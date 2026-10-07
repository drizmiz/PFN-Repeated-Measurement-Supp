
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, os, sys, time
import numpy as np, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(_PKG_ROOT/'experiments/support'))
import core
import sizehook as SH

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.environ.get('CKPT_V3', str(_PKG_ROOT/'models/tabpfn-v3-regressor-v3_default.ckpt')))
p.add_argument("--funcs", default="quadratic,tanh,step,sine")
p.add_argument("--sigmas", default="0,0.05,0.3")
p.add_argument("--seeds", default="0,1,2,3,4,5")
p.add_argument("--m", type=int, default=12)
p.add_argument("--R", type=int, default=64)
p.add_argument("--layers", default="0,1,2,3,4,5,6,7,23")
p.add_argument("--tag", default="run")
a = p.parse_args()
dev = "cuda"
LAYERS = [int(s) for s in a.layers.split(",")]
STAGES = ["xonly", "input", "embedder", "icl_in"] + [f"icl_{L}" for L in LAYERS]


class Capture:
    """Forward hooks that store the train-row tensors of each stage (fp32, cpu)."""

    def __init__(self, reg, n_train):
        self.model = reg.models_[0]
        self.n = n_train
        self.h = []
        self.raw = {}
        self.calls = {}
        M = self.model
        for nm in ("x_embed", "col_y_encoder", "feature_distribution_embedder", "column_aggregator",
                   "icl_blocks", "icl_y_encoder"):
            assert hasattr(M, nm), nm

        def keep(key, sel=lambda o: o):
            def hook(mod, args, out):
                self.calls[key] = self.calls.get(key, 0) + 1
                self.raw[key] = sel(out).detach().float()
            return hook

        self.h.append(M.x_embed.register_forward_hook(keep("xonly")))
        self.h.append(M.col_y_encoder.register_forward_hook(keep("ycol")))
        self.h.append(M.feature_distribution_embedder.register_forward_hook(
            keep("embedder", lambda o: o[0] if isinstance(o, tuple) else o)))
        self.h.append(M.icl_blocks[0].register_forward_pre_hook(
            lambda mod, args: keep("icl_in")(mod, args, args[0])))
        for L in LAYERS:
            self.h.append(M.icl_blocks[L].register_forward_hook(
                keep(f"icl_{L}", lambda o: o[0] if isinstance(o, tuple) else o)))

        self.h.append(M.col_y_encoder.register_forward_pre_hook(
            lambda mod, args: keep("y_in")(mod, args, args[0])))

    def close(self):
        for h in self.h:
            h.remove()

    def stages(self):
        assert all(c == 1 for c in self.calls.values()), self.calls
        n = self.n
        R = self.raw
        xo = R["xonly"][:, :n]                                     
        yc = R["ycol"][:, :n]                                   
        assert xo.shape[0] == 1
        inp = xo + yc.unsqueeze(2)
        out = {"xonly": xo[0].reshape(n, -1), "input": inp[0].reshape(n, -1),
               "embedder": R["embedder"][0, :n].reshape(n, -1),
               "icl_in": R["icl_in"][0, :n].reshape(n, -1)}
        for L in LAYERS:
            out[f"icl_{L}"] = R[f"icl_{L}"][0, :n].reshape(n, -1)
        y_in = R["y_in"].reshape(-1)[:n]
        return {k: v.double().cpu().numpy() for k, v in out.items()}, y_in.cpu().numpy()


def fit(ds, r, fp):
    X, y, lid = core.ctx(ds, r, "expanded")
    reg = SH.make_regressor(a.ckpt, fingerprint=fp, n_estimators=1, device=dev,
                            inference_precision=torch.float32)
    reg.fit(X, y)
    for mdl in reg.models_:
        mdl.inference_row_chunk_size = None
        mdl.inference_col_chunk_size = None
    return reg, y, lid


def run(reg, ds, y, override=None):
    cap = Capture(reg, len(y))
    if override is None:
        pred = np.asarray(reg.predict(ds["Xq"]), float)
    else:
        with SH.SizeOverride(reg, n=a.m, groups=override) as so:
            pred = np.asarray(reg.predict(ds["Xq"]), float)
        assert so.n_modified > 0, override
    st, y_in = cap.stages()
    cap.close()

    c = np.corrcoef(y_in, y)[0, 1] if y.std() > 0 else 1.0
    assert c > 0.999, ("row order / y mismatch", c)
    return st, core.mse_by_region(ds, pred)["pop"]


def stats(e_exp, lid, e_single):
    """e_exp (n, D) rows of the Expanded table; lid site ids; e_single (m, D)."""
    m = e_single.shape[0]
    cent = np.stack([e_exp[lid == i].mean(0) for i in range(m)])
    within = np.sqrt(((e_exp - cent[lid]) ** 2).sum(1).mean())
    between_exp = np.sqrt(((cent - cent.mean(0)) ** 2).sum(1).mean())
    between_single = np.sqrt(((e_single - e_single.mean(0)) ** 2).sum(1).mean())
    first = np.stack([e_exp[np.flatnonzero(lid == i)[0]] for i in range(m)])
    d0 = first - e_single
    shift0 = np.sqrt((d0 ** 2).sum(1).mean())
    common = np.linalg.norm(d0.mean(0))                                                       
    specific = np.sqrt(((d0 - d0.mean(0)) ** 2).sum(1).mean())                            
    shiftc = np.sqrt(((cent - e_single) ** 2).sum(1).mean())

    iu = np.triu_indices(m, 1)
    pd_exp = np.sqrt(((cent[:, None] - cent[None]) ** 2).sum(-1))[iu]
    pd_sin = np.sqrt(((e_single[:, None] - e_single[None]) ** 2).sum(-1))[iu]
    lay = float(np.corrcoef(pd_exp, pd_sin)[0, 1]) if pd_sin.std() > 0 else 1.0
    return dict(within=within, between_exp=between_exp, between_single=between_single,
                within_rel=within / max(between_exp, 1e-12),
                shift0_rel=shift0 / max(between_single, 1e-12),
                shift_common_rel=common / max(between_single, 1e-12),
                shift_specific_rel=specific / max(between_single, 1e-12),
                shiftc_rel=shiftc / max(between_single, 1e-12),
                between_ratio=between_exp / max(between_single, 1e-12), layout_corr=lay)


funcs = a.funcs.split(","); sigmas = [float(s) for s in a.sigmas.split(",")]
seeds = [int(s) for s in a.seeds.split(",")]
OVR = {"none": None, "featdist": ("featdist",), "count": ("featdist", "icl")}
fs = open(f"emb_stages_{a.tag}.csv", "w")
fs.write("func,sigma,seed,fp,override,stage,within,between_exp,between_single,within_rel,"
         "shift0_rel,shift_common_rel,shift_specific_rel,shiftc_rel,between_ratio,layout_corr\n")
fs.flush()
fm = open(f"emb_mse_{a.tag}.csv", "w")
fm.write("func,sigma,seed,fp,arm,mse_pop\n"); fm.flush()
t0 = time.time()
for func in funcs:
    for sigma in sigmas:
        for seed in seeds:
            ds = core.make_data1(func, a.m, sigma, seed)
            for fp in (True, False):
                reg1, y1, _ = fit(ds, 1, fp)
                st1, mse1 = run(reg1, ds, y1)
                fm.write(f"{func},{sigma},{seed},{int(fp)},single,{mse1}\n")
                del reg1
                regR, yR, lid = fit(ds, a.R, fp)
                for ov, groups in OVR.items():
                    stR, mseR = run(regR, ds, yR, groups)
                    fm.write(f"{func},{sigma},{seed},{int(fp)},expanded_{ov},{mseR}\n"); fm.flush()
                    for stg in STAGES:
                        s = stats(stR[stg], lid, st1[stg])
                        fs.write(",".join(str(x) for x in [func, sigma, seed, int(fp), ov, stg,
                                 s["within"], s["between_exp"], s["between_single"],
                                 s["within_rel"], s["shift0_rel"], s["shift_common_rel"],
                                 s["shift_specific_rel"], s["shiftc_rel"], s["between_ratio"],
                                 s["layout_corr"]]) + "\n")
                    fs.flush()
                    if ov == "none":
                        line = " ".join(f"{stg}:{stats(stR[stg], lid, st1[stg])['within_rel']:.2f}/"
                                        f"{stats(stR[stg], lid, st1[stg])['shift0_rel']:.2f}"
                                        for stg in ("xonly", "input", "embedder", "icl_in", "icl_0",
                                                    "icl_7"))
                        print(f"[{time.time()-t0:6.0f}s] {func} s={sigma} seed={seed} fp={int(fp)} "
                              f"mse1={mse1:.4f} mseR={mseR:.4f} | within/shift0: {line}", flush=True)
                del regR
                torch.cuda.empty_cache()
print("DONE", flush=True)

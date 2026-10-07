
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, math, os, sys, time, json
import numpy as np, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import sizehook as SH
import tabpfn.architectures.tabpfn_v3 as v3

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.environ.get('CKPT_V3', str(_PKG_ROOT/'models/tabpfn-v3-regressor-v3_default.ckpt')))
p.add_argument("--funcs", default="quadratic,tanh,step,sine")
p.add_argument("--sigmas", default="0.05,0.15,0.30")
p.add_argument("--seeds", default="0,1,2,3,4,5")
p.add_argument("--m", type=int, default=12)
p.add_argument("--R", type=int, default=64)
p.add_argument("--tag", default="run")
a = p.parse_args()
dev = "cuda"
NL = 24
DEEP = [18, 19, 21, 22]
SUBSETS = {"all": list(range(NL)), "deep4": DEEP, "L0_17": list(range(18))}


class Hook:
    def __init__(self, reg, lid, S):
        self.model = reg.models_[0]
        self.lid = torch.as_tensor(lid, dtype=torch.long, device=dev)
        self.S = S
        self.mode = "off"
        self.layers = set()
        self.T = {}
        self.W = {}
        self.V = {}
        self.calls = {}
        self._cur = None
        self._h = []

    def __enter__(self):
        self._orig = v3.scaled_dot_product_attention
        hk = self

        def patched(q, k, v, *args, **kwargs):
            cur = hk._cur
            if cur is None:
                return hk._orig(q, k, v, *args, **kwargs)
            layer, idx = cur
            hk._cur = (layer, idx + 1)
            if idx != 1 or hk.mode == "off":
                return hk._orig(q, k, v, *args, **kwargs)
            hk.calls[layer] = hk.calls.get(layer, 0) + 1
            assert q.shape[0] == 1 and k.shape[1] == len(hk.lid), (q.shape, k.shape)
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                qf, kf, vf = q[0].float(), k[0].float(), v[0].float()
                H = qf.shape[1]
                if kf.shape[1] != H:
                    kf = kf.repeat_interleave(H // kf.shape[1], dim=1)
                    vf = vf.repeat_interleave(H // vf.shape[1], dim=1)
                P = torch.softmax(torch.einsum("mhd,nhd->mhn", qf, kf) / math.sqrt(qf.shape[-1]), -1)
                if hk.mode == "record":
                    Ws = torch.zeros(P.shape[0], H, hk.S, device=P.device)
                    Ws.index_add_(2, hk.lid, P)
                    hk.W[layer] = Ws

                    vs = torch.zeros(hk.S, H, vf.shape[2], device=vf.device)
                    vs.index_add_(0, hk.lid, vf)
                    cnt = torch.bincount(hk.lid, minlength=hk.S).float()
                    vs = vs / cnt[:, None, None]
                    dev_rows = vf - vs[hk.lid]
                    within = torch.sqrt((dev_rows ** 2).sum(-1).mean(-1))                          
                    wsite = torch.zeros(hk.S, device=vf.device).index_add_(0, hk.lid, within ** 2)
                    wsite = torch.sqrt(wsite / cnt)                                        
                    hk.V[layer] = dict(vsite=vs.double().cpu().numpy(),
                                       within=float(wsite.mean()),
                                       nkv=int(k.shape[2]))
                    return hk._orig(q, k, v, *args, **kwargs)
                if hk.mode == "self":
                    T = P
                elif hk.mode == "target":
                    if layer not in hk.layers:
                        return hk._orig(q, k, v, *args, **kwargs)
                    T = hk.T[layer]
                else:
                    raise ValueError(hk.mode)
                out = torch.einsum("mhn,nhd->mhd", T, vf)[None]
            return out.to(q.dtype)

        v3.scaled_dot_product_attention = patched
        for i, blk in enumerate(self.model.icl_blocks):
            def pre(mod, args, kwargs, i=i):
                hk._cur = (i, 0)

            def post(mod, args, kwargs, output):
                hk._cur = None
            self._h.append(blk.icl_attention.register_forward_pre_hook(pre, with_kwargs=True))
            self._h.append(blk.icl_attention.register_forward_hook(post, with_kwargs=True))
        return self

    def __exit__(self, *e):
        v3.scaled_dot_product_attention = self._orig
        for h in self._h:
            h.remove()
        self._h = []
        return False

    def run(self, Xq, mode, layers=None, T=None):
        self.mode, self.layers, self.T, self.calls = mode, set(layers or []), T or {}, {}
        pred = np.asarray(self.reg.predict(Xq), float)
        n_exp = NL if mode != "off" else 0
        assert len(self.calls) == n_exp and all(c == 1 for c in self.calls.values()), self.calls
        self.mode = "off"
        return pred


def fit(ds, r, prec):
    X, y, lid = core.ctx(ds, r, "expanded")
    kw = {} if prec == "autocast" else dict(inference_precision=torch.float32)
    reg = SH.make_regressor(a.ckpt, fingerprint=True, n_estimators=1, device=dev, **kw)
    reg.fit(X, y)
    for mdl in reg.models_:
        mdl.inference_row_chunk_size = None
        mdl.inference_col_chunk_size = None
    return reg, y, lid


def value_stats(Vd, Wd, ybar_z, pop):
    """Per layer: R2, ||b1||, unexplained share, within-site spread."""
    out = []
    S = len(ybar_z)
    A_ = np.column_stack([np.ones(S), ybar_z])
    for L in range(NL):
        vs = Vd[L]["vsite"]                                    
        Wl = Wd[L][pop]                                           
        H = vs.shape[1]
        r2s, b1n, shc, shu, b1v, amp = [], [], [], [], [], []
        for h in range(H):
            V = vs[:, h, :]
            coef, *_ = np.linalg.lstsq(A_, V, rcond=None)           
            E = V - A_ @ coef
            r2s.append(1 - (E ** 2).sum() / ((V - V.mean(0)) ** 2).sum())
            b1 = coef[1]
            b1n.append(np.linalg.norm(b1)); b1v.append(b1)
            Ah = Wl[:, h, :]                                   
            u = Ah @ E                                         
            sig = Ah @ ybar_z                                
            den = np.linalg.norm(b1) * sig.std()
            shc.append(np.sqrt(((u - u.mean(0)) ** 2).sum(1).mean()) / den)
            shu.append(np.sqrt((u ** 2).sum(1).mean()) / den)

            su = u @ (b1 / np.linalg.norm(b1))
            k = np.cov(su, sig)[0, 1] / sig.var(ddof=1)
            amp.append(1 + k / np.linalg.norm(b1))
        vc = vs.mean(1)
        rms_site = np.sqrt((vc ** 2).sum(1).mean())
        rms_between = np.sqrt(((vc - vc.mean(0)) ** 2).sum(1).mean())
        out.append(dict(layer=L, R2=float(np.mean(r2s)), b1norm=float(np.mean(b1n)),
                        share_c=float(np.mean(shc)), share_u=float(np.mean(shu)),
                        within_rel_rms=Vd[L]["within"] / rms_site,
                        within_rel_between=Vd[L]["within"] / rms_between,
                        nkv=Vd[L]["nkv"], b1=np.mean(b1v, 0), amp=float(np.mean(amp))))
    return out


funcs = a.funcs.split(","); sigmas = [float(s) for s in a.sigmas.split(",")]
seeds = [int(s) for s in a.seeds.split(",")]
f1 = open(f"tp_transplant_{a.tag}.csv", "w")
f2 = open(f"tp_values_{a.tag}.csv", "w")
H1 = ["func", "sigma", "seed", "mse_r1_native", "mse_r64_native", "mse_r1_native_autocast",
      "mse_r64_native_autocast", "mse_r64_with_r1_all", "mse_r64_with_r1_deep4",
      "mse_r64_with_r1_L0_17", "mse_r1_with_r64_all", "mse_r64_uniform_within_site",
      "mse_r1_with_r64_deep4", "mse_r1_with_r64_L0_17",
      "corr_r64wr1all_r1native", "corr_r64native_r1native", "ctrl_self_r1_maxdiff",
      "ctrl_self_r64_maxdiff", "ctrl_record_r1_maxdiff", "ctrl_record_r64_maxdiff",
      "ctrl_r1w_r1_maxdiff", "top1_deep4_r1", "top1_deep4_r64", "recon_slope_r1", "recon_corr_r1",
      "recon_slope_r64", "recon_corr_r64"]
f1.write(",".join(H1) + "\n"); f1.flush()
H2 = ["func", "sigma", "seed", "r", "layer", "R2", "b1norm", "share_c", "share_u",
      "within_rel_rms", "within_rel_between", "nkv", "cos_b1_r1_r64", "ystd_ctx", "amp_b1"]
f2.write(",".join(H2) + "\n"); f2.flush()
os.makedirs(f"tp_preds_{a.tag}", exist_ok=True)

t0 = time.time()
for func in funcs:
    for sigma in sigmas:
        for seed in seeds:
            ds = core.make_data1(func, a.m, sigma, seed)
            Xq = ds["Xq"]; pop = ds["region"] == "pop"; S = a.m
            mse = lambda pr: core.mse_by_region(ds, pr)["pop"]
            res = {}; preds = {}

            for r in (1, a.R):
                reg, _, _ = fit(ds, r, "autocast")
                res[f"ac{r}"] = mse(np.asarray(reg.predict(Xq), float)); del reg
            regs, hooks, W, Vd, yz, ystd = {}, {}, {}, {}, {}, {}
            for r in (1, a.R):
                reg, y, lid = fit(ds, r, "fp32")
                hk = Hook(reg, lid, S); hk.reg = reg
                hk.__enter__()
                preds[f"nat{r}"] = hk.run(Xq, "off")
                pr = hk.run(Xq, "record")
                res[f"rec{r}"] = float(np.abs(pr - preds[f"nat{r}"]).max())
                W[r] = {L: hk.W[L] for L in range(NL)}
                Vd[r] = hk.V
                pr = hk.run(Xq, "self")
                res[f"self{r}"] = float(np.abs(pr - preds[f"nat{r}"]).max())
                ybar = ds["Y"][:, :r].mean(1)
                ystd[r] = float(y.std()); yz[r] = (ybar - y.mean()) / y.std()
                hk.__exit__()
                regs[r], hooks[r] = reg, hk
            R = a.R
            lidR = hooks[R].lid

            hooks[1].__enter__()
            preds["r1w_r1"] = hooks[1].run(Xq, "target", range(NL), {L: W[1][L] for L in range(NL)})
            res["r1w_r1"] = float(np.abs(preds["r1w_r1"] - preds["nat1"]).max())
            for nm, lay in SUBSETS.items():
                preds[f"r1_with_r64_{nm}"] = hooks[1].run(Xq, "target", lay,
                                                          {L: W[R][L] for L in range(NL)})
            hooks[1].__exit__()
            hooks[R].__enter__()
            T1 = {L: W[1][L][:, :, lidR] / R for L in range(NL)}
            for nm, lay in SUBSETS.items():
                preds[f"r64_with_r1_{nm}"] = hooks[R].run(Xq, "target", lay, T1)
            TU = {L: W[R][L][:, :, lidR] / R for L in range(NL)}
            preds["r64_uniform_within_site"] = hooks[R].run(Xq, "target", range(NL), TU)
            hooks[R].__exit__()
            del T1, TU

            top1 = {r: float(np.mean([W[r][L].mean(1).max(1).values.cpu().numpy()[pop]
                                      for L in DEEP])) for r in (1, R)}
            c = lambda u, w: float(np.corrcoef(u[pop], w[pop])[0, 1])
            rec = {}
            for r in (1, R):                                                                           
                Wd4 = np.mean([W[r][L].mean(1).double().cpu().numpy() for L in DEEP], 0)
                rc = (Wd4 @ ds["Y"][:, :r].mean(1))[pop]
                rec[r] = (float(np.polyfit(rc, preds[f"nat{r}"][pop], 1)[0]),
                          float(np.corrcoef(rc, preds[f"nat{r}"][pop])[0, 1]))
            row = [func, sigma, seed, mse(preds["nat1"]), mse(preds[f"nat{R}"]), res["ac1"],
                   res[f"ac{R}"], mse(preds["r64_with_r1_all"]), mse(preds["r64_with_r1_deep4"]),
                   mse(preds["r64_with_r1_L0_17"]), mse(preds["r1_with_r64_all"]),
                   mse(preds["r64_uniform_within_site"]), mse(preds["r1_with_r64_deep4"]),
                   mse(preds["r1_with_r64_L0_17"]),
                   c(preds["r64_with_r1_all"], preds["nat1"]), c(preds[f"nat{R}"], preds["nat1"]),
                   res["self1"], res[f"self{R}"], res["rec1"], res[f"rec{R}"], res["r1w_r1"],
                   top1[1], top1[R], rec[1][0], rec[1][1], rec[R][0], rec[R][1]]
            f1.write(",".join(str(x) for x in row) + "\n"); f1.flush()
            np.savez(f"tp_preds_{a.tag}/{func}_{sigma}_{seed}.npz", Xq=Xq[:, 0], fq=ds["fq"],
                     pop=pop, **preds)

            Wnp = {r: {L: W[r][L].double().cpu().numpy() for L in range(NL)} for r in (1, R)}
            vst = {r: value_stats(Vd[r], Wnp[r], yz[r], pop) for r in (1, R)}
            for r in (1, R):
                for L in range(NL):
                    s = vst[r][L]
                    b1a, b1b = vst[1][L]["b1"], vst[R][L]["b1"]
                    cs = float(b1a @ b1b / (np.linalg.norm(b1a) * np.linalg.norm(b1b)))
                    f2.write(",".join(str(x) for x in [func, sigma, seed, r, L, s["R2"], s["b1norm"],
                             s["share_c"], s["share_u"], s["within_rel_rms"],
                             s["within_rel_between"], s["nkv"], cs, ystd[r], s["amp"]]) + "\n")
            f2.flush()
            print(f"[{time.time()-t0:7.1f}s] {func} s={sigma} seed={seed}  r1={row[3]:.5f} "
                  f"r64={row[4]:.5f} r64w1all={row[7]:.5f} deep4={row[8]:.5f} L0_17={row[9]:.5f} "
                  f"r1w64={row[10]:.5f} (d4 {row[12]:.5f} L0_17 {row[13]:.5f}) unif={row[11]:.5f} | self {res['self1']:.1e}/{res[f'self{R}']:.1e} "
                  f"rec {res['rec1']:.1e}/{res[f'rec{R}']:.1e} r1w_r1 {res['r1w_r1']:.1e} "
                  f"top1 {top1[1]:.3f}/{top1[R]:.3f} recon slope {rec[1][0]:.2f}/{rec[R][0]:.2f} corr {rec[1][1]:.3f}/{rec[R][1]:.3f}", flush=True)
            del regs, hooks, W, Wnp
            torch.cuda.empty_cache()
print("DONE", flush=True)

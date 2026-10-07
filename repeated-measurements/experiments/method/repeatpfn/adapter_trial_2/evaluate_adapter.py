from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, str(_PKG_ROOT/'experiments/support'), str(_PKG_ROOT/'experiments/support')):
    sys.path.insert(0, p)
CKPT = os.environ.get('CKPT_V3', str(_PKG_ROOT/'models/tabpfn-v3-regressor-v3_default.ckpt'))
FUNCS = ["quadratic", "abs", "tanh", "step", "sine", "linear"]
SIGMAS = [0.05, 0.5]; SEEDS = range(6); RS = [4, 16, 64, 256]; M = 12
SEL = [18, 19, 21, 22]
CELLS_B = "step,0.5,24,7,4;sine,1,12,7,4;tanh,0.5,12,7,4;step,0.5,24,7,16;linear,0.5,12,4,4"
ACOLS = ["func", "sigma", "seed", "r", "arm", "model", "n_rows", "mse_pop", "align_fail", "secs"]
BCOLS = ["func", "sigma", "m", "seed", "r", "arm", "model", "n_rows", "mse_pop", "align_fail", "secs"]
PCOLS = ["func", "sigma", "seed", "r", "model", "n_rows", "xstar", "truth", "pred", "top1_sel", "top3_sel", "neff_sel", "neff_sel_median_grid", "secs"]
DCOLS = ["design", "func", "sigma", "seed", "arm", "model", "n_rows", "mse_pop", "align_fail", "secs"]


def run(a):
    import torch, core, designs, sizehook as SH, unb_common as U, ct_sweep as CS, ct_methods as CT
    from adapter import AdapterHook, load_adapter, unchunk
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_pre = torch.load(CKPT, map_location="cpu", weights_only=False)["inference_config"].get("ENABLE_GPU_PREPROCESSING", False)
    extra = {"ENABLE_GPU_PREPROCESSING": bool(gpu_pre)}
    adapters = {}
    for V in a.variants.split(","):
        path = getattr(a, "ckpt_" + V) or os.path.join(HERE, "run_%s" % V, "best.pt")
        if os.path.exists(path):
            adapters["ad_" + V] = load_adapter(path, dev); print("adapter %s <- %s" % (V, path), flush=True)
        else:
            print("WARNING: no checkpoint for variant %s (%s) -- skipped" % (V, path), flush=True)
    models = ["sh"] + sorted(adapters)
    os.makedirs(a.outdir, exist_ok=True)
    print("device=%s gpu_pre=%s models=%s n_est=%d" % (dev, gpu_pre, models, a.n_est), flush=True)
    stats = {"align_fail": 0, "fits": 0}

    def fit_predict(which, Xc, yc, Xq, n_est=None, probe=None):
        reg = SH.make_regressor(CKPT, fingerprint=True, n_estimators=n_est or a.n_est, device=dev, extra_inference_config=extra, inference_precision=torch.float32)
        reg.fit(Xc, yc); unchunk(reg); stats["fits"] += 1
        if which == "sh" and probe is None:
            return np.asarray(reg.predict(Xq), float), 0
        with AdapterHook(reg, adapters.get(which), Xc, strict=False, probe_layers=None if probe is None else probe[0], probe_reducer=None if probe is None else probe[1]) as hk:
            pred = np.asarray(reg.predict(Xq), float)
        if hk.n_align_fail:
            stats["align_fail"] += hk.n_align_fail; print("  ALIGN FAIL (%s): %s" % (which, hk.align_msgs[0]), flush=True)
        return pred, hk.n_align_fail

    def opener(path, cols, keyf):
        done = set()
        if os.path.exists(path):
            for row in csv.DictReader(open(path)):
                done.add(keyf(row))
        fh = open(path, "a", newline=""); w = csv.writer(fh)
        if not done: w.writerow(cols); fh.flush()
        return fh, w, done

    t0 = time.time()
    if a.part in ("a", "all"):
        fh, w, done = opener(os.path.join(a.outdir, "eval_a.csv"), ACOLS, lambda r: (r["func"], float(r["sigma"]), int(r["seed"]), int(r["r"]), r["arm"]))
        for seed in SEEDS:
            for sigma in SIGMAS:
                for func in FUNCS:
                    ds = designs.make("synth1d", func, M, sigma, seed); Xq = ds["Xq"]
                    jobs = [(1, "single_sh", "sh", "expanded")]
                    for r in RS:
                        jobs += [(r, "expanded_sh", "sh", "expanded"), (r, "mean_sh", "sh", "avg")]
                        jobs += [(r, "expanded_%s" % mdl, mdl, "expanded") for mdl in sorted(adapters)] + [(r, "mean_%s" % mdl, mdl, "avg") for mdl in sorted(adapters)]
                    for r, arm, mdl, ck in jobs:
                        if (func, float(sigma), seed, r, arm) in done: continue
                        t1 = time.time(); Xc, yc, _ = core.ctx(ds, r, ck); pred, af = fit_predict(mdl, Xc, yc, Xq)
                        w.writerow([func, "%g" % sigma, seed, r, arm, mdl, len(yc), "%.8f" % core.mse_by_region(ds, pred)["pop"], af, "%.1f" % (time.time() - t1)]); fh.flush()
                    print("  (a) seed %d sigma %g %s done  fits=%d  t=%.0fs" % (seed, sigma, func, stats["fits"], time.time() - t0), flush=True)
        fh.close()

    if a.part in ("b", "all"):
        fh, w, done = opener(os.path.join(a.outdir, "eval_b.csv"), BCOLS, lambda r: (r["func"], float(r["sigma"]), int(r["m"]), int(r["seed"]), int(r["r"]), r["arm"]))
        npz = os.path.join(a.outdir, "curves_b.npz"); store = dict(np.load(npz)) if os.path.exists(npz) else {}
        for cell in a.cells.split(";"):
            func, sig, m, seed, r = cell.split(","); sig, m, seed, r = float(sig), int(m), int(seed), int(r)
            ds = designs.make("synth1d", func, m, sig, seed); Xq = ds["Xq"]; key = "%s_%g_%d_%d_%d" % (func, sig, m, seed, r)
            store[key + "_Xq"] = Xq.ravel(); store[key + "_fq"] = ds["fq"]; store[key + "_region"] = ds["region"]; store[key + "_X"] = ds["X"].ravel(); store[key + "_Y"] = ds["Y"][:, :r]
            jobs = [(1, "single_sh", "sh", "expanded"), (r, "expanded_sh", "sh", "expanded"), (r, "mean_sh", "sh", "avg")] + [(r, "expanded_%s" % mdl, mdl, "expanded") for mdl in sorted(adapters)]
            for rr, arm, mdl, ck in jobs:
                if (func, sig, m, seed, r, arm) in done: continue
                t1 = time.time(); Xc, yc, _ = core.ctx(ds, rr, ck); pred, af = fit_predict(mdl, Xc, yc, Xq); mse = core.mse_by_region(ds, pred)["pop"]
                store[key + "_" + arm] = pred; store[key + "_" + arm + "_mse"] = mse
                w.writerow([func, "%g" % sig, m, seed, r, arm, mdl, len(yc), "%.8f" % mse, af, "%.1f" % (time.time() - t1)]); fh.flush()
            np.savez(npz, **store); print("  (b) %s done  t=%.0fs" % (cell, time.time() - t0), flush=True)
        fh.close()

    if a.part in ("c", "all"):
        fh, w, done = opener(os.path.join(a.outdir, "probe_c.csv"), PCOLS, lambda r: (r["func"], int(r["seed"]), int(r["r"]), r["model"]))
        for func in ["quadratic", "step"]:
            for seed in range(10):
                ds = designs.make("synth1d", func, M, 0.15, seed); xs = np.sort(ds["X"].ravel()); gaps = np.diff(xs); g = int(np.argmax(gaps))
                xstar = xs[g] + 0.25 * gaps[g]; Xq = ds["Xq"]; iq = int(np.argmin(np.abs(Xq.ravel() - xstar)))
                for r in [1, 4, 16, 64, 256]:
                    for mdl in models:
                        if (func, seed, r, mdl) in done: continue
                        t1 = time.time(); Xc, yc, lid = core.ctx(ds, r, "expanded"); L = int(lid.max()) + 1
                        lid_t = torch.as_tensor(lid, dtype=torch.long, device=dev); st = {}

                        def reducer(layer, probs):
                            Mh = torch.zeros(probs.shape[0], probs.shape[1], L, device=probs.device, dtype=probs.dtype)
                            Mh.index_add_(2, lid_t, probs); Mh = Mh / Mh.sum(2, keepdim=True).clamp_min(1e-30)
                            st.setdefault(layer, []).append(Mh.mean(1).double().cpu().numpy())

                        pred, _ = fit_predict(mdl, Xc, yc, Xq, n_est=1, probe=(SEL, reducer))
                        W = {l: np.mean(st[l], axis=0) for l in st}
                        wm = np.mean([W[l] for l in SEL], axis=0); s = np.sort(wm, axis=1)[:, ::-1]; ne = 1.0 / np.clip((wm ** 2).sum(1), 1e-30, None)
                        w.writerow([func, 0.15, seed, r, mdl, len(yc), "%.4f" % xstar, "%.4f" % ds["fq"][iq], "%.4f" % pred[iq], "%.4f" % s[iq, 0], "%.4f" % s[iq, :3].sum(),
                                    "%.3f" % ne[iq], "%.3f" % float(np.median(ne)), "%.1f" % (time.time() - t1)]); fh.flush()
                print("  (c) %s seed %d done  t=%.0fs" % (func, seed, time.time() - t0), flush=True)
        fh.close()

    if a.part in ("d", "all"):
        fh, w, done = opener(os.path.join(a.outdir, "eval_d.csv"), DCOLS, lambda r: (r["design"], r["func"], float(r["sigma"]), int(r["seed"]), r["arm"]))
        npz = os.path.join(a.outdir, "curves_d.npz"); store = dict(np.load(npz)) if os.path.exists(npz) else {}
        for seed in range(4):
            for func in U.FUNCS:
                for sigma in [0.5, 1.0]:
                    ds = core.make_data1(func, U.M, sigma, seed, rmax=64); X, Xq = ds["X"], ds["Xq"]
                    for design in ["alt", "mix"]:
                        r = CS.r_vec(design, U.M); reads = U.site_readings(ds, r); s2 = U.pooled_sigma2(reads)
                        ybar = np.array([y.mean() for y in reads]); se = np.sqrt(s2 / r); yfirst = np.array([y[0] for y in reads])
                        Xexp, yexp = U.build_table("expanded", X, reads); key = "%s_%s_%g_%d" % (design, func, sigma, seed)
                        store[key + "_fq"] = ds["fq"]; store[key + "_r"] = r; store[key + "_ybar"] = ybar

                        def emit(arm, mdl, pred, nrows, t1, af=0):
                            store[key + "_" + arm] = pred.astype(np.float32)
                            w.writerow([design, func, "%g" % sigma, seed, arm, mdl, nrows, "%.8f" % core.mse_by_region(ds, pred)["pop"], af, "%.1f" % (time.time() - t1)]); fh.flush()

                        f0q = None
                        for arm, mdl, Xc, yc in [("single_sh", "sh", X, yfirst), ("mean_sh", "sh", X, ybar), ("expanded_sh", "sh", Xexp, yexp)] + [("expanded_%s" % mdl, mdl, Xexp, yexp) for mdl in sorted(adapters)]:
                            if arm == "mean_sh" or (design, func, sigma, seed, arm) not in done:
                                t1 = time.time(); pred, af = fit_predict(mdl, Xc, yc, Xq)
                                if arm == "mean_sh": f0q = pred
                                if (design, func, sigma, seed, arm) not in done: emit(arm, mdl, pred, len(yc), t1, af)
                        if (design, func, sigma, seed, "twostage_sh") not in done:
                            t1 = time.time(); f0s, _ = fit_predict("sh", X, ybar, X)
                            emit("twostage_sh", "sh", CT.twostage(X, ybar, se, Xq, lambda Xc, yc, Xqq: fit_predict("sh", Xc, yc, Xqq)[0], f0_sites=f0s, f0_query=f0q), len(ybar), t1)
                        if (design, func, sigma, seed, "gp_ref") not in done:
                            t1 = time.time(); pred, kern = U.gp_ref_predict(X, ybar, r, s2, Xq, seed=seed); emit("gp_ref", "gp", pred, U.M, t1)
                        np.savez(npz, **store)
                    print("  (d) seed %d %s sigma %g done  fits=%d  t=%.0fs" % (seed, func, sigma, stats["fits"], time.time() - t0), flush=True)
        fh.close()
    print("EVAL DONE %.0fs  fits=%d  alignment failures=%d" % (time.time() - t0, stats["fits"], stats["align_fail"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=str(_PKG_ROOT/'runs'/'adapter_trial_2/eval')); p.add_argument("--part", default="all", choices=["a", "b", "c", "d", "all"])
    p.add_argument("--variants", default="L,V,LV"); p.add_argument("--n_est", type=int, default=8); p.add_argument("--cells", default=CELLS_B)
    for V in ("L", "V", "LV"):
        p.add_argument("--ckpt_%s" % V, default=None)
    run(p.parse_args())

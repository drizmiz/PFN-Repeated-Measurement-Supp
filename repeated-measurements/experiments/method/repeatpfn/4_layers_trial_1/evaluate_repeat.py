from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(_PKG_ROOT/'experiments/support')); sys.path.insert(0, str(_PKG_ROOT/'experiments/support'))
CKPT = os.environ.get('CKPT_V3', str(_PKG_ROOT/'models/tabpfn-v3-regressor-v3_default.ckpt'))
FUNCS = ["quadratic", "abs", "tanh", "step", "sine", "linear"]
SIGMAS = [0.05, 0.5]; SEEDS = range(6); RS = [4, 16, 64, 256]; M = 12
SEL = [18, 19, 21, 22]
COLS = ["func", "sigma", "seed", "r", "arm", "model", "n_rows", "mse_pop", "secs"]
PCOLS = ["func", "sigma", "seed", "r", "model", "n_rows", "xstar", "truth", "pred", "top1_sel", "top3_sel", "neff_sel", "neff_sel_median_grid", "secs"]


def run(a):
    import torch, core, designs, sizehook as SH
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_pre = torch.load(CKPT, map_location="cpu", weights_only=False)["inference_config"].get("ENABLE_GPU_PREPROCESSING", False)
    extra = {"ENABLE_GPU_PREPROCESSING": bool(gpu_pre)}
    models = {"sh": CKPT, "ft": a.ckpt}
    os.makedirs(a.outdir, exist_ok=True)
    print("device=%s ckpt=%s gpu_pre=%s" % (dev, a.ckpt, gpu_pre), flush=True)

    def fit_predict(which, Xc, yc, Xq, n_est):
        reg = SH.make_regressor(models[which], fingerprint=True, n_estimators=n_est, device=dev, extra_inference_config=extra)
        reg.fit(Xc, yc); return np.asarray(reg.predict(Xq), float)

    t0 = time.time()
    if a.part in ("a", "all"):
        out = os.path.join(a.outdir, "eval.csv"); done = set()
        if os.path.exists(out):
            for row in csv.DictReader(open(out)):
                done.add((row["func"], float(row["sigma"]), int(row["seed"]), int(row["r"]), row["arm"]))
        fh = open(out, "a", newline=""); w = csv.writer(fh)
        if not done: w.writerow(COLS); fh.flush()
        n = 0
        for seed in SEEDS:
            for sigma in SIGMAS:
                for func in FUNCS:
                    ds = designs.make("synth1d", func, M, sigma, seed); Xq = ds["Xq"]
                    jobs = [(1, "single_%s" % mdl, mdl, "expanded") for mdl in ("sh", "ft")]
                    for r in RS:
                        jobs += [(r, "%s_%s" % (kind, mdl), mdl, ck) for kind, ck in (("expanded", "expanded"), ("mean", "avg")) for mdl in ("sh", "ft")]
                    for r, arm, mdl, ck in jobs:
                        if (func, float(sigma), seed, r, arm) in done: continue
                        t1 = time.time(); Xc, yc, _ = core.ctx(ds, r, ck); pred = fit_predict(mdl, Xc, yc, Xq, a.n_est)
                        w.writerow([func, "%g" % sigma, seed, r, arm, mdl, len(yc), "%.8f" % core.mse_by_region(ds, pred)["pop"], "%.1f" % (time.time() - t1)]); fh.flush(); n += 1
                    print("  (a) seed %d sigma %g %s done  fits=%d  t=%.0fs" % (seed, sigma, func, n, time.time() - t0), flush=True)
        fh.close()

    if a.part in ("b", "all"):
        out = os.path.join(a.outdir, "probe.csv"); done = set()
        if os.path.exists(out):
            for row in csv.DictReader(open(out)):
                done.add((row["func"], int(row["seed"]), int(row["r"]), row["model"]))
        fh = open(out, "a", newline=""); w = csv.writer(fh)
        if not done: w.writerow(PCOLS); fh.flush()
        for func in ["quadratic", "step"]:
            for seed in range(10):
                ds = designs.make("synth1d", func, M, 0.15, seed); xs = np.sort(ds["X"].ravel()); gaps = np.diff(xs); g = int(np.argmax(gaps))
                xstar = xs[g] + 0.25 * gaps[g]; Xq = ds["Xq"]; iq = int(np.argmin(np.abs(Xq.ravel() - xstar)))
                for r in [1, 4, 16, 64, 256]:
                    for mdl in ("sh", "ft"):
                        if (func, seed, r, mdl) in done: continue
                        t1 = time.time(); Xc, yc, lid = core.ctx(ds, r, "expanded"); L = int(lid.max()) + 1
                        lid_t = torch.as_tensor(lid, dtype=torch.long, device=dev); store = {}

                        def reducer(layer, probs):
                            Mh = torch.zeros(probs.shape[0], probs.shape[1], L, device=probs.device, dtype=probs.dtype)
                            Mh.index_add_(2, lid_t, probs); Mh = Mh / Mh.sum(2, keepdim=True).clamp_min(1e-30)
                            store.setdefault(layer, []).append(Mh.mean(1).double().cpu().numpy()); return None

                        reg = SH.make_regressor(models[mdl], fingerprint=True, n_estimators=1, device=dev, extra_inference_config=extra); reg.fit(Xc, yc)
                        for mm in reg.models_: mm.inference_row_chunk_size = None; mm.inference_col_chunk_size = None
                        with SH.AttentionProbe(reg, layers=SEL, reducer=reducer): pred = np.asarray(reg.predict(Xq), float)
                        W = {l: np.mean(store[l], axis=0) for l in store}
                        wm = np.mean([W[l] for l in SEL], axis=0); s = np.sort(wm, axis=1)[:, ::-1]; ne = 1.0 / np.clip((wm ** 2).sum(1), 1e-30, None)
                        w.writerow([func, 0.15, seed, r, mdl, len(yc), "%.4f" % xstar, "%.4f" % ds["fq"][iq], "%.4f" % pred[iq], "%.4f" % s[iq, 0], "%.4f" % s[iq, :3].sum(),
                                    "%.3f" % ne[iq], "%.3f" % float(np.median(ne)), "%.1f" % (time.time() - t1)]); fh.flush()
                print("  (b) %s seed %d done  t=%.0fs" % (func, seed, time.time() - t0), flush=True)
        fh.close()

    if a.part in ("c", "all"):
        store = {}
        for func in ["quadratic", "step", "sine"]:
            ds = designs.make("synth1d", func, M, 0.15, 0); Xq = ds["Xq"]
            store["%s_Xq" % func] = Xq.ravel(); store["%s_fq" % func] = ds["fq"]; store["%s_X" % func] = ds["X"].ravel(); store["%s_region" % func] = ds["region"]
            for r in [1, 64, 256]:
                store["%s_ybar_%d" % (func, r)] = ds["Y"][:, :r].mean(1)
                for arm, mdl, ck in (("expanded_sh", "sh", "expanded"), ("expanded_ft", "ft", "expanded"), ("mean_sh", "sh", "avg")):
                    Xc, yc, _ = core.ctx(ds, r, ck); pred = fit_predict(mdl, Xc, yc, Xq, a.n_est)
                    store["%s_%s_%d" % (func, arm, r)] = pred; store["%s_%s_%d_mse" % (func, arm, r)] = core.mse_by_region(ds, pred)["pop"]
            print("  (c) %s done  t=%.0fs" % (func, time.time() - t0), flush=True)
        np.savez(os.path.join(a.outdir, "curves.npz"), **store)
    print("EVAL DONE %.0fs" % (time.time() - t0), flush=True)


def summarise(outdir):
    L = []
    ev = os.path.join(outdir, "eval.csv")
    if os.path.exists(ev):
        rows = list(csv.DictReader(open(ev))); by = {}
        for r in rows:
            by.setdefault((r["func"], r["sigma"], int(r["seed"])), {}).setdefault(int(r["r"]), {})[r["arm"]] = float(r["mse_pop"])

        pairs = [("expanded_ft", "expanded_sh"), ("expanded_ft", "mean_sh"), ("mean_ft", "mean_sh"), ("expanded_sh", "mean_sh")]
        L.append("\n### (a) paired cells, MSE_pop x1e-3 (medians over the cells of the row; wins = paired count where the first arm is strictly lower)\n")
        L.append("| sigma | r | cells | single_sh | expanded_sh | expanded_ft | mean_sh | mean_ft | " + " | ".join("%s < %s" % p for p in pairs) + " | med ratio exp_ft/exp_sh | med ratio exp_ft/mean_sh | med ratio mean_ft/mean_sh |")
        L.append("|" + "---|" * (8 + len(pairs) + 3))
        for sigma in ["0.05", "0.5"]:
            for r in RS:
                cells = [(k, d[r]) for k, d in by.items() if k[1] == sigma and r in d and "single_sh" in d.get(1, {})]
                if not cells: continue
                med = lambda arm: 1e3 * np.median([d[arm] for _, d in cells if arm in d]) if any(arm in d for _, d in cells) else float("nan")
                sing = 1e3 * np.median([by[k][1]["single_sh"] for k, _ in cells])
                wins = ["%d/%d" % (sum(dd[p[0]] < dd[p[1]] for _, dd in cells if p[0] in dd and p[1] in dd),
                                   sum(1 for _, dd in cells if p[0] in dd and p[1] in dd)) for p in pairs]
                rat = lambda p: np.median([dd[p[0]] / dd[p[1]] for _, dd in cells if p[0] in dd and p[1] in dd]) if any(p[0] in dd and p[1] in dd for _, dd in cells) else float("nan")
                L.append("| %s | %d | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %s | %.3f | %.3f | %.3f |" % (
                    sigma, r, len(cells), sing, med("expanded_sh"), med("expanded_ft"), med("mean_sh"), med("mean_ft"), " | ".join(wins), rat(pairs[0]), rat(pairs[1]), rat(pairs[2])))
        L.append("\n### (a) per-function medians at r = 64 / 256 (over seeds), MSE_pop x1e-3\n")
        L.append("| sigma | r | func | single_sh | expanded_sh | expanded_ft | mean_sh | mean_ft |")
        L.append("|---|---|---|---|---|---|---|---|")
        for sigma in ["0.05", "0.5"]:
            for r in [64, 256]:
                for f in FUNCS:
                    cells = [d for k, d in by.items() if k[1] == sigma and k[0] == f and r in d]
                    if not cells: continue
                    med = lambda arm: 1e3 * np.median([c[r][arm] for c in cells if arm in c[r]])
                    L.append("| %s | %d | %s | %.2f | %.2f | %.2f | %.2f | %.2f |" % (sigma, r, f, 1e3 * np.median([c[1]["single_sh"] for c in cells if "single_sh" in c.get(1, {})]),
                                                                                 med("expanded_sh"), med("expanded_ft"), med("mean_sh"), med("mean_ft")))
        L.append("\n### (a) forgetting on plain tables (one reading per site): single_ft vs single_sh\n")
        L.append("| sigma | cells | median single_sh | median single_ft | median ratio ft/sh | ft better | ft worse | within 5% |")
        L.append("|---|---|---|---|---|---|---|---|")
        for sigma in ["0.05", "0.5", "all"]:
            pr = [(d[1]["single_ft"], d[1]["single_sh"]) for k, d in by.items() if (sigma == "all" or k[1] == sigma) and 1 in d and "single_ft" in d[1] and "single_sh" in d[1]]
            if not pr: continue
            rat = [f / s for f, s in pr]
            L.append("| %s | %d | %.2f | %.2f | %.3f | %d | %d | %d |" % (sigma, len(pr), 1e3 * np.median([s for _, s in pr]), 1e3 * np.median([f for f, _ in pr]), np.median(rat),
                                                                    sum(f < s for f, s in pr), sum(f > s for f, s in pr), sum(abs(x - 1) <= 0.05 for x in rat)))
        L.append("\n### (a) forgetting on Mean-tables: mean_ft vs mean_sh, all r\n")
        L.append("| sigma | cells | median ratio | ft better | ft worse | within 5% |")
        L.append("|---|---|---|---|---|---|")
        for sigma in ["0.05", "0.5", "all"]:
            pr = [(d[r]["mean_ft"], d[r]["mean_sh"]) for k, d in by.items() if (sigma == "all" or k[1] == sigma) for r in d if r != 1 and "mean_ft" in d[r] and "mean_sh" in d[r]]
            if not pr: continue
            rat = [f / s for f, s in pr]
            L.append("| %s | %d | %.3f | %d | %d | %d |" % (sigma, len(pr), np.median(rat), sum(f < s for f, s in pr), sum(f > s for f, s in pr), sum(abs(x - 1) <= 0.05 for x in rat)))
    pb = os.path.join(outdir, "probe.csv")
    if os.path.exists(pb):
        rows = list(csv.DictReader(open(pb)))
        L.append("\n### (b) concentration at x* (quarter into the widest gap), layers 18/19/21/22 averaged, n_estimators = 1, sigma 0.15, seeds 0-9\n")
        L.append("| func | r | n_rows | n_eff sh (median) | n_eff ft | top-1 sh | top-1 ft | n_eff grid-median sh | ft | \\|pred - truth\\| sh | ft |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for func in ["quadratic", "step"]:
            for r in [1, 4, 16, 64, 256]:
                g = {m: [x for x in rows if x["func"] == func and int(x["r"]) == r and x["model"] == m] for m in ("sh", "ft")}
                if not g["sh"] or not g["ft"]: continue
                med = lambda m, c: np.median([float(x[c]) for x in g[m]])
                err = lambda m: np.median([abs(float(x["pred"]) - float(x["truth"])) for x in g[m]])
                L.append("| %s | %d | %s | %.2f | %.2f | %.3f | %.3f | %.2f | %.2f | %.3f | %.3f |" % (func, r, g["sh"][0]["n_rows"], med("sh", "neff_sel"), med("ft", "neff_sel"),
                         med("sh", "top1_sel"), med("ft", "top1_sel"), med("sh", "neff_sel_median_grid"), med("ft", "neff_sel_median_grid"), err("sh"), err("ft")))
    cv = os.path.join(outdir, "curves.npz")
    if os.path.exists(cv):
        z = np.load(cv, allow_pickle=True)
        L.append("\n### (c) curve cells (sigma 0.15, seed 0), MSE_pop x1e-3\n")
        L.append("| func | r | expanded_sh | expanded_ft | mean_sh |"); L.append("|---|---|---|---|---|")
        for func in ["quadratic", "step", "sine"]:
            for r in [1, 64, 256]:
                L.append("| %s | %d | %.2f | %.2f | %.2f |" % (func, r, *[1e3 * float(z["%s_%s_%d_mse" % (func, arm, r)]) for arm in ("expanded_sh", "expanded_ft", "mean_sh")]))
    text = "\n".join(L); print(text); return text


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None); p.add_argument("--outdir", default=str(_PKG_ROOT/'runs'/'repeatpfn/eval1'))
    p.add_argument("--n_est", type=int, default=8); p.add_argument("--part", default="all", choices=["a", "b", "c", "all"])
    p.add_argument("--summary", default=None); p.add_argument("--summary_out", default=None)
    a = p.parse_args()
    if a.summary:
        t = summarise(a.summary)
        if a.summary_out: open(a.summary_out, "w").write(t)
    else:
        assert a.ckpt and os.path.exists(a.ckpt), a.ckpt
        run(a)


import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import sys, os, time, argparse, csv, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, str(_PKG_ROOT/'experiments/support'))
import core, sizehook
from unb_hook import LogWeightHook, unchunk
import unb_common as U

CKPT = os.environ.get('CKPT_V3', str(_PKG_ROOT/'models/tabpfn-v3-regressor-v3_default.ckpt'))
core.R_MAX = 64

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=U.SEEDS)
ap.add_argument("--funcs", nargs="+", default=U.FUNCS)
ap.add_argument("--sigmas", type=float, nargs="+", default=U.SIGMAS)
ap.add_argument("--designs", nargs="+", default=U.DESIGNS)
ap.add_argument("--out", required=True)
ap.add_argument("--preds", default=str(_PKG_ROOT/'runs'/'unb_preds'))
args = ap.parse_args()
os.makedirs(args.preds, exist_ok=True)

reg = sizehook.make_regressor(CKPT, fingerprint=True, n_estimators=8, device="cuda",
                              inference_precision=torch.float32)

done = set()
if os.path.exists(args.out):
    with open(args.out) as f:
        for row in csv.DictReader(f):
            done.add((row["func"], float(row["sigma"]), int(row["seed"]), row["design"], row["arm"]))
fh = open(args.out, "a", newline="")
wr = csv.DictWriter(fh, fieldnames=U.CSV_COLS)
if os.path.getsize(args.out) == 0:
    wr.writeheader(); fh.flush()


def tfm_fit(Xc, yc):
    reg.fit(Xc, yc); unchunk(reg)


def emit(base, arm, nrows, pred, secs, note=""):
    s = U.score(ds, pred, r)
    row = dict(base, arm=arm, nrows=int(nrows), secs=round(secs, 3), note=note,
               mse_pop=s["pop"], mse_site=s["site"], mse_gap=s["gap"],
               mse_pop_hi=s["pop_hi"], mse_pop_lo=s["pop_lo"], mse_site_hi=s["site_hi"], mse_site_lo=s["site_lo"])
    wr.writerow(row); fh.flush()
    return s["pop"]


t_all = time.time()
for seed in args.seeds:
    for func in args.funcs:
        for sigma in args.sigmas:
            ds = core.make_data1(func, U.M, sigma, seed, rmax=64)
            X, Xq = ds["X"], ds["Xq"]
            for design in args.designs:
                cell = (func, sigma, seed, design)
                if all((func, sigma, seed, design, a) in done for a in U.ARMS):
                    continue
                r = U.r_vec(design)
                reads = U.site_readings(ds, r)
                s2 = U.pooled_sigma2(reads)
                ybar = np.array([y.mean() for y in reads])
                base = dict(func=func, sigma=sigma, seed=seed, design=design, sigma2_hat=s2)
                preds = {"truth": ds["fq"], "x_sites": X[:, 0], "r": r, "ybar": ybar,
                         "y_first": np.array([y[0] for y in reads]), "xq": Xq[:, 0], "region": ds["region"]}
                res = {}
                t0 = time.time()
                for arm in U.TFM_TABLE_ARMS:
                    Xc, yc = U.build_table(arm, X, reads)
                    t = time.time(); tfm_fit(Xc, yc); p = np.asarray(reg.predict(Xq), float)
                    res[arm] = emit(base, arm, len(yc), p, time.time() - t); preds[arm] = p

                Xc, yc = U.build_table("mean", X, reads)
                tfm_fit(Xc, yc)
                for arm, lw, which in (("mean_logw", np.log(r), "test"),
                                       ("mean_logw_prec", np.log(r / s2), "test"),
                                       ("mean_logw_both", np.log(r), "both"),
                                       ("mean_logw_b025", 0.25 * np.log(r), "test"),
                                       ("mean_logw_b05", 0.5 * np.log(r), "test")):
                    t = time.time()
                    with LogWeightHook(reg, lw, which=which) as hk:
                        p = np.asarray(reg.predict(Xq), float)
                    res[arm] = emit(base, arm, len(yc), p, time.time() - t,
                                    note="calls=%d applied=%d" % (hk.n_icl_calls, hk.n_applied)); preds[arm] = p
                t = time.time()
                p, kern = U.gp_ref_predict(X, ybar, r, s2, Xq, seed=seed)
                res["gp_ref"] = emit(base, "gp_ref", U.M, p, time.time() - t, note=kern); preds["gp_ref"] = p
                np.savez(os.path.join(args.preds, "%s_%.2f_s%d_%s.npz" % cell), **preds)
                print("[%6.0fs] %-9s sig=%.2f seed=%d %-5s (%4.1fs) " % (time.time() - t_all, func, sigma, seed, design, time.time() - t0)
                      + " ".join("%s=%.4f" % (a, res[a]) for a in U.ARMS), flush=True)
fh.close()
print("DONE", flush=True)

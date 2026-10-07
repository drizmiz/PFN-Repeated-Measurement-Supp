
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, csv, glob, sys, time, warnings              
import numpy as np              
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rdcore as RC              
import interfaces as IF              
from classical import gp, parse_seeds              

OUT = os.environ.get("RD_OUT", os.path.join(HERE, "out"))
LEARNER = "GP_rows"
TOLD = "GP_told"


def _tables(ds, seed, r, arm):
    U = RC.unit(ds, seed)
    xm, xs = U["XL"].mean(0), U["XL"].std(0)
    xs[xs == 0] = 1.0
    XL = (U["XL"] - xm) / xs
    XQall = (np.vstack([U["XQ"], U["XL"]]) - xm) / xs
    Xc, yc, XQc = IF.build(arm, XL, U["Z"][:, :r], XQall)
    return U, Xc, yc, XQc


def gp_told(alpha):
    """Same kernel as classical.gp WITHOUT the white-noise term: the noise variance of each site mean
    is supplied as alpha (on the scale normalize_y gives the kernel), as submission_scripts/classical.gp_refs."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    return GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=0, alpha=alpha)


def run_fit(ds, seed, r, learner=LEARNER):
    """GP_rows: the GP on the m*r rows (arm expanded). GP_told: the GP on the m site means with noise
    variance s2 / r supplied, s2 = pooled within-site variance of the r context readings (arm mean)."""
    warnings.filterwarnings("ignore")
    t1 = time.time()
    arm = "mean" if learner == TOLD else "expanded"
    U, Xc, yc, XQc = _tables(ds, seed, r, arm)
    try:
        if learner == TOLD:
            s2 = float(np.mean(U["Z"][:, :r].var(axis=1, ddof=1)))
            v = float(np.std(yc)) ** 2
            a = max(s2 / r / (v if v > 0 else 1.0), 1e-10)
            pred = gp_told(a).fit(Xc, yc).predict(XQc)
        else:
            pred = gp().fit(Xc, yc).predict(XQc)
        pred = np.asarray(pred, float).ravel()
        if not np.all(np.isfinite(pred)):
            raise FloatingPointError("non-finite prediction")
        return RC.score(U, r, arm, learner, pred, len(yc), time.time() - t1)
    except Exception as e:
        return RC.fail_row(U, r, arm, learner, len(yc), time.time() - t1, "%s: %s" % (type(e).__name__, e))


def verify_fit(ds, r):
    warnings.filterwarnings("ignore")
    U, Xc, yc, XQc = _tables(ds, 0, r, "mean")
    pred = np.asarray(gp().fit(Xc, yc).predict(XQc), float).ravel()
    row = RC.score(U, r, "mean", "GP", pred, len(yc), 0.0)
    return ds, r, float(row[RC.COLS.index("mse")])


def verify(datasets, a):
    from joblib import Parallel, delayed
    ref = {}
    for p in glob.glob(a.ref):
        with open(p, newline="") as fh:
            for x in csv.DictReader(fh):
                if x["learner"] == "GP" and x["arm"] == "mean" and x["seed"] == "0" and x["status"] == "ok":
                    ref[(x["dataset"], int(x["r"]))] = float(x["mse"])
    jobs = [(d, r) for d in datasets for r in RC.rlist(d)[1:] if (d, r) in ref]
    print("verify: %d stored GP mean-arm fits of seed 0 against %d reference rows" % (len(jobs), len(ref)), flush=True)
    worst, bad = 0.0, 0
    for d, r, v in Parallel(n_jobs=a.jobs)(delayed(verify_fit)(d, r) for d, r in jobs):
        rel = abs(v - ref[(d, r)]) / max(abs(ref[(d, r)]), 1e-12)
        worst = max(worst, rel)
        if rel > a.verify_tol:
            bad += 1
            print("  MISMATCH %s r=%d  here %.8g  stored %.8g  rel %.2e" % (d, r, v, ref[(d, r)], rel), flush=True)
    ok = bad == 0 and len(jobs) > 0
    print("VERIFY %s (%d fits, %d beyond tol %.0e, worst %.2e)" % ("OK" if ok else "FAILED", len(jobs), bad,
                                                                   a.verify_tol, worst), flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(RC.DEFAULT_DATASETS))
    p.add_argument("--seeds", default="0-19")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--ref", default="")
    p.add_argument("--verify-tol", type=float, default=1e-3)
    a = p.parse_args()
    datasets = a.datasets.split(",")
    if a.verify:
        sys.exit(0 if verify(datasets, a) else 1)
    tasks, skipped = [], []
    for s in parse_seeds(a.seeds):
        for d in datasets:
            for r in RC.rlist(d)[1:]:
                n = RC.n_rows(d, r, "expanded")
                (tasks if n <= a.max_rows else skipped).append((s, d, r, n, LEARNER))
                tasks.append((s, d, r, RC.n_rows(d, r, "mean"), TOLD))
    by_n = {}
    for t in tasks:
        if t[4] == LEARNER:
            by_n[t[3]] = by_n.get(t[3], 0) + 1
    n_told = sum(1 for t in tasks if t[4] == TOLD)
    print("realdata gp_rows | GP on rows: %d fits, %d skipped (rows > %d) | GP on means with noise told: %d | row fits by row count: %s"
          % (len(tasks) - n_told, len(skipped), a.max_rows, n_told, "  ".join("%d:%d" % kv for kv in sorted(by_n.items()))), flush=True)
    if a.estimate:
        return
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "gprows_s0of1.csv")
    done = RC.done_keys(glob.glob(os.path.join(OUT, "gprows_*.csv")))
    tasks = [t for t in tasks if (t[1], t[4], t[0], t[2], "mean" if t[4] == TOLD else "expanded") not in done]
    tasks.sort(key=lambda t: (t[0], -t[3]))                                                              
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    fh = open(out, "a", newline="")
    w = csv.writer(fh, lineterminator="\n")
    if new:
        w.writerow(RC.COLS); fh.flush()
    print("  to do %d | already done %d | jobs %d (1 BLAS thread each) | cpu_count %s -> %s"
          % (len(tasks), len(done), a.jobs, os.cpu_count(), out), flush=True)
    from joblib import Parallel, delayed
    t0 = time.time(); nok = nfail = 0
    it = Parallel(n_jobs=a.jobs, return_as="generator_unordered", batch_size=1, pre_dispatch="2*n_jobs")(
        delayed(run_fit)(d, s, r, L) for s, d, r, _, L in tasks)
    for i, row in enumerate(it):
        w.writerow(row); fh.flush()
        if row[-1] == "ok":
            nok += 1
        else:
            nfail += 1
            print("FAIL %s seed %s r %s: %s" % (row[0], row[2], row[3], row[-1]), flush=True)
        if (i + 1) % 25 == 0 or i + 1 == len(tasks):
            print("  [%d/%d fits] ok %d fail %d t=%.0fs" % (i + 1, len(tasks), nok, nfail, time.time() - t0), flush=True)
    fh.close()
    print("REALDATA GP_ROWS DONE ok %d fail %d %.0fs" % (nok, nfail, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()


import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rdcore as RC
import interfaces as IF

OUT = os.environ.get("RD_OUT", os.path.join(HERE, "out"))


def gp():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    k = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e2), nu=2.5)
         + WhiteKernel(1e-2, (1e-8, 1e2)))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=0)


def learners():
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    return {"GP": gp,
            "GBM": lambda: GradientBoostingRegressor(random_state=0),
            "RF": lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1),
            "NN": lambda: MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, random_state=0)}


def run_unit(ds, seed, done):
    warnings.filterwarnings("ignore")
    U = RC.unit(ds, seed)
    xm, xs = U["XL"].mean(0), U["XL"].std(0)
    xs[xs == 0] = 1.0
    XL = (U["XL"] - xm) / xs
    XQall = (np.vstack([U["XQ"], U["XL"]]) - xm) / xs
    rows = []
    for name, mk in learners().items():
        for (r, arm) in RC.fits(ds):
            if (ds, name, seed, r, arm) in done:
                continue
            Xc, yc, XQc = IF.build(arm, XL, U["Z"][:, :r], XQall)
            t1 = time.time()
            try:
                if name == "GP" and arm == "expanded":
                    Xg, yg, _ = IF.build("mean", XL, U["Z"][:, :r], XQall)                                 
                    pred = mk().fit(Xg, yg).predict(XQc)
                else:
                    pred = mk().fit(Xc, yc).predict(XQc)
                pred = np.asarray(pred, float).ravel()
                if not np.all(np.isfinite(pred)):
                    raise FloatingPointError("non-finite prediction")
                row = RC.score(U, r, arm, name, pred, len(yc), time.time() - t1)
            except Exception as e:
                row = RC.fail_row(U, r, arm, name, len(yc), time.time() - t1, "%s: %s" % (type(e).__name__, e))
            rows.append(row)
    return ds, seed, rows


def parse_seeds(s):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main():
    from joblib import Parallel, delayed
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(RC.DEFAULT_DATASETS))
    p.add_argument("--seeds", default="0-19")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "classical_s%dof%d.csv" % (a.shard, a.of))
    done = RC.done_keys([os.path.join(OUT, f) for f in os.listdir(OUT) if f.startswith("classical_") and f.endswith(".csv")])
    units = [(d, s) for s in parse_seeds(a.seeds) for d in a.datasets.split(",")]
    units = [u for i, u in enumerate(units) if i % a.of == a.shard]
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    fh = open(out, "a", newline="")
    w = csv.writer(fh, lineterminator="\n")
    if new:
        w.writerow(RC.COLS); fh.flush()
    print("classical | shard %d/%d | %d units | %d fits done | jobs %d -> %s" % (a.shard, a.of, len(units), len(done),
                                                                             a.jobs, out), flush=True)
    t0 = time.time(); k = 0
    import joblib
    if tuple(int(x) for x in joblib.__version__.split(".")[:2]) >= (1, 4):
        gen = Parallel(n_jobs=a.jobs, return_as="generator_unordered")(delayed(run_unit)(d, s, done) for d, s in units)
    else:                                                                                           
        def batches():
            for i in range(0, len(units), a.jobs):
                for res in Parallel(n_jobs=a.jobs)(delayed(run_unit)(d, s, done) for d, s in units[i:i + a.jobs]):
                    yield res
        gen = batches()
    for ds, seed, rows in gen:
        for row in rows:
            w.writerow(row)
            if row[-1] != "ok":
                print("FAIL %s %s seed %s r %s %s: %s" % (row[1], ds, seed, row[3], row[4], row[-1]), flush=True)
        fh.flush(); k += 1
        print("  [%d/%d] %s seed %d (%d rows) t=%.0fs" % (k, len(units), ds, seed, len(rows), time.time() - t0), flush=True)
    fh.close()
    print("CLASSICAL DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

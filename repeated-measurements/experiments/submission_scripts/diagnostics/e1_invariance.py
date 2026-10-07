
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
import designs_sub as DS              
import grid              

LEARNERS = ["OLS", "KRR", "TREE", "GBM", "GPfix", "RF", "RFnoboot", "NN", "GP"]
RCONFS = ["4", "64", "random", "clustered"]
COLS = ["family", "func", "truth", "seed", "rconf", "r_total", "learner", "weighted_ok",
        "maxdiff_wmean", "maxdiff_umean", "site_maxdiff_wmean", "site_maxdiff_umean", "mse_rows", "mse_wmean", "mse_umean", "secs", "key"]


def cells():
    out = []
    for s in grid.SEEDS_SYNTH:                                   
        for f in grid.FUNCS_1D:
            out.append(dict(family="synth1d", func=f, truth="", d=1, m=grid.M_DEF, spacing="random",
                            sig=grid.SIG_DEF, seed=s))
    for i, s in enumerate(grid.SEEDS_SEMI):
        for ds_, fh in grid.SEMI:
            out.append(dict(family="semisynth", func=ds_, truth=fh, d=grid.SEMI_D[ds_], m=grid.M_DEF,
                            spacing="", sig=grid.SIGMULT_DEF, seed=s))
    return out


def rvec_of(ds, rconf, seed):
    if rconf in ("random", "clustered"):
        return DS.r_vector(ds, rconf, seed)
    return np.full(len(ds["X"]), int(rconf))


def _gp_learned():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    k = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e2), nu=2.5)
         + WhiteKernel(1e-2, (1e-8, 1e2)))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=0)


def _gp_alpha(alpha, fixed):
    """GP without a white-noise term and with the noise given as alpha (per row)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    if fixed:                                                                                  
        k = ConstantKernel(1.0, "fixed") * Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5)
        return GaussianProcessRegressor(kernel=k, alpha=alpha, optimizer=None, normalize_y=False)
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    return GaussianProcessRegressor(kernel=k, alpha=alpha, normalize_y=True, random_state=0)


def make(name):
    from sklearn.linear_model import LinearRegression
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    return {"OLS": lambda: LinearRegression(),
            "KRR": lambda: KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5),
            "TREE": lambda: DecisionTreeRegressor(random_state=0),
            "GBM": lambda: GradientBoostingRegressor(random_state=0),
            "RF": lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1),
            "RFnoboot": lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1,
                                                      bootstrap=False),
            "NN": lambda: MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, random_state=0)}[name]()


def fit_predict(name, X, y, Xq, w=None):
    """Returns (prediction, weighted_ok). w = per-row sample weights or None."""
    if w is None:
        return make(name).fit(X, y).predict(Xq), True
    try:
        return make(name).fit(X, y, sample_weight=w).predict(Xq), True
    except TypeError:                                                                            
        return make(name).fit(X, y).predict(Xq), False


def run_cell(c, truthdir, done):
    recs, fails = [], []
    try:
        ds = DS.make(c, truthdir=truthdir)
    except Exception as e:
        return recs, ["FAIL design %s %s: %s" % (c["func"], c["seed"], e)]
    X, Xq, Y = ds["X"], ds["Xq"], ds["Y"]
    if c["family"] == "semisynth":                                                                     
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        X, Xq = (X - mu) / sd, (Xq - mu) / sd
    sig2 = float(ds["sigma_abs"]) ** 2
    site = ds["region"] == "site"
    for rconf in RCONFS:
        rv = rvec_of(ds, rconf, int(c["seed"]))
        Xr = np.repeat(X, rv, axis=0)
        yr = np.concatenate([Y[i, :rv[i]] for i in range(len(X))])
        ybar = np.array([Y[i, :rv[i]].mean() for i in range(len(X))])
        num = sum((rv[i] - 1) * Y[i, :rv[i]].var(ddof=1) for i in range(len(X)) if rv[i] >= 2)
        den = sum(rv[i] - 1 for i in range(len(X)) if rv[i] >= 2)
        s2 = num / den if den > 0 else 0.0
        for L in LEARNERS:
            key = "%s|%s|%s|%s|%s" % (c["family"], c["func"], c["seed"], rconf, L)
            if key in done:
                continue
            t0 = time.time()
            try:
                ok = True
                if L == "GPfix":
                    a = max(sig2, 1e-8)
                    p_rows = _gp_alpha(a, True).fit(Xr, yr).predict(Xq)
                    p_w = _gp_alpha(a / rv, True).fit(X, ybar).predict(Xq)
                    p_u = _gp_alpha(a, True).fit(X, ybar).predict(Xq)
                elif L == "GP":
                    p_rows = _gp_learned().fit(Xr, yr).predict(Xq)
                    v = float(np.std(ybar)) ** 2 or 1.0                                               
                    p_w = _gp_alpha(np.maximum(s2 / rv / v, 1e-10), False).fit(X, ybar).predict(Xq)
                    p_u = _gp_learned().fit(X, ybar).predict(Xq)
                else:
                    p_rows, _ = fit_predict(L, Xr, yr, Xq)
                    p_w, ok = fit_predict(L, X, ybar, Xq, w=rv.astype(float))
                    p_u, _ = fit_predict(L, X, ybar, Xq)
                sc = {k: DS.score(ds, p)["pop"] for k, p in (("rows", p_rows), ("w", p_w), ("u", p_u))}
                recs.append(dict(family=c["family"], func=c["func"], truth=c["truth"], seed=c["seed"],
                                 rconf=rconf, r_total=int(rv.sum()), learner=L, weighted_ok=int(ok),
                                 maxdiff_wmean="%.6g" % np.max(np.abs(p_w - p_rows)),
                                 maxdiff_umean="%.6g" % np.max(np.abs(p_u - p_rows)),
                                 site_maxdiff_wmean="%.6g" % np.max(np.abs(p_w - p_rows)[site]),
                                 site_maxdiff_umean="%.6g" % np.max(np.abs(p_u - p_rows)[site]),
                                 mse_rows="%.8g" % sc["rows"], mse_wmean="%.8g" % sc["w"],
                                 mse_umean="%.8g" % sc["u"], secs="%.2f" % (time.time() - t0), key=key))
            except Exception as e:
                fails.append("FAIL %s: %s %s" % (key, type(e).__name__, str(e)[:200]))
    return recs, fails


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--truthdir", default=str(_PKG_ROOT/'data/truth'))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--families", default="synth1d,semisynth")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    import sklearn
    print("E1 | sklearn %s | numpy %s" % (sklearn.__version__, np.__version__), flush=True)
    out = os.path.join(a.outdir, "e1_invariance.csv")
    done = set()
    if os.path.exists(out):
        with open(out, newline="") as fh:
            done = {r["key"] for r in csv.DictReader(fh)}
    cs = [c for c in cells() if c["family"] in a.families.split(",")]
    if a.limit:
        cs = cs[:a.limit]
    print("cells %d x %d count configs x %d learners | already done %d -> %s"
          % (len(cs), len(RCONFS), len(LEARNERS), len(done), out), flush=True)
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
    if new:
        w.writeheader()
    from joblib import Parallel, delayed
    t0, n, nf = time.time(), 0, 0
    gen = Parallel(n_jobs=a.jobs, return_as="generator_unordered")(
        delayed(run_cell)(c, a.truthdir, done) for c in cs)
    for i, (recs, fails) in enumerate(gen):
        for r in recs:
            w.writerow(r)
        fh.flush()
        n += len(recs)
        for f in fails:
            print(f, flush=True)
        nf += len(fails)
        if (i + 1) % 20 == 0 or i + 1 == len(cs):
            print("  [%d/%d cells] rows %d fail %d t=%.0fs" % (i + 1, len(cs), n, nf, time.time() - t0),
                  flush=True)
    fh.close()
    print("E1 DONE rows %d fail %d %.0fs" % (n, nf, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

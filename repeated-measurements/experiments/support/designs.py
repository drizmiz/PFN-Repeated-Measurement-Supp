
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_AA = str(_PKG_ROOT/'experiments/support')
sys.path.insert(0, _AA)
import core              

R_MAX = 512
core.R_MAX = R_MAX                                                      
SEEDS_SYNTH = list(range(20))
SEEDS_SEMI = list(range(7101, 7107))
FUNCS_1D = ["quadratic", "abs", "linear", "tanh", "step", "sine"]
FUNCS_D = ["additive", "friedman", "step"]
DIMS = [2, 5, 10]
SIGMAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]
RS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def make(family, func=None, m=12, sigma=0.15, seed=0, d=None, dataset=None, fhat=None,
         sigmult=None, nq=400):
    if family == "synth1d":
        core.R_MAX = R_MAX
        ds = core.make_data1(func, m, sigma, seed, rmax=R_MAX)
        ds["meta"] = dict(family=family, func=func, m=m, sigma=sigma, seed=seed, d=1, rmax=R_MAX)
        return ds
    if family == "synthd":
        ds = core.make_datad(func, d, m, sigma, seed, nq=nq, rmax=R_MAX)
        ds["meta"] = dict(family=family, func=func, m=m, sigma=sigma, seed=seed, d=d, rmax=R_MAX)
        return ds
    if family == "semisynth":
        return _semisynth(dataset, fhat, m, sigmult, seed, nq=min(nq, 300))
    raise ValueError(family)


_CACHE = {}


def _semisynth(dataset, fhat, m, sigmult, seed, nq=300):
    """Exactly the construction in bias_var/semisynth_AB.py, cached per (dataset, fhat)."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn import datasets as D
    key = (dataset, fhat, nq)
    if key not in _CACHE:
        if dataset == "calhousing":
            dd = D.fetch_california_housing(); Xall, yall = dd.data, dd.target
        elif dataset == "diabetes":
            dd = D.load_diabetes(); Xall, yall = dd.data, dd.target
        elif dataset == "wine":
            dd = D.fetch_openml("wine_quality", version=1, as_frame=True)
            Xall = dd.data.select_dtypes("number").to_numpy()
            yall = dd.target.astype(float).to_numpy()
        elif dataset == "concrete":                                
            dd = D.fetch_openml(data_id=44959, as_frame=True, parser="auto")
            Xall = dd.data.select_dtypes("number").to_numpy(float); yall = dd.target.astype(float).to_numpy()
        elif dataset == "energy":                                                
            dd = D.fetch_openml(data_id=44960, as_frame=True, parser="auto")
            Xall = dd.data.select_dtypes("number").to_numpy(float); yall = dd.target.astype(float).to_numpy()
        else:
            raise ValueError(dataset)
        rng0 = np.random.default_rng(0)
        perm = rng0.permutation(len(Xall)); ntr = int(0.7 * len(Xall))
        Xtr, ytr, Xte = Xall[perm[:ntr]], yall[perm[:ntr]], Xall[perm[ntr:]]
        truth = (GradientBoostingRegressor(random_state=0).fit(Xtr, ytr) if fhat == "gbm" else
                 make_pipeline(StandardScaler(), Nystroem(n_components=300, random_state=0),
                               Ridge(alpha=1.0)).fit(Xtr, ytr))
        resid_sd = float(np.std(yall[perm[ntr:]] - truth.predict(Xte)))
        qi = rng0.choice(len(Xte), size=min(nq, len(Xte)), replace=False)
        XQ = Xte[qi]; fQ = truth.predict(XQ)
        mu_f, sd_f = float(fQ.mean()), float(fQ.std())
        _CACHE[key] = dict(Xtr=Xtr, truth=truth, XQ=XQ, fq=(fQ - mu_f) / sd_f,
                           mu_f=mu_f, sd_f=sd_f, sd_resid=resid_sd / sd_f, d=Xall.shape[1])
    c = _CACHE[key]
    sig = sigmult * c["sd_resid"]
    g = np.random.default_rng(seed)
    li = g.choice(len(c["Xtr"]), size=m, replace=False)
    XL = c["Xtr"][li]
    fL = (c["truth"].predict(XL) - c["mu_f"]) / c["sd_f"]
    Z = g.normal(0, 1, (m, R_MAX))
    return dict(d=c["d"], X=XL, Y=fL[:, None] + sig * Z, Xq=c["XQ"], fq=c["fq"],
                region=np.array(["pop"] * len(c["XQ"])),
                meta=dict(family="semisynth", dataset=dataset, fhat=fhat, m=m, sigmult=sigmult,
                          sigma=sig, seed=seed, d=c["d"], rmax=R_MAX, nq=len(c["XQ"])))


def rows(ds, r, kind):
    """single / expanded / mean, with lid mapping each context row to its site."""
    X, Y = ds["X"], ds["Y"][:, :r]
    if kind == "single":
        return X.copy(), Y[:, 0].copy(), np.arange(len(X))
    if kind == "expanded":
        return np.repeat(X, r, axis=0), Y.reshape(-1).copy(), np.repeat(np.arange(len(X)), r)
    if kind == "mean":
        return X.copy(), Y.mean(1), np.arange(len(X))
    raise ValueError(kind)


def score(ds, pred):
    return core.mse_by_region(ds, np.asarray(pred, float))


if __name__ == "__main__":
    ds = make("synth1d", "quadratic", 12, 0.15, 0)
    assert ds["Y"].shape == (12, R_MAX)
    Xc, yc, lid = rows(ds, 64, "expanded"); assert len(yc) == 768 and lid[63] == 0 and lid[64] == 1
    dd = make("synthd", "friedman", 24, 0.15, 0, d=5); assert dd["X"].shape == (24, 5)
    print("designs.py ok: synth1d Y", ds["Y"].shape, "| synthd X", dd["X"].shape, "| R_MAX", core.R_MAX)

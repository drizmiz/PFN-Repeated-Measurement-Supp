
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, os
import numpy as np


def load(name):
    from sklearn import datasets as D
    if name == "calhousing":
        d = D.fetch_california_housing(); return d.data, d.target
    if name == "diabetes":
        d = D.load_diabetes(); return d.data, d.target
    if name == "wine":
        d = D.fetch_openml("wine_quality", version=1, as_frame=True)
        return d.data.select_dtypes("number").to_numpy(), d.target.astype(float).to_numpy()
    if name == "concrete":
        d = D.fetch_openml(data_id=44959, as_frame=True, parser="auto")
        return d.data.select_dtypes("number").to_numpy(float), d.target.astype(float).to_numpy()
    if name == "energy":
        d = D.fetch_openml(data_id=44960, as_frame=True, parser="auto")
        return d.data.select_dtypes("number").to_numpy(float), d.target.astype(float).to_numpy()
    raise SystemExit("unknown dataset " + name)


def fit_truth(kind, X, y):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "gbm":
        return GradientBoostingRegressor(random_state=0).fit(X, y)
    return make_pipeline(StandardScaler(), Nystroem(n_components=300, random_state=0),
                         Ridge(alpha=1.0)).fit(X, y)


def setup(dataset, fhat, nquery=300):
    Xall, yall = load(dataset)
    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(len(Xall))
    ntr = int(0.7 * len(Xall))
    Xtr, ytr, Xte = Xall[perm[:ntr]], yall[perm[:ntr]], Xall[perm[ntr:]]
    truth = fit_truth(fhat, Xtr, ytr)
    resid_sd = float(np.std(yall[perm[ntr:]] - truth.predict(Xte)))
    qi = rng0.choice(len(Xte), size=min(nquery, len(Xte)), replace=False)
    XQ = Xte[qi]
    fQ = truth.predict(XQ)
    mu_f, sd_f = float(fQ.mean()), float(fQ.std())
    return dict(Xtr=Xtr, truth=truth, XQ=XQ, fq=(fQ - mu_f) / sd_f, mu_f=mu_f, sd_f=sd_f,
                sd_resid=resid_sd / sd_f, d=Xall.shape[1])


if __name__ == "__main__":
    import sklearn
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", default="diabetes:gbm,wine:gbm,calhousing:gbm,concrete:gbm,energy:kernel")
    p.add_argument("--only-missing", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("sklearn", sklearn.__version__, "numpy", np.__version__, flush=True)
    for pair in a.pairs.split(","):
        ds, fh = pair.split(":")
        path = os.path.join(a.out, "ticlapp_truth_%s_%s.npz" % (ds, fh))
        if a.only_missing and os.path.exists(path):
            print("exists, kept: %s" % path, flush=True)
            continue
        S = setup(ds, fh, 300)
        np.savez(path, Xtr=S["Xtr"], XQ=S["XQ"], predTr=S["truth"].predict(S["Xtr"]),
                 fQ=S["truth"].predict(S["XQ"]), fq=S["fq"], mu_f=S["mu_f"], sd_f=S["sd_f"],
                 sd_resid=S["sd_resid"], d=S["d"], sklearn_version=sklearn.__version__)
        print("%s %s | d=%d | sd_resid=%.6f | nq=%d | ntr=%d -> %s"
              % (ds, fh, S["d"], S["sd_resid"], len(S["XQ"]), len(S["Xtr"]), path), flush=True)
    print("EXPORT DONE")

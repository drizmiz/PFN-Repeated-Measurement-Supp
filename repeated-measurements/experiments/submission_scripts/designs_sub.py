
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core                                     

R_MAX = 512
M_MAX = 96
core.R_MAX = R_MAX
core.M_MAX = M_MAX
NQ_D = 400

R_SET_RANDOM = (1, 4, 16, 64)
R_SET_LEFT = (16, 64)
R_SET_RIGHT = (1, 4)


def make_synth1d(func, m, sigma, seed, spacing="random"):
    core.R_MAX, core.M_MAX = R_MAX, M_MAX
    if spacing == "random":
        ds = core.make_data1(func, m, sigma, seed, rmax=R_MAX)
    elif spacing == "even":
        x = np.linspace(core.LO, core.HI, m + 2)[1:-1]
        z = np.random.default_rng([seed, 23]).standard_normal((M_MAX, R_MAX))[:m]
        fx = core.f_true1(func, x)
        gap = 0.5 * (x[1:] + x[:-1])
        Xq = np.concatenate([core.POP_GRID, x, gap])
        ds = dict(d=1, func=func, m=m, sigma=sigma, seed=seed, X=x[:, None],
                  Y=fx[:, None] + sigma * z, Xq=Xq[:, None], fq=core.f_true1(func, Xq),
                  region=np.array(["pop"] * core.N_POP + ["site"] * m + ["gap"] * (m - 1)))
    else:
        raise ValueError(spacing)
    ds["fX"] = core.f_true1(func, ds["X"][:, 0])
    return ds


IRR_TAG = 4401


def f_irrelevant(X):
    mu, sd = core._moments("friedman", 2)
    return (core._raw_friedman(np.asarray(X, float)[:, :2]) - mu) / sd


def _coords(seed, tag, n, d):
    return np.column_stack([np.random.default_rng([int(seed), tag, j]).uniform(-2, 2, n)
                            for j in range(d)])


def make_synthd(func, d, m, sigma, seed, nq=NQ_D):
    if func in core.FUNCSD:
        ds = core.make_datad(func, d, m, sigma, seed, nq=nq, rmax=R_MAX)
        mu, sd = core._moments(func, d)
        ds["fX"] = (core._RAWD[func](ds["X"]) - mu) / sd
        return ds
    if func != "irrelevant":
        raise ValueError(func)
    X = _coords(seed, IRR_TAG, m, d)
    Xpop = _coords(seed, IRR_TAG + 1, nq, d)
    z = np.random.default_rng([int(seed), IRR_TAG + 2]).standard_normal((m, R_MAX))
    fx, fpop = f_irrelevant(X), f_irrelevant(Xpop)
    return dict(d=d, func=func, m=m, sigma=sigma, seed=seed, X=X, Y=fx[:, None] + sigma * z,
                Xq=np.vstack([Xpop, X]), fq=np.concatenate([fpop, fx]), fX=fx,
                region=np.array(["pop"] * nq + ["site"] * m))


SEMI_TRUTH = {"diabetes": "gbm", "wine": "gbm", "calhousing": "gbm", "concrete": "gbm",
              "energy": "kernel"}
_TRUTH = {}


def truth_path(truthdir, dataset, fhat):
    return os.path.join(truthdir, "ticlapp_truth_%s_%s.npz" % (dataset, fhat))


def load_truth(truthdir, dataset, fhat):
    key = (truthdir, dataset, fhat)
    if key not in _TRUTH:
        p = truth_path(truthdir, dataset, fhat)
        if not os.path.exists(p):
            raise FileNotFoundError("truth export missing: %s (run truth_export.py under venv3)" % p)
        ex = np.load(p)
        fQ = ex["fQ"]
        mu_f, sd_f = float(ex["mu_f"]), float(ex["sd_f"])
        assert abs(float(fQ.mean()) - mu_f) < 1e-10 and abs(float(fQ.std()) - sd_f) < 1e-10
        _TRUTH[key] = dict(Xtr=ex["Xtr"], predTr=ex["predTr"], XQ=ex["XQ"],
                           fq=(fQ - mu_f) / sd_f, mu_f=mu_f, sd_f=sd_f,
                           sd_resid=float(ex["sd_resid"]), d=int(ex["d"]),
                           sklearn=str(ex["sklearn_version"]))
    return _TRUTH[key]


def make_semisynth(dataset, fhat, m, sigmult, seed, truthdir):
    c = load_truth(truthdir, dataset, fhat)
    sig = sigmult * c["sd_resid"]
    g = np.random.default_rng(seed)
    li = g.choice(len(c["Xtr"]), size=m, replace=False)
    XL = c["Xtr"][li]
    fL = (c["predTr"][li] - c["mu_f"]) / c["sd_f"]
    Z = g.normal(0, 1, (m, R_MAX))
    nq = len(c["XQ"])
    return dict(d=c["d"], X=XL, Y=fL[:, None] + sig * Z, fX=fL,
                Xq=np.vstack([c["XQ"], XL]), fq=np.concatenate([c["fq"], fL]),
                region=np.array(["pop"] * nq + ["site"] * m), sigma_abs=sig,
                truth_sklearn=c["sklearn"])


def make(spec, truthdir=None):
    """spec: dict with family, func (or dataset), truth, d, m, spacing, sig, seed."""
    fam = spec["family"]
    if fam == "synth1d":
        ds = make_synth1d(spec["func"], int(spec["m"]), float(spec["sig"]), int(spec["seed"]),
                          spec.get("spacing", "random"))
        ds["sigma_abs"] = float(spec["sig"])
    elif fam == "synthd":
        ds = make_synthd(spec["func"], int(spec["d"]), int(spec["m"]), float(spec["sig"]),
                         int(spec["seed"]))
        ds["sigma_abs"] = float(spec["sig"])
    elif fam == "semisynth":
        ds = make_semisynth(spec["func"], spec["truth"], int(spec["m"]), float(spec["sig"]),
                            int(spec["seed"]), truthdir)
    else:
        raise ValueError(fam)
    return ds


def r_by_rank(scheme, seed, m):
    """Readings per RANK of the first coordinate (rank 0 = leftmost site)."""
    if scheme == "random":
        return np.random.default_rng([int(seed), 77]).choice(R_SET_RANDOM, size=m)
    if scheme == "clustered":
        g = np.random.default_rng([int(seed), 78])
        h = m // 2
        return np.concatenate([g.choice(R_SET_LEFT, size=h), g.choice(R_SET_RIGHT, size=m - h)])
    raise ValueError(scheme)


def r_total(scheme, seed, m):
    return int(r_by_rank(scheme, seed, m).sum())


def r_balanced(matched_to, seed, m):
    return int(np.floor(r_total(matched_to, seed, m) / m + 0.5))


def r_vector(ds, scheme, seed):
    """Per-site readings in the SITE order of ds['X'] (assigned by rank of the first coordinate)."""
    m = len(ds["X"])
    rr = r_by_rank(scheme, seed, m)
    ranks = np.argsort(np.argsort(ds["X"][:, 0], kind="stable"), kind="stable")
    return rr[ranks].astype(int)


def rows(ds, arm, r=None, rvec=None):
    """single / expanded / mean. Equal r: pass r. Unequal: pass rvec (site i uses Y[i, :r_i])."""
    X, Y = ds["X"], ds["Y"]
    if arm == "single":
        return X.copy(), Y[:, 0].copy()
    if rvec is None:
        rvec = np.full(len(X), int(r))
    rvec = np.asarray(rvec, int)
    if arm == "expanded":
        return (np.repeat(X, rvec, axis=0),
                np.concatenate([Y[i, :rvec[i]] for i in range(len(X))]))
    if arm == "mean":

                                                                                               

        return X.copy(), np.array([Y[i, 0] if np.all(Y[i, :rvec[i]] == Y[i, 0])
                                   else Y[i, :rvec[i]].mean() for i in range(len(X))])
    raise ValueError(arm)


def score(ds, pred):
    out = core.mse_by_region(ds, np.asarray(pred, float).ravel())
    return {k: out.get(k, float("nan")) for k in ("pop", "site", "gap")}

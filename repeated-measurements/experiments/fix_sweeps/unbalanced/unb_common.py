
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import math
import numpy as np

M = 24
FUNCS = ["quadratic", "tanh", "step", "sine", "abs"]
SIGMAS = [0.05, 0.15, 0.30]
SEEDS = list(range(8))
DESIGNS = ["alt", "clust", "bal"]
TFM_TABLE_ARMS = ["single", "expanded", "mean", "mean_dup", "kmeans_2"]
HOOK_ARMS = ["mean_logw", "mean_logw_prec", "mean_logw_both", "mean_logw_b025", "mean_logw_b05"]
ARMS = TFM_TABLE_ARMS + HOOK_ARMS + ["gp_ref"]


def r_vec(design, m=M):
    i = np.arange(m)
    if design == "alt":
        return np.where(i % 2 == 0, 64, 1)
    if design == "clust":
        return np.where(i < m // 2, 64, 1)
    if design == "bal":
        return np.full(m, 32)
    raise ValueError(design)


def site_readings(ds, r):
    """list of length m; entry i = the first r_i readings at site i (nested)."""
    return [ds["Y"][i, : int(r[i])] for i in range(len(r))]


def pooled_sigma2(reads):
    ss, df = 0.0, 0
    for y in reads:
        if len(y) >= 2:
            ss += float(np.sum((y - y.mean()) ** 2))
            df += len(y) - 1
    if df == 0:
        raise RuntimeError("no site with r_i >= 2")
    return ss / df


def build_table(name, X, reads):
    """-> (Xc, yc) for the table-side arms. X is (m, 1); reads a list of 1-D arrays."""
    m = len(reads)
    r = np.array([len(y) for y in reads])
    ybar = np.array([y.mean() for y in reads])
    if name == "single":
        return X.copy(), np.array([y[0] for y in reads])
    if name == "expanded":
        return np.repeat(X, r, axis=0), np.concatenate(reads)
    if name == "mean":
        return X.copy(), ybar
    if name == "mean_dup":
        return np.repeat(X, r, axis=0), np.repeat(ybar, r)
    if name == "kmeans_2":
        rows, ys = [], []
        for i in range(m):
            y = reads[i]
            if len(y) >= 2:
                for g in np.array_split(np.arange(len(y)), 2):
                    rows.append(X[i]); ys.append(y[g].mean())
            else:
                rows.append(X[i]); ys.append(y[0])
        return np.vstack(rows), np.array(ys)
    raise ValueError(name)


def gp_ref_predict(X, ybar, r, sigma2, Xq, seed=0):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    kern = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    alpha = sigma2 / np.asarray(r, float)
    gp = GaussianProcessRegressor(kernel=kern, alpha=alpha, normalize_y=False,
                                  n_restarts_optimizer=4, random_state=seed)
    gp.fit(X, ybar)
    return gp.predict(Xq), str(gp.kernel_)


def score(ds, pred, r):
    """MSE against truth by region, plus the nearest-site high-r / low-r split."""
    from core import mse_by_region
    out = mse_by_region(ds, pred)
    x_sites = ds["X"][:, 0]
    xq = ds["Xq"][:, 0]
    nearest = np.abs(xq[:, None] - x_sites[None, :]).argmin(1)
    rn = np.asarray(r)[nearest]
    err2 = (pred - ds["fq"]) ** 2
    reg = ds["region"]
    rmax, rmin = int(np.max(r)), int(np.min(r))
    for region in ("pop", "site"):
        k = reg == region
        hi = k & (rn == rmax)
        lo = k & (rn == rmin)
        out["%s_hi" % region] = float(err2[hi].mean()) if hi.any() else float("nan")
        out["%s_lo" % region] = float(err2[lo].mean()) if (lo.any() and rmin != rmax) else float("nan")
    return out


CSV_COLS = ["func", "sigma", "seed", "design", "arm", "nrows", "sigma2_hat",
            "mse_pop", "mse_site", "mse_gap", "mse_pop_hi", "mse_pop_lo", "mse_site_hi", "mse_site_lo",
            "secs", "note"]

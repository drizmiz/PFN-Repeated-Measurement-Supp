from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import math
import numpy as np

LO, HI = -2.0, 2.0
M_RANGE = (8, 32)
R_MAX = 64
P_COMMON_R = 0.70
SIGMA_RANGE = (0.03, 1.0)
KINDS = ["expanded", "mean", "plain"]
P_KIND = [0.55, 0.15, 0.30]
FAMILIES = ["rbf", "poly", "gp_long"]
P_FAMILY = [0.40, 0.30, 0.30]
P_STEP = 0.20
GP_LONG_LS = (0.8, 2.0)
GP_SHORT_LS = (0.15, 0.40)                                      
GP_GRID = np.linspace(-2.1, 2.1, 256)
NQ = 100
Q_MARGIN = 0.10
Q_MIN = 8


def _gp_draw(rng, ls):
    """GP-RBF draw on GP_GRID; evaluated anywhere through the exact (smooth) kernel interpolant
    f(x) = K(x, grid) K(grid, grid)^-1 g, so no linear-interpolation kinks."""
    d = GP_GRID[:, None] - GP_GRID[None, :]
    K = np.exp(-0.5 * d ** 2 / ls ** 2) + 1e-6 * np.eye(len(GP_GRID))
    L = np.linalg.cholesky(K)
    g = L @ rng.standard_normal(len(GP_GRID))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, g))
    return lambda x: np.exp(-0.5 * (np.asarray(x, float)[:, None] - GP_GRID[None, :]) ** 2 / ls ** 2) @ alpha


def sample_function(rng, family=None):
    """-> (f: callable on 1-D arrays, meta). family in FAMILIES or 'gp_short' (validation)."""
    family = family or str(rng.choice(FAMILIES, p=P_FAMILY))
    meta = dict(family=family)
    if family == "rbf":
        k = int(rng.integers(2, 6)); c = rng.uniform(LO, HI, k)
        w = np.exp(rng.uniform(math.log(0.3), math.log(1.5), k)); a = rng.standard_normal(k); b = 0.5 * rng.standard_normal()
        base = lambda x: (a[None, :] * np.exp(-0.5 * (x[:, None] - c[None, :]) ** 2 / w[None, :] ** 2)).sum(1) + b * x
        meta.update(k=k)
    elif family == "poly":
        D = int(rng.integers(1, 4)); coef = rng.standard_normal(D + 1)
        base = lambda x: sum(coef[j] * (x / 2.0) ** j for j in range(D + 1))
        meta.update(degree=D)
    elif family == "gp_long":
        ls = float(rng.uniform(*GP_LONG_LS)); base = _gp_draw(rng, ls); meta.update(ls=ls)
    elif family == "gp_short":
        ls = float(rng.uniform(*GP_SHORT_LS)); base = _gp_draw(rng, ls); meta.update(ls=ls)
    else:
        raise ValueError(family)
    has_step = bool(rng.random() < P_STEP) if family != "gp_short" else False
    c0 = float(rng.uniform(-1.5, 1.5)) if has_step else 0.0
    s = float(rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])) if has_step else 0.0

    def raw(x):
        x = np.asarray(x, float); out = base(x)
        if has_step:
            out = out + s * (x >= c0)
        return out

    ref = raw(rng.uniform(LO, HI, 2000))
    mu, sd = float(ref.mean()), max(float(ref.std()), 1e-6)
    f = lambda x: (raw(x) - mu) / sd
    meta.update(has_step=has_step, family_tag=family + ("+step" if has_step else ""))
    return f, meta


def draw_r(rng, size, r_max):
    """integer r in {1..r_max}, log-uniform: round(exp(U[0, log r_max]))."""
    return np.clip(np.rint(np.exp(rng.uniform(0.0, math.log(r_max), size))), 1, r_max).astype(int)


def make_table(rng, kind=None, n_query=NQ, r_fixed=None, r_max=R_MAX, sigma_fixed=None, family=None):
    kind = kind or str(rng.choice(KINDS, p=P_KIND))
    m = int(rng.integers(M_RANGE[0], M_RANGE[1] + 1))
    x = np.sort(rng.uniform(LO, HI, m))
    f, fmeta = sample_function(rng, family)
    sigma = float(sigma_fixed) if sigma_fixed is not None else float(np.exp(rng.uniform(math.log(SIGMA_RANGE[0]), math.log(SIGMA_RANGE[1]))))
    if kind == "plain":
        r = np.ones(m, int); r_mode = "plain"
    elif r_fixed is not None:
        r = np.full(m, int(r_fixed)); r_mode = "common"
    elif rng.random() < P_COMMON_R:
        rc = 1
        while rc == 1:
            rc = int(draw_r(rng, 1, r_max)[0])
        r = np.full(m, rc); r_mode = "common"
    else:
        r = draw_r(rng, m, r_max); r_mode = "persite"
        if (r == 1).all():
            r[rng.integers(m)] = 2
    fx = f(x)
    reads = [fx[i] + sigma * rng.standard_normal(int(r[i])) for i in range(m)]
    if kind == "expanded" or kind == "plain":
        X_ctx = np.repeat(x, r)[:, None]; y_ctx = np.concatenate(reads); site_id = np.repeat(np.arange(m), r)
    else:
        X_ctx = x[:, None]; y_ctx = np.array([yy.mean() for yy in reads]); site_id = np.arange(m)
    cand = rng.uniform(LO, HI, 2 * n_query); f_cand = f(cand)
    rg = y_ctx.max() - y_ctx.min(); lo_y, hi_y = y_ctx.min() - Q_MARGIN * rg, y_ctx.max() + Q_MARGIN * rg
    inside = (f_cand >= lo_y) & (f_cand <= hi_y)
    keep = np.flatnonzero(inside)[:n_query]; n_clipped = 0
    if len(keep) < Q_MIN:
        viol = np.maximum(lo_y - f_cand, f_cand - hi_y); order = np.argsort(viol)
        extra = np.array([i for i in order if i not in set(keep.tolist())])[: Q_MIN - len(keep)]
        n_clipped = len(extra); keep = np.concatenate([keep, extra]).astype(int)
    Xq = cand[keep][:, None]; yq = np.clip(f_cand[keep], lo_y, hi_y)
    meta = dict(kind=kind, m=m, n_rows=len(y_ctx), nq=len(keep), r=r, r_mode=r_mode, r_common=int(r[0]) if r_mode != "persite" else -1,
                r_max=int(r.max()), sigma=sigma, x_sites=x, f_sites=fx, site_id=site_id, n_dropped=int((~inside).sum()), n_clipped=n_clipped, **fmeta)
    return dict(X_ctx=X_ctx.astype(np.float64), y_ctx=y_ctx.astype(np.float64), X_query=Xq.astype(np.float64), y_query=yq.astype(np.float64), meta=meta)


def table_to_xy(tab):
    X = np.vstack([tab["X_ctx"], tab["X_query"]]); y = np.concatenate([tab["y_ctx"], tab["y_query"]])
    return X, y, len(tab["y_ctx"])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("repeat_prior_1b.py unit test")
    for t in range(10):
        tab = make_table(rng, r_max=4 if t < 5 else 64); mt = tab["meta"]; X, y, m = table_to_xy(tab)
        assert X.shape == (mt["n_rows"] + mt["nq"], 1) and np.isfinite(X).all() and np.isfinite(y).all()
        rg = y[:m].max() - y[:m].min()
        assert y[m:].min() >= y[:m].min() - Q_MARGIN * rg - 1e-12 and y[m:].max() <= y[:m].max() + Q_MARGIN * rg + 1e-12
        if t < 5: assert mt["r_max"] <= 4
        if mt["kind"] == "expanded":
            assert mt["n_rows"] == mt["r"].sum() and len(np.unique(tab["X_ctx"])) == mt["m"]
            yb = np.array([tab["y_ctx"][mt["site_id"] == i].mean() for i in range(mt["m"])])
            assert (np.abs(yb - mt["f_sites"]) <= 5 * mt["sigma"] / np.sqrt(mt["r"]) + 1e-9).all()
        if mt["kind"] == "plain": assert mt["n_rows"] == mt["m"] and (mt["r"] == 1).all()
        print("  t=%d kind=%-8s family=%-12s m=%2d r_mode=%-7s r=%-3d rmax=%-3d n_rows=%4d nq=%3d sigma=%.3f y_ctx sd=%.2f dropped=%d clipped=%d" % (
            t, mt["kind"], mt["family_tag"], mt["m"], mt["r_mode"], mt["r_common"], mt["r_max"], mt["n_rows"], mt["nq"], mt["sigma"], tab["y_ctx"].std(), mt["n_dropped"], mt["n_clipped"]))
    v = make_table(np.random.default_rng(1), kind="expanded", r_fixed=64, family="gp_short"); assert v["meta"]["family"] == "gp_short" and v["meta"]["r_common"] == 64 and v["meta"]["ls"] <= 0.4
    p = make_table(np.random.default_rng(2), kind="plain", sigma_fixed=0.05, family="poly"); assert p["meta"]["sigma"] == 0.05 and p["meta"]["family"] == "poly"
    a = make_table(np.random.default_rng(5)); b = make_table(np.random.default_rng(5)); assert np.array_equal(a["X_ctx"], b["X_ctx"]) and np.array_equal(a["y_query"], b["y_query"])
    xs = np.linspace(-2, 2, 400)
    for fam in ["poly", "gp_long", "gp_short", "rbf"]:
        rr = [float(np.abs(np.diff(sample_function(np.random.default_rng(100 + i), fam)[0](xs), 2)).mean()) for i in range(20)]
        print("  family %-8s mean |2nd diff| %.4f" % (fam, np.mean(rr)))
    rng = np.random.default_rng(1); n = 1000; kinds = {}; fams = {}; rmodes = {}; rows = []; steps = 0; rc = []
    for _ in range(n):
        mt = make_table(rng)["meta"]; kinds[mt["kind"]] = kinds.get(mt["kind"], 0) + 1; fams[mt["family"]] = fams.get(mt["family"], 0) + 1
        rmodes[mt["r_mode"]] = rmodes.get(mt["r_mode"], 0) + 1; rows.append(mt["n_rows"]); steps += mt["has_step"]
        if mt["r_mode"] == "common": rc.append(mt["r_common"])
    print("  mix over %d tables: kinds %s (target 0.55/0.15/0.30); families %s (target 0.40/0.30/0.30); r_mode %s; step share %.3f (target 0.20)" % (
        n, {k: round(v / n, 3) for k, v in kinds.items()}, {k: round(v / n, 3) for k, v in fams.items()}, rmodes, steps / n))
    print("  rows: mean %.0f, median %.0f, max %d; common r: median %d, mean %.1f, distinct %d" % (np.mean(rows), np.median(rows), max(rows), np.median(rc), np.mean(rc), len(set(rc))))
    print("repeat_prior_1b.py OK")

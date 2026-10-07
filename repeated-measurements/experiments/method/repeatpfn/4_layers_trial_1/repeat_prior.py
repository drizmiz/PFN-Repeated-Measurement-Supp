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
R_CHOICES = [1, 2, 4, 8, 16, 32, 64]
P_COMMON_R = 0.70
SIGMA_RANGE = (0.03, 1.0)
KINDS = ["expanded", "mean", "plain"]
P_KIND = [0.55, 0.15, 0.30]
P_STEP = 0.20
NQ = 100
Q_MARGIN = 0.10
Q_MIN = 8


def sample_function(rng):
    """-> (f: callable on 1-D arrays, meta). Random RBF bumps + linear trend (+ optional step),
    standardised on a uniform reference sample."""
    k = int(rng.integers(2, 6))
    c = rng.uniform(LO, HI, k)
    w = np.exp(rng.uniform(math.log(0.3), math.log(1.5), k))
    a = rng.standard_normal(k)
    b = 0.5 * rng.standard_normal()
    has_step = bool(rng.random() < P_STEP)
    c0 = float(rng.uniform(-1.5, 1.5)) if has_step else 0.0
    s = float(rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])) if has_step else 0.0

    def raw(x):
        x = np.asarray(x, float)
        out = (a[None, :] * np.exp(-0.5 * (x[:, None] - c[None, :]) ** 2 / w[None, :] ** 2)).sum(1) + b * x
        if has_step:
            out = out + s * (x >= c0)
        return out

    ref = raw(rng.uniform(LO, HI, 2000))
    mu, sd = float(ref.mean()), max(float(ref.std()), 1e-6)
    f = lambda x: (raw(x) - mu) / sd
    return f, dict(family="rbf%d%s" % (k, "+step" if has_step else ""), k=k, has_step=has_step)


def make_table(rng, kind=None, n_query=NQ, r_fixed=None):
    """kind / r_fixed force the table kind / a common r (used for the fixed validation set)."""
    kind = kind or str(rng.choice(KINDS, p=P_KIND))
    m = int(rng.integers(M_RANGE[0], M_RANGE[1] + 1))
    x = np.sort(rng.uniform(LO, HI, m))
    f, fmeta = sample_function(rng)
    sigma = float(np.exp(rng.uniform(math.log(SIGMA_RANGE[0]), math.log(SIGMA_RANGE[1]))))
    if kind == "plain":
        r = np.ones(m, int); r_mode = "plain"
    elif r_fixed is not None:
        r = np.full(m, int(r_fixed)); r_mode = "common"
    elif rng.random() < P_COMMON_R:
        r = np.full(m, int(rng.choice(R_CHOICES[1:]))); r_mode = "common"
    else:
        r = rng.choice(R_CHOICES, size=m).astype(int); r_mode = "persite"
        if (r == 1).all():
            r[rng.integers(m)] = 2
    fx = f(x)
    reads = [fx[i] + sigma * rng.standard_normal(int(r[i])) for i in range(m)]
    if kind == "expanded" or kind == "plain":
        X_ctx = np.repeat(x, r)[:, None]; y_ctx = np.concatenate(reads)
        site_id = np.repeat(np.arange(m), r)
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
                sigma=sigma, x_sites=x, f_sites=fx, site_id=site_id, n_dropped=int((~inside).sum()), n_clipped=n_clipped, **fmeta)
    return dict(X_ctx=X_ctx.astype(np.float64), y_ctx=y_ctx.astype(np.float64),
                X_query=Xq.astype(np.float64), y_query=yq.astype(np.float64), meta=meta)


def table_to_xy(tab):
    X = np.vstack([tab["X_ctx"], tab["X_query"]]); y = np.concatenate([tab["y_ctx"], tab["y_query"]])
    return X, y, len(tab["y_ctx"])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("repeat_prior.py unit test")
    for t in range(8):
        tab = make_table(rng); mt = tab["meta"]; X, y, m = table_to_xy(tab)
        assert X.shape == (mt["n_rows"] + mt["nq"], 1) and np.isfinite(X).all() and np.isfinite(y).all()
        rg = y[:m].max() - y[:m].min()
        assert y[m:].min() >= y[:m].min() - Q_MARGIN * rg - 1e-12 and y[m:].max() <= y[:m].max() + Q_MARGIN * rg + 1e-12
        if mt["kind"] == "expanded":
            assert mt["n_rows"] == mt["r"].sum() and len(np.unique(tab["X_ctx"])) == mt["m"]
            yb = np.array([tab["y_ctx"][mt["site_id"] == i].mean() for i in range(mt["m"])])
            assert (np.abs(yb - mt["f_sites"]) <= 5 * mt["sigma"] / np.sqrt(mt["r"]) + 1e-9).all()
        if mt["kind"] == "plain":
            assert mt["n_rows"] == mt["m"] and (mt["r"] == 1).all()
        if mt["kind"] == "mean":
            assert mt["n_rows"] == mt["m"]
        print("  t=%d kind=%-8s family=%-10s m=%2d r_mode=%-7s r=%-3d n_rows=%4d nq=%3d sigma=%.3f  y_ctx sd=%.2f  f_q sd=%.2f  dropped=%d clipped=%d" % (
            t, mt["kind"], mt["family"], mt["m"], mt["r_mode"], mt["r_common"], mt["n_rows"], mt["nq"], mt["sigma"], tab["y_ctx"].std(), tab["y_query"].std(), mt["n_dropped"], mt["n_clipped"]))
    a = make_table(np.random.default_rng(5)); b = make_table(np.random.default_rng(5))
    assert np.array_equal(a["X_ctx"], b["X_ctx"]) and np.array_equal(a["y_query"], b["y_query"])
    rng = np.random.default_rng(1); n = 3000; kinds = {}; rmodes = {}; rows = []; nqs = []; steps = 0; ncl = 0; rc = {}
    for _ in range(n):
        mt = make_table(rng)["meta"]; kinds[mt["kind"]] = kinds.get(mt["kind"], 0) + 1; rmodes[mt["r_mode"]] = rmodes.get(mt["r_mode"], 0) + 1
        rows.append(mt["n_rows"]); nqs.append(mt["nq"]); steps += mt["has_step"]; ncl += mt["n_clipped"] > 0
        if mt["r_mode"] == "common": rc[mt["r_common"]] = rc.get(mt["r_common"], 0) + 1
    print("  mix over %d tables: kinds %s (target 0.55/0.15/0.30); r_mode %s; step share %.3f (target 0.20)" % (n, {k: round(v / n, 3) for k, v in kinds.items()}, rmodes, steps / n))
    print("  rows: mean %.0f, median %.0f, max %d; nq kept mean %.1f (min %d); tables with clipped queries %d/%d; common r counts %s" % (
        np.mean(rows), np.median(rows), max(rows), np.mean(nqs), min(nqs), ncl, n, dict(sorted(rc.items()))))
    print("repeat_prior.py OK")

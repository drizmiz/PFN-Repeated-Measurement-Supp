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
P_KIND = [0.60, 0.15, 0.25]
DIMS = [1, 2, 5]
P_D = [0.5, 0.25, 0.25]
FAMILIES = ["rbf", "mlp", "poly"]
P_FAMILY = [0.4, 0.35, 0.25]
P_STEP = 0.20
NQ = 100
Q_MARGIN = 0.10
Q_MIN = 8
N_REF = 2000

_ACTS = {"tanh": np.tanh, "relu": lambda z: np.maximum(z, 0.0), "sin": np.sin,
         "gelu": lambda z: 0.5 * z * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))}


def _rbf_1d(rng):
    k = int(rng.integers(2, 6)); c = rng.uniform(LO, HI, k)
    w = np.exp(rng.uniform(math.log(0.3), math.log(1.5), k)); a = rng.standard_normal(k); b = 0.5 * rng.standard_normal()
    return lambda x: (a[None, :] * np.exp(-0.5 * (x[:, None] - c[None, :]) ** 2 / w[None, :] ** 2)).sum(1) + b * x, "rbf%d" % k


def _poly_1d(rng):
    deg = int(rng.integers(1, 4)); coef = rng.standard_normal(deg + 1) / (1.0 + np.arange(deg + 1)) ** 0.5
    return lambda x: sum(coef[j] * (x / 2.0) ** j for j in range(deg + 1)), "poly%d" % deg


def _mlp(rng, d):
    w1, w2 = int(rng.choice([8, 16, 32, 64])), int(rng.choice([8, 16, 32, 64]))
    a1, a2 = _ACTS[str(rng.choice(list(_ACTS)))], _ACTS[str(rng.choice(list(_ACTS)))]
    g = float(rng.uniform(0.8, 2.0))                                                 
    W1 = rng.standard_normal((d, w1)) * g / math.sqrt(d); b1 = rng.standard_normal(w1) * 0.5
    W2 = rng.standard_normal((w1, w2)) * g / math.sqrt(w1); b2 = rng.standard_normal(w2) * 0.5
    W3 = rng.standard_normal((w2, 1)) / math.sqrt(w2)
    return lambda X: (a2(a1(X @ W1 + b1) @ W2 + b2) @ W3).ravel(), "mlp%d-%d" % (w1, w2)


def sample_function(rng, d):
    """-> (f: (n, d) -> (n,), meta).  Standardised on a uniform reference sample of [-2,2]^d."""
    fam = str(rng.choice(FAMILIES, p=P_FAMILY))
    if fam == "mlp":
        raw, name = _mlp(rng, d)
    else:
        parts = [(_rbf_1d if fam == "rbf" else _poly_1d)(rng) for _ in range(d)]
        name = "+".join(p[1] for p in parts) if d > 1 else parts[0][1]
        raw = lambda X: sum(parts[j][0](X[:, j]) for j in range(d)) / math.sqrt(d)
    has_step = bool(rng.random() < P_STEP)
    c0 = float(rng.uniform(-1.5, 1.5)) if has_step else 0.0
    s = float(rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])) if has_step else 0.0

    def raw2(X):
        X = np.asarray(X, float).reshape(len(X), d)
        out = raw(X)
        if has_step:
            out = out + s * (X[:, 0] >= c0)
        return out

    ref = raw2(rng.uniform(LO, HI, (N_REF, d)))
    mu, sd = float(ref.mean()), max(float(ref.std()), 1e-6)
    f = lambda X: (raw2(X) - mu) / sd
    return f, dict(family=fam, fname=name + ("+step" if has_step else ""), has_step=has_step)


def sample_gp_function(rng, d=1):
    """Validation family (NOT in the training prior): a GP-RBF draw with length-scale U[0.3, 1.5],
    realised on a 400-point grid and linearly interpolated; standardised as above."""
    ell = float(rng.uniform(0.3, 1.5)); g = np.linspace(LO, HI, 400)
    K = np.exp(-0.5 * (g[:, None] - g[None, :]) ** 2 / ell ** 2) + 1e-8 * np.eye(len(g))
    z = np.linalg.cholesky(K) @ rng.standard_normal(len(g))
    raw = lambda X: np.interp(np.asarray(X, float).reshape(-1), g, z)
    ref = raw(rng.uniform(LO, HI, N_REF)); mu, sd = float(ref.mean()), max(float(ref.std()), 1e-6)
    return (lambda X: (raw(X) - mu) / sd), dict(family="gp", fname="gp_ell%.2f" % ell, has_step=False, ell=ell)


def _build(rng, f, fmeta, d, kind, r_fixed, n_query):
    m = int(rng.integers(M_RANGE[0], M_RANGE[1] + 1))
    x = rng.uniform(LO, HI, (m, d))
    if d == 1:
        x = np.sort(x, axis=0)
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
    if kind in ("expanded", "plain"):
        X_ctx = np.repeat(x, r, axis=0); y_ctx = np.concatenate(reads); site_id = np.repeat(np.arange(m), r)
    else:
        X_ctx = x.copy(); y_ctx = np.array([yy.mean() for yy in reads]); site_id = np.arange(m)
    cand = rng.uniform(LO, HI, (2 * n_query, d)); f_cand = f(cand)
    rg = y_ctx.max() - y_ctx.min(); lo_y, hi_y = y_ctx.min() - Q_MARGIN * rg, y_ctx.max() + Q_MARGIN * rg
    inside = (f_cand >= lo_y) & (f_cand <= hi_y)
    keep = np.flatnonzero(inside)[:n_query]; n_clipped = 0
    if len(keep) < Q_MIN:
        viol = np.maximum(lo_y - f_cand, f_cand - hi_y); order = np.argsort(viol); ks = set(keep.tolist())
        extra = np.array([i for i in order if i not in ks])[: Q_MIN - len(keep)]
        n_clipped = len(extra); keep = np.concatenate([keep, extra]).astype(int)
    Xq = cand[keep]; yq = np.clip(f_cand[keep], lo_y, hi_y)
    meta = dict(kind=kind, d=d, m=m, n_rows=len(y_ctx), nq=len(keep), r=r, r_mode=r_mode, r_common=int(r[0]) if r_mode != "persite" else -1,
                sigma=sigma, x_sites=x, f_sites=fx, site_id=site_id, n_dropped=int((~inside).sum()), n_clipped=n_clipped, **fmeta)
    tab = dict(X_ctx=X_ctx.astype(np.float64), y_ctx=y_ctx.astype(np.float64), X_query=Xq.astype(np.float64), y_query=yq.astype(np.float64), meta=meta)
    finite = all(np.isfinite(tab[k]).all() for k in ("X_ctx", "y_ctx", "X_query", "y_query"))
    return tab if finite else None


def make_table(rng, kind=None, r_fixed=None, d=None, n_query=NQ):
    """kind / r_fixed / d force the table kind / a common r / the dimension. Redraws on non-finite."""
    for _ in range(20):
        k = kind or str(rng.choice(KINDS, p=P_KIND)); dd = int(d or rng.choice(DIMS, p=P_D))
        f, fmeta = sample_function(rng, dd)
        tab = _build(rng, f, fmeta, dd, k, r_fixed, n_query)
        if tab is not None:
            return tab
    raise RuntimeError("could not draw a finite table")


def make_gp_table(rng, kind="expanded", r_fixed=64, n_query=NQ):
    for _ in range(20):
        f, fmeta = sample_gp_function(rng)
        tab = _build(rng, f, fmeta, 1, kind, r_fixed, n_query)
        if tab is not None:
            return tab
    raise RuntimeError("could not draw a finite GP table")


def table_to_xy(tab):
    X = np.vstack([tab["X_ctx"], tab["X_query"]]); y = np.concatenate([tab["y_ctx"], tab["y_query"]])
    return X, y, len(tab["y_ctx"])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("adapter_prior.py unit test")
    for t in range(10):
        tab = make_table(rng); mt = tab["meta"]; X, y, m = table_to_xy(tab)
        assert X.shape == (mt["n_rows"] + mt["nq"], mt["d"]) and np.isfinite(X).all() and np.isfinite(y).all()
        rg = y[:m].max() - y[:m].min()
        assert y[m:].min() >= y[:m].min() - Q_MARGIN * rg - 1e-12 and y[m:].max() <= y[:m].max() + Q_MARGIN * rg + 1e-12
        if mt["kind"] == "expanded":
            assert mt["n_rows"] == mt["r"].sum() and len(np.unique(tab["X_ctx"], axis=0)) == mt["m"]
        if mt["kind"] == "plain":
            assert mt["n_rows"] == mt["m"] and (mt["r"] == 1).all()
        print("  t=%d kind=%-8s d=%d f=%-16s m=%2d r_mode=%-7s r=%-3d n_rows=%4d nq=%3d sigma=%.3f y_ctx sd=%.2f f_q sd=%.2f dropped=%d clipped=%d" % (
            t, mt["kind"], mt["d"], mt["fname"], mt["m"], mt["r_mode"], mt["r_common"], mt["n_rows"], mt["nq"], mt["sigma"], tab["y_ctx"].std(), tab["y_query"].std(), mt["n_dropped"], mt["n_clipped"]))
    g = make_gp_table(np.random.default_rng(3), "expanded", 64); print("  gp table: rows %d m %d ell %.2f" % (g["meta"]["n_rows"], g["meta"]["m"], g["meta"]["ell"]))
    a = make_table(np.random.default_rng(5)); b = make_table(np.random.default_rng(5))
    assert np.array_equal(a["X_ctx"], b["X_ctx"]) and np.array_equal(a["y_query"], b["y_query"])
    rng = np.random.default_rng(1); n = 3000; kinds = {}; rmodes = {}; rows = []; fams = {}; ds = {}; steps = 0; rc = {}; nqs = []
    for _ in range(n):
        mt = make_table(rng)["meta"]; kinds[mt["kind"]] = kinds.get(mt["kind"], 0) + 1; rmodes[mt["r_mode"]] = rmodes.get(mt["r_mode"], 0) + 1
        fams[mt["family"]] = fams.get(mt["family"], 0) + 1; ds[mt["d"]] = ds.get(mt["d"], 0) + 1; rows.append(mt["n_rows"]); steps += mt["has_step"]; nqs.append(mt["nq"])
        if mt["r_mode"] == "common": rc[mt["r_common"]] = rc.get(mt["r_common"], 0) + 1
    print("  mix over %d tables: kinds %s (target .60/.15/.25); r_mode %s; families %s; d %s; step share %.3f" % (n, {k: round(v / n, 3) for k, v in kinds.items()}, rmodes, fams, ds, steps / n))
    print("  rows: mean %.0f median %.0f max %d; nq mean %.1f min %d; common r counts %s" % (np.mean(rows), np.median(rows), max(rows), np.mean(nqs), min(nqs), dict(sorted(rc.items()))))
    print("adapter_prior.py OK")

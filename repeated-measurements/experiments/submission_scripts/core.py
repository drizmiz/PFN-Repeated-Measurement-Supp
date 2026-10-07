from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import math, zlib
import numpy as np

LO, HI = -2.0, 2.0
M_MAX, R_MAX = 48, 64
N_POP = 400
POP_GRID = LO + (np.arange(N_POP) + 0.5) * (HI - LO) / N_POP


_RAW1 = {
    "quadratic": lambda x: x ** 2,
    "sine": lambda x: np.sin(np.pi * x),
    "linear": lambda x: x,
    "cubic": lambda x: x ** 3,
    "abs": lambda x: np.abs(x),
    "tanh": lambda x: np.tanh(3.0 * x),
    "step": lambda x: (x >= 0).astype(float),
    "bump": lambda x: np.exp(-0.5 * (x / 0.3) ** 2),
}
FUNCS1 = list(_RAW1)
_REF = np.linspace(LO, HI, 20001)
_STD1 = {k: (float(np.mean(g(_REF))), float(np.std(g(_REF)))) for k, g in _RAW1.items()}


def f_true1(name, x):
    mu, sd = _STD1[name]
    return (_RAW1[name](np.asarray(x, float)) - mu) / sd


def make_data1(func, m, sigma, seed, rmax=R_MAX):
    """1-D design.  Readings are NESTED in r: use Y[:, :r]."""
    x_all = np.random.default_rng([seed, 11]).uniform(LO, HI, M_MAX)
    z_all = np.random.default_rng([seed, 23]).standard_normal((M_MAX, R_MAX))
    x, z = x_all[:m], z_all[:m, :rmax]
    o = np.argsort(x)
    x, z = x[o], z[o]
    fx = f_true1(func, x)
    Y = fx[:, None] + sigma * z
    gap = 0.5 * (x[1:] + x[:-1])
    Xq = np.concatenate([POP_GRID, x, gap])
    region = np.array(["pop"] * N_POP + ["site"] * m + ["gap"] * (m - 1))
    return dict(d=1, func=func, m=m, sigma=sigma, seed=seed,
                X=x[:, None], Y=Y, Xq=Xq[:, None], fq=f_true1(func, Xq), region=region)


def _raw_additive(X):
    comps = (lambda t: np.sin(1.5 * t), lambda t: 0.5 * t ** 2, lambda t: np.tanh(2 * t),
             lambda t: 0.6 * t, lambda t: np.cos(2.0 * t))
    out = np.zeros(len(X))
    for j in range(X.shape[1]):
        out += comps[j % 5](X[:, j]) / math.sqrt(1.0 + j)
    return out


def _raw_friedman(X):
    U = (X + 2.0) / 4.0
    d = X.shape[1]
    if d < 5:
        return 10 * np.sin(np.pi * U[:, 0] * U[:, 1]) + 20 * (U[:, 0] - 0.5) ** 2 + 5 * U[:, 1]
    return (10 * np.sin(np.pi * U[:, 0] * U[:, 1]) + 20 * (U[:, 2] - 0.5) ** 2 + 10 * U[:, 3]
            + 5 * U[:, 4])


def _raw_step(X):
    d = X.shape[1]
    out = 2.0 * ((X[:, 0] > 0) & (X[:, 1] > 0)) - 1.5 * (X[:, 0] < -0.5) + 1.0 * (X[:, 1] > 0.8)
    if d >= 5:
        out = out + 1.5 * ((X[:, 2] > 0) & (X[:, 3] < 0)) - 1.0 * (X[:, 4] > 1.0)
    return out


_RAWD = {"additive": _raw_additive, "friedman": _raw_friedman, "step": _raw_step}
FUNCSD = list(_RAWD)
_MS = {}


def _moments(func, d):
    if (func, d) not in _MS:
        Xp = np.random.default_rng(987654 + d).uniform(-2, 2, (200000, d))
        v = _RAWD[func](Xp)
        _MS[(func, d)] = (float(v.mean()), float(v.std()))
    return _MS[(func, d)]


def make_datad(func, d, m, sigma, seed, nq=1000, rmax=R_MAX):
    key = zlib.crc32(f"{func}|{d}|{m}|{sigma}".encode())
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(key)]))
    X = rng.uniform(-2, 2, (m, d))
    Xpop = rng.uniform(-2, 2, (nq, d))
    mu, sd = _moments(func, d)
    fall = (_RAWD[func](np.vstack([X, Xpop])) - mu) / sd
    fx, fpop = fall[:m], fall[m:]
    z = rng.standard_normal((m, rmax))
    Y = fx[:, None] + sigma * z
    Xq = np.vstack([Xpop, X])
    return dict(d=d, func=func, m=m, sigma=sigma, seed=seed, X=X, Y=Y, Xq=Xq,
                fq=np.concatenate([fpop, fx]),
                region=np.array(["pop"] * nq + ["site"] * m))


def ctx(ds, r, kind):
    """kind: expanded = all r*m rows; avg = m location averages."""
    X, Y = ds["X"], ds["Y"][:, :r]
    if kind == "expanded":
        return np.repeat(X, r, axis=0), Y.reshape(-1).copy(), np.repeat(np.arange(len(X)), r)
    if kind == "avg":
        return X.copy(), Y.mean(1), np.arange(len(X))
    raise ValueError(kind)


def mse_by_region(ds, pred):
    out = {}
    for reg in np.unique(ds["region"]):
        k = ds["region"] == reg
        out[reg] = float(np.mean((pred[k] - ds["fq"][k]) ** 2))
    return out

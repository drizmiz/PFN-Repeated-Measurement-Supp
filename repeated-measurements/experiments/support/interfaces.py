
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import numpy as np

NAMES = ["single", "expanded", "mean",
         "kmeans_2", "kmeans_4", "kmeans_8",
         "mean_sd", "mean_se", "mean_logr", "mean_sd_logr"]


def _sd(Y):
    """Within-location SD with ddof=1; zero when r=1 (no spread is observable)."""
    return Y.std(axis=1, ddof=1) if Y.shape[1] > 1 else np.zeros(Y.shape[0])


def _with_feature(XL, XQ, feat):
    """Append one extra column; the query rows get the context mean of that column."""
    Xc = np.column_stack([XL, feat])
    XQc = np.column_stack([XQ, np.full(len(XQ), float(np.mean(feat)))])
    return Xc, XQc


def build(name, XL, Y, XQ):
    XL = np.asarray(XL, float)
    Y = np.asarray(Y, float)
    XQ = np.asarray(XQ, float)
    m, r = Y.shape
    ybar = Y.mean(axis=1)

    if name == "single":
        return XL.copy(), Y[:, 0].copy(), XQ.copy()

    if name == "expanded":
        return np.repeat(XL, r, axis=0), Y.reshape(-1).copy(), XQ.copy()

    if name == "mean":
        return XL.copy(), ybar, XQ.copy()

    if name.startswith("kmeans_"):
        K = int(name.split("_")[1])
        k = min(K, r)

        groups = np.array_split(np.arange(r), k)
        rows, ys = [], []
        for g in groups:
            rows.append(XL)
            ys.append(Y[:, g].mean(axis=1))
        return np.vstack(rows), np.concatenate(ys), XQ.copy()

    if name == "mean_sd":
        Xc, XQc = _with_feature(XL, XQ, _sd(Y))
        return Xc, ybar, XQc

    if name == "mean_se":
        Xc, XQc = _with_feature(XL, XQ, _sd(Y) / np.sqrt(r))
        return Xc, ybar, XQc

    if name == "mean_logr":
        Xc, XQc = _with_feature(XL, XQ, np.full(m, np.log(r)))
        return Xc, ybar, XQc

    if name == "mean_sd_logr":
        Xc = np.column_stack([XL, _sd(Y), np.full(m, np.log(r))])
        XQc = np.column_stack([XQ, np.full(len(XQ), float(_sd(Y).mean())),
                               np.full(len(XQ), np.log(r))])
        return Xc, ybar, XQc

    raise ValueError("unknown interface: %s" % name)


if __name__ == "__main__":

    rng = np.random.default_rng(0)
    XL = rng.uniform(-2, 2, (12, 1)); XQ = np.linspace(-2, 2, 7)[:, None]
    for r in (1, 2, 5, 64):
        Y = rng.normal(0, 1, (12, r))
        for nm in NAMES:
            Xc, yc, XQc = build(nm, XL, Y, XQ)
            assert len(Xc) == len(yc) and Xc.shape[1] == XQc.shape[1], (nm, r)
        print("r=%-3d ok  rows per interface: %s" % (
            r, {nm: len(build(nm, XL, Y, XQ)[1]) for nm in NAMES}))

from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))


import argparse

import gc

import json

import os

from pathlib import Path

import subprocess

import sys

import time

import numpy as np

import pandas as pd

from scipy.optimize import minimize_scalar

from scipy.special import expit

from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent

REPO = HERE

RAW = HERE/'runs'

RESULTS = HERE / 'results'

SEEDS = range(6201, 6207)

FUNCTIONS = ('linear',)

T = np.linspace(0, 1, 81)

def truth(x, function):
    return x if function == 'linear' else x*x if function == 'quadratic' else np.sin(2*x)

def derivative(x, function):
    return np.ones_like(x) if function == 'linear' else 2*x if function == 'quadratic' else 2*np.cos(2*x)

def design(seed, function):
    rng = np.random.default_rng(seed)
    x = -2 + 4*(np.arange(12) + rng.uniform(.15, .85, 12))/12
    gaps = x[:-1, None] + np.diff(x)[:, None]*T
    gaps[:, -1] = x[1:]
    q, inverse = np.unique(gaps, return_inverse=True)
    y = truth(x, function)
    d = derivative(gaps, function)
    eligible = (np.abs(np.diff(y)/np.diff(x)) >= .25) & (np.min(np.abs(d), axis=1) >= .25) & (np.min(d, axis=1)*np.max(d, axis=1) > 0)
    return dict(x=x, y=y, q=q, inverse=inverse.reshape(gaps.shape), eligible=eligible, f=truth(q, function))

def path(function, seed, arm, r):
    return RAW / function / f'seed{seed}' / f'{arm}_r{r}.npz'

def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def shape(a):
    if a < 1e-5:
        return T.copy()
    return (expit(a*(T-.5))-expit(-a/2))/(expit(a/2)-expit(-a/2))

def fit_shape(values):
    result = minimize_scalar(lambda a: float(np.mean((values-shape(a))**2)), bounds=(0, 80), method='bounded')
    options = [(0., float(np.mean((values-T)**2))), (float(result.x), float(result.fun)), (80., float(np.mean((values-shape(80))**2)))]
    return min(options, key=lambda item: item[1])

def normalized(data, readout):
    means = data[readout][data['inverse']]
    delta = means[:, -1]-means[:, 0]
    available = data['eligible'] & (np.abs(delta) >= .1*np.abs(np.diff(data['y'])))
    curves = np.full_like(means, np.nan)
    curves[available] = (means[available]-means[available, :1])/delta[available, None]
    return curves, available

def metrics(data, means):
    curve = means[data['inverse']]
    target = data['f'][data['inverse']]
    gaps = np.diff(data['x'])
    sec = np.abs(np.diff(data['y'])/gaps)
    slope = np.abs(np.gradient(curve, T, axis=1)/gaps[:, None])
    near = (T <= .2) | (T >= .8)
    mid = (T >= .4) & (T <= .6)
    eligible = data['eligible']
    scaled = slope[eligible]/sec[eligible, None]
    directed = (np.gradient(curve, T, axis=1)/gaps[:,None])[eligible]*np.sign(np.diff(data['y'])[eligible,None])/sec[eligible,None]
    endpoint_ratio = (curve[:,-1]-curve[:,0])[eligible]/np.diff(data['y'])[eligible]
    near_slope = float(scaled[:, near].mean())
    middle_slope = float(scaled[:, mid].mean())
    widths = {}
    for tol in (.05, .1, .2):
        per_gap = []
        for i in np.flatnonzero(eligible):
            extent = []
            for side in (curve[i, :41], curve[i, ::-1][:41]):
                ok = np.abs(side-side[0]) <= tol*abs(data['y'][i+1]-data['y'][i])
                first = np.flatnonzero(~ok)
                extent.append((first[0]-1 if len(first) else 40)/40)
            per_gap.append(np.mean(extent))
        widths[f'anchored_width_{tol:g}'] = float(np.mean(per_gap))
    loss = (curve-target)**2
    sites = np.r_[curve[:, 0], curve[-1, -1]]
    return dict(n_eligible=int(eligible.sum()), near_slope=near_slope, middle_slope=middle_slope,
                near_signed_slope=float(directed[:,near].mean()), middle_signed_slope=float(directed[:,mid].mean()),
                endpoint_sign_accuracy=float((endpoint_ratio>0).mean()), endpoint_ratio_min=float(endpoint_ratio.min()),
                concentration=middle_slope/near_slope,
                near_flat_fraction=float((scaled[:, near] <= .25).mean()), **widths,
                site_mse=float(np.mean((sites-data['y'])**2)), near_mse=float(loss[:, near].mean()),
                middle_mse=float(loss[:, mid].mean()), hull_mse=float(np.trapezoid(loss, T, axis=1)@gaps/gaps.sum()))


from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import numpy as np

DEVELOPMENT_SEEDS = tuple(range(2026092901, 2026092907))
CONFIRMATION_SEEDS = tuple(range(2026092921, 2026092933))
COUNTS = (1, 4, 16, 64)


def evaluate_function(X, case):
    X = np.asarray(X, dtype=float)
    q = int(case.get('q', case.get('d_effective', 1)))
    family = case['function']
    z = X[:, :q]
    if family == 'linear':
        return z[:, 0] / np.sqrt(4 / 3)
    if family == 'quadratic':
        return (z[:, 0] ** 2 - 4 / 3) / np.sqrt(64 / 45)
    if family == 'sine':
        return np.sqrt(2) * np.sin(np.pi * z[:, 0])
    if family == 'sine_additive':
        return np.sin(np.pi * z).sum(axis=1) / np.sqrt(q / 2)
    if family == 'sine_interaction':
        a = np.sin(np.pi * z)
        if q == 1:
            return np.sqrt(2) * a[:, 0]
        return (a[:, :-1] * a[:, 1:]).sum(axis=1) / np.sqrt((q - 1) / 4)
    raise ValueError(family)


def _rng(seed, stream, index=0):
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream, int(index)]))


def _features(seed, stream, n, d, law):
    columns = []
    for j in range(d):
        u = _rng(seed, stream, j).random(n)
        if law == 'uniform':
            x = -2 + 4 * u
        elif law == 'two_clusters':
            x = np.where(u < .5, -2 + 3 * u, .5 + 3 * (u - .5))
        else:
            raise ValueError(law)
        columns.append(x)
    return np.column_stack(columns)


def canonical_case(case):
    c = dict(case)
    c.setdefault('q', c.get('d_effective', 1))
    c.setdefault('d', c['q'])
    c.setdefault('noise_sd', c.get('sigma', .5))
    c.setdefault('noise_bank', 0)
    c.setdefault('input_law', 'uniform')
    c.setdefault('panel', 'custom')
    c['case_id'] = (f"{c['function']}__{c['input_law']}__m{c['m']}__d{c['d']}"
                    f"__q{c['q']}__s{c['noise_sd']:g}__seed{c['seed']}__bank{c['noise_bank']}")
    return c


def make_case(case):
    c = canonical_case(case)
    m, d, seed, bank = (int(c[k]) for k in ('m', 'd', 'seed', 'noise_bank'))
    X = _features(seed, 10, m, d, c['input_law'])
    Q = _features(seed, 11, 512, d, 'uniform')
    f_X, f_Q = evaluate_function(X, c), evaluate_function(Q, c)
    sd = float(c['noise_sd'])
    Y = f_X[:, None] + sd * _rng(seed, 20, bank).normal(size=(m, 64))
    Y_B = f_X[:, None] + sd * _rng(seed, 21, bank).normal(size=(m, 64))
    Y_Q_B = f_Q[:, None] + sd * _rng(seed, 22, bank).normal(size=(512, 64))
    return dict(X=X, Y=Y, Y_B=Y_B, Y_Q_B=Y_Q_B, Q=Q,
                f_X=f_X, f_Q=f_Q, truth_fn=lambda Z: evaluate_function(Z, c),
                active_dims=list(range(int(c['q']))), case=c)


def iter_cases(panel='core', stage='development', seeds=None):
    seeds = tuple(seeds or (DEVELOPMENT_SEEDS if stage == 'development' else CONFIRMATION_SEEDS))
    specs = []
    if panel == 'core':
        for family, m, q in [('quadratic', 12, 1), ('quadratic', 128, 1),
                              ('sine_additive', 48, 5), ('sine_interaction', 48, 5)]:
            for sd in (.05, .5):
                specs.append(dict(function=family, m=m, q=q, d=q, noise_sd=sd))
    elif panel == 'noise':
        for family, m, q in [('quadratic', 12, 1), ('sine_additive', 48, 5),
                              ('sine_interaction', 48, 5)]:
            for sd in (.05, .2, .5, 1.):
                specs.append(dict(function=family, m=m, q=q, d=q, noise_sd=sd))
    elif panel == 'coverage':
        for m in (12, 48, 128):
            for sd in (.05, .5):
                specs.append(dict(function='quadratic', m=m, q=1, d=1, noise_sd=sd))
        for sd in (.05, .5):
            specs.append(dict(function='sine', m=48, q=1, d=1, noise_sd=sd, input_law='two_clusters'))
    elif panel == 'dimension':
        for family in ('sine_additive', 'sine_interaction'):
            for q in (1, 2, 5, 10, 20):
                if q == 1 and family == 'sine_interaction':
                    continue
                for sd in (.05, .5):
                    specs.append(dict(function=family, m=48, q=q, d=q, noise_sd=sd))
    elif panel == 'nuisance':
        for family, q, m in [('quadratic', 1, 12), ('sine_additive', 5, 48)]:
            for extra in (0, 20, 100):
                for sd in (.05, .5):
                    specs.append(dict(function=family, m=m, q=q, d=q + extra, noise_sd=sd))
    elif panel == 'endpoints':
        for family, q, d, m in [('sine_additive',20,20,48),
                                 ('sine_interaction',20,20,48),
                                 ('quadratic',1,101,12),
                                 ('sine_additive',5,105,48)]:
            for sd in (.05,.5):
                specs.append(dict(function=family,m=m,q=q,d=d,noise_sd=sd))
    else:
        raise ValueError(panel)
    return [canonical_case(dict(spec, seed=seed, panel=panel, stage=stage))
            for spec in specs for seed in seeds]


def represent(X, Y, r, method):
    """Only observed arrays enter this representation builder."""
    if method == 'single':
        return X, Y[:, 0]
    if method == 'mean':
        return X, Y[:, :r].mean(axis=1)
    if method == 'expanded':
        return np.repeat(X, r, axis=0), Y[:, :r].reshape(-1)
    if method == 'copy':
        return np.repeat(X, r, axis=0), np.repeat(Y[:, 0], r)
    raise ValueError(method)

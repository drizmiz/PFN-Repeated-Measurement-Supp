from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))


from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class Probe:
    query: np.ndarray
    path_indices: np.ndarray                                 
    edges: np.ndarray                       
    t: np.ndarray
    increment_base: np.ndarray
    increment_tip: np.ndarray
    region: np.ndarray
    direction: np.ndarray
    h_fraction: np.ndarray
    site_edge: np.ndarray
    site_t: np.ndarray
    scale: np.ndarray


def _directions(q: int, active_dims: tuple[int, ...], edge_unit: np.ndarray,
                rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """Eight unit directions per site in context-standardized coordinates."""
    active = np.asarray(active_dims, dtype=int)
    inactive = np.asarray([j for j in range(q) if j not in active_dims], dtype=int)
    counts = (3, 2, 3) if len(inactive) else (4, 4, 0)
    vectors: list[np.ndarray] = []
    names: list[str] = []
    for name, dimensions, count in [('active', active, counts[0]),
                                    ('neighbor', None, counts[1]),
                                    ('irrelevant', inactive, counts[2])]:
        for k in range(count):
            if name == 'neighbor':
                unit = edge_unit * (1 if k % 2 == 0 else -1)
            else:
                unit = np.zeros(q)
                if len(dimensions) == 1:
                    unit[dimensions[0]] = 1 if k % 2 == 0 else -1
                else:
                    unit[dimensions] = rng.normal(size=len(dimensions))
                    unit /= np.linalg.norm(unit)
                    if k % 2:
                        unit *= -1
            vectors.append(unit)
            names.append(name)
    return np.stack(vectors), names


def build_probe(X: np.ndarray, active_dims: tuple[int, ...] | list[int], seed: int,
                max_edges: int = 8, n_t: int = 41, max_sites: int = 16,
                h_fractions: tuple[float, ...] = (.02, .05, .1)) -> Probe:
    """Choose MST edges and perturbations using only the context geometry.

    Distances and directions use column-standardized X; returned queries are in
    the original feature coordinates. The same selected graph and sites are
    reused for every model and repetition count of a case.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 2:
        raise ValueError('X must contain at least two context rows')
    q = X.shape[1]
    active_dims = tuple(int(j) for j in active_dims)
    if not active_dims or min(active_dims) < 0 or max(active_dims) >= q:
        raise ValueError('active_dims must identify at least one valid column')
    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    center = X.mean(axis=0)
    Z = (X - center) / scale
    distances = cdist(Z, Z)
    graph = minimum_spanning_tree(distances).toarray()
    edges = np.argwhere(graph > 0)
    edges = np.sort(edges, axis=1)
    lengths = np.linalg.norm(Z[edges[:, 1]] - Z[edges[:, 0]], axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0], lengths))
    take = np.unique(np.rint(np.linspace(0, len(order)-1, min(max_edges, len(order)))).astype(int))
    edges = edges[order[take]]
    t = np.linspace(0., 1., n_t)
    paths = (1-t[None, :, None])*X[edges[:, 0], None, :] + t[None, :, None]*X[edges[:, 1], None, :]
    points = [row for row in paths.reshape(-1, q)]
    path_indices = np.arange(len(points)).reshape(len(edges), n_t)


    sites: list[tuple[int, float, str]] = []
    for edge_id in range(len(edges)):
        sites.append((edge_id, .5, 'middle'))
    for edge_id in range(len(edges)):
        sites.append((edge_id, .1 if edge_id % 2 == 0 else .9, 'near'))
    sites = sites[:max_sites]
    base: list[int] = []
    tip: list[int] = []
    region: list[str] = []
    direction: list[str] = []
    fractions: list[float] = []
    site_edges: list[int] = []
    site_ts: list[float] = []
    for edge_id, site_t, place in sites:
        a, b = edges[edge_id]
        z_site = (1-site_t)*Z[a] + site_t*Z[b]
        edge_vec = Z[b]-Z[a]
        edge_unit = edge_vec/np.linalg.norm(edge_vec)

                                                                             

        local_distance = np.linalg.norm(edge_vec)

        vectors, names = _directions(q, active_dims, edge_unit,
                                     np.random.default_rng(np.random.SeedSequence([seed, edge_id])))
        base_id = len(points)
        points.append(z_site*scale + center)
        for vec, name in zip(vectors, names):
            for fraction in h_fractions:
                base.append(base_id)
                tip.append(len(points))
                points.append((z_site + fraction*local_distance*vec)*scale + center)
                region.append(place)
                direction.append(name)
                fractions.append(fraction)
                site_edges.append(edge_id)
                site_ts.append(site_t)
    return Probe(np.asarray(points), path_indices, edges, t,
                 np.asarray(base, int), np.asarray(tip, int),
                 np.asarray(region), np.asarray(direction),
                 np.asarray(fractions), np.asarray(site_edges, int),
                 np.asarray(site_ts), scale)


def _tv80_width(curve: np.ndarray, t: np.ndarray, floor: float) -> float:
    weights = np.abs(np.diff(curve))
    total = weights.sum()
    if total <= floor:
        return float('nan')
    target = .8*total
    best = float('inf')
    for start in range(len(weights)):
        running = 0.
        for end in range(start, len(weights)):
            running += weights[end]
            if running >= target:
                best = min(best, t[end+1]-t[start])
                break
    return float(best)


def summarize_probe(probe: Probe, pred: np.ndarray, truth: np.ndarray,
                    label_scale: float) -> tuple[list[dict], list[dict]]:
    """Return aggregate increment rows and per-edge path rows.

    S is a ratio of mean squared increments, not a mean of pointwise ratios.
    Ratios are missing if teacher energy is below 1e-8 label_scale squared.
    All absolute energies and the pointwise effective fraction remain reported.
    """
    pred = np.asarray(pred, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    if len(pred) != len(probe.query) or len(truth) != len(pred):
        raise ValueError('Predictions, truth, and probe query lengths differ')
    label_scale = max(float(label_scale), 1e-12)
    dp = pred[probe.increment_tip]-pred[probe.increment_base]
    df = truth[probe.increment_tip]-truth[probe.increment_base]
    floor = 1e-8*label_scale**2
    rows: list[dict] = []
    for place in np.unique(probe.region):
        for name in np.unique(probe.direction):
            for fraction in np.unique(probe.h_fraction):
                mask = (probe.region == place) & (probe.direction == name) & (probe.h_fraction == fraction)
                if not mask.any():
                    continue
                pe = float(np.mean(dp[mask]**2))
                fe = float(np.mean(df[mask]**2))
                rows.append(dict(region=str(place), direction=str(name), h_fraction=float(fraction),
                                 n_increments=int(mask.sum()), pred_energy=pe, truth_energy=fe,
                                 response_ratio=pe/fe if fe > floor else float('nan'),
                                 effective_fraction=float(np.mean(df[mask]**2 > floor)),
                                 energy_floor=floor))
    for name in np.unique(probe.direction):
        for fraction in np.unique(probe.h_fraction):
            near = next((row for row in rows if row['region'] == 'near' and row['direction'] == name and row['h_fraction'] == fraction), None)
            mid = next((row for row in rows if row['region'] == 'middle' and row['direction'] == name and row['h_fraction'] == fraction), None)
            if near and mid:
                near['near_middle_response_ratio'] = near['response_ratio']/mid['response_ratio'] if np.isfinite(near['response_ratio']) and np.isfinite(mid['response_ratio']) and mid['response_ratio'] > 0 else float('nan')
                near['near_middle_pred_energy_ratio'] = near['pred_energy']/mid['pred_energy'] if mid['pred_energy'] > 0 else float('nan')
                near['near_middle_truth_energy_ratio'] = near['truth_energy']/mid['truth_energy'] if mid['truth_energy'] > 0 else float('nan')
    paths_pred = pred[probe.path_indices]
    paths_truth = truth[probe.path_indices]
    path_rows = []
    for edge_id, (p, f) in enumerate(zip(paths_pred, paths_truth)):
        path_rows.append(dict(edge_id=edge_id, context_a=int(probe.edges[edge_id, 0]),
                              context_b=int(probe.edges[edge_id, 1]),
                              pred_tv80_width=_tv80_width(p, probe.t, 1e-8*label_scale),
                              truth_tv80_width=_tv80_width(f, probe.t, 1e-8*label_scale),
                              pred_total_variation=float(np.abs(np.diff(p)).sum()),
                              truth_total_variation=float(np.abs(np.diff(f)).sum()),
                              path_mse=float(np.mean((p-f)**2))))
    return rows, path_rows

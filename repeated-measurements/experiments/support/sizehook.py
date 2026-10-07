from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))


import dataclasses
import math
from typing import Callable, Iterable, Sequence, Union

import numpy as np
import torch

GROUPS = ("featdist", "rowembed", "icl", "decoder")
NSpec = Union[int, float, Callable[[int, "ScalingSite"], float]]


                      

@dataclasses.dataclass(frozen=True)
class ScalingSite:
    """One SoftmaxScalingMLP instance and where it sits."""

    name: str                                              
    group: str                                  
    layer: int                          
    module: torch.nn.Module = dataclasses.field(compare=False, repr=False)


def _get_model(reg_or_model) -> torch.nn.Module:
    """Accept a fitted TabPFNRegressor/Classifier or a bare TabPFNV3 module."""
    if isinstance(reg_or_model, torch.nn.Module):
        return reg_or_model
    if hasattr(reg_or_model, "models_"):
        models = reg_or_model.models_
        if len(models) != 1:
            raise ValueError("sizehook supports a single underlying model")
        return models[0]
    return reg_or_model.model_


def find_scaling_sites(reg_or_model) -> list[ScalingSite]:
    """Enumerate every SoftmaxScalingMLP with its group and layer index."""
    model = _get_model(reg_or_model)
    sites = []
    for name, mod in model.named_modules():
        if type(mod).__name__ != "SoftmaxScalingMLP":
            continue
        parts = name.split(".")
        if parts[0] == "feature_distribution_embedder":
            group, layer = "featdist", int(parts[2])
        elif parts[0] == "icl_blocks":
            group, layer = "icl", int(parts[1])
        elif parts[0] == "many_class_decoder":
            group, layer = "decoder", 0
        elif parts[0] == "column_aggregator":
            group, layer = "rowembed", int(parts[2])
        else:
            raise RuntimeError(f"unrecognised SoftmaxScalingMLP location: {name}")
        sites.append(ScalingSite(name, group, layer, mod))
    return sites


def _select(sites, groups, layers) -> list[ScalingSite]:
    if groups is None:
        groups = GROUPS
    if isinstance(groups, str):
        groups = (groups,)
    for g in groups:
        if g not in GROUPS:
            raise ValueError(f"unknown group {g!r}; choose from {GROUPS}")
    out = []
    for s in sites:
        if s.group not in groups:
            continue
        if layers is not None:
            lay = layers.get(s.group) if isinstance(layers, dict) else layers
            if lay is not None and s.layer not in set(lay):
                continue
        out.append(s)
    return out


              

class SizeOverride:
    """Context manager: replace the `n` that selected SoftmaxScalingMLP modules receive.

    Args:
        reg_or_model: fitted TabPFNRegressor (or the TabPFNV3 module).
        n: replacement value. A number (int or float -- log is taken, so 37.5 is legal), or a
            callable `f(true_n, site) -> number`, e.g. `lambda n, s: n / 64`.
        groups: subset of ("featdist", "rowembed", "icl", "decoder"); None = all.
        layers: None = all layers; an iterable of layer indices applied to every selected group; or a
            dict {group: iterable-or-None}.
        rows: "all" | "train" | "test".  Only meaningful for the "icl" group, where the train-row
            queries and the test-row queries go through the scaling module in two separate calls
            (train first). "test" changes only how TEST rows read the context (the read-out);
            "train" changes only how the context rows attend to each other (representation
            building). Requires the default non-cached fit_mode ("fit_preprocessors").

    Attributes after use:
        n_calls: number of scaling-module calls seen on selected modules.
        n_modified: number of calls whose n was actually replaced by a DIFFERENT value.
        log: list of (site.name, row_kind, true_n, used_n).
    """

    def __init__(self, reg_or_model, n: NSpec, groups=None, layers=None, rows: str = "all"):
        if rows not in ("all", "train", "test"):
            raise ValueError("rows must be 'all', 'train' or 'test'")
        self.model = _get_model(reg_or_model)
        self.n = n
        self.rows = rows
        self.sites = _select(find_scaling_sites(self.model), groups, layers)
        if rows != "all" and any(s.group != "icl" for s in self.sites):
            raise ValueError("rows='train'/'test' is only defined for groups=('icl',)")
        if rows != "all" and getattr(self.model.config, "icl_num_kv_heads_test", None) is None:
            raise ValueError("train/test calls are not separate in this checkpoint")
        self._handles = []
        self._parity: dict[str, int] = {}
        self.n_calls = 0
        self.n_modified = 0
        self.log: list[tuple[str, str, int, float]] = []

    def _resolve(self, true_n: int, site: ScalingSite) -> float:
        val = self.n(true_n, site) if callable(self.n) else self.n
        if not (isinstance(val, (int, float)) and math.isfinite(val) and val >= 1):
            raise ValueError(f"replacement n must be a finite number >= 1, got {val!r}")
        return val

    def _make_hook(self, site: ScalingSite):
        def hook(mod, args, kwargs):
            q = args[0] if args else kwargs["q_BSHD"]
            true_n = args[1] if len(args) > 1 else kwargs["n"]
            kind = "all"
            if site.group == "icl" and self.rows != "all":
                idx = self._parity.get(site.name, 0)
                self._parity[site.name] = idx + 1
                kind = "train" if idx % 2 == 0 else "test"
                if kind == "train" and q.shape[1] != int(true_n):
                    raise RuntimeError(
                        "train/test call order assumption violated "
                        f"(q rows {q.shape[1]} vs n {true_n}); use rows='all'"
                    )
            self.n_calls += 1
            if self.rows != "all" and kind != self.rows:
                self.log.append((site.name, kind, int(true_n), float(true_n)))
                return None
            new_n = self._resolve(int(true_n), site)
            if new_n != true_n:
                self.n_modified += 1
            self.log.append((site.name, kind, int(true_n), float(new_n)))

            return (q, new_n), {}

        return hook

    def __enter__(self):
        def reset(mod, args, kwargs):
            self._parity.clear()

        self._handles.append(self.model.register_forward_pre_hook(reset, with_kwargs=True))
        for s in self.sites:
            self._handles.append(
                s.module.register_forward_pre_hook(self._make_hook(s), with_kwargs=True)
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


class SizeRecorder:
    """Context manager: record, per scaling module, the n it receives and the query shape.

    `records` is a list of dicts: name, group, layer, call_idx, n, q_shape.
    `summary()` gives {name: sorted set of n}.
    """

    def __init__(self, reg_or_model):
        self.model = _get_model(reg_or_model)
        self.sites = find_scaling_sites(self.model)
        self.records: list[dict] = []
        self._handles = []
        self._count: dict[str, int] = {}

    def __enter__(self):
        for s in self.sites:
            def hook(mod, args, kwargs, s=s):
                q = args[0] if args else kwargs["q_BSHD"]
                n = args[1] if len(args) > 1 else kwargs["n"]
                i = self._count.get(s.name, 0)
                self._count[s.name] = i + 1
                self.records.append(dict(name=s.name, group=s.group, layer=s.layer, call_idx=i,
                                         n=float(n), q_shape=tuple(q.shape)))
            self._handles.append(s.module.register_forward_pre_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def summary(self) -> dict[str, list[float]]:
        out: dict[str, set] = {}
        for r in self.records:
            out.setdefault(r["name"], set()).add(r["n"])
        return {k: sorted(v) for k, v in out.items()}


@torch.no_grad()
def base_scale_vector(site: ScalingSite, n: float) -> np.ndarray:
    """base_mlp(log n) of one scaling module, shape (num_heads, head_dim), float32."""
    p = next(site.module.base_mlp.parameters())
    logn = torch.tensor([[math.log(max(float(n), 1.0))]], device=p.device, dtype=p.dtype)
    out = site.module.base_mlp(logn).view(site.module.num_heads, site.module.head_dim)
    return out.float().cpu().numpy()


                                                      

class AttentionProbe:
    """Context manager: capture the ICL-stage attention of TEST rows over TRAIN rows.

    The fused SDPA kernels never materialise attention weights, so for the selected ICL layers we
    recompute softmax(q_scaled k^T / sqrt(d)) in float32 from the exact (already size-scaled)
    queries and keys that are handed to the kernel. Composes with SizeOverride (enter the override
    first or second, order does not matter) because the override acts upstream, on q.

    Args:
        layers: iterable of ICL layer indices (None = all 24).
        reducer: optional `f(layer, probs)` with probs a float32 CUDA tensor of shape
            (n_test, n_heads, n_train); its return value is stored in `out[layer]`.
            Default stores the full array on CPU as float16-safe float32 numpy.
    Notes:
        test rows use KV head 0 only (icl_num_kv_heads_test=1) but all 8 QUERY heads.
        Requires the non-cached path (default fit_mode) and batch size 1 (n_estimators=1) per call;
        with n_estimators>1 `out[layer]` holds a list, one entry per ensemble member forward.
    """

    def __init__(self, reg_or_model, layers: Iterable[int] | None = None,
                 reducer: Callable[[int, torch.Tensor], object] | None = None):
        self.model = _get_model(reg_or_model)
        self.layers = None if layers is None else set(layers)
        self.reducer = reducer
        self.out: dict[int, list] = {}
        self._cur = None                                                             
        self._handles = []

    def __enter__(self):
        import tabpfn.architectures.tabpfn_v3 as v3

        self._v3 = v3
        self._orig = v3.scaled_dot_product_attention
        probe = self

        def patched(q, k, v, *args, **kwargs):
            cur = probe._cur
            if cur is not None:
                layer, idx = cur
                probe._cur = (layer, idx + 1)
                if idx == 1 and k is not None:                                                  

                                                                                                 
                    with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
                        qf = q[0].float()             
                        kf = k[0].float()               
                        if kf.shape[1] != qf.shape[1]:
                            kf = kf.repeat_interleave(qf.shape[1] // kf.shape[1], dim=1)
                        logits = torch.einsum("mhd,nhd->mhn", qf, kf) / math.sqrt(qf.shape[-1])
                        probs = torch.softmax(logits, dim=-1)
                        res = probe.reducer(layer, probs) if probe.reducer else probs.cpu().numpy()
                        probe.out.setdefault(layer, []).append(res)
            return probe._orig(q, k, v, *args, **kwargs)

        v3.scaled_dot_product_attention = patched
        for i, blk in enumerate(self.model.icl_blocks):
            if self.layers is not None and i not in self.layers:
                continue
            att = blk.icl_attention

            def pre(mod, args, kwargs, i=i):
                probe._cur = (i, 0)

            def post(mod, args, kwargs, output):
                probe._cur = None

            self._handles.append(att.register_forward_pre_hook(pre, with_kwargs=True))
            self._handles.append(att.register_forward_hook(post, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        self._v3.scaled_dot_product_attention = self._orig
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._cur = None
        return False


                                                               

def make_regressor(ckpt: str, *, fingerprint: bool = True, n_estimators: int = 1,
                   never_categorical: bool = True, device: str = "cuda", random_state: int = 0,
                   softmax_temperature: float = 0.9, extra_inference_config: dict | None = None,
                   **kwargs):
    """Build a TabPFNRegressor with the duplicate-sensitive preprocessing under explicit control.

    fingerprint=False removes the hash column. With the default (True) the GPU preprocessing
        pipeline appends one extra pseudo-random U(0,1) feature; rows with IDENTICAL x get DIFFERENT
        values (collision counter re-hash), and a TEST row whose x equals a measured location
        receives the value of the FIRST train replicate at that location.
    never_categorical=True sets MIN_UNIQUE_FOR_NUMERICAL_FEATURES=1. With the shipped default (4), a
        numeric column with < 4 distinct values is silently re-typed as CATEGORICAL (shuffled ordinal
        codes) as soon as the context has > 100 rows -- i.e. a design with m <= 3 locations changes
        its x-encoding between r=1 and r=64. Irrelevant for m >= 4 but removed here for safety.
    Other duplicate/n-sensitive steps, for reference (not changed here):
        * ensemble member 0 (the only one at n_estimators=1): 'squashing_scaler_default' robust
          scaler (median / quantile range -> invariant to uniform replication), no y-transform.
        * members with 'quantile_uni' (n_estimators>=5): n_quantiles depends on the number of rows.
        * every other member uses a 'safepower' y-transform.
        * softmax_temperature=0.9 sharpens the predictive distribution (set 1.0 to get raw logits).
    """
    from tabpfn import TabPFNRegressor

    ic = {"FINGERPRINT_FEATURE": bool(fingerprint)}
    if never_categorical:
        ic["MIN_UNIQUE_FOR_NUMERICAL_FEATURES"] = 1
    if extra_inference_config:
        ic.update(extra_inference_config)
    return TabPFNRegressor(model_path=ckpt, n_estimators=n_estimators, device=device,
                           random_state=random_state, softmax_temperature=softmax_temperature,
                           ignore_pretraining_limits=True, inference_config=ic, **kwargs)


@dataclasses.dataclass
class Predictive:
    """Predictive distribution of one future reading at each query row (raw y units).

    The law is a piecewise-uniform histogram over 5000 buckets (`borders`, 5001 edges, raw y units)
    with half-normal outer tails; `probs[i, b]` is the mass of bucket b at query i.
    """

    mean: np.ndarray
    median: np.ndarray
    sd: np.ndarray
    quantile_levels: np.ndarray
    quantiles: np.ndarray                       
    probs: np.ndarray                                
    borders: np.ndarray                    

    def _cum(self) -> np.ndarray:
        cum = np.concatenate([np.zeros((len(self.probs), 1)), np.cumsum(self.probs, 1)], 1)
        return cum / cum[:, -1:]

    def cdf(self, y_grid) -> np.ndarray:
        """F_i(y) for every query i on a common grid -> (n_query, n_grid)."""
        yy = np.atleast_1d(np.asarray(y_grid, dtype=np.float64))
        cum = self._cum()
        return np.stack([np.interp(yy, self.borders, cum[i]) for i in range(len(cum))])

    def pit(self, y_obs) -> np.ndarray:
        """Row-wise probability integral transform F_i(y_obs[i]) -> (n_query,)."""
        y_obs = np.asarray(y_obs, dtype=np.float64)
        cum = self._cum()
        return np.array([np.interp(y_obs[i], self.borders, cum[i]) for i in range(len(cum))])

    def crps(self, y_obs) -> np.ndarray:
        """CRPS_i = integral (F_i(t) - 1{t >= y_obs[i]})^2 dt, exact for the piecewise-linear CDF
        (outer half-normal tails are treated as uniform over their bucket; their mass is ~1e-6)."""
        y_obs = np.asarray(y_obs, dtype=np.float64)
        b = self.borders
        out = np.empty(len(self.probs))
        cum = self._cum()
        for i in range(len(out)):
            y = float(np.clip(y_obs[i], b[0], b[-1]))
            edges = np.insert(b, np.searchsorted(b, y), y)
            F = np.interp(edges, b, cum[i])
            H = (edges >= y).astype(np.float64)

            Hl = H[:-1].copy()
            Hl[edges[1:] <= y] = 0.0
            Hl[edges[:-1] >= y] = 1.0
            a0, a1 = F[:-1] - Hl, F[1:] - Hl
            out[i] = np.sum(np.diff(edges) * (a0 * a0 + a0 * a1 + a1 * a1) / 3.0)
        return out

    def interval(self, level: float = 0.90):
        lo = self._q((1 - level) / 2)
        hi = self._q(1 - (1 - level) / 2)
        return lo, hi

    def _q(self, p: float) -> np.ndarray:
        cum = self._cum()
        return np.array([np.interp(p, cum[i], self.borders) for i in range(len(cum))])


def predictive(reg, X, quantile_levels: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> Predictive:
    """Run `reg.predict(X, output_type="full")` ONCE and return mean / median / sd / quantiles /
    full histogram law in raw y units.

    How the regressor builds it (regressor.py, tabpfn 8.5.0): the network emits 5000 logits per
    query over fixed z-space borders; logits are divided by softmax_temperature (0.9); each ensemble
    member's bucket probabilities are re-binned onto the common z-space borders and averaged
    (probability average unless average_before_softmax); `logits` returned by "full" are the LOG of
    that averaged probability vector; `criterion` is a FullSupportBarDistribution whose borders are
    z-borders * std(y_train) + mean(y_train). Mean uses bucket midpoints (half-normal tail means).
    """
    full = reg.predict(X, output_type="full", quantiles=list(quantile_levels))
    crit, logits = full["criterion"], full["logits"]
    probs = torch.softmax(logits.float(), -1).double().cpu().numpy()
    borders = crit.borders.double().cpu().numpy()
    var = crit.variance(logits.float().to(crit.borders.device)).cpu().numpy()
    return Predictive(
        mean=np.asarray(full["mean"], dtype=np.float64),
        median=np.asarray(full["median"], dtype=np.float64),
        sd=np.sqrt(np.clip(var, 0, None)).astype(np.float64),
        quantile_levels=np.asarray(quantile_levels, dtype=np.float64),
        quantiles=np.stack([np.asarray(q, dtype=np.float64) for q in full["quantiles"]], 1),
        probs=probs,
        borders=borders,
    )


def noop_test(reg, X) -> dict:
    """Control: overriding n -> n in every module must reproduce the baseline bit-exactly."""
    base = reg.predict(X, output_type="full")
    with SizeOverride(reg, n=lambda n, s: n) as ov:
        same = reg.predict(X, output_type="full")
    a, b = base["logits"].float(), same["logits"].float()
    finite = torch.isfinite(a) & torch.isfinite(b)                                          
    d_logit = float((a[finite] - b[finite]).abs().max())
    d_mean = float(np.abs(base["mean"] - same["mean"]).max())
    return dict(bit_exact=bool(torch.equal(a, b)), max_abs_diff_logits=d_logit,
                max_abs_diff_mean=d_mean, n_calls=ov.n_calls, n_modified=ov.n_modified,
                n_sites=len(ov.sites))

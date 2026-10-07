from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))


from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import importlib.metadata
from functools import lru_cache
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
from types import MethodType

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np


CONFIG = json.loads(Path(__file__).with_name("models.json").read_text(encoding="utf-8"))
MODELS = tuple(CONFIG)


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"{package} is not installed in this interpreter; see models.json and use a separate environment"
        ) from exc


def _inputs(X, y, Q):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    Q = np.asarray(Q, dtype=np.float32)
    if X.ndim != 2 or Q.ndim != 2 or X.shape[1] != Q.shape[1] or len(X) != len(y):
        raise ValueError("Expected X (n,p), y (n,), and Q (q,p) with matching p")
    if len(X) == 0 or len(Q) == 0:
        raise ValueError("Context and query must both be nonempty")
    return X, y, Q


def _local_checkpoint(name, model_dir=None):
    info = CONFIG[name]
    filename = info["checkpoint_file"]
    if name == "tabicl2" and os.environ.get("STUDY1_TABICL_CHECKPOINT") and model_dir is None:
        path = Path(os.environ["STUDY1_TABICL_CHECKPOINT"])
    else:
        base = Path(model_dir) if model_dir else Path(os.environ.get("STUDY1_MODEL_DIR", Path.home() / ".cache" / "tabpfn"))
        path = base / filename
    if not path.is_file() and name == "tabicl2" and model_dir is None:
        cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--jingang--TabICL" / "snapshots"
        path = next(iter(sorted(cache.glob("*/" + filename))), path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} checkpoint {filename}: {path}")
    return path


def _n_estimators(ensemble):
    if ensemble == "default":
        return None
    n = int(ensemble)
    if n < 1:
        raise ValueError("ensemble must be 'default' or a positive integer")
    return n


@contextmanager
def _tabpfn3_count_intervention(model, context_rows, query_rows, unique_sites, scope):
    """Replace only the specified v3 count argument, then restore the class hook."""
    from tabpfn.architectures.tabpfn_v3 import SoftmaxScalingMLP

    roots = getattr(model, "models_", None)
    if roots is None:
        roots = [getattr(model, "model_", None)]
    if not isinstance(roots, (list, tuple)):
        roots = [roots]
    selected = {}
    for root in roots:
        if root is None:
            continue
        for module_name, module in root.named_modules():
            if isinstance(module, SoftmaxScalingMLP):
                family = "feature" if "feature_distribution_embedder" in module_name else "icl" if "icl_blocks" in module_name else "other"
                if scope == "all" or family == scope:
                    selected[id(module)] = (module_name, family)
    if not selected:
        raise RuntimeError(f"No TabPFN-3 {scope} count modules found in fitted model")

    original = SoftmaxScalingMLP.forward
    calls = []
    replacements = {context_rows: unique_sites, context_rows + query_rows: unique_sites + query_rows}

    def forward(module, q, n):
        ni = int(n)
        target = selected.get(id(module))
        changed = target is not None and ni in replacements
        no = replacements[ni] if changed else ni
        if target is not None:
            calls.append(dict(module=target[0], family=target[1], n_in=ni, n_out=no, changed=changed))
        return original(module, q, no if changed else n)

    SoftmaxScalingMLP.forward = forward
    try:
        yield calls
    finally:
        SoftmaxScalingMLP.forward = original


@contextmanager
def _tabdpt_beta_intervention(model, unique_sites):
    """Freeze only the row-attention beta argument at the distinct-site count."""
    from tabdpt.model import TransformerEncoderLayer

    layers = [module for module in model.model.modules()
              if isinstance(module, TransformerEncoderLayer)]
    if not layers:
        raise RuntimeError("No TabDPT attention layers found in fitted model")

                                                                              
    fixed_rows = int(unique_sites + model.model.n_thinking_rows)
    audit = dict(layers=len(layers), fixed_beta_rows=fixed_rows,
                 thinking_rows=int(model.model.n_thinking_rows))

    def make_hook(original):
        def get_scale_param(_module, eval_pos, *, device, dtype):
            return original(fixed_rows, device=device, dtype=dtype)
        return get_scale_param

    try:
        for layer in layers:
            layer.get_scale_param = MethodType(make_hook(layer.get_scale_param), layer)
        yield audit
    finally:
        for layer in layers:
            del layer.get_scale_param


@dataclass
class FittedBackend:
    model_name: str
    model: object
    X: np.ndarray
    y: np.ndarray
    seed: int
    ensemble: object
    metadata: dict


@lru_cache(maxsize=1)
def _limix_predictor(checkpoint, config, device):
    """Reuse one predictor; every prediction supplies a fresh observed context."""
    import sys
    import torch
    repo = str(Path(checkpoint).parent.parent)
    sys.path.insert(0, repo)
    try:
        from inference.predictor import LimiXPredictor
    finally:
        sys.path.pop(0)
    return LimiXPredictor(device=torch.device(device), model_path=checkpoint,
                          inference_config=config)


def fit_backend(model_name, X, y, seed, device="cuda", ensemble="default", model_dir=None):
    """Fit a fresh official regression model once on observed context rows."""
    if model_name not in CONFIG:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODELS}")
    X, y, _ = _inputs(X, y, np.asarray(X)[:1])
    n_est = _n_estimators(ensemble)
    info = CONFIG[model_name]
    meta = dict(model=model_name, package=info["package"], checkpoint=info["checkpoint"],
                checkpoint_revision=info["revision"], source_revision=info.get("source_revision"),
                actual_input_rows=int(len(X)), unique_sites=int(len(np.unique(X, axis=0))),
                ensemble=ensemble, readout="predictive_mean",
                retrieval=info["retrieval"], truncation=info["truncation"],
                device=str(device), seed=int(seed))
    start = time.perf_counter()

    if model_name in ("tabpfn3", "tabpfn35"):
        meta["package_version"] = _version("tabpfn")
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        if model_name == "tabpfn35" and not hasattr(ModelVersion, "V3_5"):
            raise RuntimeError(f"tabpfn {meta['package_version']} cannot load V3_5; install the current TabPFN release in a separate environment")
        version = ModelVersion.V3 if model_name == "tabpfn3" else ModelVersion.V3_5
        path = _local_checkpoint(model_name, model_dir)
        opts = dict(model_path=str(path), device=device, random_state=int(seed), show_progress_bar=False,
                    categorical_features_indices=[],
                    inference_config={"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0})
        if n_est is not None:
            opts["n_estimators"] = n_est
        model = TabPFNRegressor.create_default_for_version(version, **opts)
        model.fit(X, y)
        meta.update(checkpoint=str(path), ensemble=getattr(model, "n_estimators_", getattr(model, "n_estimators", n_est)),
                    readout="TabPFN output_type=mean",
                    precision=str(getattr(model, "inference_precision", "native default")),
                    fit_mode=str(getattr(model, "fit_mode", "native default")))
    elif model_name == "tabicl2":
        meta["package_version"] = _version("tabicl")
        from tabicl import TabICLRegressor
        path = _local_checkpoint(model_name, model_dir)
        opts = dict(model_path=str(path), allow_auto_download=False, device=device,
                    random_state=int(seed))
        if n_est is not None:
            opts["n_estimators"] = n_est
        model = TabICLRegressor(**opts)
        model.fit(X, y)
        meta.update(checkpoint=str(path), ensemble=model.n_estimators,
                    readout="TabICL output_type=mean", precision=f"use_amp={model.use_amp}")
    elif model_name == "tabdpt":
        meta["package_version"] = _version("tabdpt")
        if not meta["package_version"].startswith("1.3."):
            raise RuntimeError(f"TabDPT-1.3 requires tabdpt==1.3.*, found {meta['package_version']}")
        from tabdpt import TabDPTRegressor

                                                      
        path = Path(model_dir) / info["checkpoint_file"] if model_dir else None
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
        opts = dict(device=device)
        if path is not None:
            opts["model_weight_path"] = str(path)
        model = TabDPTRegressor(**opts)
        model.fit(X, y)
        predict_defaults = inspect.signature(model.predict).parameters
        count = n_est if n_est is not None else predict_defaults["n_ensembles"].default
        context_size = predict_defaults["context_size"].default
        meta.update(checkpoint=str(path) if path else info["checkpoint"], ensemble=count,
                    retrieval="official default context_reduction (none for v1.3)",
                    truncation=f"official default context_size={context_size!r}; no cap when None",
                    readout="TabDPT output_type=mean (official default)")
    elif model_name == "tabfm":
        meta["package_version"] = _version("tabfm")
        import tabfm
        import torch
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

        if model_dir is not None and not Path(model_dir).is_dir():
            raise FileNotFoundError(f"TabFM checkpoint directory missing: {model_dir}")
        foundation = tabfm.tabfm_v1_0_0_pytorch.load(
            model_type="regression", checkpoint_path=str(model_dir) if model_dir else None, device=device)
        opts = dict(model=foundation, random_state=int(seed))
        if n_est is not None:
            opts["n_estimators"] = n_est
        model = tabfm.TabFMRegressor(**opts)
        model.fit(X, y)
        cap = model.max_num_rows
        row_patterns = [config[3] for configs in model.ensemble_generator_.ensemble_configs_.values()
                        for config in configs]
        context_row_indices = ("all" if all(pattern is None for pattern in row_patterns)
                               else [list(range(len(X))) if pattern is None else np.asarray(pattern, dtype=int).tolist()
                                     for pattern in row_patterns])
        meta.update(checkpoint=str(model_dir) if model_dir else info["checkpoint"],
                    checkpoint_revision=info["revision"] if model_dir else "HF default revision (not pinned)",
                    ensemble=getattr(model, "n_estimators", n_est), readout="TabFMRegressor.predict",
                    retrieval="no row sampling" if context_row_indices == "all" else "official ensemble row sampling",
                    context_row_indices=context_row_indices,
                    context_rows_per_member=len(X) if context_row_indices == "all" else [len(v) for v in context_row_indices],
                    truncation=f"official max_num_rows={cap}; {min(len(X), cap) if cap else len(X)} rows per ensemble context")
    elif model_name == "limix2":
        if n_est is not None:
            raise ValueError("LimiX-2 official predictor has no ensemble parameter")
        import torch
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

        if model_dir is None:
            raise ValueError("LimiX-2 requires model_dir=<official LimiX repository root>")
        repo = Path(model_dir)
        path = repo / "cache" / info["checkpoint_file"]
        config = repo / "config" / "reg_default_noretrieval_v2.json"
        if not path.is_file() or not config.is_file():
            raise FileNotFoundError(f"LimiX-2 needs {path} and {config}")
        hits_before = _limix_predictor.cache_info().hits
        model = _limix_predictor(str(path), str(config), str(device))
        revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        meta.update(package_version=_version("limix"), checkpoint=str(path),
                    repository_commit=revision,
                    predictor_reused=_limix_predictor.cache_info().hits > hits_before,
                    ensemble="single model", retrieval="disabled by reg_default_noretrieval_v2.json",
                    readout="LimiXPredictor.predict Regression")

    meta["fit_seconds"] = time.perf_counter() - start
    return FittedBackend(model_name, model, X, y, int(seed), meta["ensemble"], meta)


def predict_backend(fitted, Q, intervention=None):
    """Predict one query array; re-use `fitted` across matched methods.

    Intervention: {'scope': 'feature'|'all', 'unique_sites': int} for TabPFN-3.
    The class hook is restored even when prediction raises.
    """
    X = fitted.X
    Q = np.asarray(Q, dtype=np.float32)
    if Q.ndim != 2 or Q.shape[1] != X.shape[1] or len(Q) == 0:
        raise ValueError("Expected nonempty Q (q,p) with p matching fitted X")
    name, model = fitted.model_name, fitted.model
    meta = dict(fitted.metadata, query_rows=int(len(Q)), intervention=intervention)
    if intervention is not None:
        scopes = {"tabpfn3": ("feature", "all"), "tabdpt": ("beta48",)}
        if name not in scopes or set(intervention) != {"scope", "unique_sites"} or intervention["scope"] not in scopes[name]:
            raise ValueError("unsupported model/count intervention")
        unique_sites = int(intervention["unique_sites"])
        if unique_sites != meta["unique_sites"]:
            raise ValueError("intervention unique_sites must equal observed unique X rows")
    start = time.perf_counter()
    if name in ("tabpfn3", "tabpfn35"):
        ctx = (_tabpfn3_count_intervention(model, len(X), len(Q), unique_sites, intervention["scope"])
               if intervention is not None else nullcontext([]))
        with ctx as calls:
            pred = model.predict(Q, output_type="mean")
        meta.update(count_calls=calls, intervention_changed_calls=sum(c["changed"] for c in calls))
        if intervention is not None and not meta["intervention_changed_calls"]:
            raise RuntimeError("TabPFN-3 count intervention matched no actual module calls")
    elif name == "tabicl2":
        pred = model.predict(Q, output_type="mean")
    elif name == "tabdpt":
        kwargs = dict(seed=fitted.seed)
        if fitted.ensemble != inspect.signature(model.predict).parameters["n_ensembles"].default:
            kwargs["n_ensembles"] = fitted.ensemble
        ctx = (_tabdpt_beta_intervention(model, unique_sites)
               if intervention is not None else nullcontext(None))
        with ctx as beta_audit:
            pred = model.predict(Q, **kwargs)
        if beta_audit is not None:
            meta["beta_intervention"] = beta_audit
    elif name == "tabfm":
        pred = model.predict(Q)
        if str(meta["device"]).startswith("cuda"):
            import torch
            meta["torch_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(meta["device"]))
            meta["torch_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(meta["device"]))
    elif name == "limix2":
        pred = model.predict(X, fitted.y, Q, task_type="Regression")
        if str(meta["device"]).startswith("cuda"):
            import torch
            meta["torch_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(meta["device"]))
            meta["torch_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(meta["device"]))
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if len(pred) != len(Q) or not np.all(np.isfinite(pred)):
        raise RuntimeError(f"{name} returned invalid predictive mean shape/values: {pred.shape}")
    meta["predict_seconds"] = time.perf_counter() - start
    return pred, meta


def fit_predict(model_name, X, y, Q, seed, device="cuda", ensemble="default", model_dir=None, intervention=None):
    """Convenience call returning one native predictive mean and metadata."""
    fitted = fit_backend(model_name, X, y, seed, device, ensemble, model_dir)
    return predict_backend(fitted, Q, intervention)

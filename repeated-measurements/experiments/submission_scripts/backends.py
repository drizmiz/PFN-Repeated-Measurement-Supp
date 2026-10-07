
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import importlib.metadata as _md
import os

N_ESTIMATORS = 8
RANDOM_STATE = 0
DEVICE = os.environ.get("DEVICE", "cuda")

CKPT_V3 = os.environ.get("CKPT_V3", os.environ.get('CKPT_V3', str(_PKG_ROOT/'models/tabpfn-v3-regressor-v3_default.ckpt')))

_EXPECTED_PKG = {"tabpfn3": ("tabpfn", "8."), "tabpfn35": ("tabpfn", "9."), "tabicl": ("tabicl", "2."),
                 "tabdpt": ("tabdpt", "1.3"), "limix2": (None, None), "tabfm": ("tabfm", "1.")}

                                                                                          

                                                                                         

HF_REPO_LIMIX = ("stable-ai/LimiX-2", "LimiX-2.ckpt")
LIMIX_REPO = os.environ.get("LIMIX_REPO", os.environ.get('LIMIX_REPO', str(_PKG_ROOT/'external/LimiX')))
LIMIX_REG_CONFIG = os.path.join(LIMIX_REPO, "config", "reg_default_noretrieval_v2.json")                            

_TABFM_MODEL = None
_LIMIX_PRED = None
TABFM_MAX_ROWS = int(os.environ.get("TABFM_MAX_ROWS", "8192"))                                                                                            


class _LimiXRegressor:
    """sklearn-shaped wrapper around inference.predictor.LimiXPredictor (fit stores the context)."""

    def __init__(self):
        import torch
        from huggingface_hub import hf_hub_download
        from inference.predictor import LimiXPredictor
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("limix2 weights are licence-gated on Hugging Face: accept the licence on "
                               "the stable-ai/LimiX-2 page and set HF_TOKEN")
        global _LIMIX_PRED
        if _LIMIX_PRED is None:                                                                    
            ckpt = hf_hub_download(repo_id=HF_REPO_LIMIX[0], filename=HF_REPO_LIMIX[1],
                                   token=os.environ["HF_TOKEN"])
            _LIMIX_PRED = LimiXPredictor(device=torch.device(DEVICE), model_path=ckpt,
                                         inference_config=LIMIX_REG_CONFIG)
        self.model = _LIMIX_PRED

    def fit(self, X, y):
        import numpy as np
        self.X_, self.y_ = np.asarray(X, float), np.asarray(y, float)
        return self

    def predict(self, Xq):
        import numpy as np
        out = self.model.predict(self.X_, self.y_, np.asarray(Xq, float), task_type="Regression")
        return np.asarray(out, float).reshape(-1)


def _make_tabfm():
    import inspect
    from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("tabfm weights are licence-gated on Hugging Face: accept the licence on the "
                           "google/tabfm-1.0.0-pytorch page and set HF_TOKEN")
    global _TABFM_MODEL
    if _TABFM_MODEL is None:                                                                      

                                                                          
        _TABFM_MODEL = tabfm_v1_0_0_pytorch.load(model_type="regression", device=DEVICE)
    model = _TABFM_MODEL
    params = inspect.signature(TabFMRegressor).parameters
    kw = {"model": model}
    if "max_num_rows" in params:
        kw["max_num_rows"] = TABFM_MAX_ROWS
    else:
        raise RuntimeError("TabFMRegressor has no max_num_rows argument in this version; its default "
                           "context is 100 sampled rows, which would silently subsample the Expanded "
                           "table. Parameters: %s" % list(params))
    for k, v in (("random_state", RANDOM_STATE), ("seed", RANDOM_STATE), ("device", DEVICE)):
        if k in params:
            kw[k] = v
    return TabFMRegressor(**kw)


def _make_tabdpt():
    import inspect
    from tabdpt import TabDPTRegressor
    params = inspect.signature(TabDPTRegressor).parameters
    kw = {}
    for k, v in (("device", DEVICE), ("random_state", RANDOM_STATE), ("seed", RANDOM_STATE)):
        if k in params:
            kw[k] = v
    reg = TabDPTRegressor(**kw)

                                                                                            

                                                                           
    pp = inspect.signature(reg.predict).parameters
    _fit = reg.fit
    def fit(X, y, _f=_fit):                                                                        
        reg._n_ctx = len(y)
        _f(X, y)
        return reg
    reg.fit = fit
    _orig = reg.predict
    def predict(X, _o=_orig):
        kw2 = {}
        if "seed" in pp:
            kw2["seed"] = RANDOM_STATE
        if "context_size" in pp:
            d = pp["context_size"].default
            kw2["context_size"] = max(int(d) if isinstance(d, int) else 0, int(getattr(reg, "_n_ctx", 0)))
        return _o(X, **kw2)
    reg.predict = predict
    return reg


def _check(backend):
    pkg, prefix = _EXPECTED_PKG[backend]
    if pkg is None:                                                                 
        import importlib
        try:
            importlib.import_module("inference.predictor")
        except ImportError as e:
            raise RuntimeError("%s needs the LimiX repository installed (pip install -e external/LimiX) "
                               "in this interpreter: %s" % (backend, e))
        return "git"
    try:
        v = _md.version(pkg)
    except _md.PackageNotFoundError:
        raise RuntimeError("%s needs package %s, not installed in this interpreter (see the table "
                           "in backends.py for which venv to use)" % (backend, pkg))
    if not v.startswith(prefix):
        raise RuntimeError("%s expects %s %s* but this interpreter has %s %s -- wrong venv"
                           % (backend, pkg, prefix, pkg, v))
    return v


def make(backend, n_estimators=N_ESTIMATORS):
    """Construct a fresh, unfitted regressor for `backend` with the paper's fixed configuration."""
    _check(backend)
    if backend == "tabpfn3":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(model_path=CKPT_V3, n_estimators=n_estimators, device=DEVICE,
                               random_state=RANDOM_STATE, ignore_pretraining_limits=True)
    if backend == "tabpfn35":
        if not os.environ.get("TABPFN_TOKEN"):
            raise RuntimeError("tabpfn35 weights are licence-gated: accept the licence at "
                               "https://ux.priorlabs.ai (Licenses tab), copy the API key from "
                               "https://ux.priorlabs.ai/account, and set TABPFN_TOKEN")
        from tabpfn import TabPFNRegressor

                                                                              
        return TabPFNRegressor(n_estimators=n_estimators, device=DEVICE,
                               random_state=RANDOM_STATE, ignore_pretraining_limits=True)
    if backend == "tabicl":
        from tabicl import TabICLRegressor
        return TabICLRegressor(device=DEVICE, random_state=RANDOM_STATE)
    if backend == "tabdpt":
        return _make_tabdpt()
    if backend == "limix2":
        return _LimiXRegressor()
    if backend == "tabfm":
        return _make_tabfm()
    raise ValueError("unknown backend %r; choose from %s" % (backend, sorted(_EXPECTED_PKG)))


def describe(backend):
    """One line for the reproducibility checklist."""
    v = _check(backend)
    extra = {"tabpfn3": "checkpoint " + os.path.basename(CKPT_V3),
             "tabpfn35": "package-default 3.5 regressor weights (licence-gated download)",
             "tabicl": "package-default weights",
             "tabdpt": "Layer6/TabDPT package-default weights, package-default ensemble",
             "limix2": "stable-ai/LimiX-2 LimiX-2.ckpt, config " + LIMIX_REG_CONFIG,
             "tabfm": "google/tabfm-1.0.0-pytorch regression checkpoint, max_num_rows=%d" % TABFM_MAX_ROWS}[backend]
    if backend == "tabicl":
        try:
            from tabicl import TabICLRegressor
            import inspect
            d = inspect.signature(TabICLRegressor).parameters.get("checkpoint_version")
            if d is not None:
                extra += ", checkpoint_version default = %r" % (d.default,)
        except Exception:
            pass
    return ("%s: %s %s, %s, n_estimators=%d, random_state=%d, device=%s, fingerprint on, "
            "default softmax temperature, ignore_pretraining_limits=True, no test-time tuning"
            % (backend, _EXPECTED_PKG[backend][0], v, extra, N_ESTIMATORS, RANDOM_STATE, DEVICE))


if __name__ == "__main__":
    import sys
    b = sys.argv[1] if len(sys.argv) > 1 else "tabpfn3"
    print(describe(b))

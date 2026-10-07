from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import os
os.environ.setdefault('TABPFN_DISABLE_TELEMETRY','1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
import time
from pathlib import Path
from contextlib import contextmanager
import numpy as np

MODELS=('tabpfn3','tabicl2','tabpfn2')

def checkpoint_paths():
    cache=Path(os.environ.get('STUDY1_MODEL_DIR',str(Path.home()/'.cache'/'tabpfn')))
    paths={'tabpfn3':cache/'tabpfn-v3-regressor-v3_default.ckpt','tabpfn2':cache/'tabpfn-v2-regressor.ckpt'}
    override=os.environ.get('STUDY1_TABICL_CHECKPOINT')
    if override: paths['tabicl2']=Path(override)
    elif (cache/'tabicl-regressor-v2-20260212.ckpt').exists(): paths['tabicl2']=cache/'tabicl-regressor-v2-20260212.ckpt'
    else:
        candidates=sorted((Path.home()/'.cache'/'huggingface'/'hub'/'models--jingang--TabICL'/'snapshots').glob('*/tabicl-regressor-v2-20260212.ckpt'))
        paths['tabicl2']=candidates[0] if candidates else cache/'tabicl-regressor-v2-20260212.ckpt'
    return paths

def make_model(name,seed,device='cuda',ensemble='1'):
    import torch
    torch.set_num_threads(int(os.environ.get('STUDY1_THREADS','4')))
    path=checkpoint_paths()[name]
    if not path.is_file(): raise FileNotFoundError(f'Missing declared checkpoint: {path}')
    if name=='tabicl2':
        from tabicl import TabICLRegressor
        options=dict(model_path=str(path),allow_auto_download=False,device=device,random_state=int(seed),use_amp=False,n_jobs=4)
        if str(ensemble)!='default': options['n_estimators']=int(ensemble)
        return TabICLRegressor(**options)
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    opts=dict(model_path=str(path),device=device,random_state=int(seed),show_progress_bar=False,fit_mode='fit_preprocessors',inference_precision=torch.float32,ignore_pretraining_limits=True,categorical_features_indices=[],inference_config={'MIN_UNIQUE_FOR_NUMERICAL_FEATURES':0})
    if str(ensemble)!='default': opts.update(n_estimators=int(ensemble))
    return TabPFNRegressor.create_default_for_version(ModelVersion.V3 if name=='tabpfn3' else ModelVersion.V2,**opts)

@contextmanager
def size_intervention(model,name,actual,replacement,scope='all'):
    """Only modify eligible module calls of the supplied fitted model; restore afterward."""
    if name!='tabpfn3': raise ValueError('Module interventions are specified only for TabPFN-3.')
    from tabpfn.architectures.tabpfn_v3 import SoftmaxScalingMLP
    roots=getattr(model,'models_',None)
    if roots is None: roots=[getattr(model,'model_',None)]
    if not isinstance(roots,(list,tuple)): roots=[roots]
    selected={}
    for root in roots:
        if root is None: continue
        for module_name,module in root.named_modules():
            if isinstance(module,SoftmaxScalingMLP):
                family='feature' if 'feature_distribution_embedder' in module_name else 'icl' if 'icl_blocks' in module_name else 'other'
                if scope=='all' or scope==family: selected[id(module)]=(module_name,family)
    if not selected: raise RuntimeError(f'No size modules for {scope}; inspect installed architecture.')
    original=SoftmaxScalingMLP.forward
    calls=[]
    def forward(module,q,n):
        ni=int(n); target=selected.get(id(module)); changed=target is not None and ni==actual
        if target is not None: calls.append(dict(module=target[0],family=target[1],n_in=ni,n_out=replacement if changed else ni,changed=changed))
        return original(module,q,replacement if changed else n)
    SoftmaxScalingMLP.forward=forward
    try: yield calls
    finally: SoftmaxScalingMLP.forward=original

def predict(model,name,X,quantiles=199,batch_size=256):
    levels=((np.arange(quantiles)+.5)/quantiles).tolist()
    atoms=[]; native=[]
    for start in range(0,len(X),batch_size):
        xx=np.asarray(X[start:start+batch_size],dtype=np.float32)
        if name=='tabicl2': out=model.predict(xx,output_type=['mean','quantiles'],alphas=levels)
        else: out=model.predict(xx,output_type='main',quantiles=levels)
        a=np.asarray(out['quantiles'],dtype=float)

        if a.shape==(quantiles,len(xx)): a=a.T
        if a.shape!=(len(xx),quantiles): raise ValueError(f'Unexpected {name} quantile shape {a.shape}')
        atoms.append(a)
        native.append(np.asarray(out['mean'],dtype=float).reshape(-1))
    return np.concatenate(atoms),np.concatenate(native)

def fit_model(name,seed,X,y,device,ensemble):
    model=make_model(name,seed,device,ensemble)
    start=time.perf_counter();model.fit(np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.float32))
    return model,time.perf_counter()-start


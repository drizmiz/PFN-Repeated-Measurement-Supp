
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import gc
import json
import numpy as np
import pandas as pd
import study
from linear_mechanism import predict


def main():
    import torch
    from backends import make_model,size_intervention
    from tabpfn.preprocessing.torch.torch_standard_scaler import TorchStandardScaler
    out=study.RESULTS/'replication_repair';out.mkdir(exist_ok=True)
    raw=study.RAW/'replication_repair';raw.mkdir(exist_ok=True)
    rows=[]
    factor=float(np.sqrt((12*64-1)/(64*(12-1))))
    for seed in study.SEEDS:
        data=study.design(seed,'linear')
        model=make_model('tabpfn3',seed,'cuda','1')
        config=dict(model.inference_config);config['FINGERPRINT_FEATURE']=False
        model.set_params(inference_config=config)
        model.fit(np.repeat(data['x'],64)[:,None].astype('float32'),np.repeat(data['y'],64).astype('float32'))
        scaler=model.models_[0].standard_scaler
        original=TorchStandardScaler.fit
        corrected=[]
        def adjusted(self,x):
            result=original(self,x)
            if self is scaler and x.shape[0]==768:
                result=dict(result);result['std']=result['std']*factor
                corrected.append(int(x.shape[0]))
            return result
        TorchStandardScaler.fit=adjusted
        try:
            with size_intervention(model,'tabpfn3',768,12,'all'):
                native,atoms=predict(model,data,{})
        finally: TorchStandardScaler.fit=original
        old=dict(np.load(study.RAW/'fingerprint_probe'/f'seed{seed}_fp0_raw_r1.npz'))
        np.savez_compressed(raw/f'seed{seed}.npz',native_mean=native,atom_mean=atoms.mean(1))
        row=dict(seed=seed,scale_factor=factor,corrected_calls=len(corrected),
            max_native_gap=float(np.max(np.abs(native-old['native_mean']))),
            rms_native_gap=float(np.sqrt(np.mean((native-old['native_mean'])**2))),
            max_atom_mean_gap=float(np.max(np.abs(atoms.mean(1)-old['atom_mean']))),**study.metrics(data,native))
        rows.append(row);print(json.dumps(row),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out/'design_metrics.csv',index=False)


if __name__=='__main__':main()

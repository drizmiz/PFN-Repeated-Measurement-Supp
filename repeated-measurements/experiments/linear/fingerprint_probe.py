
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import gc
import inspect
import json
import time
import numpy as np
import pandas as pd
import study
from linear_mechanism import instrument,predict


def main():
    import torch
    from backends import make_model
    out=study.RESULTS/'fingerprint_probe';out.mkdir(exist_ok=True)
    raw=study.RAW/'fingerprint_probe';raw.mkdir(exist_ok=True)
    rows=[]; mappings=[]; attention=[]; fidelity=[]
    started=time.perf_counter()
    for seed in study.SEEDS:
        data=study.design(seed,'linear')
        for fingerprint,r in ((True,64),(False,1),(False,64)):
            model=make_model('tabpfn3',seed,'cuda','1')
            if not fingerprint:
                config=dict(model.inference_config)
                config['FINGERPRINT_FEATURE']=False
                model.set_params(inference_config=config)
            model.fit(np.repeat(data['x'],r)[:,None].astype('float32'),np.repeat(data['y'],r).astype('float32'))
            root=model.models_[0]
            def capture(module,args,kwargs):
                bound=inspect.signature(module.forward).bind_partial(*args,**kwargs).arguments
                xx=bound['x']; yy=bound['y']
                xx=xx['main'] if isinstance(xx,dict) else xx
                yy=yy['main'] if isinstance(yy,dict) else yy
                labels=yy.detach().cpu().numpy().reshape(-1)
                values,indices,counts=np.unique(labels,return_inverse=True,return_counts=True)
                expected=np.repeat(np.arange(12),r)
                order_ok=np.array_equal(indices,expected)
                features=xx[:len(labels)].detach().cpu().numpy()[:,0,:]
                spreads=[]
                for i in range(12):
                    group=features[indices==i]
                    spreads.append(np.ptp(group,axis=0))
                mappings.append(dict(seed=seed,r=r,fingerprint=fingerprint,n_features=features.shape[1],
                    unique_labels=len(values),all_counts_r=bool(np.all(counts==r)),contiguous_site_order=bool(order_ok),
                    max_within_site_feature_range=float(np.max(spreads)),
                    per_feature_max_range=json.dumps(np.max(spreads,axis=0).tolist()),
                    configured_fingerprint=bool(model.inference_config_.FINGERPRINT_FEATURE)))
                if not order_ok:
                    raise RuntimeError('Physical site order requires a new attention mapping.')
            handle=root.register_forward_pre_hook(capture,with_kwargs=True)
            for scope in (['raw'] if fingerprint or r==1 else ['raw','all']):
                state=dict(chosen=np.unique(data['inverse'][:,[8,40,72]].ravel()))
                records=[];calls=[]
                with instrument(model,r,scope,data,state,records,calls):
                    native,atoms=predict(model,data,state)
                for record in records: record.update(seed=seed,fingerprint=fingerprint)
                attention.extend(records)
                np.savez_compressed(raw/f'seed{seed}_fp{int(fingerprint)}_{scope}_r{r}.npz',native_mean=native,atom_mean=atoms.mean(1))
                rows.append(dict(seed=seed,r=r,fingerprint=fingerprint,scope=scope,**study.metrics(data,native)))
                if fingerprint:
                    old=dict(np.load(study.path('linear',seed,'raw',r)))
                    fidelity.append(dict(seed=seed,max_difference=float(np.max(np.abs(native-old['native_mean'])))))
                print(json.dumps(dict(seed=seed,r=r,fingerprint=fingerprint,scope=scope,seconds=time.perf_counter()-started)),flush=True)
            handle.remove();del model;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out/'design_metrics.csv',index=False)
    pd.DataFrame(rows).groupby(['fingerprint','scope','r']).mean(numeric_only=True).to_csv(out/'condition_metrics.csv')
    pd.DataFrame(mappings).drop_duplicates().to_csv(out/'input_mapping.csv',index=False)
    pd.DataFrame(attention).to_csv(out/'attention_replay.csv',index=False)
    pd.DataFrame(fidelity).to_csv(out/'fidelity.csv',index=False)
    study.write_json(out/'completion.json',dict(fits=18,predictions=len(rows),seconds=time.perf_counter()-started))


if __name__=='__main__': main()

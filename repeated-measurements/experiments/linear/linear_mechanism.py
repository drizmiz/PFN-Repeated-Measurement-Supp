
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

from collections import Counter
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import study

OUT=study.RESULTS/'linear_mechanism'
RAW=study.RAW/'linear_mechanism'


@contextmanager
def instrument(model,r,scope,data,state,records,calls):
    import torch
    import tabpfn.architectures.tabpfn_v3 as arch
    root=model.models_[0]
    mapping={id(mod):name for name,mod in root.named_modules() if isinstance(mod,arch.SoftmaxScalingMLP)}
    nlayer=len(root.icl_blocks)
    forward=arch.SoftmaxScalingMLP.forward
    sdpa=arch._batched_scaled_dot_product_attention
    def family(name):
        return 'feature' if 'feature_distribution_embedder' in name else 'icl' if 'icl_blocks' in name else 'other'
    def selected(name):
        fam=family(name)
        if scope=='all': return True
        if scope in ('feature','icl'): return fam==scope
        if scope in ('icl_early','icl_late') and fam=='icl':
            layer=int(name.split('icl_blocks.')[1].split('.')[0])
            return (layer<nlayer//2) if scope=='icl_early' else (layer>=nlayer//2)
        return False
    def adjusted(module,q,n):
        name=mapping.get(id(module),'unknown'); change=selected(name) and int(n)==12*r
        calls.append(dict(module=name,family=family(name),n_in=int(n),n_out=12 if change else int(n),changed=change))
        return forward(module,q,12 if change else n)
    def replay(q,k,v,softmax_scaling_layer=None,*args,**kwargs):
        name=mapping.get(id(softmax_scaling_layer),'unknown')
        if scope=='raw' and name.startswith('icl_blocks.') and k is not None and k.shape[1]==12*r:
            nq=state['stop']-state['start']
            if q.shape[1]==nq: test=q
            elif q.shape[1]==12*r+nq: test=q[:,12*r:]
            elif q.shape[1]==12*r: test=None
            else: raise ValueError(f'Unexpected ICL query axis {q.shape}, nq={nq}, r={r}')
            if test is not None:
                chosen=state['chosen']; local=chosen[(chosen>=state['start'])&(chosen<state['stop'])]
                if len(local):
                    test=test[:,local-state['start']].detach()
                    keys=k.detach()
                    grouped=keys.reshape(keys.shape[0],12,r,*keys.shape[2:])
                    duplicate_error=float((grouped-grouped[:,:,:1]).abs().max())
                    keys=keys.repeat_interleave(test.shape[2]//keys.shape[2],dim=2)
                    for mode,n in [('actual',12*r),('site_count',12)]:
                        scaled=forward(softmax_scaling_layer,test,n)
                        logits=torch.einsum('bshd,bnhd->bhsn',scaled,keys).float()/math.sqrt(test.shape[-1])
                        weights=torch.softmax(logits,dim=-1).reshape(test.shape[0],test.shape[2],len(local),12,r).sum(-1)
                        entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1)
                        bary=(weights*torch.as_tensor(data['x'],device=weights.device,dtype=weights.dtype)).sum(-1)
                        values={
                            'entropy':entropy.mean((0,1)).cpu().numpy(),
                            'effective_sites':entropy.exp().mean((0,1)).cpu().numpy(),
                            'max_site_mass':weights.max(-1).values.mean((0,1)).cpu().numpy(),
                            'barycenter':bary.mean((0,1)).cpu().numpy(),
                        }
                        for i,qi in enumerate(local):
                            records.append(dict(layer=int(name.split('.')[1]),mode=mode,r=r,query_index=int(qi),
                                x=float(data['q'][qi]),duplicate_key_max=duplicate_error,
                                **{key:float(value[i]) for key,value in values.items()}))
        return sdpa(q,k,v,softmax_scaling_layer,*args,**kwargs)
    arch.SoftmaxScalingMLP.forward=adjusted
    arch._batched_scaled_dot_product_attention=replay
    try: yield
    finally:
        arch.SoftmaxScalingMLP.forward=forward
        arch._batched_scaled_dot_product_attention=sdpa


def predict(model,data,state):
    means=[];atoms=[]
    levels=((np.arange(199)+.5)/199).tolist()
    for start in range(0,len(data['q']),256):
        state.update(start=start,stop=min(start+256,len(data['q'])))
        output=model.predict(data['q'][start:state['stop'],None].astype('float32'),output_type='main',quantiles=levels)
        means.append(np.asarray(output['mean']).reshape(-1))
        atoms.append(np.asarray(output['quantiles']).T)
    return np.concatenate(means),np.concatenate(atoms)


def main():
    from backends import fit_model
    import torch
    OUT.mkdir(parents=True,exist_ok=True); RAW.mkdir(parents=True,exist_ok=True)
    rows=[]; replay_rows=[]; fidelity=[]; coverage=[]
    start=time.perf_counter()
    for seed in study.SEEDS:
        data=study.design(seed,'linear')
        chosen=np.unique(data['inverse'][:,[8,40,72]].ravel())
        for r in (1,64):
            model,_=fit_model('tabpfn3',seed,np.repeat(data['x'],r)[:,None],np.repeat(data['y'],r),'cuda','1')
            for scope in (['raw'] if r==1 else ['raw','feature','icl','all','icl_early','icl_late']):
                state=dict(chosen=chosen); records=[];calls=[]
                with instrument(model,r,scope,data,state,records,calls):
                    native,atoms=predict(model,data,state)
                for row in records: row['seed']=seed
                replay_rows.extend(records)
                for item,n in Counter((c['module'],c['family'],c['n_in'],c['n_out'],c['changed']) for c in calls).items():
                    coverage.append(dict(seed=seed,r=r,scope=scope,module=item[0],family=item[1],n_in=item[2],n_out=item[3],changed=item[4],calls=n))
                np.savez_compressed(RAW/f'seed{seed}_{scope}_r{r}.npz',native_mean=native,atom_mean=atoms.mean(1))
                rows.append(dict(seed=seed,r=r,scope=scope,**study.metrics(data,native)))
                if scope in ('raw','all'):
                    old=dict(np.load(study.path('linear',seed,'raw' if scope=='raw' else 'size_m',r)))
                    fidelity.append(dict(seed=seed,r=r,scope=scope,max_native_difference=float(np.max(np.abs(native-old['native_mean']))),
                        max_atom_difference=float(np.max(np.abs(atoms-old['atoms'])))))
                print(json.dumps(dict(seed=seed,r=r,scope=scope,seconds=round(time.perf_counter()-start,2))),flush=True)
            del model
            gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT/'design_metrics.csv',index=False)
    pd.DataFrame(rows).groupby(['scope','r']).mean(numeric_only=True).to_csv(OUT/'condition_metrics.csv')
    pd.DataFrame(replay_rows).to_csv(OUT/'attention_replay.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'module_call_coverage.csv',index=False)
    pd.DataFrame(fidelity).to_csv(OUT/'fidelity.csv',index=False)
    study.write_json(OUT/'completion.json',dict(fits=12,predictions=len(rows),replay_rows=len(replay_rows),seconds=time.perf_counter()-start))


if __name__=='__main__': main()

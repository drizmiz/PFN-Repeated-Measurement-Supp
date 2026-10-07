
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

from pathlib import Path
import json
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from scipy.stats import invgamma
from validate import cue_cost,sufficient
OUT=Path(__file__).resolve().parent/'results'

def cell_mean(mean,sse,r,locations,V,query,K,t,prior_mean,b=1):
    bits=((np.arange(2**K)[:,None]>>np.arange(K))&1)*2-1
    N,m=mean.shape;ans=[];a=4.;prior_b=3*prior_mean
    for st in range(0,N,256):
        ys=mean[st:st+256]; loc=locations[st:st+256]; vv=V[st:st+256]; q= query[st:st+256]
        cand=bits[:,loc].transpose(1,0,2)
        residual=sse[st:st+256,None]+r*((ys[:,None,:]-cand)**2).sum(axis=2)
        lp=b*t*(vv@bits.T)-(a+m*r/2)*np.log(prior_b+.5*residual)
        prob=np.exp(lp-logsumexp(lp,axis=1)[:,None])
        ans.append((prob*bits[:,q].T).sum(axis=1))
    return np.concatenate(ans)

def cell_checks(N=8192):
    rows=[]
    for K in [2,4,8]:
      for m in [2,4]:
       for nu in [.003,.03,.3]:
        rng=np.random.default_rng(902811);t=.25
        H=rng.choice([-1.,1.],size=(N,K));loc=rng.integers(K,size=(N,m));query=rng.integers(K,size=N)
        T=np.take_along_axis(H,loc,axis=1);target=H[np.arange(N),query]
        actual_s=3*nu/rng.gamma(4.,1.,size=N);V=t*H+rng.normal(size=(N,K))
        first=T+np.sqrt(actual_s)[:,None]*rng.normal(size=(N,m))
        single=cell_mean(first,np.zeros(N),1,loc,V,query,K,t,nu)
        empty=(loc!=query[:,None]).all(axis=1)
        for r in [4,16]:
            ym,w=sufficient(T,actual_s,first,r,rng)
            grouped=cell_mean(ym,w,r,loc,V,query,K,t,nu)
            row=cell_mean(ym,w,r,loc,V,query,K,t,nu,b=r)
            direct=(row-target)**2-(single-target)**2
            net=(row-grouped)**2-(single-grouped)**2
            identity=direct-net
            predicted_empty=np.tanh(r*t*V[np.arange(N),query])
            err=abs(row[empty]-predicted_empty[empty])
            rows.append(dict(K=K,m=m,r=r,nu=nu,t=t,N=N,net=float(net.mean()),net_se=float(net.std(ddof=1)/np.sqrt(N)),direct_net=float(direct.mean()),direct_se=float(direct.std(ddof=1)/np.sqrt(N)),identity_z=float(identity.mean()/(identity.std(ddof=1)/np.sqrt(N))),low_noise_limit=(1-1/K)**m*cue_cost(t,r,beta=1),empty_prediction_max_error=float(err.max()) if len(err) else 0,uncovered_fraction=float(empty.mean())))
        print('cells',K,m,nu,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'full_cell_posteriors.csv',index=False)

def uniform_noise_posterior(y,w,r,order_s=520,order_t=36):

    if r==1:
        nodes,weights=leggauss(order_s);sv=(1+nodes)/2;ls=np.log(sv);logw=np.log(weights/2)
    else:
        ls=np.linspace(-32,0,order_s);ds=ls[1]-ls[0];sv=np.exp(ls)
        ws=np.ones(order_s)*ds;ws[[0,-1]]*=.5
        logw=np.log(ws)+ls
    hx,hw=hermgauss(order_t);lh=np.log(hw)-.5*np.log(np.pi)
    ans=[];varmeans=[];cdf=[];m=y.shape[1]
    for st in range(0,len(y),64):
        ym=y[st:st+64];ww=w[st:st+64]
        T=ym[:,None,:,None]+np.sqrt(2*sv[None,:,None,None]/r)*hx
        lp=np.log(2)-2*np.logaddexp(T,-T);ll=logsumexp(lp+lh,axis=-1)
        tw=np.exp(lp+lh-ll[...,None]);g=(tw*np.tanh(T)).sum(axis=-1).sum(axis=-1)/np.sqrt(m)
        post=ll.sum(axis=-1)+logw
        if r>1:post-=m*(r-1)/2*ls[None,:]+ww[:,None]/(2*sv[None,:])
        ps=np.exp(post-logsumexp(post,axis=1)[:,None])
        ans.append((ps*g).sum(axis=1));varmeans.append(ps@sv)
        cdf.append((ps*(sv[None,:]<=ww[:,None])).sum(axis=1))
    return np.concatenate(ans),np.concatenate(varmeans),np.concatenate(cdf)

def frozen_checks(N=2048):
    rng=np.random.default_rng(902812);m=2
    T=np.arctanh(rng.uniform(-1,1,(N,m)));truth=np.tanh(T).sum(axis=1)/np.sqrt(m)
    mu0,_,_=uniform_noise_posterior(T,np.zeros(N),1,64,48)
    e1=(mu0-truth)**2
    rows=[];checks=[]
    ref0,_,_=uniform_noise_posterior(T[:512],np.zeros(512),1,128,96)
    checks.append(dict(kind='Single noiseless calibration',N=512,max_prediction_diff=float(abs(ref0-mu0[:512]).max())))
    for s in [.03,.003,.0003,.00003]:
        rs=np.random.default_rng(902813)
        first=T+np.sqrt(s)*rs.normal(size=(N,m));single,_,_=uniform_noise_posterior(first,np.zeros(N),1,64,48)
        ref1,_,_=uniform_noise_posterior(first[:512],np.zeros(512),1,128,96)
        checks.append(dict(kind='Single noisy',s=s,N=512,max_prediction_diff=float(abs(ref1-single[:512]).max())))
        for r in [4,8]:
            mean,w=sufficient(T,np.full(N,s),first,r,rs)
            pred,vs,cdf=uniform_noise_posterior(mean,w,r)
            if s in [.03,.00003] and r==4:
                ref,rv,rc=uniform_noise_posterior(mean[:128],w[:128],r,1040,64)
                checks.append(dict(s=s,r=r,max_prediction_diff=float(abs(ref-pred[:128]).max()),max_variance_diff=float(abs(rv-vs[:128]).max())))
            for t in [.03,.2]:
                D=cue_cost(t,r,beta=.5)
                delta=D+(pred-truth)**2-(single-truth)**2
                rows.append(dict(s=s,r=r,m=m,N=N,t=t,net=float(delta.mean()),net_se=float(delta.std(ddof=1)/np.sqrt(N)),limit=D-e1.mean(),cue_cost=D,single_calibration=e1.mean(),row_response_mse=float(((pred-truth)**2).mean()),posterior_variance_over_sse=float((vs/w).mean()),variance_over_sse_limit=1/(m*(r-1)-4),posterior_cdf_ratio_one=float(cdf.mean()),cdf_limit=float(invgamma.cdf(1,a=m*(r-1)/2-1,scale=.5))))
        print('frozen',s,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'frozen_low_noise.csv',index=False)
    (OUT/'frozen_quadrature_checks.json').write_text(json.dumps(checks,indent=2))

if __name__=='__main__':
    cell_checks();frozen_checks()

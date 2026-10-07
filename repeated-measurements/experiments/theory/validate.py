
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp, expit, ndtr
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent/'results'

def logistic_posterior(mean, sse, count, prior_scale, ns=24, nt=24):
    """Prior T_i=atanh(U_i), U_i~Uniform[-1,1]; S~scale*Uniform[.5,1.5]."""
    gx,gw=leggauss(ns); vs=prior_scale*(1+.5*gx); lw_s=np.log(gw/2)
    hx,hw=hermgauss(nt); lw_t=np.log(hw)-.5*np.log(np.pi)
    vals=[]; vari=[]
    m=mean.shape[1]
    for st in range(0,len(mean),256):
        ym=mean[st:st+256]; ww=sse[st:st+256]
        tt=ym[:,None,:,None]+np.sqrt(2*vs[None,:,None,None]/count)*hx
        lp=np.log(2)-2*np.logaddexp(tt,-tt)
        ll=logsumexp(lp+lw_t,axis=-1)
        wt=np.exp(lp+lw_t-ll[...,None])
        gg=np.tanh(tt)
        mg=(wt*gg).sum(axis=-1)
        vg=(wt*gg**2).sum(axis=-1)-mg**2
        gm=mg.sum(axis=-1)/np.sqrt(m); gv=vg.sum(axis=-1)/m
        ls=ll.sum(axis=-1)+lw_s
        if count>1:ls-=.5*m*(count-1)*np.log(vs)[None,:]+ww[:,None]/(2*vs[None,:])
        ps=np.exp(ls-logsumexp(ls,axis=1)[:,None])
        mu=(ps*gm).sum(axis=1)
        vals.append(mu);vari.append((ps*(gv+gm**2)).sum(axis=1)-mu**2)
    return np.concatenate(vals),np.concatenate(vari)

def cue_cost(t,r,beta=.5):
    fun=lambda z: beta**2*(np.tanh(r*t*(t+z))-np.tanh(t*(t+z)))**2*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    return quad(fun,-11,11,epsabs=2e-13,epsrel=2e-12)[0]

def sufficient(mean_true, actual_s, first, r, rng):
    if r==1:return first,np.zeros(len(first))
    extra=mean_true+np.sqrt(actual_s/(r-1))[:,None]*rng.normal(size=mean_true.shape)
    ym=(first+(r-1)*extra)/r
    w=(r-1)/r*((first-extra)**2).sum(axis=1)
    if r>2:w+=actual_s*rng.chisquare(mean_true.shape[1]*(r-2),len(first))
    return ym,w

def phase_checks(N):
    m=2;beta=.5;K=8/15;rs=[1,2,3,4,6,8,12,16];Qs=[3,12,56,240]
    rows=[];quadchecks=[]
    for nu in [.02,.005,.00125,.0003125]:
        rng=np.random.default_rng(902801)
        tt=np.arctanh(rng.uniform(-1,1,(N,m)))
        actual_s=nu*rng.uniform(.5,1.5,N)
        first=tt+np.sqrt(actual_s)[:,None]*rng.normal(size=(N,m))
        truth=np.tanh(tt).sum(axis=1)/np.sqrt(m)
        one,_=logistic_posterior(first,np.zeros(N),1,nu)
        base_error=(one-truth)**2
        for r in rs:
            ym,w=sufficient(tt,actual_s,first,r,rng)
            pred,pvar=logistic_posterior(ym,w,r,nu)
            gain_square=(pred-one)**2
            gain_direct=base_error-(pred-truth)**2
            gap=gain_direct-gain_square
            if r in [1,16] and nu in [.02,.0003125]:
                refined,_=logistic_posterior(ym[:512],w[:512],r,nu,48,48)
                quadchecks.append(dict(nu=nu,r=r,N=512,max_prediction_diff=float(abs(refined-pred[:512]).max())))
            for Q in Qs:
                t=np.sqrt(nu*K/(Q*beta**2))
                D=cue_cost(t,r,beta); lead=nu*K*((r-1)**2/Q-(1-1/r))
                net=D-gain_square.mean()
                rows.append(dict(nu=nu,m=m,r=r,Q=Q,t=t,N=N,B=float(gain_square.mean()),B_se=float(gain_square.std(ddof=1)/np.sqrt(N)),B_leading=nu*K*(1-1/r),D=D,D_leading=nu*K*(r-1)**2/Q,net=net,leading_net=lead,net_direct=D-gain_direct.mean(),net_direct_se=float(gain_direct.std(ddof=1)/np.sqrt(N)),identity_gap=float(gap.mean()),identity_se=float(gap.std(ddof=1)/np.sqrt(N)),posterior_response_mse=float(((pred-truth)**2).mean()),conditional_variance=float(pvar.mean())))
        print('phase',nu,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'count_noise_phases.csv',index=False)
    (OUT/'quadrature_checks.json').write_text(json.dumps(quadchecks,indent=2))
    fig,axs=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for ax,Q in zip(axs,Qs[:3]):
        for nu in df.nu.unique():
            d=df[(df.Q==Q)&(df.nu==nu)]
            ax.plot(d.r,d.net/nu,'o-',ms=3,label=f'Prior mean variance {nu:g}')
        d=df[(df.Q==Q)&(df.nu==df.nu.min())]
        ax.plot(d.r,d.leading_net/d.nu,'k--',lw=1.3,label='Derived leading term')
        ax.axhline(0,color='grey',lw=.7);ax.set_title(f'Benefit / cue ratio = {Q:g}')
        ax.set_xlabel('Readings per location');ax.set_ylabel('Risk difference / prior mean variance')
        ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=6.6)
    fig.savefig(OUT/'count_noise_phases.pdf');fig.savefig(OUT/'count_noise_phases.png',dpi=150);plt.close(fig)
    return df

def finite_bank_prediction(mean,sse,r,values,log_location,prior_shape=2.,prior_scale=1.):

    N,m=mean.shape
    q=sse[:,None]+r*((mean[:,None,:]-values[None,:,:])**2).sum(axis=2)
    beta=prior_scale+.5*q;alpha=prior_shape+m*r/2
    log_response=-alpha*np.log(beta)
    def posterior(mult):
        logits=log_response+mult*log_location
        return np.exp(logits-logsumexp(logits,axis=1)[:,None])
    prow=posterior(r);pgroup=posterior(1)
    svar=beta/(alpha-1)
    return prow,pgroup,svar

def large_count_checks(N):
    m=4;kappa=.3;logL=np.array([0.,m*kappa]);rs=[1,2,4,8,16,32,64,128,512]

    x=np.array([.08,.27,.54,.83]);f0=np.sin(2*np.pi*x)+.2*np.cos(3*np.pi*x)
    vals=np.stack([f0,f0+1]);boundary=1/np.expm1(2*kappa)
    rows=[]
    for s in [.3,.8,1.1,1.4,2.,3.]:
        rng=np.random.default_rng(902802);truth=np.broadcast_to(f0,(N,m))
        ss=np.full(N,s);first=truth+np.sqrt(s)*rng.normal(size=(N,m))
        p1,_,_=finite_bank_prediction(first,np.zeros(N),1,vals,logL)
        base=p1[:,1]**2
        for r in rs:
            mean,w=sufficient(truth,ss,first,r,rng)
            pr,pg,sv=finite_bank_prediction(mean,w,r,vals,logL)
            dr=pr[:,1]**2-base
            rows.append(dict(s=s,m=m,r=r,N=N,noise_boundary=boundary,score_difference=m*kappa-m/2*np.log1p(1/s),posterior_alternative=float(pr[:,1].mean()),selected_alternative_fraction=float((pr[:,1]>.5).mean()),row_risk=float((pr[:,1]**2).mean()),grouped_risk=float((pg[:,1]**2).mean()),single_risk=float(base.mean()),net=float(dr.mean()),net_se=float(dr.std(ddof=1)/np.sqrt(N)),variance_if_alternative=float(sv[:,1].mean()),variance_limit=s+1))
    df=pd.DataFrame(rows);df.to_csv(OUT/'fixed_prior_large_count.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for s in df.s.unique():
        d=df[df.s==s];axs[0].plot(d.r,d.posterior_alternative,'o-',ms=3,label=f'Physical variance {s:g}')
        axs[1].plot(d.r,d.net,'o-',ms=3)
    for ax in axs:ax.set_xscale('log',base=2);ax.set_xlabel('Readings per location');ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Posterior probability of alternative');axs[0].legend(fontsize=7)
    axs[1].set_ylabel('Expanded minus Single risk');axs[1].axhline(0,color='grey',lw=.7)
    fig.suptitle(f'Fixed unknown-variance prior; derived boundary s = {boundary:.3f}',fontsize=11)
    fig.savefig(OUT/'fixed_prior_large_count.pdf');fig.savefig(OUT/'fixed_prior_large_count.png',dpi=150);plt.close(fig)
    return df

def coverage_checks():
    rows=[];t=.25;amp=1.;rs=[2,4,8,16];ms=[4,8,16,32];dims=[1,2,3,4];bins=3
    for d in dims:
        K=bins**d
        for m in ms:
            p0=(1-1/K)**m
            for r in rs:
                delta=cue_cost(t,r,beta=1.)
                for nu in [.01,.1,.5]:
                    gx,gw=leggauss(100);sv=nu*(1+.5*gx)
                    pe=float((gw/2*ndtr(-amp/np.sqrt(sv))).sum())
                    lower=amp**2*(p0*delta-4*(1-p0)*pe)
                    rows.append(dict(d=d,K=K,m=m,r=r,nu=nu,t=t,p_uncovered=p0,delta=delta,low_noise_limit=amp**2*p0*delta,finite_lower=lower,noise_error_bound=pe))
    df=pd.DataFrame(rows);df.to_csv(OUT/'coverage_certificates.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for d in dims:
        sub=df[(df.d==d)&(df.r==8)&(df.nu==.01)]
        axs[0].plot(sub.m,sub.low_noise_limit,'o-',label=f'Intrinsic dimension {d}')
    axs[0].set_xlabel('Distinct sampled locations');axs[0].set_ylabel('Exact low-noise excess risk');axs[0].legend(fontsize=8)
    for r in rs:
        sub=df[(df.m==16)&(df.r==r)&(df.nu==.1)]
        axs[1].plot(sub.d,sub.finite_lower,'o-',label=f'{r} readings per location')
    axs[1].set_xlabel('Intrinsic dimension (3 regions per coordinate)');axs[1].set_ylabel('Finite-noise lower bound on excess risk');axs[1].legend(fontsize=7.5)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    axs[1].axhline(0,color='grey',lw=.7)
    fig.savefig(OUT/'coverage_certificates.pdf');fig.savefig(OUT/'coverage_certificates.png',dpi=150);plt.close(fig)
    return df

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=16384);arg=ap.parse_args();OUT.mkdir(exist_ok=True)
    st=time.time();p=phase_checks(arg.n);l=large_count_checks(arg.n);c=coverage_checks()
    q=p[(p.nu==p.nu.min())&(p.r>1)&(abs(p.leading_net)>1e-12)]
    summary=dict(N=arg.n,phase_conditions=len(p),fixed_prior_conditions=len(l),coverage_certificates=len(c),smallest_variance_sign_matches=int((np.sign(q.net)==np.sign(q.leading_net)).sum()),smallest_variance_sign_tests=len(q),runtime_seconds=time.time()-st,predictor_uses_realized_variance=False,protocol='Phase model changes known prior scale but integrates each unobserved sampled S; fixed-prior model keeps the same IG(2,1) prior at every physical s. Coverage rows are analytic certificates, not additional Monte Carlo tests.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

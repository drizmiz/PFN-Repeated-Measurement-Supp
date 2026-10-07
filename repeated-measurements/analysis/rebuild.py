"""Recompute paper summaries from saved scores and prediction arrays."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize

MODELS = ['tabpfn3','tabpfn35','tabicl','tabdpt','limix2','tabfm']
NAMES = dict(zip(MODELS,['TabPFN-3','TabPFN-3.5','TabICLv2','TabDPT','LimiX-2','TabFM']))
NAMES.update(GP='GP',NN='MLP',RF='RF',GBM='GBM')
STYLE = dict(zip(MODELS+['GP','NN','RF','GBM'],
    [('#D55E00','o'),('#E69F00','v'),('#0072B2','^'),('#AA4499','s'),
     ('#009E73','D'),('#666666','P'),('#0072B2','o'),('#009E73','^'),('#666666','D'),('#AA4499','s')]))
COL = {'single':'#626262','expanded':'#D55E00','mean':'#AA4499','control':'#0072B2'}


def default(d, family):
    q = d.family.eq(family) & d.m.eq(12) & d.r_scheme.eq('equal')
    if family == 'synth1d': q &= d.spacing.eq('random')
    return q & np.isclose(d.sigma_or_sigmult, .15 if family == 'synth1d' else .25)


def paired(d, score='mse_pop', model='backend', cell='cell_id'):
    single = d[d.arm.eq('single')][[cell,model,score]].rename(columns={score:'single_mse'})
    assert not single.duplicated([cell,model]).any()
    q = d.merge(single,on=[cell,model],how='left',validate='many_to_one')
    if q.single_mse.isna().any() and 'family' in d:
        # Unequal-count cells reuse the Single fit of the identical site/reading bank.
        keys=['family','func_or_dataset','truth','d','m','spacing','sigma_or_sigmult','seed',model]
        keys=[k for k in keys if k in d]
        s=d[d.arm.eq('single')][keys+[score]].copy()
        assert (s.groupby(keys,dropna=False)[score].nunique()<=1).all()
        s=s.drop_duplicates(keys).rename(columns={score:'shared_single'})
        q=q.merge(s,on=keys,how='left',validate='many_to_one')
        q['single_mse']=q.single_mse.fillna(q.shared_single)
        q=q.drop(columns='shared_single')
    assert q.single_mse.notna().all()
    q['ratio_to_single'] = q[score]/q.single_mse
    return q


def rebuild(root, out):
    data = root/'data'; figs = out/'figures'; tables = out/'tables'
    figs.mkdir(parents=True,exist_ok=True); tables.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    read = lambda name: pd.read_csv(data/'results'/f'{name}.csv.gz',low_memory=False)
    write = lambda d,name: d.to_csv(tables/f'{name}.csv',index=False,float_format='%.17g')
    def save(fig,name):
        fig.savefig(figs/f'{name}.pdf',bbox_inches='tight',
            metadata={'CreationDate':None,'ModDate':None,'Author':None,'Title':name})
        plt.close(fig)

    main = read('main'); f1 = read('figure1'); real = read('real')
    pp = paired(main)
    write(pp.groupby(['axis','family','func_or_dataset','truth','d','m','spacing',
        'sigma_or_sigmult','r_scheme','backend','arm','r'],dropna=False).agg(
        n=('seed','size'),median_mse=('mse_pop','median'),median_paired_ratio=('ratio_to_single','median'),
        harmful=('ratio_to_single',lambda x: int((x>1).sum()))).reset_index(),'main_scan')

    ratios = f1.pivot(index=['learner','seed'],columns='r',values='mse_pop')
    normalized = ratios.div(ratios[1],axis=0)
    write(normalized.groupby(level='learner').median().reset_index(),'figure1_paired_ratios')
    write(ratios.groupby(level='learner').agg(['median','size']).reset_index(),'figure1_marginal_mse')
    harm = ((ratios[256]>ratios[1]).groupby(level='learner').sum()).rename('harmful_seeds').reset_index()
    write(harm,'figure1_harm_counts')
    def curve(ax,b):
        z = np.load(data/'predictions'/f'preds_quadratic_s0_{b}_sig0.1.npz')
        pop = z['region'].astype(str)=='pop'; x=z['Xq'].ravel()[pop]; order=np.argsort(x)
        ax.plot(x[order],z['fq'][pop][order],'--',color='.5',lw=1,label='Truth')
        for r,c,ls in [(1,'#0072B2','-'),(64,'#D55E00','-.')]:
            ax.plot(x[order],z[f'pred_r{r}'][pop][order],ls,color=c,lw=1.2,label=f'{r} reading'+('s' if r>1 else ''))
        ax.plot(z['X'].ravel(),z['ybar_r64'],'o',color='k',ms=2.5,label='Site means')
        ax.set_xlabel('$x$'); ax.grid(alpha=.15)
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.1))
    curve(axes[0],'tabpfn3');axes[0].set_title('TabPFN-3, 12 sites');axes[0].legend(fontsize=5.8,frameon=False)
    for ax,keys,title in [(axes[1],MODELS,'Six TFMs'),(axes[2],['GP','NN','GBM','RF'],'Conventional learners')]:
        for b in keys:
            v=normalized.loc[b].median();c,m=STYLE[b]
            ax.plot(v.index,v,color=c,marker=m,ms=3,lw=1.2,label=NAMES[b])
        ax.axhline(1,color='.4',ls=':',lw=.8);ax.set_xscale('log',base=2)
        ax.set_xticks([1,4,16,64,256],labels=[1,4,16,64,256]);ax.set_title(title+', 20 seeds')
        ax.set_xlabel('Readings per site');ax.legend(fontsize=5.8,frameon=False);ax.grid(alpha=.15)
    axes[1].set_ylabel('Median paired MSE ratio');axes[2].set_ylim(0,1.25)
    fig.tight_layout(pad=.4);save(fig,'fig1_failure')
    fig,axes=plt.subplots(1,6,figsize=(12.6,2.35),sharex=True,sharey=True)
    for ax,b in zip(axes,MODELS):curve(ax,b);ax.set_title(NAMES[b])
    axes[0].set_ylabel('Prediction');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.15,1,1));save(fig,'fig_fits_strip')

    marginal = pd.concat([main[default(main,f)] for f in ['synth1d','semisynth']])
    three = marginal.groupby(['family','backend','arm','r']).agg(n=('seed','size'),median_mse=('mse_pop','median')).reset_index()
    write(three,'three_tables')
    fig,axes=plt.subplots(4,3,figsize=(6.75,7.2),sharex=True)
    for group,f in enumerate(['synth1d','semisynth']):
        for i,b in enumerate(MODELS):
            ax=axes[2*group+i//3,i%3];q=three[three.family.eq(f)&three.backend.eq(b)]
            s=q.loc[q.arm.eq('single'),'median_mse'].item();ax.axhline(s,color=COL['single'],ls='--',lw=1,label='Single')
            for arm,label,m,ls in [('expanded','Expanded','o','-'),('mean','Mean','^','-.')]:
                v=q[q.arm.eq(arm)].sort_values('r')
                ax.plot(np.r_[1,v.r],np.r_[s,v.median_mse],color=COL[arm],marker=m,ls=ls,ms=2.5,lw=1,label=label)
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_title(NAMES[b])
            ax.set_xticks([1,4,16,64,256],labels=[1,4,16,64,256])
            if i%3==0: ax.set_ylabel(('One-dimensional' if group==0 else 'Real covariates')+'\nMedian MSE')
            if i//3==1: ax.set_xlabel('Readings per site')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.96));save(fig,'fig_three_tables_review')

    rd = real[~real.dataset.eq('superconductivity_r16')]
    rd = paired(rd,score='mse',model='learner',cell='dataset_seed') if 'dataset_seed' in rd else rd.copy()
    rd['dataset_seed']=rd.dataset+'|'+rd.seed.astype(str)
    rd=paired(rd,score='mse',model='learner',cell='dataset_seed')
    write(rd.groupby(['dataset','learner','arm','r']).agg(n=('seed','size'),median_mse=('mse','median'),
        median_paired_ratio=('ratio_to_single','median'),harmful=('ratio_to_single',lambda x:int((x>1).sum()))).reset_index(),'real_models')
    # Each panel uses its own manuscript cohort; zero-noise Mean equals Single.
    panels=[('synth1d','noise','sigma_or_sigmult','Noise SD'),('synth1d','m','m','Sites'),
            ('semisynth','noise','sigma_or_sigmult','Noise multiplier'),('semisynth','m','m','Sites'),
            ('synthd','dim','d','Irrelevant dimensions')]
    heat=[];titles=[];labels=[]
    for family,axis,col,title in panels:
        axis_mask=pp.axis.str.contains(r'(?:^|;)'+axis+r'(?:;|$)',regex=True,na=False)
        q=pp[pp.family.eq(family)&axis_mask&pp.arm.eq('expanded')&pp.r.eq(64)&pp.backend.isin(MODELS)]
        if family=='synthd': q=q[q.func_or_dataset.eq('irrelevant')]
        if axis=='noise': q=q[q.m.eq(12)&q.r_scheme.eq('equal')]
        if axis=='m': q=q[q.r_scheme.eq('equal')&np.isclose(q.sigma_or_sigmult,.15 if family=='synth1d' else .25)]
        if family=='synth1d': q=q[q.spacing.eq('random')]
        h=q.groupby([col,'backend']).ratio_to_single.median().unstack().reindex(columns=MODELS)
        heat.append(h);titles.append(('1D: ' if family=='synth1d' else 'Real covariates: ' if family=='semisynth' else '')+title);labels.append(col)
    desc=pd.read_csv(root/'config/real_datasets.csv')
    names=desc.sort_values('within_sd').dataset.tolist()
    endpoint=rd[rd.arm.eq('expanded')&rd.r.eq(rd.R_max)&rd.learner.isin(MODELS)]
    heat.append(endpoint.groupby(['dataset','learner']).ratio_to_single.median().unstack().reindex(index=names,columns=MODELS));titles.append('Real repeated observations');labels.append('dataset')
    fig,axes=plt.subplots(2,3,figsize=(8.2,5.1))
    for ax,h,title,col in zip(axes.ravel(),heat,titles,labels):
        im=ax.imshow(np.log2(h.values),aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(0,-3,3))
        ax.set_xticks(range(6),labels=[NAMES[b] for b in MODELS],rotation=35,ha='right',fontsize=7)
        ax.set_yticks(range(len(h)),labels=[f'{x:g}' if isinstance(x,(int,float)) else str(x) for x in h.index],fontsize=7)
        ax.set_title(title);write(h.reset_index(),'overview_'+str(len(list(tables.glob('overview_*')))))
    fig.subplots_adjust(hspace=.65,wspace=.48,bottom=.17,right=.9)
    cax=fig.add_axes([.93,.2,.015,.6]);fig.colorbar(im,cax=cax,label='log2 median paired MSE ratio')
    save(fig,'fig_overview')

    site=main[default(main,'synth1d')&main.backend.isin(MODELS)&main.arm.eq('expanded')]
    fig,axes=plt.subplots(1,6,figsize=(11,2.1),sharex=True)
    for ax,b in zip(axes,MODELS):
        for col,label,c,m in [('mse_pop','Fresh inputs','#0072B2','o'),('mse_site','Training sites','#D55E00','s')]:
            v=site[site.backend.eq(b)].groupby('r')[col].median()
            baseline=main[default(main,'synth1d')&main.backend.eq(b)&main.arm.eq('single')][col].median()
            ax.plot(np.r_[1,v.index],np.r_[baseline,v],color=c,marker=m,ms=2,lw=1,label=label)
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_title(NAMES[b]);ax.set_xlabel('Readings per site')
    axes[0].set_ylabel('Median MSE');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.14,1,1));save(fig,'fig_site_offsite')

    a=read('attention_example');meta=json.loads((root/'config/attention_example.json').read_text())
    pred=a.pivot(index='x',columns='condition',values='prediction').sort_index()
    neff=a.pivot(index='x',columns='condition',values='neff').sort_index()
    truth=a.pivot(index='x',columns='condition',values='truth').sort_index()
    assert pred.shape==(meta['queries'],3)
    fig,axes=plt.subplots(2,1,figsize=(3.35,2.72),sharex=True)
    for arm,label,c,ls in [('all','Expanded',COL['expanded'],'-'),('ovr','Count control',COL['control'],'--')]:
        axes[0].plot(pred.index,pred[arm]-truth[arm],color=c,ls=ls,lw=1.25,label=label)
        axes[1].plot(neff.index,neff[arm]-neff['one'],color=c,ls=ls,lw=1.25)
    for ax in axes:ax.axhline(0,color='.4',ls=':',lw=.7);ax.grid(axis='y',alpha=.15)
    axes[0].plot(meta['site_positions'],np.zeros(meta['sites']),'|',color='.5',transform=axes[0].get_xaxis_transform())
    axes[0].set_ylabel('Prediction error');axes[1].set_ylabel('Effective sites minus Single');axes[1].set_xlabel('Query input')
    axes[1].set_ylim(-1.75,.55);axes[0].legend(frameon=False,ncol=2,fontsize=7)
    fig.tight_layout();save(fig,'fig_count_control_review')

    # The query is a quarter of the widest gap, matching the recorded weight-map design.
    zs=[np.load(data/'predictions'/f'weights_{arm}.npz') for arm in ['one','all']]
    X=zs[0]['X'].ravel();j=np.argmax(np.diff(X));star=X[j]+.25*(X[j+1]-X[j])
    qi=int(np.argmin(abs(zs[0]['Xq'].ravel()-star)))
    fig,axes=plt.subplots(1,2,figsize=(8.2,3),sharey=True)
    for ax,z,label in zip(axes,zs,['Single','Expanded (64 readings)']):
        w=z['W'][qi];x=z['Xq'].ravel();o=np.argsort(x)
        ax.plot(x[o],z['fq'][o],'--',color='.4',label='Truth');ax.plot(x[o],z['pred'][o],color='#0072B2',lw=1,label='Prediction')
        ax.axvspan(X[j],X[j+1],color='.93');ax.axvline(x[qi],color='#D55E00',ls=':',lw=1)
        ax.scatter(X,z['ybar'],c=w,cmap='viridis',vmin=0,vmax=.55,s=35,zorder=4)
        ax.plot(x[qi],z['fq'][qi],'*',color='#D55E00',ms=8)
        for xx,yy,ww in zip(X,z['ybar'],w):ax.annotate(f'{ww:.2f}',(xx,yy),xytext=(0,5),textcoords='offset points',ha='center',fontsize=6)
        ax.set_title(f'{label}\nEffective sites {1/np.sum(w*w):.2f}; largest weight {w.max():.2f}');ax.set_xlabel('$x$')
    axes[0].set_ylabel('Response');axes[0].legend(fontsize=7,frameon=False)
    fig.tight_layout();save(fig,'fig_weightmap')

    ar=read('attention_r');fig,axes=plt.subplots(2,2,figsize=(6.8,4.4),sharex=True)
    summ=[]
    for ax,f in zip(axes.ravel(),['quadratic','step','sine','tanh']):
        q=ar[ar.func.eq(f)]; ax2=ax.twinx()
        for col,label,c,target,m in [('neff_sel','Effective sites','#0072B2',ax,'o'),('top1_sel','Largest site weight','#D55E00',ax2,'s')]:
            g=q.groupby('r')[col];v=g.mean();se=g.std(ddof=1)/np.sqrt(g.size())
            target.errorbar(v.index,v,yerr=1.96*se,color=c,marker=m,ms=2.5,lw=1,capsize=1,label=label)
            for r in v.index:summ.append(dict(func=f,r=r,statistic=col,n=int(g.size().loc[r]),mean=v.loc[r],se=se.loc[r]))
        ax.set_xscale('log',base=2);ax.set_title(f);ax.set_xlabel('Readings per site');ax.set_ylabel('Effective sites',color='#0072B2');ax2.set_ylabel('Largest weight',color='#D55E00')
    fig.tight_layout();save(fig,'fig_attn_vs_r_20seeds');write(pd.DataFrame(summ),'attention_r')

    stage=read('embedding_stages');risk=read('embedding_risk')
    order=['xonly','input','embedder','icl_in']+[f'icl_{i}' for i in [0,1,2,3,4,5,6,7,23]]
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.4));xs=np.arange(len(order))
    for fp,sg,c,label,ls in [(0,.05,'#0072B2','Response noise .05','-'),(0,.3,'#0072B2','Response noise .30','--'),(1,0,'#D55E00','Fingerprint, no noise',':'),(1,.05,'#D55E00','Fingerprint and noise .05','-.')]:
        q=stage[stage.fp.eq(fp)&np.isclose(stage.sigma,sg)&stage.override.eq('none')]
        axes[0].plot(xs,q.groupby('stage').within_rel.median().reindex(order),color=c,ls=ls,marker='o',ms=2,label=label)
    for ov,c,label,ls in [('none','#626262','Shipped','-'),('featdist','#D55E00','Embedder count','--'),('count','#0072B2','All count','-.')]:
        q=stage[stage.fp.eq(0)&stage.sigma.eq(0)&stage.override.eq(ov)]
        axes[1].plot(xs,q.groupby('stage').shift_specific_rel.median().reindex(order),color=c,ls=ls,marker='o',ms=2,label=label)
    q=risk[risk.fp.eq(1)&np.isclose(risk.sigma,.05)].pivot(index=['func','seed'],columns='arm',values='mse_pop')
    fs=['quadratic','tanh','step','sine']
    for k,(arm,c,label,hatch) in enumerate([('expanded_none','#626262','Shipped',''),('expanded_featdist','#D55E00','Embedder count','//'),('expanded_count','#0072B2','All count','..')]):
        axes[2].bar(np.arange(4)+(k-1)*.25,[np.median(q.loc[f,arm]/q.loc[f,'single']) for f in fs],.25,color=c,label=label,hatch=hatch)
    axes[0].set_ylabel('Within-site / between-site spread');axes[1].set_ylabel('Site-specific shift / between-site spread')
    axes[0].set_title('Within-site scatter');axes[1].set_title('Same reading in repeated context');axes[2].set_title('Paired MSE ratio, noise .05')
    for ax in axes[:2]:ax.set_xticks(xs,labels=order,rotation=60,fontsize=7)
    axes[2].axhline(1,color='.5',lw=.8);axes[2].set_yscale('log');axes[2].set_xticks(range(4),labels=fs);axes[2].set_ylabel('MSE / Single MSE')
    for ax in axes:ax.legend(fontsize=6,frameon=False)
    fig.tight_layout();save(fig,'fig_embedding_stage')

    z=np.load(data/'predictions/surface.npz');sm=read('surface');xx,yy=np.meshgrid(z['axis'],z['axis'])
    arms=[('truth','True function'),('single','Single'),('expanded64','Expanded, 64 readings'),('mean64','Mean, 64 readings')]
    values=[z['truth_grid']]+[z[f'pred_{arm}_grid'] for arm,_ in arms[1:]]
    limit=np.ceil(max(np.max(abs(v)) for v in values)*2)/2
    fig=plt.figure(figsize=(7.4,5.55))
    for j,((arm,label),v) in enumerate(zip(arms,values)):
        ax=fig.add_subplot(2,2,j+1,projection='3d')
        ax.plot_surface(xx,yy,v,cmap='RdBu_r',norm=Normalize(-limit,limit),rcount=61,ccount=61,linewidth=0,antialiased=True,shade=False)
        ax.scatter(z['X'][:,0],z['X'][:,1],np.full(24,-limit),s=6,c='#414954',depthshade=False)
        ax.view_init(30,-125);ax.set(xlim=(-2,2),ylim=(-2,2),zlim=(-limit,limit),xticks=[-2,0,2],yticks=[-2,0,2],zticks=[-limit,0,limit])
        ax.set_box_aspect((1,1,.7));ax.set_xlabel('$x_1$',labelpad=-5);ax.set_ylabel('$x_2$',labelpad=-5);ax.set_title(label,fontsize=9,pad=2)
        if arm!='truth':
            row=sm[sm.arm.eq(arm)].iloc[0]
            ax.text2D(.5,-.08,f'Fresh MSE {row.fresh_mse:.3f} | near-site response {row.near_S:.2f}',transform=ax.transAxes,ha='center',fontsize=8)
    fig.subplots_adjust(left=.03,right=.96,bottom=.065,top=.95,hspace=.26,wspace=.03);save(fig,'spatial_readings_m24_main')

    tc=read('theory_count');fig,ax=plt.subplots(figsize=(3.55,2.45))
    for (nu,q),(c,m) in zip(tc.groupby('nu'),[('#0072B2','o'),('#D55E00','s'),('#AA4499','^'),('#626262','D')]):
        q=q.sort_values('r');ax.errorbar(q.r,q.net_fixed_t,yerr=1.96*q.B_se,color=c,marker=m,ms=3,lw=1,capsize=1.5,label=f'nu={nu:g}')
        ax.plot(q.r,q.leading_fixed_t,'--',color=c,lw=1)
    ax.axhline(0,color='.5',lw=.7);ax.set_xscale('log',base=2);ax.set_xlabel('Readings per site');ax.set_ylabel('Risk difference from Single');ax.legend(frameon=False,fontsize=7);fig.tight_layout();save(fig,'theory_count')
    tcov=read('theory_coverage');fig,ax=plt.subplots(figsize=(3.55,2.45));palette={2:'#0072B2',4:'#D55E00',8:'#AA4499'}
    for (K,m),q in tcov.groupby(['K','m']):
        ax.errorbar(q.low_noise_limit,q.direct_net,yerr=1.96*q.direct_se,fmt='o' if m==2 else '^',color=palette[K],ms=4,capsize=2,label=f'K={K}, m={m}')
    upper=max(tcov.low_noise_limit.max(),(tcov.direct_net+1.96*tcov.direct_se).max())*1.07
    ax.plot([0,upper],[0,upper],'--',color='.5',lw=1);ax.set(xlim=(0,upper),ylim=(0,upper),xlabel='Predicted low-noise risk difference',ylabel='Direct posterior risk difference')
    ax.legend(frameon=False,fontsize=6);fig.tight_layout();save(fig,'theory_coverage')

    # Finite-bank bias correction and unbiased across-bank variance.
    bankrows=[];index=pd.read_csv(data/'predictions/banks_index.csv')
    with np.load(data/'predictions/banks.npz') as z:
        for row in index.itertuples(index=False):
            pred=z[f'p{row.index}'];f=z[f'f{row.index}'];region=z[f'region{row.index}']
            for region_name in ['pop','site']:
                mask=region==region_name
                for ai,arm in enumerate(['single','expanded','mean']):
                    v=pred[ai][:,mask]
                    variance=v.var(axis=0,ddof=1).mean();bias2=np.mean((v.mean(axis=0)-f[mask])**2)-variance/8
                    total=np.mean((v-f[mask])**2)
                    assert np.isclose(total,bias2+variance,rtol=1e-10,atol=1e-12)
                    bankrows.append(dict(backend=row.backend,family=row.family,func=row.func,sig=row.sig,seed=row.seed,region=region_name,arm=arm,bias2=bias2,variance=variance,total=total))
    banks=pd.DataFrame(bankrows);write(banks,'banks_per_design')
    write(banks.groupby(['backend','family','sig','region','arm']).agg(n=('seed','size'),bias2=('bias2','median'),variance=('variance','median'),total=('total','median')).reset_index(),'banks_summary')
    for name in ['four_layers','all_layers','adapter']:
        q=read('training_'+name);v=q.pivot(index=['func','seed'],columns='arm',values='mse_pop');v.columns=[x.replace('_ad_V','_ft') for x in v.columns]
        assert len(v)==36
        write(pd.DataFrame([dict(trial=name,n=36,expanded_sh=v.expanded_sh.median(),expanded_ft=v.expanded_ft.median(),
            mean_sh=v.mean_sh.median(),mean_ft=v.mean_ft.median(),expanded_wins=int((v.expanded_ft<v.expanded_sh).sum()))]),'training_'+name)
    lin=read('linear');write(lin.groupby(['scope','r']).mean(numeric_only=True).reset_index(),'linear_count_scopes')
    e1=read('e1');e1['cohort']=np.where(e1.rconf.isin(['r4','r64','equal4','equal64']),'equal','unequal')
    write(e1.groupby(['learner','rconf']).agg(n=('seed','size'),weighted_matches=('maxdiff_wmean',lambda x:int((x<=1e-6).sum())),
        unweighted_matches=('maxdiff_umean',lambda x:int((x<=1e-6).sum())),weighted_site_matches=('site_maxdiff_wmean',lambda x:int((x<=1e-6).sum()))).reset_index(),'invariance')
    e2=paired(read('e2'));write(e2.groupby(['family','d','m','sigma_or_sigmult','backend','arm','r']).agg(n=('seed','size'),
        median_mse=('mse_pop','median'),median_paired_ratio=('ratio_to_single','median')).reset_index(),'count_control')
    e6=read('e6');write(e6.groupby(['sig','arm']).agg(n=('seed','size'),median_mse=('mse_pop','median'),median_neff=('neff_pop_med','median')).reset_index(),'effective_count')
    gp=read('gp');gp['rows_single_ratio']=gp.A/gp.S;gp['precision_single_ratio']=gp.C/gp.S;gp['relative_gap']=abs(gp.A-gp.C)/np.maximum(abs(gp.A),1e-15)
    write(gp.groupby(['axis','family','r']).agg(n=('seed','size'),rows_ratio=('rows_single_ratio','median'),precision_ratio=('precision_single_ratio','median'),relative_gap=('relative_gap','median')).reset_index(),'gp_comparison')
    e4=read('e4');write(e4.groupby(['part','family','backend','r_scheme','r','arm']).median(numeric_only=True).reset_index(),'uncertainty')
    report={'model_calls':0,'figures':sorted(p.name for p in figs.glob('*.pdf')),
        'main_unique_fits':int(len(main)),'bank_designs':len(index),'bank_identity_checked':True,
        'training_cells_per_trial':36,'aggregation':'Paired ratios and marginal MSE medians are computed separately.'}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Rebuilt {len(report["figures"])} figures and saved summaries to {out}. No model calls.')

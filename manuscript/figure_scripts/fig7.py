"""Fig. 7: (a) QMC-anchored m(h) vs search depth; (b) presence probabilities (N<=24, 4000 resamples);
(c) tie-line geometry in the (c,m) plane: plateau value = intersection of the tie line with c = c_AB."""
import json, numpy as np
from style import *
S='/mnt/user-data/outputs/sectors/'
def curve(pl):
    h=[];m=[]
    for mm,a,b in pl: h+=[a,b]; m+=[mm,mm]
    return h,m
sets=[('old','restricted search','0.55',(0,(3,1.5))),('N16',r'all sectors, $N\leq16$','#9ecae1','-'),
      ('N21',r'all sectors, $N\leq21$','#3182bd','-'),('N24',r'all sectors, $N\leq24$','#08306b','-')]
fig,axs=plt.subplots(1,3,figsize=(DCOL,2.55),gridspec_kw=dict(wspace=0.36,width_ratios=[1.15,1,1]))
a=axs[0]
for k,lab,c,ls in sets:
    d=json.load(open(S+f'resample_{k}.json')); h,m=curve(d['central']); a.plot(h,m,color=c,ls=ls,lw=1.2 if ls=='-' else 1.0,label=lab)
a.axhline(0.2,color='#D55E00',lw=0.7,ls=':'); a.axhline(1-1/np.sqrt(2),color='0.6',lw=0.6,ls=':')
a.axvspan(0.09,0.18,color='#f2f2f2',zorder=0); a.text(0.135,0.004,'window',ha='center',fontsize=7,color='0.4')
a.text(0.298,0.192,r'$1/5$',color='#D55E00',fontsize=7.5,ha='right',va='top'); a.text(0.298,0.285,r'$m_{\max}$',color='0.4',fontsize=7.5,ha='right',va='top')
a.set_xlim(0,0.30); a.set_ylim(-0.005,0.31); a.set_xlabel(r'$h/J$'); a.set_ylabel(r'$m$')
a.legend(loc='upper left',bbox_to_anchor=(0.0,0.96),frameon=False,fontsize=6.6,handlelength=1.6,borderaxespad=0.2)
b=axs[1]; d=json.load(open(S+'resample_N24.json'))
for s in d['presence']:
    if s['m']<=0 or s['frac']<0.05: continue
    c='#08306b' if s['frac']>=0.95 else ('#9ecae1' if s['frac']<0.5 else '#3182bd')
    b.errorbar(s['onset'],s['frac'],xerr=[[s['onset']-s['onset16']],[s['onset84']-s['onset']]],fmt='o',ms=3,color=c,elinewidth=0.7,capsize=1.2)
    if s['frac']>=0.95:
        off={0.033:(-12,-12),0.083:(4,-12),0.2:(-8,-12),0.293:(-22,6)}.get(round(s['m'],3),(3,-10))
        b.annotate('%.3f'%s['m'],(s['onset'],s['frac']),xytext=off,textcoords='offset points',fontsize=7)
b.axvspan(0.09,0.18,color='#f2f2f2',zorder=0)
b.axhline(0.95,color='0.6',lw=0.6,ls=':'); b.set_xlim(0,0.30); b.set_ylim(0,1.08)
b.set_xlabel(r'onset field $h/J$ (median)'); b.set_ylabel('presence probability')
c_=axs[2]; cAB=np.sqrt(2)-1
c_.axvline(cAB,color='k',lw=0.6,ls=':'); c_.text(cAB+0.006,0.006,r'$c_{\rm AB}$',fontsize=7.5)
c_.axhline(0.2,color='#D55E00',lw=0.7,ls=':')
for k,lab,col,ls in [sets[1],sets[3]]:
    P=json.load(open(S+f'path_{k}.json')); info=P['info']; rows=P['rows']
    segs=[]; cur=None; h0=None
    for h,w in rows:
        key=tuple(sorted(w))
        if key!=cur:
            if cur is not None: segs.append((cur,h0,h))
            cur=key; h0=h
    segs.append((cur,h0,rows[-1][0]))
    for key,ha,hb in segs:
        if hb-ha<0.004 or ha<0.09 or ha>0.26 or len(key)!=2: continue
        pts=np.array([(info[q]['sigma']/(1+info[q]['sigma']),info[q]['tau']/(2*(1+info[q]['sigma']))) for q in key])
        c_.plot(pts[:,0],pts[:,1],'-',color=col,lw=1.0); c_.plot(pts[:,0],pts[:,1],'o',color=col,ms=3)
        x=pts[:,0]; y=pts[:,1]; mi=y[0]+(y[1]-y[0])*(cAB-x[0])/(x[1]-x[0]); c_.plot(cAB,mi,'x',color=col,ms=5,mew=1.2)
c_.plot([],[],'-o',color=sets[1][2],ms=3,label=sets[1][1]); c_.plot([],[],'-o',color=sets[3][2],ms=3,label=sets[3][1])
c_.set_xlim(0.15,0.65); c_.set_ylim(0.0,0.23); c_.set_xlabel(r'square fraction $c$'); c_.set_ylabel(r'$m$')
for ax_,l in zip(axs,'abc'): ax_.text(-0.2,1.02,f'({l})',transform=ax_.transAxes,fontsize=10,fontweight='bold',va='bottom')
plt.savefig('figs/fig7.pdf'); plt.savefig('figs/fig7.png',dpi=200)

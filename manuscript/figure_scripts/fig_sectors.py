"""(a) QMC-anchored ground-state magnetization m(h): earlier phase set vs all-sector search N<=16, 21, 24.
(b) presence probability of each plateau (N<=24, 4000 resamples) vs its median onset field; 16-84% onset bars."""
import json, numpy as np
from style import plt, COL, label
def curve(pl):
    h=[];m=[]
    for mm,a,b in pl: h+= [a,b]; m+=[mm,mm]
    return h,m
fig,ax=plt.subplots(2,1,figsize=(COL,4.6),gridspec_kw=dict(height_ratios=[1.25,1],hspace=0.32))
a=ax[0]
sets=[('resample_old.json','earlier phase set','0.55',(0,(3,1.5))),('resample_N16.json',r'all sectors $N\leq16$','#8ecae6','-'),
      ('resample_N21.json',r'all sectors $N\leq21$','#2a7ab8','-'),('resample_N24.json',r'all sectors $N\leq24$','#1d3557','-')]
for f,lab,c,ls in sets:
    d=json.load(open(f)); h,m=curve(d['central']); a.plot(h,m,color=c,ls=ls,lw=1.2 if ls=='-' else 1.0,label=lab)
a.axhline(0.2,color='#e76f51',lw=0.7,ls=':'); a.text(0.298,0.193,r'hub-free bound $m=1/5$',color='#e76f51',fontsize=7.5,va='top',ha='right')
a.axhline(1-1/np.sqrt(2),color='0.6',lw=0.6,ls=':')
a.set_xlim(0,0.30); a.set_ylim(-0.005,0.31); a.set_xlabel(r'$h/J$'); a.set_ylabel(r'$m$')
a.legend(loc='upper left',frameon=False,bbox_to_anchor=(0.0,0.93)); label(a,'(a)',-0.16,1.0)
b=ax[1]; d=json.load(open('resample_N24.json'))
for s in d['presence']:
    if s['m']<=0 or s['frac']<0.05: continue
    c='#1d3557' if s['frac']>=0.95 else ('#8ecae6' if s['frac']<0.5 else '#2a7ab8')
    b.errorbar(s['onset'],s['frac'],xerr=[[s['onset']-s['onset16']],[s['onset84']-s['onset']]],fmt='o',ms=3.2,color=c,elinewidth=0.8,capsize=1.5)
    if s['frac']>=0.6: b.annotate('%.3f'%s['m'],(s['onset'],s['frac']),xytext=(-28,-11) if s['m']>0.25 else ((4,-10) if s['m']!=0.082 else (4,-12)),textcoords='offset points',fontsize=7)
b.axhline(0.95,color='0.6',lw=0.6,ls=':'); b.set_xlim(0,0.30); b.set_ylim(0,1.08)
b.set_xlabel(r'median onset field $h/J$'); b.set_ylabel('presence probability'); label(b,'(b)',-0.16,1.0)
fig.savefig('/mnt/user-data/outputs/fieldfig/fig_sectors.pdf'); fig.savefig('/mnt/user-data/outputs/fieldfig/fig_sectors.png',dpi=200)

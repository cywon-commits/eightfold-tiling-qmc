import numpy as np, json
from style import *
C=json.load(open('/mnt/user-data/outputs/dense/candidates.json')); new=json.load(open('/mnt/user-data/outputs/dense/lp_big.json'))['new']
pts={}
for c in C: pts[(round(c['sigma'],3),round(c['tau'],3))]=min(pts.get((round(c['sigma'],3),round(c['tau'],3)),9),c['eps6'])
for n,s,t,e in new: k=(round(s,3),round(t,3)); pts[k]=min(pts.get(k,9),e)
for k,v in {(1.5,1.0):0.02646,(0.5,1.0):0.03719,(2.5,0.333):0.00614,(3.5,0.2):0.00351}.items(): pts.setdefault(k,v)
fig,ax=plt.subplots(figsize=(COL,2.7))
cc=np.linspace(0,1,100)
ax.plot(cc,(1-cc)/2,'-',color='#0072B2',lw=1.2); ax.plot(cc,cc/3,'-',color='#009E73',lw=1.2)
ax.axhline(0.2,color='#D55E00',lw=1.2); ax.axhline(1/3,color='0.55',lw=0.9,ls='--')
ax.text(0.07,0.415,r'$m=(1-c)/2$',color='#0072B2',fontsize=8,rotation=-40)
ax.text(0.78,0.272,r'$m=c/3$',color='#009E73',fontsize=8,rotation=29)
ax.text(0.86,0.205,r'$1/5$',color='#D55E00',fontsize=8,va='bottom'); ax.text(0.86,0.338,r'$1/3$',color='0.45',fontsize=8,va='bottom')
X=[];Y=[];Cv=[]
for (s,t),e in pts.items():
    if t<=0: continue
    X.append(s/(1+s)); Y.append(t/(2*(1+s))); Cv.append(e/t)
sc=ax.scatter(X,Y,c=Cv,cmap='viridis',vmin=0.017,vmax=0.038,s=16,ec='k',lw=0.35,zorder=3,label='restricted search')
SV=json.load(open('/mnt/user-data/outputs/sectors/screen_N24.json'))['survivors']
X2=[x['sigma']/(1+x['sigma']) for x in SV if x['tau']>0]; Y2=[x['tau']/(2*(1+x['sigma'])) for x in SV if x['tau']>0]; C2=[x['eps_big']/x['tau'] for x in SV if x['tau']>0]
ax.scatter(X2,Y2,c=C2,cmap='viridis',vmin=0.017,vmax=0.038,s=17,marker='D',ec='k',lw=0.35,zorder=4,label=r'all sectors, $N\leq24$')
ax.legend(loc='upper right',frameon=False,fontsize=7,handletextpad=0.1,borderaxespad=0.2)
cb=plt.colorbar(sc,ax=ax,pad=0.02,fraction=0.05); cb.set_label(r'$\varepsilon/\tau$ ($J$, LSWT)'); cb.ax.tick_params(labelsize=7.5)
c0=np.sqrt(2)-1; ax.axvline(c0,color='k',lw=0.6,ls=':')
ax.text(c0+0.012,0.475,'AB',fontsize=8)
ax.annotate('$z=5$ crystal',xy=(0.6,0.2),xytext=(0.80,0.045),fontsize=7.5,arrowprops=dict(arrowstyle='-',lw=0.6))
ax.annotate('dice',xy=(1/3,1/3),xytext=(0.40,0.40),fontsize=7.5,arrowprops=dict(arrowstyle='-',lw=0.6))
ax.set_xlim(0,1); ax.set_ylim(0,0.5); ax.set_xlabel(r'square fraction $c=S/N$'); ax.set_ylabel(r'$m=(N_A-N_B)/N$')
plt.savefig('figs/fig4.pdf'); plt.savefig('figs/fig4.png',dpi=220)

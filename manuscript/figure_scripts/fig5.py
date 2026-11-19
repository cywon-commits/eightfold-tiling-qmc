import numpy as np, json
from style import *
from matplotlib.patches import Polygon
v2=np.array([1.,0]); v0=np.array([0.,1]); v3=np.array([1,1])/np.sqrt(2)
fig=plt.figure(figsize=(DCOL,2.5)); gs=fig.add_gridspec(1,2,width_ratios=[2.0,1.0],wspace=0.18)
ax=fig.add_subplot(gs[0])
runs=[3,3,1,1,1,1,2,2,3]; word=''.join(('H' if i%2==0 else 'V')*r for i,r in enumerate(runs))
x,y=0,8; path=[(x,y)]
for c in word:
    x,y=(x+1,y) if c=='H' else (x,y-1); path.append((x,y))
Hrow={}
for (a,b),c in zip(path[:-1],word):
    if c=='H': Hrow[a]=b
X0,X1=-1,x+1; Y0,Y1=y-1,9
def below(fx,fy):
    yy=Hrow.get(fx,8 if fx<0 else y); return fy+1<=yy
def pos(i,j,low): return i*v2+j*v0+(v3 if low else 0)
for fx in range(X0,X1):
    for fy in range(Y0,Y1):
        lo=below(fx,fy); P=[pos(fx,fy,lo),pos(fx+1,fy,lo),pos(fx+1,fy+1,lo),pos(fx,fy+1,lo)]
        ax.add_patch(Polygon(P,fc=TILE['sq'],ec='0.65',lw=0.4))
tau=[(1 if word[k]=='H' else -1)*(-1)**k for k in range(len(word))]
for k,c in enumerate(word):
    p,q=path[k],path[k+1]; P=[pos(*p,0),pos(*q,0),pos(*q,1),pos(*p,1)]
    ax.add_patch(Polygon(P,fc='#cfe3f3' if c=='V' else '#d8ecd1',ec='0.35',lw=0.6))
    cc=(pos(*p,0)+pos(*q,1))/2
    ax.text(*cc,'+' if tau[k]>0 else '−',ha='center',va='center',fontsize=8.5,color='#b2182b' if tau[k]>0 else '#2166ac',fontweight='bold',zorder=5)
for k in range(1,len(word)):
    if word[k]!=word[k-1]:
        p=path[k]; up5=word[k-1]=='H'; par_up=(p[0]+p[1])%2
        for low,zz in ((0,5 if up5 else 3),(1,3 if up5 else 5)):
            par=(par_up+low)%2
            ax.scatter(*pos(*p,low),s=34 if zz==5 else 26,c=ZC[zz],marker='o' if par==0 else 's',ec='k',lw=0.5,zorder=4)
ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(X0-0.3,X1+0.9); ax.set_ylim(Y0+0.2,Y1+0.4)
ax.text(X1+0.85,Y1+0.2,'runs (3,3,1,1,1,1,2,2,3)\n'+r'$\tau_k=(-1)^k\sigma_k$',ha='right',va='top',fontsize=8.5)
ax.scatter([],[],s=26,c=ZC[3],ec='k',lw=0.5,label='$z=3$'); ax.scatter([],[],s=34,c=ZC[5],ec='k',lw=0.5,label='$z=5$')
ax.scatter([],[],s=26,c='w',marker='o',ec='k',lw=0.5,label='sublattice A'); ax.scatter([],[],s=26,c='w',marker='s',ec='k',lw=0.5,label='sublattice B')
ax.legend(loc='lower left',bbox_to_anchor=(0.0,0.0),ncol=2,frameon=False,handletextpad=0.2,columnspacing=0.8)
ax=fig.add_subplot(gs[1]); T=json.load(open('/home/claude/rib/turns.json'))
ev=[(int(k),np.mean([x[2] for x in v])) for k,v in T.items() if int(k)%2==0]
pol=[(1,0.02646),(3,0.0184),(3,0.0178),(5,0.0176),(5,0.0177)]
ax.plot([e[0] for e in ev],[e[1] for e in ev],'s',color='#0072B2',ms=4.5,label=r'neutral ($k$ even)')
ax.plot([p[0] for p in pol],[p[1] for p in pol],'o',color='#D55E00',ms=4.5,label=r'polarized arrays ($k$ odd)')
ax.axhline(0.0177,color='k',lw=0.6,ls=':'); ax.text(8.6,0.01795,r'$\varepsilon_t$',fontsize=8.5,va='bottom')
ax.annotate(r'$\varepsilon_t+J_1$',xy=(1,0.02646),xytext=(2.0,0.0247),fontsize=8.5,arrowprops=dict(arrowstyle='-',lw=0.6))
ax.set_xlabel('turn spacing $k$'); ax.set_ylabel(r'energy per turn ($J$)'); ax.set_xlim(0,9.5); ax.set_ylim(0.016,0.028)
ax.legend(frameon=False,loc='center right',fontsize=7.5); ax.set_xticks([1,3,5,7,9])
fig.text(0.02,0.98,'(a)',fontsize=10,fontweight='bold',va='top'); fig.text(0.66,0.98,'(b)',fontsize=10,fontweight='bold',va='top')
plt.savefig('figs/fig5.pdf'); plt.savefig('figs/fig5.png',dpi=200)

import numpy as np, json
from style import *
from matplotlib.patches import Polygon
from grid_exact import tiling, NRM
CEN=(0.0,0.0,1/12,1/3)
def bands(nb,n,m,w=0.03):
    return [np.concatenate([(cen+h-w/2+(w/(c//nb))*(np.arange(c//nb)+0.5)) for h in np.arange(nb)/nb]) for c,cen in zip((n,m,n,m),CEN)]
def spread(n,m):
    off=(0.137,0.311,0.523,0.719); return [(np.arange(c)+o)/c for c,o in zip((n,m,n,m),off)]
Ev=NRM/np.linalg.norm(NRM,axis=1)[:,None]; ea=np.eye(4,dtype=np.int64)
def draw(ax,pos,ms=1.0):
    N,B,z,info=tiling(pos); canon=info['canon']; verts=info['verts']
    for K,a,b in info['raw']:
        q=[K,K+ea[a],K+ea[a]+ea[b],K+ea[b]]; P=np.array([x@Ev for x in q])
        ax.add_patch(Polygon(P,fc=TILE['sq'] if abs(a-b)==2 else TILE['rh'],ec='0.55',lw=0.3))
        for x,p in zip(q,P):
            zz=z[verts[tuple(canon(x)[0])]]
            if zz!=4: ax.scatter(*p,s=ms*(4+3*zz),c=ZC[zz],ec='k',lw=0.3,zorder=3)
    ax.set_aspect('equal'); ax.axis('off'); ax.autoscale_view(); return N,z
fig=plt.figure(figsize=(DCOL,2.65)); gs=fig.add_gridspec(1,3,width_ratios=[0.85,1.05,1.0],wspace=0.32)
ax=fig.add_subplot(gs[0]); N,z=draw(ax,spread(6,4),0.8); ax.set_title(f'approximant, $N={N}$\n'+r'$z\neq4$ on %d%% of sites'%(round(100*np.mean(z!=4))),fontsize=8.5)
ax=fig.add_subplot(gs[1]); N,z=draw(ax,bands(1,6,4),1.6); ax.set_title(f'faceted, $N={N}$\none defect quintet per period',fontsize=8.5)
for zz in (3,5,6): ax.scatter([],[],s=1.6*(4+3*zz),c=ZC[zz],ec='k',lw=0.3,label=f'$z={zz}$')
ax.legend(loc='upper center',bbox_to_anchor=(0.5,0.0),ncol=3,frameon=False,handletextpad=0.1,columnspacing=0.8)
ax=fig.add_subplot(gs[2]); D=json.load(open('/mnt/user-data/outputs/bands.json'))
x=np.linspace(0,0.085,50)
for nb,c,mk,lab in ((1,'#0072B2','o','1 quintet'),(2,'#D55E00','s','4 quintets')):
    r=[y for y in D['res'] if y['nb']==nb]; Nn=np.array([y['N'] for y in r],float); E=np.array([y['dE'] for y in r])
    M=np.stack([np.ones_like(Nn),-Nn**-.5,Nn**-1.],1); p=np.linalg.lstsq(M,E,rcond=None)[0]
    ax.plot(Nn**-.5,E,mk,color=c,ms=4.5,label=lab+r' ($E_\infty=%.3f$)'%p[0]); ax.plot(x,p[0]-p[1]*x+p[2]*x**2,'-',color=c,lw=1)
ax.plot(x,-2.035*x,'k--',lw=0.9,label=r'$-2.03/\sqrt{N}$ (square)'); ax.axhline(0,color='0.7',lw=0.5)
ax.set_xlabel(r'$N^{-1/2}$'); ax.set_ylabel(r'$E-Ne_{\rm sq}$ per period ($J$)'); ax.set_xlim(0,0.085); ax.set_ylim(-0.33,0.16)
ax.legend(frameon=False,loc='lower left',fontsize=7.5,handlelength=1.3,borderaxespad=0.2)
for xx,t in ((0.125,'(a)'),(0.36,'(b)'),(0.665,'(c)')): fig.text(xx,0.98,t,fontsize=10,fontweight='bold',va='top')
plt.savefig('figs/fig2.pdf'); plt.savefig('figs/fig2.png',dpi=200)

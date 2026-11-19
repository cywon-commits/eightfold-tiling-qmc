import numpy as np, json
from style import *
from matplotlib.patches import Polygon
EV=np.array([[0,1],[1,-1],[1,0],[1,1]],float); EV/=np.linalg.norm(EV,axis=1)[:,None]
C=json.load(open('/home/claude/dense/candidates.json'))
def small(counts,tau): c=[c for c in C if c['counts']==counts and abs(c['tau']-tau)<1e-3][0]; return c['tiles'],c['T']
def crystal(f):
    d=json.load(open(f))[0]
    return (d['tiles'],d['T']) if isinstance(d,dict) else (d[2],d[3])
panels=[('isolated $z=6$ hubs\n'+r'$\sigma=1/2,\ \tau=1/9$',small('(3, 0, 3, 3)',1/9),3),
        ('$z=5$ carriers\n'+r'$\sigma=1/2,\ \tau=1/3$',small('(3, 0, 3, 3)',1/3),3),
        ('$z=5$ crystal\n'+r'$\sigma=3/2,\ \tau=1$',crystal('/home/claude/hf/cryst_a3_n1.json'),3),
        ('dice\n'+r'$\sigma=1/2,\ \tau=1$',crystal('/home/claude/hf/phase_a2_n2.json'),4)]
fig,axs=plt.subplots(1,4,figsize=(DCOL,2.2),gridspec_kw=dict(wspace=0.08))
for ax,(title,(tiles,T),L),lab in zip(axs,panels,'abcd'):
    T1,T2=[np.array(t) for t in T]; tl=[(np.array(R),a,b) for R,a,b in tiles]
    nb={}; allt=[]; shown=set()
    for i in range(-1,L+1):
        for j in range(-1,L+1):
            for R,a,b in tl:
                Rs=R+i*T1+j*T2; K=[tuple(x) for x in Rs]
                for s in range(4): nb.setdefault(K[s],set()).update([K[(s+1)%4],K[s-1]])
                if 0<=i<L and 0<=j<L: allt.append((Rs,a,b)); shown.update(K)
    cnt={0:0,1:0}
    for v in nb: cnt[sum(v)%2]+=1
    maj=0 if cnt[0]>=cnt[1] else 1
    for Rs,a,b in allt: ax.add_patch(Polygon(Rs@EV,fc=TILE['sq'] if abs(a-b)==2 else TILE['rh'],ec='0.55',lw=0.3))
    for v in shown:
        z=len(nb[v])
        if z==4: continue
        p=np.array(v)@EV; ax.scatter(*p,s=3+2.4*z,c=ZC.get(z,'k'),marker='o' if sum(v)%2==maj else 's',ec='k',lw=0.35,zorder=3)
    ax.set_aspect('equal'); ax.axis('off'); ax.autoscale_view(); ax.set_title(title,fontsize=8.5)
    ax.text(0.0,1.02,f'({lab})',transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom')
for z in (3,5,6): axs[1].scatter([],[],s=3+2.4*z,c=ZC[z],ec='k',lw=0.35,label=f'$z={z}$')
axs[1].scatter([],[],s=20,c='w',marker='o',ec='k',lw=0.4,label='majority'); axs[1].scatter([],[],s=20,c='w',marker='s',ec='k',lw=0.4,label='minority')
fig.legend(*axs[1].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(0.5,0.0),handletextpad=0.1,columnspacing=1.0)
plt.savefig('figs/fig6.pdf'); plt.savefig('figs/fig6.png',dpi=200)

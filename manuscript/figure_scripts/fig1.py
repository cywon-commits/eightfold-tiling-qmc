import numpy as np, itertools
from style import *
from matplotlib.patches import Polygon, FancyArrowPatch, Circle
from scipy.spatial import ConvexHull
from matplotlib.path import Path
k=np.arange(4); E=np.stack([np.cos(k*np.pi/4),np.sin(k*np.pi/4)],1); Pp=np.stack([np.cos(3*k*np.pi/4),np.sin(3*k*np.pi/4)],1)
cube=np.array(list(itertools.product([0,1],repeat=4))); W=cube@Pp; win=Path(W[ConvexHull(W).vertices]); off=np.array([0.1234,0.0567])
R=7; pts=np.array(list(itertools.product(range(-R,R+1),repeat=4))); V=pts[win.contains_points(pts@Pp-off)]
idx={tuple(v):i for i,v in enumerate(V)}; X=V@E; I4=np.eye(4,dtype=int)
edges=[(i,idx[tuple(v+I4[a])]) for i,v in enumerate(V) for a in range(4) if tuple(v+I4[a]) in idx]
z=np.bincount(np.array(edges).ravel(),minlength=len(V)); par=V.sum(1)%2
tiles=[]
for v in V:
    for a,b in itertools.combinations(range(4),2):
        q=[v,v+I4[a],v+I4[a]+I4[b],v+I4[b]]
        if all(tuple(x) in idx for x in q): tiles.append((np.array([X[idx[tuple(x)]] for x in q]),(b-a)==2))
fig=plt.figure(figsize=(DCOL,2.55)); gs=fig.add_gridspec(1,3,width_ratios=[0.95,1.15,1.0],wspace=0.08)
# (a) edge star and tiles
ax=fig.add_subplot(gs[0]); ax.set_aspect('equal'); ax.axis('off'); c0=np.array([-1.5,0.6])
for j in range(8):
    th=j*np.pi/4; d=np.array([np.cos(th),np.sin(th)])*0.75
    ax.annotate('',xy=c0+d,xytext=c0,arrowprops=dict(arrowstyle='-|>',lw=0.8,color='0.25',mutation_scale=7))
    if j<4: ax.text(*(c0+d*1.28),r'$\mathbf{v}_%d$'%(j+1),ha='center',va='center',fontsize=8.5)
ax.text(c0[0],c0[1]-1.2,'edge vectors',ha='center',fontsize=8.5)
sq=np.array([[0,0],[1,0],[1,1],[0,1]])*0.9+np.array([0.25,0.2])
ax.add_patch(Polygon(sq,fc=TILE['sq'],ec='0.3',lw=0.8))
for p,s in zip(sq,(0,1,0,1)): ax.scatter(*p,s=26,c='#c44e00' if s==0 else 'white',ec='k',lw=0.6,zorder=3)
ax.text(0.7,1.36,'square',ha='center',fontsize=8.5)
v1=np.array([1,0]); v2=np.array([np.cos(np.pi/4),np.sin(np.pi/4)]); o=np.array([-0.6,-1.35])
rh=np.array([o,o+v1,o+v1+v2,o+v2])*1.0
ax.add_patch(Polygon(rh,fc=TILE['rh'],ec='0.3',lw=0.8))
for p,s in zip(rh,(1,0,1,0)): ax.scatter(*p,s=26,c='#c44e00' if s==0 else 'white',ec='k',lw=0.6,zorder=3)
ax.text(o[0]+0.85,o[1]-0.12,'rhombus, obtuse on A:\n'+r'$\tau=+1$',ha='center',va='top',fontsize=8)
ax.scatter([],[],s=26,c='#c44e00',ec='k',lw=0.6,label='A'); ax.scatter([],[],s=26,c='white',ec='k',lw=0.6,label='B')
ax.legend(loc='upper right',bbox_to_anchor=(1.07,0.86),ncol=1,frameon=False,handletextpad=0.2)
ax.set_xlim(-2.5,2.6); ax.set_ylim(-2.45,1.65)
# (b) AB patch colored by z
ax=fig.add_subplot(gs[1]); ax.set_aspect('equal'); ax.axis('off'); Rv=4.6
drawn=[(p,s) for p,s in tiles if np.all(np.linalg.norm(p,axis=1)<Rv)]
corner={tuple(np.round(p,6)) for poly,_ in drawn for p in poly}
for p,s in drawn: ax.add_patch(Polygon(p,fc=TILE['sq'] if s else TILE['rh'],ec='0.45',lw=0.45))
ins=np.array([tuple(np.round(x,6)) in corner for x in X])
for zz in (3,5,6,7,8):
    s=ins&(z==zz); ax.scatter(X[s,0],X[s,1],s=6+3*zz,c=ZC[zz],ec='k',lw=0.35,zorder=3,label=f'$z={zz}$')
ax.legend(loc='upper center',bbox_to_anchor=(0.5,0.02),ncol=5,frameon=False,handletextpad=0.05,columnspacing=0.5,fontsize=8)
ax.set_xlim(-Rv,Rv); ax.set_ylim(-Rv,Rv)
# (c) direction wheels
ax=fig.add_subplot(gs[2]); ax.set_aspect('equal'); ax.axis('off'); fam=[1,2,3,4,1,2,3,4]
def wheel(cx,cy,P,title,star):
    for j in range(8):
        th=j*np.pi/4; f=fam[j]; col='#c44e00' if f in P else '#2a7ab8'
        ax.plot([cx,cx+0.8*np.cos(th)],[cy,cy+0.8*np.sin(th)],color=col,lw=2.2 if j in star else 0.8,alpha=1 if j in star else 0.55,solid_capstyle='round')
        ax.text(cx+1.0*np.cos(th),cy+1.0*np.sin(th),str(f),ha='center',va='center',fontsize=8,color=col)
    ax.text(cx,cy-1.45,title,ha='center',va='top',fontsize=8.5)
wheel(-1.25,0.0,{1,2},'interval\n{12|34}\n$z=4$ only',[0,2,4,6])
wheel(1.25,0.0,{1,3},'non-interval\n{13|24}\n$z=4,6,8$',[0,1,2,3,4,5,6,7])
ax.text(-0.05,1.35,r'$P$',color='#c44e00',ha='right',fontsize=9); ax.text(0.05,1.35,r'$Q$',color='#2a7ab8',ha='left',fontsize=9)
ax.set_xlim(-2.4,2.4); ax.set_ylim(-2.6,1.6)
for xx,t in ((0.125,'(a)'),(0.425,'(b)'),(0.715,'(c)')): fig.text(xx,0.97,t,fontsize=10,fontweight='bold',va='top')
plt.savefig('figs/fig1.pdf'); plt.savefig('figs/fig1.png',dpi=200)

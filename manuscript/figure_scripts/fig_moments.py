import numpy as np, json
from style import *
from matplotlib.patches import Polygon
from matplotlib.colors import TwoSlopeNorm
from phases import graph, EV
from lswt_local import local
show=[('hub',8,'(a)','$z=6$ hubs\n$m=1/27$'),('B15',6,'(b)','3/5 pairs\n$m=1/15$'),
      ('E10',12,'(c)','hub-free crystal\n$m=1/5$'),('dice',4,'(d)','dice\n$m=1/3$')]
P=json.load(open('perphase.json'))
fig=plt.figure(figsize=(DCOL,2.6)); gs=fig.add_gridspec(1,5,width_ratios=[1,1,1,1,1.25],wspace=0.12,bottom=0.2,top=0.86)
norm=TwoSlopeNorm(vmin=-0.45,vcenter=0,vmax=0.45); cmap=plt.cm.RdBu_r
for k,(key,L,lab,title) in enumerate(show):
    ax=fig.add_subplot(gs[k]); g=graph(key,L); r=local(g['N'],g['B'])
    cen=np.array([ (R@EV).mean(0) for R in g['tiles']]); c0=np.median(cen,0)
    W=4.2; sel=[t for t in range(len(cen)) if abs(cen[t,0]-c0[0])<W and abs(cen[t,1]-c0[1])<W]
    for t in sel:
        R=g['tiles'][t]; a,b=g['kinds'][t]
        ax.add_patch(Polygon(R@EV,fc='#ece6da' if abs(a-b)==2 else '#d5dde6',ec='0.6',lw=0.25))
    pts={}
    for R,q in ((g['tiles'][t],g['ids'][t]) for t in sel):
        for c,v in zip(R@EV,q): pts.setdefault(v,[]).append(c)
    xs=[];ys=[];cs=[];ss=[]
    for v,cl in pts.items():
        for c in cl: xs.append(c[0]); ys.append(c[1]); cs.append(r['Sz'][v]); ss.append(0.9*(r['z'][v]-1.5)**2)
    ax.scatter(xs,ys,c=cs,s=ss,cmap=cmap,norm=norm,ec='k',lw=0.25,zorder=3)
    ax.set_aspect('equal'); ax.axis('off'); ax.autoscale_view(); ax.set_title(title,fontsize=8.5,pad=2)
    ax.text(-0.08,1.17,lab,transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom')
# (e) mean |Sz| by class vs phase moment
ax=fig.add_subplot(gs[4]); order=['hub','B15','P24','Q23','AD24','AC22','E10','z5crystal','AE23','G11','dice']
mk={3:'o',4:'s',5:'D',6:'h'}
for cls,col in (('maj','#b2182b'),('min','#2166ac')):
    for zz in (3,4,5,6):
        xs=[];ys=[]
        for key in order:
            c=P[key]['cls'].get(f'{cls}{zz}')
            if c and c[1]>0.02: xs.append(P[key]['m']); ys.append(abs(c[0]))
        if xs: ax.plot(xs,ys,mk[zz],color=col,ms=4,mfc=col if cls=='maj' else 'w',mew=0.8)
ax.axhline(0.5-0.1966,color='0.5',lw=0.6,ls='--'); ax.text(0.335,0.308,'square',fontsize=7.5,ha='right',va='bottom',color='0.4')
for zz in (3,4,5,6): ax.plot([],[],mk[zz],color='0.3',ms=4,mfc='w',label=f'$z={zz}$')
ax.plot([],[],'s',color='#b2182b',ms=4,label='majority'); ax.plot([],[],'s',color='#2166ac',ms=4,mfc='w',label='minority')
ax.legend(frameon=False,fontsize=7,loc='upper left',ncol=2,handletextpad=0.1,columnspacing=0.5,borderaxespad=0.1)
ax.set_xlabel(r'phase moment $m$'); ax.set_ylabel(r'$|\langle S_i^z\rangle|$ (LSWT)',labelpad=1); ax.set_xlim(0,0.35); ax.set_ylim(0.28,0.46)
ax.yaxis.set_label_position('right'); ax.yaxis.tick_right()
ax.text(-0.02,1.03,'(e)',transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom')
cax=fig.add_axes([0.2,0.10,0.36,0.03]); cb=plt.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
cb.set_label(r'$\langle S_i^z\rangle$ (LSWT; red along $h$); symbol size $\propto z$',fontsize=8); cb.ax.tick_params(labelsize=7.5)
plt.savefig('fig_moments.pdf'); plt.savefig('fig_moments.png',dpi=200)

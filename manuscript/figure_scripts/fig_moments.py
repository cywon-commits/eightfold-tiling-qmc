import numpy as np, json
from style import *
from matplotlib.patches import Polygon
from matplotlib.colors import TwoSlopeNorm
from phases import graph, EV
from lswt_local import local
show=[('hub',8,'(a)','$z=6$ hubs\n$m=1/27$'),('B15',6,'(b)','3/5 pairs\n$m=1/15$'),
      ('E10',12,'(c)','hub-free crystal\n$m=1/5$'),('dice',4,'(d)','dice\n$m=1/3$')]
P=json.load(open('perphase.json'))
from matplotlib.colors import Normalize
fig=plt.figure(figsize=(DCOL,4.9))
gs=fig.add_gridspec(2,3,width_ratios=[1,1,0.95],wspace=0.10,hspace=0.20,left=0.02,right=0.93,bottom=0.04,top=0.95)
NA=Normalize(0.29,0.44); NB=Normalize(0.29,0.37); CA=plt.cm.Reds; CB=plt.cm.Blues
rng=[]
for k,(key,L,lab,title) in enumerate(show):
    ax=fig.add_subplot(gs[k//2,k%2]); g=graph(key,L); r=local(g['N'],g['B'])
    cen=np.array([(R@EV).mean(0) for R in g['tiles']]); c0=np.median(cen,0)
    W=4.6; sel=[t for t in range(len(cen)) if abs(cen[t,0]-c0[0])<W and abs(cen[t,1]-c0[1])<W]
    for t in sel:
        R=g['tiles'][t]; a_,b_=g['kinds'][t]
        ax.add_patch(Polygon(R@EV,fc='#f2ede3' if abs(a_-b_)==2 else '#e6eaef',ec='0.62',lw=0.3))
    pts={}
    for R,q in ((g['tiles'][t],g['ids'][t]) for t in sel):
        for c,v in zip(R@EV,q): pts[(round(c[0],4),round(c[1],4))]=v
    xy=np.array(list(pts.keys())); vs=np.array(list(pts.values())); sz=r['Sz'][vs]; rng+=list(sz)
    up=sz>0
    ax.scatter(xy[up,0],xy[up,1],c=sz[up],cmap=CA,norm=NA,s=22,ec='k',lw=0.35,zorder=3,marker='o')
    ax.scatter(xy[~up,0],xy[~up,1],c=-sz[~up],cmap=CB,norm=NB,s=22,ec='k',lw=0.35,zorder=3,marker='s')
    ax.set_aspect('equal'); ax.axis('off'); ax.autoscale_view()
    ax.set_title(r'$\bf{%s}$  %s'%(lab,title.replace('\n',', ')),loc='left',fontsize=9,pad=3)
print('Sz range: min %.3f max %.3f'%(min(rng),max(rng)))
# (e) mean |Sz| by class vs phase moment
ax=fig.add_subplot(gs[0,2]); order=['hub','B15','P24','Q23','AD24','AC22','E10','z5crystal','AE23','G11','dice']
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
cax1=fig.add_axes([0.715,0.30,0.20,0.022]); cb1=plt.colorbar(plt.cm.ScalarMappable(norm=NA,cmap=CA),cax=cax1,orientation='horizontal')
cb1.set_label(r'majority $\langle S_i^z\rangle$ (along $h$; circles)',fontsize=8); cb1.ax.tick_params(labelsize=7.5)
cax2=fig.add_axes([0.715,0.14,0.20,0.022]); cb2=plt.colorbar(plt.cm.ScalarMappable(norm=NB,cmap=CB),cax=cax2,orientation='horizontal')
cb2.set_label(r'minority $|\langle S_i^z\rangle|$ (against $h$; squares)',fontsize=8); cb2.ax.tick_params(labelsize=7.5)
fig.text(0.815,0.40,'LSWT, field-polarized ground state',fontsize=7.5,ha='center',color='0.35')
plt.savefig('fig_moments.pdf'); plt.savefig('fig_moments.png',dpi=200)

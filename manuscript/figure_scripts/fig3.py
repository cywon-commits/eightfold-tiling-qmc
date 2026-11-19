import numpy as np
from style import *
from matplotlib.patches import Polygon, FancyArrowPatch
c1,c3=0.15,0.40
O0,O1,O2,O3=(0,0),(1,0),(1,1),(0,1)
A=(c1,0);F=(c3,0);E=(0,c3);C=(0,1-c1);D=(c1,1);G=(c3,1);H=(1,c3);B=(1,1-c1)
P=((c1+c3)/2,(c3-c1)/2); Q=((1+c1+c3)/2,(1+c3-c1)/2)
faces=[([O0,A,P,E],4),([A,F,P],3),([F,O1,H,Q,P],5),([Q,H,B],3),([C,O3,D],3),([G,O2,B,Q],4),([E,P,Q,G,D,C],6)]
col={3:'#f4a261',4:'#ffffff',5:'#8ecae6',6:'#2a7ab8'}
fig,axs=plt.subplots(1,2,figsize=(COL,1.85),gridspec_kw=dict(wspace=0.35))
ax=axs[0]
for poly,s in faces:
    p=np.array(poly); ax.add_patch(Polygon(p,fc=col[s],ec='none',alpha=0.85))
    cx,cy=p.mean(0); ax.text(cx,cy,str(s),ha='center',va='center',fontsize=8.5,fontweight='bold',color='w' if s==6 else 'k')
ax.plot([0,1,1,0,0],[0,0,1,1,0],color='0.2',lw=1.0)
ax.plot([A[0],B[0]],[A[1],B[1]],color='#7b2cbf',lw=1.2); ax.plot([C[0],D[0]],[C[1],D[1]],color='#7b2cbf',lw=1.2)
ax.plot([E[0],F[0]],[E[1],F[1]],color='#2d6a4f',lw=1.2); ax.plot([G[0],H[0]],[G[1],H[1]],color='#2d6a4f',lw=1.2)
ax.text(0.5,-0.09,r'$L_0$',ha='center',va='top',fontsize=8.5); ax.text(-0.04,0.72,r'$L_2$',ha='right',va='center',fontsize=8.5)
ax.text(1.03,0.86,r'$L_1$',color='#7b2cbf',fontsize=8.5,va='center'); ax.text(-0.04,0.40,r'$L_3$',color='#2d6a4f',fontsize=8.5,ha='right',va='center')
ax.set_xlim(-0.2,1.18); ax.set_ylim(-0.22,1.05); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('faces {3,3,3,4,4,5,6}',fontsize=8.5)
ax=axs[1]
ax.plot([0,1,1,0,0],[0,0,1,1,0],color='0.2',lw=1.0); ax.plot([0,1],[0,1],color='#c1121f',lw=1.0); ax.plot([0,1],[1,0],color='#c1121f',lw=1.0)
for (x,y),t in (((0.2,0.5),'I'),((0.5,0.8),'II'),((0.8,0.5),'III'),((0.5,0.2),'IV')): ax.text(x,y,t,ha='center',va='center',fontsize=9)
for (a,b) in (((0.27,0.62),(0.38,0.73)),((0.62,0.73),(0.73,0.62)),((0.73,0.38),(0.62,0.27)),((0.38,0.27),(0.27,0.38))):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,color='0.3',lw=0.8,connectionstyle='arc3,rad=-0.3'))
ax.scatter([c1],[c3],s=14,c='k',zorder=4)
ax.set_xlabel(r'$c_1$',labelpad=1); ax.set_ylabel(r'$c_3$',labelpad=1); ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_aspect('equal')
ax.set_title(r'chambers, $C_4$ map',fontsize=8.5)
for sp in ax.spines.values(): sp.set_visible(False)
fig.text(0.01,0.98,'(a)',fontsize=10,fontweight='bold',va='top'); fig.text(0.52,0.98,'(b)',fontsize=10,fontweight='bold',va='top')
plt.savefig('figs/fig3.pdf'); plt.savefig('figs/fig3.png',dpi=220)

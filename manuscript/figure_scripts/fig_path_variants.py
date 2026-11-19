"""Four candidate layouts for the energy-balance figure (Fig. 8)."""
import json, numpy as np
from style import *
import pathdata as P
from matplotlib.colors import to_rgb
hs=P.hs; q=P.q; GRAY='0.55'
LAB=['canted $z\\equiv4$ facets']+P.CATS; COL=[GRAY]+P.CC
FR=np.concatenate([P.F0[:,:,None],P.FC],2)          # (NS,H,7)
FRc=np.concatenate([P.f0c[:,None],P.FCc],1)         # central (H,7)
show=[k for k in range(7) if FR[:,:,k].max()>1e-3]
RS=json.load(open('/mnt/user-data/outputs/sectors/resample_N24.json'))
steps=[s for s in RS['presence'] if s['frac']>=0.95 and s['m']>0]
def window(ax,hatch=False):
    ax.axvspan(0.09,0.18,color='#f2f2f2',zorder=0)
    if hatch: ax.axvspan(0.09,0.18,facecolor='none',edgecolor='0.6',hatch='////',lw=0,zorder=0.5)
def panel_bc(axb,axc,legend_loc_b='lower right'):
    for a in (axb,axc): window(a)
    for A,col,lab in ((P.EX,'#2166ac','in-plane ($xx{+}yy$)'),(P.EZ,'#b2182b','normal ($zz$)')):
        lo,md,hi=q(A); axb.fill_between(hs,lo,hi,color=col,alpha=0.25,lw=0); axb.plot(hs,md,color=col,lw=1.3,label=lab)
    axb.set_xlim(0,0.3); axb.set_xlabel(r'$h/J$'); axb.set_ylabel(r'exchange ($J$ per site)')
    axb.legend(frameon=False,fontsize=7.5,loc=legend_loc_b,handlelength=1.2)
    for A,col,lab in ((P.EEX-P.ESQ,'#d95f02',r'exchange $-\,e_{\rm sq}$'),(P.EZM,'#1b9e77',r'Zeeman'),(P.TOT-P.ESQ,'k',r'total $-\,e_{\rm sq}$')):
        lo,md,hi=q(1e3*A); axc.fill_between(hs,lo,hi,color=col,alpha=0.22,lw=0); axc.plot(hs,md,color=col,lw=1.3,label=lab)
    axc.axhline(0,color='0.6',lw=0.5,ls=':'); axc.set_xlim(0,0.3); axc.set_xlabel(r'$h/J$'); axc.set_ylabel(r'energy ($10^{-3}J$ per site)')
    axc.legend(frameon=False,fontsize=7.5,loc='lower left',handlelength=1.2)
def tag(ax,l,x=-0.2,y=1.0): ax.text(x,y,f'({l})',transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom')
def save(fig,name): fig.savefig(f'../figs/{name}.pdf'); fig.savefig(f'../figs/{name}.png',dpi=170); plt.close(fig)

# ---------- V1: stacked central composition + step-field bars ----------
fig=plt.figure(figsize=(DCOL,3.0))
gs=fig.add_gridspec(2,3,height_ratios=[0.16,1],hspace=0.05,wspace=0.42,left=0.07,right=0.985,bottom=0.15,top=0.93)
ax=fig.add_subplot(gs[1,0]); st=fig.add_subplot(gs[0,0],sharex=ax)
window(ax,hatch=True); base=np.zeros(len(hs))
for k in show:
    ax.fill_between(hs,base,base+FRc[:,k],color=COL[k],lw=0,alpha=0.9); 
    mid=base+FRc[:,k]/2; i=np.argmax(FRc[:,k]) 
    if FRc[i,k]>0.25:
        nm={0:'canted',1:'hubs',2:'ribbons',3:'window\ncrystals',4:'$m{=}1/5$',5:'$m{>}1/5$',6:'dice'}[k]
        ax.text(hs[i] if k not in (3,) else 0.135,mid[i],nm,fontsize=6.8,ha='center',va='center',color='w' if k in (4,6,1) else 'k')
    base=base+FRc[:,k]
ax.set_xlim(0,0.3); ax.set_ylim(0,1); ax.set_xlabel(r'$h/J$'); ax.set_ylabel('fraction of sites')
for s in steps:
    st.errorbar(s['onset'],0,xerr=[[s['onset']-s['onset16']],[s['onset84']-s['onset']]],fmt='|',color='k',ms=6,elinewidth=1.2,capsize=2)
    ha={0.033:'right',0.083:'left'}.get(round(s['m'],3),'center')
    st.text(s['onset']+(-0.002 if ha=='right' else 0.002 if ha=='left' else 0),0.7,'$m$=%.3f'%s['m'],fontsize=6.3,ha=ha,va='bottom')
st.set_ylim(-1,1.9); st.axis('off'); st.text(0.0,0.0,'steps:',fontsize=6.5,ha='left',va='center')
tag(ax,'a',-0.28,1.13)
axb=fig.add_subplot(gs[:,1]); axc=fig.add_subplot(gs[:,2]); panel_bc(axb,axc); tag(axb,'b'); tag(axc,'c')
save(fig,'fig_path_v1')

# ---------- V2: small multiples per class ----------
fig=plt.figure(figsize=(DCOL,4.4))
outer=fig.add_gridspec(1,2,width_ratios=[1.05,1],wspace=0.24,left=0.075,right=0.985,bottom=0.09,top=0.96)
left=outer[0].subgridspec(len(show),1,hspace=0.12); right=outer[1].subgridspec(2,1,hspace=0.38)
axs=[]
for r,k in enumerate(show):
    a=fig.add_subplot(left[r],sharex=axs[0] if axs else None); axs.append(a); window(a)
    lo,md,hi=q(FR[:,:,k]); a.fill_between(hs,lo,hi,color=COL[k],alpha=0.3,lw=0); a.plot(hs,md,color=COL[k],lw=1.2)
    a.set_ylim(0,1.05); a.set_yticks([0,1]); a.tick_params(labelsize=7)
    a.text(0.20 if k<=2 else 0.004,0.55,LAB[k],ha='left',va='center',fontsize=7.4,color=COL[k] if k else '0.3',
           bbox=dict(fc='white',ec='none',alpha=0.85,pad=0.8),zorder=5)
    if r<len(show)-1: plt.setp(a.get_xticklabels(),visible=False)
axs[-1].set_xlim(0,0.3); axs[-1].set_xlabel(r'$h/J$'); fig.text(0.02,0.53,'fraction of sites',rotation=90,va='center',fontsize=9)
tag(axs[0],'a',-0.13,1.05)
axb=fig.add_subplot(right[0]); axc=fig.add_subplot(right[1]); panel_bc(axb,axc); tag(axb,'b',-0.22); tag(axc,'c',-0.22)
save(fig,'fig_path_v2')

# ---------- V3: heat map of mean fraction ----------
fig=plt.figure(figsize=(DCOL,3.0))
gs=fig.add_gridspec(1,3,width_ratios=[1.25,1,1],wspace=0.42,left=0.16,right=0.985,bottom=0.15,top=0.88)
ax=fig.add_subplot(gs[0]); mean=FR.mean(0)[:,show].T
im=ax.imshow(mean,aspect='auto',origin='lower',extent=[hs[0],hs[-1],-0.5,len(show)-0.5],cmap='Greys',vmin=0,vmax=1,interpolation='nearest')
for r,k in enumerate(show):      # central path: fraction drawn inside each row (bottom = 0, top = 1)
    ax.plot(hs,r-0.42+0.84*FRc[:,k],color=COL[k] if k else '#c9a227',lw=1.1)
    ax.axhline(r+0.5,color='0.8',lw=0.4)
ax.set_yticks(range(len(show))); ax.set_yticklabels([LAB[k] for k in show],fontsize=7)
for t,k in zip(ax.get_yticklabels(),show): t.set_color(COL[k] if k else '0.3')
ax.axvline(0.09,color='#c9a227',lw=0.7,ls='--'); ax.axvline(0.18,color='#c9a227',lw=0.7,ls='--')
ax.set_xlabel(r'$h/J$'); ax.set_xlim(0,0.3)
cax=ax.inset_axes([0.55,1.03,0.45,0.045]); cb=fig.colorbar(im,cax=cax,orientation='horizontal'); cb.ax.tick_params(labelsize=6.5,pad=1)
cax.text(-0.03,0.5,'mean fraction',transform=cax.transAxes,ha='right',va='center',fontsize=7)
tag(ax,'a',-0.62)
axb=fig.add_subplot(gs[1]); axc=fig.add_subplot(gs[2]); panel_bc(axb,axc); tag(axb,'b'); tag(axc,'c')
save(fig,'fig_path_v3')

# ---------- V4: m(h) + phase strip on top, (b),(c) below ----------
fig=plt.figure(figsize=(DCOL,4.8))
top=fig.add_gridspec(2,1,height_ratios=[1,0.17],hspace=0.06,left=0.08,right=0.80,bottom=0.50,top=0.97)
bot=fig.add_gridspec(1,2,wspace=0.3,left=0.08,right=0.80,bottom=0.08,top=0.38)
ax=fig.add_subplot(top[0]); sp=fig.add_subplot(top[1],sharex=ax)
window(ax); lo,md,hi=q(P.M); ax.fill_between(hs,lo,hi,color='#08306b',alpha=0.25,lw=0); l1,=ax.plot(hs,md,color='#08306b',lw=1.4)
l2,=ax.plot(hs,2*P.szc,color='#08306b',lw=0.8,ls='--')
ax.axhline(0.2,color='#D55E00',lw=0.7,ls=':'); ax.set_ylabel(r'$m$'); ax.set_ylim(-0.005,0.31); ax.set_xlim(0,0.3)
plt.setp(ax.get_xticklabels(),visible=False); ax.text(0.297,0.205,'$1/5$',color='#D55E00',fontsize=7.5,ha='right')
leg1=ax.legend([l1,l2],['median (band: 16–84%)','central path'],frameon=False,fontsize=7.5,loc='upper left')
mode=np.array([np.bincount(P.DOM[:,i],minlength=7).argmax() for i in range(len(hs))])
prob=np.array([np.mean(P.DOM[:,i]==mode[i]) for i in range(len(hs))])
img=np.ones((1,len(hs),4))
for i in range(len(hs)): img[0,i,:3]=to_rgb(COL[mode[i]]); img[0,i,3]=0.15+0.85*prob[i]
sp.imshow(img,aspect='auto',extent=[hs[0],hs[-1],0,1],interpolation='nearest'); sp.set_yticks([]); sp.set_xlabel(r'$h/J$',labelpad=1)
sp.text(1.01,0.5,'phase with most sites\n(opacity = probability)',transform=sp.transAxes,fontsize=6.8,va='center')
hand=[plt.Rectangle((0,0),1,1,color=COL[k]) for k in show]
ax.legend(hand,[LAB[k] for k in show],frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(1.01,1.0),title='strip colors',title_fontsize=7)
ax.add_artist(leg1); tag(ax,'a',-0.08)
axb=fig.add_subplot(bot[0]); axc=fig.add_subplot(bot[1]); panel_bc(axb,axc); tag(axb,'b',-0.2); tag(axc,'c',-0.2)
save(fig,'fig_path_v4')
print('done')

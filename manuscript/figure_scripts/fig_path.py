"""Energy balance along the ground-state path (N<=24) with its uncertainty: median and 16-84% band over 500 redraws
of the phase energies (same error model as Fig. 7). Central path (no redraw) dashed in (b),(c)."""
import numpy as np, json, glob
from style import *
ESQ=-0.669437; s0=1/np.sqrt(2); chi=0.065
D=np.load('/mnt/user-data/outputs/sectors/path_redraws_N24.npz',allow_pickle=True)
W=D['W']; hs=D['hs']; S=D['sigma']; T=D['tau']; EPS=D['eps']; qk=D['qkey']; NS,H,n=W.shape
# QMC split of exchange energy into zz and xy parts (seeded runs, largest cell)
SEED='/mnt/user-data/outputs/fieldfig/runs_seeds/'; byk={}
for f in glob.glob(SEED+'*_s*.json'):
    r=json.load(open(f)); byk.setdefault(r['key'],{}).setdefault(r['L'],[]).append(r)
fz={k:np.mean([r['E_zz']/r['E_J'] for r in d[max(d)]]) for k,d in byk.items()}
mP=T/(2*(1+S))                                     # phase moment m=(N_A-N_B)/N
mm=np.array([mP[i] for i in range(n) if qk[i] in fz]); ff=np.array([fz[qk[i]] for i in range(n) if qk[i] in fz])
cf=np.polyfit(mm,ff,1)                             # unmeasured phases: zz share interpolated in m
FZ=np.array([fz[qk[i]] if qk[i] in fz else np.polyval(cf,mP[i]) for i in range(n)])
print('zz share vs m fit:',cf,' rms',np.std(ff-np.polyval(cf,mm)))
ET=ESQ+EPS/(1+S)                                   # exchange energy per site of each phase (central eps)
# canted square facets (L=16 fixed-M split as before)
sq=sorted([json.load(open(f)) for f in glob.glob('/mnt/user-data/outputs/fieldfig/runs/sq16_*.json')],key=lambda r:r['fixM'])
M=np.array([r['fixM'] for r in sq]); x=M*(M+1)/256**2
bz=np.polyfit(x,[r['E_zz']/256 for r in sq],1)[0]; bx=np.polyfit(x,[r['E_xy']/256 for r in sq],1)[0]; gz=bz/(bz+bx); gx=bx/(bz+bx)
def sq_h(h): ms=chi*h; c=ms**2/(2*chi); return ESQ/3+gz*c, 2*ESQ/3+gx*c, ms
SQ=np.array([sq_h(h) for h in hs])
def cat(i):
    if qk[i]=='hub': return 0
    if qk[i]=='runs3': return 1
    if qk[i]=='dice': return 5
    if abs(mP[i]-0.2)<1e-6: return 3
    return 4 if mP[i]>0.2 else 2
CATS=['isolated $z=6$ hubs','ribbons','window crystals','crystals on $m=1/5$','crystals with $m>1/5$','dice']
CC=['#4c72b0','#8da0cb','#1b9e77','#e7298a','#d95f02','#b3002d']; ci=np.array([cat(i) for i in range(n)])
def decomp(Wt):
    f=Wt*(1+S[None,:])/(1+s0); f0=1-f.sum(1)
    Fc=np.stack([f[:,ci==c].sum(1) for c in range(6)],1)
    ezz=(f*(ET*FZ)[None,:]).sum(1)+f0*SQ[:,0]; exy=(f*(ET*(1-FZ))[None,:]).sum(1)+f0*SQ[:,1]
    sz=(f*(mP/2)[None,:]).sum(1)+f0*SQ[:,2]
    return f0,Fc,ezz,exy,sz
R=[decomp(W[s]) for s in range(NS)]
F0=np.array([r[0] for r in R]); FC=np.array([r[1] for r in R]); EZ=np.array([r[2] for r in R]); EX=np.array([r[3] for r in R]); SZ=np.array([r[4] for r in R])
EEX=EZ+EX; EZM=-hs[None,:]*SZ; TOT=EEX+EZM
q=lambda A:np.percentile(A,[16,50,84],axis=0)
fig,axs=plt.subplots(1,3,figsize=(DCOL,3.05),gridspec_kw=dict(wspace=0.38,bottom=0.32,top=0.93))
for a in axs: a.axvspan(0.09,0.18,color='#f2f2f2',zorder=0)
axs[0].axvspan(0.09,0.18,facecolor='none',edgecolor='0.55',hatch='////',lw=0,zorder=0.5)
axs[0].text(0.135,0.5,'approximants;\nidentity depends\non $N_{\\max}$',ha='center',va='center',fontsize=6.8,color='0.25',zorder=5,
            bbox=dict(fc='white',ec='none',alpha=0.8,pad=1.2))
ax=axs[0]; lo,md,hi=q(F0); ax.fill_between(hs,lo,hi,color='0.6',alpha=0.35,lw=0); ax.plot(hs,md,color='0.45',lw=1.1,label='canted $z\\equiv4$')
for c in range(6):
    lo,md,hi=q(FC[:,:,c])
    if hi.max()<1e-3: continue
    ax.fill_between(hs,lo,hi,color=CC[c],alpha=0.25,lw=0); ax.plot(hs,md,color=CC[c],lw=1.1,label=CATS[c])
ax.set_xlim(0,0.3); ax.set_ylim(0,1.02); ax.set_xlabel(r'$h/J$'); ax.set_ylabel('fraction of sites')
fig.legend(*ax.get_legend_handles_labels(),frameon=False,fontsize=7.5,loc='lower center',ncol=4,bbox_to_anchor=(0.5,0.0),handlelength=1.2,columnspacing=1.0)
ax=axs[1]
for A,col,lab in ((EX,'#2166ac','in-plane ($xx{+}yy$)'),(EZ,'#b2182b','normal ($zz$)')):
    lo,md,hi=q(A); ax.fill_between(hs,lo,hi,color=col,alpha=0.25,lw=0); ax.plot(hs,md,'-',color=col,lw=1.3,label=lab)
ax.set_xlim(0,0.3); ax.set_xlabel(r'$h/J$'); ax.set_ylabel(r'energy ($J$ per site)')
ax.legend(frameon=False,fontsize=7.5,loc='lower right',bbox_to_anchor=(1.0,0.12),handlelength=1.2)
ax=axs[2]
for A,col,lab in ((EEX-ESQ,'#d95f02',r'exchange $-\,e_{\rm sq}$'),(EZM,'#1b9e77',r'Zeeman $-h\langle S^z\rangle$'),(TOT-ESQ,'k',r'total $-\,e_{\rm sq}$')):
    lo,md,hi=q(1e3*A); ax.fill_between(hs,lo,hi,color=col,alpha=0.22,lw=0); ax.plot(hs,md,'-',color=col,lw=1.3,label=lab)
ax.axhline(0,color='0.6',lw=0.5,ls=':'); ax.set_xlim(0,0.3); ax.set_xlabel(r'$h/J$'); ax.set_ylabel(r'energy ($10^{-3}J$ per site)')
ax.legend(frameon=False,fontsize=7.5,loc='lower left')
for a,l in zip(axs,'abc'): a.text(-0.3,1.0,f'({l})',transform=a.transAxes,fontsize=10,fontweight='bold',va='bottom')
plt.savefig('fig_path.pdf'); plt.savefig('fig_path.png',dpi=200)
# numbers for the text
def at(A,h): i=np.argmin(abs(hs-h)); return np.percentile(A[:,i],[16,50,84])
print('end: zz',at(EZ,0.299),'xy',at(EX,0.299),'exchange rise',np.percentile(EEX[:,-1]-EEX[:,0],[16,50,84]))
cross=[hs[np.argmax(EZ[s]<EX[s])] if (EZ[s]<EX[s]).any() else np.nan for s in range(NS)]
print('zz/xy crossing h',np.nanpercentile(cross,[16,50,84]))
print('hub fraction after onset (h=0.08):',at(FC[:,:,0],0.08),'f0',at(F0,0.08))
print('window max total-energy band width (1e-3J):',np.max(np.percentile(1e3*TOT,84,0)-np.percentile(1e3*TOT,16,0)))
print('end fractions dice',at(FC[:,:,5],0.299),'m>1/5',at(FC[:,:,4],0.299))

import numpy as np, json, glob
from style import *
RUN='/mnt/user-data/outputs/fieldfig/runs/'
SEED='/mnt/user-data/outputs/fieldfig/runs_seeds/'
ESQ=-0.669437; s0=1/np.sqrt(2)
import sys; sys.path.insert(0,'/mnt/user-data/outputs/sectors')
PATH=json.load(open('/mnt/user-data/outputs/sectors/path_N24.json'))
Q={}
byk={}
for f in glob.glob(SEED+'*_s*.json'):
    r=json.load(open(f)); byk.setdefault(r['key'],{}).setdefault(r['L'],[]).append(r)
for k,d in byk.items():
    rs=d[max(d)]; N=rs[0]['N']
    Q[k]=dict(ezz=np.mean([r['E_zz'] for r in rs])/N,exy=np.mean([r['E_xy'] for r in rs])/N,e=np.mean([r['E_J'] for r in rs])/N,sz=rs[0]['fixM']/N,sigma=rs[0]['sigma'])
# canted square facets from fixed-M runs on L=16
sq=sorted([json.load(open(f)) for f in glob.glob(RUN+'sq16_*.json')],key=lambda r:r['fixM'])
N16=256; M=np.array([r['fixM'] for r in sq]); x=M*(M+1)/N16**2
ezz=np.array([r['E_zz']/N16 for r in sq]); exy=np.array([r['E_xy']/N16 for r in sq]); etot=ezz+exy
bz=np.polyfit(x,ezz,1)[0]; bx=np.polyfit(x,exy,1)[0]; chi_fit=1/(2*(bz+bx)); chi=0.065
fz=bz/(bz+bx); fx=bx/(bz+bx)            # how the canting cost m^2/(2chi) splits between zz and xy (L=16 sectors)
print('square: chi(L=16 tower)=%.4f, used chi=%.3f, split zz:xy = %.2f:%.2f'%(chi_fit,chi,fz,fx))
def sq_h(h): ms=chi*h; c=ms**2/(2*chi); return ESQ/3+fz*c, 2*ESQ/3+fx*c, ms
info=PATH['info']
for k,q in Q.items():
    if k not in info: continue
    et=ESQ+info[k]['eps']/(1+q['sigma']); q['ezz'],q['exy']=et*q['ezz']/q['e'],et*q['exy']/q['e']; q['e']=et
rows=PATH['rows']; hs=np.array([h for h,w in rows])
keys=[]
for h,w in rows:
    for k in w:
        if k not in keys: keys.append(k)
missing=[k for k in keys if k not in Q]; print('phases on path:',keys,'without QMC split:',missing)
lab={'hub':'$z=6$ hubs','runs3':'ribbons','z5crystal':'$z=5$ crystal','dice':'dice'}
for k in keys:
    if k not in lab: lab[k]=r'%s ($\sigma=%.2f$)'%(k,info[k]['sigma'])
pal=['#4c72b0','#8da0cb','#a6d854','#66c2a5','#ffd92f','#e5c494','#fc8d62','#e78ac3','#b3002d','#8c564b','#17becf','#bcbd22','#7f7f7f','#9467bd','#2ca02c']
colors=dict(zip(keys,pal))
F={k:np.zeros(len(hs)) for k in keys}; f0=np.ones(len(hs))
Ezz=np.zeros(len(hs)); Exy=np.zeros(len(hs)); Sz=np.zeros(len(hs))
for t,(h,w) in enumerate(rows):
    tot=0
    for k,xw in w.items():
        q=Q[k]; f=xw*(1+q['sigma'])/(1+s0); F[k][t]=f; tot+=f
        Ezz[t]+=f*q['ezz']; Exy[t]+=f*q['exy']; Sz[t]+=f*q['sz']
    f0[t]=1-tot; z0,x0,m0=sq_h(h); Ezz[t]+=f0[t]*z0; Exy[t]+=f0[t]*x0; Sz[t]+=f0[t]*m0
Eex=Ezz+Exy; Ez=-hs*Sz
steps=[hs[i] for i in range(1,len(hs)) if any(abs(F[k][i]-F[k][i-1])>1e-3 for k in keys)]
fig,axs=plt.subplots(1,3,figsize=(DCOL,3.05),gridspec_kw=dict(wspace=0.38,bottom=0.32,top=0.93))
ax=axs[0]; base=np.zeros(len(hs))
ax.fill_between(hs,0,f0,color='#eeeeee',lw=0,label='canted $z\\equiv4$')
base=f0.copy()
def cat(k):
    if k=='hub': return 'isolated $z=6$ hubs'
    if k=='runs3': return 'ribbons'
    if k=='dice': return 'dice'
    s_,t_=info[k]['sigma'],info[k]['tau']; m_=t_/(2*(1+s_))
    if abs(m_-0.2)<1e-6: return 'crystals on $m=1/5$'
    if m_>0.2: return 'crystals with $m>1/5$'
    return 'window crystals'
CATS=['isolated $z=6$ hubs','ribbons','window crystals','crystals on $m=1/5$','crystals with $m>1/5$','dice']
CC=dict(zip(CATS,['#4c72b0','#8da0cb','#66c2a5','#e78ac3','#fc8d62','#b3002d']))
FC={c:np.zeros(len(hs)) for c in CATS}
for k in keys: FC[cat(k)]+=F[k]
for c in CATS:
    if FC[c].max()<1e-3: continue
    ax.fill_between(hs,base,base+FC[c],color=CC[c],lw=0,label=c); base+=FC[c]
print({k:cat(k) for k in keys})
ax.set_xlim(0,0.3); ax.set_ylim(0,1); ax.set_xlabel(r'$h/J$'); ax.set_ylabel('fraction of sites')
fig.legend(*ax.get_legend_handles_labels(),frameon=False,fontsize=7.5,loc='lower center',ncol=4,bbox_to_anchor=(0.5,0.0),handlelength=1.0,columnspacing=1.0)
ax=axs[1]
hh=np.linspace(0,0.3,200); zq=[sq_h(h) for h in hh]
ax.plot(hh,[z[1] for z in zq],'--',color='#2166ac',lw=0.8); ax.plot(hh,[z[0] for z in zq],'--',color='#b2182b',lw=0.8)
ax.plot(hs,Exy,'-',color='#2166ac',lw=1.5,label='in-plane ($xx{+}yy$)')
ax.plot(hs,Ezz,'-',color='#b2182b',lw=1.5,label='normal ($zz$)')
ax.set_xlim(0,0.3); ax.set_xlabel(r'$h/J$'); ax.set_ylabel(r'energy ($J$ per site)')
ax.legend(frameon=False,fontsize=7.5,loc='lower right',bbox_to_anchor=(1.0,0.12),handlelength=1.2)
ax=axs[2]
ax.plot(hs,1e3*(Eex-ESQ),'-',color='#d95f02',lw=1.5,label=r'exchange $-\,e_{\rm sq}$')
ax.plot(hs,1e3*Ez,'-',color='#1b9e77',lw=1.5,label=r'Zeeman $-h\langle S^z\rangle$')
ax.plot(hs,1e3*(Eex+Ez-ESQ),'-',color='k',lw=1.3,label=r'total $-\,e_{\rm sq}$')
ax.axhline(0,color='0.6',lw=0.5,ls=':')
ax.set_xlim(0,0.3); ax.set_xlabel(r'$h/J$'); ax.set_ylabel(r'energy ($10^{-3}J$ per site)')
ax.legend(frameon=False,fontsize=7.5,loc='lower left')
for a in axs[1:]:
    for s in steps: a.axvline(s,color='0.75',lw=0.4,ls=':')
for a,l in zip(axs,'abc'): a.text(-0.3,1.0,f'({l})',transform=a.transAxes,fontsize=10,fontweight='bold',va='bottom')
plt.savefig('fig_path.pdf'); plt.savefig('fig_path.png',dpi=200)
# check the exact step relation  Delta E_ex = h Delta S^z  at each jump
for i in range(1,len(hs)):
    dS=Sz[i]-Sz[i-1]
    if dS>1e-4: print('step h=%.4f  dE_ex=%.5f  h*dSz=%.5f'%(hs[i],Eex[i]-Eex[i-1],hs[i]*dS))
json.dump(dict(h=hs.tolist(),Ezz=Ezz.tolist(),Exy=Exy.tolist(),Sz=Sz.tolist(),f0=f0.tolist(),F={k:v.tolist() for k,v in F.items()}),open('/mnt/user-data/outputs/fieldfig/path_decomp.json','w'))

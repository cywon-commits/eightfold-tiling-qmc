"""Gallery of the window crystals (hub-free z=3/z=5 pair crystals with m<1/5) on the N<=24 ground-state paths.
Each panel: a patch of the tiling, z=3 (majority) and z=5 (minority) sites marked; label gives sigma, m, QMC/estimated cost,
the selection frequency over 500 redraws, and the smallest energy above the lower envelope over the field window."""
import sys, json, numpy as np
sys.path.insert(0,'/mnt/user-data/outputs/manuscript/figure_scripts'); sys.path.insert(0,'/mnt/user-data/outputs/sectors')
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import pathdata as P
from style import *
from matplotlib.patches import Polygon
import phases as PH
from window import phases as allphases, lp
D=np.load('/mnt/user-data/outputs/sectors/path_redraws_N24.npz',allow_pickle=True)
names=D['names']; qk=D['qkey']; S=P.S; T=P.T; mP=P.mP; W=P.W
NP=json.load(open('/mnt/user-data/outputs/fieldfig/newphases.json'))
SV=json.load(open('/mnt/user-data/outputs/sectors/screen_N24.json'))['survivors']
OLD={p['name']:p for p in json.load(open('/mnt/user-data/outputs/sectors/phases.json'))['phases']}
# reduced cost (energy above the lower envelope, J per rhombus) for every phase over h, central energies
AP=allphases('/mnt/user-data/outputs/sectors/compare_N24.json'); sg=np.array([p[1] for p in AP]); ta=np.array([p[2] for p in AP]); ep=np.array([p[3] for p in AP])
hs=np.linspace(0.08,0.20,61); RED=np.array([lp(sg,ta,ep,h)[1] for h in hs])       # (H, nphases)
apname=[p[0] for p in AP]
# collect window crystals
seen=set(); items=[]
for i in range(len(S)):
    if P.cat(i)!=2: continue
    f=W[:,:,i]*(1+S[i])/(1+P.s0); freq=(f>0.05).any(1).mean()
    if freq<0.02: continue
    key=qk[i] or str(names[i])
    if key in seen: continue
    seen.add(key)
    # structure
    if key in NP: v=NP[key]; tiles,Tm,N=v['tiles'],v['T'],v['N']
    elif key in ('p4062','fourfam'):
        tiles,Tm,_=PH.PHASES[key][0](); N=len(tiles)
    else:
        c=[x for x in SV if abs(x['sigma']-S[i])<1e-6 and abs(x['tau']-T[i])<1e-6 and abs(x['eps_big']-D['eps'][i]/ (1.114+1.152*(x['rH']-1)) )<2e-3]
        if not c: c=[x for x in SV if abs(x['sigma']-S[i])<1e-6 and abs(x['tau']-T[i])<1e-6]
        c.sort(key=lambda x:x['N']); x=c[0]; tiles,Tm,N=x['tiles'],x['T'],x['N']
    j=apname.index(str(names[i])) if str(names[i]) in apname else None
    gap=RED[:,j].min() if j is not None else np.nan
    items.append(dict(key=key,N=N,sigma=S[i],tau=T[i],m=mP[i],eps=D['eps'][i],err=D['err'][i],qmc=bool(qk[i]),freq=freq,gap=gap,tiles=tiles,T=Tm))
items.sort(key=lambda d:(round(d['m'],4),-d['freq']))
print(len(items),'window crystals')
from fractions import Fraction as F
nc=8; nr=int(np.ceil(len(items)/nc))
fig,axs=plt.subplots(nr,nc,figsize=(DCOL,DCOL*nr/nc*1.10))
for ax in axs.ravel(): ax.axis('off')
for idx,(ax,it) in enumerate(zip(axs.ravel(),items)):
    it['id']='W%d'%(idx+1)
    T1,T2=[np.array(t) for t in it['T']]; tl=[(np.array(R),a_,b_) for R,a_,b_ in it['tiles']]
    EV=PH.EV if it['key'] in ('p4062','fourfam') else PH.EV8
    p1,p2=T1@EV,T2@EV; Wd=4.0
    K=int(np.ceil(3*Wd/min(np.linalg.norm(p1),np.linalg.norm(p2))))+2
    cell_c=np.mean([(R@EV).mean(0) for R,a_,b_ in tl],0)
    edges={}; polys=[]
    for i1 in range(-K,K+1):
        for i2 in range(-K,K+1):
            sh=i1*T1+i2*T2; ps=sh@EV
            if np.linalg.norm(cell_c+ps)>Wd*1.6+np.linalg.norm(p1)+np.linalg.norm(p2): continue
            for R,a_,b_ in tl:
                C=R+sh
                xy_=C@EV
                if np.abs(xy_.mean(0)-cell_c).max()>Wd+1: continue
                polys.append((xy_,abs(a_-b_)==2))
                for u in range(4):
                    e=tuple(sorted((tuple(C[u]),tuple(C[(u+1)%4])))); edges[e]=1
    for xy_,isq in polys: ax.add_patch(Polygon(xy_,fc='#f2ede3' if isq else '#dfe6ee',ec='0.6',lw=0.25))
    deg={}
    for u,v in edges: deg[u]=deg.get(u,0)+1; deg[v]=deg.get(v,0)+1
    P3=[];P5=[]
    for v,d in deg.items():
        x=np.array(v)@EV
        if np.abs(x-cell_c).max()>Wd-0.2: continue
        if d==3: P3.append(x)
        elif d==5: P5.append(x)
    P3=np.array(P3).reshape(-1,2); P5=np.array(P5).reshape(-1,2)
    ax.scatter(P3[:,0],P3[:,1],s=9,c='#c0392b',zorder=3,lw=0); ax.scatter(P5[:,0],P5[:,1],s=10,c='#1f5fa8',marker='s',zorder=3,lw=0)
    c0=cell_c
    ax.set_aspect('equal'); ax.set_xlim(c0[0]-Wd,c0[0]+Wd); ax.set_ylim(c0[1]-Wd,c0[1]+Wd)
    ax.set_title(it['id'],fontsize=7.5,pad=1.5)
    # z=3 sites per primitive cell (pairs), from the periodic graph
    PH.PHASES['_g']=((lambda it=it:(it['tiles'],it['T'],it['sigma'])),'_g',it['tau'])
    g=PH.graph('_g',2); z=np.bincount(g['B'].ravel(),minlength=g['N']); it['n3']=int((z==3).sum()//4); it['n5']=int((z==5).sum()//4)
fig.text(0.5,0.002,r'red dots: $z=3$ (majority), blue squares: $z=5$ (minority); beige: squares, blue-gray: rhombi.  $\varepsilon$: QMC cost per rhombus ($J$; * = Eq. (15) estimate).  sel: fraction of 500 redraws in which it fills >5% of sites.  gap: smallest energy above the ground-state envelope for $0.08<h/J<0.20$ ($J$ per rhombus).',
         ha='center',va='bottom',fontsize=0.1,alpha=0)
rest=axs.ravel()[len(items):]
if len(rest):
    k=rest[0]; k.set_xlim(0,1); k.set_ylim(0,1)
    k.scatter([0.12],[0.82],s=16,c='#c0392b',lw=0); k.text(0.24,0.82,'$z=3$ (majority)',va='center',fontsize=6.5)
    k.scatter([0.12],[0.62],s=16,c='#1f5fa8',marker='s',lw=0); k.text(0.24,0.62,'$z=5$ (minority)',va='center',fontsize=6.5)
    k.add_patch(plt.Rectangle((0.06,0.36),0.12,0.12,fc='#f2ede3',ec='0.6',lw=0.4)); k.text(0.24,0.42,'square',va='center',fontsize=6.5)
    k.add_patch(Polygon([[0.06,0.14],[0.15,0.14],[0.21,0.24],[0.12,0.24]],fc='#dfe6ee',ec='0.6',lw=0.4)); k.text(0.24,0.19,'rhombus',va='center',fontsize=6.5)
plt.subplots_adjust(left=0.005,right=0.995,top=0.97,bottom=0.005,wspace=0.05,hspace=0.18)
plt.savefig('/mnt/user-data/outputs/manuscript/figs/fig_window_gallery.pdf'); plt.savefig('/mnt/user-data/outputs/manuscript/figs/fig_window_gallery.png',dpi=200)
json.dump([{k:v for k,v in it.items() if k not in ('tiles',)} for it in items],open('/mnt/user-data/outputs/fieldfig/gallery/window_crystals.json','w'),indent=1,default=float)
def frac(x,d=30):
    f=F(x).limit_denominator(d); return '%d'%f.numerator if f.denominator==1 else '%d/%d'%(f.numerator,f.denominator)
rows=[]
for it in items:
    nm=it['key'] if not it['key'].startswith('sec') else '--'
    rows.append('%s & %s & %d & $%s$ & $%s$ & %d & %.4f%s & %.0f & %.1f \\\\'%(it['id'],nm,it['N'],frac(it['sigma']),frac(it['m'],60),it['n3'],it['eps'],'' if it['qmc'] else '$^*$',100*it['freq'],1e4*max(it['gap'],0)))
open('/mnt/user-data/outputs/manuscript/tab_window.tex','w').write('\n'.join(rows)+'\n')
for it in items: assert it['n3']==it['n5'], it['key']; 
print('check m=n3/(2N):',all(abs(it['m']-it['n3']/(2*it['N']))<1e-9 for it in items))
for it in items: print('%-22s N=%-3s sigma=%.3f m=%.4f eps=%.5f%s freq=%.2f gap=%.1e'%(it['key'],it['N'],it['sigma'],it['m'],it['eps'],'' if it['qmc'] else '*',it['freq'],it['gap']))

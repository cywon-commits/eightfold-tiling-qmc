"""Energy balance along the ground-state path (N<=24) with its uncertainty: median and 16-84% band over 500 redraws
of the phase energies (same error model as Fig. 7). Central path (no redraw) dashed in (b),(c)."""
import numpy as np, json, glob
import numpy as np
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
C0=decomp(D['W0'])                                  # central path (no redraw)
F0=np.array([r[0] for r in R]); FC=np.array([r[1] for r in R]); EZ=np.array([r[2] for r in R]); EX=np.array([r[3] for r in R]); SZ=np.array([r[4] for r in R])
EEX=EZ+EX; EZM=-hs[None,:]*SZ; TOT=EEX+EZM
EEX=EZ+EX; EZM=-hs[None,:]*SZ; TOT=EEX+EZM
M=2*SZ                                              # m = 2<S^z>/site
q=lambda A:np.percentile(A,[16,50,84],axis=0)
f0c,FCc,ezc,exc,szc=C0
# phase of largest site fraction among polarized classes, per redraw
DOM=np.argmax(np.concatenate([F0[:,:,None],FC],2),2)   # 0 = canted facets, 1.. = class+1

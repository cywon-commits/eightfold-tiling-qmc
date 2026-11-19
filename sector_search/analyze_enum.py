"""Screen enumerated sector tilings against the existing phase set with the field LP (LSWT level),
then refine survivors (large supercell LSWT + Hartree ratio) and run the calibrated LP."""
import json, glob, sys, os, numpy as np
from scipy.optimize import linprog
from esq_h import e_sq
from crystal import supercell_graph
from lswt import lswt_energy
from hartree import energies
ESQ=-0.6579474209435209; s0=1/np.sqrt(2); e0=e_sq(0); chi=0.065/0.058; rho=2-np.sqrt(2)
e1=ESQ; d_sq=(-e1-0.5)/2; e2=-2*d_sq**2
OLD=json.load(open('/home/claude/fieldfig/phases.json'))['phases']
HS=np.linspace(0,0.32,321)
def lp(sig,tau,eps,h):
    d=(e0-e_sq(h))*chi; c=eps+d*(1+sig)-h*tau/4
    r=linprog(c,A_ub=np.stack([np.ones_like(sig),sig]),b_ub=[1,s0],bounds=[(0,None)]*len(sig),method='highs')
    y=r.ineqlin.marginals            # <=0 duals
    red=c-y[0]-y[1]*sig              # reduced costs (>=0)
    return r,red
def load(Nmax):
    recs=[]; info=dict(sectors=0,exhausted=0,states=0)
    for N in range(1,Nmax+1):
        fns=[f'enum_N{N}.json'] if os.path.exists(f'enum_N{N}.json') else sorted(glob.glob(f'enum_N{N}_part*.json'))
        for s in [s for fn in fns for s in json.load(open(fn))]:
            info['sectors']+=1; info['exhausted']+=s['done']; info['states']+=s['states']
            for r in s['recs']:
                if r['eps'] is None: continue
                r['N']=N; r['T']=s['T']; recs.append(r)
    return recs,info
def best_per_comp(recs):
    b={}
    for r in recs:
        k=(round(r['sigma'],6),round(r['tau'],6))
        if k not in b or r['eps']<b[k]['eps']: b[k]=r
    return list(b.values())
def screen(recs,margin):
    """records whose LSWT reduced cost comes within `margin` of zero somewhere on the field grid (old phases with LSWT eps,
    rescaled by a common factor 1.2 so that both sets are compared on the same footing)"""
    sig=np.array([p['sigma'] for p in OLD]+[r['sigma'] for r in recs]); tau=np.array([p['tau'] for p in OLD]+[r['tau'] for r in recs])
    eps=1.2*np.array([p['eps'] for p in OLD]+[r['eps'] for r in recs]); n0=len(OLD)
    minred=np.full(len(recs),np.inf); active=set()
    for h in HS:
        r,red=lp(sig,tau,eps,h); minred=np.minimum(minred,red[n0:])
        for i in np.nonzero(r.x[n0:]>1e-7)[0]: active.add(i)
    return minred,active
def refine(r,target=600):
    tiles=[(np.array(R),a,b) for R,a,b in r['tiles']]; T=[np.array(t) for t in r['T']]
    L=2
    while r['N']*L*L<target: L+=2
    N,B=supercell_graph(tiles,T[0],T[1],L); R=sum(1 for _,a,b in tiles if abs(a-b)!=2)*L*L
    E,S=lswt_energy(N,B); E1,E2,_=energies(N,B); dE1=(E1-N*e1)/R; dE2=(E2-N*e2)/R
    rH=1+dE2/dE1; return dict(eps_big=(E-N*ESQ)/R,tau_big=2*(2*S)/R,L_big=L,N_big=N,rH=rH,ratio=1.114+1.152*(rH-1))
def refine_lswt(r,target):
    tiles=[(np.array(R),a,b) for R,a,b in r['tiles']]; T=[np.array(t) for t in r['T']]
    L=2
    while r['N']*L*L<target: L+=2
    N,B=supercell_graph(tiles,T[0],T[1],L); R=sum(1 for _,a,b in tiles if abs(a-b)!=2)*L*L
    E,S=lswt_energy(N,B); return (E-N*ESQ)/R, 2*(2*S)/R, L
def old_calibrated():
    import io,contextlib
    src=open('analyze_hartree.py').read(); g={'__file__':__file__}
    with contextlib.redirect_stdout(io.StringIO()): exec(src[:src.index("s0=1/np.sqrt(2)")],g)
    return [(p['name'],p['sigma'],p['tau'],p['eps'],g['pred'](p['name'])[0] or 1.2) for p in OLD]
def staircase(P,label):
    sig=np.array([p[1] for p in P]); tau=np.array([p[2] for p in P]); eps=np.array([p[3]*p[4] for p in P])
    prev=None; steps=[]
    for h in np.linspace(0,0.32,3201):
        r,red=lp(sig,tau,eps,h); tt=(r.x*tau).sum()
        if prev is None or abs(tt-prev)>1e-6:
            act={P[i][0]:round(float(r.x[i]),3) for i in range(len(P)) if r.x[i]>1e-6}; steps.append((round(float(h),4),round(float(rho*tt/2),4),act)); prev=tt
    print(f'--- {label}'); [print('  h=%.4f m=%.4f %s'%s) for s in steps]; return steps
if __name__=='__main__':
    Nmax=int(sys.argv[1]); margin=float(sys.argv[2]) if len(sys.argv)>2 else 0.001
    recs,info=load(Nmax)
    nz4=sum(1 for r in recs if sum(r['z'])==r['z'][4]); recs=[r for r in recs if sum(r['z'])!=r['z'][4]]
    comps={}
    for r in recs: comps.setdefault((round(r['sigma'],6),round(r['tau'],6)),[]).append(r)
    print(f"N<={Nmax}: sectors {info['sectors']}, exhausted {info['exhausted']} ({info['exhausted']/info['sectors']:.3f}), states {info['states']}, z4-only {nz4}, other {len(recs)}, compositions {len(comps)}",flush=True)
    comp=[]
    for k,L in comps.items():
        L.sort(key=lambda r:r['eps']); best=None
        for r in L[:5]:
            e,t,Lb=refine_lswt(r,600)
            if best is None or e<best[0]: best=(e,r)
        r=dict(best[1]); r['eps']=best[0]; comp.append(r)
    minred,active=screen(comp,margin)
    surv=[comp[i] for i in range(len(comp)) if minred[i]<margin]
    print(f'screen: {len(comp)} compositions; within margin {margin} of the LSWT lower envelope: {len(surv)}',flush=True)
    out=[]
    for r in surv:
        x=refine(r,1500); x.update(sigma=r['sigma'],tau=r['tau'],eps600=r['eps'],N=r['N'],T=r['T'],tiles=r['tiles'],z=r['z'])
        out.append(x); print(f"  N={r['N']:2d} T={r['T']} sig={r['sigma']:.3f} tau={r['tau']:.3f} eps600={r['eps']:.5f} eps1500={x['eps_big']:.5f} rH={x['rH']:.3f} ratio={x['ratio']:.3f} z3..7={r['z'][3:8]}",flush=True)
    OLDc=old_calibrated()
    st0=staircase(OLDc,'old phase set (QMC/Hartree calibrated)')
    NEW=[(f"sec N{x['N']} s={x['sigma']:.3f} t={x['tau']:.3f}",x['sigma'],x['tau'],x['eps_big'],x['ratio']) for x in out]
    st1=staircase(OLDc+NEW,'old + all-sector survivors')
    json.dump(dict(info=info,nz4=nz4,ncomp=len(comp),survivors=out,steps_old=st0,steps_new=st1),open(f'screen_N{Nmax}.json','w'))

"""Seeded QMC energies of all path phases -> eps per rhombus; resampled field staircase (presence probabilities).
LP with two resources (rhombi <= 1, squares-in-phases <= s0) solved exactly by enumerating basic solutions."""
import json, glob, sys, numpy as np, itertools
from esq_h import e_sq
from analyze_enum import old_calibrated, lp
E_SQ_TL=-0.669437; E_SQ_ERR=0.000005
s0=1/np.sqrt(2); rho=2-np.sqrt(2); e0=e_sq(0); chi=0.065/0.058
KEY={'small(3, 0, 3, 3) t=0.111':'hub','runs3 d5':'runs3','small(4, 0, 6, 2) t=0.400':'p4062','small(6, 0, 4, 2) t=0.400':'p4062',
     'small(3, 0, 3, 3) t=0.333':'z5carrier','4-1-4-1 t=0.500':'fourfam','small(3, 0, 3, 1) t=1.000':'z5crystal','6-0-6-2 t=1.000':'z5crystal',
     '4-2-2-2 t=0.500':'p4222','2-0-6-4 t=0.500':'kappa89','dice':'dice','small(2, 0, 2, 2) t=1.000':'dice'}
OLDKEY={'hub':'small(3, 0, 3, 3) t=0.111','runs3':'runs3 d5','p4062':'small(4, 0, 6, 2) t=0.400','z5carrier':'small(3, 0, 3, 3) t=0.333',
 'fourfam':'4-1-4-1 t=0.500','z5crystal':'small(3, 0, 3, 1) t=1.000','p4222':'4-2-2-2 t=0.500','kappa89':'2-0-6-4 t=0.500','dice':'dice'}
LC=json.load(open('/mnt/user-data/outputs/sectors/lswt_qmc_cells.json'))
EINF={k:v['eps'] for k,v in json.load(open('/mnt/user-data/outputs/fieldfig/newphases.json')).items()}
EINF.update({k:[p['eps'] for p in json.load(open('/mnt/user-data/outputs/sectors/phases.json'))['phases'] if p['name']==n][0] for k,n in OLDKEY.items()})
def qmc_table(D='/mnt/user-data/outputs/fieldfig/runs_seeds/'):
    runs={}
    for f in glob.glob(D+'*_s*.json'):
        d=json.load(open(f)); runs.setdefault((d['key'],d['L']),[]).append(d)
    for k,L in [(f.split('/')[-1].rsplit('_',1)[0],int(f.split('_')[-1][:-5])) for f in glob.glob('/mnt/user-data/outputs/fieldfig/runs/*.json')]:
        if k=='sq16': continue
        d=json.load(open(f'/mnt/user-data/outputs/fieldfig/runs/{k}_{L}.json')); d['seed']=1
        if (k,L) in runs and any(x['seed']==1 for x in runs[(k,L)]): continue
        runs.setdefault((k,L),[]).append(d)
    tab={}
    for (k,L),rs in runs.items():
        N,R=rs[0]['N'],rs[0]['R']; e=np.array([r['E']/N for r in rs]); er=np.array([r['E_err']/N for r in rs])
        m=e.mean(); stat=max(er.mean()/np.sqrt(len(e)), e.std(ddof=1)/np.sqrt(len(e)) if len(e)>1 else 0)
        eps=(m-E_SQ_TL)*N/R; err=np.hypot(stat,E_SQ_ERR)*N/R
        # same-cell ratio: eps_QMC(L)/eps_LSWT(L) times the converged LSWT cost (removes most of the finite-size shift)
        el=LC.get(f'{k}|{L}'); einf=EINF.get(k)
        fac=(einf/el) if (el and einf and el>0) else 1.0
        tab.setdefault(k,{})[L]=dict(N=N,R=R,seeds=len(e),e_site=m,e_err=stat,eps_direct=eps,eps=eps*fac,eps_err=err*fac,fs_factor=fac,
                                     chi2=float(((e-m)**2/er**2).sum()/max(len(e)-1,1)))
    # systematic finite-size uncertainty: change of the corrected estimate between the two largest cells,
    # or, with one cell only, the size of the LSWT same-cell correction itself
    for k,d in tab.items():
        Ls=sorted(d); top=d[Ls[-1]]
        fs=abs(top['eps']-d[Ls[-2]]['eps']) if len(Ls)>1 else abs(top['fs_factor']-1)*top['eps']
        top['fs_err']=fs; top['eps_err_stat']=top['eps_err']; top['eps_err']=float(np.hypot(top['eps_err'],fs))
    return tab

def extend_key(P):
    """map phase names to measured QMC keys, including supercell duplicates (same sigma, tau, LSWT eps)"""
    NPh=json.load(open('/mnt/user-data/outputs/fieldfig/newphases.json'))
    for kk,v in NPh.items(): KEY[f"sec N{v['N']} s={v['sigma']:.3f} t={v['tau']:.3f}"]=kk
    for p in P:
        if p[0] in KEY: continue
        for kk,v in NPh.items():
            if abs(v['sigma']-p[1])<1e-6 and abs(v['tau']-p[2])<1e-6 and abs(v['eps']-p[3])<3e-4: KEY[p[0]]=kk; break
def solve_all(c,sig,tau):
    """c: (H,n) costs. returns m(h) of the optimal basic solution."""
    H,n=c.shape; best=np.zeros(H); bm=np.zeros(H)
    xs=np.minimum(1,np.where(sig>0,s0/np.maximum(sig,1e-12),1))
    for i in range(n):
        obj=c[:,i]*xs[i]; better=obj<best-1e-12; best=np.where(better,obj,best); bm=np.where(better,xs[i]*tau[i],bm)
    for i,j in itertools.combinations(range(n),2):
        if abs(sig[i]-sig[j])<1e-12: continue
        xj=(s0-sig[i])/(sig[j]-sig[i]); xi=1-xj
        if xi<0 or xj<0: continue
        obj=c[:,i]*xi+c[:,j]*xj; better=obj<best-1e-12; best=np.where(better,obj,best); bm=np.where(better,xi*tau[i]+xj*tau[j],bm)
    return rho*bm/2
def plateaus(hs,m):
    """list of (m, h_start, h_end) of constant-m segments"""
    out=[]; s=0
    for k in range(1,len(m)+1):
        if k==len(m) or abs(m[k]-m[s])>1e-9:
            out.append((round(float(m[s]),3),float(hs[s]),float(hs[k-1]))); s=k
    return out
if __name__=='__main__':
    tab=qmc_table(); print(json.dumps({k:{L:{kk:(round(v,6) if isinstance(v,float) else v) for kk,v in d.items()} for L,d in t.items()} for k,t in tab.items()},indent=0))
    P=old_calibrated()
    CMP=sys.argv[2] if len(sys.argv)>2 else None
    if CMP:
        C=json.load(open(CMP)); P=P+[tuple(n) for n in C['new']]
        extend_key(P)
    names=[p[0] for p in P]; sig=np.array([p[1] for p in P]); tau=np.array([p[2] for p in P])
    eps_l=np.array([p[3] for p in P]); ratio=np.array([p[4] for p in P])
    eps=eps_l*ratio; err=0.018*eps_l          # Hartree-predicted phases: calibration rms on the ratio
    isq=np.zeros(len(P),bool)
    for i,nm in enumerate(names):
        k=KEY.get(nm)
        if k and k in tab:
            L=max(tab[k]); eps[i]=tab[k][L]['eps']; err[i]=tab[k][L]['eps_err']; isq[i]=True
    hs=np.linspace(0,0.32,641); d=np.array([(e0-e_sq(h))*chi for h in hs])
    def costs(e): return e[None,:]+d[:,None]*(1+sig[None,:])-hs[:,None]*tau[None,:]/4
    # prune: keep phases within 0.004 of the envelope (reduced cost) somewhere, plus all QMC phases
    keep=set(np.nonzero(isq)[0])
    for h in hs[::8]:
        r,red=lp(sig,tau,eps,h); keep|=set(np.nonzero(red<0.004)[0])
    keep=sorted(keep); print('phases kept for resampling:',len(keep),'of',len(P))
    S,T_,EP,ER=sig[keep],tau[keep],eps[keep],err[keep]
    m0=solve_all(EP[None,:]+d[:,None]*(1+S[None,:])-hs[:,None]*T_[None,:]/4,S,T_)
    pl0=plateaus(hs,m0); print('central staircase:'); [print('  m=%.3f  h=%.4f..%.4f'%p) for p in pl0]
    rng=np.random.default_rng(1); NS=int(sys.argv[1]) if len(sys.argv)>1 else 4000
    pres={}; edges={}
    for s in range(NS):
        e=EP+ER*rng.standard_normal(len(EP))
        m=solve_all(e[None,:]+d[:,None]*(1+S[None,:])-hs[:,None]*T_[None,:]/4,S,T_)
        seg={}
        for mm,a,b in plateaus(hs,m):
            seg[mm]=(min(a,seg[mm][0]),max(b,seg[mm][1])) if mm in seg else (a,b)
        for mm,(a,b) in seg.items(): pres.setdefault(mm,[]).append((a,b))
    print(f'presence over {NS} resamples (plateau m: fraction, median onset, median width):')
    summ=[]
    for mm in sorted(pres):
        v=np.array(pres[mm]); f=len(v)/NS
        if f<0.01: continue
        summ.append(dict(m=mm,frac=f,onset=float(np.median(v[:,0])),onset16=float(np.percentile(v[:,0],16)),onset84=float(np.percentile(v[:,0],84)),width=float(np.median(v[:,1]-v[:,0]))))
        print('  m=%.3f  %.3f  h=%.4f [%.4f,%.4f]  width %.4f'%(mm,f,summ[-1]['onset'],summ[-1]['onset16'],summ[-1]['onset84'],summ[-1]['width']))
    json.dump(dict(qmc=tab,central=pl0,presence=summ,kept=[names[i] for i in keep]),open(sys.argv[3] if len(sys.argv)>3 else '/mnt/user-data/outputs/fieldfig/resample.json','w'),indent=1,default=float)

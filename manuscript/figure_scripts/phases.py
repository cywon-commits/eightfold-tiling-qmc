"""Load the homogeneous phases used in the paper and build supercell graphs with vertex coordinates."""
import numpy as np, json
D='/mnt/user-data/outputs/'
C=json.load(open(D+'dense/candidates.json'))
EV=np.array([[0,1],[1,-1],[1,0],[1,1]],float); EV/=np.linalg.norm(EV,axis=1)[:,None]
def small(counts,tau): c=[c for c in C if c['counts']==counts and abs(c['tau']-tau)<1e-3][0]; return c['tiles'],c['T'],c['sigma']
def big(name,tau):
    d=json.load(open(D+f'dense/big_{name}.json')); v=[v for k,v in d.items() if abs(float(k)-tau)<1e-3][0]; return v['tiles'],v['T'],v['sigma']
def z5():
    d=json.load(open(D+'field_hull/cryst_a3_n1.json'))[0]; return d[2],d[3],1.5
PHASES={  # key: (loader, label, tau)
 'hub':      (lambda: small('(3, 0, 3, 3)',1/9), r'isolated $z=6$ hubs', 1/9),
 'z5carrier':(lambda: small('(3, 0, 3, 3)',1/3), r'$z=5$ carriers ($\sigma=1/2$)', 1/3),
 'p4062':    (lambda: small('(4, 0, 6, 2)',0.4), r'$(4,0,6,2)$', 0.4),
 'fourfam':  (lambda: big('4-1-4-1',0.5), r'four-family $(4,1,4,1)$', 0.5),
 'p4222':    (lambda: big('4-2-2-2',0.5), r'four-family $(4,2,2,2)$', 0.5),
 'kappa89':  (lambda: big('2-0-6-4',0.5), r'$\kappa_B=8/9$', 0.5),
 'z5crystal':(z5, r'$z=5$ crystal', 1.0),
 'dice':     (lambda: small('(2, 0, 2, 2)',1.0), 'dice', 1.0),
}
_NP=json.load(open(D+'fieldfig/newphases.json'))
EV8=np.array([[np.cos(j*np.pi/4),np.sin(j*np.pi/4)] for j in range(4)])   # star used by the sector search
_EVK={}
for _k,_v in _NP.items():
    PHASES[_k]=((lambda v=_v:(v['tiles'],v['T'],v['sigma'])),_k,_v['tau']); _EVK[_k]=EV8
def ev(key):
    """projection star for drawing: sector-search phases use e_j=(cos j pi/4, sin j pi/4); earlier phases use EV"""
    return _EVK.get(key,EV)
def graph(key,L):
    tiles,T,sig=PHASES[key][0](); T1,T2=[np.array(t) for t in T]; tl=[(np.array(R),a,b) for R,a,b in tiles]
    A=np.stack([L*T1,L*T2],1).astype(float); pinv=np.linalg.pinv(A)
    def canon(v):
        ab=np.floor(v@pinv.T+1e-7).astype(np.int64); return v-ab[:,:1]*(L*T1)-ab[:,1:]*(L*T2)
    allR=np.concatenate([R+i*T1+j*T2 for i in range(L) for j in range(L) for R,a,b in tl])
    kinds=[(a,b) for i in range(L) for j in range(L) for R,a,b in tl]
    Cn=canon(allR); vid={}; ids=np.array([vid.setdefault(tuple(c),len(vid)) for c in Cn]).reshape(-1,4)
    E=set()
    for q in ids:
        for s in range(4): p,r=q[s],q[(s+1)%4]; E.add((min(p,r),max(p,r)))
    keys=list(vid); X=np.array(keys)@EV
    R=sum(1 for _,a,b in tl if abs(a-b)!=2)*L*L
    return dict(N=len(vid),B=np.array(sorted(E)),X=X,lift=np.array(keys),tiles=allR.reshape(-1,4,4),kinds=kinds,R=R,sigma=sig,ids=ids)

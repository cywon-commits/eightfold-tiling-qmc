"""General de Bruijn builder for a winding sector (T1,T2) and flip-BFS over all tilings in that sector."""
import numpy as np, itertools, math
from collections import deque
from flips import Torus
from sectors import pl, PAIRS
E4=np.eye(4,dtype=np.int64)
ANG=np.pi/4
def make_canon(T1,T2):
    A=np.stack([T1,T2],1).astype(float); pinv=np.linalg.pinv(A)
    def canon(v):
        v=np.atleast_2d(v); ab=np.floor(v@pinv.T+1e-7).astype(np.int64)
        return v-ab[:,:1]*T1-ab[:,1:]*T2
    return canon
def build(T,rng):
    T1=np.array(T[0],dtype=np.int64); T2=np.array(T[1],dtype=np.int64)
    phi=rng.random(4)
    def f(P): return T1*P[0]+T2*P[1]-phi
    tiles=[]
    for j,k in PAIRS:
        M=np.array([[T1[j],T2[j]],[T1[k],T2[k]]],float); d=round(np.linalg.det(M))
        if d==0: continue
        corners=np.array([[0,0],[1,0],[0,1],[1,1]],float)
        fj=corners@M[0]-phi[j]; fk=corners@M[1]-phi[k]
        cnt=0
        for m in range(int(np.floor(fj.min()))-1,int(np.ceil(fj.max()))+2):
            for n in range(int(np.floor(fk.min()))-1,int(np.ceil(fk.max()))+2):
                P=np.linalg.solve(M,[m+phi[j],n+phi[k]])
                if not (0<=P[0]<1 and 0<=P[1]<1): continue
                K=np.floor(f(P)).astype(np.int64); K[j]=m-1; K[k]=n-1
                tiles.append((np.array([K,K+E4[j],K+E4[j]+E4[k],K+E4[k]]),j,k)); cnt+=1
        assert cnt==abs(d),(j,k,cnt,d)
    o=Torus.__new__(Torus); o.canon=make_canon(T1,T2); o.tiles=tiles; o.T=[T1,T2]; o.rebuild()
    return o
def valid(o):
    """edges = 2N and every vertex angle sum = 2 pi"""
    if o.N!=len(o.tiles): return False
    ang={v:0.0 for v in o.vlist}
    for cs,(R,a,b) in zip(o.ctiles,o.tiles):
        th=(b-a)*ANG
        for i,c in enumerate(cs): ang[c]+=th if i%2==0 else np.pi-th
    return all(abs(x-2*np.pi)<1e-9 for x in ang.values())
def key(T):
    best=None
    allR=np.stack([R for R,a,b in T.tiles]); ab=[(a,b) for R,a,b in T.tiles]
    for v in T.vlist:
        C=T.canon((allR-np.array(v)).reshape(-1,4)).astype(np.int64).reshape(allR.shape)
        k=tuple(sorted((tuple(map(tuple,sorted(c.tolist()))),a,b) for c,(a,b) in zip(C,ab)))
        if best is None or k<best: best=k
    return best
def bfs(T,cap,seed=0):
    rng=np.random.default_rng(seed); o=build(T,rng); assert valid(o)
    seen={key(o):o}; q=deque([o])
    while q:
        x=q.popleft()
        for v in x.z3():
            y=x.copy(); y.T=x.T
            if len({ti for ti,i in x.vt[v]})!=3: continue      # degenerate on tiny tori
            try: y.flip(v)
            except AssertionError: continue
            k=key(y)
            if k not in seen:
                if len(seen)>=cap: return list(seen.values()),False
                seen[k]=y; q.append(y)
    return list(seen.values()),True

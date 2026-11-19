"""Torus rhombus tiling with hexagon flips (z=3 vertex v -> v + s_a e_a + s_b e_b + s_c e_c)."""
import numpy as np, itertools, collections
from grid_exact import tiling
from lswt import lswt_energy
E4=np.eye(4,dtype=np.int64)
class Torus:
    def __init__(self,pos):
        N,B,z,info=tiling(pos); self.canon=info['canon']
        self.tiles=[]                      # list of (raw corners 4x4 int array, a, b)
        for K,a,b in info['raw']:
            self.tiles.append((np.array([K,K+E4[a],K+E4[a]+E4[b],K+E4[b]]),a,b))
        self.rebuild()
    def C(self,x): return tuple(self.canon(x)[0].astype(np.int64))
    def rebuild(self):
        self.vt=collections.defaultdict(list); self.ctiles=[]
        allR=np.concatenate([R for R,a,b in self.tiles]); allC=self.canon(allR).astype(np.int64)
        for ti in range(len(self.tiles)):
            cs=[tuple(x) for x in allC[4*ti:4*ti+4]]; self.ctiles.append(cs)
            for i,c in enumerate(cs): self.vt[c].append((ti,i))
        self.vlist=list(self.vt); self.vid={v:i for i,v in enumerate(self.vlist)}
        E=set()
        for cs in self.ctiles:
            for s in range(4):
                p,q=self.vid[cs[s]],self.vid[cs[(s+1)%4]]; E.add((min(p,q),max(p,q)))
        self.B=np.array(sorted(E)); self.N=len(self.vlist)
        self.z=np.bincount(self.B.ravel(),minlength=self.N)
        par=np.array([sum(v)%2 for v in self.vlist]); self.M=int((par==0).sum()-(par==1).sum())
    def z3(self): return [v for v in self.vlist if len(self.vt[v])==3]
    def flip(self,v):
        vr=np.array(v,dtype=np.int64); new=[]; dirs={}
        for ti,i in self.vt[v]:
            R,a,b=self.tiles[ti]; Rs=R+(vr-R[i])
            for r in Rs:
                d=r-vr
                if np.abs(d).sum()==1: k=int(np.nonzero(d)[0][0]); dirs[k]=int(d[k])
        assert len(dirs)==3
        vp=vr+sum(s*E4[k] for k,s in dirs.items())
        rm=set(ti for ti,i in self.vt[v])
        for a,b in itertools.combinations(sorted(dirs),2):
            sa,sb=dirs[a],dirs[b]
            new.append((np.array([vp,vp-sa*E4[a],vp-sa*E4[a]-sb*E4[b],vp-sb*E4[b]]),a,b))
        self.tiles=[t for ti,t in enumerate(self.tiles) if ti not in rm]+new
        self.rebuild()
    def energy(self): return lswt_energy(self.N,self.B)[0]
    def copy(self):
        o=Torus.__new__(Torus); o.canon=self.canon; o.tiles=[(R.copy(),a,b) for R,a,b in self.tiles]; o.rebuild(); return o

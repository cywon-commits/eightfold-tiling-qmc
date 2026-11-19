"""Periodic tilings: search with flips on small tori, then LSWT on L x L supercells."""
import numpy as np, itertools, collections, sys
from grid_exact import tiling
from flips import Torus
from lswt import lswt_energy
ESQ=-0.6579474209435209
def make_torus(pos):
    T=Torus(pos); N,B,z,info=tiling(pos); T.T=[np.array(t,dtype=np.int64) for t in info['T']]; return T
def supercell_graph(tiles,T1,T2,L):
    A=np.stack([L*T1,L*T2],1).astype(float); pinv=np.linalg.pinv(A)
    def canon(v):
        ab=np.floor(v@pinv.T+1e-7).astype(np.int64); return v-ab[:,:1]*(L*T1)-ab[:,1:]*(L*T2)
    allR=[]
    for i in range(L):
        for j in range(L):
            for R,a,b in tiles: allR.append(R+i*T1+j*T2)
    allR=np.concatenate(allR); C=canon(allR)
    vid={}; ids=np.array([vid.setdefault(tuple(c),len(vid)) for c in C]).reshape(-1,4)
    E=set()
    for q in ids:
        for s in range(4):
            p,r=q[s],q[(s+1)%4]; E.add((min(p,r),max(p,r)))
    return len(vid),np.array(sorted(E))
def stats(T):
    par=np.array([sum(v)%2 for v in T.vlist]); maj=0 if T.M>=0 else 1
    zA=np.bincount(T.z[par==maj],minlength=9)[3:]; zB=np.bincount(T.z[par!=maj],minlength=9)[3:]
    R=sum(1 for _,a,b in T.tiles if abs(a-b)!=2); return zA,zB,R
def evaluate(T,Ls=(2,3,4)):
    out=[]
    for L in Ls:
        N,B=supercell_graph(T.tiles,T.T[0],T.T[1],L); E,S=lswt_energy(N,B); out.append((L,N,E/N))
    return out

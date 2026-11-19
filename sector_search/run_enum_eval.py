"""Exhaustive (capped) flip enumeration over all D8-reduced winding sectors with N in [N0,N1].
usage: python3 run_enum.py N0 N1 CAP"""
import json, sys, time, numpy as np
from build import bfs
from crystal import supercell_graph
from lswt import lswt_energy
ESQ=-0.6579474209435209
def evaluate(o,N):
    L=2
    while N*L*L<144: L+=2
    n,B=supercell_graph(o.tiles,o.T[0],o.T[1],L)
    if len(B)!=2*n: return None
    E,S=lswt_energy(n,B)
    R=sum(1 for _,a,b in o.tiles if abs(a-b)!=2)*L*L; Sq=len(o.tiles)*L*L-R
    if R==0: return dict(eps=None,sq_only=True,e_site=E/n)
    z=np.bincount(np.bincount(B.ravel(),minlength=n),minlength=9).tolist()
    return dict(eps=(E-n*ESQ)/R,tau=2*(2*S)/R,sigma=Sq/R,L=L,z=z)

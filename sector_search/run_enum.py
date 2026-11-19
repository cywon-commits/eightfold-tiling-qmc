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
N0,N1,CAP=map(int,sys.argv[1:4])
PART=tuple(map(int,sys.argv[4].split('/'))) if len(sys.argv)>4 else None
S=json.load(open('sectors_N24.json'))
for N in range(N0,N1+1):
    t=time.time(); out=[]; nd=0; ns=0
    SN=[r for r in S if r['N']==N]
    if PART: SN=SN[PART[0]::PART[1]]
    for r in SN:
        st,done=bfs(r['T'],CAP); nd+=done; ns+=len(st)
        recs=[]
        for o in st:
            ev=evaluate(o,N)
            if ev is None or ev.get('sq_only'): continue
            ev['tiles']=[(R.tolist(),int(a),int(b)) for R,a,b in o.tiles]; recs.append(ev)
        out.append(dict(T=r['T'],p=r['p'],done=done,states=len(st),recs=recs))
    json.dump(out,open(f'enum_N{N}.json' if not PART else f'enum_N{N}_part{PART[0]}of{PART[1]}.json','w'))
    print(f'N={N}: sectors={len(out)} exhausted={nd} states={ns} {time.time()-t:.0f}s',flush=True)

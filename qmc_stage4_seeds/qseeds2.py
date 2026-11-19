"""Seeded QMC, priority queue after the all-sector search (new path phases first). usage: WORKER NWORKERS"""
import json, os, sys, time, numpy as np
from phases import graph
from ribbon import build
from lswt import signs
from crystal import supercell_graph
from sse_split import simulate
OUT='/mnt/user-data/outputs/fieldfig/runs_seeds/'
NP=json.load(open('/mnt/user-data/outputs/fieldfig/newphases.json'))
def Lfor(n,target):
    L=2
    while n*L*L<target: L+=2
    return L
new=[(k,Lfor(v['N'],450)) for k,v in sorted(NP.items())]
core=[('hub',6),('z5crystal',6),('dice',6),('fourfam',4),('runs3',30)]
old=[('z5carrier',4),('p4062',2),('kappa89',2),('p4222',2)]
jobs=[(k,L,s) for s in (1,2,3) for k,L in new]+[(k,L,s) for s in (1,2,3) for k,L in core]+[(k,L,s) for s in (1,2,3) for k,L in old]
w,nw=int(sys.argv[1]),int(sys.argv[2])
for i,(key,L,seed) in enumerate(jobs):
    if i%nw!=w: continue
    fn=OUT+f'{key}_{L}_s{seed}.json'
    if os.path.exists(fn) or (seed==1 and os.path.exists(f'/mnt/user-data/outputs/fieldfig/runs/{key}_{L}.json')): continue
    t=time.time()
    if key in NP:
        v=NP[key]; tiles=[(np.array(R),a,b) for R,a,b in v['tiles']]; T=[np.array(x) for x in v['T']]
        N,B=supercell_graph(tiles,T[0],T[1],L); P=signs(N,B); R=sum(1 for _,a,b in tiles if abs(a-b)!=2)*L*L
        extra=dict(R=R,sigma=v['sigma'])
    elif key=='runs3':
        N,B=build(L,[((i5*5,0),'HHHVVV'*(L//3)) for i5 in range(L//5)]); B=np.array(B); P=signs(N,B)
        extra=dict(R=2*L*(L//5),sigma=2.5)
    else:
        g=graph(key,L); N=g['N']; B=g['B']; P=np.array([1 if sum(v)%2==0 else -1 for v in g['lift']])
        extra=dict(R=g['R'],sigma=g['sigma'])
    r=simulate(N,B,300.0,0.0,1000,40,75,seed,verbose=False,fixM=0.5*abs(np.sum(P)))
    r.update(key=key,L=L,wall=round(time.time()-t,1),**extra)
    for k in ('bins_E','bins_Ezz'): r[k]=list(map(float,r[k])) if k in r else None
    r.pop('Sz_local',None); r.pop('Sz_local_err',None)
    json.dump(r,open(fn,'w'))
    print(f"{key} L={L} s={seed} N={N}: e={r['E_J']/N:.6f}±{r['E_err']/N:.6f} {r['wall']}s",flush=True)

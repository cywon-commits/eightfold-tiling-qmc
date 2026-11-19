"""Seeded QMC of all ground-state-path phases (Lieb top sector, h=0; E(h)=E0-hM exactly).
usage: python3 qseeds.py WORKER NWORKERS"""
import json, os, sys, time, numpy as np
from phases import graph
from ribbon import build
from lswt import signs
from sse_split import simulate
OUT='/mnt/user-data/outputs/fieldfig/runs_seeds/'
big=[('hub',6),('z5carrier',6),('p4062',4),('kappa89',4),('p4222',4),('fourfam',4),('z5crystal',6),('dice',6),('runs3',30)]
small=[('hub',4),('z5carrier',4),('p4062',2),('kappa89',2),('p4222',2),('fourfam',2),('z5crystal',4),('dice',4),('runs3',15)]
jobs=[(k,L,s) for s in (1,2,3) for (k,L) in big]+[(k,L,s) for s in (2,3) for (k,L) in small]
w,nw=int(sys.argv[1]),int(sys.argv[2])
for i,(key,L,seed) in enumerate(jobs):
    if i%nw!=w: continue
    fn=OUT+f'{key}_{L}_s{seed}.json'
    if os.path.exists(fn): continue
    t=time.time()
    if key=='runs3':
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

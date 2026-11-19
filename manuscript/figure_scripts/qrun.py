import json, os, sys, time, numpy as np
from phases import graph
from sse_split import simulate
OUT='/mnt/user-data/outputs/fieldfig/runs/'
jobs=[('dice',4,4000),('z5crystal',4,4000),('fourfam',2,6000),('p4222',2,6000),('kappa89',2,6000),('p4062',2,6000),('hub',4,3000),('z5carrier',4,3000)]
jobs+=[('sq16',M,4000) for M in (0,2,4,6)]
for key,L,sw in jobs:
    fn=OUT+f'{key}_{L}.json'
    if os.path.exists(fn): continue
    t=time.time()
    if key=='sq16':
        Ls=16; B=[]
        for x in range(Ls):
            for y in range(Ls):
                i=x*Ls+y; B+=[(i,((x+1)%Ls)*Ls+y),(i,x*Ls+(y+1)%Ls)]
        N=Ls*Ls; fixM=L; extra={}
    else:
        g=graph(key,L); N=g['N']; B=g['B']; P=np.array([1 if sum(v)%2==0 else -1 for v in g['lift']])
        fixM=0.5*abs(P.sum()); extra=dict(R=g['R'],sigma=g['sigma'])
    r=simulate(N,B,300.0,0.0,1000,40,sw//40,1,verbose=False,fixM=fixM)
    r.update(key=key,L=L,wall=round(time.time()-t,1),**extra); r.pop('bins_E'); r.pop('bins_Ezz')
    json.dump(r,open(fn,'w'))
    print(f"{key} L={L} N={N} M={fixM}: e={r['E_J']/N:.5f} ezz={r['E_zz']/N:.5f}±{r['E_zz_err']/N:.5f} exy={r['E_xy']/N:.5f} {r['wall']}s",flush=True)

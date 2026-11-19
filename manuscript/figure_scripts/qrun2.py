import json, os, time, numpy as np
from ribbon import build
from lswt import signs
from sse_split import simulate
OUT='/mnt/user-data/outputs/fieldfig/runs/'
for L in (15,):
    fn=OUT+f'runs3_{L}.json'
    if os.path.exists(fn): continue
    N,B=build(L,[((i*5,0),'HHHVVV'*(L//3)) for i in range(L//5)]); B=np.array(B); P=signs(N,B)
    t=time.time(); r=simulate(N,B,300.0,0.0,1000,40,100,1,verbose=False,fixM=0.5*abs(P.sum()))
    r.update(key='runs3',L=L,R=2*L*(L//5),sigma=2.5,wall=round(time.time()-t,1)); r.pop('bins_E'); r.pop('bins_Ezz')
    json.dump(r,open(fn,'w')); print('runs3',L,N,r['E_J']/N,r['E_zz']/N,r['E_xy']/N,flush=True)

"""QMC-anchored central LP: active phases (with sigma) along h, and energy lowering between search depths."""
import json, sys, numpy as np, io, contextlib
from esq_h import e_sq
from analyze_enum import old_calibrated, lp
from resample import qmc_table, KEY, extend_key
s0=1/np.sqrt(2); e0=e_sq(0); chi=0.065/0.058; rho=2-np.sqrt(2)
T=qmc_table(); NPh=json.load(open('/mnt/user-data/outputs/fieldfig/newphases.json'))
def phases(cmp):
    P=old_calibrated()
    if cmp: P=P+[tuple(n) for n in json.load(open(cmp))['new']]
    extend_key(P); out=[]
    for nm,sg,ta,el,r in P:
        k=KEY.get(nm); e=el*r
        if k and k in T: e=T[k][max(T[k])]['eps']
        out.append((nm,sg,ta,e,k))
    return out
hs=np.linspace(0,0.30,301)
if __name__=='__main__':
    res={}
    for lab,cmp in [('old',None),('N16','compare_N16.json'),('N21','compare_N21.json'),('N24','compare_N24.json')]:
        P=phases(cmp); sig=np.array([p[1] for p in P]); tau=np.array([p[2] for p in P]); eps=np.array([p[3] for p in P])
        E=[];prev=None; print('---',lab)
        for h in hs:
            r,red=lp(sig,tau,eps,h); E.append(r.fun); act=tuple(sorted((P[i][4] or P[i][0],round(P[i][1],3),round(P[i][2],3),round(float(r.x[i]),3)) for i in np.nonzero(r.x>1e-6)[0]))
            tt=(r.x*tau).sum()
            if act!=prev: print('  h=%.3f m=%.4f  %s'%(h,rho*tt/2,act)); prev=act
        res[lab]=np.array(E)
    for a,b in [('old','N16'),('N16','N21'),('N21','N24')]:
        d=res[b]-res[a]; w=(hs>=0.09)&(hs<=0.18)
        print(f'{a}->{b}: max lowering per rhombus in window 0.09-0.18: {(-d[w]).max():.5f} J, mean {(-d[w]).mean():.5f}; outside window max {(-d[~w]).max():.5f}')

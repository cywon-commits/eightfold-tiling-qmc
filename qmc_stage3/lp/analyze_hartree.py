"""QMC-calibrated staircase with per-phase ratios predicted from LSWT correlators:
r_QMC ~= 1.114 + 1.152 (r_H - 1),  r_H = 1 + dE_Hartree/dE_LSWT  (fit to 8 measured phases, rms 0.018)."""
import json, numpy as np, sys
from scipy.optimize import linprog
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from esq_h import e_sq
_D=os.path.dirname(os.path.abspath(__file__)); T=json.load(open(os.path.join(_D,'phases.json'))); P=T['phases']; rH=json.load(open(os.path.join(_D,'rH_all.json')))
meas={'small(3, 0, 3, 3) t=0.333':1.201,'small(3, 0, 3, 3) t=0.111':0.892,'small(2, 0, 2, 4) t=0.500':1.245,'2-0-6-4 t=0.500':1.247,
      '4-1-4-1 t=0.500':1.201,'runs3 d5':1.045,'dice':1.250,'small(2, 0, 2, 2) t=1.000':1.250,'small(3, 0, 3, 1) t=1.000':1.287,'6-0-6-2 t=1.000':1.287}
S3=json.load(open(os.path.join(_D,'stage3_measured.json'))) if os.path.exists(os.path.join(_D,'stage3_measured.json')) else {}
def pred(name):
    if name in meas: return meas[name],'QMC'
    if name in S3: return S3[name],'QMC stage3'
    if ' t=' in name:
        b,t=name.split(' t='); kk=f'{b}|{float(t):.3f}'
        if kk in S3: return S3[kk],'QMC stage3'
    if name in rH: return 1.114+1.152*(rH[name]-1),'Hartree'
    if ' t=' in name:
        base,t=name.split(' t='); key=f'{base}|{float(t):.3f}'
        if key in rH: return 1.114+1.152*(rH[key]-1),'Hartree'
    return None,'none'
sig=np.array([p['sigma'] for p in P]); tau=np.array([p['tau'] for p in P]); eps=[]; src={}
for p in P:
    r,s=pred(p['name']); src[s]=src.get(s,0)+1
    eps.append(p['eps']*(r if r is not None else 1.2))
eps=np.array(eps); print('ratio sources:',src)
s0=1/np.sqrt(2); rho=2-np.sqrt(2); e0=e_sq(0); chi=0.065/0.058
print('\n| h (J) | m | phases |\n|---|---|---|'); prev=None; curve=[]
for h in np.linspace(0,0.32,6401):
    d=(e0-e_sq(h))*chi; c=eps+d*(1+sig)-h*tau/4
    r=linprog(c,A_ub=np.stack([np.ones_like(sig),sig]),b_ub=[1,s0],bounds=[(0,None)]*len(P),method='highs'); tt=(r.x*tau).sum(); curve.append((h,rho*tt/2))
    if prev is None or abs(tt-prev)>1e-6:
        print('| %.4f | %.4f | %s |'%(h,rho*tt/2,', '.join('%s (%.2f)'%(P[i]['name'],r.x[i]) for i in range(len(P)) if r.x[i]>1e-6))); prev=tt
json.dump(curve,open(os.path.join(_D,'lp_qmc_hartree.json'),'w'))

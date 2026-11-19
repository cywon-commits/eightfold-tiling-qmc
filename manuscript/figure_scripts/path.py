"""Ground-state path from the QMC-calibrated linear program: phase weights x_P(h) per rhombus."""
import json, numpy as np
from scipy.optimize import linprog
from esq_h import e_sq
exec(open('/mnt/user-data/outputs/qmc_stage3/lp/analyze_hartree.py').read().split("eps=np.array(eps)")[0].split("_D=os.path")[0])
_D='.'; T=json.load(open('phases.json')); P=T['phases']; rH=json.load(open('rH_all.json')); S3=json.load(open('stage3_measured.json'))
src=open('/mnt/user-data/outputs/qmc_stage3/lp/analyze_hartree.py').read()
exec(src[src.index('meas={'):src.index("sig=np.array")])
sig=np.array([p['sigma'] for p in P]); tau=np.array([p['tau'] for p in P])
eps=np.array([p['eps']*(pred(p['name'])[0] or 1.2) for p in P])
s0=1/np.sqrt(2); e0=e_sq(0); chi_scale=0.065/0.058
KEY={'small(3, 0, 3, 3) t=0.111':'hub','runs3 d5':'runs3','small(4, 0, 6, 2) t=0.400':'p4062','small(6, 0, 4, 2) t=0.400':'p4062',
     'small(3, 0, 3, 3) t=0.333':'z5carrier','4-1-4-1 t=0.500':'fourfam','small(3, 0, 3, 1) t=1.000':'z5crystal','6-0-6-2 t=1.000':'z5crystal',
     '4-2-2-2 t=0.500':'p4222','2-0-6-4 t=0.500':'kappa89','dice':'dice','small(2, 0, 2, 2) t=1.000':'dice'}
def path(hs):
    rows=[]
    for h in hs:
        d=(e0-e_sq(h))*chi_scale; c=eps+d*(1+sig)-h*tau/4
        r=linprog(c,A_ub=np.stack([np.ones_like(sig),sig]),b_ub=[1,s0],bounds=[(0,None)]*len(P),method='highs')
        w={}
        for i in np.nonzero(r.x>1e-7)[0]:
            k=KEY.get(P[i]['name'],P[i]['name']); w[k]=w.get(k,0)+r.x[i]
        rows.append((h,w))
    return rows
if __name__=='__main__':
    rows=path(np.linspace(0,0.3,3001)); seen=set()
    for h,w in rows:
        for k in w: seen.add(k)
    print('phases on path:',seen)
    json.dump([(h,w) for h,w in rows],open('path.json','w'))

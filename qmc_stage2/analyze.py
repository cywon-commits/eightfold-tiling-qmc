"""Stage-2 analysis: QMC/LSWT ratios of eps per rhombus, then the QMC-calibrated magnetization staircase (LP)."""
import json, glob, os, numpy as np
from scipy.optimize import linprog
HERE=os.path.dirname(os.path.abspath(__file__)); import sys; sys.path.insert(0,os.path.join(HERE,'lp'))
from esq_h import e_sq
ESQ_QMC=-0.669437; ESQ_LSWT=-0.6579474209435209
runs={}
for f in glob.glob(os.path.join(HERE,'runs','*.json')):
    r=json.load(open(f)); runs.setdefault((r['graph'],r['beta']),[]).append(r)
print('## Phases: eps per rhombus = (E - N e_sq)/R\n\n| graph | beta | seeds | eps QMC | eps LSWT | ratio |\n|---|---|---|---|---|---|')
ratio={}
for (g,b),rs in sorted(runs.items()):
    w=np.array([1/r['E_err']**2 for r in rs]); E=np.sum(w*[r['E'] for r in rs])/w.sum(); s=1/np.sqrt(w.sum())
    N=rs[0]['N']; R=rs[0]['R']; eq=(E-N*ESQ_QMC)/R; el=rs[0]['eps_LSWT']; rr=eq/el; se=s/R
    print(f'| {g} | {b} | {len(rs)} | {eq:.5f} ± {se:.5f} | {el:.5f} | {rr:.3f} ± {se/el:.3f} |')
    base=g.split('_')[0]+('' if not g.startswith('runs3') else '_d5'); base={'runs3':'runs3_d5'}.get(base,base)
    if b==300: ratio.setdefault(base,[]).append((rr,se/el,N))
# use the largest graph for each phase
R_use={k:sorted(v,key=lambda x:-x[2])[0][0] for k,v in ratio.items()}
T=json.load(open(os.path.join(HERE,'lp','phases.json')))
R_use.update({k:v for k,v in T['stage1_ratios'].items()})
default=float(np.mean(list(R_use.values())))
print('\nratios used:',{k:round(v,3) for k,v in R_use.items()},' default for unmeasured phases = %.3f'%default)
P=T['phases']; mm=T['measured_map']
sig=np.array([p['sigma'] for p in P]); tau=np.array([p['tau'] for p in P])
eps=np.array([p['eps']*R_use.get(mm.get(p['name'],''),default) for p in P])
s0=1/np.sqrt(2); rho=2-np.sqrt(2); e0=e_sq(0); chi_scale=0.065/0.058
print('\n## QMC-calibrated staircase (AB composition, canting with chi_QMC)\n\n| h (J) | m | phases |\n|---|---|---|')
prev=None
for h in np.linspace(0,0.32,6401):
    d=(e0-e_sq(h))*chi_scale; c=eps+d*(1+sig)-h*tau/4
    r=linprog(c,A_ub=np.stack([np.ones_like(sig),sig]),b_ub=[1,s0],bounds=[(0,None)]*len(P),method='highs'); tt=(r.x*tau).sum()
    if prev is None or abs(tt-prev)>1e-6:
        print(f"| {h:.4f} | {rho*tt/2:.4f} | {', '.join(f'{P[i]['name']} ({r.x[i]:.2f})' for i in range(len(P)) if r.x[i]>1e-6)} |"); prev=tt

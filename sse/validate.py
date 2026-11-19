"""Validation of sse_det.py against exact diagonalization (takes ~1-2 min).
Pass criterion: |E_QMC - E_ED| < 2.5 sigma for every test."""
import json, numpy as np, os
from sse_det import simulate
from ed_sector import ed
here=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(here,'..','qmc_inputs')
tests=[]
for name,ref in (('dice_L1_N12',-7.86916032),('z5crystal_L1_N15',-10.08806815)):
    g=json.load(open(os.path.join(D,name+'.json'))); r=simulate(g['N'],g['bonds'],60.0,0.0,2000,40,300,1,verbose=False)
    tests.append((name+' beta=60',r['E'],r['E_err'],ref))
L=4; B=[]
for x in range(L):
    for y in range(L):
        i=x*L+y; B+=[(i,((x+1)%L)*L+y),(i,x*L+(y+1)%L)]
for M in (0,1,2):
    ref=ed(16,B,8+M); r=simulate(16,B,50.0,0.0,2000,40,300,5+M,verbose=False,fixM=M)
    tests.append((f'4x4 square, fixM={M}, beta=50',r['E_J'],r['E_J_err'],ref))
ok=True
for name,E,err,ref in tests:
    z=(E-ref)/err; ok&=abs(z)<2.5
    print(f'{name:34s} QMC {E:.5f} +- {err:.5f}   ED {ref:.6f}   {z:+.2f} sigma  {"OK" if abs(z)<2.5 else "FAIL"}')
print('ALL PASS' if ok else 'SOME TESTS FAILED')

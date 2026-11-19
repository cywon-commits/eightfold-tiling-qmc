"""LSWT eps per rhombus on exactly the supercells used in QMC (for same-cell ratios r_L = eps_QMC(L)/eps_LSWT(L))."""
import sys, json, glob, numpy as np
sys.path.insert(0,'/home/claude/fieldfig')
from phases import graph
from lswt import lswt_energy
from ribbon import build
ESQ=-0.6579474209435209
cells=set()
for f in glob.glob('/mnt/user-data/outputs/fieldfig/runs_seeds/*_s*.json')+glob.glob('/mnt/user-data/outputs/fieldfig/runs/*.json'):
    d=json.load(open(f))
    if d.get('key','sq16')=='sq16' or 'sq16' in f: continue
    cells.add((d['key'],d['L']))
out={}
for k,L in sorted(cells):
    if k=='runs3':
        N,B=build(L,[((i*5,0),'HHHVVV'*(L//3)) for i in range(L//5)]); B=np.array(B); R=2*L*(L//5)
    else:
        g=graph(k,L); N,B,R=g['N'],g['B'],g['R']
    E,S=lswt_energy(N,B); out[f'{k}|{L}']=(E-N*ESQ)/R; print(k,L,N,round(out[f'{k}|{L}'],5),flush=True)
json.dump(out,open('/mnt/user-data/outputs/sectors/lswt_qmc_cells.json','w'),indent=1)

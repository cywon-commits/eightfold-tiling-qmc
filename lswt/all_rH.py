import json, numpy as np, glob, sys
from hartree import energies
from crystal import supercell_graph
from ribbon import build
e1=-0.6579474209435209; d_sq=(-e1-0.5)/2; e2=-2*d_sq**2
def rH(N,B,R):
    E1,E2,_=energies(N,B); dE1=(E1-N*e1)/R; dE2=(E2-N*e2)/R; return 1+dE2/dE1, dE1
def tiles_graph(p,L):
    tiles=[(np.array(R),a,b) for R,a,b in p['tiles']]; T=[np.array(t) for t in p['T']]
    N,B=supercell_graph(tiles,T[0],T[1],L); R=sum(1 for _,a,b in tiles if abs(a-b)!=2)*L*L; return N,B,R
out={}
def word(runs): return ''.join(('H' if i%2==0 else 'V')*r for i,r in enumerate(runs))
rib={'runs3 d5':((3,3),5,30),'runs3 d9':((3,3),9,27),'runs5 d7':((5,5),7,35),'HV d5':((1,1),5,30),'runs 1133':((1,1,3,3),5,20),
     'runs 111133':((1,1,1,1,3,3),5,25),'runs 33111111':((3,3,1,1,1,1,1,1),5,30)}
for name,(runs,d,L) in rib.items():
    w=word(runs); h=w.count('H'); N,B=build(L,[((i*d,0),w*(L//h)) for i in range(L//d)]); R=2*L*(L//d)
    out[name]=rH(N,np.array(B),R)[0]; print(name,round(out[name],3),flush=True)
for c in json.load(open('/home/claude/dense/candidates.json')):
    n=f"small{c['counts']} t={c['tau']:.3f}"
    if n in out: continue
    L=4 if len(c['tiles'])<=30 else 2
    N,B,R=tiles_graph(c,L); out[n]=rH(N,B,R)[0]; print(n,round(out[n],3),flush=True)
for f in sorted(glob.glob('/home/claude/dense/big_*.json')):
    base=f.split('big_')[1].replace('.json','')
    for k,p in json.load(open(f)).items():
        N,B,R=tiles_graph(p,2)
        try: r,_=rH(N,B,R)
        except Exception as e: continue
        # name must match phases.json naming: '<base> t=<tau at L4>' ; store by base and rounded tau
        out[f'{base}|{float(k):.3f}']=r; print(base,k,round(r,3),flush=True)
json.dump(out,open('rH_all.json','w'),indent=1)

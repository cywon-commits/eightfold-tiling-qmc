import numpy as np, json, sys
from phases import graph, PHASES
from lswt_local import local
ESQ=-0.6579474209435209
import sys as _s
KEYS=_s.argv[1:] or list(PHASES)
try: out=json.load(open('perphase.json'))
except Exception: out={}
for key in KEYS:
    res={}
    g0=graph(key,1); n1=g0['N']
    Ls=[L for L in (2,3,4,6,8) if n1*L*L<=1100][-2:] if key not in ('fourfam','p4222','kappa89') else (2,3)
    for L in Ls:
        g=graph(key,L); r=local(g['N'],g['B']); N=g['N']
        cls={}
        for zz_,P,s in zip(r['z'],r['P'],r['Sz']):
            k=('maj' if P>0 else 'min',int(zz_)); cls.setdefault(k,[]).append(s)
        res[L]=dict(N=N,R=g['R'],e=r['E']/N,ezz=r['zz'].sum()/N,exy=r['xy'].sum()/N,m=abs(r['P'].sum())/N,
                    cls={f'{a}{b}':(float(np.mean(v)),len(v)/N) for (a,b),v in sorted(cls.items())})
    L=max(res); x=res[L]
    print(f"{key:10s} L={L} N={x['N']:4d} m={x['m']:.4f} e={x['e']:.5f} ezz={x['ezz']:.5f} exy={x['exy']:.5f} (L small: e={res[min(res)]['e']:.5f})")
    print('            ', ', '.join(f"{k}:{v[0]:+.3f}({v[1]:.2f})" for k,v in x['cls'].items()))
    out[key]=res[L]
json.dump(out,open('perphase.json','w'),indent=1)

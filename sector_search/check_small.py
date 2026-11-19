"""independent brute-force check of sector counts for N<=3 (all 2x4 matrices with entries in [-3,3])"""
import numpy as np, itertools
from sectors import canon_sector, PAIRS
from collections import Counter
r=np.arange(-3,4); V=np.array(list(itertools.product(r,repeat=4)))
V=V[np.abs(V).sum(1)>0]
found=set()
for v1 in V:
    P=np.stack([v1[j]*V[:,k]-v1[k]*V[:,j] for j,k in PAIRS],1)
    ok=(P>=0).all(1)&(P.sum(1)>0)&(P.sum(1)<=3)
    for v2 in V[ok]: found.add(canon_sector((tuple(v1),tuple(v2))))
import json
S=json.load(open('sectors_N12.json')); mine=Counter(r['N'] for r in S if r['N']<=3)
def N(c):
    from sectors import pl; return sum(pl(c))
print('brute',Counter(N(c) for c in found),'enum',mine, 'same set:',found=={tuple(map(tuple,r['canon'])) for r in S if r['N']<=3})

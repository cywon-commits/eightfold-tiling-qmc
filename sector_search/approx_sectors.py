"""Sectors with a prescribed square/rhombus ratio sigma = S/R (S squares, R rhombi), N = S+R, D8 x GL2 reduced."""
import sys, json, math
from sectors import comps, lattice_from_pl, orient, pl, canon_sector
def sectors_comp(S,R):
    N=S+R; found={}
    for p in comps(N):
        if sum(p)!=N or p[1]+p[4]!=S: continue
        a01,a02,a03,a12,a13,a23=p
        if a01*a23-a02*a13+a03*a12!=0: continue
        g=math.gcd(*p); q=[x//g for x in p]; T0=lattice_from_pl(q)
        for a in range(1,g+1):
            if g%a: continue
            d=g//a
            for b in range(d):
                T=orient((tuple(a*T0[0][i]+b*T0[1][i] for i in range(4)),tuple(d*T0[1][i] for i in range(4))))
                if T is None: continue
                c=canon_sector(T); found.setdefault(c,T)
    return found
if __name__=='__main__':
    S,R=int(sys.argv[1]),int(sys.argv[2]); F=sectors_comp(S,R)
    rows=[dict(canon=c,T=[list(T[0]),list(T[1])],p=pl(T),N=S+R,sq=S,even=(sum(T[0])%2==0 and sum(T[1])%2==0)) for c,T in F.items()]
    json.dump(rows,open(f'sectors_S{S}R{R}.json','w')); print(S,R,len(rows),'sectors')

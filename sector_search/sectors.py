"""All winding sectors of periodic eightfold rhombus tilings with N <= Nmax tiles (= vertices).
A sector is a rank-2 lattice Lambda = <T1,T2> in Z^4 (lifted periods).  Valid iff all Pluecker minors
p_jk = T1_j T2_k - T1_k T2_j have one sign (taken >= 0).  N = sum p_jk ; tiles of type (j,k) = p_jk.
Squares: (0,2),(1,3).  Reduction: GL2(Z) (row HNF) x D8 (signed column permutations)."""
import numpy as np, itertools, math
PAIRS=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
def pl(T):
    T1,T2=T; return tuple(int(T1[j]*T2[k]-T1[k]*T2[j]) for j,k in PAIRS)
def hnf(T):
    """row Hermite normal form of a 2x4 integer matrix of rank 2"""
    A=[list(map(int,T[0])),list(map(int,T[1]))]; r=0
    for c in range(4):
        if r==2: break
        # euclid on rows r..1 in column c
        while True:
            nz=[i for i in range(r,2) if A[i][c]!=0]
            if not nz: break
            i0=min(nz,key=lambda i:abs(A[i][c])); A[r],A[i0]=A[i0],A[r]
            done=True
            for i in range(r+1,2):
                if A[i][c]!=0:
                    q=A[i][c]//A[r][c]; A[i]=[a-q*b for a,b in zip(A[i],A[r])]
                    if A[i][c]!=0: done=False
            if done: break
        if any(A[i][c]!=0 for i in range(r,2)):
            if A[r][c]<0: A[r]=[-a for a in A[r]]
            for i in range(r):
                q=A[i][c]//A[r][c]; A[i]=[a-q*b for a,b in zip(A[i],A[r])]
            r+=1
    return tuple(map(tuple,A))
ROT=lambda v:(-v[3],v[0],v[1],v[2])
REF=lambda v:(v[0],-v[3],-v[2],-v[1])
def d8(T):
    out=[]; cur=[tuple(T[0]),tuple(T[1])]
    for refl in (0,1):
        c=[REF(v) for v in cur] if refl else cur
        for k in range(8):
            out.append(c); c=[ROT(v) for v in c]
    return out
def orient(T):
    p=pl(T)
    if all(x>=0 for x in p): return T
    if all(x<=0 for x in p): return (T[1],T[0])
    return None
def canon_sector(T):
    best=None
    for U in d8(T):
        U=orient(U); H=hnf(U)
        if best is None or H<best: best=H
    return best
def sectors(Nmax):
    """enumerate via Pluecker vectors: p>=0, sum<=Nmax, p01 p23 - p02 p13 + p03 p12 = 0;
    saturated lattice for p/g, then all index-g sublattices."""
    found={}
    for p in comps(Nmax):
        if sum(p)==0: continue
        a01,a02,a03,a12,a13,a23=p
        if a01*a23-a02*a13+a03*a12!=0: continue
        g=math.gcd(*p); q=[x//g for x in p]
        T0=lattice_from_pl(q)
        if T0 is None: raise RuntimeError(('no lattice',p))
        for a in range(1,g+1):
            if g%a: continue
            d=g//a
            for b in range(d):
                T=(tuple(a*T0[0][i]+b*T0[1][i] for i in range(4)),tuple(d*T0[1][i] for i in range(4)))
                T=orient(T)
                if T is None: continue
                assert pl(T)==tuple(p),(p,T)
                c=canon_sector(T); found.setdefault(c,T)
    return found
def int_kernel(C):
    """integer kernel basis (saturated) of an integer matrix C (r x 4) via unimodular column ops"""
    C=[list(r) for r in C]; n=4; U=[[int(i==j) for j in range(n)] for i in range(n)]
    def colop(i,j,q):   # col_j -= q col_i
        for r in C: r[j]-=q*r[i]
        for r in U: r[j]-=q*r[i]
    def swap(i,j):
        for r in C: r[i],r[j]=r[j],r[i]
        for r in U: r[i],r[j]=r[j],r[i]
    piv=0
    for row in range(len(C)):
        if piv==n: break
        while True:
            nz=[j for j in range(piv,n) if C[row][j]!=0]
            if not nz: break
            j0=min(nz,key=lambda j:abs(C[row][j])); swap(piv,j0)
            for j in range(piv+1,n):
                if C[row][j]: colop(piv,j,C[row][j]//C[row][piv])
            if all(C[row][j]==0 for j in range(piv+1,n)): break
        if any(C[row][j] for j in range(piv,n)): piv+=1
    return [tuple(U[i][j] for i in range(n)) for j in range(piv,n)]
def lattice_from_pl(q):
    """saturated rank-2 lattice with primitive Pluecker vector q"""
    om={pr:x for pr,x in zip(PAIRS,q)}
    def w(i,j): return om[(i,j)] if i<j else (-om[(j,i)] if i>j else 0)
    # plane vectors c_j(i)=w(i,j); complement = integer kernel of the plane basis (as rows)
    cs=[tuple(w(i,j) for i in range(4)) for j in range(4)]
    cs=[c for c in cs if any(c)]
    A=np.array(cs,dtype=float)
    if np.linalg.matrix_rank(A)!=2: return None
    perp=int_kernel(cs)            # 2 vectors spanning plane^perp (integers)
    base=int_kernel(perp)          # saturated plane lattice
    if len(base)!=2: return None
    T=(base[0],base[1]); p=pl(T)
    g=math.gcd(*p)
    if g!=1: return None
    if p==tuple(q): return T
    if p==tuple(-x for x in q): return (T[1],T[0])
    return None
def comps(Nmax):
    def rec(pre,rem,k):
        if k==6: yield tuple(pre); return
        for x in range(rem+1): yield from rec(pre+[x],rem-x,k+1)
    yield from rec([],Nmax,0)
if __name__=='__main__':
    import sys,json,time
    Nmax=int(sys.argv[1]); t=time.time(); S=sectors(Nmax)
    rows=[]
    for c,T in S.items():
        p=pl(T); rows.append(dict(canon=c,T=[list(T[0]),list(T[1])],p=p,N=sum(p),sq=p[1]+p[4],even=(sum(T[0])%2==0 and sum(T[1])%2==0)))
    rows.sort(key=lambda r:(r['N'],r['p']))
    json.dump(rows,open(f'sectors_N{Nmax}.json','w'))
    from collections import Counter
    print(len(rows),'sectors',Counter(r['N'] for r in rows),f'{time.time()-t:.0f}s')

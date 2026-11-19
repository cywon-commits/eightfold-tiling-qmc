import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
def ed(N,B,nup):
    states=[s for s in range(1<<N) if bin(s).count('1')==nup]; idx={s:i for i,s in enumerate(states)}
    r=[];c=[];v=[]
    for i,s in enumerate(states):
        d=0.0
        for a,b in B:
            if ((s>>a)&1)==((s>>b)&1): d+=0.25
            else: d-=0.25; r.append(idx[s^((1<<a)|(1<<b))]); c.append(i); v.append(0.5)
        r.append(i); c.append(i); v.append(d)
    H=coo_matrix((v,(r,c)),shape=(len(states),)*2).tocsr()
    return eigsh(H,k=1,which='SA')[0][0]

"""Second-order (Oguchi/Hartree) correction from LSWT bond correlators.
Rotated frame: H2 = sum_i A_i n_i + sum_b (a_i a_j + h.c.) * S J,  A_i = S J z_i.  With q,p quadratures:
 delta_b = 1/2 - <(q_i+q_j)^2>/4 - <(p_i-p_j)^2>/4   (bond fluctuation, >0)
 E_LSWT = -J sum_b (S^2 + 2 S delta_b),   E_2 = -J sum_b delta_b^2   (=> E ~ -J sum_b (S+delta_b)^2)."""
import numpy as np, json
def bond_deltas(N,bonds,S=0.5,J=1.0,g=1e-9):
    bonds=np.asarray(bonds); Adj=np.zeros((N,N)); Adj[bonds[:,0],bonds[:,1]]=1; Adj[bonds[:,1],bonds[:,0]]=1
    z=Adj.sum(1); A=np.diag(J*S*z+g); B=J*S*Adj
    K=A+B; M=A-B                                 # H = 1/2 q^T K q + 1/2 p^T M p - TrA/2
    e,V=np.linalg.eigh(K); e=np.clip(e,1e-14,None); Kh=(V*np.sqrt(e))@V.T; Kmh=(V/np.sqrt(e))@V.T
    w,U=np.linalg.eigh(Kh@M@Kh); w=np.clip(w,1e-18,None)
    X=(U*np.sqrt(np.sqrt(w)))@U.T; Xi=(U/np.sqrt(np.sqrt(w)))@U.T   # (Kh M Kh)^{+1/4}, ^{-1/4}
    Gp=0.5*Kh@Xi@Xi@Kh                           # <p p> = 1/2 K^{1/2} W^{-1/2} K^{1/2}
    Gq=0.5*Kmh@X@X@Kmh                           # <q q> = 1/2 K^{-1/2} W^{+1/2} K^{-1/2}
    i,j=bonds[:,0],bonds[:,1]
    qq=Gq[i,i]+Gq[j,j]+2*Gq[i,j]; pp=Gp[i,i]+Gp[j,j]-2*Gp[i,j]
    d=0.5-qq/4-pp/4
    return d
def energies(N,bonds,S=0.5,J=1.0):
    d=bond_deltas(N,bonds,S,J); E1=-J*np.sum(S*S+2*S*d); E2=-J*np.sum(d*d); return E1,E2,d
if __name__=='__main__':
    from lswt import lswt_energy
    L=12; B=[]
    for x in range(L):
        for y in range(L):
            k=x*L+y; B+=[(k,((x+1)%L)*L+y),(k,x*L+(y+1)%L)]
    E1,E2,d=energies(L*L,B); print('square L=12: E1/N=%.6f (lswt %.6f)  E2/N=%.6f  delta=%.5f'%(E1/L/L,lswt_energy(L*L,np.array(B))[0]/L/L,E2/L/L,d.mean()))

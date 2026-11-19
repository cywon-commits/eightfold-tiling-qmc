"""LSWT local moments and bond-resolved zz / xy exchange (collinear Lieb state, majority along h).
A tiny field h_reg (commutes with H) lifts the zero mode of the ferrimagnetic multiplet; covariances are h-independent."""
import numpy as np
from lswt import signs
def local(N,bonds,S=0.5,J=1.0,h_reg=0.01):
    bonds=np.asarray(bonds); P=signs(N,bonds).astype(float)
    if P.sum()<0: P=-P
    Adj=np.zeros((N,N)); Adj[bonds[:,0],bonds[:,1]]=1; Adj[bonds[:,1],bonds[:,0]]=1
    z=Adj.sum(1); A=np.diag(J*S*z+h_reg*P); B=J*S*Adj
    e,V=np.linalg.eigh(A+B); assert e.min()>0
    Kh=(V*np.sqrt(e))@V.T; Kmh=(V/np.sqrt(e))@V.T
    w,U=np.linalg.eigh(Kh@(A-B)@Kh); assert w.min()>0
    X=(U*w**0.25)@U.T; Xi=(U*w**-0.25)@U.T
    Gp=0.5*Kh@Xi@Xi@Kh; Gq=0.5*Kmh@X@X@Kmh
    n=0.5*(np.diag(Gq)+np.diag(Gp)-1); Sz=P*(S-n)
    i,j=bonds[:,0],bonds[:,1]
    zz=J*(-S*S+S*(n[i]+n[j])); xy=J*S*(Gq[i,j]-Gp[i,j])
    return dict(P=P,z=z,n=n,Sz=Sz,zz=zz,xy=xy,E=zz.sum()+xy.sum())

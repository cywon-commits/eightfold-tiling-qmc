"""Square-lattice Heisenberg AF in a field: classical canting + LSWT zero-point energy (Zhitomirsky-Chernyshev form).
e(h) = -2JS^2 - h^2/(16J) + 2JS < sqrt((1+g)(1-cos2t g)) - 1 >,  sin t = h/(8JS)."""
import numpy as np
q=(np.arange(1200)+0.5)/1200*2*np.pi; Q1,Q2=np.meshgrid(q,q); G=(np.cos(Q1)+np.cos(Q2))/2
def e_sq(h,J=1.0,S=0.5):
    st=h/(8*J*S); c2=1-2*st**2
    return -2*J*S*S - h*h/(16*J) + 2*J*S*np.mean(np.sqrt(np.clip((1+G)*(1-c2*G),0,None))-1)
if __name__=='__main__':
    e0=e_sq(0); print('e_sq(0)=%.6f'%e0)
    for h in (0.02,0.05,0.08,0.1,0.12,0.15,0.17,0.2):
        d=e_sq(h)-e0; print('h=%.2f  Delta e=%.6f  classical=%.6f  chi_eff=%.4f'%(h,d,-h*h/16,-2*d/h**2))

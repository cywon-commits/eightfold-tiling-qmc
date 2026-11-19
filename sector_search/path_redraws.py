"""Ground-state path under redraws of the phase energies (same error model as resample.py), N<=24 set.
Stores per redraw: weight of every kept phase vs h (basic solutions solved exactly)."""
import json, sys, itertools, numpy as np
from esq_h import e_sq
from analyze_enum import old_calibrated, lp
from resample import qmc_table, extend_key, KEY, s0, chi, e0
NS=int(sys.argv[1]) if len(sys.argv)>1 else 400
tab=qmc_table(); P=old_calibrated()+[tuple(n) for n in json.load(open('compare_N24.json'))['new']]; extend_key(P)
names=[p[0] for p in P]; sig=np.array([p[1] for p in P]); tau=np.array([p[2] for p in P])
eps=np.array([p[3]*p[4] for p in P]); err=0.018*np.array([p[3] for p in P]); qk=[None]*len(P)
for i,nm in enumerate(names):
    k=KEY.get(nm)
    if k and k in tab: L=max(tab[k]); eps[i]=tab[k][L]['eps']; err[i]=tab[k][L]['eps_err']; qk[i]=k
hs=np.linspace(0,0.30,301); d=np.array([(e0-e_sq(h))*chi for h in hs])
keep=set(i for i in range(len(P)) if qk[i])
for h in hs[::6]:
    r,red=lp(sig,tau,eps,h); keep|=set(np.nonzero(red<0.004)[0])
keep=sorted(keep); S,T_,EP,ER=sig[keep],tau[keep],eps[keep],err[keep]; n=len(keep)
pairs=[(i,j) for i,j in itertools.combinations(range(n),2) if abs(S[i]-S[j])>1e-12]
PI=np.array([p[0] for p in pairs]); PJ=np.array([p[1] for p in pairs])
XJ=(s0-S[PI])/(S[PJ]-S[PI]); XI=1-XJ; ok=(XI>=0)&(XJ>=0); PI,PJ,XI,XJ=PI[ok],PJ[ok],XI[ok],XJ[ok]
XS=np.minimum(1,np.where(S>0,s0/np.maximum(S,1e-12),1))
def solve(e):
    C=e[None,:]+d[:,None]*(1+S[None,:])-hs[:,None]*T_[None,:]/4      # (H,n)
    W=np.zeros((len(hs),n))
    single=C*XS[None,:]; bi=single.argmin(1); bs=single[np.arange(len(hs)),bi]
    pc=C[:,PI]*XI[None,:]+C[:,PJ]*XJ[None,:]; bp=pc.argmin(1); bpv=pc[np.arange(len(hs)),bp]
    for t in range(len(hs)):
        best=min(0.0,bs[t],bpv[t])
        if best==0.0 and bs[t]>=-1e-12 and bpv[t]>=-1e-12: continue
        if bs[t]<=bpv[t]: W[t,bi[t]]=XS[bi[t]]
        else: q=bp[t]; W[t,PI[q]]=XI[q]; W[t,PJ[q]]=XJ[q]
    return W
rng=np.random.default_rng(7); out=np.zeros((NS,len(hs),n),dtype=np.float32)
for s in range(NS): out[s]=solve(EP+ER*rng.standard_normal(n))
np.savez_compressed('path_redraws_N24.npz',W=out,hs=hs,keep=np.array(keep),sigma=S,tau=T_,names=np.array([names[i] for i in keep]),qkey=np.array([qk[i] or '' for i in keep]),eps=EP,err=ER)
print('kept',n,'redraws',NS)

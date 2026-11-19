"""QMC-anchored central ground-state path for a given search depth -> path json (h, {phase: x per rhombus}), plus phase table."""
import json, sys, numpy as np
from window import phases, lp, hs
cmp=sys.argv[1]; out=sys.argv[2]
P=phases(cmp); sig=np.array([p[1] for p in P]); tau=np.array([p[2] for p in P]); eps=np.array([p[3] for p in P])
rows=[]; info={}
for h in np.linspace(0,0.3,3001):
    r,red=lp(sig,tau,eps,h); w={}
    for i in np.nonzero(r.x>1e-7)[0]:
        k=P[i][4] or P[i][0]; w[k]=w.get(k,0)+float(r.x[i]); info[k]=dict(sigma=P[i][1],tau=P[i][2],eps=P[i][3],qmc=P[i][4] is not None)
    rows.append((float(h),w))
json.dump(dict(rows=rows,info=info),open(out,'w')); print(info)

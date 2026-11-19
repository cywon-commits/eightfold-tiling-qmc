"""Compare staircases: old phase set vs old + all-sector survivors, at three calibration levels.
Survivors identical to an old phase (same sigma, tau, eps within 3e-4) are merged into it."""
import json, sys, numpy as np, io, contextlib
from analyze_enum import old_calibrated, staircase, OLD
Nmax=sys.argv[1]
S=json.load(open(f'screen_N{Nmax}.json'))['survivors']
rH=json.load(open('rH_all.json'))
oldc=old_calibrated()
def hartree_ratio(name,rat):
    b=name; k=None
    if name in rH: return 1.114+1.152*(rH[name]-1)
    if ' t=' in name:
        base,t=name.split(' t='); k=f'{base}|{float(t):.3f}'
        if k in rH: return 1.114+1.152*(rH[k]-1)
    return rat
new=[]; merged=[]
for x in S:
    m=[p for p in oldc if abs(p[1]-x['sigma'])<1e-6 and abs(p[2]-x['tau'])<1e-6 and abs(p[3]-x['eps_big'])<3e-4]
    if m: merged.append((x['N'],x['sigma'],x['tau'],m[0][0])); continue
    new.append((f"sec N{x['N']} s={x['sigma']:.3f} t={x['tau']:.3f}",x['sigma'],x['tau'],x['eps_big'],x['ratio'],x))
print('survivors identical to old phases:',len(merged)); [print('  ',m) for m in merged]
print('genuinely new compositions/energies:',len(new))
for n in new: print('  %-28s eps=%.5f ratio=%.3f z3..7=%s T=%s'%(n[0],n[3],n[4],n[5]['z'][3:8],n[5]['T']))
NEW=[n[:5] for n in new]
with contextlib.redirect_stdout(io.StringIO()):
    lsw_old=[(p[0],p[1],p[2],p[3],1.0) for p in oldc]; lsw_new=[(n[0],n[1],n[2],n[3],1.0) for n in NEW]
    har_old=[(p[0],p[1],p[2],p[3],hartree_ratio(p[0],p[4])) for p in oldc]
res={}
for lab,A,B in [('LSWT',lsw_old,lsw_new),('Hartree-all',har_old,NEW),('QMC+Hartree',oldc,NEW)]:
    res[lab]=dict(old=staircase(A,lab+' old'),new=staircase(A+B,lab+' old+new'))
json.dump(dict(merged=merged,new=[n[:5] for n in new],res=res),open(f'compare_N{Nmax}.json','w'),default=float)

"""Stage-3: blind test of r_QMC = 1.114 + 1.152 (r_H - 1), then the staircase with the new ratios."""
import json, glob, os, numpy as np, subprocess, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ESQ_QMC=-0.669437
print('## Blind test of the Hartree prediction\n\n| graph | N | eps QMC | eps LSWT | r_QMC measured | r_QMC predicted | difference |\n|---|---|---|---|---|---|---|')
meas={}
for f in sorted(glob.glob(os.path.join(HERE,'runs','*.json'))):
    r=json.load(open(f)); eq=(r['E']-r['N']*ESQ_QMC)/r['R']; se=r['E_err']/r['R']; rr=eq/r['eps_LSWT']; sr=se/r['eps_LSWT']
    print('| %s | %d | %.5f ± %.5f | %.5f | %.3f ± %.3f | %.3f | %+.3f |'%(r['graph'],r['N'],eq,se,r['eps_LSWT'],rr,sr,r['r_QMC_predicted'],rr-r['r_QMC_predicted']))
    k=r['phase_key']
    if k not in meas or r['N']>meas[k][1]: meas[k]=(rr,r['N'])
json.dump({k:v[0] for k,v in meas.items()},open(os.path.join(HERE,'lp','stage3_measured.json'),'w'),indent=1)
print('\nPass criterion: |difference| <= 0.04 (about 2x the fit rms 0.018) for the largest size of every phase.')

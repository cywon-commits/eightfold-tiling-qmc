"""Table: QMC eps per rhombus (all seeds, largest L), LSWT eps, ratio, Hartree prediction (recorded before the run)."""
import json
from resample import qmc_table
from analyze_enum import old_calibrated
NP=json.load(open('/mnt/user-data/outputs/fieldfig/newphases.json'))
T=qmc_table(); OLDc={p[0]:p for p in old_calibrated()}
OLDKEY={'hub':'small(3, 0, 3, 3) t=0.111','runs3':'runs3 d5','p4062':'small(4, 0, 6, 2) t=0.400','z5carrier':'small(3, 0, 3, 3) t=0.333',
 'fourfam':'4-1-4-1 t=0.500','z5crystal':'small(3, 0, 3, 1) t=1.000','p4222':'4-2-2-2 t=0.500','kappa89':'2-0-6-4 t=0.500','dice':'dice'}
rows=[]
for k,d in sorted(T.items()):
    Ls=sorted(d); L=Ls[-1]; v=d[L]
    if k in NP: el=NP[k]['eps']; pred=NP[k]['ratio']; src='Hartree (blind)'
    else: p=OLDc[OLDKEY[k]]; el=p[3]; pred=p[4]; src='stage-1/3 QMC'
    fs=(d[Ls[0]]['eps']-v['eps']) if len(Ls)>1 else None
    rows.append(dict(phase=k,L=L,N=v['N'],seeds=v['seeds'],eps_qmc=v['eps'],eps_direct=v['eps_direct'],fs_factor=v['fs_factor'],err=v['eps_err'],eps_lswt=el,ratio=v['eps']/el,prev_ratio=pred,prev_src=src,fs_shift=fs))
    print('%-10s L=%2d N=%4d seeds=%d  eps=%.5f±%.5f (direct %.5f, fs x%.3f) LSWT %.5f  ratio %.3f  (before: %.3f %s)%s'%(k,L,v['N'],v['seeds'],v['eps'],v['eps_err'],v['eps_direct'],v['fs_factor'],el,v['eps']/el,pred,src,
          '' if fs is None else '  eps(L=%d)-eps(L=%d)=%+.5f'%(Ls[0],L,fs)))
json.dump(rows,open('/mnt/user-data/outputs/fieldfig/qmc_seed_table.json','w'),indent=1)

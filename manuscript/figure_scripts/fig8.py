import numpy as np, json
from style import *
R=json.load(open('/mnt/user-data/outputs/hartree/ratios.json'))
kB={'hub':0.154,'runs3':0.2,'m1partner':0.5,'fourfam':0.533,'kappa89':0.889,'kappa1':1.0,'z5':1.0,'dice':2.0}
keys=['hub','runs3','m1partner','fourfam','kappa89','kappa1','z5','dice']
rH=np.array([r['r2'] for r in R]); rQ=np.array([r['rq'] for r in R]); K=np.array([kB[k] for k in keys])
M=json.load(open('/mnt/user-data/outputs/qmc_stage3/qmc_inputs/manifest.json'))
meas3={'small(4, 0, 6, 2) t=0.400':1.165,'4-2-2-2|0.500':1.222,'5-0-3-3|0.417':1.222,'6-0-6-2|0.333':1.113}
b=[]
for r in M:
    if r['phase_key'] in meas3 and r['name'].endswith(('L4','L3')) :
        s,t=r['sigma'],r['tau']; m=t/(2*(1+s)); b.append((4*m/(1-m),r['r_H'],meas3[r['phase_key']],r['r_QMC_predicted']))
b=np.array(b)
# blind test II: phases found by the all-sector search; Hartree prediction recorded before QMC
NP=json.load(open('/mnt/user-data/outputs/fieldfig/newphases.json')); TB={r['phase']:r for r in json.load(open('/mnt/user-data/outputs/fieldfig/qmc_seed_table.json'))}
b2=[]
for k,v in NP.items():
    if k not in TB: continue
    m=v['tau']/(2*(1+v['sigma'])); rh_=1+(v['ratio']-1.114)/1.152
    b2.append((4*m/(1-m),rh_,TB[k]['ratio'],v['ratio'],TB[k]['err']/v['eps']))
b2=np.array(b2); dev=b2[:,2]-b2[:,3]
print('blind II: n=%d rms=%.4f mean=%+.4f max=%.4f'%(len(b2),np.sqrt((dev**2).mean()),dev.mean(),abs(dev).max()))
fig,axs=plt.subplots(1,2,figsize=(DCOL,2.45),gridspec_kw=dict(wspace=0.28))
ax=axs[0]
ax.plot(K,rQ,'o',color='#0072B2',ms=5,label='fit set (8 phases)'); ax.plot(b[:,0],b[:,2],'*',color='#D55E00',ms=8,label='blind test I (4 phases)'); ax.errorbar(b2[:,0],b2[:,2],yerr=b2[:,4],fmt='^',color='#009E73',ms=4,mfc='w',mew=0.9,elinewidth=0.7,label='blind test II (%d phases)'%len(b2))
ax.axhline(1,color='0.6',lw=0.6,ls='--')
for k,x,y,dx,dy in (('hub',0.154,0.892,0.06,-0.005),('dice',2.0,1.25,-0.22,-0.035),('z5',1.0,1.287,0.06,-0.06)):
    ax.text(x+dx,y+dy,{'hub':'isolated hub','dice':'dice','z5':'$z=5$ crystal'}[k],fontsize=7.5)
ax.set_xlabel(r'mean minority charge $\kappa_B$'); ax.set_ylabel(r'$\varepsilon_{\rm QMC}/\varepsilon_{\rm LSWT}$'); ax.set_xlim(0,2.15); ax.set_ylim(0.82,1.34)
ax.legend(frameon=False,loc='lower right',fontsize=7.5)
ax=axs[1]; x=np.linspace(0.78,1.17,10)
ax.plot(x,1.114+1.152*(x-1),'-',color='0.4',lw=1,label=r'$1.114+1.152\,(r_H-1)$')
ax.plot(rH,rQ,'o',color='#0072B2',ms=5,label='fit set'); ax.plot(b[:,1],b[:,2],'*',color='#D55E00',ms=8,label='blind test I'); ax.errorbar(b2[:,1],b2[:,2],yerr=b2[:,4],fmt='^',color='#009E73',ms=4,mfc='w',mew=0.9,elinewidth=0.7,label='blind test II')
ax.set_xlabel(r'Hartree ratio $r_H=1+\Delta E_2/\Delta E_1$'); ax.set_ylabel(r'$r_{\rm QMC}$'); ax.set_xlim(0.77,1.18); ax.set_ylim(0.82,1.34)
ax.legend(frameon=False,loc='upper left',fontsize=7.5)
fig.text(0.075,0.99,'(a)',fontsize=10,fontweight='bold',va='top'); fig.text(0.53,0.99,'(b)',fontsize=10,fontweight='bold',va='top')
plt.savefig('figs/fig8.pdf'); plt.savefig('figs/fig8.png',dpi=200)


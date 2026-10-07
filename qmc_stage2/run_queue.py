"""Stage-2 SSE queue (resumable, parallel).  python run_queue.py -j 4 [--extra-seeds 2]"""
import json, os, sys, time, argparse
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'sse'))
IN=os.path.join(HERE,'qmc_inputs'); OUT=os.path.join(HERE,'runs'); os.makedirs(OUT,exist_ok=True)
JOBS=[(n,300,40,500) for n in ('m1partner_s0.5_t0.333_L4','hub_s0.5_t0.111_L4','kappa1_s0.25_t0.5_L4','kappa89_s0.375_t0.5_L2',
                              'kappa89_s0.375_t0.5_L4','fourfam_4141_t0.5_L2','fourfam_4141_t0.5_L4','runs3_d5_L15')]
JOBS+=[('m1partner_s0.5_t0.333_L6',300,30,400),('kappa1_s0.25_t0.5_L6',300,30,400),('runs3_d5_L30',300,30,400),('hub_s0.5_t0.111_L4',600,30,400)]
def run(job):
    name,beta,nb,spb,seed=job
    fn=os.path.join(OUT,f'{name}_b{beta}_s{seed}.json')
    if os.path.exists(fn): return 'skip '+os.path.basename(fn)
    from sse_det import simulate
    g=json.load(open(os.path.join(IN,name+'.json'))); t=time.time()
    r=simulate(g['N'],g['bonds'],float(beta),0.0,1000,nb,spb,seed,verbose=False)
    r.update(graph=name,R=g['R'],E_LSWT=g['E_LSWT'],eps_LSWT=g['eps_LSWT'],wall_s=round(time.time()-t,1))
    json.dump(r,open(fn+'.tmp','w'),indent=1); os.replace(fn+'.tmp',fn)
    return f"{name} beta={beta} seed={seed}: E={r['E']:.5f}+-{r['E_err']:.5f} (LSWT {g['E_LSWT']:.5f}) {r['wall_s']}s"
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('-j',type=int,default=1); ap.add_argument('--extra-seeds',type=int,default=0); a=ap.parse_args()
    jobs=[(n,b,nb,s,seed) for n,b,nb,s in JOBS for seed in range(1,2+a.extra_seeds)]
    jobs.sort(key=lambda j:-j[1]*json.load(open(os.path.join(IN,j[0]+'.json')))['N'])
    with Pool(a.j) as p:
        for m in p.imap_unordered(run,jobs): print(m,flush=True)

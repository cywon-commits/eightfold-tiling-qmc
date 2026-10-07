"""Stage-3 SSE queue (resumable, parallel):  python run_queue.py -j 4"""
import json, os, sys, time, argparse
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'sse'))
IN=os.path.join(HERE,'qmc_inputs'); OUT=os.path.join(HERE,'runs'); os.makedirs(OUT,exist_ok=True)
NAMES=[n for n in ('p4062_t0.400_L2','p4062_t0.400_L4','p4222_t0.500_L2','p4222_t0.500_L4','p5033_t0.417_L2','p5033_t0.417_L4','p6062_t0.333_L2','p6062_t0.333_L3')]
def run(job):
    name,beta,nb,spb,seed=job; fn=os.path.join(OUT,f'{name}_b{beta}_s{seed}.json')
    if os.path.exists(fn): return 'skip '+os.path.basename(fn)
    from sse_det import simulate
    g=json.load(open(os.path.join(IN,name+'.json'))); t=time.time()
    r=simulate(g['N'],g['bonds'],float(beta),0.0,1000,nb,spb,seed,verbose=False)
    r.update(graph=name,R=g['R'],E_LSWT=g['E_LSWT'],eps_LSWT=g['eps_LSWT'],phase_key=g['phase_key'],r_QMC_predicted=g['r_QMC_predicted'],wall_s=round(time.time()-t,1))
    json.dump(r,open(fn+'.tmp','w'),indent=1); os.replace(fn+'.tmp',fn)
    return f"{name}: E={r['E']:.5f}+-{r['E_err']:.5f} (LSWT {g['E_LSWT']:.5f}) {r['wall_s']}s"
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('-j',type=int,default=1); a=ap.parse_args()
    jobs=[(n,300,40,500,1) for n in NAMES]
    jobs.sort(key=lambda j:-json.load(open(os.path.join(IN,j[0]+'.json')))['N'])
    with Pool(a.j) as p:
        for m in p.imap_unordered(run,jobs): print(m,flush=True)

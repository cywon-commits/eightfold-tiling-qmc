import json, sys, time
from build import bfs
from run_enum_eval import evaluate
f=sys.argv[1]; CAP=int(sys.argv[2]); S=json.load(open(f)); out=[]; t=time.time(); nd=0; ns=0
for r in S:
    st,done=bfs(r['T'],CAP); nd+=done; ns+=len(st); recs=[]
    for o in st:
        ev=evaluate(o,r['N'])
        if ev is None or ev.get('sq_only'): continue
        ev['tiles']=[(R.tolist(),int(a),int(b)) for R,a,b in o.tiles]; recs.append(ev)
    out.append(dict(T=r['T'],p=r['p'],done=done,states=len(st),recs=recs))
json.dump(out,open(f.replace('sectors_','enum_'),'w')); print(f,'sectors',len(S),'exhausted',nd,'states',ns,f'{time.time()-t:.0f}s')

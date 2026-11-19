"""Square lattice L x L torus with family-3 ribbons inserted along staircase paths (steps H=+x, V=-y).
Each path vertex is split into an upper(left) copy and a lower(right) copy joined by a v3 edge."""
import numpy as np
from lswt import lswt_energy
ESQ=-0.6579474209435209
D={'R':(1,0),'L':(-1,0),'U':(0,1),'D':(0,-1)}
ANG={(1,0):0,(0,1):90,(-1,0):180,(0,-1):270}
def build(L,paths):
    """paths: list of (start(x,y), word) ; word letters H,V ; each path must close on the torus."""
    onp={}   # vertex -> (path id, index, a_in, b_out)
    for pid,(st,w) in enumerate(paths):
        x,y=st; pts=[(x%L,y%L)]
        for c in w:
            x,y=(x+1,y) if c=='H' else (x,y-1); pts.append((x%L,y%L))
        assert pts[-1]==pts[0], 'path not closed'
        pts=pts[:-1]; n=len(pts)
        for k,p in enumerate(pts):
            assert p not in onp, 'paths overlap'
            a=(1,0) if w[k-1]=='H' else (0,-1)        # incoming step direction
            b=(1,0) if w[k]=='H' else (0,-1)          # outgoing
            onp[p]=(pid,k,a,b,pts[(k+1)%n],pts[k-1])
    vid={}
    def V(p,side=None):
        key=(p,side) if p in onp else (p,None); return vid.setdefault(key,len(vid))
    def side_of(p,u):
        # u: direction from p to an off-path neighbour; left of travel = 'U' copy
        pid,k,a,b,nx,pv=onp[p]
        ab=ANG[b]; ain=ANG[(-a[0],-a[1])]; au=ANG[u]
        # counterclockwise from outgoing b: directions met before reaching reversed-incoming are on the left
        return 'U' if (au-ab)%360 < (ain-ab)%360 else 'D'
    E=set()
    def add(i,j): E.add((min(i,j),max(i,j)))
    for x in range(L):
        for y in range(L):
            p=(x,y)
            for u in ((1,0),(0,1)):
                q=((x+u[0])%L,(y+u[1])%L)
                if p in onp and q in onp and (onp[p][4]==q or onp[q][4]==p) and onp[p][0]==onp[q][0]:
                    add(V(p,'U'),V(q,'U')); add(V(p,'D'),V(q,'D')); continue
                sp=side_of(p,u) if p in onp else None
                sq=side_of(q,(-u[0],-u[1])) if q in onp else None
                add(V(p,sp),V(q,sq))
    for p in onp: add(V(p,'U'),V(p,'D'))
    N=len(vid); B=np.array(sorted(E)); return N,B
def energy(L,paths):
    N,B=build(L,paths); assert len(B)==2*N, (len(B),N)
    E,S=lswt_energy(N,B); R=sum(len(w) for _,w in paths)
    dE=E-N*ESQ+2.035/np.sqrt(N)          # remove square-lattice Casimir term
    return dict(N=N,R=R,eps=dE/R,tau=2*(2*S)/R/1.0 if True else None,M=2*S)

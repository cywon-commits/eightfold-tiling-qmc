"""SSE QMC for the S=1/2 Heisenberg antiferromagnet H = J sum_<ij> S_i.S_j - h sum_i S^z_i on a bipartite graph.

Update scheme (Sandvik, AIP Conf. Proc. 1297, 135 (2010)):
  * diagonal update: insert/remove diagonal bond operators (matrix element 1/2 on antiparallel pairs)
  * deterministic loop update: ALL loops of the linked-vertex list are built once per sweep (every vertex
    belongs to exactly one loop, loops always close, no length cap) and each loop is flipped independently.
    With a uniform field the Zeeman term commutes with H_J, so it enters only as the weight exp(beta*h*M)
    of the time-0 state; each loop is then flipped with heat-bath probability r/(1+r), r = exp(beta*h*dM_loop)
    (dM_loop = change of M = sum S^z produced by flipping that loop).  h = 0 gives probability 1/2.
Cost per sweep is O(n + N), n = number of operators.  No sign problem on bipartite graphs.
Energy: <H_J> = -<n>/beta + N_b/4 (J = 1), total E = <H_J> - h <M>.
"""
import numpy as np, numba as nb, json, time, sys, argparse

@nb.njit(cache=True)
def diag_update(spin, ops, bsites, nb_, beta, n):
    L = ops.shape[0]
    for p in range(L):
        op = ops[p]
        if op == 0:
            b = np.random.randint(nb_)
            if spin[bsites[b,0]] != spin[bsites[b,1]]:
                if np.random.random()*2.0*(L-n) < beta*nb_:
                    ops[p] = 2*b+2; n += 1          # diagonal: 2b+2 ; offdiag: 2b+3  (0 = identity)
        elif op % 2 == 0:
            if np.random.random()*beta*nb_ < 2.0*(L-n+1):
                ops[p] = 0; n -= 1
        else:
            b = (op-2)//2
            spin[bsites[b,0]] = -spin[bsites[b,0]]; spin[bsites[b,1]] = -spin[bsites[b,1]]
    return n

@nb.njit(cache=True)
def loop_update(spin, ops, bsites, N, beta, h, fixm):
    L = ops.shape[0]
    vlist = -np.ones(4*L, np.int64)
    frst = -np.ones(N, np.int64); last = -np.ones(N, np.int64)
    for p in range(L):
        op = ops[p]
        if op == 0: continue
        b = (op-2)//2; v0 = 4*p
        for k in range(2):
            i = bsites[b,k]; v1 = last[i]
            if v1 != -1:
                vlist[v1] = v0+k; vlist[v0+k] = v1
            else:
                frst[i] = v0+k
            last[i] = v0+k+2
    for i in range(N):
        if frst[i] != -1:
            vlist[frst[i]] = last[i]; vlist[last[i]] = frst[i]
    # label loops
    lid = -np.ones(4*L, np.int64); nl = 0
    for v0 in range(0, 4*L, 2):
        if vlist[v0] < 0 or lid[v0] >= 0: continue
        v1 = v0
        while True:
            lid[v1] = nl; v2 = v1 ^ 1; lid[v2] = nl
            v1 = vlist[v2]
            if v1 == v0: break
        nl += 1
    # Zeeman change of each loop from the time-0 spins
    dM = np.zeros(nl, np.float64)
    for i in range(N):
        if frst[i] != -1:
            dM[lid[frst[i]]] += -spin[i]          # flipping S^z = spin/2 changes M by -spin
    flip = np.zeros(nl, np.bool_)
    for l in range(nl):
        if fixm:                                  # canonical (fixed S^z) sampling: only M-conserving loops
            flip[l] = (dM[l] == 0.0) and (np.random.random() < 0.5)
        elif h == 0.0:
            flip[l] = np.random.random() < 0.5
        else:
            r = np.exp(beta*h*dM[l]); flip[l] = np.random.random() < r/(1.0+r)
    for v in range(0, 4*L, 2):
        if lid[v] >= 0 and flip[lid[v]]:
            ops[v >> 2] ^= 1
    for i in range(N):
        if frst[i] != -1:
            if flip[lid[frst[i]]]: spin[i] = -spin[i]
        elif not fixm:                            # free spin (no operator acts on it)
            if h == 0.0:
                if np.random.random() < 0.5: spin[i] = -spin[i]
            else:
                r = np.exp(-beta*h*spin[i])
                if np.random.random() < r/(1.0+r): spin[i] = -spin[i]
    return nl

@nb.njit(cache=True)
def run_block(spin, ops, bsites, N, beta, h, n, nsweep, adjust, fixm):
    nsum = 0.0; msum = 0.0; ops_out = ops
    for s in range(nsweep):
        n = diag_update(spin, ops_out, bsites, bsites.shape[0], beta, n)
        if adjust:
            Lnew = n + n//3
            if Lnew > ops_out.shape[0]:
                new = np.zeros(Lnew, np.int64); new[:ops_out.shape[0]] = ops_out; ops_out = new
        loop_update(spin, ops_out, bsites, N, beta, h, fixm)
        nsum += n; msum += 0.5*spin.sum()
    return n, ops_out, nsum/nsweep, msum/nsweep

def simulate(N, bonds, beta, h=0.0, n_eq=2000, n_bins=50, sweeps_per_bin=200, seed=1, init_spin=None, verbose=True, fixM=None):
    """fixM: if not None, sample only the sector S^z_tot = fixM (canonical); h is then irrelevant for the dynamics."""
    np.random.seed(seed); nb.njit(lambda s: np.random.seed(s))(seed)
    bsites = np.asarray(bonds, np.int64); Nb = len(bsites)
    if init_spin is None:   # Neel state from bipartition
        from collections import deque
        sub = -np.ones(N, np.int64); adj = [[] for _ in range(N)]
        for a,b in bsites: adj[a].append(b); adj[b].append(a)
        for s0 in range(N):
            if sub[s0] >= 0: continue
            sub[s0] = 0; q = deque([s0])
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if sub[y] < 0: sub[y] = 1-sub[x]; q.append(y)
        spin = np.where(sub==0, 1, -1).astype(np.int64)
    else: spin = np.array(init_spin, np.int64)
    fixm = fixM is not None
    if fixm:                                     # raise/lower spins until S^z_tot = fixM
        target = int(round(2*fixM)); cur = int(spin.sum())
        idx = np.random.permutation(N)
        for i in idx:
            if cur == target: break
            if cur < target and spin[i] == -1: spin[i] = 1; cur += 2
            elif cur > target and spin[i] == 1: spin[i] = -1; cur -= 2
        assert cur == target, 'fixM incompatible with N'
    # beta-doubling equilibration (Sandvik): start at beta0 <= 2, duplicate the operator string at each doubling.
    # Needed with h != 0 so that the system can choose its S^z sector while loops still wind in imaginary time.
    k = 0
    while beta/2**k > 2.0: k += 1
    b_cur = beta/2**k
    ops = np.zeros(max(20, int(b_cur*Nb/2)), np.int64); n = 0
    t0 = time.time()
    per = max(50, n_eq)                      # n_eq sweeps at EVERY beta level (low levels are cheap)
    for lev in range(k+1):
        for _ in range(5):
            n, ops, _, _ = run_block(spin, ops, bsites, N, b_cur, h, n, max(1, per//5), True, fixm)
        if lev < k:
            ops = np.concatenate([ops, ops]); n *= 2; b_cur *= 2
    teq = time.time()-t0; t0 = time.time()
    nbins = []; mbins = []
    for k in range(n_bins):
        n, ops, nm, mm = run_block(spin, ops, bsites, N, beta, h, n, sweeps_per_bin, False, fixm)
        nbins.append(nm); mbins.append(mm)
    tm = time.time()-t0
    nbins = np.array(nbins); mbins = np.array(mbins)
    EJ = -nbins/beta + Nb/4.0; E = EJ - h*mbins
    def err(x): return x.std(ddof=1)/np.sqrt(len(x))
    res = dict(N=int(N), Nb=int(Nb), beta=beta, h=h, fixM=fixM, seed=seed, n_eq=n_eq, n_bins=n_bins, sweeps_per_bin=sweeps_per_bin,
               E=float(E.mean()), E_err=float(err(E)), E_J=float(EJ.mean()), E_J_err=float(err(EJ)),
               M=float(mbins.mean()), M_err=float(err(mbins)), n_ops=float(nbins.mean()), L_cut=int(ops.shape[0]),
               t_equil_s=round(teq,2), t_meas_s=round(tm,2), sweeps_per_s=round(n_bins*sweeps_per_bin/tm,1),
               bins_E=E.tolist())
    if verbose: print(f"N={N} beta={beta} h={h}: E = {res['E']:.6f} +- {res['E_err']:.6f}  M={res['M']:.3f}  <n>={res['n_ops']:.0f}  {res['sweeps_per_s']} sweeps/s")
    return res

def tau_int(bins_small, sweeps_per_bin):
    """crude integrated autocorrelation (in sweeps) from bin variance vs naive variance is not available
    from bins alone; use rebinning: tau ~ 0.5 * (var(bins of size B)*B / var(bins of size 1)) - see analyze()."""
    raise NotImplementedError

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='deterministic-loop SSE on a bond-list graph (qmc_inputs/*.json)')
    ap.add_argument('graph'); ap.add_argument('--beta', type=float, default=60); ap.add_argument('--h', type=float, default=0.0)
    ap.add_argument('--neq', type=int, default=2000); ap.add_argument('--nbins', type=int, default=50)
    ap.add_argument('--spb', type=int, default=200); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=None); ap.add_argument('--fixM', type=float, default=None, help='canonical sector S^z_tot (recommended for field studies)')
    a = ap.parse_args()
    g = json.load(open(a.graph))
    r = simulate(g['N'], g['bonds'], a.beta, a.h, a.neq, a.nbins, a.spb, a.seed, fixM=a.fixM)
    r['graph'] = g.get('name', a.graph); r['E_LSWT'] = g.get('E_LSWT')
    json.dump(r, open(a.out or f"sse_{r['graph']}_b{a.beta}_h{a.h}_s{a.seed}.json", 'w'), indent=1)

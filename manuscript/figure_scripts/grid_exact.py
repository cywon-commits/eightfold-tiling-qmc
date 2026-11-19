"""Exact de Bruijn construction on the torus: every line crossing is one tile; its four surrounding faces are vertices."""
import numpy as np, itertools
NRM = np.array([[0, 1], [1, -1], [1, 0], [1, 1]], float)
def tiling(pos):
    pos = [np.sort(np.asarray(p) % 1) for p in pos]; n = [len(p) for p in pos]
    T1 = np.array([0, n[1], n[2], n[3]]); T2 = np.array([n[0], -n[1], 0, n[3]])
    pinv = np.linalg.pinv(np.stack([T1, T2], 1).astype(float))
    def canon(v):
        v = np.atleast_2d(v); ab = np.floor(v @ pinv.T + 1e-7).astype(np.int64)
        return v - ab[:, :1] * T1 - ab[:, 1:] * T2
    def K_at(P, skip):
        K = np.zeros(4, np.int64)
        for c in range(4):
            if c in skip: continue
            u = P @ NRM[c]; fl = np.floor(u); K[c] = n[c] * int(fl) + np.searchsorted(pos[c], u - fl)
        return K
    tiles = []
    for a, b in itertools.combinations(range(4), 2):
        A = np.array([NRM[a], NRM[b]])
        for i, pa in enumerate(pos[a]):
            for j, pb in enumerate(pos[b]):
                for ia in range(-3, 4):
                    for ib in range(-3, 4):
                        P = np.linalg.solve(A, [pa + ia, pb + ib])
                        if not (0 <= P[0] < 1 and 0 <= P[1] < 1): continue
                        K = K_at(P, (a, b)); K[a] = n[a] * ia + i; K[b] = n[b] * ib + j
                        tiles.append((K, a, b))
    ea = np.eye(4, dtype=np.int64)
    corners = []; E = set(); verts = {}
    def vid(v):
        t = tuple(canon(v)[0]); 
        if t not in verts: verts[t] = len(verts)
        return verts[t]
    for K, a, b in tiles:
        q = [vid(K), vid(K + ea[a]), vid(K + ea[a] + ea[b]), vid(K + ea[b])]
        for s, t in ((0, 1), (1, 2), (2, 3), (3, 0)): E.add((min(q[s], q[t]), max(q[s], q[t])))
    B = sorted(E); N = len(verts); z = np.bincount(np.array(B).ravel(), minlength=N)
    sq = sum(1 for _, a, b in tiles if b - a == 2)
    return N, B, z, dict(tiles=len(tiles), squares=sq, rhombi=len(tiles) - sq, T=(T1.tolist(), T2.tolist()), raw=tiles, verts=verts, canon=canon)

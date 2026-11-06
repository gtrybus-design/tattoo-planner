"""MakeHuman base mesh parts (hands; later feet and head), fitted onto our anatomy.

The MakeHuman base mesh (CC0, see assets/makehuman/SOURCE.md) has properly modelled hands: knuckles, webbing
between the fingers and nails. A part is fitted in two steps:
1. rough: its joint markers are lined up with our rig's joints (a similarity transform, then a smooth
   radial-basis-function warp so every joint lands exactly on ours);
2. fine: a smooth displacement nudges it toward the old hand surface (a rough guide for thickness) and out of
   our bones, so it keeps its own detail (only a smooth offset is added) but sits on our anatomy.
"""
import json
import numpy as np
from fastgeo import add_at, nearest_index, nsum

# MakeHuman finger k (1 = thumb ... 5 = little finger) -> ours; MakeHuman joint markers k-1 .. k-4 are, for the
# thumb, the base of its metacarpal, the knuckle, the middle joint and the tip; for the fingers the knuckle,
# middle joint, end joint and tip (the end of the last finger bone, inside the fingertip)
MH_FINGER = {1: 'thumb', 2: 'f_index', 3: 'f_middle', 4: 'f_ring', 5: 'f_pinky'}


def load_mh(path_dir):
    """Body vertices (metres), body faces (quads), joint marker centres and the rig's weights."""
    V, F, G, g = [], [], [], None
    for line in open(f'{path_dir}/base.obj'):
        if line.startswith('v '): V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('g '): g = line.split()[1]
        elif line.startswith('f '):
            F.append([int(x.split('/')[0]) - 1 for x in line.split()[1:]]); G.append(g)
    V = np.array(V) / 10.0                                       # (decimetres -> metres)
    marks = {}
    for f, g in zip(F, G):
        if g.startswith('joint-'): marks.setdefault(g, set()).update(f)
    marks = {k: V[sorted(v)].mean(0) for k, v in marks.items()}
    body = np.array([f for f, g in zip(F, G) if g == 'body'])
    T = quads_to_tris(body)
    if np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum() < 0:
        body = body[:, ::-1]                                     # (faces outward)
    W = json.load(open(f'{path_dir}/default_weights.mhw'))['weights']
    return V, body, marks, W


def mh_bone_to_ours(b):
    """MakeHuman hand / forearm bone -> our rig bone (without side), or None."""
    side = b[-2:]
    n = b[:-2]
    if n == 'wrist' or n.startswith('metacarpal'): return 'hand'
    if n.startswith('finger'):
        k, j = n[6:].split('-')
        return MH_FINGER[int(k)] + '.0' + j
    if n.startswith('lowerarm'): return 'forearm'
    return None


def hand_landmarks_mh(marks, s):
    """MakeHuman joint markers for one hand (s = 'l' or 'r'), in the order of hand_landmarks_ours."""
    P = [marks[f'joint-{s}-hand'], marks[f'joint-{s}-elbow']]
    for k in range(1, 6):
        P += [marks[f'joint-{s}-finger-{k}-{j}'] for j in range(1, 5)]
    return np.array(P)


def hand_landmarks_ours(heads, tips, idx, sfx):
    """Our joints in the same order: wrist, elbow, then each finger base -> tip (thumb from its metacarpal)."""
    P = [heads[idx['hand' + sfx]], heads[idx['forearm' + sfx]]]
    for k in range(1, 6):
        f_ = MH_FINGER[k]
        P += [heads[idx[f'{f_}.0{j}{sfx}']] for j in (1, 2, 3)] + [tips[idx[f'{f_}.03{sfx}']]]
    return np.array(P)


def similarity(A, B):
    """Scale, rotation and shift taking points A onto B (least squares)."""
    mA, mB = A.mean(0), B.mean(0); a_, b_ = A - mA, B - mB
    U, S, Vt = np.linalg.svd(b_.T @ a_ / len(A)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (a_ ** 2).sum(1).mean()
    return lambda P: (P - mA) @ (s * R).T + mB


def rbf_warp(A, B, reg=1e-6):
    """Smooth warp taking points A exactly onto B (biharmonic radial basis functions plus an affine part)."""
    n = len(A)
    K = np.linalg.norm(A[:, None] - A[None], axis=-1) + reg * np.eye(n)
    Pm = np.c_[np.ones(n), A]
    M = np.zeros((n + 4, n + 4)); M[:n, :n] = K; M[:n, n:] = Pm; M[n:, :n] = Pm.T
    rhs = np.zeros((n + 4, 3)); rhs[:n] = B
    sol = np.linalg.solve(M, rhs)
    w, aff = sol[:n], sol[n:]

    def f(P):
        return np.linalg.norm(P[:, None] - A[None], axis=-1) @ w + np.c_[np.ones(len(P)), P] @ aff
    return f


def quads_to_tris(Q):
    return np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])


def vertex_normals(V, T):
    N = np.zeros_like(V)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    for k in range(3): add_at(N, T[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def smooth_field(D, T, n, passes):
    i = np.concatenate([T[:, 0], T[:, 1], T[:, 2], T[:, 1], T[:, 2], T[:, 0]])
    j = np.concatenate([T[:, 1], T[:, 2], T[:, 0], T[:, 0], T[:, 1], T[:, 2]])
    deg = np.maximum(np.bincount(i, minlength=n), 1)[:, None]
    for _ in range(passes):
        acc = nsum(i, j, D); D = 0.5 * D + 0.5 * acc / deg
    return D


def nearest(Q, pool):
    """Index of the nearest pool point for every query point (a KD-tree: tools/fastgeo.py)."""
    return nearest_index(Q, pool)


def fit_part(mh, s, A, ours, joint, prox, bones_P, guide_P, start=-0.06, reach_mh=0.25, clear=0.0025, pull=0.5,
             reach=0.004, floor=None, fade=-0.01):
    """Fit one MakeHuman part (a hand or a foot). A / ours: matching landmarks (MakeHuman's, ours); joint, prox: our
    joint where the part starts (wrist, ankle) and the one above it (elbow, knee); MakeHuman's are A[0], A[1].
    Takes the part and from `start` (metres along the limb, - toward the body) on. Returns its points, triangles,
    weights by MakeHuman bone, and where each point sits along the limb from our joint.
    floor: if given, the sole is kept flat on that height (feet). fade: where along the limb the fine fit fades in
    (over 2 cm from there)."""
    V, body, marks, W = mh
    ax_mh = (A[0] - A[1]) / np.linalg.norm(A[0] - A[1])
    side = V[:, 0] > 0.05 if s == 'l' else V[:, 0] < -0.05
    keep_v = side & ((V - A[0]) @ ax_mh > start) & (np.linalg.norm(V - A[0], axis=1) < reach_mh)   # (not the hip etc.)
    Q = body[keep_v[body].all(1)]
    used = np.unique(Q); remap = np.full(len(V), -1); remap[used] = np.arange(len(used))
    P = V[used]; T = quads_to_tris(remap[Q])
    # 1. rough: similarity, then the radial-basis warp onto our landmarks
    sim = similarity(A, ours)
    P = sim(P); A2 = sim(A)
    P = rbf_warp(A2, ours)(P)
    u = (joint - prox) / np.linalg.norm(joint - prox)
    if floor is not None: P = flatten_sole(P, T, floor)
    # 2. fine: a smooth displacement toward the guide surface (by `pull` of the way, at most `reach`) and out of
    # the bones (at least `clear` from them), repeated a few times
    on_part = np.clip(((P - joint) @ u - fade) / 0.02, 0, 1)[:, None]   # (the part above the cut is cut off anyway)
    for _ in range(4):
        N = vertex_normals(P, T)
        g = guide_P[nearest(P, guide_P)]
        off = np.clip(((g - P) * N).sum(1), -reach, reach) * pull
        b = bones_P[nearest(P, bones_P)]
        d_b = ((P - b) * N).sum(1)
        out = np.maximum(clear - d_b, 0) * (np.linalg.norm(P - b, axis=1) < 0.02)
        D = smooth_field(N * (off + out)[:, None], T, len(P), 20) * on_part
        if floor is not None: D[:, 1] *= np.clip((P[:, 1] - floor) / 0.01, 0, 1)    # (the sole stays put)
        P = P + D
    if floor is not None: P = flatten_sole(P, T, floor)          # (again: the fine fit may have lifted it)
    # weights by MakeHuman bone, for the points kept
    Wp = {}
    for b_, iw in W.items():
        if b_[-2:] != ('.L' if s == 'l' else '.R'): continue
        for i, w in iw:
            if remap[i] >= 0: Wp.setdefault(b_, []).append((remap[i], w))
    return P, T, Wp, (P - joint) @ u


def flatten_sole(P, T, floor):
    """The sole flat on the floor: a plane is fitted through the sole (points facing down, in the bottom 2.5 cm)
    and the bottom 3 cm above it moved up or down so that plane lies on the floor (fading out by 3 cm up), so the foot
    stands flat whatever tilt the warp gave it."""
    N = vertex_normals(P, T)
    sole_ = (N[:, 1] < -0.7) & (P[:, 1] < P[:, 1].min() + 0.025)
    a_, *_ = np.linalg.lstsq(np.c_[P[sole_][:, [0, 2]], np.ones(sole_.sum())], P[sole_][:, 1], rcond=None)
    # (the plane through the sole's lowest layer: shifted down to its lowest points)
    plane = np.c_[P[:, [0, 2]], np.ones(len(P))] @ a_
    plane += (P[sole_, 1] - plane[sole_]).min()
    f_ = np.clip(1 - (P[:, 1] - plane) / 0.03, 0, 1)
    P = P.copy(); P[:, 1] -= (plane - floor) * f_
    return P


def fit_hand(mh, s, ours, wrist, elbow, bones_P, guide_P):
    """One hand: from 6 cm up the forearm, landmarks as hand_landmarks_mh / hand_landmarks_ours."""
    return fit_part(mh, s, hand_landmarks_mh(mh[2], s), ours, wrist, elbow, bones_P, guide_P)


def foot_landmarks_mh(mh, s):
    """MakeHuman foot landmarks: ankle, knee, heel (the back of the heel's skin, 1 cm up), then for each toe its
    base knuckle and tip (the big toe also its middle joint)."""
    V, body, marks, W = mh
    P = [marks[f'joint-{s}-ankle'], marks[f'joint-{s}-knee'], heel_back(V, marks[f'joint-{s}-ankle'], s)]
    P += [marks[f'joint-{s}-toe-1-1'], marks[f'joint-{s}-toe-1-2'], marks[f'joint-{s}-toe-1-3']]
    for k in range(2, 6): P += [marks[f'joint-{s}-toe-{k}-1'], marks[f'joint-{s}-toe-{k}-4']]
    return np.array(P)


def heel_back(V, ankle, s, up=0.01):
    """The back of the heel: the rearmost skin point within 1 cm of `up` above the lowest point near the ankle."""
    side = V[:, 0] > 0.05 if s == 'l' else V[:, 0] < -0.05
    near = side & (np.linalg.norm(V - ankle, axis=1) < 0.12)
    y0 = V[near, 1].min()
    band = near & (np.abs(V[:, 1] - (y0 + up)) < 0.005)
    return V[band][np.argmin(V[band, 2])]


def mh_leg_bone_to_ours(b):
    n = b[:-2]
    if n == 'foot': return 'foot'
    if n.startswith('toe'): return 'toe'
    if n.startswith('lowerleg'): return 'shin'
    return None


def boundary_loop(T, n):
    """The single boundary loop of an open triangle mesh, as an ordered list of point indices."""
    E = {}
    for t in T:
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            E[(a, b)] = E.get((a, b), 0) + 1
    bnd = [(a, b) for (a, b) in E if (b, a) not in E]
    nxt = {a: b for a, b in bnd}
    loops, seen = [], set()
    for a0 in nxt:
        if a0 in seen: continue
        loop, a = [], a0
        while a not in seen:
            seen.add(a); loop.append(a); a = nxt.get(a)
            if a is None: break
        loops.append(loop)
    return loops


def bridge(P, la, lb, u, c):
    """Triangles joining two loops of points (indices into P) that run around the axis u through c. Each loop comes
    in the order its own mesh runs along its open edge. Both are walked the same way round, starting from their
    closest pair, and zipped by how far along each loop the points are (so it works for any loop shape). Each
    triangle runs its loop edge the opposite way to that loop's own mesh, so it faces the same way as both."""
    la, lb = np.asarray(la), np.asarray(lb)
    own = {(int(x), int(y)) for l in (la, lb) for x, y in zip(l, np.roll(l, -1))}     # (each mesh's own direction)
    e1 = np.cross(u, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)

    def turn(l):          # which way round the axis a loop runs (+1 / -1)
        q = P[l] - c; x, y = q @ e1, q @ e2
        return np.sign(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    if turn(la) != turn(lb): lb = lb[::-1]
    lb = np.roll(lb, -int(np.argmin(np.linalg.norm(P[lb] - P[la[0]], axis=1))))

    def along(l):         # how far along the loop each point is (0 to 1), plus 1 at the end
        d = np.linalg.norm(P[np.roll(l, -1)] - P[l], axis=1)
        return np.r_[0, np.cumsum(d)] / d.sum()
    ta, tb = along(la), along(lb)
    na, nb = len(la), len(lb)
    tris, i, j = [], 0, 0
    while i < na or j < nb:
        if j >= nb or (i < na and ta[i + 1] <= tb[j + 1]):
            tris.append((la[i], la[(i + 1) % na], lb[j % nb])); i += 1
        else:
            tris.append((la[i % na], lb[(j + 1) % nb], lb[j])); j += 1
    T = np.array(tris)
    flip = np.array([any((int(t[k]), int(t[(k + 1) % 3])) in own for k in range(3)) for t in T])
    T[flip] = T[flip][:, ::-1]
    return T


# ---------------------------------------------------------------- head
def mh_head_bone_to_ours(b):
    """MakeHuman neck / head / face bone -> ours: the lower neck rides our neck, everything above it our head."""
    if b in ('neck01', 'neck02'): return 'spine.004'
    if b.startswith(('neck03', 'head', 'jaw', 'eye', 'tongue', 'levator', 'oculi', 'orbicularis', 'oris', 'risorius',
                     'temporalis', 'special', 'mylohyoid', 'platysma', 'masseter', 'depressor', 'zygomatic', 'nasal',
                     'mentalis', 'buccinator', 'procerus', 'corrugator', 'frontalis')): return 'spine.005'
    return None


def mh_face_landmarks(mh):
    """Skin landmarks on the MakeHuman head, found from its shape (metres, MakeHuman's own frame).
    Midline (|x| < 3 mm): glabella, nasion, nose tip, subnasale, upper / lower lip, chin front and bottom, top and
    back of the head; each side: cheekbone (zygion), the ear's centre; plus the eye centres and the neck joint."""
    V, body, marks, W = mh
    eL, eR = marks['joint-l-eye'], marks['joint-r-eye']
    ye, ze = (eL[1] + eR[1]) / 2, (eL[2] + eR[2]) / 2
    head = np.unique(body[(V[body][..., 1] > marks['joint-neck'][1]).all(1)])
    H = V[head]
    M = H[np.abs(H[:, 0]) < 0.003]
    F = M[M[:, 2] > ze - 0.02]                                  # (the face: the front of the midline)
    # its outer profile: the most forward point every 2 mm of height, then dropping any more than 5 mm behind both
    # its neighbours (the inside of the mouth or a nostril, where a slice missed the outer surface)
    yb = np.round(F[:, 1] / 0.002).astype(int)
    F = np.array([F[yb == b][np.argmax(F[yb == b, 2])] for b in np.unique(yb)])
    for _ in range(5):
        z = F[:, 2]; inner = np.r_[False, (z[1:-1] < np.minimum(z[:-2], z[2:]) - 0.005), False]
        if not inner.any(): break
        F = F[~inner]
    band = lambda lo, hi: F[(F[:, 1] > lo) & (F[:, 1] < hi)]
    prn = F[np.argmax(F[:, 2])]
    g_ = band(ye + 0.005, ye + 0.035); g = g_[np.argmax(g_[:, 2])]
    n_ = band(ye - 0.01, g[1]); n = n_[np.argmin(n_[:, 2])]
    # the mouth line: half-way between the upper and lower lip muscles (oris05, oris01); the lips are the most
    # forward points within 12 mm above and below it, subnasale the deepest point between the upper lip and the
    # nose tip; then the chin: its lowest point and its most forward point just above that (MakeHuman's profile is
    # nearly flat from the lower lip down, so the mentolabial sulcus is not used as a landmark)
    lip = lambda b: np.average(V[[i for i, w in W[b] if w > 0.3]], axis=0, weights=[w for i, w in W[b] if w > 0.3])
    st_y = (lip('oris05')[1] + lip('oris01')[1]) / 2
    ls_ = band(st_y, st_y + 0.012); ls = ls_[np.argmax(ls_[:, 2])]
    li_ = band(st_y - 0.012, st_y); li = li_[np.argmax(li_[:, 2])]
    sn_ = band(ls[1], prn[1]); sn = sn_[np.argmin(sn_[:, 2])]
    me_ = M[(M[:, 2] > ze) & (M[:, 1] < li[1] - 0.02)]; me = me_[np.argmin(me_[:, 1])]
    pg_ = band(me[1] + 0.005, me[1] + 0.022); pg = pg_[np.argmax(pg_[:, 2])]
    top = M[np.argmax(M[:, 1])]
    op_ = M[M[:, 1] > ye]; op = op_[np.argmin(op_[:, 2])]
    out = {'eye.L': eL, 'eye.R': eR, 'g': g, 'n': n, 'prn': prn, 'sn': sn, 'ls': ls, 'li': li, 'pg': pg, 'me': me,
           'top': top, 'op': op, 'neck': marks['joint-neck']}
    for s_, sg in (('L', 1), ('R', -1)):
        zy_ = H[(np.abs(H[:, 1] - (ye - 0.01)) < 0.01) & (H[:, 2] > ze - 0.055) & (H[:, 2] < ze - 0.01)]
        out['zy.' + s_] = zy_[np.argmax(zy_[:, 0] * sg)]
        ear = H[(H[:, 0] * sg > 0.06) & (H[:, 1] > ye - 0.05) & (H[:, 1] < ye + 0.02) & (H[:, 2] < ze - 0.06)]
        out['ear.' + s_] = ear.mean(0)
    return out


def close_openings(P, T, lm, W, remap):
    """Close the eyes and the mouth (in MakeHuman's own frame, before any warping): the insides of the eye sockets
    (behind the eyelid rims) and of the mouth (every point with skin right in front of it) are deleted, and each
    opening left is capped with a small clean patch (a ring half-way in and a middle point).
    Returns the new points and triangles, where each old point went (-1: deleted), and how many were deleted."""
    lip = lambda b: np.average(np.array([P[remap[i]] for i, w in W[b] if w > 0.3 and remap[i] >= 0]), axis=0)
    st = (lip('oris05') + lip('oris01')) / 2
    mouth_c = np.array([0, st[1], lm['ls'][2]])
    inside = np.zeros(len(P), bool)
    for k in ('eye.L', 'eye.R'):
        c = lm[k]; d = P - c; lat = np.linalg.norm(d[:, :2], axis=1)
        inside |= (lat < 0.016) & (d[:, 2] < 0.009) & (d[:, 2] > -0.04)
    d = P - mouth_c
    cand = np.where((np.abs(d[:, 0]) < 0.04) & (np.abs(d[:, 1]) < 0.035) & (d[:, 2] > -0.07))[0]
    Pc = P[cand]
    for k in range(0, len(cand), 256):
        q = Pc[k:k + 256]
        cover = (np.abs(q[:, None, 0] - Pc[None, :, 0]) < 0.004) & (np.abs(q[:, None, 1] - Pc[None, :, 1]) < 0.004) & \
                (Pc[None, :, 2] - q[:, None, 2] > 0.003)
        inside[cand[k:k + 256][cover.any(1)]] = True
    keepT = T[~inside[T].any(1)]
    alive = np.zeros(len(P), bool); alive[keepT.ravel()] = True
    newi = np.full(len(P), -1); newi[alive] = np.arange(alive.sum())
    P2, T2 = P[alive], newi[keepT]
    caps = []
    for loop in boundary_loop(T2, len(P2)):
        cen = P2[loop].mean(0)
        if cen[1] < lm['me'][1] - 0.01: continue                 # (the neck's open end stays open)
        is_eye = min(np.linalg.norm(cen - lm[k]) for k in ('eye.L', 'eye.R')) < 0.03
        # (eyes slightly forward, the mouth slightly back; any leftover edge deep inside the head just closed flat)
        push = np.array([0, 0, -0.001 if (not is_eye and np.linalg.norm(cen - mouth_c) < 0.03) else 0])
        ring = (P2[loop] + cen) / 2 + push * 0.6
        r0 = len(P2); c0 = r0 + len(loop)
        P2 = np.vstack([P2, ring, cen + push])
        n = len(loop)
        for k in range(n):
            a_, b_ = loop[k], loop[(k + 1) % n]
            ra, rb = r0 + k, r0 + (k + 1) % n
            caps += [(b_, a_, ra), (b_, ra, rb), (rb, ra, c0)]          # (the reverse of the rim's own direction)
    if caps: T2 = np.vstack([T2, np.array(caps)])
    return P2, T2, newi, int(inside.sum())


def flatten_face(P, T, lm, sigma=0.012):
    """A plain, mannequin-like face: on the front of the head (from above the brow down to the chin, in front of the
    ears) the small shapes are blurred away: each point's depth (front to back) is averaged with the face around it
    (about 1.2 cm across, seen from the front), so the eyelids, lips and creases melt into the face while the bigger
    shapes (brow, cheeks, chin) stay. The nose is left as it is and kept out of the averaging, and the change fades
    out over 1-2 cm toward the sides, the forehead, the chin and the nose. (In MakeHuman's own frame.)"""
    ze = (lm['eye.L'][2] + lm['eye.R'][2]) / 2
    sm = lambda v, lo, hi: np.clip((v - lo) / (hi - lo), 0, 1)
    # (fading out above the chin, so the chin keeps its shape)
    face = sm(P[:, 2], lm['ear.L'][2] + 0.015, lm['ear.L'][2] + 0.04) * sm(P[:, 1], lm['me'][1] + 0.008, lm['me'][1] + 0.025) * \
           sm(-P[:, 1], -(lm['g'][1] + 0.045), -(lm['g'][1] + 0.025))
    nose = sm(-np.abs(P[:, 0]), -0.026, -0.016) * sm(P[:, 1], lm['sn'][1] - 0.012, lm['sn'][1] - 0.004) * \
           sm(-P[:, 1], -(lm['n'][1] + 0.006), -(lm['n'][1] - 0.002)) * sm(P[:, 2], ze + 0.004, ze + 0.014)
    N = vertex_normals(P, T)
    front = sm(N[:, 2], 0.0, 0.3)
    w = smooth_field((face * (1 - nose) * front)[:, None], T, len(P), 6)[:, 0]
    src = np.where((nose < 0.05) & (N[:, 2] > 0.6) & (P[:, 2] > lm['ear.L'][2] + 0.03))[0]   # (what's averaged: the front)
    on = np.where(w > 1e-3)[0]
    z = P[:, 2].copy()
    for k in range(0, len(on), 512):
        q = P[on[k:k + 512]]
        g = np.exp(-(((q[:, None, :2] - P[None, src, :2]) ** 2).sum(-1)) / (2 * sigma ** 2))
        z[on[k:k + 512]] = (g * P[src, 2]).sum(1) / np.maximum(g.sum(1), 1e-12)
    P0 = P
    P = P.copy()
    P[:, 2] = (1 - w) * P[:, 2] + w * z
    P = smooth_mesh(P, T, w, 6)
    # any triangle the blur turned inside out (where the face folded under: the nostrils, the lip line) is smoothed
    # out with its neighbours until none are left
    n0 = np.cross(P0[T[:, 1]] - P0[T[:, 0]], P0[T[:, 2]] - P0[T[:, 0]])
    for _ in range(30):
        bad = (np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]) * n0).sum(1) < 0
        if not bad.any(): break
        fix = np.zeros(len(P)); fix[T[bad].ravel()] = 1
        P = smooth_mesh(P, T, smooth_field(fix[:, None], T, len(P), 2)[:, 0].clip(0, 1) * 2, 3)
    return P


def smooth_mesh(P, T, w, passes):
    """A few gentle smoothing passes, each point by its own amount w (0-1)."""
    i = np.concatenate([T[:, 0], T[:, 1], T[:, 2], T[:, 1], T[:, 2], T[:, 0]])
    j = np.concatenate([T[:, 1], T[:, 2], T[:, 0], T[:, 0], T[:, 1], T[:, 2]])
    deg = np.maximum(np.bincount(i, minlength=len(P)), 1)[:, None]
    for _ in range(passes):
        acc = nsum(i, j, P); P = P + 0.5 * w[:, None] * (acc / deg - P)
    return P


def dense_points(P, T, step=0.002):
    """Points spread over every triangle at most `step` apart."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    n = np.clip(np.ceil(np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                                np.linalg.norm(a - c, axis=1)], 0) / step), 1, 40).astype(int)
    out = []
    for k in np.unique(n):
        sel = np.where(n == k)[0]
        uv = np.array([(i / k, j / k) for i in range(k + 1) for j in range(k + 1 - i)])
        out.append((a[sel, None] + uv[None, :, 0:1] * (b - a)[sel, None] + uv[None, :, 1:2] * (c - a)[sel, None]).reshape(-1, 3))
    return np.vstack(out)


def accentuate(P, T, lm, nose_k=1.7, jaw_out=0.006, chin_fwd=0.004, line_out=0.003):
    """A stronger nose and jaw (in MakeHuman's own frame, before the head is made plain, so they still read after
    the blur): the nose stands out further from the face (its height above the cheeks beside it times nose_k,
    fading in from its edges); the jaw angles are pushed out sideways by jaw_out, the chin forward by chin_fwd, and
    the band along the jawline between them outward by line_out, for a squarer, more defined jaw."""
    sm = lambda v, lo, hi: np.clip((v - lo) / (hi - lo), 0, 1)
    g = lambda c, r: np.exp(-((P - c) ** 2).sum(1) / (2 * r ** 2))
    P = P.copy()
    # the nose: everything in the middle strip from the bridge to just under the tip, raised off the cheek level
    base = lm['prn'][2] - 0.0225                                     # (the cheeks beside the nose sit about here)
    nose = sm(-np.abs(P[:, 0]), -0.024, -0.012) * sm(P[:, 1], lm['sn'][1] - 0.008, lm['sn'][1]) * \
           sm(-P[:, 1], -(lm['n'][1] + 0.004), -(lm['n'][1] - 0.006)) * (P[:, 2] > base)
    P[:, 2] += nose * (P[:, 2] - base) * (nose_k - 1)
    # the jaw: its angles (the widest point 2 cm above the bottom of the chin, toward the ears) out to the sides, the
    # chin forward, and the jawline between them outward (the lower 4.5 cm of the face, in front of the ears)
    N = vertex_normals(P, T)
    y0 = lm['me'][1] + 0.02
    for sg in (1, -1):
        band = P[(np.abs(P[:, 1] - y0) < 0.003) & (P[:, 2] > lm['ear.L'][2] - 0.03) & (P[:, 2] < lm['pg'][2])]
        go = band[np.argmax(band[:, 0] * sg)]
        P[:, 0] += sg * jaw_out * g(go, 0.02) * (P[:, 0] * sg > 0)
    P[:, 2] += chin_fwd * g(lm['pg'] - [0, 0, 0.005], 0.015)
    line = sm(P[:, 1], lm['me'][1] - 0.01, lm['me'][1]) * sm(-P[:, 1], -(lm['me'][1] + 0.045), -(lm['me'][1] + 0.025)) * \
           sm(P[:, 2], lm['ear.L'][2] - 0.01, lm['ear.L'][2] + 0.01) * sm(-N[:, 1], -0.2, 0.3)
    horiz = N * [1, 0, 1]; horiz /= np.maximum(np.linalg.norm(horiz, axis=1, keepdims=True), 1e-9)
    P += horiz * (line_out * line)[:, None]
    return P


def fill_face(P, T, lm, cell=0.002, radius=0.025):
    """A flat face: no hollows or bumps on the front of the head except the nose. The face is seen from the front
    as a height map (front-to-back depth on a 2 mm grid, the nose left out), every dip narrower than about 2 x radius
    filled (a closing with a disc of that radius: the eye sockets, under the brow, around the mouth), the result
    smoothed about 1.2 cm across, and each face point set onto it; the nose is left as it is, and it all fades out
    over 1-2 cm toward the sides, the forehead, the chin and the nose. (In MakeHuman's own frame.)"""
    from scipy import ndimage
    sm = lambda v, lo, hi: np.clip((v - lo) / (hi - lo), 0, 1)
    N = vertex_normals(P, T)
    face = sm(P[:, 2], lm['ear.L'][2] + 0.02, lm['ear.L'][2] + 0.045) * sm(P[:, 1], lm['me'][1] + 0.005, lm['me'][1] + 0.02) * \
           sm(-P[:, 1], -(lm['g'][1] + 0.045), -(lm['g'][1] + 0.025)) * sm(N[:, 2], 0.0, 0.3)
    nose = sm(-np.abs(P[:, 0]), -0.026, -0.016) * sm(P[:, 1], lm['sn'][1] - 0.01, lm['sn'][1] - 0.002) * \
           sm(-P[:, 1], -(lm['n'][1] + 0.006), -(lm['n'][1] - 0.002)) * (P[:, 2] > lm['prn'][2] - 0.03)
    src = (face > 0.01) & (nose < 0.05)
    lo = np.array([-0.09, lm['me'][1] - 0.02]); shape = tuple(np.ceil(np.array([0.18, lm['g'][1] + 0.07 - lo[1]]) / cell).astype(int) + 1)
    ij = np.clip(np.floor((P[:, :2] - lo) / cell).astype(int), 0, np.array(shape) - 1)
    Hm = np.full(shape, -np.inf)
    np.maximum.at(Hm, (ij[src, 0], ij[src, 1]), P[src, 2])
    empty = ~np.isfinite(Hm)                       # (cells no face point fell in: the nearest filled cell's depth)
    _, near_ij = ndimage.distance_transform_edt(empty, return_indices=True)
    Hm = Hm[tuple(near_ij)]
    r = int(round(radius / cell)); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    Hc = ndimage.grey_closing(Hm, footprint=(xx ** 2 + yy ** 2) <= r * r)
    Hc = ndimage.gaussian_filter(Hc, 6)            # (and smoothed about 1.2 cm across: no bumps either)
    level = ndimage.map_coordinates(Hc, ((P[:, :2] - lo) / cell).T, order=1, mode='nearest')
    w = smooth_field((face * (1 - nose))[:, None], T, len(P), 4)[:, 0]
    P = P.copy()
    P[:, 2] = (1 - w) * P[:, 2] + w * level
    return P


def mannequin(P, T, cell=0.005, close=3, blur=10):
    """A plain, mannequin-like head: the head is filled in as a solid, small hollows closed (about 2 x close cells
    across) and the solid blurred, and its outline taken as the new surface, so the eyelids, lips and creases
    melt away while the overall shape, the nose and the ears stay as soft forms. Its open end (the neck) is capped
    first so it is a closed solid. Returns the new points and triangles."""
    import envelope as env
    caps, P2 = [], P
    for loop in boundary_loop(T, len(P)):
        c0 = len(P2); P2 = np.vstack([P2, P2[loop].mean(0)])
        caps += [(loop[(k + 1) % len(loop)], loop[k], c0) for k in range(len(loop))]
    T2 = np.vstack([T, np.array(caps)]) if caps else T
    return env.envelope([dense_points(P2, T2)], cell=cell, radius=0, blur_passes=blur, close=close)


def fit_head(mh, A, ours, neck, up, cut, skull_P, skull_N, eyes_mh, neck_guide=None, clear=0.003, reach=0.015):
    """Fit the MakeHuman head, keeping its own (normal) shape: its eyes and mouth are closed, it is scaled and placed
    as a whole onto a few of our landmarks (A: MakeHuman's, ours), rebuilt as a plain, mannequin-like head (see
    mannequin()), and only pushed out wherever our skull would come within
    `clear` of its skin (skull_P / skull_N: the skull's outer points and outward directions). neck, up: our neck
    base and the neck's direction; cut: where along it the head starts. neck_guide: our old neck around the cut,
    which the bottom of the MakeHuman neck is pulled onto so the join is level. eyes_mh: MakeHuman's landmarks.
    Returns points, triangles, weights by MakeHuman bone, and where each point sits along the neck from `neck`."""
    V, body, marks, W = mh
    up_mh = (marks['joint-head'] - marks['joint-neck']); up_mh /= np.linalg.norm(up_mh)
    s_mh = (V - marks['joint-neck']) @ up_mh
    rad = np.linalg.norm(V - marks['joint-neck'] - np.outer(s_mh, up_mh), axis=1)
    keep_v = (s_mh > -0.06) & ((rad < 0.11) | (s_mh > 0.02))          # (the neck, not the shoulders)
    Q = body[keep_v[body].all(1)]
    used = np.unique(Q); remap = np.full(len(V), -1); remap[used] = np.arange(len(used))
    P = V[used]; T = quads_to_tris(remap[Q])
    P, T, newi, n_closed = close_openings(P, T, eyes_mh, W, remap)
    remap = np.where(remap >= 0, newi[np.maximum(remap, 0)], -1)
    # 1. placed on our head as a whole: one size, turn and shift that best fits our landmarks (no bending, so it
    # keeps MakeHuman's normal proportions)
    P = accentuate(P, T, eyes_mh)            # (a stronger nose and jaw, still in MakeHuman's frame)
    P = fill_face(P, T, eyes_mh)             # (and a flat face: no hollows at the eyes or around the mouth)
    P = similarity(A, ours)(P)
    # 2. made plain: rebuilt as a blurred solid (no eyelids, lips or creases); each new point takes the MakeHuman
    # weights of the old point nearest it
    Pm, Tm = mannequin(P, T)
    nn = nearest(Pm, P)
    by_pt = {}
    for b_, iw in W.items():
        for i, w in iw:
            if remap[i] >= 0: by_pt.setdefault(remap[i], []).append((b_, w))
    W = {}
    for k, j in enumerate(nn):
        for b_, w in by_pt.get(j, []): W.setdefault(b_, []).append((k, w))
    P, T = Pm, Tm
    remap = None
    # 2. out of the skull where it would show through, and the bottom of the neck onto our old neck
    for _ in range(4):
        N = vertex_normals(P, T)
        s_ = (P - neck) @ up
        j = nearest(P, skull_P)
        under = ((P - skull_P[j]) * N).sum(1)
        off = np.maximum(clear - under, 0) * (np.linalg.norm(P - skull_P[j], axis=1) < 0.02)
        if neck_guide is not None:
            gN = neck_guide[nearest(P, neck_guide)]
            f_n = np.clip((cut + 0.04 - s_) / 0.03, 0, 1)
            off = off * (1 - f_n) + np.clip(((gN - P) * N).sum(1), -reach, reach) * f_n
        P = P + smooth_field(N * off[:, None], T, len(P), 20)
    s_ = (P - neck) @ up
    return P, T, W, s_, n_closed

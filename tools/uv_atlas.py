"""Flat skin map (UV atlas) for the body skin in body-rig.js, at real-world scale.

Run after tools/fit_rig.py (needs the venv: python3 -m venv .venv && .venv/bin/pip install numpy scipy libigl pillow):
    .venv/bin/python tools/uv_atlas.py .

A closed body can't be flattened without cuts and stretch, so the skin is cut into regions a tattoo
artist would recognise (torso front / back split down the sides, sleeves split along the inner arm,
legs along the inner leg, head front / back, hands and feet top / bottom, each finger and toe), and each
region is flattened with SLIM (symmetric Dirichlet), which keeps lengths and areas as close to true as it
can. Every region is laid out at the same scale, so 1 cm on the map is ~1 cm of skin.

Points on a seam are split in two. The copies go after the existing points (so the anchors and everything
else indexed by skin point stay valid), and `dup` lists the original point of each copy; the page copies
the posed position of the original onto each copy so seams never open.

Reads body-rig.js (never changes it) and writes skin-uv.js (new triangle list, seam copies, UVs),
skin-map.png (printable template: 1 cm grid, outlines, labels, stretch shading) and skin-map.js (a
skin-tone grid texture for the 3D view).
"""
import base64, json, os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra
import igl
from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1] if len(sys.argv) > 1 else '.'
t0 = time.time()
MARGIN = 0.02          # gap between regions on the map (m)
TEMPLATE_PX = 4096     # printable template
PREVIEW_PX = 2048      # texture embedded for the 3D view


def log(*a): print(f'[{time.time() - t0:5.1f}s]', *a, file=sys.stderr)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rigpack
b64 = rigpack.b64                     # (big arrays written compactly; the page unpacks them)
def dec(s, t): return np.frombuffer(base64.b64decode(s), t)


R = rigpack.load_js(f'{out}/body-rig.js')     # (unpacked back to plain base64, whatever format it was written in)
B = R['body']
p = B['position']
V = np.array(p['min']) + dec(p['q'], np.uint16).reshape(-1, 3) / 65535 * (np.array(p['max']) - np.array(p['min']))
N = dec(B['normal'], np.float32).reshape(-1, 3)
T = dec(B['index'], np.uint32).reshape(-1, 3).astype(np.int64)
SI = dec(B['skinIndex'], np.uint8).reshape(-1, 4)
W8 = dec(B['weight8'], np.uint8).reshape(-1, 4)
# which body-rig.js this map belongs to: point count, triangle count and a sum over the packed positions
fingerprint = [len(V), len(T), int(dec(p['q'], np.uint16).astype(np.int64).sum() % 2 ** 31)]
nV = len(V)
names = [b['name'] for b in R['bones']]
log('skin', nV, 'points', len(T), 'triangles')

# ---------------------------------------------------------------- clean triangles
# The surface was decimated by merging nearby points, which leaves a few doubled triangles and edges
# shared by 4+ triangles. Doubled triangles are dropped (they cover the same spot); the remaining
# non-manifold edges are simply treated as cuts.
_, keep = np.unique(np.sort(T, 1), axis=0, return_index=True)
T = T[np.sort(keep)]
T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
log('dropped', len(dec(B['index'], np.uint32)) // 3 - len(T), 'doubled triangles')


def repair(T):
    """Make the skin a clean surface: triangles facing one way, every edge shared by exactly two triangles,
    no pinched points. Triangles at a bad spot are peeled away and the small hole left is filled again
    from the points around it (no new points, so everything indexed by skin point stays valid)."""
    def edge_info(T):
        d = np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]
        _, inv, cnt = np.unique(np.sort(d, 1), axis=0, return_inverse=True, return_counts=True)
        _, dinv, dc = np.unique(d[:, 0] * nV + d[:, 1], return_inverse=True, return_counts=True)
        return cnt[inv.ravel()].reshape(3, -1).T, dc[dinv.ravel()].reshape(3, -1).T

    def fill_loop(loop):
        """Ear clipping, always cutting off the corner with the shortest new edge."""
        loop, tris = list(loop), []
        while len(loop) > 3:
            n = len(loop)
            i = int(np.argmin([np.linalg.norm(V[loop[i - 1]] - V[loop[(i + 1) % n]]) for i in range(n)]))
            tris.append([loop[i - 1], loop[i], loop[(i + 1) % n]]); loop.pop(i)
        return tris + [loop]

    for it in range(10):
        T = igl.bfs_orient(T)[0]
        for _ in range(30):
            ec, dc = edge_info(T)
            bad = (ec > 2).any(1) | (dc > 1).any(1) | ((ec == 1).sum(1) >= 2) | (~igl.is_vertex_manifold(T)[T]).any(1)
            if not bad.any(): break
            T = T[~bad]
        loops = igl.boundary_loop_all(T)
        if not loops: break
        dset = set(map(tuple, np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]].tolist()))
        fill = []
        for loop in loops:
            if (loop[0], loop[1]) in dset: loop = loop[::-1]    # the hole runs the other way from its faces
            fill += fill_loop(loop)
        T = np.vstack([T, np.array(fill, np.int64)])
        log(f'repair pass {it}: filled {len(loops)} holes')
    # each closed piece (body, every finger and toe) facing outwards
    A = sp.coo_matrix((np.ones(len(T) * 3), (T.ravel(), np.repeat(T[:, 0], 3))), (nV, nV))
    _, pc = connected_components(A, directed=False)
    vol = np.zeros(pc.max() + 1)
    np.add.at(vol, pc[T[:, 0]], np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])))
    inward = vol[pc[T[:, 0]]] < 0
    T[inward] = T[inward][:, ::-1]
    log(f'repair: {len(T)} triangles')
    return T


T = repair(T)
nF = len(T)

he = np.stack([T, np.roll(T, -1, 1)], 2).reshape(-1, 2)          # half-edge f*3+k: T[f,k] -> T[f,k+1]
ekey = np.sort(he, 1)
eu, einv, ecnt = np.unique(ekey, axis=0, return_inverse=True, return_counts=True)
einv = einv.ravel()
nE = len(eu)

# face adjacency over manifold edges (exactly 2 triangles, opposite directions)
order = np.argsort(einv, kind='stable')
first = np.searchsorted(einv[order], np.arange(nE))
two = np.where(ecnt == 2)[0]
h1, h2 = order[first[two]], order[first[two] + 1]
consistent = he[h1, 0] == he[h2, 1]
pair_e, pair_h1, pair_h2 = two[consistent], h1[consistent], h2[consistent]
fa, fb = pair_h1 // 3, pair_h2 // 3


def face_components(mask_pairs, n=nF):
    A = sp.coo_matrix((np.ones(mask_pairs.sum()), (fa[mask_pairs], fb[mask_pairs])), (n, n))
    return connected_components(A, directed=False)


# ---------------------------------------------------------------- regions
_, comp = face_components(np.ones(len(fa), bool))
main = np.bincount(comp).argmax()
fcen = V[T].mean(1)

REGION_BONES = {
    'head': ['spine.004', 'spine.005'],
    'torso': ['spine', 'spine.001', 'spine.002', 'spine.003', 'shoulder.L', 'shoulder.R', 'pelvis.L', 'pelvis.R',
              'arm_pull.L', 'arm_pull.R'],
}
for s in ('.L', '.R'):
    REGION_BONES['arm' + s] = ['upper_arm' + s, 'forearm' + s]
    REGION_BONES['hand' + s] = ['hand' + s, 'thumb.01' + s] + [n for n in names if n.startswith('carpal') and n.endswith(s)]
    REGION_BONES['leg' + s] = ['thigh' + s, 'shin' + s]
    REGION_BONES['foot' + s] = ['foot' + s, 'heel.02' + s]
    REGION_BONES['toes' + s] = ['toe' + s]          # (the MakeHuman feet: the toes are part of the skin)
# fingers are part of the skin (the MakeHuman hands): each finger and the thumb is its own region
for s in ('.L', '.R'):
    for f_ in ('f_index', 'f_middle', 'f_ring', 'f_pinky', 'thumb'):
        REGION_BONES[f'finger{s} {f_}'] = [n for n in names if n.startswith(f_ + '.0') and n.endswith(s) and n != 'thumb.01' + s]
regions = list(REGION_BONES)
bone_region = np.zeros(len(names), int)
for r, bl in enumerate(regions):
    for b in REGION_BONES[bl]: bone_region[names.index(b)] = r
score = np.zeros((nV, len(regions)))
for k in range(4): np.add.at(score, (np.arange(nV), bone_region[SI[:, k]]), W8[:, k] / 255)
# smooth the scores across the skin so region borders run smoothly instead of zig-zagging
ai, aj = he[:, 0], he[:, 1]
deg = np.maximum(np.bincount(ai, minlength=nV), 1)[:, None]
for _ in range(40):
    acc = np.zeros_like(score); np.add.at(acc, ai, score[aj]); np.add.at(acc, aj, score[ai])
    score = 0.5 * score + 0.25 * acc / deg
flab = score[T].sum(1).argmax(1)
flab[comp != main] = -1        # finger and toe tubes: handled below
# the MakeHuman feet: every toe is its own region (body-rig.js says which toe each point belongs to); what the toe
# bone covers besides (the webbing, the ball of the foot) goes with the foot
if 'digit' in B:
    DG = dec(B['digit'], np.uint8)[T]
    for s in ('.L', '.R'):
        flab[flab == regions.index('toes' + s)] = regions.index('foot' + s)
        for k in range(1, 6):
            on = (DG == k).all(1) & ((fcen[:, 0] > 0) if s == '.L' else (fcen[:, 0] < 0))
            if on.any(): regions.append(f'toes{s} {k}'); flab[on] = len(regions) - 1


def tidy(lab, active):
    """Keep only the biggest piece of each label; smaller islands join the neighbour they touch most."""
    for _ in range(10):
        same = active[fa] & active[fb] & (lab[fa] == lab[fb])
        _, c = face_components(same)
        changed = False
        for l in np.unique(lab[active]):
            cs = np.unique(c[active & (lab == l)])
            if len(cs) < 2: continue
            big = cs[np.argmax([np.sum(c == x) for x in cs])]
            for x in cs:
                if x == big: continue
                fs = np.where(c == x)[0]
                m = np.isin(fa, fs) & ~np.isin(fb, fs) & active[fb]
                m2 = np.isin(fb, fs) & ~np.isin(fa, fs) & active[fa]
                nb = np.r_[lab[fb[m]], lab[fa[m2]]]
                if len(nb): lab[fs] = np.bincount(nb).argmax(); changed = True
        if not changed: break
    return lab


flab = tidy(flab, flab >= 0)

# ---------------------------------------------------------------- split some regions in two halves
# torso and head: front / back along the sides (half-way between front and back at each height);
# hands: palm / back of hand; feet: top / sole.
chart_name = np.empty(nF, object)


def halves(fsel, f, pos, neg):
    """Split the faces fsel by the sign of the per-vertex field f (smoothed), naming the two halves."""
    fv = f[T[fsel]].mean(1)
    chart_name[fsel] = np.where(fv >= 0, pos, neg)


for r, nm in enumerate(regions):
    fsel = np.where(flab == r)[0]
    if nm in ('torso', 'head'):
        vs = np.unique(T[fsel])
        bins = np.round(V[vs, 1] / 0.01).astype(int)
        zc = {}
        for b in np.unique(bins):
            z = V[vs[bins == b], 2]; zc[b] = (z.max() + z.min()) / 2
        keys = np.array(sorted(zc)); vals = np.array([zc[k] for k in keys])
        vals = np.convolve(np.pad(vals, 4, mode='edge'), np.ones(9) / 9, 'valid')   # a smooth side line
        f = V[:, 2] - np.interp(V[:, 1] / 0.01, keys, vals)
        halves(fsel, f, nm + ' front', nm + ' back')
    elif nm.startswith('hand'):
        vs = np.unique(T[fsel]); c = V[vs].mean(0)
        _, _, ax = np.linalg.svd(V[vs] - c, full_matrices=False)
        n = ax[2]
        # the back of the hand is the side facing away from the thigh (palms face in at rest)
        side = 1 if nm.endswith('.L') else -1
        if n[0] * side < 0: n = -n
        halves(fsel, (V - c) @ n, nm + ' back', nm + ' palm')
    elif nm.startswith('foot'):
        vs = np.unique(T[fsel])
        halves(fsel, V[:, 1] - (V[vs, 1].min() + 0.012), nm + ' top', nm + ' sole')
    elif nm.startswith('toes'):
        if not len(fsel): continue                 # (the toe bone's own region: all moved to the foot and the toes)
        vs = np.unique(T[fsel])
        halves(fsel, V[:, 1] - (V[vs, 1].min() + 0.01), nm + ' top', nm + ' under')
    elif not nm.startswith('finger'):
        chart_name[fsel] = nm
# fingers: split along their length into the nail side (the same way as the back of that hand) and the front
for r, nm in enumerate(regions):
    if not nm.startswith('finger'): continue
    fsel = np.where(flab == r)[0]
    if not len(fsel): continue
    side = nm[6:8]
    vs = np.unique(T[fsel]); cen = V[vs].mean(0)
    n = fcen[chart_name == f'hand{side} back'].mean(0) - fcen[chart_name == f'hand{side} palm'].mean(0)
    axis = np.linalg.svd(V[vs] - cen, full_matrices=False)[2][0]     # along the finger
    n -= axis * (n @ axis); n /= np.linalg.norm(n)
    halves(fsel, (V - cen) @ n, nm + ' top', nm + ' under')

# finger and toe tubes: each one is its own closed tube; split into back (nail side) and front
tube_comps = [c for c in np.unique(comp) if c != main]
for c in tube_comps:
    fsel = np.where(comp == c)[0]
    vs = np.unique(T[fsel]); cen = V[vs].mean(0)
    side = '.L' if cen[0] > 0 else '.R'
    if cen[1] < 0.15:
        kind, n = 'toe', np.array([0, 1.0, 0])
    else:
        kind = 'finger'
        # nail side: same direction as the back of that hand
        hf = chart_name == f'hand{side} back'
        hb = chart_name == f'hand{side} palm'
        n = fcen[hf].mean(0) - fcen[hb].mean(0)
        axis = np.linalg.svd(V[vs] - cen, full_matrices=False)[2][0]     # along the finger
        n -= axis * (n @ axis); n /= np.linalg.norm(n)
    fv = ((V[T[fsel]].mean(1) - cen) @ n)
    lab_ = f'{kind}{side} #{c}'
    chart_name[fsel] = np.where(fv >= 0, lab_ + ' top', lab_ + ' under')

# tidy the halves too (a half must be one piece)
cn_list = sorted(set(chart_name))
cn_idx = np.array([cn_list.index(x) for x in chart_name])
cn_idx = tidy(cn_idx, np.ones(nF, bool))
chart_name = np.array(cn_list, object)[cn_idx]

# ---------------------------------------------------------------- seams inside tube-shaped regions
# Sleeves and legs are tubes (open at both ends): cut each along the inner side (facing the body's
# midline), the way sleeve templates are drawn: cheapest path between the two openings, where the
# cost is low on the inner line and high elsewhere.
cut_edge = np.zeros(nE, bool)
same_chart = chart_name[fa] == chart_name[fb]
cut_edge[pair_e[~same_chart]] = True
nonpair = np.ones(nE, bool); nonpair[pair_e] = False
cut_edge[nonpair] = True       # non-manifold / inconsistent edges


def split_points():
    """Corners joined across uncut edges become one point; each original point may become several."""
    joined = ~cut_edge[pair_e]
    h1, h2 = pair_h1[joined], pair_h2[joined]
    # half-edge h: a = T[f,k] (corner h), b = T[f,k+1] (corner f*3+(k+1)%3); the twin runs b -> a
    nxt = lambda h: (h // 3) * 3 + (h % 3 + 1) % 3
    ca = np.r_[h1, nxt(h1)]; cb = np.r_[nxt(h2), h2]
    A = sp.coo_matrix((np.ones(len(ca)), (ca, cb)), (nF * 3, nF * 3))
    _, cc = connected_components(A, directed=False)
    orig = T.ravel()
    # one new point per corner class; the first class of each original point keeps its index
    cls, first_corner = np.unique(cc, return_index=True)
    cls_orig = orig[first_corner]
    o = np.argsort(cls_orig, kind='stable')
    keeps = np.r_[True, cls_orig[o][1:] != cls_orig[o][:-1]]
    new_id = np.empty(len(cls), np.int64)
    new_id[o[keeps]] = cls_orig[o[keeps]]
    new_id[o[~keeps]] = nV + np.arange((~keeps).sum())
    extra = cls_orig[o[~keeps]]
    lookup = np.empty(cc.max() + 1, int); lookup[cls] = new_id
    return lookup[cc].reshape(-1, 3), extra.astype(np.int64)


def boundary_loops(fsel):
    """Boundary loops (lists of original point ids) of the faces fsel, given the current cuts, biggest first.
    Traced on the cut-open mesh, so a point that sits on a seam (and so on the boundary twice) is handled."""
    TT_, extra_ = split_points()
    src_ = np.r_[np.arange(nV), extra_]
    vs, Fl = np.unique(TT_[fsel], return_inverse=True)
    loops = igl.boundary_loop_all(Fl.reshape(-1, 3).astype(np.int64))
    return sorted([list(src_[vs[np.array(l)]]) for l in loops], key=len, reverse=True)


def cut_path(fsel, src_pts, dst_pts, cost):
    """Cut along the cheapest edge path from any of src_pts to any of dst_pts, over the faces fsel."""
    hs = (fsel[:, None] * 3 + np.arange(3)).ravel()
    e = np.unique(einv[hs]); e = e[~cut_edge[e]]
    a, b = eu[e, 0], eu[e, 1]
    w = np.linalg.norm(V[a] - V[b], axis=1) * (cost[a] + cost[b]) / 2
    G = sp.coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), (nV, nV)).tocsr()
    dist, pred, srcs = dijkstra(G, indices=np.array(src_pts), min_only=True, return_predecessors=True)
    end = dst_pts[np.argmin(dist[dst_pts])]
    path = [end]
    while pred[path[-1]] >= 0: path.append(pred[path[-1]])
    pe = np.sort(np.array([path[:-1], path[1:]]).T, 1)
    idx = np.searchsorted(eu[:, 0] * nV + eu[:, 1], pe[:, 0] * nV + pe[:, 1])
    cut_edge[idx] = True
    return len(path)


bone_head = {b['name']: np.array(b['head']) for b in R['bones']}
for nm in sorted(set(chart_name)):
    if not nm.startswith(('arm', 'leg')): continue
    fsel = np.where(chart_name == nm)[0]
    loops = boundary_loops(fsel)
    side = 1 if nm.endswith('.L') else -1
    a0, a1 = (('upper_arm', 'hand') if nm.startswith('arm') else ('thigh', 'foot'))
    axis = bone_head[a1 + nm[-2:]] - bone_head[a0 + nm[-2:]]; axis /= np.linalg.norm(axis)
    inward = np.array([-side, 0, 0.0]); inward -= axis * (inward @ axis); inward /= np.linalg.norm(inward)
    s = N @ inward                                   # 1 on the inner line, -1 on the outer
    cost = 1 + 30 * ((1 - s) / 2) ** 2
    n = cut_path(fsel, loops[0], np.array(loops[1]), cost) if len(loops) > 1 else 0
    log(f'{nm}: {len(loops)} openings, inner seam {n} points')


# ---------------------------------------------------------------- cut the mesh, make every chart a disc
def build_charts():
    joined = ~cut_edge[pair_e]
    _, fcomp = face_components(joined)
    return fcomp


# Charts that still have holes (more than one boundary loop) get a seam from each hole to the outer edge;
# closed pieces (no boundary at all) get a single short cut to open them.
for it in range(3):
    fcomp = build_charts()
    fixed = 0
    for c in np.unique(fcomp):
        fsel = np.where(fcomp == c)[0]
        loops = boundary_loops(fsel)
        if len(loops) > 1:
            for l in loops[1:]:
                cut_path(fsel, l, np.array(loops[0]), np.ones(nV)); fixed += 1
    if not fixed: break
    log(f'pass {it}: joined {fixed} holes to their chart edge')

fcomp = build_charts()
TT, extra = split_points()
nV2 = nV + len(extra)
src_of = np.r_[np.arange(nV), extra]
V2 = V[src_of]
log(f'{len(np.unique(fcomp))} charts, {len(extra)} seam copies')

# ---------------------------------------------------------------- flatten each chart
UV = np.zeros((nV2, 2))
charts = []
for c in np.unique(fcomp):
    fsel = np.where(fcomp == c)[0]
    Fg = TT[fsel]
    vs, Fl = np.unique(Fg, return_inverse=True); Fl = Fl.reshape(-1, 3)
    Vl = V2[vs]
    nm = chart_name[fsel[0]]
    area3 = np.linalg.norm(np.cross(Vl[Fl[:, 1]] - Vl[Fl[:, 0]], Vl[Fl[:, 2]] - Vl[Fl[:, 0]]), axis=1).sum() / 2
    if len(fsel) < 4: continue
    euler = len(vs) - len(np.unique(np.sort(np.r_[Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]], 1), axis=0)) + len(Fl)
    bnd = igl.boundary_loop(Fl)
    if euler != 1 or len(bnd) < 3:
        log(f'  ! {nm}: not a disc (euler {euler}), skipped'); continue
    # start: Tutte embedding (always fold-free on a disc), boundary on a circle
    n = len(vs)
    bc = igl.map_vertices_to_circle(np.asfortranarray(Vl), bnd.astype(np.int32))
    i_, j_ = np.r_[Fl[:, 0], Fl[:, 1], Fl[:, 2]], np.r_[Fl[:, 1], Fl[:, 2], Fl[:, 0]]
    Adj = sp.coo_matrix((np.ones(len(i_) * 2), (np.r_[i_, j_], np.r_[j_, i_])), (n, n)).tocsr()
    Adj.data[:] = 1
    L = sp.diags(np.asarray(Adj.sum(1)).ravel()) - Adj
    free = np.setdiff1d(np.arange(n), bnd)
    U0 = np.zeros((n, 2)); U0[bnd] = bc
    from scipy.sparse.linalg import spsolve
    rhs = -L[free][:, bnd] @ bc
    sol = spsolve(L[free][:, free].tocsc(), rhs)
    U0[free] = sol
    U0 *= np.sqrt(area3 / np.pi)
    a0 = U0[Fl[:, 1]] - U0[Fl[:, 0]]; b0 = U0[Fl[:, 2]] - U0[Fl[:, 0]]
    if (a0[:, 0] * b0[:, 1] - a0[:, 1] * b0[:, 0]).sum() < 0: U0[:, 0] *= -1   # SLIM wants it the right way round
    data = igl.slim_precompute(np.asfortranarray(Vl), np.asfortranarray(Fl.astype(np.int32)), np.asfortranarray(U0),
                               igl.SYMMETRIC_DIRICHLET, np.zeros(0, np.int32), np.zeros((0, 2)), 0.0)
    U = np.array(igl.slim_solve(data, 60 if n > 2000 else 30))
    # rotate so the body's 'up' points up on the map
    d1 = Vl[Fl[:, 1]] - Vl[Fl[:, 0]]; d2 = Vl[Fl[:, 2]] - Vl[Fl[:, 0]]
    u1 = U[Fl[:, 1]] - U[Fl[:, 0]]; u2 = U[Fl[:, 2]] - U[Fl[:, 0]]
    # per triangle Jacobian J (2x3, maps 3D tangent to UV) via the 2x2 local frame
    e1 = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    nrm = np.cross(d1, d2); dbl = np.linalg.norm(nrm, axis=1); nrm /= dbl[:, None]
    e2 = np.cross(nrm, e1)
    P = np.stack([np.c_[(d1 * e1).sum(1), (d1 * e2).sum(1)], np.c_[(d2 * e1).sum(1), (d2 * e2).sum(1)]], 1)   # rows: d1,d2 in frame
    Q = np.stack([u1, u2], 1)
    J2 = np.linalg.solve(P, Q).transpose(0, 2, 1)          # UV = J2 @ local
    up = np.stack([e1[:, 1], e2[:, 1]], 1)                 # world up (0,1,0) in each frame
    if nm.endswith('sole') or 'toe' in nm or nm.startswith('foot'):
        up = np.stack([e1[:, 2], e2[:, 2]], 1)            # feet: toes point up on the map
    g = (np.einsum('fij,fj->fi', J2, up) * dbl[:, None]).sum(0)
    ang = np.arctan2(g[0], g[1])
    Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    U = U @ Rm.T
    J2 = np.einsum('ij,fjk->fik', Rm, J2)
    if np.linalg.det(J2).mean() < 0:                     # mirrored: flip u
        U[:, 0] *= -1; J2[:, 0] *= -1
    sv = np.linalg.svd(J2, compute_uv=False)
    flips = int((np.linalg.det(J2) <= 0).sum())
    stretch = sv[:, 0] * sv[:, 1]                          # area ratio map / skin
    worst = np.maximum(sv[:, 0], 1 / np.maximum(sv[:, 1], 1e-9))
    U = U * [1, -1]                                    # to sheet coordinates (y down), so up stays up
    UV[vs] = U - U.min(0)
    charts.append({'name': nm, 'faces': fsel, 'verts': vs, 'size': U.max(0) - U.min(0), 'area': area3,
                   'area_ratio': stretch, 'worst': worst, 'flips': flips})
    log(f'  {nm:28s} {n:6d} pts {area3 * 1e4:7.0f} cm2  area ratio {np.percentile(stretch, 5):.2f}-{np.percentile(stretch, 95):.2f}'
        f'  >10% length error on {100 * np.average(worst > 1.1, weights=dbl):4.1f}% of skin  flips {flips}')

# ---------------------------------------------------------------- lay the charts out on one sheet
# Same scale for every chart: big ones first, in rows (torso, legs, arms, head, then hands, feet, digits).
def group_rank(nm):
    for i, k in enumerate(('torso', 'leg', 'arm', 'head', 'hand', 'foot', 'finger', 'toe')):
        if nm.startswith(k): return i
    return 9


charts.sort(key=lambda c: (group_rank(c['name']), -c['size'][1]))
total = sum((c['size'][0] + MARGIN) * (c['size'][1] + MARGIN) for c in charts)
W = max(np.sqrt(total * 1.25), max(c['size'][0] for c in charts) + 2 * MARGIN)
x = y = MARGIN; row_h = 0
for c in charts:
    if x + c['size'][0] + MARGIN > W: x = MARGIN; y += row_h + MARGIN; row_h = 0
    c['at'] = np.array([x, y]); x += c['size'][0] + MARGIN; row_h = max(row_h, c['size'][1])
H = y + row_h + MARGIN
S = max(W, H)                                         # square sheet, S metres on a side
for c in charts:
    UV[c['verts']] = (UV[c['verts']] + c['at'] + [(S - W) / 2, (S - H) / 2]) / S
UV[:, 1] = 1 - UV[:, 1]                               # sheet y runs down; v runs up
log(f'sheet {S * 100:.0f} x {S * 100:.0f} cm  ({S * 1000 / TEMPLATE_PX:.2f} mm per pixel at {TEMPLATE_PX} px)')

# ---------------------------------------------------------------- template images
def render(px, skin_tone):
    k = px / S
    bg = (217, 176, 140) if skin_tone else (255, 255, 255)
    img = Image.new('RGB', (px, px), bg)
    mask = Image.new('L', (px, px), 0)
    dr, dm = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    pix = np.c_[UV[:, 0] * px, (1 - UV[:, 1]) * px]
    for c in charts:
        tri = pix[TT[c['faces']]]
        r = np.log2(np.clip(c['area_ratio'], 0.25, 4))          # >0: map bigger than skin (stretched on print)
        for t, rr in zip(tri, r):
            pts = [tuple(q) for q in t]
            if skin_tone: col = bg
            else:
                a = min(abs(rr) / 0.4, 1)                         # full colour at ~30% area error
                col = (255, int(255 - 120 * a), int(255 - 200 * a)) if rr > 0 else (int(255 - 200 * a), int(255 - 120 * a), 255)
                if a < 0.12: col = (255, 255, 255)
            dr.polygon(pts, fill=col)
            dm.polygon(pts, fill=255)
    # 1 cm grid (5 cm lines stronger), only inside the charts
    grid = Image.new('RGB', (px, px), (0, 0, 0)); dg = ImageDraw.Draw(grid)
    ga = Image.new('L', (px, px), 0); dga = ImageDraw.Draw(ga)
    step = 0.01 * k
    for i in range(int(S / 0.01) + 1):
        o = i * step; strong = i % 5 == 0
        a_ = 150 if strong else 60
        w_ = max(1, round(px / 2048)) * (2 if strong else 1)
        for xy in (((o, 0), (o, px)), ((0, o), (px, o))):
            dga.line(xy, fill=a_, width=w_)
    col = (90, 50, 30) if skin_tone else (40, 70, 140)
    gridc = Image.new('RGB', (px, px), col)
    alpha = Image.fromarray((np.array(ga, np.uint16) * np.array(mask, np.uint16) // 255).astype(np.uint8))
    img = Image.composite(gridc, img, alpha)
    # outlines and names
    dr = ImageDraw.Draw(img)
    lw = max(1, px // 1024)
    try: font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', max(12, px // 110))
    except OSError: font = ImageFont.load_default()
    for c in charts:
        Fc = TT[c['faces']]
        e = np.sort(np.r_[Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]], 1)
        u_, cnt = np.unique(e, axis=0, return_counts=True)
        for a, b in u_[cnt == 1]:
            dr.line([tuple(pix[a]), tuple(pix[b])], fill=(30, 30, 30) if not skin_tone else (120, 70, 45), width=lw)
        if not skin_tone and not c['name'].startswith(('finger', 'toe')):
            cx, cy = pix[c['verts']].mean(0)
            label = c['name'].replace('.L', ' (L)').replace('.R', ' (R)')
            dr.text((cx, cy), label, fill=(0, 0, 0), font=font, anchor='mm')
    if not skin_tone:
        dr.text((px * 0.01, px * 0.99), f'Skin map, true scale: 1 square = 1 cm ({S * 1000 / px:.2f} mm per pixel). '
                f'Blue = map smaller than skin, orange = map larger than skin.', fill=(0, 0, 0), font=font, anchor='ld')
    return img


render(TEMPLATE_PX, False).save(f'{out}/skin-map.png', optimize=True)
prev = render(PREVIEW_PX, True)
import io
buf = io.BytesIO(); prev.save(buf, 'JPEG', quality=85)
with open(f'{out}/skin-map.js', 'w') as fh:
    fh.write('// Generated by tools/uv_atlas.py: skin-tone 1 cm grid on the skin map, for the 3D view.\n')
    fh.write("window.SKIN_MAP = 'data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode() + "';\n")
log('wrote skin-map.png and skin-map.js')

# ---------------------------------------------------------------- write skin-uv.js
# Kept out of body-rig.js so fit_rig.py and this script never overwrite each other. The page checks the
# fingerprint below and ignores a skin-uv.js made for an older body-rig.js.
uvdata = {
    'fingerprint': fingerprint,
    'index': b64(TT.astype(np.uint32)),
    'dup': b64(extra.astype(np.int32)),
    'uv': b64(UV.astype(np.float32)),
    'metersPerUV': float(S),
    'charts': [{'name': c['name'], 'min': UV[c['verts']].min(0).round(5).tolist(), 'max': UV[c['verts']].max(0).round(5).tolist()}
               for c in charts],
}
with open(f'{out}/skin-uv.js', 'w') as fh:
    fh.write('// Generated by tools/uv_atlas.py from body-rig.js: flat skin map (seams, UVs) for the body skin.\n')
    fh.write('window.SKIN_UV = ' + json.dumps(uvdata) + ';\n')
log('wrote skin-uv.js:', nV2, 'skin points (', len(extra), 'seam copies )')

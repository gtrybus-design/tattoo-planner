"""Build a skin surface directly from the anatomy: everything inside the muscles and bones plus a thin
skin layer becomes 'body', and the outline of that is extracted as a smooth mesh (surface nets).

Gaps narrower than about twice the skin thickness get bridged (like real skin over the sternum), wider
ones stay open (the armpit, between the legs). Parts that must never be bridged to each other (fingers)
are built as separate groups and merged afterwards.
"""
import numpy as np


def sample_triangles(P, F):
    """Corners, edge midpoints and centre of every triangle: dense enough to rasterize a surface."""
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    return np.vstack([P, (a + b) / 2, (b + c) / 2, (c + a) / 2, (a + b + c) / 3])


def _sphere_offsets(r):
    rng = np.arange(-r, r + 1)
    g = np.stack(np.meshgrid(rng, rng, rng, indexing='ij'), -1).reshape(-1, 3)
    return g[(g ** 2).sum(1) <= r * r + 0.5]


def _shift_or(dst, src, off):
    """dst |= src shifted by integer offset off (no wrap-around)."""
    sl_d, sl_s = [], []
    for o, n in zip(off, src.shape):
        if o >= 0: sl_d.append(slice(o, n)); sl_s.append(slice(0, n - o))
        else: sl_d.append(slice(0, n + o)); sl_s.append(slice(-o, n))
    dst[tuple(sl_d)] |= src[tuple(sl_s)]


def solid(points, origin, cell, shape, radius, close=0):
    """Voxels inside (anatomy + `radius` voxels of skin), with enclosed space filled in."""
    occ = np.zeros(shape, bool)
    ijk = np.floor((points - origin) / cell).astype(int)
    ok = np.all((ijk >= 0) & (ijk < shape), 1)
    occ[tuple(ijk[ok].T)] = True
    # grow by radius + close, then (below) shrink back by close: hollows and grooves narrower than about
    # 2 x close get filled, so the layer ends up level, thick in dips and thin over high points
    dil = _dilate(occ, radius + close)
    # outside = empty space you can reach in a straight line from the edge of the grid along x, y or z
    outside = np.zeros(shape, bool)
    for ax in range(3):
        fwd = np.logical_or.accumulate(dil, axis=ax)
        bwd = np.flip(np.logical_or.accumulate(np.flip(dil, ax), axis=ax), ax)
        outside |= ~fwd | ~bwd
    inside = ~outside
    if close: inside = ~_dilate(~inside, close)
    return inside


def blur(f, passes=2):
    """Soften a voxel field: `passes` repeats of a 1-2-1 average along each axis, done in one pass per axis.
    Repeating [1, 2, 1] / 4 n times is the same as one average with the binomial weights C(2n, k) / 4^n
    (each repeat convolves the kernel with itself once more), so the result is identical and much faster."""
    from scipy.ndimage import convolve1d
    from math import comb
    if passes <= 0: return f
    k = np.array([comb(2 * passes, i) for i in range(2 * passes + 1)], np.float64) / 4.0 ** passes
    k = k[k > 1e-7]; k /= k.sum()                      # (the far tails are negligible)
    for ax in range(3): f = convolve1d(f, k.astype(f.dtype), axis=ax, mode='nearest')
    return f


CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(CORNERS[a] - CORNERS[b]).sum() == 1]


def surface_nets(f, origin, cell, iso=0.5):
    """Smooth surface through f = iso (one vertex per crossing cell, quads across crossing edges)."""
    X, Y, Z = f.shape
    s = f > iso
    cv = np.stack([f[i:X - 1 + i, j:Y - 1 + j, k:Z - 1 + k] for i, j, k in CORNERS], -1)   # cells x 8
    cs = cv > iso
    active = cs.any(-1) & ~cs.all(-1)
    cid = np.argwhere(active)
    vals = cv[active]
    acc = np.zeros((len(cid), 3)); cnt = np.zeros(len(cid))
    for a, b in EDGES:
        fa, fb = vals[:, a], vals[:, b]
        m = (fa > iso) != (fb > iso)
        t = (iso - fa[m]) / (fb[m] - fa[m])
        acc[m] += CORNERS[a] + t[:, None] * (CORNERS[b] - CORNERS[a]); cnt[m] += 1
    verts = origin + (cid + acc / cnt[:, None]) * cell
    vid = np.full(active.shape, -1, np.int64); vid[tuple(cid.T)] = np.arange(len(cid))

    quads = []
    for d in range(3):
        u, v = (d + 1) % 3, (d + 2) % 3
        lo = [slice(1, n - 1) for n in (X, Y, Z)]; hi = list(lo)
        lo[d] = slice(0, s.shape[d] - 1); hi[d] = slice(1, s.shape[d])
        cross = s[tuple(lo)] != s[tuple(hi)]
        p = np.argwhere(cross); p[:, u] += 1; p[:, v] += 1          # voxel at the low end of each crossing edge
        inside_low = s[tuple(p.T)]
        eu, ev = np.eye(3, dtype=int)[u], np.eye(3, dtype=int)[v]
        q = np.stack([vid[tuple((p - eu - ev).T)], vid[tuple((p - ev).T)], vid[tuple(p.T)], vid[tuple((p - eu).T)]], 1)
        q[~inside_low] = q[~inside_low][:, ::-1]                   # face outward
        quads.append(q[(q >= 0).all(1)])
    Q = np.vstack(quads)
    tris = np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    return verts, tris


def envelope(groups, cell=0.003, radius=2, blur_passes=1, margin=18, shrink=0, close=0):
    """groups: list of point arrays; each is made solid on its own (so groups never bridge each other).
    radius: skin thickness in voxels, one number or one per group."""
    radii = radius if isinstance(radius, (list, tuple)) else [radius] * len(groups)
    closes = close if isinstance(close, (list, tuple)) else [close] * len(groups)
    allp = np.vstack(groups)
    origin = allp.min(0) - margin * cell
    shape = tuple(np.ceil((allp.max(0) + margin * cell - origin) / cell).astype(int) + 1)
    inside = np.zeros(shape, bool)
    for pts, r, cl in zip(groups, radii, closes):
        lo = np.maximum(np.floor((pts.min(0) - origin) / cell).astype(int) - margin, 0)
        hi = np.minimum(np.ceil((pts.max(0) - origin) / cell).astype(int) + margin, np.array(shape))
        sub = solid(pts, origin + lo * cell, cell, tuple(hi - lo), r, cl)
        inside[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= sub
    inside = opening(inside, 1)
    if shrink: inside = ~_dilate(~inside, shrink)       # pull the surface back in by `shrink` voxels
    f = blur(inside.astype(np.float32), blur_passes)
    return surface_nets(f, origin, cell)


FACE6 = [o for o in _sphere_offsets(1) if any(o) and np.abs(o).sum() == 1]
CUBE26 = [np.array([i, j, k]) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]


def _dilate(m, r):
    """Grow by about r voxels: alternating one-voxel steps (6 and 26 neighbours) make a near-round ball,
    and r small steps are far faster than one big ball for large r."""
    out = m.copy()
    for i in range(r):
        prev = out.copy()
        for off in (FACE6 if i % 2 == 0 else CUBE26): _shift_or(out, prev, off)
    return out


def opening(m, r):
    """Remove fins and specks thinner than about 2r voxels (shrink, then grow back)."""
    eroded = ~_dilate(~m, r)
    return _dilate(eroded, r) & m


def envelope_pieces(pieces, cell=0.003):
    """One continuous surface from several pieces. pieces: list of (name, groups, radii, closes, blur).
    Each piece is made solid and blurred on its own; then the pieces are blended softly where they meet
    (soft union a + b - ab, no crease), except the legs, which are only blended with the rest of the body
    and never with each other, so the inner thighs can't fuse. A piece may carry a 6th item (center, radius):
    then it is only blended softly within that radius of the center (e.g. an arm around its shoulder joint)
    and combined hard further away, so an arm hanging against the ribs stays a separate surface."""
    allp = np.vstack([g for p in pieces for g in p[1]])
    pad = max(max(r + c for r, c in zip(p[2], p[3])) + p[4] + 4 for p in pieces)
    origin = allp.min(0) - pad * cell
    shape = tuple(np.ceil((allp.max(0) + pad * cell - origin) / cell).astype(int) + 1)
    body = np.zeros(shape, np.float32)
    legs = []
    for piece in pieces:
        name, groups, radii, closes, blur_n = piece[:5]
        soft = piece[5] if len(piece) > 5 else None
        pts_all = np.vstack(groups)
        m = max(r + c for r, c in zip(radii, closes)) + blur_n + 4
        lo = np.maximum(np.floor((pts_all.min(0) - origin) / cell).astype(int) - m, 0)
        hi = np.minimum(np.ceil((pts_all.max(0) - origin) / cell).astype(int) + m, np.array(shape))
        inside = np.zeros(tuple(hi - lo), bool)
        for pts, r, cl in zip(groups, radii, closes):
            glo = np.maximum(np.floor((pts.min(0) - origin) / cell).astype(int) - (r + cl + 3), lo)
            ghi = np.minimum(np.ceil((pts.max(0) - origin) / cell).astype(int) + (r + cl + 3), hi)
            sub = solid(pts, origin + glo * cell, cell, tuple(ghi - glo), r, cl)
            a, b = glo - lo, ghi - lo
            inside[a[0]:b[0], a[1]:b[1], a[2]:b[2]] |= sub
        inside = opening(inside, 1)
        f = blur(inside.astype(np.float32), blur_n)
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        if name.startswith('leg'):
            legs.append((sl, f))
        elif soft is None:
            region = body[sl]; region += f * (1 - region)
        else:
            c_, r_ = soft
            ax = [origin[k] + (np.arange(lo[k], hi[k]) + 0.5) * cell - c_[k] for k in range(3)]
            d = np.sqrt(ax[0][:, None, None] ** 2 + ax[1][None, :, None] ** 2 + ax[2][None, None, :] ** 2)
            t = np.clip((1.5 * r_ - d) / (0.5 * r_), 0, 1).astype(np.float32)      # 1 within r, 0 beyond 1.5 r
            region = body[sl]
            region[...] = (region + f * (1 - region)) * t + np.maximum(region, f) * (1 - t)
    out = body.copy()
    for sl, f in legs:
        b = body[sl]
        np.maximum(out[sl], b + f - b * f, out=out[sl])
    return surface_nets(out, origin, cell)

"""Fast versions of the two operations the build spends most of its time on.

1. "Which point in this cloud is nearest to each of my points?"  Comparing every point with every other
   point costs (number of queries) x (pool size): with 80,000 skin points and 200,000 muscle points that is
   16 billion distances. A KD-tree (scipy's cKDTree) sorts the pool into nested boxes once, then each query
   only looks inside the few boxes around it, so it costs about (queries) x log(pool size). Same answers,
   typically 50-200x faster.

2. "Add these values into these slots, where a slot can appear many times" (np.add.at). numpy's add.at is
   a known slow path (it handles each repeat one by one in a generic loop). np.bincount does the same sum
   in one tight C loop: give it the slot of every value and the values as weights, and it returns the total
   per slot. The results match add.at up to the last bit of rounding.

Lesson for other projects: when a script is slow, profile it first (python -m cProfile -o out.prof script.py),
then look for these two patterns -- all-pairs distances and scatter-adds -- before anything clever.
"""
import numpy as np
from scipy.spatial import cKDTree


def add_at(a, idx, v):
    """Same as np.add.at(a, idx, v), via np.bincount. idx: one index array (into the first axis of a), or a
    tuple of index arrays (one per axis, broadcast together). Changes a in place."""
    if isinstance(idx, tuple):
        idx = np.broadcast_arrays(*[np.asarray(i) for i in idx])
        v = np.broadcast_to(v, idx[0].shape)
        flat = np.ravel_multi_index(tuple(i.ravel() for i in idx), a.shape[:len(idx)])
        rest = int(np.prod(a.shape[len(idx):], dtype=np.int64))
        if rest != 1: raise ValueError('add_at: tuple index must address single elements')
        s = np.bincount(flat, weights=np.asarray(v, np.float64).ravel(), minlength=int(np.prod(a.shape)))
    else:
        idx = np.asarray(idx).ravel()
        row = int(np.prod(a.shape[1:], dtype=np.int64))
        v = np.broadcast_to(v, (len(idx),) + a.shape[1:]).reshape(len(idx), row)
        flat = (idx[:, None] * row + np.arange(row)).ravel()
        s = np.bincount(flat, weights=np.asarray(v, np.float64).ravel(), minlength=a.shape[0] * row)
    a += s.reshape(a.shape).astype(a.dtype, copy=False)


def tree(pool):
    return cKDTree(np.asarray(pool, np.float64))


def nearest_index(Q, pool, margin=None, fallback=True):
    """Index of the nearest pool point for each query point. With fallback=False, -1 where the nearest one is
    farther than margin."""
    d, j = tree(pool).query(np.asarray(Q, np.float64), workers=-1)
    if not fallback: j = np.where(d <= margin, j, -1)
    return j.astype(int)


def knn(Q, pool, k):
    """The k nearest pool points of each query point: indices and distances (nearest first)."""
    d, j = tree(pool).query(np.asarray(Q, np.float64), k=k, workers=-1)
    return j.reshape(len(Q), k).astype(int), d.reshape(len(Q), k)


# 3. Smoothing loops repeat "for every point, add up its neighbours' values" hundreds of times on the same mesh.
#    That sum is a matrix product: A @ X, where A[i, j] = 1 for each (point i, neighbour j) pair. A sparse matrix
#    stores only those pairs. Building it costs about one scatter-add; every use after that is ~8x faster.
#    The matrix is kept (cached) for as long as the index arrays it was built from exist.
import weakref
import scipy.sparse as _sp
_nsum_cache = {}


def nsum(i, j, X, n=None):
    """Same as: acc = np.zeros((n,) + X.shape[1:]); np.add.at(acc, i, X[j]) -- the sum over each point's
    neighbours (i: the point, j: its neighbour, one entry per pair). n defaults to len(X), and the result then has
    X's type, like np.zeros_like(X); with n given it is float64, like np.zeros(n)."""
    dt = X.dtype if n is None else np.float64
    n = len(X) if n is None else n
    key = (id(i), id(j), n, len(X))
    hit = _nsum_cache.get(key)
    if hit is None or hit[0]() is not i or hit[1]() is not j:
        A = _sp.csr_matrix((np.ones(len(i)), (np.asarray(i).ravel(), np.asarray(j).ravel())), (n, len(X)))
        _nsum_cache[key] = hit = (weakref.ref(i), weakref.ref(j), A)
        weakref.finalize(i, _nsum_cache.pop, key, None)
    A = hit[2]
    shp = X.shape
    out = A @ np.asarray(X).reshape(shp[0], -1)
    return out.reshape((n,) + shp[1:]).astype(dt, copy=False)

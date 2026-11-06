"""Compact packing for the arrays in body-rig.js and skin-uv.js.

Every array used to be written as plain base64 of its raw bytes. Big ones are now written as
    {"z": base64(gzip(byte planes)), "w": item size, "n": byte count}
"Byte planes" means all the first bytes of the items, then all the second bytes, and so on (a byte shuffle):
neighbouring numbers share their high bytes, so gzip finds far more repeats (triangle lists shrink ~4x).
An array identical to one already written is stored as {"ref": "body.index"} (the fat and the skin share
their triangles). The page unpacks everything once at startup (unpackRig in index.html), so the rest of the
code sees the same bytes as before.

    pack(obj)    plain base64 strings -> packed (needs the item size of each array: see WIDTH / b64())
    unpack(obj)  packed -> plain base64 strings, exactly as before (for the Python tools that read the files)
    b64(a)       one numpy array -> packed (or plain base64 when small); used by the writers
"""
import base64, gzip
import numpy as np

MIN_BYTES = 4096    # smaller arrays stay plain base64 (not worth the wrapper)


def _shuffle(raw, w):
    if w == 1: return raw
    return np.frombuffer(raw, np.uint8).reshape(-1, w).T.tobytes()


def _unshuffle(raw, w):
    if w == 1: return raw
    return np.frombuffer(raw, np.uint8).reshape(w, -1).T.tobytes()


def pack_bytes(raw, w):
    if len(raw) < MIN_BYTES: return base64.b64encode(raw).decode()
    if len(raw) % w: w = 1
    z = gzip.compress(_shuffle(raw, w), 9, mtime=0)    # (mtime=0: same input -> same file)
    return {'z': base64.b64encode(z).decode(), 'w': w, 'n': len(raw)}


def b64(a):
    a = np.ascontiguousarray(a)
    return pack_bytes(a.tobytes(), a.dtype.itemsize)


def unpack_bytes(v):
    raw = _unshuffle(gzip.decompress(base64.b64decode(v['z'])), v['w'])
    assert len(raw) == v['n']
    return raw


def is_packed(v): return isinstance(v, dict) and 'z' in v and 'w' in v and 'n' in v


def dedupe(obj):
    """Replace every packed array identical to an earlier one with {"ref": "<its path>"}."""
    seen = {}

    def walk(x, path):
        if isinstance(x, dict):
            for k in list(x):
                p = f'{path}.{k}' if path else k
                v = x[k]
                if is_packed(v):
                    if v['z'] in seen: x[k] = {'ref': seen[v['z']]}
                    else: seen[v['z']] = p
                else: walk(v, p)
        elif isinstance(x, list):
            for i, v in enumerate(x): walk(v, f'{path}.{i}')
    walk(obj, '')
    return obj


def unpack(obj):
    """Packed arrays and refs back to plain base64 strings, in place (returns obj)."""
    def get(path):
        x = obj
        for k in path.split('.'): x = x[int(k)] if isinstance(x, list) else x[k]
        return x

    refs = []

    def walk(x, path):
        items = x.items() if isinstance(x, dict) else enumerate(x) if isinstance(x, list) else ()
        for k, v in list(items):
            if is_packed(v): x[k] = base64.b64encode(unpack_bytes(v)).decode()
            elif isinstance(v, dict) and set(v) == {'ref'}: refs.append((x, k, v['ref']))
            else: walk(v, f'{path}.{k}')
    walk(obj, '')
    for x, k, p in refs:
        src = get(p)
        x[k] = src if isinstance(src, str) else base64.b64encode(unpack_bytes(src)).decode()
    return obj


def load_js(path):
    """The object in a 'window.X = {...};' file, unpacked to plain base64 strings."""
    import json
    s = open(path).read()
    return unpack(json.loads(s[s.index('{'):s.rindex('}') + 1]))

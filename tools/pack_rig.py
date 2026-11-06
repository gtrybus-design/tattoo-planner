"""Repack an existing body-rig.js / skin-uv.js into the compact format (see rigpack.py), without a rebuild.

usage: python3 -I tools/pack_rig.py [project dir]

Files that are already packed are left as they are. The packed file is checked to unpack to exactly the same
data before it replaces the old one (written to .tmp first, then swapped in).
"""
import copy, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rigpack

# item size of each array, by key (only affects how well it compresses, never the data)
WIDTH = {'q': 2, 'origin': 2, 'insertion': 2,
         'skinIndex': 1, 'weight8': 1, 'weight': 1, 'digit': 1, 'color': 1, 'sign': 1, 'glue': 1, 't': 1,
         'group': 1, 'endBone': 1, 'bone': 1}
SKIP = {'gz', 'name', 'type', 'parent', 'follows'}    # already compressed, or not arrays


def pack_plain(obj):
    def walk(x):
        items = x.items() if isinstance(x, dict) else enumerate(x) if isinstance(x, list) else ()
        for k, v in list(items):
            if isinstance(v, str) and k not in SKIP and len(v) >= 64:
                raw = rigpack.base64.b64decode(v, validate=True)
                x[k] = rigpack.pack_bytes(raw, WIDTH.get(k, 4))
            else: walk(v)
    walk(obj)
    return rigpack.dedupe(obj)


def repack(path):
    s = open(path).read()
    head, body = s[:s.index('{')], s[s.index('{'):s.rindex('}') + 1]
    obj = json.loads(body)
    if '"z":' in body[:2_000_000] and any(rigpack.is_packed(v) for v in _values(obj)):
        print(path, 'already packed'); return
    orig = copy.deepcopy(obj)
    packed = pack_plain(obj)
    text = json.dumps(packed, separators=(',', ':'))
    assert rigpack.unpack(json.loads(text)) == orig, 'packed file does not unpack to the same data'
    with open(path + '.tmp', 'w') as fh:
        fh.write(head + text + ';\n')
    os.replace(path + '.tmp', path)
    print(f'{path}: {len(s) / 1e6:.1f} MB -> {os.path.getsize(path) / 1e6:.1f} MB')


def _values(x):
    if isinstance(x, dict):
        for v in x.values(): yield v; yield from _values(v)
    elif isinstance(x, list):
        for v in x: yield from _values(v)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else '.'
    for f in ('body-rig.js', 'skin-uv.js'):
        if os.path.exists(f'{out}/{f}'): repack(f'{out}/{f}')

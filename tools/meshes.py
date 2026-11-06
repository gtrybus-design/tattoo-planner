"""Read the .meshes files written by convert_fbx.py."""
import json, struct
import numpy as np


def read_meshes(path):
    blob = open(path, 'rb').read()
    hl = struct.unpack('<I', blob[:4])[0]
    header = json.loads(blob[4:4 + hl])
    off = 4 + hl
    nv = sum(h['verts'] for h in header); nt = sum(h['tris'] for h in header)
    P = np.frombuffer(blob, np.float32, nv * 3, off).reshape(-1, 3); off += nv * 12
    I = np.frombuffer(blob, np.uint32, nt * 3, off).reshape(-1, 3)
    out, vo, to = [], 0, 0
    for h in header:
        out.append((h['name'], P[vo:vo + h['verts']], I[to:to + h['tris']]))
        vo += h['verts']; to += h['tris']
    return out

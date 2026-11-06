"""Fit a rigged base-mesh body onto a real skeleton and muscles (Z-Anatomy).

usage: python3 -I fit_rig.py <anatomy-src dir> <male_base_mesh.glb> <output dir>
(first convert the Z-Anatomy FBX files with tools/convert_fbx.py)

1. Every joint of the base mesh's rig is moved onto the matching joint of the real skeleton
   (shoulders, elbows, every finger knuckle, hips, knees, ankles, spine levels...).
2. The mesh is smoothed (Loop subdivision) and shrink-wrapped over the real muscles,
   plus a thin skin layer.
3. Every real bone is attached to the rig bone it belongs to, and every muscle is weighted to
   the joints it spans, so posing the rig moves skeleton, muscles and skin together.

Anatomy: Z-Anatomy (built on BodyParts3D), CC BY-SA 4.0.
Base mesh: "Male Base Mesh" by Orange Juice Games / BoQsc, CC0.
"""
import os, sys, re, json, struct, base64, gzip, pickle, zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meshes import read_meshes
import fastgeo                     # KD-tree nearest-point search and a fast scatter-add (see tools/fastgeo.py)
from fastgeo import add_at

src, glb_path, out = sys.argv[1:4]
SRC = src   # (kept separately: 'src' gets reused as a variable name further down)

SKIN = 0.006          # skin + thin fat layer over the muscles (m)
SUBDIVISIONS = 2

# ---------------------------------------------------------------- Z-Anatomy
# Z-Anatomy is in cm. This maps it onto the frame of the BodyParts3D data it was built from
# (m, y up, z forward, +x = the body's left), measured by matching 10 bone centres.
Z_SCALE, Z_OFFSET = 0.0100398, np.array([-0.00057, -0.06808, 0.10933])


def weld_mesh(P, I, tol=1e-3):
    key = np.round(P / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel(); F = inv[I]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return P[first] * Z_SCALE + Z_OFFSET, F


def decimate(V, F, cell, keep_sides=True):
    """Merge vertices closer than `cell` (keeps the shape, drops detail we can't see).
    Only vertices facing the same way are merged, so thin sheets of muscle keep both their sides."""
    N = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3): add_at(N, F[:, k], fn)
    facing = np.abs(N).argmax(1) * 2 + (N[np.arange(len(N)), np.abs(N).argmax(1)] > 0)
    key = np.c_[np.round(V / cell).astype(np.int64), facing] if keep_sides else np.round(V / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    cnt = np.bincount(inv); Q = np.zeros((len(cnt), 3))
    for c in range(3): Q[:, c] = np.bincount(inv, V[:, c]) / cnt
    I = inv[F]; I = I[(I[:, 0] != I[:, 1]) & (I[:, 1] != I[:, 2]) & (I[:, 0] != I[:, 2])]
    return Q, I


BONE_RE = re.compile(
    r'^(Humerus|Radius|Ulna|Clavicle|Scapula|Femur|Tibia|Fibula|Patella|Talus|Calcaneus|Navicular_bone|Cuboid_bone|'
    r'(Medial|Intermediate|Lateral)_cuneiform_bone|\w+_metatarsal_bone|\w+_metacarpal_bone|(Proximal|Middle|Distal)_phalanx_of_\w+|'
    r'(Scaphoid|Lunate|Triquetrum|Pisiform|Trapezium|Trapezoid|Capitate|Hamate)_bone|Hip_bone|Sacrum|Coccyx|Vertebra_\w+|'
    r'Atlas_\(C1\)|Axis_\(C2\)|[A-Z][a-z]+_rib|Costal_cartilage_of_\w+_rib|Manubrium_of_sternum|Body_of_sternum|Xiphoid_process|'
    r'Frontal_bone|Parietal_bone|Occipital_bone|Temporal_bone|Sphenoid_bone|Ethmoid_bone|Maxilla|Mandible|Zygomatic_bone|'
    r'Nasal_bone|Lacrimal_bone|Palatine_bone|Vomer|Inferior_nasal_concha|Hyoid_bone|Thyroid_cartilage|Cricoid_cartilage|'
    r'\w+_(incisor|canine|premolar|molar_tooth))[lr]?$')
BONES = {}
for n, P, I in read_meshes(f'{src}/zanatomy/SkeletalSystem100.meshes'):
    if BONE_RE.match(n) and len(P) > 30 and n not in BONES:
        BONES[n] = weld_mesh(P, I)
print('bones', len(BONES), file=sys.stderr)
zb = lambda n: BONES[n][0]


# Frame: floor under the feet, scaled to 1.80 m, hip joints at z = 0
skull = np.vstack([zb(n) for n in ['Frontal_bone', 'Parietal_bonel', 'Parietal_boner', 'Occipital_bone']])
FLOOR = -0.075
TOP = skull[:, 1].max() + 0.006                     # + scalp
K = 1.80 / (TOP - FLOOR)
ARM_OUT = np.radians(35)   # the body is built in an A-pose (arms 35 degrees out) so the armpits are open
SHOULDER = ZC = None       # set from the landmarks below


def rot_arm(P, side=1):
    """Swing points of an arm outward around the shoulder joint (raw coords)."""
    P = np.array(P, float)
    if side < 0: P[..., 0] *= -1
    d = P[..., :2] - SHOULDER[:2]
    c, s = np.cos(ARM_OUT), np.sin(ARM_OUT)
    x = SHOULDER[0] + d[..., 0] * c - d[..., 1] * s
    y = SHOULDER[1] + d[..., 0] * s + d[..., 1] * c
    P[..., 0], P[..., 1] = x, y
    if side < 0: P[..., 0] *= -1
    return P


def app(P):
    P = np.array(P, float)
    return np.stack([P[..., 0] * K, (P[..., 1] - FLOOR) * K, (P[..., 2] - ZC) * K], -1)


def end(V, toward, frac=0.12):
    """Centre of the end of a bone nearest to a point."""
    d = np.linalg.norm(V - toward, axis=1)
    return V[d <= np.quantile(d, frac)].mean(0)


def far(V, frm):
    return V[np.argmax(np.linalg.norm(V - frm, axis=1))]


def sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    return np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0][:3]


def body_center(name):
    """Centre of a vertebral body (the front half of the vertebra)."""
    V = zb(name); front = V[V[:, 2] > np.median(V[:, 2])]
    return np.array([0.0, front[:, 1].mean(), front[:, 2].mean()])


# ---------------------------------------------------------------- landmarks (left side, raw coords)
L = {}
H = zb('Humerusl')
L['upper_arm'] = sphere(H[H[:, 1] > H[:, 1].max() - 0.035])
lo = H[H[:, 1] < H[:, 1].min() + 0.02]; L['forearm'] = (lo.min(0) + lo.max(0)) / 2
R_ = zb('Radiusl'); lo = R_[R_[:, 1] < R_[:, 1].min() + 0.015]; L['hand'] = (lo.min(0) + lo.max(0)) / 2
W = L['hand']
for rig, num in [('f_index', 'second'), ('f_middle', 'third'), ('f_ring', 'fourth'), ('f_pinky', 'fifth')]:
    for k, part in [('01', 'Proximal'), ('02', 'Middle'), ('03', 'Distal')]:
        L[f'{rig}.{k}'] = end(zb(f'{part}_phalanx_of_{num}_finger_of_handl'), W)
    L[f'{rig}.03_tip'] = far(zb(f'Distal_phalanx_of_{num}_finger_of_handl'), W)
L['capitate'] = end(zb('Capitate_bonel'), W, 0.25)     # head of the capitate: the second wrist joint (between the carpal rows)
L['thumb.01'] = end(zb('First_metacarpal_bonel'), W)
L['thumb.02'] = end(zb('Proximal_phalanx_of_first_finger_of_handl'), W)
L['thumb.03'] = end(zb('Distal_phalanx_of_first_finger_of_handl'), W)
L['thumb.03_tip'] = far(zb('Distal_phalanx_of_first_finger_of_handl'), W)
C = zb('Claviclel'); L['shoulder'] = end(C, np.array([0.0, C[:, 1].mean(), C[:, 2].max()]))

F_ = zb('Femurl'); top = F_[F_[:, 1] > F_[:, 1].max() - 0.06]
L['thigh'] = sphere(top[np.abs(top[:, 0]) < np.median(np.abs(top[:, 0]))])
lo = F_[F_[:, 1] < F_[:, 1].min() + 0.025]; L['shin'] = (lo.min(0) + lo.max(0)) / 2
L['foot'] = zb('Talusl').mean(0)
toe_tip = far(zb('Distal_phalanx_of_second_finger_of_footl'), L['foot'])
L['toe'] = (end(zb('First_metatarsal_bonel'), toe_tip) + end(zb('Fifth_metatarsal_bonel'), toe_tip)) / 2
L['toe_tip'] = toe_tip
Cal = zb('Calcaneusl')
L['heel.02'] = end(Cal, np.array([Cal[:, 0].mean(), Cal[:, 1].min(), Cal[:, 2].min()]))

hip_mid = np.array([0.0, L['thigh'][1], L['thigh'][2]])
L['spine'] = hip_mid - np.array([0, 0.05, 0])
L['spine.001'] = body_center('Vertebra_L5') - np.array([0, 0.02, 0])
L['spine.002'] = (body_center('Vertebra_T12') + body_center('Vertebra_L1')) / 2
target_y = L['shoulder'][1] - 0.13
L['spine.003'] = min((body_center(f'Vertebra_T{i}') for i in range(1, 9)), key=lambda c: abs(c[1] - target_y))
L['spine.004'] = (body_center('Vertebra_C7') + body_center('Vertebra_T1')) / 2
head_top = skull[np.argmax(skull[:, 1])] * [0, 1, 1] + [0, 0.006, 0]
L['spine.005'] = L['spine.004'] + 0.19 * (head_top - L['spine.004'])   # same place along the neck as the base rig
L['spine.005_tip'] = head_top

SHOULDER, ZC = L['upper_arm'].copy(), L['thigh'][2]
L0 = {k: v.copy() for k, v in L.items()}     # landmarks before the arms are swung out
ARM_KEYS = {'upper_arm', 'forearm', 'hand', 'capitate', 'thumb.01', 'thumb.02', 'thumb.03', 'thumb.03_tip'} | \
    {k for k in L if k.startswith('f_')}
for k in ARM_KEYS: L[k] = rot_arm(L[k])
print('landmarks:', {k: np.round(app(v), 3).tolist() for k, v in L.items()}, file=sys.stderr)

# ---------------------------------------------------------------- base mesh (glTF)
blob = open(glb_path, 'rb').read()
jlen = struct.unpack('<I', blob[12:16])[0]
G = json.loads(blob[20:20 + jlen]); BIN = blob[20 + jlen + 8:]


def accessor(k):
    a = G['accessors'][k]; bv = G['bufferViews'][a['bufferView']]
    dt = {5126: np.float32, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32}[a['componentType']]
    nc = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
    return np.frombuffer(BIN, dt, a['count'] * nc, bv.get('byteOffset', 0) + a.get('byteOffset', 0)).reshape(a['count'], nc)


def node_matrix(nd):
    T = np.eye(4)
    x, y, z, w = nd.get('rotation', [0, 0, 0, 1])
    T[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]) * nd.get('scale', [1, 1, 1])
    T[:3, 3] = nd.get('translation', [0, 0, 0])
    return T


parent_of = {c: i for i, nd in enumerate(G['nodes']) for c in nd.get('children', [])}


def world(i):
    m = node_matrix(G['nodes'][i])
    while i in parent_of: i = parent_of[i]; m = node_matrix(G['nodes'][i]) @ m
    return m


prim = G['meshes'][0]['primitives'][0]
skin = G['skins'][0]
joints = skin['joints']
bone_names = [G['nodes'][j]['name'] for j in joints]
bone_parent = [bone_names.index(G['nodes'][parent_of[j]]['name']) if G['nodes'][parent_of[j]]['name'] in bone_names else -1 for j in joints]
IBM = accessor(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
P0 = accessor(prim['attributes']['POSITION']).astype(float)
UV = accessor(prim['attributes']['TEXCOORD_0']).astype(float)
J4 = accessor(prim['attributes']['JOINTS_0']).astype(int)
W4 = accessor(prim['attributes']['WEIGHTS_0']).astype(float)
TRI = accessor(prim['indices']).reshape(-1, 3).astype(int)
NB = len(bone_names)
Wd = np.zeros((len(P0), NB))
for k in range(4): add_at(Wd, (np.arange(len(P0)), J4[:, k]), W4[:, k])
Wd /= Wd.sum(1, keepdims=True)

# rest-pose vertex positions in rig space (normally IBM = inverse(joint world))
Pr = np.zeros_like(P0)
for b, j in enumerate(joints):
    M = world(j) @ IBM[b]
    Pr += Wd[:, b:b + 1] * (P0 @ M[:3, :3].T + M[:3, 3])
heads0 = np.array([world(j)[:3, 3] for j in joints])


def b2a(P):   # base mesh axes (x forward, z = right) -> ours (x = left, z forward)
    P = np.asarray(P, float)
    return np.stack([-P[..., 2], P[..., 1], P[..., 0]], -1)


Pr, heads0 = b2a(Pr), b2a(heads0)
idx = {n: i for i, n in enumerate(bone_names)}


def side_of(n): return 1 if n.endswith('.L') else -1 if n.endswith('.R') else 0
def base_name(n): return n[:-2] if side_of(n) else n


# ---------------------------------------------------------------- target joints (app coords)
def target(key, side):
    p = app(L[key]).copy()
    if side < 0: p[0] *= -1
    return p


heads1, tails0, tails1 = np.zeros((NB, 3)), np.zeros((NB, 3)), np.zeros((NB, 3))
children = {b: [c for c in range(NB) if bone_parent[c] == b] for b in range(NB)}
CHAIN_CHILD = {'spine': 'spine.001', 'spine.001': 'spine.002', 'spine.002': 'spine.003', 'spine.003': 'spine.004',
               'spine.004': 'spine.005', 'shoulder': 'upper_arm', 'upper_arm': 'forearm', 'forearm': 'hand',
               'hand': 'f_middle.01', 'thigh': 'shin', 'shin': 'foot', 'foot': 'toe',
               'thumb.01': 'thumb.02', 'thumb.02': 'thumb.03'}
for f in ['f_index', 'f_middle', 'f_ring', 'f_pinky']:
    CHAIN_CHILD[f'{f}.01'] = f'{f}.02'; CHAIN_CHILD[f'{f}.02'] = f'{f}.03'
LEAF_TIP = {f'{f}.03': f'{f}.03_tip' for f in ['f_index', 'f_middle', 'f_ring', 'f_pinky']}
LEAF_TIP.update({'thumb.03': 'thumb.03_tip', 'toe': 'toe_tip', 'spine.005': 'spine.005_tip'})
FOLLOW = {'pelvis': 'spine', 'heel.02': 'foot'}    # bones that just ride along with another bone

for b, n in enumerate(bone_names):
    s, bn = side_of(n), base_name(n)
    if bn in FOLLOW: continue
    heads1[b] = target(bn, s)
    sfx = n[len(bn):]
    if bn in CHAIN_CHILD:
        c = idx[CHAIN_CHILD[bn] + sfx]
        tails0[b], tails1[b] = heads0[c], target(CHAIN_CHILD[bn], s)
    else:
        tails1[b] = target(LEAF_TIP[bn], s)
        dom = Wd[:, b] > 0.5
        tails0[b] = Pr[dom][np.argmax(((Pr[dom] - heads0[b]) @ (tails1[b] - heads1[b])))] if dom.any() else heads0[b]
bone_follow = {b: (idx[FOLLOW[base_name(n)] + ('' if FOLLOW[base_name(n)] == 'spine' else n[len(base_name(n)):])])
               for b, n in enumerate(bone_names) if base_name(n) in FOLLOW}


def rotation_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), np.dot(a, b)
    if c < -0.999999: return -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def frame(axis):
    a = axis / np.linalg.norm(axis)
    l = np.array([1.0, 0, 0]) - a * a[0]
    if np.linalg.norm(l) < 1e-6: l = np.array([0, 0, 1.0]) - a * a[2]
    l /= np.linalg.norm(l)
    f = np.cross(a, l)
    if f[2] < 0 or (abs(f[2]) < 1e-6 and f[1] < 0): f = -f
    return a, l, f


XF = []
for b in range(NB):
    if b in bone_follow:
        XF.append(None); continue
    d0, d1 = tails0[b] - heads0[b], tails1[b] - heads1[b]
    R = rotation_between(d0, d1)
    s = np.linalg.norm(d1) / max(np.linalg.norm(d0), 1e-9)
    a0 = d0 / np.linalg.norm(d0)
    XF.append(dict(R=R, s=s, h0=heads0[b], a0=a0, h1=heads1[b], len=np.linalg.norm(d1), fr=frame(d1)))
for b, o in bone_follow.items():
    XF[b] = XF[o]
    heads1[b] = XF[o]['h1'] + XF[o]['R'] @ (heads0[b] - XF[o]['h0']) * XF[o]['s']

# ---------------------------------------------------------------- which rig bone each real bone belongs to
ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth']
FINGER_RIG = {'second': 'f_index', 'third': 'f_middle', 'fourth': 'f_ring', 'fifth': 'f_pinky'}
ARM_BONE = re.compile(r'Humerus|Radius|Ulna|carpal|metacarpal|finger_of_hand|Scaphoid|Lunate|Triquetrum|Pisiform|Trapezium|Trapezoid|Capitate|Hamate')


def side_suffix(n):
    """'.L' / '.R' when the name ends in l/r and its mirror twin exists (so 'Vomer' stays central)."""
    if n[-1] in 'lr' and (n[:-1] + ('r' if n[-1] == 'l' else 'l')) in BONES:
        return '.L' if n[-1] == 'l' else '.R'
    return ''


def rig_bone_for(n):
    s = side_suffix(n)
    m = re.match(r'(Proximal|Middle|Distal)_phalanx_of_(\w+)_finger_of_(hand|foot)', n)
    if m:
        part, num, where = m.groups()
        if where == 'foot': return 'toe' + s
        if num == 'first': return ('thumb.02' if part == 'Proximal' else 'thumb.03') + s
        return FINGER_RIG[num] + {'Proximal': '.01', 'Middle': '.02', 'Distal': '.03'}[part] + s
    if n.startswith('First_metacarpal'): return 'thumb.01' + s
    if re.search(r'metacarpal|Scaphoid|Lunate|Triquetrum|Pisiform|Trapezium|Trapezoid|Capitate|Hamate', n): return 'hand' + s
    if re.match(r'Radius|Ulna', n): return 'forearm' + s
    if n.startswith('Humerus'): return 'upper_arm' + s
    if re.match(r'Clavicle|Scapula', n): return 'shoulder' + s
    if re.search(r'Talus|Calcaneus|Navicular|Cuboid|cuneiform|metatarsal', n): return 'foot' + s
    if re.match(r'Tibia|Fibula|Patella', n): return 'shin' + s
    if n.startswith('Femur'): return 'thigh' + s
    if re.match(r'Hip_bone|Sacrum|Coccyx', n): return 'spine'
    if n.startswith('Vertebra_L'): return 'spine.001'
    if n.startswith('Vertebra_T'): return 'spine.003' if int(n[10:]) <= 6 else 'spine.002'
    m = re.match(r'(?:Costal_cartilage_of_)?([A-Za-z]+)_rib', n)   # 'Fifth_rib' and 'Costal_cartilage_of_fifth_rib'
    if m: return 'spine.003'          # the ribcage moves as one rigid unit
    if re.match(r'Manubrium|Body_of_sternum|Xiphoid', n): return 'spine.003'
    if re.match(r'Vertebra_C|Hyoid|Thyroid|Cricoid', n): return 'spine.004'
    return 'spine.005'   # skull, jaw, teeth, atlas, axis


# ---------------------------------------------------------------- muscles: load, thin out, weight to the joints they span
HIDDEN = re.compile(   # deep or internal muscles that never shape the outside of the body
    r'intercostal|Diaphragm|Transversus|Palmar_aponeurosis|pharyn|laryn|Palat|gloss|Tongue|arytenoid|thyro|crico|uvula|'
    r'^(Superior|Inferior)_oblique_muscle|^(Superior|Inferior|Medial|Lateral)_rectus_muscle|palpebrae|bursa|sheath|tympani|'
    r'Stapedius|Levator_ani|Coccygeus|sphincter|Multifidus|Rotatores|Interspinal|Intertransvers|Levatores|Subcostal|'
    r'Psoas_minor|Iliacus|Quadratus_lumborum|Obturator|Gemell|Piriformis|Quadratus_femoris|Trochlea|Cricothyroid|'
    r'^Fascia|fascia[lr]?$|_fascia_', re.I)   # fascia sheets would hide the muscle shapes
TENDON = re.compile(r'tendon|aponeuros|tract|fascia|retinacul|Linea|ligament', re.I)
MUS = []
for n, P, I in read_meshes(f'{src}/zanatomy/MuscularSystem100.meshes'):
    if HIDDEN.search(n): continue
    V, F = decimate(*weld_mesh(P, I), 0.006)
    if len(F) >= 4: MUS.append((n, V, F))
print('muscles', len(MUS), 'verts', sum(len(v) for _, v, _ in MUS), file=sys.stderr)


def to_raw(P):
    P = np.asarray(P, float)
    return np.stack([P[..., 0] / K, P[..., 1] / K + FLOOR, P[..., 2] / K + ZC], -1)


def targetU(key, side):   # joint position before the arms were swung out
    p = app(L0[key]).copy()
    if side < 0: p[0] *= -1
    return p


seg_h, seg_t, seg_b = [], [], []
for b, n in enumerate(bone_names):
    bn = base_name(n)
    if bn in FOLLOW: continue
    seg_h.append(targetU(bn, side_of(n))); seg_t.append(targetU(CHAIN_CHILD.get(bn) or LEAF_TIP[bn], side_of(n))); seg_b.append(b)
seg_h, seg_t, seg_b = np.array(seg_h), np.array(seg_t), np.array(seg_b)
seg_side = np.array([side_of(bone_names[b]) for b in seg_b])
ARM = {'upper_arm', 'forearm', 'hand'} | {base_name(n) for n in bone_names if n.startswith(('f_', 'thumb'))}
LEG = {'thigh', 'shin', 'foot', 'toe', 'heel.02'}


def joint_weights(P):
    """Weights to the two nearest rig bones (like automatic skinning), P in app coords before the arm swing."""
    D = np.empty((len(P), len(seg_b)))
    for k, (h, t) in enumerate(zip(seg_h, seg_t)):
        ab = t - h; u = np.clip(((P - h) @ ab) / (ab @ ab), 0, 1)
        D[:, k] = np.linalg.norm(P - (h + u[:, None] * ab), axis=1)
        if seg_side[k] > 0: D[P[:, 0] < -0.005, k] = np.inf
        if seg_side[k] < 0: D[P[:, 0] > 0.005, k] = np.inf
    near = np.argsort(D, 1)[:, :2]
    w = 1 / np.maximum(np.take_along_axis(D, near, 1), 0.004) ** 4
    return seg_b[near], w / w.sum(1, keepdims=True)


ARM_MUSCLE = re.compile(
    r'deltoid|biceps_brachii|triceps|Brachialis|Coracobrachialis|Brachioradialis|Anconeus|Pronator|Supinator|carpi|Palmaris|'
    r'digitorum_(superficialis|profundus)|Extensor_digitorum_muscle|digiti_minimi(?!.*foot)|indicis|pollicis|of_hand|'
    r'Lumbrical_muscles?_of_hand|Opponens|intermuscular_septum_of_arm|Extensor_digitorum[lr]?$|Palmar_interossei|'
    r'Common_flexor_tendon_sheath|retinaculum_of_wrist|metacarpal_ligament', re.I)
LEG_MUSCLE = re.compile(
    r'Vastus|Rectus_femoris|Sartorius|Gracilis|Adductor_(longus|brevis|magnus|minimus)|biceps_femoris|Semitendinosus|'
    r'Semimembranosus|gastrocnemius|Soleus|Tibialis|Fibularis|digitorum_longus|hallucis|Popliteus|Plantaris|'
    r'Tensor_fasciae_latae|Iliotibial|Calcaneal_tendon|of_foot|digitorum_brevis|Quadratus_plantae|Articularis_genus|Pectineus', re.I)
def swing_arms(P, name, bidx, w):
    """Apply the rest-pose arm swing. Only arm muscles swing with the arm (by how much each point follows
    the arm bones); lats, teres major and pecs only near where they attach to the humerus; the rest stays."""
    side = 1 if name.rstrip(')')[-1] == 'l' else -1
    if ARM_MUSCLE.search(name) and 'foot' not in name:
        share = sum(np.where([base_name(bone_names[b]) in ARM for b in bidx[:, k]], w[:, k], 0) for k in range(bidx.shape[1]))
    elif re.search(r'Latissimus_dorsi|Teres_major|pectoralis_major', name, re.I):
        # these stay snug on the trunk: only the tendon end moves, carried by the attachment point on the
        # humerus (full within 3 cm of it, fading out by 8 cm), so nothing swings out sideways with the arm
        sh, el = app(L0['upper_arm']), app(L0['forearm'])
        if side < 0: sh[0] *= -1; el[0] *= -1
        ins = sh + 0.18 * (el - sh)
        share = np.clip((0.08 - np.linalg.norm(P - ins, axis=1)) / 0.05, 0, 1)
        ins2 = app(rot_arm(to_raw(ins[None]), side))[0]
        return P + share[:, None] * (ins2 - ins)
    else:
        return P
    Q = app(rot_arm(to_raw(P), side))
    return P + share[:, None] * (Q - P)


def region(b):
    n = bone_names[b]; bn = base_name(n); s = 'L' if n.endswith('.L') else 'R'
    return 'arm' + s if bn in ARM else 'leg' + s if bn in LEG else 'axial'


mus_P, mus_F, mus_B, mus_W, mus_C, mus_id, off = [], [], [], [], [], [], 0
pools = {k: [] for k in ['armL', 'armR', 'legL', 'legR', 'axial']}
for i, (n, V, F) in enumerate(MUS):
    Pa = app(V)
    bidx, w = joint_weights(Pa)
    Pr_ = swing_arms(Pa, n, bidx, w)
    mus_P.append(Pr_); mus_F.append(F + off); mus_B.append(bidx); mus_W.append(w); mus_id.append(np.full(len(V), i)); off += len(V)
    rng = np.random.default_rng(zlib.crc32(n.encode()))   # (crc32, not hash(): Python randomises hash() every run)
    base = np.array([217, 210, 195]) if TENDON.search(n) else np.array([178, 74, 63]) * rng.uniform(0.85, 1.12)
    mus_C.append(np.tile(np.clip(base, 0, 255), (len(V), 1)))
    # the skin wraps each region over that region's muscles, decided by anatomy (by name), not by distance:
    # e.g. the serratus sits right under the hanging arm but belongs to the trunk
    sd = 'L' if n.rstrip(')')[-1] == 'l' else 'R'
    reg = ('leg' + sd) if (LEG_MUSCLE.search(n) or 'foot' in n) else ('arm' + sd) if ARM_MUSCLE.search(n) else 'axial'
    pools[reg].append(Pr_)
mus_P, mus_F = np.vstack(mus_P), np.vstack(mus_F)
mus_B, mus_W, mus_C = np.vstack(mus_B), np.vstack(mus_W), np.vstack(mus_C).astype(np.uint8)

# bones join the pools too (where there's no muscle, e.g. shins, knuckles, collarbones)
SKEL = []
for n, (V, F) in BONES.items():
    rb = idx[rig_bone_for(n)]
    if ARM_BONE.search(n): V = rot_arm(V, 1 if side_suffix(n) == '.L' else -1)
    SKEL.append((n, rb, V, F))
    pools[region(rb)].append(app(V))
pools = {k: np.vstack(v) for k, v in pools.items()}


# ---------------------------------------------------------------- rigid fit: rig joints onto the skeleton
def deform(P):
    out = np.zeros_like(P)
    for b in range(NB):
        m = Wd[:, b] > 1e-4
        if not m.any(): continue
        x = XF[b]
        out[m] += Wd[m, b:b + 1] * (x['h1'] + ((P[m] - x['h0']) @ x['R'].T) * x['s'])
    return out


Pf = deform(Pr)


# ---------------------------------------------------------------- Loop subdivision (keeps UV seams)
def weld(P, tol=1e-5):
    key = np.round(P / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return first, inv.ravel()


def loop_subdivide(Pw, Tw, Ts, attrs, wmap):
    edges = {}
    for t, (a, b, c) in enumerate(Tw):
        for u, v, o in ((a, b, c), (b, c, a), (c, a, b)):
            edges.setdefault((min(u, v), max(u, v)), []).append(o)
    nbr = [set() for _ in range(len(Pw))]
    boundary = [[] for _ in range(len(Pw))]
    for (u, v), opp in edges.items():
        nbr[u].add(v); nbr[v].add(u)
        if len(opp) == 1: boundary[u].append(v); boundary[v].append(u)
    newV = np.zeros_like(Pw)
    for i in range(len(Pw)):
        if boundary[i]:
            newV[i] = 0.75 * Pw[i] + 0.125 * (Pw[boundary[i][0]] + Pw[boundary[i][-1]])
        else:
            n = len(nbr[i]); beta = 3 / (8 * n) if n > 3 else 3 / 16
            newV[i] = (1 - n * beta) * Pw[i] + beta * Pw[list(nbr[i])].sum(0)
    eidx, eP = {}, []
    for (u, v), opp in edges.items():
        eidx[(u, v)] = len(Pw) + len(eP)
        eP.append(0.375 * (Pw[u] + Pw[v]) + 0.125 * (Pw[opp[0]] + Pw[opp[1]]) if len(opp) == 2 else 0.5 * (Pw[u] + Pw[v]))
    Pw2 = np.vstack([newV, np.array(eP)])
    # split-vertex side: originals keep their attributes, edge points average them
    sidx, s_w, s_attr = {}, list(wmap), {k: list(v) for k, v in attrs.items()}
    ns = len(wmap)

    def split_edge(sa, sb):
        nonlocal ns
        k = (min(sa, sb), max(sa, sb))
        if k not in sidx:
            sidx[k] = ns; ns += 1
            wa, wb = wmap[sa], wmap[sb]
            s_w.append(eidx[(min(wa, wb), max(wa, wb))])
            for name in s_attr: s_attr[name].append((attrs[name][sa] + attrs[name][sb]) / 2)
        return sidx[k]
    Tw2, Ts2 = [], []
    for (a, b, c), (sa, sb, sc) in zip(Tw, Ts):
        ab, bc, ca = eidx[(min(a, b), max(a, b))], eidx[(min(b, c), max(b, c))], eidx[(min(c, a), max(c, a))]
        sab, sbc, sca = split_edge(sa, sb), split_edge(sb, sc), split_edge(sc, sa)
        Tw2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        Ts2 += [(sa, sab, sca), (sb, sbc, sab), (sc, sca, sbc), (sab, sbc, sca)]
    return Pw2, np.array(Tw2), np.array(Ts2), {k: np.array(v) for k, v in s_attr.items()}, np.array(s_w)


first, wmap = weld(Pf)
Pw, Tw, Ts = Pf[first], wmap[TRI], TRI
attrs = {'uv': UV, 'w': Wd}
for _ in range(SUBDIVISIONS):
    Pw, Tw, Ts, attrs, wmap = loop_subdivide(Pw, Tw, Ts, attrs, wmap)


# ---------------------------------------------------------------- shrink-wrap onto the muscles and bones
def normals(P, T):
    N = np.zeros_like(P)
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    for k in range(3): add_at(N, T[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)   # (points on no triangle stay 0)


def voxel_downsample(P, cell=0.005):
    _, first_ = np.unique(np.round(P / cell).astype(np.int64), axis=0, return_index=True)
    return P[first_].astype(np.float32)


def nearest(Q, pool, margin=0.07):
    """Nearest anatomy point for each query point (NaN when nothing is within the margin)."""
    d, j = fastgeo.tree(pool).query(np.asarray(Q, np.float64), distance_upper_bound=margin, workers=-1)
    out = np.full(Q.shape, np.nan); ok = np.isfinite(d)
    out[ok] = pool[j[ok]]
    return out


pools_ds = {k: voxel_downsample(v) for k, v in pools.items()}
Ww = np.zeros((len(Pw), NB)); Ww[wmap] = attrs['w']
dom = np.array([base_name(bone_names[b]) for b in Ww.argmax(1)])
vside = np.where(Pw[:, 0] >= 0, 'L', 'R')
KEEP = ['spine.005', 'foot', 'toe', 'heel.02']   # head and feet keep the base mesh shape (no nose or sole pads in the data)
keep_w = sum(Ww[:, b] for b, n in enumerate(bone_names) if base_name(n) in KEEP)
ARM = {'upper_arm', 'forearm', 'hand'} | {base_name(n) for n in bone_names if n.startswith(('f_', 'thumb'))}
groups = []
for sd in 'LR':
    groups.append((np.isin(dom, list(ARM)) & (vside == sd), pools_ds['arm' + sd]))
    groups.append(((dom == 'shoulder') & (vside == sd), np.vstack([pools_ds['arm' + sd], pools_ds['axial']])))
    groups.append((np.isin(dom, ['thigh', 'shin']) & (vside == sd), pools_ds['leg' + sd]))
axial_mask = np.isin(dom, ['spine', 'spine.001', 'spine.002', 'spine.003', 'spine.004', 'pelvis'])
groups.append((axial_mask, pools_ds['axial']))

ei = np.concatenate([Tw[:, 0], Tw[:, 1], Tw[:, 2], Tw[:, 1], Tw[:, 2], Tw[:, 0]])
ej = np.concatenate([Tw[:, 1], Tw[:, 2], Tw[:, 0], Tw[:, 0], Tw[:, 1], Tw[:, 2]])
deg = np.bincount(ei, minlength=len(Pw)).astype(float)


def smooth_field(D, passes):
    for _ in range(passes):
        acc = fastgeo.nsum(ei, ej, D)
        D = 0.5 * D + 0.5 * acc / deg
    return D


def taubin(P, passes=2, lam=0.5, mu=-0.53):
    """Smooth out bumps without shrinking the surface."""
    for _ in range(passes):
        for k in (lam, mu):
            acc = fastgeo.nsum(ei, ej, P)
            P = P + k * (acc / deg[:, None] - P)
    return P


# Vertices shared between bones (shoulders, armpits, hips) move less, so the surface can't tear
confidence = Ww.max(1) ** 2
P_start = Pw.copy()
MAX_MOVE = 0.08
for it in range(20):
    N = normals(Pw, Tw)
    delta = np.zeros(len(Pw))
    for mask, pool in groups:
        near = nearest(Pw[mask], pool)
        dd = ((near - Pw[mask]) * N[mask]).sum(1) + SKIN    # how far to move along the normal
        delta[mask] = np.clip(np.nan_to_num(dd), -0.012, 0.012)
    delta = smooth_field(delta * (1 - keep_w) * confidence, 4)
    Pw = Pw + N * (0.7 * delta)[:, None]
    Pw = taubin(Pw, 1)
    D = Pw - P_start; n = np.linalg.norm(D, axis=1, keepdims=True)
    Pw = P_start + D * np.minimum(1, MAX_MOVE / np.maximum(n, 1e-9))
    print(f'wrap pass {it}: mean move {np.abs(delta).mean() * 1000:.1f} mm', file=sys.stderr)

# Where the arm meets the chest the rigid fit squeezes the skin into a fold (the base mesh's arms
# started at 45 degrees). Relax the surface there: vertices shared between an arm bone and the torso.
arm_w = sum(Ww[:, b] for b, n in enumerate(bone_names) if base_name(n) in {'upper_arm', 'shoulder'})
torso_w = sum(Ww[:, b] for b, n in enumerate(bone_names) if base_name(n) in {'spine.002', 'spine.003', 'spine.004'})
blend = smooth_field(np.minimum(arm_w, torso_w) * 2, 4)
blend = np.clip(blend * 2, 0, 1)[:, None]
# Armpit: the skin between the upper arm and the side of the chest is too far from any muscle for the
# main wrap, so it hangs across as a web. Pull it into the hollow between the pec (front), the lats
# (back), the ribs/serratus (inside) and the arm, wrapping onto arm and trunk muscles together.
zone = smooth_field(np.minimum(arm_w, torso_w), 6) > 0.02
for sd, sgn in (('L', 1), ('R', -1)):
    m = zone & (Pw[:, 0] * sgn > 0)
    pool = np.vstack([pools_ds['arm' + sd], pools_ds['axial']])
    start = Pw[m].copy()
    for it in range(20):
        N = normals(Pw, Tw)
        near = nearest(Pw[m], pool, margin=0.15)
        dd = np.clip(np.nan_to_num(((near - Pw[m]) * N[m]).sum(1) + SKIN), -0.012, 0.012)
        delta = np.zeros(len(Pw)); delta[m] = dd
        delta = smooth_field(delta, 3)
        Pw[m] += N[m] * (0.7 * delta[m])[:, None]
        D = Pw[m] - start; n = np.linalg.norm(D, axis=1, keepdims=True)
        Pw[m] = start + D * np.minimum(1, 0.12 / np.maximum(n, 1e-9))
    print(f'armpit {sd}: {m.sum()} skin points, mean pull {np.linalg.norm(Pw[m] - start, axis=1).mean() * 1000:.0f} mm', file=sys.stderr)
Pw = taubin(Pw, 2)
# a thin layer of fat softens the step from the ribcage down to the waist
trunk = smooth_field(axial_mask.astype(float), 3)[:, None] * (1 - keep_w[:, None])
for _ in range(6):
    for k in (0.5, -0.53):
        acc = fastgeo.nsum(ei, ej, Pw)
        Pw = Pw + k * trunk * (acc / deg[:, None] - Pw)

# put the soles on the floor
dy = -Pw[:, 1].min()
Pw[:, 1] += dy; heads1[:, 1] += dy
print('floor shift', round(dy, 4), 'height', round(Pw[:, 1].max(), 3), file=sys.stderr)

# smooth normals on the welded surface
Nw = normals(Pw, Tw)
pos, nor, uv = Pw[wmap], Nw[wmap], attrs['uv']
Wf = attrs['w']
top4 = np.argsort(-Wf, 1)[:, :4]
w4 = np.take_along_axis(Wf, top4, 1); w4 /= w4.sum(1, keepdims=True)
print('body verts', len(pos), 'tris', len(Ts), file=sys.stderr)

# ---------------------------------------------------------------- skeleton bones attached to rig bones
skP, skI, skB, base_i = [], [], [], 0
skN = []      # which real bone each skeleton point comes from
for n, rb, V, F in SKEL:
    if re.search(r'tooth|incisor|canine|premolar', n): continue
    Q, I = decimate(V, F, 0.005)
    Q = app(Q); Q[:, 1] += dy
    skP.append(Q); skI.append(I + base_i); skB.append(np.full(len(Q), rb)); skN.extend([n] * len(Q)); base_i += len(Q)
skP, skI, skB = np.vstack(skP).astype(np.float32), np.vstack(skI).astype(np.uint32), np.concatenate(skB).astype(np.uint8)
print('skeleton verts', len(skP), 'tris', len(skI), file=sys.stderr)

mus_P[:, 1] += dy
print('muscle verts', len(mus_P), 'tris', len(mus_F), file=sys.stderr)


def nearest_index(Q, pool, margin=0.25, fallback=True):
    return fastgeo.nearest_index(Q, pool, margin, fallback)   # (a KD-tree: tools/fastgeo.py)


# Muscles take their joint weights from the skin right above them: the skin's weights were set up
# by an artist (a lat follows the spine, not the arm hanging next to it), so muscles bend like the skin.
# Only skin on the same side of the body is used, so nothing crosses the midline.
near = np.zeros(len(mus_P), int)
for side in (1, -1):
    m = mus_P[:, 0] * side >= 0
    skin_side = np.where(Pw[:, 0] * side >= -0.01)[0]
    near[m] = skin_side[nearest_index(mus_P[m], Pw[skin_side])]
Wm = Ww[near].copy()
# Arm muscles are those whose points mostly follow an arm bone; every other muscle belongs to the trunk.
mus_id = np.concatenate(mus_id)
arm_cols = [b for b, bn in enumerate(bone_names) if base_name(bn) in ARM]
is_arm_pt = Wm[:, arm_cols].sum(1) > 0.5
arm_muscle = np.bincount(mus_id, is_arm_pt) / np.bincount(mus_id) > 0.5
trunk_pt = ~arm_muscle[mus_id]
# On trunk muscles the arm only pulls within ~5 cm of the upper-arm bone (where the lats, pecs and deltoid
# attach); further out on the chest and back the pull fades to the upper spine.
for side in ('.L', '.R'):
    ua, fa = idx['upper_arm' + side], idx['forearm' + side]
    h, t = heads1[ua], heads1[fa]
    ab = t - h; u = np.clip(((mus_P - h) @ ab) / (ab @ ab), 0, 1)
    d = np.linalg.norm(mus_P - (h + u[:, None] * ab), axis=1)
    keep = np.where(trunk_pt, np.clip((0.05 - d) / 0.025, 0, 1), 1)
    moved = Wm[:, ua] * (1 - keep)
    Wm[:, ua] -= moved; Wm[:, idx['spine.003']] += moved
# Muscles on the shoulder blade ride with it (the 'shoulder' bone) instead of the arm, and the deltoid
# runs from the collarbone / shoulder blade (not the spine) to the arm.
SCAPULAR = re.compile(r'Serratus_anterior|Subscapularis|Infraspinatus|Supraspinatus|Teres_minor|Pectoralis_minor|Subclavius|Rhomboid|Levator_scapulae', re.I)
names_arr = np.array([n for n, _, _ in MUS])
for i, n in enumerate(names_arr):
    scap, delt = SCAPULAR.search(n), re.search('deltoid', n, re.I)
    if not (scap or delt): continue
    sfx = '.L' if n.rstrip(')')[-1] == 'l' else '.R'
    m = mus_id == i; sh = idx['shoulder' + sfx]
    move = [idx['upper_arm' + sfx]] if scap else [b for b, bn in enumerate(bone_names) if bn.startswith('spine')]
    Wm[np.ix_(m, [sh])] += Wm[np.ix_(m, move)].sum(1, keepdims=True)
    Wm[np.ix_(m, move)] = 0
# The glutes belong to the pelvis and only attach to the thigh bone at one end. Each glute point follows the
# thigh by how far along the muscle it sits, from its pelvis end (0) to its end on the femur (1): only the last
# ~30% near the femur follows the thigh (the gluteus maximus, mostly a pelvis muscle, only the last 20% and at
# most 70%), so swinging the leg out moves the attachment instead of swinging the belly in with the thigh.
GLUTE = re.compile(r'Gluteus|Tensor_fasciae_latae|Piriformis', re.I)
skN_arr = np.array(skN)
for i, n in enumerate(names_arr):
    if not GLUTE.search(n): continue
    sfx = '.L' if n.rstrip(')')[-1] == 'l' else '.R'
    m = mus_id == i; th = idx['thigh' + sfx]
    fem = skP[skN_arr == 'Femur' + ('l' if sfx == '.L' else 'r')]
    P_ = mus_P[m]
    ins = P_[np.argmin(np.linalg.norm(P_ - fem[nearest_index(P_, fem, margin=0.08)], axis=1))]   # the end on the femur
    org = P_[np.argmax(np.linalg.norm(P_ - ins, axis=1))]                                      # the far end, on the pelvis
    ab = ins - org; t_ = np.clip(((P_ - org) @ ab) / (ab @ ab), 0, 1)
    if re.search('maximus', n, re.I):
        follow = 0.7 * np.clip((t_ - 0.8) / 0.2, 0, 1) ** 2
    else:
        f_ = np.clip((t_ - 0.7) / 0.3, 0, 1); follow = f_ * f_ * (3 - 2 * f_)
    tot = Wm[m].sum(1)
    Wm[np.ix_(m, [th])] = 0
    Wm[m] *= ((tot * (1 - follow)) / np.maximum(Wm[m].sum(1), 1e-9))[:, None]
    Wm[m, idx['spine']] += np.where(Wm[m].sum(1) < 1e-9, tot * (1 - follow), 0)
    Wm[m, th] = tot * follow
# The pecs sit on the ribcage, which moves as one piece: their trunk share all goes to the ribcage joint,
# so twisting the chest turns them as a whole instead of shearing them across the middle.
lower = [idx['spine.001'], idx['spine.002']]
for i, n in enumerate(names_arr):
    if not re.search(r'pectoralis', n, re.I): continue
    m = mus_id == i
    Wm[np.ix_(m, [idx['spine.003']])] += Wm[np.ix_(m, lower)].sum(1, keepdims=True)
    Wm[np.ix_(m, lower)] = 0
top2 = np.argsort(-Wm, 1)[:, :2]
w2 = np.take_along_axis(Wm, top2, 1); w2 /= w2.sum(1, keepdims=True)
mus_B, mus_W = top2, w2

# Lats, teres major and pecs run from the trunk up to the upper arm. They act like fibres stretched between
# two anchors: the trunk end stays put and the arm end follows the insertion point on the humerus without
# swinging out with the arm's rotation. A helper bone per side carries that insertion point (the page moves
# it each frame); each muscle point follows it by its share along the fibre, so the muscle lengthens
# along its fibres (vertically for the lats) instead of fanning out sideways.
SPAN = re.compile(r'Latissimus_dorsi|Teres_major|pectoralis_major', re.I)
helpers = []
for side, sfx in ((1, '.L'), (-1, '.R')):
    ua, fa = idx['upper_arm' + sfx], idx['forearm' + sfx]
    ins = heads1[ua] + 0.18 * (heads1[fa] - heads1[ua])    # where these muscles attach, high on the humerus
    hb = NB + len(helpers)
    helpers.append({'name': 'arm_pull' + sfx, 'parent': idx['spine.003'], 'head': np.round(ins, 5).tolist(), 'upper_arm': ua})
    for i, (n, _, _) in enumerate(MUS):
        if not SPAN.search(n) or n.rstrip(')')[-1] != ('l' if side > 0 else 'r'): continue
        m = mus_id == i
        d = np.linalg.norm(mus_P[m] - ins, axis=1)
        t = np.clip(1 - d / 0.15, 0, 1) ** 1.5                    # 1 at the insertion, 0 from 15 cm away (the rest stays snug)
        trunk = Wm[m].copy(); trunk[:, [b for b, bn in enumerate(bone_names) if base_name(bn) in ARM or bn.startswith('shoulder')]] = 0
        tb = np.where(trunk.max(1) > 0, trunk.argmax(1), idx['spine.003'])
        mus_B[m] = np.c_[np.full(m.sum(), hb), tb]
        mus_W[m] = np.c_[t, 1 - t]
# The heel tendon gets the same kind of helper: it rides on the shin but sits wherever the tendon's attachment
# on the heel bone is, moving with it without turning, so the tendon slides with the heel as a straight cord.
for sfx in ('.L', '.R'):
    ten = np.isin(mus_id, [i for i, (n, _, _) in enumerate(MUS) if n == 'Calcaneal_tendon' + sfx[1].lower()])
    if not ten.any(): continue
    T_ = mus_P[ten]; ins = T_[T_[:, 1] < T_[:, 1].min() + 0.01].mean(0)
    helpers.append({'name': 'heel_pull' + sfx, 'parent': idx['shin' + sfx], 'head': np.round(ins, 5).tolist(), 'follow': idx['foot' + sfx]})

# Muscles lying under the lats that also reach the arm (teres major / minor) copy the lat's movement wherever the lat
# covers them (fully within 1.5 cm, fading out by 3 cm), so they stretch with it instead of bulging out.
UNDER_LAT = re.compile(r'Teres_major|Teres_minor', re.I)   # (the serratus rides the ribs and shoulder blade, not the arm)
NBH = NB + len(helpers)


def dense(B, W):
    D = np.zeros((len(B), NBH))
    for k in range(B.shape[1]): add_at(D, (np.arange(len(B)), B[:, k]), W[:, k])
    return D


for sl in 'lr':
    lat = np.isin(mus_id, [i for i, (n, _, _) in enumerate(MUS) if 'Latissimus_dorsi' in n and n.endswith(sl)])
    lat_ids = np.where(lat)[0]
    for i, (n, _, _) in enumerate(MUS):
        if not UNDER_LAT.search(n) or n.rstrip(')')[-1] != sl: continue
        m = np.where(mus_id == i)[0]
        j = lat_ids[nearest_index(mus_P[m], mus_P[lat_ids], margin=0.1)]
        f = np.clip((0.03 - np.linalg.norm(mus_P[m] - mus_P[j], axis=1)) / 0.015, 0, 1)[:, None]
        D = (1 - f) * dense(mus_B[m], mus_W[m]) + f * dense(mus_B[j], mus_W[j])
        b2 = np.argsort(-D, 1)[:, :2]; w2_ = np.take_along_axis(D, b2, 1); w2_ /= w2_.sum(1, keepdims=True)
        mus_B[m], mus_W[m] = b2, w2_
        print(f'{n}: {int((f > 0.5).sum())} of {len(m)} points follow the lat', file=sys.stderr)


# ---------------------------------------------------------------- skin built from the anatomy itself
# Everything inside the muscles and bones plus a ~6-8 mm skin layer becomes the body; its outline is the skin.
# Each finger is built on its own so neighbouring fingers are never joined by a web of skin.
from envelope import envelope, envelope_pieces


def bp3d(name):   # eyes and outer ears come from BodyParts3D (same man, same coordinates)
    files = [l.split('\t')[0] for l in open(f'{SRC}/isa_parts.tsv') if l.split('\t')[1] == name]
    V = np.vstack([[[float(t) for t in l.split()[1:4]] for l in open(f'{SRC}/objs-isa/isa_BP3D_4.0_obj_99/{fid}.obj') if l.startswith('v ')] for fid in files])
    return np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1) / 1000


def bp3d_mesh(name):
    """A BodyParts3D part as a mesh (points in our frame, triangles)."""
    files = [l.split('\t')[0] for l in open(f'{SRC}/isa_parts.tsv') if l.split('\t')[1] == name]
    Vs, Fs, n0 = [], [], 0
    for fid in files:
        lines = open(f'{SRC}/objs-isa/isa_BP3D_4.0_obj_99/{fid}.obj').read().split('\n')
        V = np.array([[float(t) for t in l.split()[1:4]] for l in lines if l.startswith('v ')])
        F = np.array([[int(t.split('/')[0]) - 1 for t in l.split()[1:4]] for l in lines if l.startswith('f ')])
        Vs.append(V); Fs.append(F + n0); n0 += len(V)
    V = np.vstack(Vs)
    return app(np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1) / 1000) + [0, dy, 0], np.vstack(Fs)


extra = [bp3d(n) for n in ['Left sclera', 'Right sclera', 'Left cornea', 'Right cornea', 'External ear']]
n_eyes = sum(len(e) for e in extra[:4]); n_ear = len(extra[4])
nose = re.compile(r'alar_cartilage|Nasal_septal_cartilage')
extra += [weld_mesh(P, I)[0] for n, P, I in read_meshes(f'{src}/zanatomy/SkeletalSystem100.meshes') if nose.search(n) and n[-1] not in 'ij']
extra = app(np.vstack(extra)); extra[:, 1] += dy

# ---- leaner muscles, same frame: every muscle point moves 15% of the way toward the bone under it
# (fading out for points more than 8 cm from any bone), so muscles get thinner while the skeleton stays put
n_mus0 = len(mus_P)
q = skP[nearest_index(mus_P, skP, margin=0.08)]
d = np.linalg.norm(mus_P - q, axis=1)
k = 0.15 * np.clip((0.08 - d) / 0.03, 0, 1) + 0.15 * (d <= 0.05)
k = np.minimum(k, 0.15)
# the shoulder muscles (deltoids) get an extra 15% on top
delt = np.isin(mus_id, [i for i, (n, _, _) in enumerate(MUS) if 'deltoid' in n.lower()])
k = np.where(delt, 1 - (1 - k) * 0.85, k)
# the rest of the arm muscles get an extra 15% too (smaller arms)
arm_m = np.isin(mus_id, [i for i, (n, _, _) in enumerate(MUS) if ARM_MUSCLE.search(n) and 'deltoid' not in n.lower()])
k = np.where(arm_m, 1 - (1 - k) * 0.85, k)
# the lats and teres major an extra 20%, so the back is smaller and less V-tapered
v_back = np.isin(mus_id, [i for i, (n, _, _) in enumerate(MUS) if re.search(r'Latissimus_dorsi|Teres_major', n, re.I)])
k = np.where(v_back, 1 - (1 - k) * 0.8, k)
# the chest (pecs) an extra 15%, so it is smaller
chest = np.isin(mus_id, [i for i, (n, _, _) in enumerate(MUS) if re.search(r'pectoralis', n, re.I)])
k = np.where(chest, 1 - (1 - k) * 0.85, k)
# the gluteus maximus is not slimmed but grown 10% away from the bone (a fuller, rounder butt)
glute = np.isin(mus_id, [i for i, (n, _, _) in enumerate(MUS) if re.search(r'Gluteus_maximus', n, re.I)])
k = np.where(glute, -0.10 * np.clip((0.12 - d) / 0.04, 0, 1), k)
print(f'gluteus maximus points grown: {int(glute.sum())}', file=sys.stderr)
mus_P = mus_P + (q - mus_P) * k[:, None]
print(f'muscles slimmed: mean move {np.mean(d * k) * 1000:.1f} mm ({int(delt.sum())} shoulder-muscle points extra)', file=sys.stderr)

# ---- wider standard stance: each leg swung out 6 degrees at the hip, baked into the rest pose
# (bones, muscles and joint positions), so the inner thighs no longer touch
LEG_OUT = np.radians(6)
LEG_SUB = {s_: [idx[n + s_] for n in ('thigh', 'shin', 'foot', 'heel.02', 'toe')] for s_ in ('.L', '.R')}


def leg_rot(P, sfx):
    h = heads1[idx['thigh' + sfx]]; a_ = LEG_OUT * (1 if sfx == '.L' else -1)
    c, s_ = np.cos(a_), np.sin(a_); d = np.asarray(P, float) - h
    return h + np.stack([d[..., 0] * c - d[..., 1] * s_, d[..., 0] * s_ + d[..., 1] * c, d[..., 2]], -1)


for sfx in ('.L', '.R'):
    sub = LEG_SUB[sfx]
    share = sum(np.where(np.isin(mus_B[:, k], sub), mus_W[:, k], 0) for k in range(mus_B.shape[1]))
    m = share > 0
    mus_P[m] += share[m, None] * (leg_rot(mus_P[m], sfx) - mus_P[m])
    m = np.isin(skB, sub); skP[m] = leg_rot(skP[m], sfx)
    for b_ in sub[1:]: heads1[b_] = leg_rot(heads1[b_], sfx)
    for b_ in sub: tails1[b_] = leg_rot(tails1[b_] + [0, dy, 0], sfx) - [0, dy, 0]
    for h in helpers:                       # (the heel tendon's helper moves with the foot it sits on)
        if h.get('follow') in sub: h['head'] = np.round(leg_rot(np.array(h['head']), sfx), 5).tolist()
print('legs swung out', round(np.degrees(LEG_OUT), 1), 'degrees each', file=sys.stderr)
anat_P = np.vstack([mus_P, skP, extra])
anat_B = np.vstack([mus_B, np.c_[skB, skB], np.full((len(extra), 2), idx['spine.005'])])
anat_W = np.vstack([mus_W, np.c_[np.ones(len(skB)), np.zeros(len(skB))], np.tile([1.0, 0.0], (len(extra), 1))])
lead = np.where(anat_W[:, 0] >= anat_W[:, 1], anat_B[:, 0], anat_B[:, 1])
# Hand: the palm is one piece, but every point past the knuckles (tendons and small muscles included)
# goes with the finger it is closest to, so neighbouring fingers are never glued together by skin.
def seg_dist(P, a, b):
    ab = b - a; u = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + u[:, None] * ab), axis=1)


group = np.full(len(anat_P), 'body', dtype=object)
for sfx in ('.L', '.R'):
    hand_bones = [idx[n] for n in bone_names if n.endswith(sfx) and (n.startswith(('f_', 'thumb', 'hand')))]
    m = np.isin(lead, hand_bones)
    H = lambda n: heads1[idx[n + sfx]]
    parts = {'palm' + sfx: [(H('hand'), H('f_middle.01')), (H('hand'), H('f_index.01')), (H('hand'), H('f_pinky.01')),
                            (H('thumb.01'), H('thumb.02'))],
             'thumb' + sfx: [(H('thumb.02'), H('thumb.03'))]}
    for f_ in ['f_index', 'f_middle', 'f_ring', 'f_pinky']:
        parts[f_ + sfx] = [(H(f_ + '.01'), H(f_ + '.02')), (H(f_ + '.02'), H(f_ + '.03'))]
    keys = list(parts)
    D = np.stack([np.min([seg_dist(anat_P[m], a, b) for a, b in parts[k]], 0) for k in keys], 1)
    group[m] = np.array(keys, dtype=object)[D.argmin(1)]
# the head (skull, face, eyes, nose, ears) is its own piece too, with a thicker layer
group[lead == idx['spine.005']] = 'head'
# each leg and each arm is its own piece too (joined to the trunk at hip and shoulder), so the inner
# thighs, or a hanging arm and the side of the chest, never fuse together
for sfx in ('.L', '.R'):
    legb = [idx[n + sfx] for n in ('thigh', 'shin', 'foot', 'heel.02', 'toe')]
    armb = [idx[n + sfx] for n in ('upper_arm', 'forearm')]
    group[(group == 'body') & np.isin(lead, legb)] = 'leg' + sfx
    group[(group == 'body') & np.isin(lead, armb)] = 'arm' + sfx
# below the hip joints, anything still in the trunk piece goes to the leg on its side (no web at the crotch)
hip_y = (heads1[idx['thigh.L']][1] + heads1[idx['thigh.R']][1]) / 2
low = (group == 'body') & (anat_P[:, 1] < hip_y)
group[low & (anat_P[:, 0] >= 0)] = 'leg.L'; group[low & (anat_P[:, 0] < 0)] = 'leg.R'
# The trunk stays snug under the arms: lat, teres and pec ends sticking out sideways past the ribcage
# (they run out to the humerus) are left out of the trunk's skin, so its filling and blurring can't
# stretch a web of skin from the side of the chest to the arm. The arm piece covers where they attach.
spine_b = [idx[n] for n in bone_names if n.startswith('spine') and n != 'spine.005']
ribs = skP[np.isin(skB, spine_b)]
sh_y = min(heads1[idx['upper_arm.L']][1], heads1[idx['upper_arm.R']][1])
band = np.round(anat_P[:, 1] / 0.02).astype(int)
rib_x = {b: np.abs(ribs[np.round(ribs[:, 1] / 0.02).astype(int) == b, 0]).max(initial=0) for b in np.unique(band)}
side_x = np.array([rib_x[b] for b in band])
off_side = (group == 'body') & (anat_P[:, 1] < sh_y) & (side_x > 0) & (np.abs(anat_P[:, 0]) > side_x + 0.02)
group[off_side] = 'drop'
print('trunk points past the side of the ribcage left out of the skin', int(off_side.sum()), file=sys.stderr)
group = group.astype(str)

# The hands themselves come from the modeled hands of the base mesh (already fitted to our knuckles):
# smooth, full fingers instead of skin shrunk over tendons. The anatomy's hand points are left out of
# the skin, and the base hand is built finger by finger so nothing webs; it joins the forearm at the wrist.
HAND_PREFIX = ('palm', 'f_', 'thumb')
# anatomy above the wrist joint (forearm tendons that run into the hand) stays part of the forearm
for sfx in ('.L', '.R'):
    w_, k_ = heads1[idx['hand' + sfx]], heads1[idx['f_middle.01' + sfx]]
    ax = (k_ - w_) / np.linalg.norm(k_ - w_)
    above = np.array([g.endswith(sfx) and g.startswith(HAND_PREFIX) for g in group]) & ((anat_P - w_) @ ax < 0.005)
    group[above] = 'body'
hand_anat = np.array([g.startswith(HAND_PREFIX) for g in group])


# ---------------------------------------------------------------- fingers as smooth tubes
# Each finger (and the thumb) is a tapered, slightly flattened tube along its own bones: one for the
# cartilage layer and a slightly thicker one for the skin. Every ring follows only that finger's joints
# (blending 50/50 right at each knuckle), so fingers bend independently and stay clean cylinders.
def tube(chain, radii, seg_bones, base_bone, n_hint, flat=0.85, segs=20, sub=8, tip_rings=5):
    V_, B_, W_ = [], [], []
    rings = []
    for k in range(len(chain) - 1):
        a_, b_ = chain[k], chain[k + 1]
        T = (b_ - a_) / np.linalg.norm(b_ - a_)
        S = np.cross(T, n_hint); S /= np.linalg.norm(S); F_ = np.cross(S, T)
        for j in range(sub + (1 if k == len(chain) - 2 else 0)):
            u = j / sub
            c = a_ + u * (b_ - a_); r = radii[k] + u * (radii[k + 1] - radii[k])
            prev = seg_bones[k - 1] if k > 0 else base_bone
            nxt = seg_bones[k + 1] if k + 1 < len(seg_bones) else None
            if u < 0.3 and prev is not None: bw = [(seg_bones[k], 1 - 0.5 * (1 - u / 0.3)), (prev, 0.5 * (1 - u / 0.3))]
            elif u > 0.7 and nxt is not None: bw = [(seg_bones[k], 1 - 0.5 * (u - 0.7) / 0.3), (nxt, 0.5 * (u - 0.7) / 0.3)]
            else: bw = [(seg_bones[k], 1.0), (seg_bones[k], 0.0)]
            rings.append((c, r, S, F_, T, bw))
    # rounded fingertip
    c, r, S, F_, T, bw = rings[-1]
    for q in range(1, tip_rings + 1):
        ang = q / (tip_rings + 1) * np.pi / 2
        rings.append((c + T * r * np.sin(ang), r * np.cos(ang), S, F_, T, bw))
    ang = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    for c, r, S, F_, T, bw in rings:
        for t in ang:
            V_.append(c + S * r * np.cos(t) + F_ * r * flat * np.sin(t))
            B_.append([bw[0][0], bw[1][0]]); W_.append([bw[0][1], bw[1][1]])
    nr = len(rings); I_ = []
    for k in range(nr - 1):
        for j in range(segs):
            a0, a1 = k * segs + j, k * segs + (j + 1) % segs
            I_ += [[a0, a0 + segs, a1], [a1, a0 + segs, a1 + segs]]
    # caps: fingertip point and closed base
    tipc = len(V_); V_.append(rings[-1][0] + rings[-1][4] * 1e-4); B_.append(B_[-1]); W_.append(W_[-1])
    basec = len(V_); V_.append(rings[0][0]); B_.append(B_[0]); W_.append(W_[0])
    for j in range(segs):
        I_ += [[(nr - 1) * segs + j, tipc, (nr - 1) * segs + (j + 1) % segs], [basec, j, (j + 1) % segs]]
    V_, I_ = np.array(V_), np.array(I_)
    if np.einsum('ij,ij->i', V_[I_[:, 0]] - V_.mean(0), np.cross(V_[I_[:, 1]] - V_[I_[:, 0]], V_[I_[:, 2]] - V_[I_[:, 0]])).sum() < 0:
        I_ = I_[:, ::-1]
    return V_, I_, np.array(B_), np.array(W_, float)


def seg_dist2(P, a_, b_):
    ab = b_ - a_; u = np.clip(((P - a_) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a_ + u[:, None] * ab), axis=1)


finger_tubes = {'cart': [], 'skin': []}
for sfx in ('.L', '.R'):
    Hd = lambda n: heads1[idx[n + sfx]]
    n_hint = np.cross(Hd('f_pinky.01') - Hd('f_index.01'), Hd('f_middle.01') - Hd('hand')); n_hint /= np.linalg.norm(n_hint)
    for f_ in ['f_index', 'f_middle', 'f_ring', 'f_pinky', 'thumb']:
        names_ = [f'{f_}.01', f'{f_}.02', f'{f_}.03']
        chain = [Hd(n) for n in names_] + [tails1[idx[names_[-1] + sfx]] + [0, dy, 0]]
        seg_b = [idx[n + sfx] for n in names_]
        pts = anat_P[group == f_ + sfx] if f_ != 'thumb' else anat_P[(group == 'thumb' + sfx)]
        r_seg = []
        for k in range(3):
            d = seg_dist2(pts, chain[k], chain[k + 1]) if len(pts) else np.array([0.008])
            near_k = d < 0.015
            r_seg.append(np.clip(np.percentile(d[near_k], 75) if near_k.sum() > 10 else 0.008, 0.0055, 0.0105))
        if f_ == 'thumb': r_seg[0] = max(r_seg[0], 0.011)
        r_pt = [r_seg[0], (r_seg[0] + r_seg[1]) / 2, (r_seg[1] + r_seg[2]) / 2, r_seg[2] * 0.85]
        base = None if f_ == 'thumb' else idx['hand' + sfx]
        if f_ == 'thumb': base = idx['hand' + sfx]
        finger_tubes['cart'].append(tube(chain, [r * 0.9 for r in r_pt], seg_b, base, n_hint))
        finger_tubes['skin'].append(tube(chain, [r * 0.95 + 0.0018 for r in r_pt], seg_b, base, n_hint))   # (fat; skin goes 1.8 mm out)
print('finger tubes', len(finger_tubes['cart']), file=sys.stderr)

# Toes the same way: each toe is a tube along its own toe bones (all on the foot's single toe joint).
def bone_axis(name):
    P = leg_rot(app(BONES[name][0]) + [0, dy, 0], '.L' if name.endswith('l') else '.R')   # toes: with the stance
    c = P.mean(0); ax = np.linalg.svd(P - c, full_matrices=False)[2][0]
    t = (P - c) @ ax
    return c + ax * np.percentile(t, 3), c + ax * np.percentile(t, 97)


toe_tubes = {'cart': [], 'skin': []}
TOE_NUM = ['first', 'second', 'third', 'fourth', 'fifth']
for sfx, sl in (('.L', 'l'), ('.R', 'r')):
    ankle = heads1[idx['foot' + sfx]]
    for k, num in enumerate(TOE_NUM):
        parts = [p for p in ('Proximal', 'Middle', 'Distal') if f'{p}_phalanx_of_{num}_finger_of_foot{sl}' in BONES]
        ends = []
        for p in parts:
            e0, e1 = bone_axis(f'{p}_phalanx_of_{num}_finger_of_foot{sl}')
            if np.linalg.norm(e0 - ankle) > np.linalg.norm(e1 - ankle): e0, e1 = e1, e0
            ends.append((e0, e1))
        chain = [ends[0][0]] + [e1 for _, e1 in ends]
        if len(chain) < 3: continue
        r0 = 0.0115 if k == 0 else 0.0082 - 0.0006 * k
        r_pt = list(np.linspace(r0, r0 * 0.8, len(chain)))
        seg_b = [idx['toe' + sfx]] * (len(chain) - 1)
        toe_tubes['cart'].append(tube(chain, [r * 0.85 for r in r_pt], seg_b, idx['foot' + sfx], np.array([0, 1.0, 0]), flat=0.8))
        toe_tubes['skin'].append(tube(chain, [r + 0.0005 for r in r_pt], seg_b, idx['foot' + sfx], np.array([0, 1.0, 0]), flat=0.8))
print('toe tubes', len(toe_tubes['skin']), file=sys.stderr)
for k in ('cart', 'skin'): finger_tubes[k] += toe_tubes[k]



# The hands themselves are the detailed rigged hand models of the WebXR Input Profiles project
# (W3C Software and Document License), fitted onto our skeleton's hand: first the whole hand is lined up
# on matching joints (position, size, which way the palm faces), then every finger bone is moved onto
# our knuckles. Each finger is built on its own so nothing webs.
def load_skinned_glb(path):
    blob = open(path, 'rb').read(); jl = struct.unpack('<I', blob[12:16])[0]
    J = json.loads(blob[20:20 + jl]); Bn = blob[20 + jl + 8:]

    def acc(k):
        a_ = J['accessors'][k]; bv = J['bufferViews'][a_['bufferView']]
        dt = {5126: np.float32, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32}[a_['componentType']]
        nc = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a_['type']]
        return np.frombuffer(Bn, dt, a_['count'] * nc, bv.get('byteOffset', 0) + a_.get('byteOffset', 0)).reshape(a_['count'], nc)
    par = {c: i for i, nd in enumerate(J['nodes']) for c in nd.get('children', [])}

    def wm(i):
        m = node_matrix(J['nodes'][i])
        while i in par: i = par[i]; m = node_matrix(J['nodes'][i]) @ m
        return m
    pr = J['meshes'][0]['primitives'][0]; sk = J['skins'][0]
    names_ = [J['nodes'][j]['name'] for j in sk['joints']]
    ibm = acc(sk['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
    P = acc(pr['attributes']['POSITION']).astype(float)
    J4_, W4_ = acc(pr['attributes']['JOINTS_0']).astype(int), acc(pr['attributes']['WEIGHTS_0']).astype(float)
    T = acc(pr['indices']).reshape(-1, 3).astype(int)
    W = np.zeros((len(P), len(names_)))
    for k in range(4): add_at(W, (np.arange(len(P)), J4_[:, k]), W4_[:, k])
    W /= W.sum(1, keepdims=True)
    rest = np.zeros_like(P)
    for b, j in enumerate(sk['joints']):
        M = wm(j) @ ibm[b]; rest += W[:, b:b + 1] * (P @ M[:3, :3].T + M[:3, 3])
    heads = np.array([wm(j)[:3, 3] for j in sk['joints']])
    return rest, T, W, names_, heads


def dense_samples(P, T, step=0.0015):
    """Points spread over every triangle at most `step` apart (so the outline has no holes)."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    n = np.clip(np.ceil(np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                                np.linalg.norm(a - c, axis=1)], 0) / step), 1, 30).astype(int)
    pts, src = [], []
    for k in np.unique(n):
        sel = np.where(n == k)[0]
        uv = np.array([(i / k, j / k) for i in range(k + 1) for j in range(k + 1 - i)])
        p = a[sel, None] + uv[None, :, 0:1] * (b - a)[sel, None] + uv[None, :, 1:2] * (c - a)[sel, None]
        pts.append(p.reshape(-1, 3)); src.append(np.repeat(sel, len(uv)))
    return np.vstack(pts), np.concatenate(src)


FINGER_MAP = {'index': 'f_index', 'middle': 'f_middle', 'ring': 'f_ring', 'pinky': 'f_pinky'}


def our_bone(jn):   # WebXR hand joint -> our rig bone (without side)
    if jn == 'wrist' or jn.endswith('-metacarpal') and not jn.startswith('thumb'): return 'hand'
    if jn.startswith('thumb'):
        return {'thumb-metacarpal': 'thumb.01', 'thumb-phalanx-proximal': 'thumb.02'}.get(jn, 'thumb.03')
    f_, part = jn.split('-finger-')
    return FINGER_MAP[f_] + {'phalanx-proximal': '.01', 'phalanx-intermediate': '.02'}.get(part, '.03')


def our_point(jn, sfx):   # where that joint sits on our skeleton (None for the metacarpal bases)
    if jn == 'wrist': return heads1[idx['hand' + sfx]]
    if jn.endswith('-tip'):
        b = idx[our_bone(jn) + sfx]; return tails1[b] + [0, dy, 0]
    if jn.endswith('-metacarpal') and not jn.startswith('thumb'): return None
    return heads1[idx[our_bone(jn) + sfx]]


def hand_child(jn, names_):
    order = ['metacarpal', 'phalanx-proximal', 'phalanx-intermediate', 'phalanx-distal', 'tip']
    if jn == 'wrist': return None
    stem = jn[:jn.index('-metacarpal')] if jn.endswith('-metacarpal') else jn[:jn.index('-phalanx')] if '-phalanx' in jn else None
    if stem is None: return None   # tips
    cur = jn[len(stem) + 1:]
    for nxt in order[order.index(cur) + 1:]:
        if f'{stem}-{nxt}' in names_: return f'{stem}-{nxt}'
    return None


BHv, bh_samples, bh_sgroup, bh_top, bh_w = [], [], [], [], []
for sfx, fname in (('.L', 'left.glb'), ('.R', 'right.glb')):
    HP, HT, HW, hn, hh = load_skinned_glb(f'{os.path.dirname(glb_path)}/webxr-hand/{fname}')
    # 1. line the whole hand up with ours (similarity transform on the joints we can match)
    pairs = [(i, our_point(n_, sfx)) for i, n_ in enumerate(hn) if our_point(n_, sfx) is not None]
    A_ = np.array([p for _, p in pairs]); B_ = hh[[i for i, _ in pairs]]
    mA, mB = A_.mean(0), B_.mean(0); a_, b_ = A_ - mA, B_ - mB
    U, S_, Vt = np.linalg.svd(a_.T @ b_ / len(A_)); Dm = np.eye(3); Dm[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rg = U @ Dm @ Vt; sg = (S_ * np.diag(Dm)).sum() / (b_ ** 2).sum(1).mean(); tg = mA - sg * Rg @ mB
    HP = HP @ (sg * Rg).T + tg; hh = hh @ (sg * Rg).T + tg
    # 2. move every finger bone onto our knuckles (stretching along the bone, keeping its thickness)
    xf = {}
    for i, n_ in enumerate(hn):
        c_ = hand_child(n_, hn)
        if c_ is None: continue
        ci = hn.index(c_)
        h0, t0 = hh[i], hh[ci]
        h1 = our_point(n_, sfx); h1 = h0 if h1 is None else h1
        t1 = our_point(c_, sfx)
        a0 = (t0 - h0) / np.linalg.norm(t0 - h0)
        xf[i] = (h0, a0, h1, rotation_between(t0 - h0, t1 - h1), np.linalg.norm(t1 - h1) / np.linalg.norm(t0 - h0))
    fitted = np.zeros_like(HP)
    for i, n_ in enumerate(hn):
        m = HW[:, i] > 1e-5
        if not m.any(): continue
        key = i if i in xf else (hn.index(n_.replace('-tip', '-phalanx-distal')) if n_.endswith('-tip') and n_ != 'thumb-tip' else
                                 hn.index('thumb-phalanx-distal') if n_ == 'thumb-tip' else None)
        if key is None or key not in xf:
            fitted[m] += HW[m, i:i + 1] * HP[m]; continue
        h0, a0, h1, R, s_ = xf[key]
        q = HP[m] - h0; ax = np.outer(q @ a0, a0)
        fitted[m] += HW[m, i:i + 1] * (h1 + (ax * s_ + (q - ax)) @ R.T)
    # groups (palm / each finger) and our joint weights for every hand vertex
    dom = [hn[j] for j in HW.argmax(1)]
    grp = np.array([('palm' if our_bone(d) in ('hand', 'thumb.01') else our_bone(d).split('.')[0]) + sfx for d in dom])
    # nothing of the palm reaches past the knuckle line: beyond it the skin goes with the nearest finger,
    # so there is only a small natural web at the base of each finger
    w0, k0 = our_point('wrist', sfx), our_point('middle-finger-phalanx-proximal', sfx)
    ax_ = (k0 - w0) / np.linalg.norm(k0 - w0)
    past = np.array([g.startswith('palm') for g in grp]) & ((fitted - w0) @ ax_ > (k0 - w0) @ ax_ - 0.006)
    if past.any():
        keys = [f_ + sfx for f_ in FINGER_MAP.values()]
        D = np.stack([seg_dist(fitted[past], our_point(f'{x}-finger-phalanx-proximal', sfx), our_point(f'{x}-finger-tip', sfx))
                      for x in FINGER_MAP], 1)
        grp[past] = np.array(keys)[D.argmin(1)]
    pts, src = dense_samples(fitted, HT)
    BHv.append(fitted); bh_samples.append(pts); bh_sgroup.append(grp[HT[src, 0]])
    Wour = np.zeros((len(fitted), NB))
    for j, n_ in enumerate(hn): Wour[:, idx[our_bone(n_) + sfx]] += HW[:, j]
    top = np.argsort(-Wour, 1)[:, :2]; w_ = np.take_along_axis(Wour, top, 1); w_ /= w_.sum(1, keepdims=True)
    bh_top.append(top); bh_w.append(w_)
    print(f'hand {sfx}: fitted {len(fitted)} vertices, scale {sg:.2f}', file=sys.stderr)
BHv, bh_samples, bh_sgroup = np.vstack(BHv), np.vstack(bh_samples), np.concatenate(bh_sgroup)
bh_top, bh_w = np.vstack(bh_top), np.vstack(bh_w)
F_all = np.vstack([mus_F, skI + len(mus_P)])
samples = [anat_P, *[(anat_P[F_all[:, 0]] + anat_P[F_all[:, k]]) / 2 for k in (1, 2)], anat_P[F_all].mean(1)]
sample_group = np.concatenate([group, group[F_all[:, 0]], group[F_all[:, 0]], group[F_all[:, 0]]])
sample_P = np.vstack(samples)
anat_samples, anat_sgroup = sample_P.copy(), sample_group.copy()        # (used again for the cartilage layer)
keep_s = ~np.array([g.startswith(HAND_PREFIX) for g in sample_group])     # drop the anatomy's own hand
toe_lead = np.isin(lead, [idx['toe.L'], idx['toe.R']])
toe_sample = np.concatenate([toe_lead, toe_lead[F_all[:, 0]], toe_lead[F_all[:, 0]], toe_lead[F_all[:, 0]]])
keep_s = ~np.array([g.startswith(('f_', 'thumb')) for g in sample_group]) & ~toe_sample   # fingers, toes: tubes
keep_s &= sample_group != 'drop'
# the first knuckle-length of every finger tube joins the palm, so palm and fingers blend at the knuckles
knuck = [V_[:2 * 20] for (V_, _, _, _) in finger_tubes['skin']]   # just the knuckle end of each finger
knuck_g = [('palm.L' if k < 5 else 'palm.R') if k < 10 else ('leg.L' if k < 15 else 'leg.R') for k in range(len(knuck))]
def lm(name):
    return app(BONES[name][0]) + [0, dy, 0]


# Crotch fill: the anatomy has no external genitalia, and below the hip joints the trunk is handed to the
# legs, which leaves a groove down the front midline. A low rounded solid in the trunk piece, from the top
# of the pubic bone down to the crotch and only in front of the hip joints, bridges the two legs (both blend
# softly into the trunk), like a mannequin. It only covers the gap: it is kept behind the surface level so
# it never bulges. The back is left alone.
pubis = np.vstack([lm('Hip_bonel'), lm('Hip_boner')])
sym = pubis[(np.abs(pubis[:, 0]) < 0.015) & (pubis[:, 2] > heads1[idx['thigh.L']][2])]   # the pubic symphysis
m_top, m_bot = sym[:, 1].max(), sym[:, 1].min() - 0.015         # (stops short of the crotch, so the legs never web and
# nothing bulges down between them)
m_c = np.array([0.0, (m_top + m_bot) / 2, np.percentile(sym[:, 2], 90) - 0.02])
m_r = np.array([0.045, (m_top - m_bot) / 2, 0.02])      # ~10 cm wide once padded; kept back so it only bridges
g_ = np.mgrid[-1:1:21j, -1:1:21j, -1:1:21j].reshape(3, -1).T
g_ = g_[(g_ ** 2).sum(1) <= 1]
mound = m_c + g_ * m_r
# Buttock cleft fill: below the hip joints each buttock belongs to its leg's piece, and the two are never blended,
# which left a slot ~9-11 cm deep between them (its walls 1.5-2 cm apart, reaching almost to the hip joints) that
# opened right up when the legs spread. A thin fill down the midline behind the hips (2 cm wide), in the trunk piece
# (blended softly with both legs), bridges the cheeks: it reaches back to ~1.2 cm short of the buttock muscles at each
# height, from just above the hip joints down to 7 cm below them (lower, it hung between the thighs as a tab, then
# rounded into a bulb under the cleft), centred on the midline, so only a shallow crease is left (the line between the
# cheeks is drawn in later).
hz_ = heads1[idx['thigh.L']][2]
glute_ = anat_P[(np.abs(anat_P[:, 0]) > 0.03) & (np.abs(anat_P[:, 0]) < 0.09) & (anat_P[:, 2] < hz_)]
cleft_fill = []
for yb_ in np.arange(hip_y - 0.07, hip_y + 0.04, 0.003):
    band_ = glute_[np.abs(glute_[:, 1] - yb_) < 0.005]
    if not len(band_): continue
    # (1.2 cm in front of the back of the buttock muscles, sinking to 3 cm toward its lower end, so it stays inside the
    # cheeks where they curve under instead of rounding out between them as a bulb)
    z_back = band_[:, 2].min() + 0.012 + 0.018 * np.clip(((hip_y - 0.02) - yb_) / 0.05, 0, 1)
    for z_ in np.arange(z_back, hz_ - 0.02, 0.003):
        for x_ in np.linspace(-0.009, 0.009, 7): cleft_fill.append((x_, yb_, z_))
cleft_fill = np.array(cleft_fill)
print(f'buttock cleft fill: {len(cleft_fill)} points', file=sys.stderr)
# Pelvis fill: the anatomy has no organs, so the bowl of the pelvis was empty, and with the legs never blended with each
# other the skin ran up between them as a slit (its walls touching) from the crotch 16 cm below the hip joints into a
# hollow ~10 cm wide inside the pelvis: ~1900 hidden skin points that folded whenever a leg moved. Filled: everything
# inside the pelvic bones (their convex hull, kept 1 cm inside it so it never reaches the surface), and a thin sheet down
# the midline between the legs (2 cm wide like the cleft fill), from the hip joints down to 9 cm below them (about where
# a real crotch is), between the cleft fill behind and the crotch mound in front.
from scipy.spatial import ConvexHull
pel_ = np.vstack([lm(n) for n in ('Hip_bonel', 'Hip_boner', 'Sacrum', 'Coccyx')])
hull_ = ConvexHull(pel_)
g_ = np.mgrid[pel_[:, 0].min():pel_[:, 0].max():0.005, pel_[:, 1].min():pel_[:, 1].max():0.005,
              pel_[:, 2].min():pel_[:, 2].max():0.005].reshape(3, -1).T
pelvis_fill = g_[(g_ @ hull_.equations[:, :3].T + hull_.equations[:, 3]).max(1) < -0.01]
peri_fill = np.array([(x_, y_, z_) for y_ in np.arange(hip_y - 0.09, hip_y, 0.003) for z_ in np.arange(hz_ - 0.03, hz_ + 0.02, 0.003)
                      for x_ in np.linspace(-0.009, 0.009, 7)])
pelvis_fill = np.vstack([pelvis_fill, peri_fill])
print(f'pelvis fill: {len(pelvis_fill)} points', file=sys.stderr)
mound = mound[mound[:, 2] > heads1[idx['thigh.L']][2]]
print(f'crotch mound: {len(mound)} points, {m_top - m_bot:.3f} m tall', file=sys.stderr)
sample_P = np.vstack([sample_P[keep_s]] + knuck)
sample_group = np.concatenate([sample_group[keep_s]] + [np.full(len(v), g) for v, g in zip(knuck, knuck_g)])
gnames = np.unique(sample_group)
groups = [sample_P[sample_group == gname] for gname in gnames]
# the model hand is already a skin surface: only a thin 3 mm layer; the head gets ~9 mm (a fuller face),
# the body ~6 mm
# This surface is the adipose (fat) layer: ~9 mm over the body and face, thin on the palms. The heavy
# smoothing below lets it fill the grooves between muscles; the skin then sits 1.8 mm outside it.
# Thickness varies: 'close' fills hollows up to about 2 x close voxels wide (face: cheeks, temples, around
# the eyes and jaw; hands: between tendons and bones), so the fat comes out level and blends between parts.
# plump: ~12 mm with strong closing on the trunk (waist, butt), arms and legs
# The fat settles into the hollows between muscles and hides their shapes: a wide closing (fills grooves
# and steps up to ~5 cm across) and heavy smoothing make it soft and plump rather than defined.
# This surface is the anatomy grown by one 3 mm voxel with its grooves closed: the muscle surface as a whole, with
# the grooves between muscles bridged (about 2 x close voxels across: ~2.4 cm on the arms, ~3 cm on the legs and trunk,
# matching a fat layer of ~9-15 mm, which fills grooves a few times its own thickness). The fat is then this surface
# moved out by each region's fat thickness (below), so it is that thick over the muscle bellies and thicker in the
# grooves, and the muscle shapes show softly through it. (It was grow 3 / close 11 everywhere and the thinning capped
# at 3 mm: the forearm came out a smooth tube ~13 mm off the muscle.)
ENV_GROW = {'arm': 1, 'leg': 1, 'trunk': 1, 'head': 3}
# (trunk 6 voxels: grooves up to ~3.5 cm wide are bridged, like the fat of a 17% build filling in between the abs and
# along the obliques; it was 5, an athletic ~15%, which left the abs and the hip ridges standing out)
ENV_CLOSE = {'arm': 4, 'leg': 5, 'trunk': 6, 'head': 8}
region_of = lambda g: 'arm' if g.startswith(('arm', 'palm')) else 'leg' if g.startswith('leg') else 'head' if g == 'head' else 'trunk'
radii = [ENV_GROW[region_of(g)] for g in gnames]
closes = [ENV_CLOSE[region_of(g)] for g in gnames]


def piece_of(g):
    for p_ in ('leg.L', 'leg.R'):
        if g == p_: return p_
    for sfx in ('.L', '.R'):
        if g in ('arm' + sfx, 'palm' + sfx): return 'arm' + sfx
    return 'head' if g == 'head' else 'trunk'


PIECES = ['trunk', 'head', 'arm.L', 'arm.R', 'leg.L', 'leg.R']
# How strongly each piece's volume is blurred before its surface is taken: strong on trunk and legs (rounds
# the steps between muscles, e.g. the ab segments, into one slope), moderate on the arms (keeps the palm),
# light on the head (keeps the face).
FAT_BLUR = {'trunk': 44, 'leg.L': 38, 'leg.R': 38, 'arm.L': 15, 'arm.R': 15, 'head': 7}   # (trunk: 17% build)
SKIN_BLUR = {'trunk': 24, 'leg.L': 24, 'leg.R': 24, 'arm.L': 10, 'arm.R': 10, 'head': 6}


def mesh_taubin(V, T, passes, lam=0.5, mu=-0.53, w=1.0):
    """Smooth a surface without shrinking it (like skin and fat evening out the muscle texture).
    w (optional, per point) sets how much each point is smoothed."""
    i = np.concatenate([T[:, 0], T[:, 1], T[:, 2], T[:, 1], T[:, 2], T[:, 0]])
    j = np.concatenate([T[:, 1], T[:, 2], T[:, 0], T[:, 0], T[:, 1], T[:, 2]])
    d = np.maximum(np.bincount(i, minlength=len(V)), 1)[:, None]
    w = np.reshape(w, (-1, 1)) if np.ndim(w) else w
    for _ in range(passes):
        for k in (lam, mu):
            acc = fastgeo.nsum(i, j, V)
            V = V + k * w * (acc / d - V)
    return V


def finish_surface(V_, T_, smooth):
    if os.environ.get('SURF_DUMP'):     # (for testing the clean-up: the raw surface, then stop)
        pickle.dump((V_, T_, smooth), open(os.environ['SURF_DUMP'], 'wb')); sys.exit(0)
    if np.einsum('ij,ij->i', V_[T_[:, 0]], np.cross(V_[T_[:, 1]], V_[T_[:, 2]])).sum() < 0: T_ = T_[:, ::-1]
    V_ = mesh_taubin(V_, T_, smooth)
    V_, T_ = clean_decimate(V_, T_, 165000)              # ~5 mm spacing is plenty
    return mesh_taubin(V_, T_, 4), T_


def clean_decimate(V, T, faces):
    """Edge-collapse decimation (libigl) down to `faces` triangles, keeping the surface one clean sheet. (Merging
    all points within 5 mm of each other, as before, joined surfaces that lie closer than that, like the two walls
    of the buttock cleft: it left 1200 doubled triangles, 2400 edges shared by three or more and 5500 creases folded
    flat over the body, which folded further as soon as it moved.) The surface nets output can have a spot where two
    sheets touch along an edge; the triangles within 1 cm of it are cut out first (edge collapse needs a clean sheet)
    and the hole is closed with a fan afterwards."""
    import igl
    T = T.astype(np.int32)
    D = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    u, inv, cnt = np.unique(D, axis=0, return_inverse=True, return_counts=True)
    bad_v = np.union1d(np.unique(u[cnt > 2]), np.where(~np.asarray(igl.is_vertex_manifold(np.asfortranarray(T))).ravel().astype(bool))[0])
    if len(bad_v):
        dmin = np.min([np.linalg.norm(V[T].mean(1) - V[v], axis=1) for v in bad_v], 0)
        T = T[dmin > 0.01]
    U, G = igl.decimate(np.asfortranarray(V), np.asfortranarray(T), faces)[:2]
    assert len(G), 'decimation failed'
    u_ = np.unique(G); r = np.full(len(U), -1); r[u_] = np.arange(len(u_)); U, G = U[u_], r[G].astype(np.int64)
    # close the holes: a fan from each boundary loop's centre
    D = np.concatenate([G[:, [0, 1]], G[:, [1, 2]], G[:, [2, 0]]])
    _, inv, cnt = np.unique(np.sort(D, 1), axis=0, return_inverse=True, return_counts=True)
    nxt = {a: b for a, b in D[cnt[inv.ravel()] == 1]}         # the open edges, as their triangles run them
    while nxt:
        a = next(iter(nxt)); loop = [a]
        while nxt[loop[-1]] != a: loop.append(nxt.pop(loop[-1]))
        nxt.pop(loop[-1]); loop = np.array(loop)
        c = len(U); U = np.vstack([U, U[loop].mean(0)])
        G = np.vstack([G, np.c_[np.roll(loop, -1), loop, np.full(len(loop), c)]])   # (running those edges the other way)
    print(f'surface decimated: {len(G)} triangles ({len(bad_v)} points where sheets touched cut out and closed)', file=sys.stderr)
    return U, G


# All pieces go into one volume, blended softly where they join (trunk with legs, arms and head), and one
# continuous surface comes out; the legs are never blended with each other.
fat_pieces, lab_pts, lab_name = [], [], []
for p_ in PIECES:
    sel = [i for i, g in enumerate(gnames) if piece_of(g) == p_]
    if not sel: continue
    piece_ = (p_, [groups[i] for i in sel], [radii[i] for i in sel], [closes[i] for i in sel], FAT_BLUR[p_])
    if p_ == 'trunk':               # (plus the crotch mound)
        piece_ = (p_, piece_[1] + [mound, cleft_fill, pelvis_fill], piece_[2] + [3, 1, 1], piece_[3] + [11, 3, 3], piece_[4])
        lab_pts.append(mound[::4]); lab_name += [p_] * len(mound[::4])
    if p_.startswith('arm'):        # arms blend softly into the trunk only around the shoulder
        piece_ += ((heads1[idx['upper_arm' + p_[3:]]].copy(), 0.06),)
    fat_pieces.append(piece_)
    for i in sel: lab_pts.append(groups[i][::4]); lab_name += [p_] * len(groups[i][::4])
SV, ST = finish_surface(*envelope_pieces(fat_pieces), 70)
STAGES = {}           # (for measuring: the surface after each step, saved when STAGE_DUMP is set)
if os.environ.get('STAGE_DUMP'):
    STAGES['arm_piece'] = [p_ for p_ in fat_pieces if p_[0] == 'arm.L'][0]
lab_pts, lab_name = np.vstack(lab_pts), np.array(lab_name)
FSET = lab_name[nearest_index(SV, lab_pts, margin=0.05)]    # which piece each point belongs to (for weights)

# ---------------------------------------------------------------- hands and feet: the MakeHuman base mesh's
# (CC0; see tools/mh_parts.py and assets/makehuman/SOURCE.md). Each is lined up on our joints and nudged onto our
# anatomy (the old surface as a rough guide for thickness), then the body surface is cut just before the joint
# (hands: 5 mm before the wrist; feet: 5.5 cm above the ankle, so the whole ankle and heel are MakeHuman's), the
# part 1 cm further on, and the gap closed with a strip of triangles, so the skin stays one surface. The body's
# later smoothing would wash out knuckles, nails and toes, so the part's points are put back to their fitted
# shape afterwards, blending in over the first 1.5 cm past the cut. Fitted parts are cached.
import hashlib, pickle
import mh_parts
MH_DIR = f'{os.path.dirname(glb_path)}/makehuman'
mh_mesh = mh_parts.load_mh(MH_DIR)
tips_now = tails1 + [0, dy, 0]


def toe_landmarks_ours(sfx, sl):
    """Our foot landmarks in the order of mh_parts.foot_landmarks_mh: ankle, knee, heel back, big toe base / middle
    joint / tip, then each other toe's base and tip (from the real toe bones, as the toe tubes use them)."""
    ankle = heads1[idx['foot' + sfx]]
    P = [ankle, heads1[idx['shin' + sfx]], mh_parts.heel_back(SV, ankle, sl)]
    for k, num in enumerate(TOE_NUM):
        ends = []
        for p_ in ('Proximal', 'Middle', 'Distal'):
            if f'{p_}_phalanx_of_{num}_finger_of_foot{sl}' not in BONES: continue
            e0, e1 = bone_axis(f'{p_}_phalanx_of_{num}_finger_of_foot{sl}')
            if np.linalg.norm(e0 - ankle) > np.linalg.norm(e1 - ankle): e0, e1 = e1, e0
            ends.append((e0, e1))
        P += [ends[0][0], ends[0][1], ends[-1][1]] if k == 0 else [ends[0][0], ends[-1][1]]
    return np.array(P)


PARTS = []    # (kind, side, joint, joint above, body cut and part cut along the limb, our bones, piece, bone map)
for sfx, sl in (('.L', 'l'), ('.R', 'r')):
    PARTS.append(('hand', sfx, sl, 'hand', 'forearm', -0.005, 0.005, ('hand', 'f_', 'thumb'), 'arm', mh_parts.mh_bone_to_ours))
for sfx, sl in (('.L', 'l'), ('.R', 'r')):
    PARTS.append(('foot', sfx, sl, 'foot', 'shin', -0.055, -0.045, ('foot', 'toe', 'heel'), 'leg', mh_parts.mh_leg_bone_to_ours))
mh_idx, mh_skin, mh_blend, mh_W, mh_fset, mh_digit, mh_kind = [], [], [], [], [], [], []
new_P, new_T, cut_loops = [SV], [], []
keep_body = np.ones(len(SV), bool)
for kind, sfx, sl, jn, pn, cut_b, cut_p, bone_pre, piece, bone_map in PARTS:
    joint, prox = heads1[idx[jn + sfx]], heads1[idx[pn + sfx]]
    u = (joint - prox) / np.linalg.norm(joint - prox)
    near_ = (np.linalg.norm(SV - joint, axis=1) < 0.25) & ((SV[:, 0] > 0) if sfx == '.L' else (SV[:, 0] < 0))
    s_body = (SV - joint) @ u
    part_b = [idx[n] for n in bone_names if n.endswith(sfx) and n.startswith(bone_pre)]
    bones_P = skP[np.isin(skB, part_b)].astype(float)
    tubes_ = finger_tubes['skin'][(0 if sfx == '.L' else 5):(5 if sfx == '.L' else 10)] if kind == 'hand' else \
        finger_tubes['skin'][(10 if sfx == '.L' else 15):(15 if sfx == '.L' else 20)]
    guide = np.vstack([SV[near_ & (s_body > cut_p)]] + [V_ + normals(V_, I_) * 0.002 for V_, I_, _, _ in tubes_])
    if kind == 'hand':
        A_mh = mh_parts.hand_landmarks_mh(mh_mesh[2], sl); ours_ = mh_parts.hand_landmarks_ours(heads1, tips_now, idx, sfx)
        kw = {}
    else:
        A_mh = mh_parts.foot_landmarks_mh(mh_mesh, sl); ours_ = toe_landmarks_ours(sfx, sl)
        kw = dict(start=-0.10, fade=cut_p - 0.01, floor=SV[:, 1].min())      # (both soles on the same floor)
    key = hashlib.sha1(b''.join([np.round(a, 5).tobytes() for a in (ours_, bones_P, guide)]) +
                       open(mh_parts.__file__, 'rb').read() + open(f'{MH_DIR}/base.obj', 'rb').read()).hexdigest()[:16]
    cache = f'{MH_DIR}/cache/{kind}{sfx}-{key}.pkl'
    was_cached = os.path.exists(cache)
    if was_cached:
        P_, T_, Wp, s_ = pickle.load(open(cache, 'rb'))
    else:
        P_, T_, Wp, s_ = mh_parts.fit_part(mh_mesh, sl, A_mh, ours_, joint, prox, bones_P, guide, **kw)
        os.makedirs(f'{MH_DIR}/cache', exist_ok=True); pickle.dump((P_, T_, Wp, s_), open(cache, 'wb'))
    keep_body &= ~(near_ & (s_body > cut_b))
    # the part past its cut
    kt = (s_[T_] > cut_p).all(1)
    used = np.unique(T_[kt]); rm = np.full(len(P_), -1); rm[used] = np.arange(len(used))
    base_ = sum(len(x) for x in new_P)
    new_P.append(P_[used]); new_T.append(rm[T_[kt]] + base_)
    Wd_ = np.zeros((len(used), NB + len(helpers)), np.float32)
    for b_, iw in Wp.items():
        ob_ = bone_map(b_)
        if ob_ is None: continue
        for i_, w_ in iw:
            if rm[i_] >= 0: Wd_[rm[i_], idx[ob_ + sfx]] += w_
    Wd_[Wd_.sum(1) < 1e-6, idx[jn + sfx]] = 1
    mh_idx.append(np.arange(len(used)) + base_); mh_skin.append(P_[used]); mh_W.append(Wd_ / Wd_.sum(1, keepdims=True))
    mh_fset.append(np.full(len(used), piece + sfx)); mh_kind.append(np.full(len(used), kind + sfx))
    # which toe each foot point belongs to (1 = big toe ... 5; 0 = not a toe), from MakeHuman's own toe bones,
    # so the skin map can give every toe its own piece
    dig = np.zeros(len(used), np.uint8); best = np.zeros(len(used))
    for b_, iw in Wp.items():
        for i_, w_ in iw:
            if rm[i_] >= 0 and w_ > best[rm[i_]]:
                best[rm[i_]] = w_; dig[rm[i_]] = int(b_[3]) if kind == 'foot' and b_.startswith('toe') else 0
    mh_digit.append(dig)
    b_ = np.clip((s_[used] - cut_p) / 0.03, 0, 1); mh_blend.append(b_ * b_ * (3 - 2 * b_))     # (over 3 cm)
    cut_loops.append((kind + sfx, joint + u * (cut_b + cut_p) / 2, u, rm[mh_parts.boundary_loop(T_[kt], len(P_))[0]] + base_))
    print(f'{kind} {sfx}: MakeHuman {kind} fitted, {len(used)} points ({"cached" if was_cached else "new"})', file=sys.stderr)
# ---------------------------------------------------------------- head: the MakeHuman base mesh's head
# Fitted onto our skull plus a realistic soft tissue depth (skin, fat and muscle over the bone) at the standard
# craniofacial landmarks, pooled from the published data (tools/tissue_depth.py: adult men, C-Table / T-Tables,
# Stephan & Simpson 2008, Stephan 2014, 2017, Hona & Stephan 2024). The landmarks are found on our skull from its
# shape; each depth is laid out along the bone's outward direction there and spread smoothly over the skull, and
# 'skull + depth' is the surface the face is fitted to. The MakeHuman face keeps its own eyelids, lips, nose and
# ears on top of that: its skin landmarks (found from its shape) are first warped onto ours, then a smooth
# displacement settles it onto skull + depth, leaving the nose, ears and eyes as MakeHuman shaped them; the eyelids
# are settled onto our eyeballs (BodyParts3D), the nose tip lined up with our nasal cartilage, the ears with ours.
if os.environ.get('HEAD_DUMP'):     # (for working on the head on its own: everything it needs, saved once)
    pickle.dump(dict(skP=skP, skN_arr=skN_arr, skI=skI, extra=extra, n_eyes=n_eyes, n_ear=n_ear, heads1=heads1, idx=idx,
                     SV=SV, dy=dy), open(os.environ['HEAD_DUMP'], 'wb'))
import tissue_depth
TD = tissue_depth.depths(f'{os.path.dirname(glb_path)}/tissue-depth/C-Table_v2018_0.csv')
print('soft tissue depths (mm): ' + ', '.join(f'{k} {v[0] * 1000:.1f}' for k, v in TD.items()), file=sys.stderr)
sk_named = lambda pat: skP[np.array([bool(re.match(pat, n_)) for n_ in skN_arr])].astype(float)
SKULL_RE = r'(Frontal_bone|Parietal_bone[lr]|Occipital_bone|Temporal_bone[lr]|Nasal_bone[lr]|Maxilla[lr]|Mandible|Zygomatic_bone[lr]|Sphenoid_bone)$'
sk_m = np.array([bool(re.match(SKULL_RE, n_)) for n_ in skN_arr])
sk_norm = normals(skP.astype(float), skI)
eyes_c = {s_: (app(bp3d(f'{nm} sclera')) + [0, dy, 0]).mean(0) for s_, nm in (('L', 'Left'), ('R', 'Right'))}
ye, ze = (eyes_c['L'][1] + eyes_c['R'][1]) / 2, (eyes_c['L'][2] + eyes_c['R'][2]) / 2
head_c = np.array([0.0, ye - 0.01, ze - 0.07])          # (roughly the middle of the head: 'outward' is away from it)
FR, NA = sk_named(r'Frontal_bone$'), sk_named(r'Nasal_bone[lr]$')
MXa, MD = sk_named(r'Maxilla[lr]$'), sk_named(r'Mandible$')
VAULT = sk_named(r'(Frontal_bone|Parietal_bone[lr]|Occipital_bone)$')
mid = lambda P, w=0.004: P[np.abs(P[:, 0]) < w]
pick = lambda P, f: P[np.argmax(f(P))]
inb = lambda P, k, lo, hi: P[(P[:, k] > lo) & (P[:, k] < hi)]
SL = {}       # landmark code -> skull point
SL['g'] = pick(inb(mid(FR), 1, ye + 0.005, ye + 0.04), lambda P: P[:, 2])
SL['n'] = pick(mid(NA, 0.006), lambda P: P[:, 1])
SL['rhi'] = pick(mid(NA, 0.006), lambda P: -P[:, 1])
mn_ = mid(NA, 0.006); SL['mn'] = pick(mn_[np.abs(mn_[:, 1] - (SL['n'][1] + SL['rhi'][1]) / 2) < 0.004], lambda P: P[:, 2])
SL['sn'] = pick(inb(mid(MXa), 1, SL['rhi'][1] - 0.035, SL['rhi'][1] - 0.01), lambda P: P[:, 2])     # (anterior nasal spine)
pr_ = mid(MXa); pr_ = pr_[pr_[:, 2] > SL['sn'][2] - 0.02]; SL['ls'] = pick(pr_, lambda P: -P[:, 1])  # (prosthion)
SL['mp'] = pick(inb(mid(MXa), 1, (SL['sn'][1] + SL['ls'][1]) / 2 - 0.004, (SL['sn'][1] + SL['ls'][1]) / 2 + 0.004), lambda P: P[:, 2])
md_ = mid(MD)
SL['li'] = pick(md_[md_[:, 2] > md_[:, 2].max() - 0.02], lambda P: P[:, 1])                          # (infradentale)
SL['me'] = pick(md_, lambda P: -P[:, 1])
SL['pg'] = pick(inb(md_, 1, SL['me'][1], (SL['me'][1] + SL['li'][1]) / 2 + 0.005), lambda P: P[:, 2])
SL['sm'] = pick(inb(md_, 1, SL['pg'][1], SL['li'][1]), lambda P: -P[:, 2])
SL['gn'] = pick(md_[md_[:, 1] < SL['pg'][1]], lambda P: P[:, 2] - P[:, 1])
SL['op'] = pick(mid(VAULT)[mid(VAULT)[:, 1] > ye - 0.02], lambda P: -P[:, 2])
SL['m'] = pick(inb(mid(FR), 1, SL['g'][1] + 0.04, SL['g'][1] + 0.06), lambda P: P[:, 2])
sg_ = mid(FR); SL['sg'] = sg_[np.argmin(np.abs(sg_[:, 1] - (SL['g'][1] + 0.018)) - 0.001 * sg_[:, 2])]
SL['v'] = pick(mid(VAULT), lambda P: P[:, 1])
dent = lambda nm: lm(nm).mean(0)          # (a tooth's centre: the teeth aren't among the skeleton's points)
M2_up = (dent('Upper_second_molar_toothl') + dent('Upper_second_molar_toothr')) / 2
for s_, sg in (('L', 1), ('R', -1)):
    sl_ = s_.lower(); e_ = eyes_c[s_]
    MX_s, ZG_s, TP_s = sk_named(rf'Maxilla{sl_}$'), sk_named(rf'Zygomatic_bone{sl_}$'), sk_named(rf'Temporal_bone{sl_}$')
    MD_s = MD[MD[:, 0] * sg > 0.005]
    SL['mso.' + s_] = pick(inb(FR[np.abs(FR[:, 0] - e_[0]) < 0.006], 1, e_[1] + 0.012, e_[1] + 0.035), lambda P: P[:, 2])
    io_ = np.vstack([MX_s, ZG_s]); SL['mio.' + s_] = pick(inb(io_[np.abs(io_[:, 0] - e_[0]) < 0.006], 1, e_[1] - 0.03, e_[1] - 0.012), lambda P: P[:, 2])
    ac_ = inb(MX_s, 1, SL['sn'][1], SL['sn'][1] + 0.015); SL['ac.' + s_] = pick(ac_[(ac_[:, 0] * sg > 0.012) & (ac_[:, 0] * sg < 0.028)], lambda P: P[:, 2])
    zy_ = inb(inb(np.vstack([ZG_s, TP_s]), 1, e_[1] - 0.035, e_[1] - 0.005), 2, e_[2] - 0.06, e_[2]); SL['zy.' + s_] = pick(zy_, lambda P: P[:, 0] * sg)
    go_ = MD_s[MD_s[:, 0] * sg > 0.03]; SL['go.' + s_] = pick(go_, lambda P: -P[:, 1] - P[:, 2])
    mr_ = inb(go_[go_[:, 2] < SL['go.' + s_][2] + 0.012], 1, SL['go.' + s_][1] + 0.015, SL['go.' + s_][1] + 0.035)
    SL['mr.' + s_] = pick(mr_, lambda P: P[:, 0] * sg)
    mm_ = (SL['me'] + SL['go.' + s_]) / 2
    mb_ = MD_s[np.linalg.norm((MD_s - mm_)[:, [0, 2]], axis=1) < 0.012]; SL['mmb.' + s_] = pick(mb_, lambda P: -P[:, 1])
    for code, tooth, bone_ in (('sC', 'Upper_canine', MX_s), ('iC', 'Lower_canine', MD_s),
                               ('sM2', 'Upper_second_molar_tooth', MX_s), ('iM2', 'Lower_second_molar_tooth', MD_s)):
        c_ = dent(tooth + sl_)
        out_ = np.array([c_[0], 0, c_[2] - M2_up[2]]) if code in ('sC', 'iC') else np.array([sg, 0, 0.0])
        out_ /= np.linalg.norm(out_)
        near_t = bone_[np.argsort(np.linalg.norm(bone_ - c_, axis=1))[:300]]     # (the bone around the tooth's root)
        SL[f'{code}.{s_}'] = pick(near_t, lambda P: (P - c_) @ out_)


def outward(p):
    """The skull's outward direction at a point (its surface normals within 4 mm, facing away from the head's middle)."""
    near = sk_m & (np.linalg.norm(skP - p, axis=1) < 0.004)
    n_ = sk_norm[near]; n_ *= np.sign(((skP[near] - head_c) * n_).sum(1, keepdims=True) + 1e-9)
    n_ = n_.sum(0) if len(n_) else p - head_c
    return n_ / np.linalg.norm(n_)


def depth_of(code):
    base = code.split('.')[0]
    if base == 'v': return (TD['m'][0] + TD['op'][0]) / 2     # (no data at the vertex: the mean of metopion and opisthocranion)
    return TD[base][0]


SKIN_L = {k: p + outward(p) * depth_of(k) for k, p in SL.items()}
print('skull landmark positions (cm): ' + ', '.join(f'{k} {np.round(p * 100, 1).tolist()}' for k, p in SL.items()), file=sys.stderr)
print('skull landmarks: ' + ', '.join(f'{k} {depth_of(k) * 1000:.1f}' for k in SL), file=sys.stderr)
# the depth spread over the skull: each point takes the landmarks' depths weighted by closeness (about 3 cm)
pool_ix = np.where(sk_m)[0]
_, first_ = np.unique(np.round(skP[pool_ix] / 0.003).astype(np.int64), axis=0, return_index=True)
pool_ix = pool_ix[first_]
skull_P = skP[pool_ix].astype(float)
skull_N = sk_norm[pool_ix] * np.sign(((skP[pool_ix] - head_c) * sk_norm[pool_ix]).sum(1, keepdims=True) + 1e-9)
L_P = np.array([SL[k] for k in SL]); L_d = np.array([depth_of(k) for k in SL])
dd = np.linalg.norm(skull_P[:, None] - L_P[None], axis=-1)
wd = np.exp(-(dd / 0.03) ** 2) + 1e-4 / (dd ** 2 + 1e-4)
skull_d = (wd * L_d).sum(1) / wd.sum(1)
# eyes, ears, nose
ears_all = extra[n_eyes:n_eyes + n_ear]
ear_c = {'L': ears_all[ears_all[:, 0] > 0].mean(0), 'R': ears_all[ears_all[:, 0] < 0].mean(0)}
nose_cart = extra[n_eyes + n_ear:]
prn_ours = nose_cart[np.argmax(nose_cart[:, 2])] + [0, 0, 0.002]     # (the nose tip: 2 mm in front of the cartilage)
mh_lm = mh_parts.mh_face_landmarks(mh_mesh)
# (only a few, robust landmarks, so the head keeps MakeHuman's own normal proportions: the eyes, brow, nose tip, chin,
# top and back of the head, cheekbones and ears; pulling the lips, philtrum and so on onto skull + depth too, and
# settling the whole face onto it, distorted it badly)
pairs = [('eye.L', eyes_c['L']), ('eye.R', eyes_c['R']), ('g', SKIN_L['g']), ('prn', prn_ours), ('me', SKIN_L['me']),
         ('top', SKIN_L['v']), ('op', SKIN_L['op']), ('zy.L', SKIN_L['zy.L']), ('zy.R', SKIN_L['zy.R']),
         ('ear.L', ear_c['L']), ('ear.R', ear_c['R'])]
A_head = np.array([mh_lm[k] for k, _ in pairs]); B_head = np.array([p for _, p in pairs])
# the neck: cut across it 1.5 cm below where the throat meets the underside of the chin (where the front of the old
# neck starts jutting forward, 1 cm past its deepest point), so the cut runs around a plain tube; cutting nearer the
# chin crossed its sloping underside at a shallow angle and left the cut edge in ragged pieces
neck0 = heads1[idx['spine.004']]
up_n = skP[skN_arr == 'Atlas_(C1)'].mean(0) - neck0; up_n /= np.linalg.norm(up_n)
s_body = (SV - neck0) @ up_n
rad_body = np.linalg.norm(SV - neck0 - np.outer(s_body, up_n), axis=1)
fz_ = [(sl_, SV[(s_body > sl_) & (s_body < sl_ + 0.005) & (rad_body < 0.12), 2].max()) for sl_ in np.arange(0, 0.08, 0.005)]
k_min = int(np.argmin([z_ for _, z_ in fz_]))
chin_start = next(sl_ for sl_, z_ in fz_[k_min:] if z_ > fz_[k_min][1] + 0.01)
cut_head = chin_start - 0.015
head_body = (s_body > cut_head) & (rad_body < 0.25)       # (everything above the cut: the shoulders are below it)
keep_body &= ~head_body
key = hashlib.sha1(b''.join([np.round(a, 5).tobytes() for a in (A_head, B_head, skull_P, skull_d, up_n, np.array([cut_head]))]) +
                   open(mh_parts.__file__, 'rb').read() + open(f'{MH_DIR}/base.obj', 'rb').read()).hexdigest()[:16]
cache = f'{MH_DIR}/cache/head-{key}.pkl'
was_cached = os.path.exists(cache)
if was_cached:
    P_, T_, Wp, s_, n_closed = pickle.load(open(cache, 'rb'))
else:
    P_, T_, Wp, s_, n_closed = mh_parts.fit_head(mh_mesh, A_head, B_head, neck0, up_n, cut_head, skull_P, skull_N, mh_lm,
                                                 neck_guide=SV[(s_body > cut_head - 0.02) & (s_body < cut_head + 0.04) & (rad_body < 0.10)])
    pickle.dump((P_, T_, Wp, s_, n_closed), open(cache, 'wb'))
cut_p = cut_head + 0.01
kt = (s_[T_] > cut_p).all(1)
# (only the main piece: any sliver of the cut left on its own is dropped)
import scipy.sparse as sp_, scipy.sparse.csgraph as cg_
Tk = T_[kt]; A_ = sp_.coo_matrix((np.ones(len(Tk) * 3), (Tk.ravel(), np.repeat(Tk[:, 0], 3))), (len(P_), len(P_)))
_, comp_ = cg_.connected_components(A_ + A_.T, directed=False)
big_ = np.bincount(comp_[Tk[:, 0]]).argmax()
kt[np.where(kt)[0][comp_[Tk[:, 0]] != big_]] = False
used = np.unique(T_[kt]); rm = np.full(len(P_), -1); rm[used] = np.arange(len(used))
base_ = sum(len(x) for x in new_P)
new_P.append(P_[used]); new_T.append(rm[T_[kt]] + base_)
Wd_ = np.zeros((len(used), NB + len(helpers)), np.float32)
for b_, iw in Wp.items():
    ob_ = mh_parts.mh_head_bone_to_ours(b_)
    if ob_ is None: continue
    for i_, w_ in iw:
        if rm[i_] >= 0: Wd_[rm[i_], idx[ob_]] += w_
Wd_[Wd_.sum(1) < 1e-6, idx['spine.005']] = 1       # (the patches closing the eyes and mouth: the head)
mh_idx.append(np.arange(len(used)) + base_); mh_skin.append(P_[used]); mh_W.append(Wd_ / Wd_.sum(1, keepdims=True))
mh_fset.append(np.full(len(used), 'head')); mh_digit.append(np.zeros(len(used), np.uint8)); mh_kind.append(np.full(len(used), 'head'))
b_ = np.clip((s_[used] - cut_p) / 0.03, 0, 1); mh_blend.append(b_ * b_ * (3 - 2 * b_))     # (over 3 cm)
cut_loops.append(('head', neck0 + up_n * (cut_head + 0.005), up_n, rm[mh_parts.boundary_loop(T_[kt], len(P_))[0]] + base_))
PARTS.append(('head', '', '', 'spine.004', None, cut_head, cut_p, (), 'head', None))
print(f'head: MakeHuman head fitted, {len(used)} points ({"cached" if was_cached else "new"}); eyes and mouth closed '
      f'({n_closed} points inside them removed); neck cut {cut_head * 100:.1f} cm up the neck', file=sys.stderr)

# cut the body, then bridge each wrist and ankle
SVall = np.vstack(new_P)
kept_T = ST[keep_body[ST].all(1)]
# tidy each cut edge: triangles left hanging on by one corner or one edge (two or three of their edges open) are
# peeled off, a few times over, so every cut edge is a simple ring the strip can follow
for _ in range(6):
    a_, b_ = kept_T.ravel(), kept_T[:, [1, 2, 0]].ravel()
    open_e = ~np.isin(a_.astype(np.int64) * len(SV) + b_, b_.astype(np.int64) * len(SV) + a_)
    loose = open_e.reshape(-1, 3).sum(1) >= 2
    # (and triangles meeting the cut edge at a single corner shared by two rings of open edges)
    deg_open = np.bincount(a_[open_e], minlength=len(SV))
    pinch = (deg_open[kept_T] > 1).any(1)
    if not (loose | pinch).any(): break
    kept_T = kept_T[~(loose | pinch)]
a_, b_ = kept_T.ravel(), kept_T[:, [1, 2, 0]].ravel()
NV_ = len(SVall)
fwd = a_.astype(np.int64) * NV_ + b_; rev = b_.astype(np.int64) * NV_ + a_
open_ = ~np.isin(fwd, rev)
nxt_ = dict(zip(a_[open_], b_[open_]))
body_loops = []
seen_ = set()
for v0 in nxt_:
    if v0 in seen_: continue
    loop_, v = [], v0
    while v is not None and v not in seen_: seen_.add(v); loop_.append(v); v = nxt_.get(v)
    body_loops.append(np.array(loop_))
bridges = []
for name_, c_, u, hl in cut_loops:
    bl = max((l for l in body_loops if np.linalg.norm(SVall[l].mean(0) - c_) < 0.06), key=len)
    bridges.append(mh_parts.bridge(SVall, bl, hl, u, c_))
    print(f'{name_}: body cut ring {len(bl)} points, hand ring {len(hl)} points, '
          f'{len(bridges[-1])} bridge triangles; {len(body_loops)} open rings in all', file=sys.stderr)
ST = np.vstack([kept_T] + new_T + bridges)
# any small hole the zipping left (3-12 points, where the two rings' order around the limb disagrees) is closed
a_, b_ = ST.ravel(), ST[:, [1, 2, 0]].ravel()
open_ = ~np.isin(a_.astype(np.int64) * NV_ + b_, b_.astype(np.int64) * NV_ + a_)
nxt_ = dict(zip(a_[open_], b_[open_])); seen_, fills = set(), []
for v0 in nxt_:
    if v0 in seen_: continue
    loop_, v = [], v0
    while v is not None and v not in seen_: seen_.add(v); loop_.append(v); v = nxt_.get(v)
    if 3 <= len(loop_) <= 12:
        fills += [(loop_[0], loop_[k + 1], loop_[k]) for k in range(1, len(loop_) - 1)]   # (reverse of the hole's edges)
if fills: ST = np.vstack([ST, np.array(fills)])
print(f'joins: {len(fills)} triangles closing tiny holes', file=sys.stderr)
# drop the body points that were cut off (renumbering everything)
alive = np.zeros(len(SVall), bool); alive[ST.ravel()] = True
ren = np.cumsum(alive) - 1
SV, ST = SVall[alive], ren[ST]
FSET = np.concatenate([FSET] + mh_fset)[alive]
digit = np.zeros(len(SV), np.uint8); digit[ren[np.concatenate(mh_idx)]] = np.concatenate(mh_digit)
mh_kind = np.concatenate(mh_kind)
mh_idx = ren[np.concatenate(mh_idx)]; mh_skin = np.vstack(mh_skin); mh_W = np.vstack(mh_W); mh_blend = np.concatenate(mh_blend)
mh_fat = mh_skin - mh_parts.vertex_normals(SV, ST)[mh_idx] * 0.0015       # (the part's fat: 1.5 mm under its skin)
print(f'hands and feet in: {len(SV)} surface points now', file=sys.stderr)
print('adipose verts', len(SV), 'tris', len(ST), file=sys.stderr)

# joint weights: copy from the muscle / bone right under each skin point, then blend a little across the skin
SW = np.zeros((len(SV), NB + len(helpers)), np.float32)
look_P, look_B, look_W = anat_P, anat_B, anat_W
for side in (1, -1):
    m = SV[:, 0] * side >= 0
    pool = np.where(look_P[:, 0] * side >= -0.005)[0]
    near = pool[nearest_index(SV[m], look_P[pool], margin=0.06)]
    rows = np.where(m)[0]
    for k in range(2): add_at(SW, (rows, look_B[near, k]), look_W[near, k])
si = np.concatenate([ST[:, 0], ST[:, 1], ST[:, 2], ST[:, 1], ST[:, 2], ST[:, 0]])
sj = np.concatenate([ST[:, 1], ST[:, 2], ST[:, 0], ST[:, 0], ST[:, 1], ST[:, 2]])
sdeg = np.bincount(si, minlength=len(SV)).astype(np.float32)[:, None]
for _ in range(4):
    acc = fastgeo.nsum(si, sj, SW)
    SW = 0.5 * SW + 0.5 * acc / np.maximum(sdeg, 1)
# the MakeHuman hands and feet bring their own rig's weights (mapped onto our hand, finger, foot and toe bones)
SW[mh_idx] = (1 - mh_blend[:, None]) * SW[mh_idx] + mh_blend[:, None] * mh_W

# Skin is one connected sheet: on the trunk, a joint's pull spreads well beyond the muscle right under it
# (turn the chest and the skin over the abs shifts too). Each trunk skin point takes a distance-weighted
# blend (about 4 cm falloff, 10 cm reach) of the weights of the trunk skin around it. Arm and leg skin is
# left out of the blend so a hanging arm doesn't mix into the side of the chest.
TRUNK = [b for b, n in enumerate(bone_names) if n.startswith(('spine', 'pelvis', 'shoulder'))] + \
        [NB + k for k, h in enumerate(helpers) if 'upper_arm' in h]
trunk_amt = SW[:, TRUNK].sum(1) / np.maximum(SW.sum(1), 1e-9)
is_trunk = trunk_amt > 0.6
cell_k = np.floor(SV[is_trunk] / 0.02).astype(np.int64)
_, cinv = np.unique(cell_k, axis=0, return_inverse=True); cinv = cinv.ravel()
cnt = np.bincount(cinv).astype(np.float32)
cP = np.stack([np.bincount(cinv, SV[is_trunk, c]) for c in range(3)], 1) / cnt[:, None]
cW = np.stack([np.bincount(cinv, SW[is_trunk, c], minlength=len(cnt)) for c in range(SW.shape[1])], 1) / cnt[:, None]
smooth_W = np.zeros_like(SW)
rows = np.where(trunk_amt > 0.3)[0]
for i in range(0, len(rows), 2048):
    r = rows[i:i + 2048]
    d2 = ((SV[r, None, :] - cP[None]) ** 2).sum(-1)
    g = np.exp(-d2 / (2 * 0.04 ** 2)) * (d2 < 0.10 ** 2) * cnt[None]
    smooth_W[r] = (g @ cW) / np.maximum(g.sum(1, keepdims=True), 1e-9)
beta = np.clip((trunk_amt - 0.3) / 0.4, 0, 1)[:, None]
SW = (1 - beta) * SW + beta * smooth_W
print(f'trunk skin weights spread over {int((beta[:, 0] > 0).sum())} points', file=sys.stderr)

# Armpit / side of the trunk: when the arm lifts, the skin stretches all the way from the bottom of the
# ribcage, more the nearer it is to the arm. Trunk skin on that side takes a share of the arm's 'pull'
# (the helper bone that follows the attachment point on the humerus without swinging out with the arm):
# 0 at the bottom of the ribcage, growing smoothly to 80% near the armpit.
rib_bottom = skP[skB == idx['spine.003'], 1].min()
for k, h in enumerate(helpers):
    if 'upper_arm' not in h: continue
    side = 1 if h['name'].endswith('.L') else -1
    ins = np.array(h['head'])
    D = np.linalg.norm(ins - np.array([ins[0] * 0.85, rib_bottom, ins[2]]))     # armpit -> bottom of the ribcage
    m = (SV[:, 0] * side > 0.02) & (SV[:, 1] > rib_bottom) & (trunk_amt > 0.4)
    t = np.clip(1 - np.linalg.norm(SV[m] - ins, axis=1) / D, 0, 1)
    t = 0.8 * (3 * t ** 2 - 2 * t ** 3)                                          # smooth growth toward the armpit
    rows = np.where(m)[0]
    SW[rows] *= (1 - t)[:, None]
    SW[rows, NB + k] += t
    print(f'{h["name"]}: {int((t > 0.05).sum())} skin points share the arm pull', file=sys.stderr)
# extra smoothing over the hands and head so muscle texture doesn't show through (skin and fat even it out)
hand_cols = [b for b, n in enumerate(bone_names) if n.startswith(('f_', 'thumb', 'hand')) or n == 'spine.005']
hand_w = np.clip(SW[:, hand_cols].sum(1) / np.maximum(SW.sum(1), 1e-9), 0, 1)[:, None]
for _ in range(25):
    for k in (0.5, -0.53):
        acc = fastgeo.nsum(si, sj, SV)
        SV = SV + k * hand_w * (acc / np.maximum(sdeg, 1) - SV)
SW = SW / np.maximum(SW.sum(1, keepdims=True), 1e-9)


def top4(W):
    t = np.argsort(-W, 1)[:, :4]; w = np.take_along_axis(W, t, 1); return t, w / np.maximum(w.sum(1, keepdims=True), 1e-9)


# ---- fat thickness by body region (athletic man, ~15% body fat). From skinfold / ultrasound studies:
# skinfold ~ 2 x (skin + fat): abdomen ~11.5 mm, thigh ~11, below the shoulder blade ~9, triceps ~7.3,
# chest ~6.6; ultrasound order thigh > triceps > upper back > abdomen > hip crest > chest > calf > forearm.
SNf = normals(SV, ST)
dom_b = SW[:, :NB].argmax(1)
dom_n = np.array([base_name(bone_names[b]) for b in dom_b])
front, back = SNf[:, 2] > 0.3, SNf[:, 2] < -0.3
BASE_T = {'spine': 7.0, 'pelvis': 7.0, 'spine.001': 7.0, 'spine.002': 5.5, 'spine.003': 4.5, 'spine.004': 3.0,
          'spine.005': 3.5, 'shoulder': 3.0, 'upper_arm': 3.5, 'forearm': 2.0, 'hand': 1.5, 'thigh': 7.0,
          'shin': 2.5, 'foot': 2.0, 'heel.02': 2.0, 'toe': 1.5}
t_mm = np.array([BASE_T.get(n, 3.0) for n in dom_n])
lower_trunk = np.isin(dom_n, ['spine', 'pelvis', 'spine.001'])
t_mm[lower_trunk & front] = 9.0                                    # lower belly
t_mm[lower_trunk & back & (SV[:, 1] < heads1[idx['spine.001']][1])] = 12.0   # buttocks (a round seat, not too full; the
# old thick envelope used to carry this volume, the thinner one leaves it to the fat)
t_mm[lower_trunk & back & (SV[:, 1] >= heads1[idx['spine.001']][1])] = 6.0   # lower back
t_mm[(dom_n == 'spine.002') & front] = 6.0                         # upper belly
t_mm[(dom_n == 'spine.003') & front] = 3.5                         # chest
t_mm[(dom_n == 'spine.003') & back] = 5.0                          # upper back
# The arms: the adipose layer carries an average man's arm size (ANSUR II men: upper arm ~32 cm around relaxed,
# forearm ~28 cm, wrist 17.6 cm). Our arm muscles were slimmed on purpose, so the fat makes up the size: ~13 mm on
# the upper arm (more at the back), and on the forearm 9 mm near the elbow thinning to 6 mm at the wrist. The
# skin (dermis) is the 2 mm layer on top of this.
t_mm[dom_n == 'upper_arm'] = 13.0
t_mm[(dom_n == 'upper_arm') & back] = 14.0                         # back of the upper arm
t_mm[(dom_n == 'upper_arm') & front] = 12.0                        # biceps
# (the armpit and the top of the shoulder carry little fat: within 6 cm of the shoulder joint the upper arm's fat thins to
# 5 mm, back to full by 12 cm, so the thick arm fat doesn't bunch into folds there when the arm moves)
for sfx in ('.L', '.R'):
    ua_ = (dom_n == 'upper_arm') & ((SV[:, 0] > 0) if sfx == '.L' else (SV[:, 0] < 0))
    f_ = np.clip((np.linalg.norm(SV[ua_] - heads1[idx['upper_arm' + sfx]], axis=1) - 0.06) / 0.06, 0, 1)
    t_mm[ua_] = 5.0 + (t_mm[ua_] - 5.0) * f_ * f_ * (3 - 2 * f_)
for sfx in ('.L', '.R'):
    el, wr = heads1[idx['forearm' + sfx]], heads1[idx['hand' + sfx]]
    on_ = (dom_n == 'forearm') & ((SV[:, 0] > 0) if sfx == '.L' else (SV[:, 0] < 0))
    f_ = np.clip(((SV[on_] - el) @ (wr - el)) / ((wr - el) @ (wr - el)), 0, 1)
    t_mm[on_] = 9.0 - 3.0 * f_          # (tapering evenly to the wrist, never thinner than it)
# The bony points of the arm carry almost no fat, so they show as firm points: the olecranon (the point of the elbow,
# at the back) and the end of the ulna at the wrist (little-finger side) thin to ~2.5 mm, fading out over 1.5 / 1 cm.
for sfx, sl_ in (('.L', 'l'), ('.R', 'r')):
    uln = skP[skN_arr == 'Ulna' + sl_].astype(float)
    el_, wr_ = heads1[idx['forearm' + sfx]], heads1[idx['hand' + sfx]]
    top_ = uln[np.linalg.norm(uln - el_, axis=1) < 0.04]
    olec = top_[np.argmax((top_ - el_) @ ((el_ - wr_) / np.linalg.norm(el_ - wr_)))]          # (the ulna's tip past the elbow)
    head_ = uln[np.linalg.norm(uln - wr_, axis=1) < 0.03]
    for pt, r_ in ((olec, 0.015), (head_.mean(0), 0.010)):
        g_ = np.exp(-np.sum((SV - pt) ** 2, 1) / (2 * r_ ** 2)) * ((SV[:, 0] > 0) if sfx == '.L' else (SV[:, 0] < 0))
        t_mm = t_mm * (1 - g_) + 2.5 * g_
t_mm[(dom_n == 'shin') & back] = 5.0                               # calf
t_mm[(dom_n == 'shin') & front] = 2.5                              # shin
t_mm[(dom_n == 'thigh') & front] = 9.0                             # front of the thigh
t_mm[(dom_n == 'thigh') & ~front] = 10.5                           # inner / back of the thigh
# The default build is an ordinary fit man at ~17% body fat (the targets above are an athletic ~15%). With the
# Jackson-Pollock 3-site equation (men, age 30) and Siri's formula, 15% is a skinfold sum of 49.7 mm and 17% of 56.6 mm:
# every skinfold ~14% thicker. A skinfold is twice (skin + fat) and the skin stays ~2 mm, so the trunk's and legs'
# fat grows a little more than 14%. (The arms are already sized to an average man from ANSUR II; the bony points stay thin.)
BODY_FAT_SCALE, SKIN_MM = 56.6 / 49.7, 2.0
trunk_leg = np.isin(dom_n, ['spine', 'pelvis', 'spine.001', 'spine.002', 'spine.003', 'thigh', 'shin'])
t_mm[trunk_leg] = (t_mm[trunk_leg] + SKIN_MM) * BODY_FAT_SCALE - SKIN_MM
# smooth the target over the surface so regions blend gradually
fi = np.concatenate([ST[:, 0], ST[:, 1], ST[:, 2], ST[:, 1], ST[:, 2], ST[:, 0]])
fj = np.concatenate([ST[:, 1], ST[:, 2], ST[:, 0], ST[:, 0], ST[:, 1], ST[:, 2]])
fdeg = np.maximum(np.bincount(fi, minlength=len(SV)), 1).astype(float)


def fsmooth(D, passes):
    for _ in range(passes):
        acc = fastgeo.nsum(fi, fj, D); D = 0.5 * D + 0.5 * acc / (fdeg if D.ndim == 1 else fdeg[:, None])
    return D


t_m = fsmooth(t_mm / 1000, 30)
# move each fat point toward its target height above the muscle / bone under it; the correction itself is
# smoothed heavily, so the fat changes thickness gradually and keeps hiding the muscle shapes
# (the thickness is measured to the closed muscle surface: the same anatomy grown and closed the same way but only
# lightly blurred, so not to the nearest groove wall, and not puffed out in hollows like the neck by the heavy blur
# above; no cap, and only light smoothing, so every region reaches its target)
from scipy.spatial import cKDTree
Vc_, _ = envelope_pieces([(p_[0], p_[1], p_[2], p_[3], 2, *p_[5:]) for p_ in fat_pieces])
h = cKDTree(Vc_).query(SV)[0]
print(f'closed muscle surface: {len(Vc_)} points; fat surface {np.median(h) * 1000:.1f} mm off it before the correction', file=sys.stderr)
if os.environ.get('STAGE_DUMP'): STAGES.update(env=SV.copy(), T=ST.copy(), t_m=t_m.copy(), h=h.copy(), SNf=SNf.copy())
corr = fsmooth(t_m - h, 20)
SV = SV + SNf * corr[:, None]
print(f'fat thickness: target {t_m.mean() * 1000:.1f} mm avg, moved {np.abs(corr).mean() * 1000:.1f} mm avg', file=sys.stderr)
# two strong smoothing rounds on the final fat
# (much lighter on the arms, so the muscle shapes show softly through the skin there)
arm_cols_ = [b for b, n in enumerate(bone_names) if n.startswith(('upper_arm', 'forearm', 'hand'))]
arm_soft = 1 - 0.7 * fsmooth(np.clip(SW[:, arm_cols_].sum(1) / np.maximum(SW.sum(1), 1e-9), 0, 1), 10)
for _ in range(2): SV = mesh_taubin(SV, ST, 50, w=arm_soft)      # (light: the fat keeps the soft muscle shapes)
if os.environ.get('STAGE_DUMP'): STAGES['thick'] = SV.copy()

# ---- the fat always wraps over the muscles, never below or behind them: every fat point is kept at least
# 1.5 mm outside the muscle right under it (along that muscle surface's outward direction, checked against
# the fat's own outward direction, since some muscle meshes face the other way)
MIN_FAT = 0.0015
mN0 = normals(mus_P, mus_F)
# (No build-time push here: the closing already wraps the fat around the muscles, and pushing out over
# every small tendon edge left the fat covered in bumps. The page keeps the fat over the muscles while posing.)
# 2 mm more fat from below the neck down, except on the hands and feet (fading in smoothly)
extra = (SV[:, 1] < heads1[idx['spine.004']][1]).astype(float)
hf = [b for b, n in enumerate(bone_names) if n.startswith(('hand', 'f_', 'thumb', 'foot', 'toe', 'heel'))]
extra *= 1 - np.clip(SW[:, hf + arm_cols_].sum(1) / np.maximum(SW.sum(1), 1e-9), 0, 1)     # (nor on the arms)
extra = fsmooth(extra, 15)
SV = SV + normals(SV, ST) * (0.002 * extra)[:, None]
# The cleft between the buttocks (back, within ~1.5 cm of the midline, from the crotch up to just above the
# hip joints) is left alone by the extra leg fat and the extra smoothing, so the two cheeks stay distinct.
# (only the line itself, within 6 mm of the midline: it keeps the line between the cheeks without keeping a chasm)
cleft = (np.abs(SV[:, 0]) < 0.006) & (SV[:, 2] < heads1[idx['thigh.L']][2]) & \
        (SV[:, 1] > hip_y - 0.14) & (SV[:, 1] < hip_y + 0.06)
keep_cleft = 1 - fsmooth(cleft.astype(float), 2)
# and 3 mm more on the legs (thighs most, fading toward the ankles; none on the feet)
leg_cols = [b for b, n in enumerate(bone_names) if n.startswith(('thigh', 'shin'))]
leg_amt = np.clip(SW[:, leg_cols].sum(1) / np.maximum(SW.sum(1), 1e-9), 0, 1)
leg_amt *= np.clip((SV[:, 1] - 0.08) / 0.25, 0, 1)
leg_amt = fsmooth(leg_amt, 15) * keep_cleft
SV = SV + normals(SV, ST) * (0.003 * leg_amt)[:, None]
# The front of the crotch is levelled: front points within 5 cm of the midline move (by up to 1.5 cm) to the
# level of the surface 4-5 cm out at the same height (forward out of a groove, or back if they stick out),
# fading out toward the sides, the crotch and the belly, so it is one flat, smooth surface.
hip_z0 = heads1[idx['thigh.L']][2]
fy0 = SV[(np.abs(SV[:, 0]) < 0.01) & (SV[:, 2] > hip_z0), 1].min()                # the crotch, at the front
band_ = np.floor((SV[:, 1] - fy0) / 0.005).astype(int)
zone = (SV[:, 2] > hip_z0) & (np.abs(SV[:, 0]) < 0.05) & (SV[:, 1] < m_top + 0.02)
side_ = zone & (np.abs(SV[:, 0]) > 0.035)
side_z = {b: SV[side_ & (band_ == b), 2].max() for b in np.unique(band_[side_])}
fill = np.zeros(len(SV))
for i in np.where(zone)[0]:
    if band_[i] not in side_z: continue
    x_ = abs(SV[i, 0]) / 0.05
    fill[i] = np.clip(side_z[band_[i]] - SV[i, 2], -0.015, 0.015) * (1 - x_ ** 2) ** 2
# only the front-facing surface, fading out from 6 cm down to 3 cm above the crotch: the underside and the
# dip where the legs part are left alone, so no ledge forms and nothing webs between the legs
fill *= np.clip((normals(SV, ST)[:, 2] - 0.3) / 0.4, 0, 1)
fill *= np.clip((SV[:, 1] - fy0 - 0.03) / 0.03, 0, 1) * np.clip((m_top + 0.02 - SV[:, 1]) / 0.02, 0, 1)
fill = fsmooth(fill, 8)
SV[:, 2] += fill
print(f'crotch levelled: {fill.max() * 1000:.1f} mm forward, {-fill.min() * 1000:.1f} mm back at most', file=sys.stderr)
# extra fat on the buttocks for a relaxed, round, plump shape (not clenched): up to 10 mm, broad and sitting
# low on each cheek, filling in right up to the cleft, so the cheeks meet in a line rather than a deep
# groove; plus 5 mm on the outer side of each hip to fill the hollow there
hip_z = heads1[idx['thigh.L']][2]
for side in (1, -1):
    r2 = ((SV[:, 0] - side * 0.075) / 0.10) ** 2 + ((SV[:, 1] - (hip_y - 0.08)) / 0.12) ** 2
    bump = 0.010 * np.exp(-2 * r2) * (SV[:, 2] < hip_z)
    r2 = ((SV[:, 0] - side * 0.15) / 0.06) ** 2 + ((SV[:, 1] - (hip_y - 0.02)) / 0.08) ** 2
    bump += 0.005 * np.exp(-2 * r2) * (SV[:, 2] < hip_z + 0.03)
    SV = SV + normals(SV, ST) * (fsmooth(bump * keep_cleft, 10))[:, None]
SV = mesh_taubin(SV, ST, 70, w=arm_soft)   # and one more smoothing round at the very end,
SV = mesh_taubin(SV, ST, 40, w=keep_cleft * arm_soft)   # plus 40 more everywhere but the cleft
# the line between the cheeks: a soft crease down the back midline (up to 5 mm deep, about 1.5 cm across), deepest near
# the tailbone and fading out above it and toward the fold under the buttocks
hz_ = heads1[idx['thigh.L']][2]
crease_ = 0.005 * np.exp(-(SV[:, 0] / 0.007) ** 2) * (SV[:, 2] < hz_ - 0.02) * \
          np.clip((SV[:, 1] - (hip_y - 0.15)) / 0.03, 0, 1) * np.clip(((hip_y + 0.06) - SV[:, 1]) / 0.04, 0, 1)
SV = SV - normals(SV, ST) * fsmooth(crease_, 4)[:, None]
# the skin between the legs, under the cleft, is pulled flat (plain smoothing, which draws bumps in), so it doesn't bulge
under_ = (np.abs(SV[:, 0]) < 0.03) & (SV[:, 1] > hip_y - 0.13) & (SV[:, 1] < hip_y - 0.05) & (np.abs(SV[:, 2] - hz_) < 0.08)
uw_ = fsmooth(under_.astype(float), 4)
for _ in range(15):
    acc = fastgeo.nsum(fi, fj, SV); SV = SV + 0.5 * uw_[:, None] * (acc / fdeg[:, None] - SV)
butt_ = (np.abs(SV[:, 0]) < 0.12) & (SV[:, 2] < hz_) & (SV[:, 1] > hip_y - 0.18) & (SV[:, 1] < hip_y + 0.08)
SV = mesh_taubin(SV, ST, 20, w=fsmooth(butt_.astype(float), 6) * 0.6)      # (and the whole seat evened out a little)

# pairs for the page, so the fat also stays over the muscle while posing
j = nearest_index(SV, mus_P, margin=0.04)
SNf = normals(SV, ST)
mN0[j] *= np.sign(np.sum(mN0[j] * SNf, 1, keepdims=True) + 1e-9)    # muscle normals facing out, like the fat
gap = ((SV - mus_P[j]) * mN0[j]).sum(1)
ok = (np.linalg.norm(SV - mus_P[j], axis=1) < 0.03) & (gap > 0)
fat_over = (np.where(ok)[0], j[ok], np.clip(gap[ok], MIN_FAT, 0.02))

# ---- adipose layer: this surface (the hands and feet are MakeHuman's now, with no finger or toe tubes: put back
# to their fitted shape, under 1.5 mm of skin)
SV[mh_idx] = (1 - mh_blend[:, None]) * SV[mh_idx] + mh_blend[:, None] * mh_fat

# ---- the joins: wrists, ankles and neck sized to an average man (ANSUR II men's means, skin: wrist 17.6 cm, narrowest
# ankle 22.9 cm, neck 43.5 cm at its base and 39.8 cm above it), across both sides of each join, then blended softly.
# Along each limb, every 5 mm slice of the fat is scaled about its centre (across the limb only) toward the size that
# gives that skin, easing in and out at the ends of each stretch; nothing is moved closer than 1.5 mm to bone or
# muscle. The MakeHuman part's own skin gets the same change, so the two sides of each join match.
SKIN_D = 0.0022                                         # (the skin layer on top of the fat)
anat_tree = cKDTree(anat_P)


def ring_perimeter(V, T, c, u, R, side=0):
    d = (V - c) @ u; sg = np.sign(d[T]); Tm = T[(sg.min(1) < 0) & (sg.max(1) > 0)]
    tot = 0.0
    for t in Tm:
        pts = [V[a] + d[a] / (d[a] - d[b]) * (V[b] - V[a]) for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])) if (d[a] < 0) != (d[b] < 0)]
        if len(pts) != 2: continue
        q = (pts[0] + pts[1]) / 2 - c; q -= u * (q @ u)
        if np.linalg.norm(q) > R or (side and (pts[0][0] + pts[1][0]) / 2 * side < 0.005): continue
        tot += np.linalg.norm(pts[1] - pts[0])
    return tot


def ease(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


# (centre, axis, radius, side, skin size, (s0, s1): where it fully applies, fade, the join's cut, where the size starts
# tapering from the limb's own size (the forearm tapers evenly down to the wrist instead of pinching in above it))
JOIN_SIZES = []
for sfx, sg in (('.L', 1), ('.R', -1)):
    wr, el = heads1[idx['hand' + sfx]], heads1[idx['forearm' + sfx]]
    JOIN_SIZES.append((wr, (wr - el) / np.linalg.norm(wr - el), 0.07, 0, 0.176, (-0.08, 0.01), 0.02, 0.0, -0.08))
    an, kn = heads1[idx['foot' + sfx]], heads1[idx['shin' + sfx]]
    JOIN_SIZES.append((an, (an - kn) / np.linalg.norm(an - kn), 0.12, sg, 0.229, (-0.09, -0.06), 0.02, -0.05, None))
JOIN_SIZES.append((neck0, up_n, 0.12, 0, None, (0.01, 0.035), 0.01, 0.03, None))   # (the neck: 43.5 at its base, 39.8 above)


def size_points(Q, plan):
    """Move points Q by the joins' sizing plan: per join, each point is scaled about the ring centre at its height."""
    Q = Q.copy()
    for c, u, R, side, sl, k, cen, w, axes, g in plan:
        d = (Q - c) @ u; r_ = Q - c - np.outer(d, u)
        on = (np.linalg.norm(r_, axis=1) < R) & (d > sl[0]) & (d < sl[-1])
        if side: on &= Q[:, 0] * side > 0.005
        if not on.any(): continue
        kk = 1 + np.interp(d[on], sl, w) * (np.interp(d[on], sl, k) - 1)
        cc = np.stack([np.interp(d[on], sl, cen[:, j]) for j in range(3)], 1)
        rr = Q[on] - cc; rr -= np.outer(rr @ u, u)
        # (and the shape: stretched along the slice's long axis and squeezed along its short one, area kept)
        near_sl = np.clip(np.round((d[on] - sl[0]) / (sl[1] - sl[0])).astype(int), 0, len(sl) - 1)
        e1 = axes[near_sl]; a1 = (rr * e1).sum(1)[:, None]; gg = g[near_sl][:, None]
        rr2 = rr + e1 * a1 * (gg - 1) + (rr - e1 * a1) * (1 / gg - 1)
        Q[on] += rr2 * kk[:, None] - rr
    return Q


size_plan = []
for c, u, R, side, size_, (s0, s1), fade, cut_s, taper in JOIN_SIZES:
    sl = np.arange(min(s0 - fade, cut_s - 0.035) - 0.005, max(s1 + fade, cut_s + 0.035) + 0.0051, 0.005)
    ks, cens, ws, axes, asp = [], [], [], [], []
    for s_ in sl:
        cc = c + u * s_
        d = (SV - cc) @ u; r_ = SV - cc - np.outer(d, u)
        m = (np.abs(d) < 0.0025) & (np.linalg.norm(r_, axis=1) < R) & ((SV[:, 0] * side > 0.005) if side else True)
        cen_ = SV[m].mean(0) - u * ((SV[m].mean(0) - cc) @ u) if m.sum() > 5 else cc
        cens.append(cen_)
        q_ = SV[m] - cen_; q_ -= np.outer(q_ @ u, u)
        if len(q_) > 5:
            ev, evec = np.linalg.eigh(np.cov(q_.T)); e1_ = evec[:, -1]; asp.append(np.sqrt(ev[-1] / max(ev[-2], 1e-12)))
        else: e1_ = np.cross(u, [0, 0, 1.0]); asp.append(1.0)
        axes.append(e1_ / np.linalg.norm(e1_))
        per = ring_perimeter(SV, ST, cc, u, R, side)
        if taper is not None and s_ < s0 + 0.06:
            per_t = ring_perimeter(SV, ST, c + u * taper, u, R, side) + 2 * np.pi * SKIN_D
            tgt = np.interp(s_, [taper, s0 + 0.06], [per_t, size_])           # (from the limb's own size down to the target)
        else:
            tgt = size_ if size_ is not None else np.interp(s_, [0.01, 0.02], [0.435, 0.398])
        ks.append((tgt - 2 * np.pi * SKIN_D) / per if per > 0.05 else 1.0)
        ws.append(ease((s_ - (s0 - fade)) / fade) * ease(((s1 + fade) - s_) / fade))
    # the shape: within 3 cm of the cut, each slice's long-to-short ratio blends from the body's (3 cm before the cut)
    # to the part's (3 cm after it)
    asp = np.array(asp); a_body, a_part = np.interp(cut_s - 0.03, sl, asp), np.interp(cut_s + 0.03, sl, asp)
    t_ = np.clip((sl - (cut_s - 0.03)) / 0.06, 0, 1)
    a_t = a_body + (a_part - a_body) * ease(t_)
    g_ = np.where(np.abs(sl - cut_s) < 0.03, np.sqrt(np.clip(a_t / np.maximum(asp, 1e-6), 0.5, 2.0)), 1.0)
    size_plan.append((c, u, R, side, sl, np.array(ks), np.array(cens), np.array(ws), np.array(axes), g_))
    print(f'join sizing at {np.round(c, 3)}: scale {min(ks):.2f}-{max(ks):.2f} over {len(sl)} slices; shape ratio {a_body:.2f} -> {a_part:.2f}', file=sys.stderr)
moved_ = size_points(SV, size_plan)
# (never closer than 1.5 mm to bone or muscle: points that would be are left where they were)
too_close = (anat_tree.query(moved_)[0] < 0.0015) & (np.linalg.norm(moved_ - SV, axis=1) > 0)
moved_[too_close] = SV[too_close]
print(f'join sizing: {int((np.linalg.norm(moved_ - SV, axis=1) > 1e-4).sum())} fat points moved, {int(too_close.sum())} held off the bone', file=sys.stderr)
SV = moved_
mh_skin = size_points(mh_skin, size_plan)
# the joins blended softly: a smoothing band 3 cm either side of each cut, strongest at the cut
join_w = np.zeros(len(SV))
for kind, sfx, sl_, jn, pn, cut_b, cut_p, *_ in PARTS:
    if kind == 'head': c_, u_ = neck0, up_n
    else:
        c_ = heads1[idx[jn + sfx]]; u_ = c_ - heads1[idx[pn + sfx]]; u_ = u_ / np.linalg.norm(u_)
    d = (SV - c_) @ u_ - (cut_b + cut_p) / 2; r_ = np.linalg.norm(SV - c_ - np.outer((SV - c_) @ u_, u_), axis=1)
    near_ = (r_ < (0.12 if kind != 'hand' else 0.07)) & (np.linalg.norm(SV - c_, axis=1) < 0.25)
    if sfx: near_ &= (SV[:, 0] > 0) if sfx == '.L' else (SV[:, 0] < 0)
    join_w = np.maximum(join_w, near_ * np.clip(1 - np.abs(d) / 0.03, 0, 1))
SV = mesh_taubin(SV, ST, 15, w=join_w)
finger_tubes['skin'] = []


def symmetrize(V, T, rounds=3):
    """Make a surface the same on both sides: each point moves half way onto the mirror image of the surface (onto the
    tangent plane of its mirrored partner, along its own normal, so no sideways jitter), a few times over. The anatomy
    is symmetric to ~1 mm, but building the surface on a 3 mm voxel grid not centred on the midline left the two sides
    ~3 mm apart (up to 3 cm around the buttock cleft). The partner is the nearest mirrored point facing the same way
    (within 60 degrees; none of the 8 nearest: the point stays): just the nearest one was often on the other wall
    of the cleft, facing the other way, and pushing points by that creased the skin flat in ~150 places there. The push
    is smoothed over the surface so neighbouring points move alike."""
    for _ in range(rounds):
        N = normals(V, T)
        Vm = V * [-1, 1, 1]; Nm = N * [-1, 1, 1]
        jj = cKDTree(Vm).query(V, k=8)[1]
        ok = (Nm[jj] * N[:, None, :]).sum(2) > 0.5
        first = ok.argmax(1); j = jj[np.arange(len(V)), first]
        off = ((Vm[j] - V) * Nm[j]).sum(1) * ok.any(1)
        off = fsmooth(np.clip(off, -0.01, 0.01), 8)        # (smoothed over the surface: neighbours move alike)
        V = V + N * (0.5 * off)[:, None]
    return V


SV = symmetrize(SV, ST)
if os.environ.get('STAGE_DUMP'): STAGES['fat_extra'] = SV.copy()
FV, FT = SV.copy(), ST.copy()
f_top4, f_w4 = top4(SW)
for V_, I_, B_, W_ in finger_tubes['skin']:
    FT = np.vstack([FT, I_ + len(FV)]); FV = np.vstack([FV, V_])
    f_top4 = np.vstack([f_top4, np.c_[B_, B_]]); f_w4 = np.vstack([f_w4, np.c_[W_, np.zeros_like(W_)]])

# ---- skin: the fat surface moved 2 mm outward (real skin is ~1.5-2 mm; it was 4) and smoothed further: same
# points and joint weights as the fat, so the two layers always move together and never poke through each other
# while posing. (Lighter smoothing on the arms.)
SV = mesh_taubin(SV + normals(SV, ST) * 0.002, ST, 40, w=keep_cleft * arm_soft)   # (the buttock cleft stays)
SV = mesh_taubin(SV, ST, 30, w=keep_cleft * arm_soft)   # one more pass over the whole body (the cleft stays)
# and an extra pass over the buttocks: the cheeks are smoothed while the cleft line between them is held
# still, so the cheeks come out round and the division stays clear
butt = (np.abs(SV[:, 0]) < 0.17) & (SV[:, 2] < heads1[idx['thigh.L']][2]) & \
       (SV[:, 1] > hip_y - 0.16) & (SV[:, 1] < hip_y + 0.08)
SV = mesh_taubin(SV, ST, 40, w=fsmooth(butt.astype(float), 10) * keep_cleft ** 2)
SV[mh_idx] = (1 - mh_blend[:, None]) * SV[mh_idx] + mh_blend[:, None] * mh_skin    # the hands' and feet's own skin
SV = mesh_taubin(SV, ST, 15, w=join_w)                  # (and the joins blended softly in the skin too)
SV = symmetrize(SV, ST)                                  # (and the skin made the same on both sides)
if os.environ.get('STAGE_DUMP'):
    STAGES.update(fat=FV.copy(), skin=SV.copy(), mus_P=mus_P.copy(), anat_P=anat_P.copy(), heads1=heads1.copy(), idx=idx)
    pickle.dump(STAGES, open(os.environ['STAGE_DUMP'], 'wb'))
s_top4, s_w4 = top4(SW)
for V_, I_, B_, W_ in finger_tubes['skin']:
    V2_ = V_ + normals(V_, I_) * 0.002
    ST = np.vstack([ST, I_ + len(SV)]); SV = np.vstack([SV, V2_])
    s_top4 = np.vstack([s_top4, np.c_[B_, B_]]); s_w4 = np.vstack([s_w4, np.c_[W_, np.zeros_like(W_)]])
print('skin verts', len(SV), '| adipose verts', len(FV), file=sys.stderr)
# (check) the skin over the left lat / teres major near the armpit: which piece shaped it, and its gap to them
lat_ids_dbg = np.where(np.isin(mid_all_ := np.concatenate([mus_id, np.full(len(mus_P) - len(mus_id), -1)]),
                               [i for i, (n, _, _) in enumerate(MUS) if re.search(r'Latissimus_dorsi|Teres_major', n) and n.endswith('l')]))[0]
sh_ = heads1[idx['upper_arm.L']]
pit = np.where((SV[:len(FSET), 0] > 0.06) & (SV[:len(FSET), 1] < sh_[1] - 0.02) & (SV[:len(FSET), 1] > sh_[1] - 0.20) & (SV[:len(FSET), 2] < 0.03))[0]
jn = lat_ids_dbg[nearest_index(SV[pit], mus_P[lat_ids_dbg], margin=0.03)]
gap_ = np.linalg.norm(SV[pit] - mus_P[jn], axis=1)
on_lat = gap_ < 0.03
print(f'armpit check: {int(on_lat.sum())} skin points over the left lat/teres; pieces ' +
      str({k: int(v) for k, v in zip(*np.unique(FSET[pit[on_lat]], return_counts=True))}) +
      f'; gap to the lat mm: median {np.median(gap_[on_lat]) * 1000:.1f}, 90% {np.percentile(gap_[on_lat], 90) * 1000:.1f}', file=sys.stderr)

# ---------------------------------------------------------------- cartilage layer (its own layer)
# A smooth, pale connective-tissue / cartilage layer to build the hands, fingers, feet, toes, ears and
# nose from. Hands and feet: it fills the gaps between tendons, bones and small muscles, sitting a few mm
# under the outer muscle surface (palm and each finger built separately so nothing webs). Ears and nose:
# it is the shape itself (outer ear and nasal cartilages).
n_mb = len(mus_P) + len(skP)
cart_label = np.full(len(anat_P), '', dtype=object)
cart_label[np.array([g.startswith(HAND_PREFIX) for g in group])] = group[np.array([g.startswith(HAND_PREFIX) for g in group])]
for sfx in ('.L', '.R'):
    cart_label[(lead == idx['foot' + sfx]) | (lead == idx['heel.02' + sfx])] = 'foot' + sfx
    cart_label[lead == idx['toe' + sfx]] = 'toe' + sfx
cart_label[n_mb + n_eyes:n_mb + n_eyes + n_ear] = 'ear'
cart_label[n_mb + n_eyes + n_ear:] = 'nose'
cart_label = cart_label.astype(str)
F_lab = cart_label[F_all[:, 0]]
c_pts = np.vstack([anat_P, *[(anat_P[F_all[:, 0]] + anat_P[F_all[:, k]]) / 2 for k in (1, 2)], anat_P[F_all].mean(1)])
c_lab = np.concatenate([cart_label, F_lab, F_lab, F_lab])


def cart_mesh(names, radius, shrink):
    names = [g for g in names if (c_lab == g).sum() > 50]
    V_, T_ = envelope([c_pts[c_lab == g] for g in names], cell=0.002, radius=radius, blur_passes=2, shrink=shrink)
    V_ = mesh_taubin(V_, T_, 8)
    V_, T_ = decimate(V_, T_, 0.003, keep_sides=False)
    u = np.unique(T_); r = np.full(len(V_), -1); r[u] = np.arange(len(u)); V_, T_ = V_[u], r[T_]
    if np.einsum('ij,ij->i', V_[T_[:, 0]], np.cross(V_[T_[:, 1]], V_[T_[:, 2]])).sum() < 0: T_ = T_[:, ::-1]
    return V_, T_


labs = np.unique(c_lab[c_lab != ''])
V0, T0 = cart_mesh([g for g in labs if g.startswith(('palm', 'foot'))], radius=2, shrink=2)   # palms and feet
V2, T2 = cart_mesh(['ear', 'nose'], radius=2, shrink=1)                                          # ears and nose
V1, T1 = V0, T0                                                                                  # (toes: tubes)
CV = np.vstack([V1, V2]); CT = np.vstack([T1, T2 + len(V1)])
# joints: copy from the nearest anatomy point on the same side (ears and nose: the head)
cart_B = np.zeros((len(CV), 2), int); cart_W = np.zeros((len(CV), 2))
for side in (1, -1):
    m = CV[:, 0] * side >= 0
    pool = np.where(anat_P[:, 0] * side >= -0.005)[0]
    near = pool[nearest_index(CV[m], anat_P[pool], margin=0.05)]
    cart_B[m], cart_W[m] = anat_B[near], anat_W[near]
head_part = np.zeros(len(CV), bool); head_part[len(V1):] = True
cart_B[head_part] = idx['spine.005']; cart_W[head_part] = [1.0, 0.0]
for V_, I_, B_, W_ in finger_tubes['cart']:       # finger cartilage: tubes on their own finger joints
    CT = np.vstack([CT, I_ + len(CV)]); CV = np.vstack([CV, V_])
    cart_B = np.vstack([cart_B, B_]); cart_W = np.vstack([cart_W, W_])
# The old hand and foot cartilage (palm, finger and toe tubes, foot) against the new MakeHuman skin: anything closer
# than 1.5 mm to it (on the hands about one point in nine poked out, by up to ~5.5 mm) is moved in to 1.5 mm under it
part_skin = mh_parts.vertex_normals(SV, ST)
for kind, sfx, sl, jn, pn, cut_b, cut_p, *_ in PARTS:
    if kind == 'head': joint, u = neck0, up_n                      # (the head: up the neck)
    else:
        joint, prox = heads1[idx[jn + sfx]], heads1[idx[pn + sfx]]
        u = (joint - prox) / np.linalg.norm(joint - prox)
    side_ = (CV[:, 0] > 0) if sfx == '.L' else (CV[:, 0] < 0) if sfx == '.R' else np.ones(len(CV), bool)
    on = np.where(((CV - joint) @ u > cut_p + 0.01) & (np.linalg.norm(CV - joint, axis=1) < (0.35 if kind == 'head' else 0.25)) & side_)[0]
    hs = mh_idx[mh_kind == kind + sfx]
    if not len(on): continue
    j = hs[nearest_index(CV[on], SV[hs], margin=0.04)]
    d = ((CV[on] - SV[j]) * part_skin[j]).sum(1)          # + outside the skin
    move = np.maximum(d + 0.0015, 0)
    CV[on] -= part_skin[j] * move[:, None]
    print(f'{kind} cartilage {sfx}: {len(on)} points; {int((d > 0).sum())} poked out of the skin (up to {max(d.max(), 0) * 1000:.1f} mm), '
          f'{int((move > 0).sum())} moved in to 1.5 mm under it; {int((d < -0.006).sum())} sit more than 6 mm under it', file=sys.stderr)
# merge into the muscle layer (muscles + cartilage)
mus_F = np.vstack([mus_F, CT + len(mus_P)]); mus_P = np.vstack([mus_P, CV])
mus_B = np.vstack([mus_B, cart_B]); mus_W = np.vstack([mus_W, cart_W])
mus_C = np.vstack([mus_C, np.tile(np.array([230, 220, 208], np.uint8), (len(CV), 1))])
print('cartilage layer verts', len(CV), '(hands/feet', len(V1), ', ears/nose', len(V2), ')', file=sys.stderr)


# put the new soles on the floor (moves everything together)
dy2 = -SV[:, 1].min()
for arr in (SV, FV, mus_P, skP, heads1): arr[:, 1] += dy2
for h in helpers: h['head'][1] = round(h['head'][1] + dy2, 5)
SN = normals(SV, ST)
tails_now = tails1 + [0, dy + dy2, 0]

# ---------------------------------------------------------------- every muscle hangs off the skeleton
# A muscle may only follow the bones it actually touches (its attachments, within 8 mm of its surface) and
# the joints between them. E.g. the biceps touches the shoulder blade and forearm, so it may follow
# shoulder blade, upper arm and forearm; a muscle touching only ribs, spine and shoulder blade can never be
# pulled by the arm. Points left with nothing to follow go to the nearest bone that muscle touches.
n_orig = len(mus_id)
near_sk = nearest_index(mus_P[:n_orig], skP, margin=0.012, fallback=False)
d_sk = np.full(n_orig, np.inf); okn = near_sk >= 0
d_sk[okn] = np.linalg.norm(mus_P[:n_orig][okn] - skP[near_sk[okn]], axis=1)


def ancestors(b):
    out = [b]
    while bone_parent[out[-1]] >= 0: out.append(bone_parent[out[-1]])
    return out


def path_between(a_, b_):
    A_, B_ = ancestors(a_), ancestors(b_)
    common = next(x for x in A_ if x in B_)
    return set(A_[:A_.index(common) + 1]) | set(B_[:B_.index(common) + 1])


fixed = 0
for i, (n, _, _) in enumerate(MUS):
    m = np.where(mus_id == i)[0]
    att = set(int(skB[near_sk[j]]) for j in m if d_sk[j] < 0.008)
    if not att: continue
    allowed = set()
    for a_ in att:
        for b_ in att: allowed |= path_between(a_, b_)
    for k, h in enumerate(helpers):
        if h.get('upper_arm') in allowed: allowed.add(NB + k)      # lats / pecs / teres keep their fibre pull
    ok = np.isin(mus_B[m], list(allowed))
    w = np.where(ok, mus_W[m], 0)
    tot = w.sum(1)
    bad = tot < 1e-6
    if bad.any():
        # nothing allowed left: follow the closest bone this muscle touches
        att_list = list(att)
        pts = mus_P[m[bad]]
        best = np.array([min(att_list, key=lambda b_: np.min(np.linalg.norm(skP[skB == b_][::5] - p, axis=1))) for p in pts])
        mus_B[m[bad]] = np.c_[best, best]; w[bad] = [1.0, 0.0]; tot[bad] = 1
    fixed += int((~ok).any(1).sum())
    mus_W[m] = w / tot[:, None]
print(f'muscle points restricted to their own bones: {fixed}', file=sys.stderr)

# The serratus lies flat on the ribcage: it follows the ribcage, and only the strip within ~3 cm of the
# shoulder blade (where it attaches along the blade's inner edge) follows the shoulder blade, fading out.
for i, (n, _, _) in enumerate(MUS):
    if 'Serratus_anterior' not in n: continue
    sfx = '.L' if n.rstrip(')')[-1] == 'l' else '.R'
    m = np.where(mus_id == i)[0]
    scap = skP[skB == idx['shoulder' + sfx]]
    d = np.linalg.norm(mus_P[m] - scap[nearest_index(mus_P[m], scap, margin=0.06)], axis=1)
    share = np.clip((0.03 - d) / 0.02, 0, 1)
    mus_B[m] = np.c_[np.full(len(m), idx['shoulder' + sfx]), np.full(len(m), idx['spine.003'])]
    mus_W[m] = np.c_[share, 1 - share]
    print(f'{n}: {int((share > 0.5).sum())} of {len(m)} points follow the shoulder blade, the rest the ribcage', file=sys.stderr)

# The heel (calcaneal / Achilles) tendon is a stiff cord from the back of the heel bone up into the calf. It is
# weighted by how far up it each point is: on the heel bone (the bottom 2 cm) it goes from the foot to the heel
# helper (which slides with the attachment without turning), then the helper's share fades in a straight line
# to none at its top, where the calf muscles (on the shin) take over. (Weighting it to the foot itself turned
# its upper part about the ankle joint and cut it forward into the leg, thinning the ankle.)
tendon_pts = {}
for i, (n, _, _) in enumerate(MUS):
    if not n.startswith('Calcaneal_tendon'): continue
    sfx = '.L' if n.rstrip(')')[-1] == 'l' else '.R'
    m = np.where(mus_id == i)[0]
    cal = skP[skN_arr == 'Calcaneus' + sfx[1].lower()]
    P_ = mus_P[m]
    bot = P_[np.argmin(np.linalg.norm(P_ - cal[nearest_index(P_, cal, margin=0.08)], axis=1) + 0.2 * P_[:, 1])]
    top = P_[np.argmax(P_[:, 1])]
    ft, sh_ = idx['foot' + sfx], idx['shin' + sfx]
    hp = NB + next(k for k, h in enumerate(helpers) if h['name'] == 'heel_pull' + sfx)
    t_ = np.clip((P_[:, 1] - bot[1] - 0.02) / max(top[1] - bot[1] - 0.02, 1e-3), 0, 1)
    low = np.clip((P_[:, 1] - bot[1]) / 0.02, 0, 1)     # on the heel bone: foot, fading into the helper
    was = np.where(mus_B[m] == ft, mus_W[m], 0).sum(1) + np.where(mus_B[m] == idx['heel.02' + sfx], mus_W[m], 0).sum(1)
    bins = np.minimum((t_ * 5).astype(int), 4)
    print(f'{n}: {len(m)} points, {(top[1] - bot[1]) * 100:.0f} cm long; foot share before, bottom to top in fifths: '
          f'{[round(float(was[bins == k].mean()), 2) if (bins == k).any() else None for k in range(5)]}', file=sys.stderr)
    lo_ = P_[:, 1] < bot[1] + 0.02
    mus_B[m] = np.where(lo_[:, None], np.c_[np.full(len(m), ft), np.full(len(m), hp)], np.c_[np.full(len(m), hp), np.full(len(m), sh_)])
    mus_W[m] = np.where(lo_[:, None], np.c_[1 - low, low], np.c_[1 - t_, t_])
    tendon_pts[sfx] = m

# ---------------------------------------------------------------- muscle layering: lats over what lies beneath
# Every lat point is paired with the closest point of the muscles underneath it, and how far above that
# muscle it sits at rest is recorded. After each pose the page pushes any lat point that sank below that
# back out along the surface beneath, so the lat wraps around the ribcage and muscles instead of passing in.
UNDER = re.compile(r'Serratus_anterior|Teres_major|Teres_minor|External_abdominal_oblique|Infraspinatus', re.I)
mid = np.concatenate([mus_id, np.full(len(mus_P) - len(mus_id), -1)])
mN = normals(mus_P, mus_F)
lay_top, lay_under, lay_gap = [], [], []
for sl in 'lr':
    lat_ids = np.where(np.isin(mid, [i for i, (n, _, _) in enumerate(MUS) if 'Latissimus_dorsi' in n and n.endswith(sl)]))[0]
    und_ids = np.where(np.isin(mid, [i for i, (n, _, _) in enumerate(MUS) if UNDER.search(n) and n.rstrip(')')[-1] == sl]))[0]
    j = und_ids[nearest_index(mus_P[lat_ids], mus_P[und_ids], margin=0.04)]
    gap = ((mus_P[lat_ids] - mus_P[j]) * mN[j]).sum(1)
    ok = (gap > 0.0005) & (np.linalg.norm(mus_P[lat_ids] - mus_P[j], axis=1) < 0.03)
    lay_top.append(lat_ids[ok]); lay_under.append(j[ok]); lay_gap.append(np.minimum(gap[ok], 0.012))
lay_top, lay_under, lay_gap = np.concatenate(lay_top), np.concatenate(lay_under), np.concatenate(lay_gap)
print('lat layering pairs', len(lay_top), file=sys.stderr)

# ...and the reverse: the muscles under the lats (serratus, teres, oblique, infraspinatus) stay blanketed
# by them. Each covered point is glued to the underside of the nearest lat point at its resting gap (fully
# when the lat is right above it, fading out over 1.5 cm toward the lat's edge), so it follows the lat
# wherever the lat goes and can't come out from under it; parts that are exposed at rest stay on the ribs.
bel_bottom, bel_cover, bel_gap, bel_sign, bel_glue = [], [], [], [], []
for sl in 'lr':
    lat_ids = np.where(np.isin(mid, [i for i, (n, _, _) in enumerate(MUS) if 'Latissimus_dorsi' in n and n.endswith(sl)]))[0]
    und_ids = np.where(np.isin(mid, [i for i, (n, _, _) in enumerate(MUS) if UNDER.search(n) and n.rstrip(')')[-1] == sl]))[0]
    j = lat_ids[nearest_index(mus_P[und_ids], mus_P[lat_ids], margin=0.04)]
    d = mus_P[j] - mus_P[und_ids]; dist = np.linalg.norm(d, axis=1)
    proj = (mN[j] * d).sum(1)
    sign = np.sign(proj); sign[sign == 0] = 1
    gap = np.abs(proj)
    ok = (dist < 0.03) & (gap > 0.0005)
    bel_bottom.append(und_ids[ok]); bel_cover.append(j[ok]); bel_gap.append(np.minimum(gap[ok], 0.012)); bel_sign.append(sign[ok])
    bel_glue.append(np.clip((0.03 - dist[ok]) / 0.015, 0, 1))
bel_bottom, bel_cover = np.concatenate(bel_bottom), np.concatenate(bel_cover)
bel_gap, bel_sign, bel_glue = np.concatenate(bel_gap), np.concatenate(bel_sign), np.concatenate(bel_glue)
print('muscles-under-lat pairs', len(bel_bottom), file=sys.stderr)

# ---------------------------------------------------------------- simple bone shapes for collisions
# A capsule around each long bone and an ellipsoid around the ribcage, measured from the real bones.
# The page pushes skin and muscle back out of these when a pose drives bone through them.
colliders = []
for sfx in ('.L', '.R'):
    for bn, child in [('upper_arm', 'forearm'), ('forearm', 'hand'), ('hand', 'f_middle.01'), ('thigh', 'shin'), ('shin', 'foot')]:
        b = idx[bn + sfx]; a_, t_ = heads1[b], heads1[idx[child + sfx]]
        pts = skP[skB == b]
        if not len(pts): continue
        r = float(np.percentile(seg_dist(pts, a_, t_), 85))
        colliders.append({'type': 'capsule', 'bone': int(b), 'a': np.round(a_, 4).tolist(), 'b': np.round(t_, 4).tolist(), 'r': round(r, 4)})
rib = skP[skB == idx['spine.003']]
cen = rib.mean(0); U_, S2, Vt2 = np.linalg.svd(rib - cen, full_matrices=False)
proj = (rib - cen) @ Vt2.T
radii = np.percentile(np.abs(proj), 95, axis=0) * 0.92
colliders.append({'type': 'ellipsoid', 'bone': idx['spine.003'], 'c': np.round(cen, 4).tolist(),
                  'axes': np.round(Vt2, 5).tolist(), 'radii': np.round(radii, 4).tolist()})
print('colliders', len(colliders), file=sys.stderr)

# the solar plexus (just below the breastbone): the point the model is pinned at
anchor = BONES['Xiphoid_process'][0] if 'Xiphoid_process' in BONES else None
if anchor is not None:
    anchor = app(anchor).mean(0); anchor[1] += dy + dy2; anchor[0] = 0.0
else:
    anchor = np.array([0.0, (heads1[idx['spine.003']][1] + heads1[idx['spine.002']][1]) / 2, 0.08])
print('anchor (solar plexus)', np.round(anchor, 3), file=sys.stderr)


# ---------------------------------------------------------------- fat and skin follow the muscles
# (done last, once the muscle weights are final). Each fat/skin point takes the joint weights of the muscle
# points right under it (6 nearest, by distance), smoothed a little over the surface, so in every pose the fat
# and skin move like the muscle layer, lat 'fibre' pulls included. It is also tied to its 4 nearest muscle
# points ('anchors'): the page copies any extra muscle adjustment (lats over serratus, bone collisions) to it.
def knn(Q, pool, k, margin=None):
    return fastgeo.knn(Q, pool, k)


n_tube = sum(len(V_) for V_, _, _, _ in finger_tubes['skin'])
n_nt = len(FV) - n_tube
NBH = NB + len(helpers)
MW = np.zeros((n_nt, NBH), np.float32)
for side in (1, -1):
    m = np.where(FV[:n_nt, 0] * side >= 0)[0]
    pool = np.where(mus_P[:, 0] * side >= -0.005)[0]
    I, D = knn(FV[m], mus_P[pool], 6)
    I = pool[I]; w = 1 / np.maximum(D, 0.002); w /= w.sum(1, keepdims=True)
    for k in range(6):
        for c in range(2): add_at(MW, (m, mus_B[I[:, k], c]), w[:, k] * mus_W[I[:, k], c])
T_nt = FT[(FT < n_nt).all(1)]
ai = np.concatenate([T_nt[:, 0], T_nt[:, 1], T_nt[:, 2], T_nt[:, 1], T_nt[:, 2], T_nt[:, 0]])
aj = np.concatenate([T_nt[:, 1], T_nt[:, 2], T_nt[:, 0], T_nt[:, 0], T_nt[:, 1], T_nt[:, 2]])
adeg = np.maximum(np.bincount(ai, minlength=n_nt), 1).astype(np.float32)[:, None]
for _ in range(3):
    acc = fastgeo.nsum(ai, aj, MW); MW = 0.5 * MW + 0.5 * acc / adeg
t4, w4 = top4(MW)
f_top4[:n_nt], f_w4[:n_nt] = t4, w4
s_top4[:n_nt], s_w4[:n_nt] = t4, w4
anc_I, anc_D = np.zeros((n_nt, 4), int), np.zeros((n_nt, 4))
for side in (1, -1):
    m = np.where(FV[:n_nt, 0] * side >= 0)[0]
    pool = np.where(mus_P[:, 0] * side >= -0.005)[0]
    I, D = knn(FV[m], mus_P[pool], 4)
    anc_I[m], anc_D[m] = pool[I], D
anc_W = 1 / np.maximum(anc_D, 0.002); anc_W /= anc_W.sum(1, keepdims=True)
print('fat/skin weights taken from the muscles; anchors', anc_I.shape, file=sys.stderr)


# ---------------------------------------------------------------- skin weights by heat diffusion
# (Baran & Popovic 2007, 'Pinocchio'.) Copying weights from the muscles leaves sharp jumps at the armpit and
# hip, which fold the skin as soon as a joint moves. Instead each skin point's weights are found by letting
# 'heat' spread over the surface from the bones: every point is warmed by the nearest bone it can see through
# the inside of the body (more strongly the closer it is), and the heat equation (L + H) w = H p spreads it,
# so weights change smoothly over several centimetres across every joint. Solved on a coarse copy of the
# surface (about 1 cm apart) and carried back to every point. Hands, feet and the face keep their weights.
from envelope import solid
HEAT_C = 30.0                # how far weights spread across a joint (larger = sharper)
heat_names = ['spine', 'spine.001', 'spine.002', 'spine.003', 'spine.004', 'spine.005']
for sfx in ('.L', '.R'):
    heat_names += [n + sfx for n in ('shoulder', 'upper_arm', 'forearm', 'hand', 'thigh', 'shin', 'foot')]
heat_b = [idx[n] for n in heat_names]


def heat_lines(b):
    kids = [c for c in range(NB) if bone_parent[c] == b and bone_names[c] in heat_names]
    if bone_names[b].startswith('hand'): kids = [idx['f_middle.01' + bone_names[b][-2:]]]
    if bone_names[b].startswith('foot'): kids = [idx['toe' + bone_names[b][-2:]]]
    ends = [heads1[c] for c in kids] or [heads1[b] + np.array([0, 0.12, 0.02])]     # (the head)
    return [(heads1[b], e) for e in ends]


# coarse copy of the surface: points merged within 1 cm (fine -> coarse map kept)
P_nt = FV[:n_nt].astype(float)
ck = np.round(P_nt / 0.010).astype(np.int64)
_, cmap = np.unique(ck, axis=0, return_inverse=True); cmap = cmap.ravel()
nc = cmap.max() + 1; cnt_c = np.bincount(cmap, minlength=nc)
CP = np.stack([np.bincount(cmap, P_nt[:, k], nc) for k in range(3)], 1) / cnt_c[:, None]
CT = cmap[T_nt]; CT = CT[(CT[:, 0] != CT[:, 1]) & (CT[:, 1] != CT[:, 2]) & (CT[:, 0] != CT[:, 2])]
ce = np.unique(np.sort(np.concatenate([CT[:, [0, 1]], CT[:, [1, 2]], CT[:, [2, 0]]]), 1), axis=0)
# inside of the body (for 'can it see the bone'): the fat surface filled in
fat_pts, _ = dense_samples(P_nt, T_nt, step=0.002)
ilo = fat_pts.min(0) - 0.02; ishape = tuple(np.ceil((fat_pts.max(0) + 0.02 - ilo) / 0.004).astype(int) + 1)
inside_v = solid(fat_pts, ilo, 0.004, ishape, 1)
def inside_at(Q):
    ii = np.clip(np.floor((Q - ilo) / 0.004).astype(int), 0, np.array(ishape) - 1)
    return inside_v[ii[..., 0], ii[..., 1], ii[..., 2]]
# distance from every coarse point to every bone, and whether the straight line to it stays inside the body
Dist = np.full((nc, len(heat_b)), np.inf); Near = np.zeros((nc, len(heat_b), 3))
for k, b in enumerate(heat_b):
    for a_, e_ in heat_lines(b):
        ab = e_ - a_; u = np.clip(((CP - a_) @ ab) / (ab @ ab), 0, 1)
        q_ = a_ + u[:, None] * ab; d = np.linalg.norm(CP - q_, axis=1)
        m = d < Dist[:, k]; Dist[m, k] = d[m]; Near[m, k] = q_[m]
visible = np.ones_like(Dist, bool)
order_ = np.argsort(Dist, 1)[:, :4]
for j in range(4):
    k = order_[:, j]; q_ = Near[np.arange(nc), k]
    t_ = np.linspace(0.15, 0.9, 6)[None, :, None]
    path = CP[:, None, :] + t_ * (q_ - CP)[:, None, :]
    visible[np.arange(nc), k] = inside_at(path).all(1)
# a point only takes heat from a bone its surface faces away from: skin wraps around its own bones, so the
# chest wall under the armpit (which faces the hanging arm) can't belong to the upper arm, and the 'can it see
# the bone' test alone fails where the arm hangs right against the ribs
CN = np.zeros_like(CP)
fn_ = np.cross(CP[CT[:, 1]] - CP[CT[:, 0]], CP[CT[:, 2]] - CP[CT[:, 0]])
for k in range(3): add_at(CN, CT[:, k], fn_)
CN /= np.maximum(np.linalg.norm(CN, axis=1, keepdims=True), 1e-12)
away = ((CP[:, None, :] - Near) * CN[:, None, :]).sum(-1) > 0.3 * Dist
Dist_raw = Dist.copy()            # (plain distances, for points no bone is left for: the inside of the mouth)
Dist = np.where(away, Dist, np.inf)
# ...and the muscle right under the skin decides between arm and trunk: skin over the deltoid, biceps and
# triceps belongs to the arm, skin over the lats, pecs and serratus to the trunk (with a 1 cm overlap at the
# border, so the switch is smooth but short). Without this the arm's weight leaks down the side of the chest
# and raising the arm pulls a web of skin from the armpit to the waist.
n_mo = len(mus_id)
arm_m = np.array([bool(ARM_MUSCLE.search(n_)) and 'foot' not in n_ for n_, _, _ in MUS])[mus_id]
arm_pts, trunk_pts = mus_P[:n_mo][arm_m], mus_P[:n_mo][~arm_m]
dA = np.linalg.norm(CP - arm_pts[nearest_index(CP, arm_pts, margin=0.08)], axis=1)
dT = np.linalg.norm(CP - trunk_pts[nearest_index(CP, trunk_pts, margin=0.08)], axis=1)
arm_cols = [k for k, n_ in enumerate(heat_names) if n_.startswith(('upper_arm', 'forearm', 'hand'))]
trunk_cols = [k for k, n_ in enumerate(heat_names) if n_.startswith('spine') and n_ not in ('spine.004', 'spine.005')]
over_arm = dA < dT + 0.01; over_trunk = dT < dA + 0.01
near_shoulder = np.min([np.linalg.norm(CP - heads1[idx['upper_arm' + sx]], axis=1) for sx in ('.L', '.R')], 0) < 0.25
Dist[np.ix_(near_shoulder & ~over_arm, arm_cols)] = np.inf
Dist[np.ix_(near_shoulder & ~over_trunk, trunk_cols)] = np.inf
Dv = np.where(visible, Dist, np.inf)
dmin = Dv.min(1); dmin = np.where(np.isfinite(dmin), dmin, Dist.min(1)); dmin = np.where(np.isfinite(dmin), dmin, Dist_raw.min(1))
Pn = (Dv <= dmin[:, None] * 1.0001).astype(float)
zero = Pn.sum(1) == 0
Pn[np.where(zero)[0], np.where(np.isfinite(Dist[zero]), Dist[zero], Dist_raw[zero] + 1e3).argmin(1)] = 1
Pn /= Pn.sum(1, keepdims=True)
h_edge = np.linalg.norm(CP[ce[:, 0]] - CP[ce[:, 1]], axis=1).mean()
Hh = HEAT_C * (h_edge / np.maximum(dmin, 0.01)) ** 2
# sharper at the shoulders: the arm-to-chest join gets a stronger pull from the bones (4x within ~10 cm of the
# shoulder joint, fading out by 18 cm), so the weights switch from arm to chest right at the armpit instead of
# spreading down the side of the chest (which folds the armpit and flares the torso when the arms go up)
SHOULDER_SHARP = 1.0   # (tried 4: no help, the leak came from which bones each point may take, fixed above)
for sfx in ('.L', '.R'):
    dj = np.linalg.norm(CP - heads1[idx['upper_arm' + sfx]], axis=1)
    t_ = np.clip((0.18 - dj) / 0.08, 0, 1)
    Hh *= 1 + (SHOULDER_SHARP - 1) * t_ * t_ * (3 - 2 * t_)
# Jacobi iterations on (L + H) w = H p (uniform graph Laplacian)
ii_ = np.concatenate([ce[:, 0], ce[:, 1]]); jj_ = np.concatenate([ce[:, 1], ce[:, 0]])
deg_c = np.bincount(ii_, minlength=nc).astype(float)
Wc = Pn.copy()
for it in range(1500):
    acc = fastgeo.nsum(ii_, jj_, Wc)
    Wn = (acc + Hh[:, None] * Pn) / (deg_c + Hh)[:, None]
    if it % 300 == 0: print(f'  heat weights: iteration {it}, change {np.abs(Wn - Wc).max():.2e}', file=sys.stderr)
    Wc = Wn
# back to every point, smoothed a little; blended out where hands, feet and the face keep their old weights
Wf = Wc[cmap]
for _ in range(3):
    acc = fastgeo.nsum(ai, aj, Wf); Wf = 0.5 * Wf + 0.5 * acc / adeg
keep_cols = [b for b, n in enumerate(bone_names) if n.startswith(('hand', 'f_', 'thumb', 'foot', 'toe', 'heel')) or n == 'spine.005']
old = np.zeros((n_nt, NBH), np.float32)
for k in range(4): add_at(old, (np.arange(n_nt), s_top4[:n_nt, k]), s_w4[:n_nt, k])
keep_amt = np.clip(old[:, keep_cols].sum(1) * 1.5 - 0.25, 0, 1)
for _ in range(4):
    acc = fastgeo.nsum(ai, aj, keep_amt); keep_amt = 0.5 * keep_amt + 0.5 * acc / adeg[:, 0]
new = np.zeros_like(old); new[:, heat_b] = Wf
blend_w = (1 - keep_amt)[:, None] * new + keep_amt[:, None] * old
# The crotch mound rides the pelvis: on the front midline the legs' share of the weights moves to the
# pelvis, fading back to the heat weights ~5 cm out to each side, so spreading or lifting a leg stretches the
# skin at the side of the mound evenly instead of dragging the mound along with one leg.
thigh_cols = [idx['thigh.L'], idx['thigh.R']]
yb, yt = m_bot + dy2, m_top + dy2
# (front-facing skin only, down to the bottom of the mound: the underside between the legs keeps its weights)
mid = np.exp(-(SV[:n_nt, 0] / 0.035) ** 2) * np.clip((normals(SV, ST)[:n_nt, 2] - 0.3) / 0.4, 0, 1) * \
      np.clip((SV[:n_nt, 1] - yb) / 0.02, 0, 1) * np.clip(((yt + 0.03) - SV[:n_nt, 1]) / 0.03, 0, 1)
print(f'crotch mound: legs share at the midline {blend_w[mid > 0.8][:, thigh_cols].sum(1).mean():.2f} before', file=sys.stderr)
moved = blend_w[:, thigh_cols] * mid[:, None]
blend_w[:, thigh_cols] -= moved; blend_w[:, idx['spine']] += moved.sum(1)
# Around the pelvis the weights are mirrored across the midline: each point is averaged with its mirror
# image (left and right bones swapped), fully on the midline and fading out ~5 cm to each side, so the two
# legs pull the middle of the crotch and buttock cleft equally and spreading them can't tear it sideways.
# (Not below the perineum, ~10 cm under the hip joints: the skin there is the inner thighs, each on its own leg; mirrored
# with the other leg down to 20 cm, the inner thigh caved in 3-5 cm when the legs came together.)
names_all = bone_names + [h['name'] for h in helpers]
swap = np.array([names_all.index(n[:-2] + ('.R' if n.endswith('.L') else '.L')) if n.endswith(('.L', '.R')) else i
                 for i, n in enumerate(names_all)])
hip_y2 = heads1[idx['thigh.L']][1]
pel = np.exp(-(SV[:n_nt, 0] / 0.03) ** 2) * np.clip((hip_y2 + 0.12 - SV[:n_nt, 1]) / 0.04, 0, 1) * \
      np.clip((SV[:n_nt, 1] - (hip_y2 - 0.12)) / 0.03, 0, 1)
twin = np.arange(n_nt)
on = np.where(pel > 0.01)[0]
twin[on] = nearest_index(SV[on] * [-1, 1, 1], SV[:n_nt], margin=0.03)   # the point on the other side
f_ = (0.5 * pel)[:, None]
blend_w = (1 - f_) * blend_w + f_ * blend_w[twin][:, swap]
print(f'pelvis weights mirrored over {int((pel > 0.1).sum())} points', file=sys.stderr)
# The skin over the side and back of the hip (gluteus medius, the top of the buttock) stays with the pelvis:
# above the hip joints the thigh's share fades out (none from 3 cm above them, all of it from 3 cm below), so
# swinging the leg out doesn't swing that skin inward over the joint and cave the hip in.
hy_ = heads1[idx['thigh.L']][1]
keep_t = np.clip((hy_ + 0.03 - SV[:n_nt, 1]) / 0.06, 0, 1)
keep_t = (keep_t * keep_t * (3 - 2 * keep_t))[:, None]
moved = blend_w[:, thigh_cols] * (1 - keep_t)
blend_w[:, thigh_cols] -= moved; blend_w[:, idx['spine']] += moved.sum(1)
print(f'hip skin: thigh share moved to the pelvis on {int((moved.sum(1) > 0.05).sum())} points', file=sys.stderr)
# The buttock cleft rides the pelvis (it runs over the sacrum and tailbone), so its two walls stay together when the legs
# move instead of each following its own leg and tearing the cleft open: within 2 cm of the midline the thighs' share
# moves to the pelvis (fading out by 4 cm), from above the hips down to the bottom of the perineum
# (10 cm below the hip joints, fading out by 13). The perineum (down to ~10 cm below the hip joints, not the inner thighs below it) gives half its thigh share
# to the pelvis, only near the midline. The
# cheeks outside that follow the legs, so a spreading leg stretches the inner cheek, as on a real body. (A band 6 cm
# wide, and the whole perineum, pulled at the thighs.) Every edge of these areas fades over 2 cm and the share is then
# smoothed over the skin: hard edges (the cleft cut off 1 cm behind the hip joints, the perineum starting 4 cm below them)
# put all-pelvis skin a few mm from skin with a fifth of it on a thigh, which folded at the bottom of the cleft.
xs_, ys_, zs_ = SV[:n_nt, 0], SV[:n_nt, 1], SV[:n_nt, 2]
hz2_ = heads1[idx['thigh.L']][2]
ease_ = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
cleft_f = ease_((0.04 - np.abs(xs_)) / 0.02) * ease_((hz2_ - zs_) / 0.02) * ease_((ys_ - (hy_ - 0.13)) / 0.03)
peri_f = 0.5 * ease_((0.025 - np.abs(xs_)) / 0.02) * ease_((hy_ - 0.01 - ys_) / 0.03) * ease_((ys_ - (hy_ - 0.12)) / 0.02) * \
         ease_((0.06 - np.abs(zs_ - hz2_)) / 0.02)
f_ = np.maximum(cleft_f, peri_f)
for _ in range(6):
    acc = fastgeo.nsum(ai, aj, f_, n_nt); f_ = 0.5 * f_ + 0.5 * acc / adeg[:, 0]
f_ = f_[:, None]
moved = blend_w[:, thigh_cols] * f_
blend_w[:, thigh_cols] -= moved; blend_w[:, idx['spine']] += moved.sum(1)
print(f'buttock cleft: thigh share moved to the pelvis on {int((moved.sum(1) > 0.05).sum())} points', file=sys.stderr)
# Below the perineum the skin between the legs is the inner thighs: each one's pelvis share moves to its own thigh (none
# 11 cm below the hip joints, all of it by 15 cm, front and back; spreading it over 7 cm folded more, not less). Blending in from the perineum and the cleft it kept up
# to a third on the pelvis, so bringing the legs together caved the top of the inner thigh in 2-4 cm.
own_f = ease_(((hy_ - 0.11) - ys_) / 0.04) * (np.abs(xs_) < 0.08)
pel_cols = [idx['spine']] + [i for i, n in enumerate(bone_names) if n == 'pelvis' or n.startswith('pelvis.')]
for sfx, side in (('.L', xs_ > 0), ('.R', xs_ <= 0)):
    f_ = (own_f * side)[:, None]
    moved = blend_w[:, pel_cols] * f_
    blend_w[:, pel_cols] -= moved; blend_w[:, idx['thigh' + sfx]] += moved.sum(1)
print(f'inner thighs: pelvis share moved to their own thigh on {int((own_f > 0.05).sum())} points', file=sys.stderr)
# The change from the pelvis's weights (upper buttock) to the thigh's (lower buttock, the fold under it) is spread over a
# taller band: the weights on the seat behind the hips are smoothed over the surface, so spreading the legs stretches
# it evenly instead of making a ledge at the fold.
seat_ = ((zs_ < hz2_ - 0.02) & (ys_ > hy_ - 0.09) & (ys_ < hy_ + 0.06) & (np.abs(xs_) < 0.16)).astype(float)   # (not the thighs)
seat_w = np.zeros(n_nt); seat_w[:] = seat_
for _ in range(4):
    acc = fastgeo.nsum(ai, aj, seat_w, n_nt); seat_w = 0.5 * seat_w + 0.5 * acc / adeg[:, 0]
for _ in range(20):
    acc = fastgeo.nsum(ai, aj, blend_w)
    blend_w = blend_w + 0.5 * seat_w[:, None] * (acc / adeg - blend_w)
print(f'seat weights smoothed over {int((seat_w > 0.5).sum())} points', file=sys.stderr)
# The skin over the lats near the armpit moves with the lat under it (its top follows the arm's pull): it takes
# the lat's weights, fully within 1.5 cm of the lat and fading out by 3 cm (smoothed over the skin; none within
# 10 cm of the shoulder joint, where those lat points ride the arm's pull and the skin sagged), so raising
# the arm stretches the skin with the lat instead of leaving it behind lifting off the muscle.
lat_all = np.where(np.isin(mus_id, [i for i, (n, _, _) in enumerate(MUS) if re.search(r'Latissimus_dorsi|Teres_major', n)]))[0]
near_sh = np.min([np.linalg.norm(SV[:n_nt] - heads1[idx['upper_arm' + sx]], axis=1) for sx in ('.L', '.R')], 0)
cand = np.where(near_sh < 0.30)[0]
jl = lat_all[nearest_index(SV[cand], mus_P[lat_all], margin=0.04)]
f_lat = np.zeros(n_nt)
f_lat[cand] = np.clip((0.03 - np.linalg.norm(SV[cand] - mus_P[jl], axis=1)) / 0.015, 0, 1) * np.clip((0.30 - near_sh[cand]) / 0.05, 0, 1) * \
              np.clip((near_sh[cand] - 0.10) / 0.06, 0, 1)    # (not right under the shoulder: there it sagged into a pouch)
for _ in range(6):
    acc = fastgeo.nsum(ai, aj, f_lat, n_nt); f_lat = 0.5 * f_lat + 0.5 * acc / adeg[:, 0]
f_lat[np.setdiff1d(np.arange(n_nt), cand)] = 0
latW = np.zeros((n_nt, blend_w.shape[1]), np.float32)
latW[cand] = dense(mus_B[jl], mus_W[jl])[:, :blend_w.shape[1]]
blend_w = (1 - f_lat[:, None]) * blend_w + f_lat[:, None] * latW
print(f'skin over the lats: {int((f_lat > 0.5).sum())} points take the lat weights', file=sys.stderr)
# The skin over the back of the ankle moves with the heel tendon under it: it takes the tendon's weights, fully
# within 1.5 cm of the tendon and fading out by 3 cm (smoothed over the skin)
ten_all = np.concatenate(list(tendon_pts.values()))
lo_, hi_ = mus_P[ten_all].min(0) - 0.04, mus_P[ten_all].max(0) + 0.04
box = np.where(np.all((SV[:n_nt] >= lo_) & (SV[:n_nt] <= hi_), 1))[0]
cand = box[np.linalg.norm(SV[box] - mus_P[ten_all][nearest_index(SV[box], mus_P[ten_all], margin=0.04)], axis=1) < 0.04]
jt = ten_all[nearest_index(SV[cand], mus_P[ten_all], margin=0.04)]
f_ten = np.zeros(n_nt)
f_ten[cand] = np.clip((0.03 - np.linalg.norm(SV[cand] - mus_P[jt], axis=1)) / 0.015, 0, 1)
for _ in range(4):
    acc = fastgeo.nsum(ai, aj, f_ten, n_nt); f_ten = 0.5 * f_ten + 0.5 * acc / adeg[:, 0]
f_ten[np.setdiff1d(np.arange(n_nt), cand)] = 0
tenW = np.zeros((n_nt, blend_w.shape[1]), np.float32)
tenW[cand] = dense(mus_B[jt], mus_W[jt])[:, :blend_w.shape[1]]
blend_w = (1 - f_ten[:, None]) * blend_w + f_ten[:, None] * tenW
print(f'skin over the heel tendon: {int((f_ten > 0.5).sum())} points take its weights', file=sys.stderr)
# The front of the chest below the collarbone rides the ribcage, not the collarbone: the collarbone's heat line
# starts near the breastbone, so the skin over the pecs took ~60% of its weights from it, and raising the arms
# (which tilts the collarbone and shoulder blade) slid that skin up into a ridge under the chin. On front-facing
# skin more than 1.5 cm below the collarbone the collarbone's share moves to the ribcage; it is kept within
# 8 cm of the shoulder joint (fading out by 14 cm), so the shoulder and armpit still move with it.
SN_nt = normals(SV, ST)[:n_nt]
for sfx in ('.L', '.R'):
    sb, sh_j = idx['shoulder' + sfx], heads1[idx['upper_arm' + sfx]]
    a_ = heads1[sb]
    side = (SV[:n_nt, 0] > 0) if sfx == '.L' else (SV[:n_nt, 0] < 0)
    u_ = np.clip((SV[:n_nt, 0] - a_[0]) / (sh_j[0] - a_[0]), 0, 1)
    clav_y = a_[1] + u_ * (sh_j[1] - a_[1])                           # collarbone height above each point
    below = np.clip((clav_y - 0.015 - SV[:n_nt, 1]) / 0.02, 0, 1)
    front = np.clip((SN_nt[:, 2] - 0.2) / 0.3, 0, 1)
    far_ = np.clip((np.linalg.norm(SV[:n_nt] - sh_j, axis=1) - 0.08) / 0.06, 0, 1)
    f_ = side * below * front * far_
    moved = blend_w[:, sb] * f_
    blend_w[:, sb] -= moved; blend_w[:, idx['spine.003']] += moved
    print(f'chest {sfx}: collarbone share moved to the ribcage on {int((moved > 0.05).sum())} skin points', file=sys.stderr)
# The top of the shoulder and the slope from the neck to it: the weights there changed from the shoulder blade to the
# upper arm (and to the neck) within ~3 cm, so raising the arm or lifting the shoulder squeezed that skin into folds
# instead of bunching it evenly. They are smoothed over the skin above the armpit (from 2 cm below the shoulder joint
# up; behind it from 8 cm below, the back fold of the armpit, which folded when the arm was pulled back), within 15 cm of the shoulder joint or 6 cm of the line from the neck to it; not the armpit, where the arm-to-
# chest change is kept sharp on purpose, and fading out toward the midline.
SHOULDER_SMOOTH = int(os.environ.get('SHOULDER_SMOOTH', 30))   # (rounds)
sh_w = np.zeros(n_nt)
for sfx in ('.L', '.R'):
    sh_j, nk = heads1[idx['upper_arm' + sfx]], heads1[idx['spine.004']]
    u_ = np.clip(((SV[:n_nt] - nk) @ (sh_j - nk)) / ((sh_j - nk) @ (sh_j - nk)), 0, 1)
    d_line = np.linalg.norm(SV[:n_nt] - (nk + u_[:, None] * (sh_j - nk)), axis=1)
    near = np.maximum(np.clip((0.15 - np.linalg.norm(SV[:n_nt] - sh_j, axis=1)) / 0.04, 0, 1), np.clip((0.06 - d_line) / 0.03, 0, 1))
    up = np.clip((SV[:n_nt, 1] - (sh_j[1] - 0.02)) / 0.03, 0, 1)
    back_ = np.clip(((sh_j[2] - 0.02) - SV[:n_nt, 2]) / 0.02, 0, 1)                       # (behind the shoulder joint)
    up = np.maximum(up, back_ * np.clip((SV[:n_nt, 1] - (sh_j[1] - 0.08)) / 0.03, 0, 1))  # (there down to 8 cm below it)
    side = np.clip((np.abs(SV[:n_nt, 0]) - 0.02) / 0.03, 0, 1) * ((SV[:n_nt, 0] > 0) if sfx == '.L' else (SV[:n_nt, 0] < 0))
    sh_w = np.maximum(sh_w, near * up * side)
for _ in range(4):
    acc = fastgeo.nsum(ai, aj, sh_w, n_nt); sh_w = 0.5 * sh_w + 0.5 * acc / adeg[:, 0]
for _ in range(SHOULDER_SMOOTH):
    acc = fastgeo.nsum(ai, aj, blend_w)
    blend_w = blend_w + 0.5 * sh_w[:, None] * (acc / adeg - blend_w)
print(f'shoulder tops: weights smoothed over {int((sh_w > 0.5).sum())} skin points', file=sys.stderr)
if not np.isfinite(blend_w).all(): print(f'WARNING: {int((~np.isfinite(blend_w)).any(1).sum())} skin points have broken weights', file=sys.stderr)
t4, w4 = top4(blend_w)
f_top4[:n_nt], f_w4[:n_nt] = t4, w4
s_top4[:n_nt], s_w4[:n_nt] = t4, w4
print(f'heat weights: {nc} coarse points, {len(heat_b)} bones, {int((keep_amt > 0.5).sum())} points keep their old weights',
      file=sys.stderr)


# ---------------------------------------------------------------- forearm twist: radius bone and twist knots
# Turning the palm (pronation / supination) happens along the whole forearm: the radius rolls around the
# ulna, about the axis from the radial head (at the elbow) to the head of the ulna (at the wrist).
# Each forearm gets a 'radius' bone that turns rigidly about that axis, and four twist knots along the
# forearm that take 25%, 50%, 75% and 100% of the turn. Forearm skin, fat and ulna-side muscles are spread
# over the knots by how far down the forearm they are; the radius and its muscles follow the radius bone.
skN = np.array(skN)
RADIAL = re.compile(r'Pronator|Brachioradialis|Supinator|carpi_radialis|Abductor_pollicis_longus|'
                    r'Extensor_pollicis|Flexor_pollicis_longus|Radial_head_of_flexor_digitorum', re.I)
twist_bones, twist_axes = [], {}
NBX = NBH
mid_all = np.concatenate([mus_id, np.full(len(mus_P) - len(mus_id), -1)])


def end_center(P, toward, mm=0.015):
    d = np.linalg.norm(P - toward, axis=1)
    return P[d < d.min() + mm].mean(0)


def dense_w(B, W, n_cols):
    D = np.zeros((len(B), n_cols), np.float32)
    for k in range(B.shape[1]): add_at(D, (np.arange(len(B)), B[:, k]), W[:, k])
    return D


KNOTS = [0.25, 0.5, 0.75, 1.0]
THIGH_KNOTS = [0.0, 0.25, 0.5, 0.75]
NCOL = NBH + 2 * (len(KNOTS) + 2) + 2 * len(THIGH_KNOTS)      # all rig bones once the twist and carpal bones are added
wrist_info = {}
for sfx, sl in (('.L', 'l'), ('.R', 'r')):
    fa, hd = idx['forearm' + sfx], idx['hand' + sfx]
    elbow, wrist = heads1[fa], heads1[hd]
    rad = skP[(skN == 'Radius' + sl)].astype(float); uln = skP[(skN == 'Ulna' + sl)].astype(float)
    rhead, uhead = end_center(rad, elbow), end_center(uln, wrist)
    twist_axes[sfx[1]] = {'point': np.round(rhead, 5).tolist(), 'dir': np.round((uhead - rhead) / np.linalg.norm(uhead - rhead), 5).tolist()}
    knot_ids = [fa]
    for k_, f_ in enumerate(KNOTS):
        twist_bones.append({'name': f'forearm_tw{k_ + 1}{sfx}', 'parent': fa, 'head': np.round(elbow + f_ * (wrist - elbow), 5).tolist(), 'frac': f_})
        knot_ids.append(NBX); NBX += 1
    twist_bones.append({'name': 'radius' + sfx, 'parent': fa, 'head': np.round(rhead, 5).tolist(), 'frac': 1.0})
    rad_b = NBX; NBX += 1
    twist_axes[sfx[1]]['knots'] = knot_ids[1:]; twist_axes[sfx[1]]['radius'] = rad_b
    skB[skN == 'Radius' + sl] = rad_b
    # The wrist bends at two joints: the first row of carpal bones turns on the end of the radius, and the
    # hand turns on that row about the head of the capitate (each takes part of the bend; the page splits it).
    # The 'carpal' bone carries the first row; it turns with the forearm twist like the hand.
    cap = target('capitate', 1 if sfx == '.L' else -1)
    twist_bones.append({'name': 'carpal' + sfx, 'parent': fa, 'head': np.round(wrist, 5).tolist(), 'frac': 1.0,
                        'mid': np.round(cap, 5).tolist()})
    car_b = NBX; NBX += 1
    row1 = np.array([bool(re.match(r'(Scaphoid|Lunate|Triquetrum|Pisiform)_bone' + sl + '$', n_)) for n_ in skN])
    skB[row1] = car_b
    wrist_info[sfx] = {'p': wrist, 'u': (wrist - elbow) / np.linalg.norm(wrist - elbow), 'hand': hd, 'carpal': car_b,
                       'fore': knot_ids + [rad_b], 'last': knot_ids[-1]}
    print(f'wrist {sfx}: capitate head {np.linalg.norm(cap - wrist) * 100:.1f} cm from the radius end, '
          f'{int(row1.sum())} first-row carpal points', file=sys.stderr)
    seg = wrist - elbow


    def spread(P, D, radial=None):
        """Move each point's forearm weight onto the two twist knots either side of it; 'radial' (0-1 per point)
        is the share that goes to the radius instead."""
        s_ = np.clip(((P - elbow) @ seg) / (seg @ seg), 0, 1)
        wf = D[:, fa].copy(); D[:, fa] = 0
        if radial is not None:
            D[:, rad_b] += wf * radial; wf = wf * (1 - radial)
        pos_ = s_ * 4; k0 = np.minimum(pos_.astype(int), 3); t_ = pos_ - k0
        kid = np.array(knot_ids)
        add_at(D, (np.arange(len(P)), kid[k0]), wf * (1 - t_))
        add_at(D, (np.arange(len(P)), kid[k0 + 1]), wf * t_)
        return D


    side = lambda P: (P[:, 0] > 0) if sfx == '.L' else (P[:, 0] < 0)
    # muscles: radial muscles near the radius follow the radius, the rest the knots
    m = side(mus_P) & np.isin(mus_B, [fa]).any(1)
    radial_mus = np.array([bool(RADIAL.search(MUS[i][0])) if i >= 0 else False for i in mid_all[m]])
    # each muscle point is shared between the radius and the twist knots by how much nearer it is to the
    # radius than to the ulna (deep muscles between the two bones get some of each); named radial muscles
    # lean further toward the radius
    dR = np.linalg.norm(mus_P[m] - rad[nearest_index(mus_P[m], rad, margin=0.06)], axis=1)
    dU = np.linalg.norm(mus_P[m] - uln[nearest_index(mus_P[m], uln, margin=0.06)], axis=1)
    share_r = dU / np.maximum(dR + dU, 1e-9)
    share_r = np.clip((share_r - 0.35) / 0.3 + 0.25 * radial_mus, 0, 1)
    share_r = share_r * share_r * (3 - 2 * share_r)
    mus_D = dense_w(mus_B[m], mus_W[m], NCOL)
    mus_twist = (m, spread(mus_P[m], mus_D, share_r))
    print(f'forearm twist {sfx}: axis {np.round(uhead - rhead, 3)}, radius share avg {share_r.mean():.2f} over {int(m.sum())} muscle points',
          file=sys.stderr)
    if sfx == '.L': mus_twist_L = mus_twist
    # skin and fat: spread over the knots by how far down the forearm they are (a straight elbow-to-wrist ramp;
    # following the muscles' own twist instead put too much twist near the elbow and dented it)
    for name in ('skin', 'fat'):
        T4, W4 = (s_top4, s_w4) if name == 'skin' else (f_top4, f_w4)
        P = SV if name == 'skin' else FV
        m = side(P) & np.isin(T4, [fa]).any(1)
        D = spread(P[m], dense_w(T4[m], W4[m], NCOL))
        t4 = np.argsort(-D, 1)[:, :4]; w4 = np.take_along_axis(D, t4, 1); w4 /= w4.sum(1, keepdims=True)
        T4[m], W4[m] = t4, w4

# ---- thigh twist: turning the leg in or out (thigh twist up to 30 degrees) is spread down the thigh the
# same way, so it doesn't all land at the hip and wring the skin there. Unlike the forearm the femur turns as
# one piece, so the thigh bone keeps the whole turn (the knee and shin follow it) and four knots from the hip
# joint down hold back 100%, 75%, 50% and 25% of it. Thigh skin, fat and muscles are spread over the knots by
# how far down the thigh they are; the femur itself stays on the thigh.
thigh_twist, side_knots = [], []
for sfx in ('.L', '.R'):
    th, sh_ = idx['thigh' + sfx], idx['shin' + sfx]
    hip, knee = heads1[th], heads1[sh_]
    knot_ids = []
    for f_ in THIGH_KNOTS:
        twist_bones.append({'name': f'thigh_tw{len(knot_ids)}{sfx}', 'parent': th, 'head': np.round(hip + f_ * (knee - hip), 5).tolist(), 'frac': f_})
        knot_ids.append(NBX); NBX += 1
    kid = np.array(knot_ids + [th])                       # twist carried: 0, 25, 50, 75, 100%
    seg = knee - hip


    def spread_thigh(P, D):
        s_ = np.clip(((P - hip) @ seg) / (seg @ seg), 0, 1)
        wt = D[:, th].copy(); D[:, th] = 0
        pos_ = s_ * 4; k0 = np.minimum(pos_.astype(int), 3); t_ = pos_ - k0
        add_at(D, (np.arange(len(P)), kid[k0]), wt * (1 - t_))
        add_at(D, (np.arange(len(P)), kid[k0 + 1]), wt * t_)
        return D


    side_knots.append(kid)
    for name in ('skin', 'fat'):
        T4, W4 = (s_top4, s_w4) if name == 'skin' else (f_top4, f_w4)
        P = SV if name == 'skin' else FV
        m = np.isin(T4, [th]).any(1)
        D = spread_thigh(P[m], dense_w(T4[m], W4[m], NCOL))
        # (a point needing more than four joints, the skin between the legs: pelvis plus two knots from each thigh, keeps
        # one knot per thigh, the one carrying more of it. Cutting the smallest instead dropped the left thigh's on one
        # point and the right's on its neighbour, so spreading the legs pushed points 2 mm apart opposite ways and folded.)
        over = (D > 1e-6).sum(1) > 4
        for k_ in side_knots:
            Do = D[over][:, k_]; keep = k_[Do.argmax(1)]; tot = Do.sum(1)
            Do[:] = 0; D[np.where(over)[0][:, None], k_[None, :]] = Do
            D[np.where(over)[0], keep] = tot
        t4 = np.argsort(-D, 1)[:, :4]; w4 = np.take_along_axis(D, t4, 1); w4 /= w4.sum(1, keepdims=True)
        T4[m], W4[m] = t4, w4
    m = np.isin(mus_B, [th]).any(1)
    thigh_twist.append((m, spread_thigh(mus_P[m], dense_w(mus_B[m], mus_W[m], NCOL))))
    print(f'thigh twist {sfx}: {int(m.sum())} muscle points spread over the knots', file=sys.stderr)



# ---- across the wrist: weights by position along the forearm, from the forearm (2 cm or more before the end
# of the radius) through the first carpal row (all of it right at the joint) to the hand (2 cm past it). The
# base mesh spread the hand's share 5 cm up the forearm, so muscle there already swung with the hand into the
# ends of the radius and ulna. Only the forearm / carpal / hand share is moved (thumb and fingers keep theirs).
def wrist_ramp(P, D, width):
    for sfx, w_ in wrist_info.items():
        s_ = (P - w_['p']) @ w_['u']
        r_ = np.linalg.norm(P - w_['p'] - s_[:, None] * w_['u'], axis=1)
        # (reaching 12 cm up the forearm: the heat-diffusion weights leave the hand 10-45% of the skin 4-9 cm up it,
        # and a band moving with the hand above a band that doesn't made a ledge across the forearm when the wrist bent)
        m = (r_ < 0.06) & (s_ > -(width + 0.10)) & (s_ < width + 0.04) & ((P[:, 0] > 0) if sfx == '.L' else (P[:, 0] < 0))
        fore = w_['fore']; sm = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
        Dm = D[m]
        f_sum = Dm[:, fore].sum(1)
        tot = f_sum + Dm[:, w_['carpal']] + Dm[:, w_['hand']]
        h = sm(s_[m] / width); f = sm(-s_[m] / width); c = 1 - h - f
        Dm[:, fore] *= (tot * f / np.maximum(f_sum, 1e-9))[:, None]
        Dm[:, w_['last']] += np.where(f_sum < 1e-9, tot * f, 0)
        Dm[:, w_['carpal']] = tot * c; Dm[:, w_['hand']] = tot * h
        D[m] = Dm
    return D


# muscles get 4 joints per point (the twist knots add one)
mus_D_all = dense_w(mus_B, mus_W, NCOL)
for mt in (mus_twist_L, mus_twist, *thigh_twist):
    mus_D_all[mt[0]] = mt[1]
mus_D_all = wrist_ramp(mus_P, mus_D_all, 0.015)
for name in ('skin', 'fat'):
    T4, W4 = (s_top4, s_w4) if name == 'skin' else (f_top4, f_w4)
    P = SV if name == 'skin' else FV
    D = wrist_ramp(P, dense_w(T4, W4, NCOL), 0.025)        # (skin: over 2.5 cm each side, so a bent wrist creases softly)
    t4 = np.argsort(-D, 1)[:, :4]; w4 = np.take_along_axis(D, t4, 1); w4 /= w4.sum(1, keepdims=True)
    T4[:], W4[:] = t4, w4
m_top4 = np.argsort(-mus_D_all, 1)[:, :4]
m_w4 = np.take_along_axis(mus_D_all, m_top4, 1); m_w4 /= np.maximum(m_w4.sum(1, keepdims=True), 1e-9)
del mus_D_all
# Skin and fat drop the crumbs of weight the heat diffusion leaves on far bones: 1% comes off every weight (anything under
# it goes to 0) and the rest is scaled back up to 100%. 0.4% of the upper arm on the lower back moved it 2 mm sideways when
# the arm rose 45 degrees while the point next to it, without that crumb, stayed put, which folded the skin there.
# (Taking off a fixed 1% rather than cutting at 1% keeps the weights continuous: no new steps between neighbours.)
for W4 in (s_w4, f_w4):
    W4[:] = np.maximum(W4 - 0.01, 0); W4 /= np.maximum(W4.sum(1, keepdims=True), 1e-9)


# ---------------------------------------------------------------- optimized centers of rotation
# Le & Hodgins 2016: each point gets a 'center of rotation', the similarity-weighted average of the surface
# whose joint weights are like its own. The page turns the point about that center with the blended
# rotation (as a quaternion), which keeps volume at twists and bends instead of collapsing like plain blending.
# Similarity only depends on, for each pair of joints (j, k), how a point's weight is split between them, so
# triangles are binned per pair by (share a = w_j + w_k, split r = w_j / a) and points compare with the bins.
def cor_centers(P, T, top, w, sigma=0.1, nr=40, na=10):
    n = len(P)
    W = np.zeros((n, NCOL), np.float32)
    for k in range(top.shape[1]): add_at(W, (np.arange(n), top[:, k]), w[:, k])
    a_, b_, c_ = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b_ - a_, c_ - a_), axis=1)
    cen = (a_ + b_ + c_) / 3
    Wt = (W[T[:, 0]] + W[T[:, 1]] + W[T[:, 2]]) / 3
    num = np.zeros((n, 3)); den = np.zeros(n)
    eps = 1e-3
    used = W > eps
    pairs = set()
    # joint pairs present at points
    nz = [np.where(r)[0] for r in used[(used.sum(1) >= 2)]]
    for r in nz:
        for i_ in range(len(r)):
            for j_ in range(i_ + 1, len(r)): pairs.add((r[i_], r[j_]))
    rb = (np.arange(nr) + 0.5) / nr; ab = (np.arange(na) + 0.5) / na
    for (j, k) in pairs:
        tm = (Wt[:, j] > eps) & (Wt[:, k] > eps)
        pm = used[:, j] & used[:, k]
        if not tm.any() or not pm.any(): continue
        av = Wt[tm, j] + Wt[tm, k]; rv = Wt[tm, j] / av
        g = area[tm] * av ** 2 * rv * (1 - rv)
        bi = np.minimum((rv * nr).astype(int), nr - 1) * na + np.minimum((av * na).astype(int), na - 1)
        S = np.bincount(bi, g, nr * na)
        SC = np.stack([np.bincount(bi, g * cen[tm, c], nr * na) for c in range(3)], 1)
        keep = S > 0
        S, SC = S[keep], SC[keep]
        brv = np.repeat(rb, na)[keep]; bav = np.tile(ab, nr)[keep]
        pi = np.where(pm)[0]
        ap = W[pi, j] + W[pi, k]; rp = W[pi, j] / ap
        fp = 2 * ap ** 2 * rp * (1 - rp)
        for c0 in range(0, len(pi), 4096):
            sl_ = slice(c0, c0 + 4096)
            E = np.exp(-((ap[sl_, None] * bav[None]) ** 2) * (rp[sl_, None] - brv[None]) ** 2 / sigma ** 2)
            num[pi[sl_]] += fp[sl_, None] * (E @ SC)
            den[pi[sl_]] += fp[sl_] * (E @ S)
    out = P.copy()                       # no center (one joint only): plain blending, same result
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok, None]
    # smooth the centers over the surface (between points that have one), so neighbouring points turn
    # about nearby centers and the surface can't tear into slivers where the center would jump
    ei = np.concatenate([T[:, 0], T[:, 1], T[:, 2], T[:, 1], T[:, 2], T[:, 0]])
    ej = np.concatenate([T[:, 1], T[:, 2], T[:, 0], T[:, 0], T[:, 1], T[:, 2]])
    okf = ok.astype(float)
    for _ in range(15):
        acc = np.zeros_like(out); add_at(acc, ei, out[ej] * okf[ej, None])
        cnt = np.bincount(ei, okf[ej], n)
        m = ok & (cnt > 0)
        out[m] = 0.5 * out[m] + 0.5 * acc[m] / cnt[m, None]
    print(f'  centers of rotation: {int(ok.sum())} of {n} points, {len(pairs)} joint pairs', file=sys.stderr)
    return out


print('optimized centers of rotation:', file=sys.stderr)
cor_skin = cor_centers(SV.astype(float), ST, s_top4, s_w4)
cor_fat = cor_centers(FV.astype(float), FT, f_top4, f_w4)
cor_mus = cor_centers(mus_P.astype(float), mus_F, m_top4, m_w4)


# ---------------------------------------------------------------- muscle order
# For each muscle point: the point of another muscle right beneath it, and the gap between them at rest.
# The page keeps that order (and at least that gap, up to 1 cm) in every pose, so muscles can't pass
# through each other. Only the outer muscle of each pair is moved.
mid_m = np.concatenate([mus_id, np.full(len(mus_P) - len(mus_id), -1)])
mNo = normals(mus_P, mus_F)
I8, D8 = knn(mus_P, mus_P, 8, margin=0.03)
other = (mid_m[I8] != mid_m[:, None]) & (mid_m[I8] >= 0)
first = np.where(other.any(1), other.argmax(1), -1)
has = (first >= 0) & (mid_m >= 0)
o_top = np.where(has)[0]; o_und = I8[o_top, first[has]]
o_gap = ((mus_P[o_top] - mus_P[o_und]) * mNo[o_und]).sum(1)
o_dist = np.linalg.norm(mus_P[o_top] - mus_P[o_und], axis=1)
# only real layers: both surfaces face the same way and the lower point is straight beneath (not a
# neighbouring muscle lying side by side)
stacked = ((mNo[o_top] * mNo[o_und]).sum(1) > 0.6) & (o_gap > 0.75 * o_dist)
keep = (o_gap > 0.0003) & (o_dist < 0.015) & stacked
o_top, o_und, o_gap = o_top[keep], o_und[keep], np.minimum(o_gap[keep], 0.01)
print(f'muscle order: {len(o_top)} pairs (muscle point over another muscle)', file=sys.stderr)


# ---------------------------------------------------------------- bone distance grids
# For every rig bone, a 3D grid of signed distances to its real bones (negative inside bone), in the rig
# bone's rest space (= world space at rest, since no bone is rotated at rest). Used by the page to count
# and fix skin, fat and muscle points that a pose has driven into bone.
from envelope import solid


def edt2(inside):
    """Squared distance (in cells) from every cell to the nearest True cell. Exact (scipy's linear-time
    Euclidean distance transform; it replaced a brute-force version here that took 10 s)."""
    from scipy.ndimage import distance_transform_edt
    if not inside.any(): return np.full(inside.shape, 1e9)
    return distance_transform_edt(~inside).astype(np.float64) ** 2


SDF_CELL, SDF_PAD, SDF_UNIT = 0.0025, 0.02, 0.00025      # 2.5 mm cells, 2 cm margin, 0.25 mm per step
sdf_grids, sdf_bytes = [], []
for b in sorted(set(skB.tolist())):
    tri = skI[(skB[skI[:, 0]] == b)]
    if len(tri) < 4: continue
    pts, _ = dense_samples(skP.astype(float), tri, step=0.0012)
    lo = pts.min(0) - SDF_PAD; hi = pts.max(0) + SDF_PAD
    shape = tuple(np.ceil((hi - lo) / SDF_CELL).astype(int) + 1)
    ins = solid(pts, lo, SDF_CELL, shape, 0)
    d_out = np.sqrt(edt2(ins)); d_in = np.sqrt(edt2(~ins))
    sd = np.where(ins, -(d_in - 0.5), d_out - 0.5) * SDF_CELL   # half a cell: distance to the boundary
    q = np.clip(np.round(sd / SDF_UNIT), -127, 127).astype(np.int8)
    sdf_grids.append({'bone': int(b), 'min': np.round(lo, 5).tolist(), 'shape': [int(v) for v in shape], 'offset': sum(len(x) for x in sdf_bytes)})
    sdf_bytes.append(q.tobytes(order='C'))
sdf_blob = gzip.compress(b''.join(sdf_bytes), 9)
print(f'bone distance grids: {len(sdf_grids)} bones, {sum(len(x) for x in sdf_bytes) / 1e6:.1f} M cells, '
      f'{len(sdf_blob) / 1e6:.2f} MB gzipped', file=sys.stderr)


# ---------------------------------------------------------------- body segment shapes (implicit skinning)
# The finished skin filled in as one solid body, split into segments by the nearest bone: chest, pelvis and each
# upper arm, forearm, thigh and shin (hands, feet and the head are segments too, but get no shape: they just
# cut the limbs off cleanly at the wrists, ankles and neck). Each segment is turned into a smooth signed-distance
# shape in its main bone's rest space. The page moves skin and fat that a pose drives inside a segment back out
# to the surface, and where two segments meet (armpit, groin, inner thighs, elbow, knee) the skin stops on the
# neighbouring segment's surface and makes a crease instead of passing through.
from envelope import blur as vol_blur
SEG_CELL, SEG_PAD, SEG_UNIT = 0.004, 0.03, 0.0005
seg_groups = {'chest': (['spine.002', 'spine.003', 'shoulder.L', 'shoulder.R'], 'spine.003'),
              'pelvis': (['spine', 'spine.001', 'pelvis.L', 'pelvis.R'], 'spine'),
              'head': (['spine.004', 'spine.005'], None)}
for sfx in ('.L', '.R'):
    seg_groups['upper_arm' + sfx] = (['upper_arm' + sfx], 'upper_arm' + sfx)
    seg_groups['forearm' + sfx] = (['forearm' + sfx], 'forearm' + sfx)
    seg_groups['hand' + sfx] = (['hand' + sfx] + [n for n in bone_names if n.endswith(sfx) and n.startswith(('f_', 'thumb'))], None)
    seg_groups['thigh' + sfx] = (['thigh' + sfx], 'thigh' + sfx)
    seg_groups['shin' + sfx] = (['shin' + sfx], 'shin' + sfx)
    seg_groups['foot' + sfx] = (['foot' + sfx, 'toe' + sfx, 'heel.02' + sfx], None)
# each rig bone as a line from its head to its children's heads (or a short way along itself at the end of a chain)
def bone_lines(names):
    L_ = []
    for n in names:
        b = idx[n]; kids = [c for c in range(NB) if bone_parent[c] == b and not bone_names[c].startswith('pelvis')]
        ends = [heads1[c] for c in kids] or [heads1[b] + (heads1[b] - heads1[bone_parent[b]]) * 0.5]
        L_ += [(heads1[b], e) for e in ends]
    return L_


skin_pts, _ = dense_samples(SV.astype(float), ST, step=0.0015)
blo = skin_pts.min(0) - SEG_PAD; bhi = skin_pts.max(0) + SEG_PAD
bshape = tuple(np.ceil((bhi - blo) / SEG_CELL).astype(int) + 1)
body_in = solid(skin_pts, blo, SEG_CELL, bshape, 0)
cells = np.argwhere(body_in); cpos = blo + (cells + 0.5) * SEG_CELL
gnames_ = list(seg_groups)
best_d = np.full(len(cells), np.inf); best_g = np.zeros(len(cells), int)
for gi, g in enumerate(gnames_):
    for a_, b_ in bone_lines(seg_groups[g][0]):
        for c0 in range(0, len(cells), 500_000):
            d = seg_dist(cpos[c0:c0 + 500_000], a_, b_)
            sel = d < best_d[c0:c0 + 500_000]
            best_d[c0:c0 + 500_000][sel] = d[sel]; best_g[c0:c0 + 500_000][sel] = gi
seg_grids, seg_bytes = [], []
for gi, g in enumerate(gnames_):
    main = seg_groups[g][1]
    if main is None: continue
    own = cells[best_g == gi]
    if not len(own): continue
    lo_c = np.maximum(own.min(0) - int(SEG_PAD / SEG_CELL), 0); hi_c = np.minimum(own.max(0) + int(SEG_PAD / SEG_CELL) + 1, bshape)
    ins = np.zeros(tuple(hi_c - lo_c), bool); o_ = own - lo_c; ins[tuple(o_.T)] = True
    sd = np.where(ins, -(np.sqrt(edt2(~ins)) - 0.5), np.sqrt(edt2(ins)) - 0.5) * SEG_CELL
    sd = vol_blur(sd.astype(np.float32), 2)          # a little smoother, so the surface normals are too
    q = np.clip(np.round(sd / SEG_UNIT), -127, 127).astype(np.int8)
    seg_grids.append({'name': g, 'bone': idx[main], 'min': np.round(blo + lo_c * SEG_CELL + 0.5 * SEG_CELL, 5).tolist(),
                      'shape': [int(v) for v in q.shape], 'offset': sum(len(x) for x in seg_bytes)})
    seg_bytes.append(q.tobytes(order='C'))
seg_blob = gzip.compress(b''.join(seg_bytes), 9)
print(f'body segment shapes: {[g["name"] for g in seg_grids]}, {sum(len(x) for x in seg_bytes) / 1e6:.1f} M cells, '
      f'{len(seg_blob) / 1e6:.2f} MB gzipped', file=sys.stderr)


# ---------------------------------------------------------------- muscle contraction (arms, chest, hips and legs)
# A muscle keeps its volume: when its two ends come closer its belly gets fatter, and when it is stretched it
# gets thinner (the muscle-primitive idea, Scheepers et al. 1997). The length is measured between the
# muscle's attachments, not from joint angles, so muscles crossing two joints (biceps, triceps long head)
# respond to both. For each muscle: its origin and insertion ends and the rig bone carrying each. The deltoid
# parts are fans from a long origin to one insertion, so each of their points gets its own origin (the spot
# on the origin strip in line with it) and its own length. Each point stores: its ends, how far along the
# muscle it is (t, 0 at the origin, 1 at the insertion), its offset out from the centre line at rest, and a
# group for the per-muscle extras (biceps, triceps heads).
names_all = bone_names + [h['name'] for h in helpers] + [t['name'] for t in twist_bones]
CONTRACT = [  # (name pattern, origin bone, insertion bone, group, fan)
    (r'^Long_head_of_biceps_brachii', 'shoulder', 'radius', 1, False),
    (r'^Short_head_of_biceps_brachii', 'shoulder', 'radius', 1, False),
    (r'^Brachialis_muscle', 'upper_arm', 'forearm', 0, False),
    (r'^Brachioradialis_muscle', 'upper_arm', 'radius', 0, False),
    (r'^Long_head_of_triceps_brachii', 'shoulder', 'forearm', 2, False),
    (r'^Lateral_head_of_triceps_brachii', 'upper_arm', 'forearm', 2, False),
    (r'^Medial_head_of_triceps_brachii', 'upper_arm', 'forearm', 3, False),
    (r'_part_of_deltoid_muscle', 'shoulder', 'upper_arm', 4, True),
    # chest: pectoralis major, a fan from the collarbone, breastbone and upper rib cartilages to the humerus
    (r'^Clavicular_head_of_pectoralis_major', 'shoulder', 'upper_arm', 5, 'pec'),
    (r'^Sternocostal_head_of_pectoralis_major', 'spine.003', 'upper_arm', 6, 'pec'),
    (r'Abdominal_part_of_pectoralis_major', 'spine.003', 'upper_arm', 7, 'pec'),
    # hips: the glutes are fans from the pelvis to the femur; tensor fasciae latae runs straight down to the
    # thigh's outer band. Medius, minimus and the tensor shorten as the leg swings out, maximus as it goes back.
    (r'^Gluteus_(maximus|medius|minimus)_muscle', 'spine', 'thigh', 8, 'fan'),
    (r'^Tensor_fasciae_latae', 'spine', 'thigh', 8, False),
    # thigh: quadriceps (shorten as the knee straightens; rectus femoris also crosses the hip) and hamstrings
    (r'^Rectus_femoris_muscle', 'spine', 'shin', 9, False),
    (r'^Vastus_(medialis|lateralis|intermedius)_muscle', 'thigh', 'shin', 9, False),
    (r'^(Long_head_of_biceps_femoris|Semitendinosus_muscle|Semimembranosus_muscle)', 'spine', 'shin', 10, False),
    (r'^Short_head_of_biceps_femoris', 'thigh', 'shin', 10, False),
    # calf: measured to the heel tendon's attachment on the heel bone, so pointing the foot shortens them;
    # gastrocnemius starts above the knee, so bending the knee shortens it too
    (r'^(Medial|Lateral)_head_of_gastrocnemius', 'thigh', 'foot', 11, 'heel'),
    (r'^Soleus_muscle', 'shin', 'foot', 11, 'heel'),
    # the rest: lats (a fan to the humerus, fullest with the arm pulled down or back), upper trapezius (shrug,
    # arm overhead), abs and obliques (crunch, side bend, twist), sternocleidomastoid (one group per side, for
    # turning the head), serratus anterior (slips from the ribs to the shoulder blade, measured row by row) and
    # the wrist flexors and extensors. From here on the origin is the end nearer the origin bone's joint.
    (r'^Latissimus_dorsi_muscle', 'spine.002', 'upper_arm', 12, 'pec'),
    (r'^Descending_part_of_trapezius', 'spine.004', 'shoulder', 13, False),
    (r'^Rectus_abdominis_muscle', 'spine', 'spine.003', 14, False),
    (r'^(External|Internal)_abdominal_oblique', 'spine.003', 'spine', 15, False),
    (r'^Sternocleidomastoid_muscle', 'spine.003', 'spine.005', 16, False),
    (r'^Serratus_anterior_muscle', 'spine.003', 'shoulder', 18, 'rows'),
    (r'^(Flexor_carpi_radialis|Palmaris_longus|Humeral_head_of_flexor_carpi_ulnaris|Extensor_carpi_radialis_(longus|brevis)|'
     r'Humeral_head_of_extensor_carpi_ulnaris)', 'upper_arm', 'hand', 19, False),
    (r'^Ulnar_head_of_(flexor|extensor)_carpi_ulnaris', 'forearm', 'hand', 19, False),
]
con_rows, con_idx, con_o, con_b, con_t, con_r, con_g = [], [], [], [], [], [], []


def end_rows(P, axis_, top):
    """Centre of the end of a muscle along its axis (top = the end with the larger projection)."""
    pr = P @ axis_
    sel = pr > pr.max() - 0.015 if top else pr < pr.min() + 0.015
    return P[sel].mean(0)


for i, (n, _, _) in enumerate(MUS):
    rule = next((r for r in CONTRACT if re.search(r[0], n)), None)
    if rule is None: continue
    sfx = '.L' if n.rstrip(')')[-1] == 'l' else '.R'
    _, ob, ib, grp, fan = rule
    sided = lambda n_: n_ + ('' if n_.startswith('spine') else sfx)      # (the spine bones have no side)
    ob_i, ib_i = names_all.index(sided(ob)), names_all.index(sided(ib))
    ids = np.where(mid_all == i)[0]
    P_ = mus_P[ids].astype(float)
    c_ = P_.mean(0); _, _, vt = np.linalg.svd(P_ - c_, full_matrices=False); ax_ = vt[0]
    leg = ib in ('thigh', 'shin', 'foot')
    sh = heads1[idx[('thigh' if leg else 'upper_arm') + sfx]]     # the origin is the end nearer this joint
    if grp >= 12: sh = heads1[idx[sided(ob)]]
    if grp == 16 and sfx == '.R': grp = 17
    a_end, b_end = end_rows(P_, ax_, True), end_rows(P_, ax_, False)
    if fan in ('pec', 'fan'):
        # the insertion: the end nearest the attachment high on the humerus (glutes: nearest the femur). The
        # origin is a long curved line, so each fibre's origin is found by direction: the points are split into
        # 12 slices across the fan (by their direction from the insertion) and each slice's far end is the
        # origin of its points.
        if fan == 'pec':
            ref = sh + 0.18 * (heads1[idx['forearm' + sfx]] - sh)
            dI = np.linalg.norm(P_ - ref, axis=1)
        else:
            fem = skP[skN_arr == 'Femur' + sfx[1].lower()]
            dI = np.linalg.norm(P_ - fem[nearest_index(P_, fem, margin=0.08)], axis=1)
        ins = P_[dI < dI.min() + 0.015].mean(0)
        D_ = P_ - ins; dist = np.linalg.norm(D_, axis=1); u_ = D_ / np.maximum(dist[:, None], 1e-9)
        e1 = u_.mean(0); e1 /= np.linalg.norm(e1)
        perp = u_ - (u_ @ e1)[:, None] * e1; _, _, vt2 = np.linalg.svd(perp, full_matrices=False); e2 = vt2[0]
        ang = np.arctan2(u_ @ e2, u_ @ e1)
        edges = np.quantile(ang, np.linspace(0, 1, 13)); bin_ = np.clip(np.searchsorted(edges, ang, 'right') - 1, 0, 11)
        base = len(con_rows)
        for k_ in range(12):
            sel = bin_ == k_; dk = dist[sel]
            con_rows.append((*P_[sel][dk > dk.max() - 0.015].mean(0), ob_i))
        con_rows.append((*ins, ib_i)); b_row = len(con_rows) - 1
        o_row = base + bin_
        O = np.array(con_rows)[o_row, :3]
    elif fan == 'rows':
        # serratus: its slips run around the ribs from the side of the chest (origin, the outer edge) to the
        # shoulder blade's inner edge (insertion); the points are split into 8 rows by height, each row with
        # its own origin and insertion spot
        yq = np.quantile(P_[:, 1], np.linspace(0, 1, 9)); row = np.clip(np.searchsorted(yq, P_[:, 1], 'right') - 1, 0, 7)
        ax_abs = np.abs(P_[:, 0]); base = len(con_rows)
        for k_ in range(8):
            sel = row == k_; xa = ax_abs[sel]
            con_rows.append((*P_[sel][xa > xa.max() - 0.015].mean(0), ob_i))
            con_rows.append((*P_[sel][xa < xa.min() + 0.015].mean(0), ib_i))
        o_row = base + 2 * row; b_row = o_row + 1
        O = np.array(con_rows)[o_row, :3]
    elif fan:   # deltoid: the insertion is the end farther from the shoulder joint
        if np.linalg.norm(a_end - sh) > np.linalg.norm(b_end - sh): a_end, b_end = b_end, a_end
        ins = b_end
        # the origin strip: points within 2 cm of the far end from the insertion, 12 spots along it
        dist = np.linalg.norm(P_ - ins, axis=1)
        strip = P_[dist > dist.max() - 0.02]
        k_ = min(12, len(strip))
        spots = strip[np.linspace(0, len(strip) - 1, k_).astype(int)]
        base = len(con_rows)
        for q in spots: con_rows.append((*q, ob_i))
        con_rows.append((*ins, ib_i)); b_row = len(con_rows) - 1
        d_ = (P_ - ins) / np.maximum(np.linalg.norm(P_ - ins, axis=1, keepdims=True), 1e-9)
        sd = (spots - ins) / np.linalg.norm(spots - ins, axis=1, keepdims=True)
        o_row = base + np.argmax(d_ @ sd.T, 1)
        O = np.array(con_rows)[o_row, :3]
    else:     # the origin is the end nearer the shoulder
        if np.linalg.norm(a_end - sh) > np.linalg.norm(b_end - sh): a_end, b_end = b_end, a_end
        # (calf: the far end is the heel tendon's attachment on the heel bone, moving with the foot)
        far_end = np.array(next(h['head'] for h in helpers if h['name'] == 'heel_pull' + sfx)) if fan == 'heel' else b_end
        con_rows.append((*a_end, ob_i)); con_rows.append((*far_end, ib_i))
        o_row = np.full(len(ids), len(con_rows) - 2); b_row = len(con_rows) - 1
        O = np.tile(a_end, (len(ids), 1))
    B_ = np.array(con_rows)[b_row, :3] if fan == 'rows' else np.array(con_rows[b_row][:3]) if fan != 'heel' else b_end
    seg_ = B_ - O     # (calf: t runs along the muscle itself)
    t_ = np.clip(np.einsum('ij,ij->i', P_ - O, seg_) / np.einsum('ij,ij->i', seg_, seg_), 0, 1)
    r_ = P_ - (O + t_[:, None] * seg_)
    con_idx.append(ids); con_o.append(o_row); con_b.append(np.broadcast_to(b_row, len(ids)).copy()); con_t.append(t_)
    con_r.append(r_); con_g.append(np.full(len(ids), grp))
con_idx, con_o, con_b = np.concatenate(con_idx), np.concatenate(con_o), np.concatenate(con_b)
con_t, con_r, con_g, con_rows = np.concatenate(con_t), np.vstack(con_r), np.concatenate(con_g), np.array(con_rows)
print(f'muscle contraction: {len(con_idx)} points, {len(con_rows)} attachment spots, '
      f'mean offset from the centre line {np.linalg.norm(con_r, axis=1).mean() * 1000:.1f} mm', file=sys.stderr)

# The lats and teres major sticking out sideways past the ribcage (the fold behind the armpit) belong to the
# A-pose: raising the arm stretches them flat against the side of the chest. Each such point stores how far it
# sticks out past the side of the ribcage at its height (plus 2 cm for the muscle itself); the page pulls it
# back in by that much as the arm on its side rises. Points carried by the arm's pull (the tendon end) are left
# alone, so the lat stays attached to the arm.
spine_b = [idx[n] for n in bone_names if n.startswith('spine') and n != 'spine.005']
ribs_ = skP[np.isin(skB, spine_b)]
rb_ = np.round(ribs_[:, 1] / 0.02).astype(int)
lat_pts = np.where(np.isin(mid_all, [i for i, (n, _, _) in enumerate(MUS) if re.search(r'Latissimus_dorsi|Teres_major', n)]))[0]
bnd = np.round(mus_P[lat_pts, 1] / 0.02).astype(int)
side_x_ = np.array([np.abs(ribs_[rb_ == b, 0]).max(initial=0) for b in bnd])
excess = np.abs(mus_P[lat_pts, 0]) - (side_x_ + 0.02)
pull_share = (m_w4[lat_pts] * (m_top4[lat_pts] >= NB) * (m_top4[lat_pts] < NBH)).sum(1)   # share on the arm's pull
keep_ = (excess > 0.0005) & (side_x_ > 0)
flat_idx = lat_pts[keep_]
flat_vec = np.zeros((len(flat_idx), 3), np.float32)
flat_vec[:, 0] = -np.sign(mus_P[flat_idx, 0]) * excess[keep_] * (1 - pull_share[keep_])
print(f'lat fold: {len(flat_idx)} points stick out past the ribcage, up to {excess[keep_].max() * 1000:.0f} mm', file=sys.stderr)


# ---------------------------------------------------------------- write
# Big arrays are stored compactly (gzipped byte planes, see tools/rigpack.py); the page unpacks them at startup.
import rigpack
b64 = rigpack.b64


def q16(P):
    """Positions packed as 16-bit integers inside their bounding box (half the size, ~0.03 mm steps)."""
    lo, hi = P.min(0), P.max(0)
    return {'min': lo.tolist(), 'max': hi.tolist(), 'q': b64(np.round((P - lo) / (hi - lo) * 65535).astype(np.uint16))}


rig = {
    'bones': [{'name': n, 'parent': p, 'head': np.round(heads1[b], 5).tolist()} for b, (n, p) in enumerate(zip(bone_names, bone_parent))] +
             [{'name': h['name'], 'parent': h['parent'], 'head': h['head'], 'follows': bone_names[h.get('upper_arm', h.get('follow'))]} for h in helpers] +
             [{'name': t['name'], 'parent': t['parent'], 'head': t['head'], 'twist': t['frac'], **({'mid': t['mid']} if 'mid' in t else {})}
              for t in twist_bones],
    'twistAxes': twist_axes,
    'anchors': {'index': b64(anc_I.astype(np.int32)), 'weight': b64(np.round(anc_W * 255).astype(np.uint8))},
    'fat': {'position': q16(FV), 'index': b64(FT.astype(np.uint32)),
            'skinIndex': b64(f_top4.astype(np.uint8)), 'weight8': b64(np.round(f_w4 * 255).astype(np.uint8)), 'center': q16(cor_fat),
            'over': {'fat': b64(fat_over[0].astype(np.int32)), 'muscle': b64(fat_over[1].astype(np.int32)),
                     'gap': b64(fat_over[2].astype(np.float32))}},
    'body': {'position': q16(SV), 'normal': b64(SN.astype(np.float32)), 'index': b64(ST.astype(np.uint32)),
             'skinIndex': b64(s_top4.astype(np.uint8)), 'weight8': b64(np.round(s_w4 * 255).astype(np.uint8)), 'center': q16(cor_skin),
             'digit': b64(np.r_[digit, np.zeros(len(SV) - len(digit), np.uint8)])},
    'skeleton': {'position': q16(skP), 'bone': b64(skB), 'index': b64(skI)},
    'colliders': colliders,
    'sdf': {'cell': SDF_CELL, 'unit': SDF_UNIT, 'grids': sdf_grids, 'gz': base64.b64encode(sdf_blob).decode()},
    'segments': {'cell': SEG_CELL, 'unit': SEG_UNIT, 'grids': seg_grids, 'gz': base64.b64encode(seg_blob).decode()},
    'anchor': {'bone': idx['spine.003'], 'point': np.round(anchor, 4).tolist()},
    'muscles': {'position': q16(mus_P), 'index': b64(mus_F.astype(np.uint32)), 'color': b64(mus_C),
                'skinIndex': b64(m_top4.astype(np.uint8)), 'weight8': b64(np.round(m_w4 * 255).astype(np.uint8)), 'center': q16(cor_mus),
                'order': {'top': b64(o_top.astype(np.int32)), 'under': b64(o_und.astype(np.int32)),
                          'gap': b64(o_gap.astype(np.float32))},
                'layers': {'top': b64(lay_top.astype(np.int32)), 'under': b64(lay_under.astype(np.int32)),
                           'gap': b64(lay_gap.astype(np.float32))},
                'below': {'bottom': b64(bel_bottom.astype(np.int32)), 'cover': b64(bel_cover.astype(np.int32)),
                          'gap': b64(bel_gap.astype(np.float32)), 'sign': b64(bel_sign.astype(np.int8)),
                          'glue': b64(np.round(bel_glue * 255).astype(np.uint8))},
                'debugLat': b64(lat_ids_dbg.astype(np.int32)),
                'latFold': {'index': b64(flat_idx.astype(np.int32)), 'in': b64(flat_vec)},
                'contract': {'ends': b64(con_rows[:, :3].astype(np.float32)), 'endBone': b64(con_rows[:, 3].astype(np.uint8)),
                             'index': b64(con_idx.astype(np.int32)), 'origin': b64(con_o.astype(np.uint16)),
                             'insertion': b64(con_b.astype(np.uint16)), 't': b64(np.round(con_t * 255).astype(np.uint8)),
                             'out': b64(con_r.astype(np.float32)), 'group': b64(con_g.astype(np.uint8))}},
}
text_ = json.dumps(rigpack.dedupe(rig), separators=(',', ':'))   # (built first: a failure here must not leave a half-written body-rig.js)
with open(f'{out}/body-rig.js.tmp', 'w') as fh:
    fh.write('// Generated by tools/fit_rig.py. Body: "Male Base Mesh" (CC0) fitted to Z-Anatomy\n')
    fh.write('// (Z-Anatomy, CC BY-SA 4.0, built on BodyParts3D (c) The Database Center for Life Science).\n')
    fh.write('window.BODY_RIG = ' + text_ + ';\n')
os.replace(f'{out}/body-rig.js.tmp', f'{out}/body-rig.js')
print('wrote', f'{out}/body-rig.js', file=sys.stderr)

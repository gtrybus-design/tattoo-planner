"""Convert Z-Anatomy FBX files into a simple binary mesh format, using three.js's FBX loader
in headless Chrome (no Blender needed).

usage: python3 -I convert_fbx.py <fbx dir> <name.fbx> [<name.fbx> ...]

For each <name>.fbx writes <name>.meshes next to it:
  uint32 header length | JSON header [{name, verts, tris}, ...] | float32 positions | uint32 indices
Positions are in the FBX file's own world space (all node transforms applied).
"""
import http.server, json, os, subprocess, sys, threading

fbx_dir, files = os.path.abspath(sys.argv[1]), sys.argv[2:]
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
PORT = 8765

PAGE = """<!doctype html><meta charset="utf-8">
<script type="importmap">{"imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
const name = new URLSearchParams(location.search).get('f');
const root = await new FBXLoader().loadAsync('/fbx/' + name);
root.updateMatrixWorld(true);
const header = [], pos = [], idx = [];
root.traverse(o => {
  if (!o.isMesh) return;
  const g = o.geometry.clone().applyMatrix4(o.matrixWorld);
  const p = g.attributes.position.array;
  const n = p.length / 3;
  const ind = g.index ? Array.from(g.index.array) : Array.from({ length: n }, (_, i) => i);
  // mirrored objects (negative scale) come out inside-out: flip their triangles back
  if (o.matrixWorld.determinant() < 0) for (let i = 0; i < ind.length; i += 3) [ind[i + 1], ind[i + 2]] = [ind[i + 2], ind[i + 1]];
  header.push({ name: o.name, verts: n, tris: ind.length / 3 });
  pos.push(new Float32Array(p)); idx.push(new Uint32Array(ind));
});
const h = new TextEncoder().encode(JSON.stringify(header));
const parts = [new Uint32Array([h.length]), h, ...pos, ...idx];
await fetch('/save?f=' + encodeURIComponent(name), { method: 'POST', body: new Blob(parts) });
</script>"""

done = threading.Event()
pending = set(files)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path.startswith('/convert.html'):
            body = PAGE.encode(); ctype = 'text/html'
        elif self.path.startswith('/fbx/'):
            name = os.path.basename(self.path[5:].split('?')[0])
            body = open(os.path.join(fbx_dir, name), 'rb').read(); ctype = 'application/octet-stream'
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        name = os.path.basename(self.path.split('f=')[1])
        from urllib.parse import unquote
        name = unquote(name)
        data = self.rfile.read(int(self.headers['Content-Length']))
        out = os.path.join(fbx_dir, os.path.splitext(name)[0] + '.meshes')
        open(out, 'wb').write(data)
        print('wrote', out, len(data), 'bytes', file=sys.stderr)
        self.send_response(200); self.end_headers()
        pending.discard(name)
        if not pending: done.set()


server = http.server.ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
profile = os.path.join(fbx_dir, '.chrome-convert')
procs = [subprocess.Popen([CHROME, '--headless=new', f'--user-data-dir={profile}-{i}', '--no-first-run',
                           '--disable-gpu', f'http://127.0.0.1:{PORT}/convert.html?f={f}'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for i, f in enumerate(files)]
ok = done.wait(600)
import shutil
for i in range(len(files)): shutil.rmtree(f'{profile}-{i}', ignore_errors=True)
for p in procs: p.kill()
server.shutdown()
sys.exit(0 if ok else 1)

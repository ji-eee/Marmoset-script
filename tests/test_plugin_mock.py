"""Integration test for CaptureFrontBackBake.py using a MOCK ``mset`` module.

This exercises the plugin's real orchestration end-to-end on a plain machine:
  * scene gathering (visible mesh filtering, hidden objects ignored)
  * reading the camera, computing the turntable center
  * capturing FRONT, physically rotating the root 180deg, capturing BACK, and
    RESTORING the original transforms exactly
  * material grouping and one-PNG-per-material output
  * a round-trip correctness check: the mock renderer paints each surface point
    with its UV encoded as colour, so a baked texel at (u,v) must reconstruct
    (u*255, v*255) within tolerance.

The mock ``mset`` reflects each object's *current* transform when rendering, so
the 180deg turntable and its restoration are genuinely validated.

Run:  python3 tests/test_plugin_mock.py
"""

import math
import os
import sys
import types
from array import array

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from projbake import linalg as la
from projbake.image import ImageRGBA
from projbake import pngio
from projbake.mesh import SceneMesh, Submesh
from make_sphere_demo import build_uv_sphere

EULER_ORDER = "YXZ"
_failures = []


def check(cond, msg):
    print(("  ok  - " if cond else "  FAIL- ") + msg)
    if not cond:
        _failures.append(msg)


def color_uv(u, v):
    return (int(u * 255) & 255, int(v * 255) & 255, 128, 255)


# ---------------------------------------------------------------------------
# Fake mset scene objects
# ---------------------------------------------------------------------------
class FakeMesh:
    def __init__(self, verts, tris, uvs, norms):
        self.vertices = verts
        self.triangles = tris
        self.uvs = uvs
        self.normals = norms


class SubMeshObject:
    def __init__(self, material, start, count):
        self.material = material
        self.startIndex = start
        self.indexCount = count

    def getChildren(self):
        return []


class FakeMaterial:
    def __init__(self, name):
        self.name = name


class SceneObject:
    _uid = 10000

    def __init__(self, name):
        SceneObject._uid += 1
        self.uid = SceneObject._uid
        self.name = name
        self.visible = True
        self.parent = None

    def getChildren(self):
        return []


class MeshObject:
    _uid = 0

    def __init__(self, name, sm, material_name):
        MeshObject._uid += 1
        self.uid = MeshObject._uid
        self.name = name
        self.mesh = sm
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]
        self.pivot = [0.0, 0.0, 0.0]
        self.visible = True
        self.invisibleToCamera = False
        self.parent = None
        self._subs = [SubMeshObject(FakeMaterial(material_name), 0, len(sm.triangles))]

    def getChildren(self):
        return list(self._subs)

    def getBounds(self):
        vs = self.mesh.vertices
        lo = [min(vs[i::3]) for i in range(3)]
        hi = [max(vs[i::3]) for i in range(3)]
        # apply current translation (meshes are at identity rotation here)
        lo = [lo[i] + self.position[i] for i in range(3)]
        hi = [hi[i] + self.position[i] for i in range(3)]
        return [lo, hi]


class FakeCamera:
    def __init__(self, pos, rot, fov):
        self.position = list(pos)
        self.rotation = list(rot)
        self.fov = fov
        self.mode = "perspective"
        self.orthoScale = 1.0


# ---------------------------------------------------------------------------
# Build the mock mset module
# ---------------------------------------------------------------------------
def make_mock_mset(scene_objects, camera, capture_size):
    m = types.ModuleType("mset")
    m.MeshObject = MeshObject
    m._objects = scene_objects
    m._camera = camera
    m._dialogs = []

    def getAllObjects():
        return list(scene_objects)

    def getCamera():
        return camera

    def getSceneBounds():
        return [[-2, -2, -2], [2, 2, 2]]

    def getScenePath():
        return ""

    def getPluginPath():
        return os.path.join(_ROOT, "CaptureFrontBackBake.py")

    def log(msg):
        pass

    def err(msg):
        print("    mset.err:", str(msg).strip())

    def showOkDialog(msg):
        m._dialogs.append(msg)

    def showOpenFolderDialog():
        return ""

    def shutdownPlugin():
        pass

    def renderCamera(path="", width=-1, height=-1, sampling=-1,
                     transparency=False, camera="", viewportPass=""):
        # forward-render the CURRENT scene state, colouring by UV
        cam = la.Camera(camera_obj.position, camera_obj.rotation, camera_obj.fov,
                        width, height, euler_order=EULER_ORDER)
        img = _render_scene(scene_objects, cam)
        if path:
            pngio.save_png(path, img)
        return _FakeImage(img, path)

    camera_obj = camera
    m.getAllObjects = getAllObjects
    m.getCamera = getCamera
    m.getSceneBounds = getSceneBounds
    m.getScenePath = getScenePath
    m.getPluginPath = getPluginPath
    m.log = log
    m.err = err
    m.showOkDialog = showOkDialog
    m.showOpenFolderDialog = showOpenFolderDialog
    m.shutdownPlugin = shutdownPlugin
    m.renderCamera = renderCamera

    # UI stubs (constructed but not exercised here)
    for cls in ("UIWindow", "UIButton", "UILabel", "UIListBox",
                "UITextFieldFloat"):
        setattr(m, cls, _make_ui_stub(cls))
    return m


class _FakeImage:
    def __init__(self, img, path):
        self._img = img
        self._path = path

    def writeOut(self, path):
        pngio.save_png(path, self._img)


def _make_ui_stub(name):
    class _Stub:
        def __init__(self, *a, **k):
            self._items = []
            self.value = 0.0
            self.selectedItem = 0
            self.text = ""

        def addElement(self, *a, **k):
            pass

        def addReturn(self, *a, **k):
            pass

        def addSpace(self, *a, **k):
            pass

        def addItem(self, s):
            self._items.append(s)

        def selectItemByName(self, s):
            pass

        def close(self):
            pass
    _Stub.__name__ = name
    return _Stub


def _render_scene(scene_objects, cam):
    """Forward-render the current scene (UV-encoded colour, nearest depth)."""
    W, H = cam.width, cam.height
    img = ImageRGBA(W, H, fill=(0, 0, 0, 0))
    depth = array("f", [float("inf")]) * (W * H)
    data = img.data
    for obj in scene_objects:
        if not obj.visible:
            continue
        sm = SceneMesh(obj.name, obj.mesh.vertices, obj.mesh.triangles,
                       obj.mesh.uvs, obj.mesh.normals,
                       position=obj.position, rotation=obj.rotation,
                       scale=obj.scale, pivot=obj.pivot)
        for _mat, (p0, p1, p2), _n, (uv0, uv1, uv2) in sm.iter_world_triangles(None, EULER_ORDER):
            a = cam.project(p0)
            b = cam.project(p1)
            c = cam.project(p2)
            if not (a[3] and b[3] and c[3]):
                continue
            x0, y0, d0 = a[0], a[1], a[2]
            x1, y1, d1 = b[0], b[1], b[2]
            x2, y2, d2 = c[0], c[1], c[2]
            minx = max(0, int(math.floor(min(x0, x1, x2))))
            maxx = min(W - 1, int(math.ceil(max(x0, x1, x2))))
            miny = max(0, int(math.floor(min(y0, y1, y2))))
            maxy = min(H - 1, int(math.ceil(max(y0, y1, y2))))
            areav = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(areav) < 1e-12:
                continue
            inv = 1.0 / areav
            id0, id1, id2 = 1.0 / d0, 1.0 / d1, 1.0 / d2
            for py in range(miny, maxy + 1):
                yc = py + 0.5
                rb = py * W
                for px in range(minx, maxx + 1):
                    xc = px + 0.5
                    w0 = ((x1 - xc) * (y2 - yc) - (x2 - xc) * (y1 - yc)) * inv
                    w1 = ((x2 - xc) * (y0 - yc) - (x0 - xc) * (y2 - yc)) * inv
                    w2 = 1.0 - w0 - w1
                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue
                    dd = 1.0 / (w0 * id0 + w1 * id1 + w2 * id2)
                    idx = rb + px
                    if dd < depth[idx]:
                        depth[idx] = dd
                        u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                        v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                        col = color_uv(u, v)
                        o = idx * 4
                        data[o], data[o + 1], data[o + 2], data[o + 3] = col
    return img


# ---------------------------------------------------------------------------
def main():
    import tempfile

    size = 256
    sphere = build_uv_sphere(rings=32, sectors=64)
    head = MeshObject("Head", FakeMesh(sphere.vertices, sphere.triangles,
                                       sphere.uvs, sphere.normals), "head")
    # Marmoset can place meshes under a plain SceneObject that has visibility
    # and hierarchy, but no transform attributes. The bake should rotate the
    # mesh instead of trying to rotate that non-transform parent.
    root = SceneObject("Scene Root")
    head.parent = root
    # a hidden object that must be ignored
    hidden = MeshObject("Hidden", FakeMesh(sphere.vertices, sphere.triangles,
                                           sphere.uvs, sphere.normals), "hidden")
    hidden.visible = False

    camera = FakeCamera((0, 0, 4), (0, 0, 0), 45.0)
    scene = [head, hidden]

    mock = make_mock_mset(scene, camera, size)
    sys.modules["mset"] = mock

    # import AFTER installing the mock
    import importlib
    if "CaptureFrontBackBake" in sys.modules:
        importlib.reload(sys.modules["CaptureFrontBackBake"])
    import CaptureFrontBackBake as plugin

    out_dir = tempfile.mkdtemp(prefix="cbb_mock_")

    # snapshot transforms to verify exact restoration
    before = ([list(head.position), list(head.rotation),
               list(head.scale), list(head.pivot)])

    print("[plugin: run_bake]")
    plugin.run_bake(out_dir, size, 70.0, 0)  # edge_blur=0 for deterministic round-trip

    # transforms restored?
    after = ([list(head.position), list(head.rotation),
              list(head.scale), list(head.pivot)])
    check(before == after, "root transforms restored exactly after bake")

    # captures written
    check(os.path.isfile(os.path.join(out_dir, "_capture_front.png")),
          "front capture PNG written")
    check(os.path.isfile(os.path.join(out_dir, "_capture_back.png")),
          "back capture PNG written")

    # two outputs per visible material (masked + full); 'hidden' must NOT appear
    outs = [f for f in os.listdir(out_dir) if f.endswith(".png")
            and not f.startswith("_capture")]
    check(any(f.endswith("_masked.png") and "head" in f for f in outs),
          "masked head texture written")
    check(any(f.endswith("_full.png") and "head" in f for f in outs),
          "full (front/back merged) head texture written")
    check(not any("hidden" in f for f in outs),
          "hidden object's material not baked (%s)" % outs)

    def _coverage(fname):
        im = pngio.load_png(os.path.join(out_dir, fname))
        op = sum(1 for i in range(im.width * im.height) if im.data[i * 4 + 3] == 255)
        return im, op / float(im.width * im.height)

    # round-trip correctness on the masked head texture
    head_png = [f for f in outs if "head" in f and f.endswith("_masked.png")][0]
    baked, frac = _coverage(head_png)
    checked = 0
    rt_fail = 0
    for (u, v) in [(0.25, 0.5), (0.25, 0.4), (0.25, 0.6), (0.75, 0.5), (0.75, 0.45)]:
        tx = int(u * baked.width)
        ty = int((1.0 - v) * baked.height)
        got = baked.get(min(tx, baked.width - 1), min(ty, baked.height - 1))
        exp = color_uv(u, v)
        if got[3] != 255:
            print("    MISS uv(%.2f,%.2f) transparent" % (u, v))
            rt_fail += 1
            continue
        err = max(abs(got[i] - exp[i]) for i in range(2))  # compare R,G (UV)
        checked += 1
        if err > 16:
            print("    BAD uv(%.2f,%.2f) got=%s exp=%s err=%d" % (u, v, got[:3], exp[:3], err))
            rt_fail += 1
    check(rt_fail == 0 and checked >= 4,
          "round-trip: masked texels reconstruct their UV (%d checked)" % checked)
    check(0.1 < frac < 0.95, "masked coverage sane (%.1f%% opaque, sides masked)" % (100 * frac))

    # the 'full' (no side mask) variant must cover MORE than the masked one
    full_png = [f for f in outs if "head" in f and f.endswith("_full.png")][0]
    _, full_frac = _coverage(full_png)
    check(full_frac > frac + 0.05,
          "full variant covers more than masked (full=%.1f%% > masked=%.1f%%)"
          % (100 * full_frac, 100 * frac))

    print()
    if _failures:
        print("PLUGIN MOCK: %d FAILURE(S)" % len(_failures))
        for f in _failures:
            print("  -", f)
        sys.exit(1)
    print("PLUGIN MOCK: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

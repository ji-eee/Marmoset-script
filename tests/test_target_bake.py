"""Tests for the Target Mesh feature (mid/low reprojection) and the new UI wiring,
using a MOCK ``mset`` so it runs without Marmoset.

Covers:
  * TARGET MODE bake: capture a VISIBLE source mesh (mid), project onto a
    DIFFERENT hidden target mesh (low) sharing the same UV layout, and verify
    the target's texels reconstruct the source's UV-encoded colour (front from
    the front capture, back from the 180-deg back capture). Also verifies the
    source transforms are restored and no intermediate captures are left.
  * UI wiring: constructing the window, refreshing the mesh list, selecting a
    target via the list and via "Use Selected Mesh", and the middle-ellipsis
    path helper — all under a mock mset.

Run:  python3 tests/test_target_bake.py
"""

import os
import sys
import tempfile
import importlib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from projbake import pngio
from make_sphere_demo import build_uv_sphere
import test_plugin_mock as tpm   # reuse MeshObject/FakeMesh/FakeCamera/make_mock_mset

_failures = []


def check(cond, msg):
    print(("  ok  - " if cond else "  FAIL- ") + msg)
    if not cond:
        _failures.append(msg)


def _mesh_obj(name, sphere, material, visible=True):
    o = tpm.MeshObject(name, tpm.FakeMesh(sphere.vertices, sphere.triangles,
                                          sphere.uvs, sphere.normals), material)
    o.visible = visible
    return o


# ---------------------------------------------------------------------------
def _augment_mock_ui(mock, sel_obj=None):
    """Give the mock mset a full-enough UI + selection API to construct the
    real CaptureBakeUI without Marmoset."""
    class LB:
        def __init__(self, title=""):
            self._items = []; self.selectedItem = 0; self.onSelect = None; self.title = title
        def addItem(self, s): self._items.append(s)
        def clearItems(self): self._items = []
        def selectItemByName(self, s):
            if s in self._items: self.selectedItem = self._items.index(s)
        def selectNone(self): self.selectedItem = -1

    class Win:
        def __init__(self, t=""): self.title = t; self.visible = True; self.width = 0; self.height = 0
        def addElement(self, *a): pass
        def addReturn(self, *a): pass
        def addSpace(self, *a): pass
        def addStretchSpace(self, *a): pass
        def clearElements(self, *a): pass
        def getElements(self): return []
        def close(self): pass

    class TF:
        def __init__(self, *a): self.value = 0; self.width = 0; self.onChange = None

    class Btn:
        def __init__(self, t=""): self.text = t; self.onClick = None
        def setIcon(self, *a): pass

    class Lbl:
        def __init__(self, t=""): self.text = t; self.fixedWidth = 0

    mock.UIListBox = LB
    mock.UIWindow = Win
    mock.UITextField = TF
    mock.UITextFieldInt = TF
    mock.UITextFieldFloat = TF
    mock.UIButton = Btn
    mock.UILabel = Lbl
    mock.getSelectedObject = lambda: sel_obj
    mock.getSelectedObjects = lambda: ([sel_obj] if sel_obj else [])


def _load_plugin(scene, camera, size, sel_obj=None):
    mock = tpm.make_mock_mset(scene, camera, size)
    _augment_mock_ui(mock, sel_obj=sel_obj)
    sys.modules["mset"] = mock
    if "CaptureFrontBackBake" in sys.modules:
        importlib.reload(sys.modules["CaptureFrontBackBake"])
    import CaptureFrontBackBake as plugin
    return plugin


# ---------------------------------------------------------------------------
def test_target_mode_reprojection():
    print("[target mode: mid -> low reprojection]")
    size = 256
    hi = build_uv_sphere(rings=32, sectors=64)   # visible mid-poly source
    lo = build_uv_sphere(rings=24, sectors=48)   # hidden low-poly target
    hi_obj = _mesh_obj("HiMid", hi, "shared", visible=True)
    lo_obj = _mesh_obj("LowTarget", lo, "shared", visible=False)
    camera = tpm.FakeCamera((0, 0, 4), (0, 0, 0), 45.0)
    scene = [hi_obj, lo_obj]
    plugin = _load_plugin(scene, camera, size)

    out = tempfile.mkdtemp(prefix="cbb_target_")
    before = (list(hi_obj.position), list(hi_obj.rotation),
              list(hi_obj.scale), list(hi_obj.pivot))
    # target-mode bake: source = visible (hi), target = hidden low-poly (lo)
    plugin.run_bake(out, size, 75.0, 0, target_mesh=lo_obj)
    after = (list(hi_obj.position), list(hi_obj.rotation),
             list(hi_obj.scale), list(hi_obj.pivot))

    check(before == after, "source transforms restored after target bake")
    outs = [f for f in os.listdir(out) if f.endswith(".png")]
    check(not any(f.startswith("_") for f in outs),
          "no intermediate _capture_*.png left (%s)" % outs)
    masked = [f for f in outs if "shared" in f and f.endswith("_masked.png")]
    full = [f for f in outs if "shared" in f and f.endswith("_full.png")]
    check(len(masked) == 1 and len(full) == 1,
          "target material written as _masked + _full (%s)" % outs)
    if not masked:
        return

    img = pngio.load_png(os.path.join(out, masked[0]))
    opaque = sum(1 for i in range(img.width * img.height) if img.data[i * 4 + 3] == 255)
    frac = opaque / float(img.width * img.height)
    check(0.1 < frac < 0.95, "target coverage sane (%.1f%% opaque)" % (100 * frac))

    def sample(u, v):
        tx = min(int(u * img.width), img.width - 1)
        ty = min(int((1.0 - v) * img.height), img.height - 1)
        return img.get(tx, ty)

    # the mock renders UV-encoded colour_uv = (u*255, v*255, 128, 255).
    # front hemisphere ~ u=0.25 (faces camera) -> sampled from FRONT capture;
    # back hemisphere ~ u=0.75 -> sampled from the 180-deg BACK capture.
    front_ok = 0
    for v in (0.4, 0.5, 0.6):
        c = sample(0.25, v)
        if c[3] == 255 and abs(c[0] - int(0.25 * 255)) <= 28 and abs(c[1] - int(v * 255)) <= 28:
            front_ok += 1
    check(front_ok >= 2,
          "front-hemisphere target texels reconstruct mid-poly UV colour (%d/3)" % front_ok)

    back_c = sample(0.75, 0.5)
    check(back_c[3] == 255 and back_c[0] > 150,
          "back-hemisphere target texel sampled from the back capture (r=%d, expect ~191)"
          % back_c[0])


def test_target_defaults_to_self_when_none():
    print("[target mode: None target keeps default self-bake]")
    size = 128
    sph = build_uv_sphere(rings=16, sectors=32)
    a = _mesh_obj("A", sph, "matA", visible=True)
    camera = tpm.FakeCamera((0, 0, 4), (0, 0, 0), 45.0)
    plugin = _load_plugin([a], camera, size)
    out = tempfile.mkdtemp(prefix="cbb_self_")
    plugin.run_bake(out, size, 75.0, 0, target_mesh=None)   # default path
    outs = [f for f in os.listdir(out) if f.endswith(".png")]
    check(any("matA" in f and f.endswith("_masked.png") for f in outs),
          "default (no target) still bakes each mesh onto itself (%s)" % outs)


def test_single_view_mode():
    print("[single view: project the current view only (no turntable)]")
    size = 256
    sph = build_uv_sphere(rings=32, sectors=64)
    obj = _mesh_obj("Solo", sph, "solo", visible=True)
    camera = tpm.FakeCamera((0, 0, 4), (0, 0, 0), 45.0)
    plugin = _load_plugin([obj], camera, size)
    out = tempfile.mkdtemp(prefix="cbb_single_")

    before_rot = list(obj.rotation)
    plugin.run_bake(out, size, 75.0, 0, target_mesh=None, single_view=True)
    check(list(obj.rotation) == before_rot,
          "single view does NOT turntable the mesh (rotation unchanged)")

    outs = [f for f in os.listdir(out) if f.endswith(".png")]
    masked = [f for f in outs if "solo" in f and f.endswith("_masked.png")]
    check(len(masked) == 1, "single-view masked output written (%s)" % outs)
    if not masked:
        return
    img = pngio.load_png(os.path.join(out, masked[0]))

    def sample(u, v):
        tx = min(int(u * img.width), img.width - 1)
        ty = min(int((1.0 - v) * img.height), img.height - 1)
        return img.get(tx, ty)

    front = sample(0.25, 0.5)   # hemisphere facing the camera -> painted
    back = sample(0.75, 0.5)    # hemisphere facing away -> masked (no back cap)
    check(front[3] == 255 and abs(front[0] - int(0.25 * 255)) <= 28,
          "current-view-facing texel painted (u=0.25 -> %s)" % (front[:3],))
    check(back[3] == 0,
          "away-facing texel masked in single view (a=%d, expect 0)" % back[3])


def test_ui_wiring():
    print("[ui wiring under mock]")
    size = 128
    hi = build_uv_sphere(rings=8, sectors=16)
    lo = build_uv_sphere(rings=6, sectors=12)
    hi_obj = _mesh_obj("HiMid", hi, "shared", visible=True)
    lo_obj = _mesh_obj("LowTarget", lo, "shared", visible=False)
    camera = tpm.FakeCamera((0, 0, 4), (0, 0, 0), 45.0)
    # "Use Selected Mesh" will return hi_obj
    plugin = _load_plugin([hi_obj, lo_obj], camera, size, sel_obj=hi_obj)

    ui = None
    try:
        ui = plugin.CaptureBakeUI()
    except Exception as e:
        check(False, "CaptureBakeUI constructs without error (%r)" % e)
        return
    check(True, "CaptureBakeUI constructs without error")

    # mesh list populated (2 meshes, incl. hidden)
    check(len(ui.mesh_objs) == 2, "mesh picker lists all meshes incl hidden (%d)" % len(ui.mesh_objs))

    # DEFAULT: screenshot mesh == target (self-bake), selected out of the box
    check(ui.target_mesh is None and ui._resolve_target() is None,
          "default target is 'same as captured mesh' (self-bake)")
    check(ui.mesh_box._items and ui.mesh_box._items[0] == plugin.CaptureBakeUI._SAME_ITEM
          and ui.mesh_box.selectedItem == 0,
          "list index 0 is the default '(same as captured mesh)' and is selected")

    # select the hidden low-poly as the target via the list box (index +1 for
    # the leading default entry)
    lo_idx = ui.mesh_objs.index(lo_obj) + 1
    ui.mesh_box.selectedItem = lo_idx
    ui._on_target_select()
    check(ui.target_mesh is lo_obj and ui._resolve_target() is lo_obj,
          "selecting a list row sets (and resolves) a different target")

    # selecting the default row again goes back to same-as-captured (None)
    ui.mesh_box.selectedItem = 0
    ui._on_target_select()
    check(ui.target_mesh is None, "selecting row 0 returns to same-as-captured default")

    # "Use Selected Mesh" picks hi_obj (the mock's selected object)
    ui._use_selected_mesh()
    check(ui.target_mesh is hi_obj, "'Use Selected Mesh' sets target to the viewport selection")

    # refresh keeps the target by uid
    ui._refresh_meshes()
    check(ui.target_mesh is hi_obj, "target survives a list refresh (matched by uid)")

    # capture mode defaults to Front+Back (turntable); "Current view only" opt-in
    check(ui._selected_single_view() is False,
          "default capture mode = Front + Back (turntable)")
    ui.mode_box.selectedItem = 1
    check(ui._selected_single_view() is True, "'Current view only' mode selectable")
    ui.mode_box.selectedItem = 0

    # folder field shows an elided path, not the raw long string
    ui.output_dir = r"C:\Users\Someone\AppData\Local\Marmoset Toolbag 5\plugins\CaptureFrontBackBake\out"
    ui._update_folder_field()
    val = ui.folder_field.value
    check(len(val) <= plugin.CaptureBakeUI._PATH_MAX_CHARS and "..." in val,
          "long output path shown middle-elided (%r)" % val)


def test_ellipsize_middle():
    print("[ellipsize_middle]")
    import CaptureFrontBackBake as plugin
    e = plugin._ellipsize_middle
    check(e("short", 46) == "short", "short path unchanged")
    long = r"C:\Users\KJH\AppData\Local\Marmoset Toolbag 5\plugins\CaptureFrontBackBake\output"
    r = e(long, 40)
    check(len(r) == 40 and "..." in r, "elided to exact budget with ellipsis (%d)" % len(r))
    check(r.startswith("C:\\Users") and r.endswith("output"),
          "keeps drive/root AND leaf visible (%r)" % r)
    check(e("abcdefghij", 4) == "abcdefghij",
          "budget too small to elide -> returned unchanged (no crash)")


def main():
    test_target_mode_reprojection()
    test_target_defaults_to_self_when_none()
    test_single_view_mode()
    test_ui_wiring()
    test_ellipsize_middle()
    print()
    if _failures:
        print("TARGET/UI TESTS: %d FAILURE(S)" % len(_failures))
        for f in _failures:
            print("  -", f)
        sys.exit(1)
    print("TARGET/UI TESTS: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

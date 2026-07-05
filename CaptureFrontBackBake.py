"""Capture Front/Back And Bake - Marmoset Toolbag 5 plugin.

Substance-Painter-style projection bake: capture the CURRENT camera view of the
visible scene (front), rotate the visible objects 180 degrees about world Y like a
turntable and capture again (back), then project both captures onto the meshes' UVs
and write one PNG per material. Overlapping/occluded areas and grazing side faces
are left transparent (masked).

INSTALL
    Copy this file AND the ``projbake/`` folder next to it into your Toolbag
    plugins directory, e.g. on Windows:
        C:\\Users\\<you>\\AppData\\Local\\Marmoset Toolbag 5\\plugins\\CaptureFrontBackBake\\
    (both ``CaptureFrontBackBake.py`` and ``projbake/`` in the same folder), then
    Edit > Plugins > Refresh and run "Capture Front Back Bake".

The heavy lifting lives in the dependency-free ``projbake`` package, which is unit
tested outside Marmoset. This file only talks to ``mset``. If the projection looks
mirrored/rotated in Marmoset, the one knob to try first is ``EULER_ORDER`` below
(see docs/marmoset-api-notes.md).
"""

import os
import sys
import traceback

import mset

# --- make the bundled projbake package importable -------------------------
def _plugin_dir():
    try:
        p = mset.getPluginPath()
        if p:
            return os.path.dirname(os.path.abspath(p))
    except Exception:
        pass
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()

_HERE = _plugin_dir()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from projbake import linalg as la          # noqa: E402
from projbake import pngio                 # noqa: E402
from projbake.image import ImageRGBA       # noqa: E402
from projbake.mesh import SceneMesh, Submesh  # noqa: E402
from projbake import bake                  # noqa: E402
from projbake import postprocess           # noqa: E402

# ===========================================================================
# Configuration
# ===========================================================================
# The single riskiest assumption: how Marmoset composes TransformObject.rotation.
# YXZ keeps the 180-degree turntable exact (ry += 180). Change here if projection
# is wrong after in-app testing.
EULER_ORDER = "YXZ"

TEXTURE_SIZES = ["512", "1024", "2048", "4096"]
DEFAULT_SIZE = "1024"
DEFAULT_SIDE_MASK_ANGLE = 75.0
DEFAULT_EDGE_BLUR = 3            # px; feathers the masked-output island edges (0 = off)
MIN_TEXTURE_SIZE = 16
MAX_TEXTURE_SIZE = 8192


def _log(msg):
    try:
        mset.log("[CaptureFrontBackBake] " + str(msg) + "\n")
    except Exception:
        print(msg)


def _err(msg):
    try:
        mset.err("[CaptureFrontBackBake] " + str(msg) + "\n")
    except Exception:
        print("ERROR:", msg)


# ===========================================================================
# Scene gathering (all mset access is here)
# ===========================================================================
def _all_objects_unique():
    """Return every scene object exactly once, whether getAllObjects() is flat
    or root-only."""
    seen = {}
    stack = list(mset.getAllObjects())
    while stack:
        o = stack.pop()
        try:
            uid = o.uid
        except Exception:
            uid = id(o)
        if uid in seen:
            continue
        seen[uid] = o
        try:
            stack.extend(o.getChildren())
        except Exception:
            pass
    return list(seen.values())


def _effective_visible(obj):
    """An object is visible only if it and all ancestors are visible."""
    cur = obj
    while cur is not None:
        try:
            if not cur.visible:
                return False
        except Exception:
            pass
        try:
            cur = cur.parent
        except Exception:
            cur = None
    return True


def _is_mesh(obj):
    try:
        return isinstance(obj, mset.MeshObject)
    except Exception:
        return type(obj).__name__ == "MeshObject"


def _collect_visible_meshes():
    out = []
    for o in _all_objects_unique():
        if not _is_mesh(o):
            continue
        if not _effective_visible(o):
            continue
        try:
            if o.invisibleToCamera:
                continue
        except Exception:
            pass
        out.append(o)
    return out


def _world_matrix_of(obj):
    """Compose the full parent-chain world matrix for a scene object using the
    plugin's Euler convention. world = parentWorld @ localTRS(pivot)."""
    chain = []
    cur = obj
    while cur is not None:
        chain.append(cur)
        try:
            cur = cur.parent
        except Exception:
            cur = None
    W = la.IDENTITY
    for node in reversed(chain):
        pos = _vec3(getattr(node, "position", (0, 0, 0)))
        rot = _vec3(getattr(node, "rotation", (0, 0, 0)))
        scl = _vec3(getattr(node, "scale", (1, 1, 1)), default=1.0)
        piv = _vec3(getattr(node, "pivot", (0, 0, 0)))
        local = la.mat_mul_chain(
            la.translation(pos),
            la.translation(piv),
            la.euler_to_matrix(rot[0], rot[1], rot[2], EULER_ORDER),
            la.scaling(scl),
            la.translation((-piv[0], -piv[1], -piv[2])),
        )
        W = la.mat_mul(W, local)
    return W


def _vec3(v, default=0.0):
    try:
        return (float(v[0]), float(v[1]), float(v[2]))
    except Exception:
        return (default, default, default)


def _submeshes_of(mesh_obj):
    """Return a list of projbake Submesh from a MeshObject's SubMeshObject
    children. Falls back to a single whole-mesh submesh."""
    subs = []
    try:
        for ch in mesh_obj.getChildren():
            if type(ch).__name__ != "SubMeshObject":
                continue
            mat = None
            try:
                if ch.material is not None:
                    mat = ch.material.name
            except Exception:
                mat = None
            try:
                start = int(ch.startIndex)
                count = int(ch.indexCount)
            except Exception:
                continue
            subs.append(Submesh(mat, start, count))
    except Exception:
        pass
    return subs


def _mesh_to_scenemesh(mesh_obj):
    """Convert a Marmoset MeshObject into a world-space projbake SceneMesh."""
    m = mesh_obj.mesh
    verts = list(m.vertices)
    tris = [int(i) for i in m.triangles]
    uvs = list(m.uvs) if m.uvs else []
    try:
        norms = list(m.normals) if m.normals else None
    except Exception:
        norms = None
    subs = _submeshes_of(mesh_obj)
    if not subs:
        # fall back to the mesh's own material name if reachable, else None
        mat = None
        try:
            mat = mesh_obj.getChildren()[0].material.name
        except Exception:
            mat = None
        subs = [Submesh(mat, 0, len(tris))]
    W = _world_matrix_of(mesh_obj)
    return SceneMesh(mesh_obj.name, verts, tris, uvs, norms,
                     submeshes=subs, world_override=W)


def _bounds_center(mesh_objs):
    """Union AABB center of the given objects (world space). Falls back to the
    whole-scene bounds, then the origin."""
    lo = [None, None, None]
    hi = [None, None, None]

    def acc(b):
        if not b:
            return
        mn, mx = b[0], b[1]
        for k in range(3):
            lo[k] = mn[k] if lo[k] is None else min(lo[k], mn[k])
            hi[k] = mx[k] if hi[k] is None else max(hi[k], mx[k])

    for o in mesh_objs:
        try:
            acc(o.getBounds())
        except Exception:
            pass
    if lo[0] is None:
        try:
            acc(mset.getSceneBounds())
        except Exception:
            pass
    if lo[0] is None:
        return (0.0, 0.0, 0.0)
    return ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5)


# ===========================================================================
# Turntable (physical rotation of one object, exact under YXZ; restores originals)
# ===========================================================================
def _is_ancestor(candidate, obj):
    """True if ``candidate`` is somewhere up ``obj``'s parent chain (so hiding it
    during isolation would also hide ``obj``)."""
    cur = getattr(obj, "parent", None)
    while cur is not None:
        if cur is candidate:
            return True
        cur = getattr(cur, "parent", None)
    return False


def _snapshot(obj):
    return {
        "position": list(obj.position),
        "rotation": list(obj.rotation),
        "scale": list(getattr(obj, "scale", [1, 1, 1])),
        "pivot": list(getattr(obj, "pivot", [0, 0, 0])),
    }


def _restore(obj, snap):
    try:
        obj.position = snap["position"]
        obj.rotation = snap["rotation"]
        obj.scale = snap["scale"]
        obj.pivot = snap["pivot"]
    except Exception as e:
        _err("failed to restore %s: %s" % (getattr(obj, "name", "?"), e))


def _apply_turntable_180(obj, center):
    """Set obj to its 180-deg-about-world-Y (through center) pose.

    Exact for the YXZ convention: pre-multiplying the world transform by
    Ry(180)-about-C gives  pos' = C + Ry180(pos+pivot-C) - pivot  and
    rot' = [rx, ry+180, rz]  (scale and pivot unchanged).
    """
    pos = _vec3(obj.position)
    piv = _vec3(getattr(obj, "pivot", (0, 0, 0)))
    rot = list(obj.rotation)
    C = center
    sx = pos[0] + piv[0] - C[0]
    sy = pos[1] + piv[1] - C[1]
    sz = pos[2] + piv[2] - C[2]
    # Ry(180): (x,y,z) -> (-x, y, -z)
    npos = [C[0] - sx - piv[0], C[1] + sy - piv[1], C[2] - sz - piv[2]]
    obj.position = npos
    obj.rotation = [rot[0], rot[1] + 180.0, rot[2]]


# ===========================================================================
# Capture
# ===========================================================================
def _capture(path, size):
    """Render the current active camera to a square PNG with transparency and
    return it decoded as an ImageRGBA."""
    mset.renderCamera(path=path, width=size, height=size, transparency=True)
    if not os.path.isfile(path):
        raise RuntimeError("renderCamera did not write %s" % path)
    return pngio.load_png(path)


def _read_camera(width, height):
    """Build the projection camera. ``width``/``height`` should be the ACTUAL
    captured image size so the projection stays self-consistent even if
    renderCamera produced a different resolution than requested."""
    cam = mset.getCamera()
    if cam is None:
        raise RuntimeError("No active camera (mset.getCamera() returned None)")
    mode = "perspective"
    try:
        mode = cam.mode
    except Exception:
        pass
    ortho_scale = 1.0
    try:
        ortho_scale = float(cam.orthoScale)
    except Exception:
        pass
    return la.Camera(
        _vec3(cam.position), _vec3(cam.rotation), float(cam.fov),
        width, height, mode=mode, ortho_scale=ortho_scale, euler_order=EULER_ORDER,
    )


# ===========================================================================
# Main bake orchestration
# ===========================================================================
def _sanitize(name):
    if not name:
        return "material"
    keep = "-_.() "
    s = "".join(c if (c.isalnum() or c in keep) else "_" for c in str(name))
    return s.strip() or "material"


def _capture_object_isolated(obj, others, size, front_path, back_path):
    """Capture ``obj`` alone (other target meshes hidden) so nothing occludes it.

    Returns ``(front_img, back_img, center)``. ``center`` is obj's own bounds
    center, used both for the physical 180-deg turntable and by the bake so the
    back projection matches. Restores visibility and transform no matter what.
    """
    to_hide = [o for o in others if o is not obj and not _is_ancestor(o, obj)]
    vis_snap = [(o, o.visible) for o in to_hide]
    try:
        for o in to_hide:
            try:
                o.visible = False
            except Exception:
                pass
        front_img = _capture(front_path, size)
        center = _bounds_center([obj])
        snap = _snapshot(obj)
        try:
            _apply_turntable_180(obj, center)
            back_img = _capture(back_path, size)
        finally:
            _restore(obj, snap)
    finally:
        for o, v in vis_snap:
            try:
                o.visible = v
            except Exception:
                pass
    return front_img, back_img, center


def run_bake(output_dir, size, side_mask_angle, edge_blur_px=DEFAULT_EDGE_BLUR):
    if not output_dir or not os.path.isdir(output_dir):
        mset.showOkDialog("Please choose a valid Output Folder first.")
        return
    size = int(size)
    edge_blur_px = int(edge_blur_px)

    meshes = _collect_visible_meshes()
    if not meshes:
        mset.showOkDialog("No visible mesh objects to bake.")
        return
    _log("visible meshes: %d" % len(meshes))

    # Two outputs per material:
    #   masked - side-masked (grazing sides transparent) + occlusion, then edge-blur
    #   full   - unmasked front/back smear, later filled to be fully opaque
    variants = [
        {"name": "masked", "side_mask_angle": float(side_mask_angle)},
        {"name": "full", "side_mask_angle": 90.0, "unmasked": True},
    ]

    # Project each object in ISOLATION (others hidden) so a nearer object never
    # erases the surface behind it, then composite per material. This is what
    # fills the areas that a single combined screenshot leaves blank.
    camera = None
    composited = {}  # {material: {variant: ImageRGBA}}
    for i, obj in enumerate(meshes):
        _log("object %d/%d %r: isolated capture..." % (i + 1, len(meshes),
                                                        getattr(obj, "name", "?")))
        fpath = os.path.join(output_dir, "_capture_front_%d.png" % i)
        bpath = os.path.join(output_dir, "_capture_back_%d.png" % i)
        front_img, back_img, center = _capture_object_isolated(
            obj, meshes, size, fpath, bpath)
        # intermediates are decoded into memory; keep the output folder clean
        for p in (fpath, bpath):
            try:
                os.remove(p)
            except OSError:
                pass
        if camera is None:
            camera = _read_camera(front_img.width, front_img.height)
            _log("capture resolution: %dx%d" % (front_img.width, front_img.height))

        sm = _mesh_to_scenemesh(obj)
        res = bake.bake_variants(
            [sm], front_img, back_img, camera, center, size, variants,
            euler_order=EULER_ORDER, log=None)
        for mat, var_imgs in res.items():
            slot = composited.setdefault(mat, {})
            for var, img in var_imgs.items():
                if var not in slot:
                    slot[var] = img               # first object owns the buffer
                else:
                    postprocess.composite_max_alpha(slot[var], img)

    results = composited
    written = []
    scene_name = "bake"
    try:
        sp = mset.getScenePath()
        if sp:
            scene_name = os.path.splitext(os.path.basename(sp))[0] or "bake"
    except Exception:
        pass
    for mat, var_imgs in results.items():
        masked = var_imgs["masked"]
        if edge_blur_px > 0:
            _log("edge-blurring %r masked output (%dpx)..." % (mat, edge_blur_px))
            masked = postprocess.edge_blur(masked, edge_blur_px)
        _log("filling %r full output (no transparency)..." % (mat,))
        full = postprocess.fill_transparent(var_imgs["full"])
        base = "%s_%s" % (_sanitize(scene_name), _sanitize(mat))
        for suffix, img in (("masked", masked), ("full", full)):
            out_path = os.path.join(output_dir, "%s_%s.png" % (base, suffix))
            pngio.save_png(out_path, img)
            written.append(out_path)
            _log("wrote %s" % out_path)

    mset.showOkDialog(
        "Bake complete.\n\n%d texture(s) written to:\n%s\n\n"
        "Per material: _masked (sides masked + edge blur), "
        "_full (front/back smear, fully opaque)." % (len(written), output_dir))


# ===========================================================================
# UI
# ===========================================================================
class CaptureBakeUI:
    def __init__(self):
        self.output_dir = self._default_output_dir()
        self.window = mset.UIWindow("Capture Front Back Bake")
        self._build()

    def _default_output_dir(self):
        try:
            sp = mset.getScenePath()
            if sp and os.path.isdir(os.path.dirname(sp)):
                return os.path.dirname(sp)
        except Exception:
            pass
        return _HERE

    # label column width; keeps all value fields left-aligned on one column
    _LABEL_WIDTH = 150.0
    _FIELD_WIDTH = 64.0

    def _row_label(self, text):
        lb = mset.UILabel(text)
        try:
            lb.fixedWidth = self._LABEL_WIDTH
        except Exception:
            pass
        return lb

    def _build(self):
        w = self.window

        # --- Output folder --------------------------------------------------
        w.addElement(self._row_label("Output Folder:"))
        browse = mset.UIButton("Browse...")
        browse.onClick = self._pick_folder
        w.addElement(browse)
        w.addReturn()
        self.folder_label = mset.UILabel(self.output_dir or "(none)")
        w.addElement(self.folder_label)
        w.addReturn()
        w.addReturn()

        # --- Texture size (one setting per row; the ListBox draws its own
        # title, so pass '' to avoid a doubled "Texture Size: Texture Size") --
        w.addElement(self._row_label("Texture Size:"))
        self.size_box = mset.UIListBox("")
        for s in TEXTURE_SIZES:
            self.size_box.addItem(s)
        try:
            self.size_box.selectItemByName(DEFAULT_SIZE)
        except Exception:
            pass
        w.addElement(self.size_box)
        w.addReturn()

        w.addElement(self._row_label("Custom Size (px):"))
        self.custom_size_field = mset.UITextFieldInt()
        try:
            self.custom_size_field.value = 0
            self.custom_size_field.width = self._FIELD_WIDTH
        except Exception:
            pass
        w.addElement(self.custom_size_field)
        w.addSpace(6)
        w.addElement(mset.UILabel("(0 = use preset)"))
        w.addReturn()

        # --- Side mask angle ------------------------------------------------
        w.addElement(self._row_label("Side Mask Angle (deg):"))
        self.angle_field = mset.UITextFieldFloat()
        try:
            self.angle_field.value = DEFAULT_SIDE_MASK_ANGLE
            self.angle_field.width = self._FIELD_WIDTH
        except Exception:
            pass
        w.addElement(self.angle_field)
        w.addReturn()

        # --- Edge blur --------------------------------------------------------
        w.addElement(self._row_label("Edge Blur (px):"))
        self.blur_field = mset.UITextFieldInt()
        try:
            self.blur_field.value = DEFAULT_EDGE_BLUR
            self.blur_field.width = self._FIELD_WIDTH
        except Exception:
            pass
        w.addElement(self.blur_field)
        w.addReturn()
        w.addReturn()

        # --- Actions --------------------------------------------------------
        bake_btn = mset.UIButton("Capture Front/Back And Bake")
        bake_btn.onClick = self._on_bake
        w.addElement(bake_btn)
        w.addReturn()
        close_btn = mset.UIButton("Close")
        close_btn.onClick = self._on_close
        w.addElement(close_btn)

    # -- callbacks ----------------------------------------------------------
    def _pick_folder(self):
        try:
            folder = mset.showOpenFolderDialog()
        except Exception as e:
            _err("folder dialog failed: %s" % e)
            return
        if folder:
            self.output_dir = folder
            try:
                self.folder_label.text = folder
            except Exception:
                pass

    def _selected_size(self):
        # a positive custom value overrides the preset dropdown
        try:
            custom = int(self.custom_size_field.value)
            if custom > 0:
                return max(MIN_TEXTURE_SIZE, min(MAX_TEXTURE_SIZE, custom))
        except Exception:
            pass
        try:
            idx = self.size_box.selectedItem
            if 0 <= idx < len(TEXTURE_SIZES):
                return int(TEXTURE_SIZES[idx])
        except Exception:
            pass
        return int(DEFAULT_SIZE)

    def _selected_angle(self):
        try:
            v = float(self.angle_field.value)
            if 0.0 < v < 90.0:
                return v
        except Exception:
            pass
        return DEFAULT_SIDE_MASK_ANGLE

    def _selected_edge_blur(self):
        try:
            v = int(self.blur_field.value)
            if v >= 0:
                return v
        except Exception:
            pass
        return DEFAULT_EDGE_BLUR

    def _on_bake(self):
        try:
            run_bake(self.output_dir, self._selected_size(),
                     self._selected_angle(), self._selected_edge_blur())
        except Exception as e:
            _err("bake failed: %s\n%s" % (e, traceback.format_exc()))
            try:
                mset.showOkDialog("Bake failed:\n%s\n\nSee the log for details." % e)
            except Exception:
                pass

    def _on_close(self):
        try:
            self.window.close()
        except Exception:
            pass
        try:
            mset.shutdownPlugin()
        except Exception:
            pass


# keep a module-level reference so the window is not garbage collected
_ui = None


def main():
    global _ui
    _ui = CaptureBakeUI()


if __name__ == "__main__":
    main()

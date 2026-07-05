"""Standalone test suite for the projbake pure-Python core.

Run with:  python3 tests/test_core.py
(no pytest dependency required, matching Marmoset's bare Python environment)
"""

import math
import os
import struct
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from projbake import linalg as la
from projbake.image import ImageRGBA
from projbake import pngio
from projbake.mesh import SceneMesh, Submesh, group_by_material
from projbake import bake

_failures = []


def check(cond, msg):
    if cond:
        print("  ok  - %s" % msg)
    else:
        print("  FAIL- %s" % msg)
        _failures.append(msg)


def approx(a, b, eps=1e-6):
    return abs(a - b) <= eps


# ---------------------------------------------------------------------------
def test_linalg():
    print("[linalg]")
    m = la.euler_to_matrix(0, 0, 0)
    check(m == la.IDENTITY, "euler(0,0,0) == identity")

    p = la.transform_point(la.rot_y(180), (1.0, 2.0, 3.0))
    check(approx(p[0], -1.0) and approx(p[1], 2.0) and approx(p[2], -3.0),
          "rot_y(180) maps (1,2,3) -> (-1,2,-3)")

    p = la.transform_point(la.rot_y(90), (1.0, 0.0, 0.0))
    check(approx(p[0], 0.0, 1e-6) and approx(p[2], -1.0, 1e-6),
          "rot_y(90) maps +X -> -Z (right-handed, Y-up)")

    # rigid inverse round-trips
    W = la.compose_trs((3, -2, 5), (10, 20, 30))
    inv = la.rigid_inverse(W)
    q = la.transform_point(la.mat_mul(inv, W), (1.0, 2.0, 3.0))
    check(approx(q[0], 1.0, 1e-5) and approx(q[1], 2.0, 1e-5) and approx(q[2], 3.0, 1e-5),
          "rigid_inverse(W) @ W == identity")


def test_camera():
    print("[camera]")
    cam = la.Camera((0, 0, 5), (0, 0, 0), 45.0, 100, 100)
    px, py, depth, front = cam.project((0, 0, 0))
    check(approx(px, 50.0, 1e-4) and approx(py, 50.0, 1e-4),
          "point straight ahead projects to image center")
    check(approx(depth, 5.0, 1e-4) and front, "depth == 5, in front")

    px, _, _, _ = cam.project((1, 0, 0))
    check(px > 50.0, "world +X projects to the right half")
    _, py, _, _ = cam.project((0, 1, 0))
    check(py < 50.0, "world +Y projects to the top half")

    # behind the camera
    _, _, _, front = cam.project((0, 0, 10))
    check(not front, "point behind camera reported not in front")

    fwd = cam.forward()
    check(approx(fwd[0], 0, 1e-6) and approx(fwd[1], 0, 1e-6) and approx(fwd[2], -1, 1e-6),
          "camera at rot(0) looks down -Z")


def test_image_sampling():
    print("[image]")
    img = ImageRGBA(2, 2)
    img.set(0, 0, (10, 0, 0, 255))
    img.set(1, 0, (20, 0, 0, 255))
    img.set(0, 1, (30, 0, 0, 255))
    img.set(1, 1, (40, 0, 0, 255))
    # center of the image -> average of the four
    s = img.sample_bilinear(1.0, 1.0)
    check(approx(s[0], 25.0, 1e-6), "bilinear center == mean(10,20,30,40)=25")
    # exact texel center
    s = img.sample_bilinear(0.5, 0.5)
    check(approx(s[0], 10.0, 1e-6), "bilinear at texel(0,0) center == 10")
    check(img.sample_bilinear(-5, -5) is None, "sample outside bounds -> None")


def _make_png_with_filter(width, height, rows_rgba, filter_type):
    """Build a PNG (color type 6, bit depth 8) applying one filter to every row.
    Used to exercise the decoder's un-filter paths."""
    stride = width * 4
    raw = bytearray()

    def paeth(a, b, c):
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        if pa <= pb and pa <= pc:
            return a
        return b if pb <= pc else c

    prev = bytearray(stride)
    for y in range(height):
        cur = bytearray(rows_rgba[y])
        enc = bytearray(stride)
        for i in range(stride):
            a = cur[i - 4] if i >= 4 else 0
            b = prev[i]
            c = prev[i - 4] if i >= 4 else 0
            if filter_type == 0:
                enc[i] = cur[i]
            elif filter_type == 1:
                enc[i] = (cur[i] - a) & 255
            elif filter_type == 2:
                enc[i] = (cur[i] - b) & 255
            elif filter_type == 3:
                enc[i] = (cur[i] - ((a + b) >> 1)) & 255
            elif filter_type == 4:
                enc[i] = (cur[i] - paeth(a, b, c)) & 255
        raw.append(filter_type)
        raw += enc
        prev = cur

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw)))
            + chunk(b"IEND", b""))


def test_pngio():
    print("[pngio]")
    img = ImageRGBA(5, 3)
    for y in range(3):
        for x in range(5):
            img.set(x, y, (x * 40, y * 80, (x + y) * 20, 255 if (x + y) % 2 else 128))
    tmp = os.path.join(tempfile.gettempdir(), "projbake_rt.png")
    pngio.save_png(tmp, img)
    back = pngio.load_png(tmp)
    same = all(img.get(x, y) == back.get(x, y) for y in range(3) for x in range(5))
    check(back.width == 5 and back.height == 3, "round-trip preserves size")
    check(same, "round-trip preserves all RGBA pixels (filter 0)")

    # decoder against every filter type
    rows = [bytes([((x * 7 + y * 13) & 255) for x in range(4 * 4)]) for y in range(4)]
    for ft in range(5):
        data = _make_png_with_filter(4, 4, rows, ft)
        dec = pngio.load_png_bytes(data)
        ok = all(bytes(dec.get(x, y)) == rows[y][x * 4:x * 4 + 4]
                 for y in range(4) for x in range(4))
        check(ok, "decode PNG with filter type %d" % ft)


def _make_palette_png(width, height, indices, palette_bytes, trns=None):
    """Build an 8-bit palette (color type 3) PNG with filter 0 rows."""
    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter None
        for x in range(width):
            raw.append(indices[y * width + x])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"PLTE", palette_bytes)
    if trns is not None:
        out += chunk(b"tRNS", trns)
    out += chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")
    return out


def test_pngio_palette():
    print("[pngio: palette]")
    # 2x1 image, palette of 2 colors
    pal = bytes([255, 0, 0, 0, 255, 0])  # index0=red, index1=green
    data = _make_palette_png(2, 1, [0, 1], pal)
    dec = pngio.load_png_bytes(data)
    check(dec.get(0, 0) == (255, 0, 0, 255) and dec.get(1, 0) == (0, 255, 0, 255),
          "valid palette PNG decodes to correct RGBA")

    # index out of range must NOT crash (malformed PLTE, 5 bytes -> 1 entry)
    bad_pal = bytes([255, 0, 0, 0, 99])  # only 1 full entry
    bad = _make_palette_png(2, 1, [0, 1], bad_pal)
    try:
        d2 = pngio.load_png_bytes(bad)
        ok = d2.get(0, 0) == (255, 0, 0, 255)  # index 1 -> fallback, no crash
        check(ok, "out-of-range palette index handled without crashing")
    except Exception as e:  # pragma: no cover
        check(False, "out-of-range palette index raised %r" % e)

    # missing PLTE -> clean ValueError, not IndexError/TypeError
    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))
    raw = bytes([0, 0, 0])  # 1 row filter + 2 indices? use 2x1
    ihdr = struct.pack(">IIBBBBB", 2, 1, 8, 3, 0, 0, 0)
    nolut = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
             + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    try:
        pngio.load_png_bytes(nolut)
        check(False, "missing PLTE should raise")
    except ValueError:
        check(True, "missing PLTE raises a clean ValueError")
    except Exception as e:  # pragma: no cover
        check(False, "missing PLTE raised wrong type %r" % e)


def _quad(name, corners, uvs, normal, material="mat", pos=(0, 0, 0)):
    """Two-triangle quad. corners: 4 world points (CCW), uvs: 4 (u,v)."""
    verts = []
    for c in corners:
        verts += [c[0], c[1], c[2]]
    norms = list(normal) * 4
    uvflat = []
    for uv in uvs:
        uvflat += [uv[0], uv[1]]
    tris = [0, 1, 2, 0, 2, 3]
    return SceneMesh(name, verts, tris, uvflat, norms,
                     position=pos, submeshes=[Submesh(material, 0, 6)])


def test_bake_front_quad():
    print("[bake: front quad]")
    # A quad in the z=0 plane facing +Z (toward a camera on +Z).
    corners = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    quad = _quad("q", corners, uvs, (0, 0, 1))
    cam = la.Camera((0, 0, 5), (0, 0, 0), 60.0, 128, 128)

    # front capture: left half red, right half green (in screen space)
    front = ImageRGBA(128, 128, fill=(0, 0, 0, 0))
    for y in range(128):
        for x in range(128):
            if x < 64:
                front.set(x, y, (255, 0, 0, 255))
            else:
                front.set(x, y, (0, 255, 0, 255))
    back = ImageRGBA(128, 128, fill=(0, 0, 0, 0))

    groups = group_by_material([quad])
    res = bake.bake_scene([quad], front, back, cam, (0, 0, 0),
                          256, 80.0, log=None)
    out = res["mat"]
    # The quad faces +Z, camera on +Z -> fully front-facing, no masking.
    # world +X (right) projects to screen right (green); +X maps to u=1 side.
    # Sample near u=0.1 (left, world -X -> screen left -> red) and u=0.9 (green).
    # texel for (u,v): tx=u*256, ty=(1-v)*256
    left = out.get(int(0.15 * 256), 128)
    right = out.get(int(0.85 * 256), 128)
    check(left[3] == 255 and left[0] > 200 and left[1] < 60,
          "left side of UV (world -X) sampled RED from front capture")
    check(right[3] == 255 and right[1] > 200 and right[0] < 60,
          "right side of UV (world +X) sampled GREEN from front capture")
    # coverage: most texels opaque
    opaque = sum(1 for i in range(256 * 256) if out.data[i * 4 + 3] == 255)
    check(opaque > 256 * 256 * 0.85, "front-facing quad mostly covered (%d texels)" % opaque)


def test_bake_side_masking():
    print("[bake: side masking]")
    # A quad whose normal is perpendicular to the view (faces +X) -> grazing ->
    # must be masked (transparent) for both front and back.
    corners = [(0, -1, -1), (0, -1, 1), (0, 1, 1), (0, 1, -1)]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    quad = _quad("side", corners, uvs, (1, 0, 0))
    cam = la.Camera((0, 0, 5), (0, 0, 0), 60.0, 128, 128)
    front = ImageRGBA(128, 128, fill=(255, 255, 255, 255))
    back = ImageRGBA(128, 128, fill=(255, 255, 255, 255))
    groups = group_by_material([quad])
    res = bake.bake_scene([quad], front, back, cam, (0, 0, 0),
                          128, 75.0)
    out = res["mat"]
    opaque = sum(1 for i in range(128 * 128) if out.data[i * 4 + 3] != 0)
    check(opaque == 0, "side-facing quad fully masked (0 opaque texels, got %d)" % opaque)


def test_bake_back_capture():
    print("[bake: back capture]")
    # A quad facing -Z (away from camera). After a 180-deg Y turntable about the
    # origin it faces the camera, so it must be sampled from the BACK capture.
    corners = [(1, -1, 0), (-1, -1, 0), (-1, 1, 0), (1, 1, 0)]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    quad = _quad("bk", corners, uvs, (0, 0, -1))
    cam = la.Camera((0, 0, 5), (0, 0, 0), 60.0, 128, 128)
    front = ImageRGBA(128, 128, fill=(0, 0, 0, 0))       # empty
    back = ImageRGBA(128, 128, fill=(0, 0, 255, 255))    # solid blue
    groups = group_by_material([quad])
    res = bake.bake_scene([quad], front, back, cam, (0, 0, 0),
                          128, 80.0)
    out = res["mat"]
    center = out.get(64, 64)
    check(center[3] == 255 and center[2] > 200 and center[0] < 60,
          "back-facing quad sampled BLUE from back capture after turntable")


def test_bake_occlusion():
    print("[bake: occlusion / overlap]")
    # Two quads facing the camera at different depths but SAME UVs. The near quad
    # (z=1, red front) should win; the far quad (z=-1) is occluded so its texels
    # must not pick up the near quad's colour incorrectly -- it should be masked
    # (transparent) because it is behind nearer geometry.
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    near = _quad("near", [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], uvs,
                 (0, 0, 1), material="near")
    far = _quad("far", [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)], uvs,
                (0, 0, 1), material="far")
    cam = la.Camera((0, 0, 5), (0, 0, 0), 60.0, 128, 128)
    front = ImageRGBA(128, 128, fill=(200, 100, 50, 255))
    back = ImageRGBA(128, 128, fill=(0, 0, 0, 0))
    meshes = [near, far]
    groups = group_by_material(meshes)
    res = bake.bake_scene(meshes, front, back, cam, (0, 0, 0), 128, 85.0)
    near_out = res["near"]
    far_out = res["far"]
    near_center = near_out.get(64, 64)
    far_center = far_out.get(64, 64)
    check(near_center[3] == 255, "near quad receives colour (visible)")
    check(far_center[3] == 0, "far quad center masked (occluded by near quad)")


def test_cross_object_occlusion():
    print("[bake: cross-object occlusion (feather-in-slot bleed)]")
    from projbake import bake as _bake
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    # head plane at z=0; feather plane JUST in front (z=0.02, within depth bias)
    head = _quad("head", [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)],
                 uvs, (0, 0, 1), material="head")
    feather = _quad("feather",
                    [(-0.3, -0.3, 0.02), (0.3, -0.3, 0.02),
                     (0.3, 0.3, 0.02), (-0.3, 0.3, 0.02)],
                    uvs, (0, 0, 1), material="feather")
    cam = la.Camera((0, 0, 5), (0, 0, 0), 60.0, 128, 128)
    # front capture: green everywhere, RED where the feather is (screen center)
    front = ImageRGBA(128, 128, fill=(0, 200, 0, 255))
    for y in range(54, 74):
        for x in range(54, 74):
            front.set(x, y, (220, 0, 0, 255))
    back = ImageRGBA(128, 128, fill=(0, 0, 0, 0))
    res = _bake.bake_scene([head, feather], front, back, cam, (0, 0, 0), 128, 85.0)

    head_img = res["head"]
    center = head_img.get(64, 64)   # UV center -> world (0,0,0) -> behind feather
    check(center[3] == 0,
          "head texel behind feather masked (id-buffer stops cross-object bleed) a=%d"
          % center[3])
    corner = head_img.get(14, 64)   # off to the side -> not behind feather -> green
    check(corner[3] == 255 and corner[1] > 150 and corner[0] < 80,
          "head texel outside the feather sampled green (%s)" % (corner[:3],))
    feather_img = res["feather"]
    fc = feather_img.get(64, 64)
    check(fc[3] == 255 and fc[0] > 150,
          "feather texel samples its own red (%s)" % (fc[:3],))


def test_edge_blur():
    print("[postprocess: edge_blur]")
    from projbake import postprocess
    img = ImageRGBA(32, 32, fill=(0, 0, 0, 0))
    for y in range(8, 24):
        for x in range(8, 24):
            img.set(x, y, (200, 50, 50, 255))
    out = postprocess.edge_blur(img, 3)
    check(out.get(16, 16) == (200, 50, 50, 255),
          "edge_blur keeps deep interior exact")
    e = out.get(25, 16)  # just outside the original opaque edge (was transparent)
    check(0 < e[3] < 255, "edge_blur feathers alpha past the edge (a=%d)" % e[3])
    check(e[0] > e[2] and e[0] > 60,
          "edge_blur bleeds real colour outward, no black halo (rgb=%s)" % (e[:3],))
    check(out.get(0, 0)[3] == 0,
          "edge_blur leaves far transparent region transparent")
    check(postprocess.edge_blur(img, 0).data == img.data,
          "edge_blur radius 0 is a no-op copy")


def main():
    test_linalg()
    test_camera()
    test_image_sampling()
    test_pngio()
    test_pngio_palette()
    test_bake_front_quad()
    test_bake_side_masking()
    test_bake_back_capture()
    test_bake_occlusion()
    test_cross_object_occlusion()
    test_edge_blur()
    print()
    if _failures:
        print("FAILED %d check(s):" % len(_failures))
        for f in _failures:
            print("  - %s" % f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

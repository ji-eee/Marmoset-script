"""End-to-end demo + round-trip correctness test on a UV sphere.

Builds a UV sphere (a stand-in for the character head), forward-renders synthetic
"front" and "back" captures whose colour ENCODES each surface point's UV, runs the
reverse-projection bake, and then checks that a baked texel at UV (u,v) recovers
the encoded colour for that same (u,v) -- i.e. the projection round-trips.

Also writes front.png / back.png / baked.png to a chosen output dir so the result
can be viewed. Run:  python3 tests/make_sphere_demo.py [out_dir]
"""

import math
import os
import sys
from array import array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from projbake import linalg as la
from projbake.image import ImageRGBA
from projbake import pngio
from projbake.mesh import SceneMesh, Submesh, group_by_material
from projbake import bake

INF = float("inf")


def build_uv_sphere(rings=48, sectors=96, radius=1.0):
    verts, norms, uvs, tris = [], [], [], []
    for r in range(rings + 1):
        v = r / rings
        theta = v * math.pi            # 0..pi from top
        st, ct = math.sin(theta), math.cos(theta)
        for s in range(sectors + 1):
            u = s / sectors
            phi = u * 2.0 * math.pi
            x = st * math.cos(phi)
            y = ct
            z = st * math.sin(phi)
            verts += [x * radius, y * radius, z * radius]
            norms += [x, y, z]
            uvs += [u, v]
    row = sectors + 1
    for r in range(rings):
        for s in range(sectors):
            a = r * row + s
            b = a + 1
            c = a + row
            d = c + 1
            tris += [a, b, c, b, d, c]
    return SceneMesh("sphere", verts, tris, uvs, norms,
                     submeshes=[Submesh("head", 0, len(tris))])


def color_front(u, v):
    return (int(u * 255) & 255, int(v * 255) & 255, 60, 255)


def color_back(u, v):
    return (60, int(u * 255) & 255, int(v * 255) & 255, 255)


def render_capture(mesh, camera, color_fn, extra, euler_order="ZYX"):
    """Forward-render: colour each pixel by the interpolated UV of the nearest
    surface. Emulates a Marmoset capture whose shading encodes UV."""
    W, H = camera.width, camera.height
    img = ImageRGBA(W, H, fill=(0, 0, 0, 0))
    depth = array("f", [INF]) * (W * H)
    data = img.data
    persp = camera.mode != "orthographic"
    for _mat, (p0, p1, p2), _n, (uv0, uv1, uv2) in mesh.iter_world_triangles(extra, euler_order):
        a = camera.project(p0)
        b = camera.project(p1)
        c = camera.project(p2)
        if not (a[3] and b[3] and c[3]):
            continue
        x0, y0, d0 = a[0], a[1], a[2]
        x1, y1, d1 = b[0], b[1], b[2]
        x2, y2, d2 = c[0], c[1], c[2]
        minx = max(0, int(math.floor(min(x0, x1, x2))))
        maxx = min(W - 1, int(math.ceil(max(x0, x1, x2))))
        miny = max(0, int(math.floor(min(y0, y1, y2))))
        maxy = min(H - 1, int(math.ceil(max(y0, y1, y2))))
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        inv = 1.0 / area
        id0, id1, id2 = (1.0 / d0, 1.0 / d1, 1.0 / d2) if persp else (d0, d1, d2)
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
                interp = w0 * id0 + w1 * id1 + w2 * id2
                d = (1.0 / interp) if persp else interp
                idx = rb + px
                if d < depth[idx]:
                    depth[idx] = d
                    u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                    vv = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                    col = color_fn(u, vv)
                    o = idx * 4
                    data[o], data[o + 1], data[o + 2], data[o + 3] = col
    return img


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_demo_out")
    os.makedirs(out_dir, exist_ok=True)

    size = 512
    sphere = build_uv_sphere()
    cam = la.Camera((0, 0, 4), (0, 0, 0), 45.0, size, size)
    rc = bake.turntable_matrix((0, 0, 0), 180.0)

    print("rendering front capture...")
    front = render_capture(sphere, cam, color_front, None)
    print("rendering back capture (rotated 180 Y)...")
    back = render_capture(sphere, cam, color_back, rc)

    pngio.save_png(os.path.join(out_dir, "front.png"), front)
    pngio.save_png(os.path.join(out_dir, "back.png"), back)

    print("baking...")
    groups = group_by_material([sphere])
    res = bake.bake_scene([sphere], groups, front, back, cam, (0, 0, 0),
                          size, 70.0, log=print)
    baked = res["head"]
    pngio.save_png(os.path.join(out_dir, "baked.png"), baked)
    print("wrote front/back/baked .png to", out_dir)

    # ---- round-trip correctness on sampled texels -------------------------
    failures = 0
    checked = 0

    def texel_of(u, v):
        return int(u * size), int((1.0 - v) * size)

    # Front hemisphere is around phi=pi/2 -> u=0.25 (facing +Z where camera is).
    # Back hemisphere around phi=3pi/2 -> u=0.75.
    for (u, v, fn, tag) in [
        (0.25, 0.5, color_front, "front"),
        (0.25, 0.35, color_front, "front"),
        (0.25, 0.65, color_front, "front"),
        (0.75, 0.5, color_back, "back"),
        (0.75, 0.4, color_back, "back"),
    ]:
        tx, ty = texel_of(u, v)
        got = baked.get(tx, ty)
        exp = fn(u, v)
        if got[3] != 255:
            print("  MISS %s uv(%.2f,%.2f): texel transparent" % (tag, u, v))
            failures += 1
            continue
        derr = max(abs(got[i] - exp[i]) for i in range(3))
        checked += 1
        status = "ok" if derr <= 14 else "FAIL"
        if derr > 14:
            failures += 1
        print("  %s %s uv(%.2f,%.2f): got=%s exp=%s maxerr=%d"
              % (status, tag, u, v, got[:3], exp[:3], derr))

    # Silhouette side (phi=0 -> u=0, faces +X, grazing) should be masked.
    tx, ty = texel_of(0.0, 0.5)
    side = baked.get(min(tx, size - 1), ty)
    print("  side uv(0.0,0.5) alpha=%d (expect 0/masked)" % side[3])
    if side[3] != 0:
        failures += 1

    # coverage stats
    opaque = sum(1 for i in range(size * size) if baked.data[i * 4 + 3] == 255)
    print("  opaque texels: %d / %d (%.1f%%)"
          % (opaque, size * size, 100.0 * opaque / (size * size)))

    print()
    if failures:
        print("DEMO ROUND-TRIP: %d FAILURE(S)" % failures)
        sys.exit(1)
    print("DEMO ROUND-TRIP OK (%d texels within tolerance)" % checked)


if __name__ == "__main__":
    main()

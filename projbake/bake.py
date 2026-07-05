"""Reverse-projection (gather) baker.

Given the front & back captures (rendered from the same camera, the back one after
rotating the objects 180 degrees about the world Y axis through the scene center),
this walks every mesh triangle in UV space and, for each output texel:

  1. finds the texel's 3D world position P and normal N (barycentric interpolation);
  2. classifies it as FRONT-facing, BACK-facing, or grazing SIDE;
  3. projects it into the appropriate capture, rejecting texels that are occluded
     (behind nearer geometry -> the "overlap bleed" case) or that land on the
     transparent background;
  4. writes the sampled colour, keeping the most head-on sample when UVs overlap.

Grazing/side texels (normal angle to camera beyond ``side_mask_angle``) are left
transparent, which is exactly the "masked side" the user wants in the PNG.

No numpy: buffers are ``array`` objects; hot loops bind locals. See
docs/projection-bake-design.md for the math and rationale.
"""

import math
from array import array

from . import linalg as la
from .image import ImageRGBA

INF = float("inf")


def turntable_matrix(center, angle_deg=180.0):
    """World-space rotation of ``angle_deg`` about the +Y axis through ``center``."""
    C = center
    return la.mat_mul_chain(
        la.translation(C),
        la.rot_y(angle_deg),
        la.translation((-C[0], -C[1], -C[2])),
    )


def _tri_screen(camera, p0, p1, p2):
    """Project a triangle; return (verts, ok) where verts is list of
    (px,py,depth) and ok is False if any vertex is behind the camera."""
    a = camera.project(p0)
    b = camera.project(p1)
    c = camera.project(p2)
    if not (a[3] and b[3] and c[3]):
        return None, False
    return ((a[0], a[1], a[2]), (b[0], b[1], b[2]), (c[0], c[1], c[2])), True


def build_depth_buffer(meshes, camera, extra=None, euler_order="YXZ"):
    """Rasterise all triangles into a nearest-depth (z) buffer of the capture size.

    Depth is the positive camera-space distance in front of the camera; smaller =
    nearer. Perspective-correct interpolation (1/depth) is used for perspective
    cameras. Returns an ``array('f')`` of length W*H (row-major, top-left origin),
    filled with +inf where nothing was drawn.
    """
    W, H = camera.width, camera.height
    depth = array("f", [INF]) * (W * H)
    persp = camera.mode != "orthographic"

    for m in meshes:
        for _mat, (p0, p1, p2), _norms, _uvs in m.iter_world_triangles(extra, euler_order):
            verts, ok = _tri_screen(camera, p0, p1, p2)
            if not ok:
                continue
            (x0, y0, d0), (x1, y1, d1), (x2, y2, d2) = verts
            _rasterize_depth(depth, W, H, x0, y0, d0, x1, y1, d1, x2, y2, d2, persp)
    return depth


def _rasterize_depth(depth, W, H, x0, y0, d0, x1, y1, d1, x2, y2, d2, persp):
    minx = int(math.floor(min(x0, x1, x2)))
    maxx = int(math.ceil(max(x0, x1, x2)))
    miny = int(math.floor(min(y0, y1, y2)))
    maxy = int(math.ceil(max(y0, y1, y2)))
    if maxx < 0 or minx >= W or maxy < 0 or miny >= H:
        return
    if minx < 0:
        minx = 0
    if miny < 0:
        miny = 0
    if maxx > W - 1:
        maxx = W - 1
    if maxy > H - 1:
        maxy = H - 1

    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(area) < 1e-12:
        return
    inv_area = 1.0 / area

    # perspective-correct: interpolate 1/depth
    if persp:
        id0, id1, id2 = 1.0 / d0, 1.0 / d1, 1.0 / d2
    else:
        id0, id1, id2 = d0, d1, d2

    for py in range(miny, maxy + 1):
        ycen = py + 0.5
        row = py * W
        for px in range(minx, maxx + 1):
            xcen = px + 0.5
            w0 = ((x1 - xcen) * (y2 - ycen) - (x2 - xcen) * (y1 - ycen)) * inv_area
            w1 = ((x2 - xcen) * (y0 - ycen) - (x0 - xcen) * (y2 - ycen)) * inv_area
            w2 = 1.0 - w0 - w1
            if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                continue
            interp = w0 * id0 + w1 * id1 + w2 * id2
            d = (1.0 / interp) if persp else interp
            idx = row + px
            if d < depth[idx]:
                depth[idx] = d


def _visible(depth_buf, W, H, px, py, depth, bias_rel, bias_abs):
    """True if a point at (px,py,depth) is the front-most surface (not occluded)."""
    ix = int(px)
    iy = int(py)
    if ix < 0 or iy < 0 or ix >= W or iy >= H:
        return False
    near = depth_buf[iy * W + ix]
    if near == INF:
        # nothing was rasterised here; treat as visible (edge of silhouette)
        return True
    return depth <= near + (bias_abs + bias_rel * depth)


def bake_group(meshes, front_img, back_img, camera, rc_matrix,
               front_depth, back_depth, tex_size, side_mask_angle_deg,
               euler_order="YXZ", flip_v=True, occlusion_bias_rel=0.01,
               occlusion_bias_abs=1e-4, require_opaque_alpha=8):
    """Bake one material group into an ImageRGBA(tex_size, tex_size).

    ``front_depth``/``back_depth`` are shared, whole-scene depth buffers so that
    other objects correctly occlude this group. ``require_opaque_alpha`` is the
    minimum capture alpha (0-255) for a sample to count as "hit the model" rather
    than the transparent background.
    """
    size = tex_size
    out = ImageRGBA(size, size)
    out_data = out.data
    score = array("f", [-2.0]) * (size * size)

    cos_thresh = math.cos(math.radians(side_mask_angle_deg))
    rc3 = la.mat3_from_mat4(rc_matrix)
    cam_pos = camera.position
    ortho = camera.mode == "orthographic"
    cam_fwd = camera.forward() if ortho else None
    Wc, Hc = camera.width, camera.height

    fsample = front_img.sample_bilinear
    bsample = back_img.sample_bilinear

    for m in meshes:
        for _mat, (p0, p1, p2), (n0, n1, n2), (uv0, uv1, uv2) in \
                m.iter_world_triangles(None, euler_order):
            _bake_triangle(
                out_data, score, size, flip_v,
                p0, p1, p2, n0, n1, n2, uv0, uv1, uv2,
                camera, cam_pos, ortho, cam_fwd, cos_thresh,
                rc_matrix, rc3, front_depth, back_depth, Wc, Hc,
                fsample, bsample, occlusion_bias_rel, occlusion_bias_abs,
                require_opaque_alpha,
            )
    return out


def _bake_triangle(out_data, score, size, flip_v,
                   p0, p1, p2, n0, n1, n2, uv0, uv1, uv2,
                   camera, cam_pos, ortho, cam_fwd, cos_thresh,
                   rc_matrix, rc3, front_depth, back_depth, Wc, Hc,
                   fsample, bsample, bias_rel, bias_abs, req_alpha):
    # UV -> texel coordinates (v-up flipped to top-left row order by default)
    def uv_px(uv):
        u, v = uv
        tx = u * size
        ty = (1.0 - v) * size if flip_v else v * size
        return tx, ty

    ax, ay = uv_px(uv0)
    bx, by = uv_px(uv1)
    cx, cy = uv_px(uv2)

    minx = int(math.floor(min(ax, bx, cx)))
    maxx = int(math.ceil(max(ax, bx, cx)))
    miny = int(math.floor(min(ay, by, cy)))
    maxy = int(math.ceil(max(ay, by, cy)))
    if maxx < 0 or minx >= size or maxy < 0 or miny >= size:
        return
    if minx < 0:
        minx = 0
    if miny < 0:
        miny = 0
    if maxx > size - 1:
        maxx = size - 1
    if maxy > size - 1:
        maxy = size - 1

    area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    if abs(area) < 1e-12:
        return
    inv_area = 1.0 / area

    project = camera.project

    for ty in range(miny, maxy + 1):
        ycen = ty + 0.5
        rowbase = ty * size
        for tx in range(minx, maxx + 1):
            xcen = tx + 0.5
            w0 = ((bx - xcen) * (cy - ycen) - (cx - xcen) * (by - ycen)) * inv_area
            w1 = ((cx - xcen) * (ay - ycen) - (ax - xcen) * (cy - ycen)) * inv_area
            w2 = 1.0 - w0 - w1
            if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                continue

            # interpolate world position & normal (barycentric is affine-exact
            # for a planar triangle regardless of UV distortion)
            P = (w0 * p0[0] + w1 * p1[0] + w2 * p2[0],
                 w0 * p0[1] + w1 * p1[1] + w2 * p2[1],
                 w0 * p0[2] + w1 * p1[2] + w2 * p2[2])
            N = (w0 * n0[0] + w1 * n1[0] + w2 * n2[0],
                 w0 * n0[1] + w1 * n1[1] + w2 * n2[1],
                 w0 * n0[2] + w1 * n1[2] + w2 * n2[2])
            N = la.v_normalize(N)

            if ortho:
                vcam = (-cam_fwd[0], -cam_fwd[1], -cam_fwd[2])
            else:
                vcam = la.v_normalize(la.v_sub(cam_pos, P))
            facing_front = N[0] * vcam[0] + N[1] * vcam[1] + N[2] * vcam[2]

            use_back = False
            facing = facing_front
            if facing_front >= cos_thresh:
                sx, sy, depth, in_front = project(P)
                if not in_front:
                    continue
                if not _visible(front_depth, Wc, Hc, sx, sy, depth, bias_rel, bias_abs):
                    continue
                sample = fsample(sx, sy)
            else:
                # try the back capture using the rotated pose
                Pr = la.transform_point(rc_matrix, P)
                Nr = la.v_normalize(la.transform_dir3(rc3, N))
                if ortho:
                    vcam_r = (-cam_fwd[0], -cam_fwd[1], -cam_fwd[2])
                else:
                    vcam_r = la.v_normalize(la.v_sub(cam_pos, Pr))
                facing_back = Nr[0] * vcam_r[0] + Nr[1] * vcam_r[1] + Nr[2] * vcam_r[2]
                if facing_back < cos_thresh:
                    continue  # grazing / side -> masked (leave transparent)
                sx, sy, depth, in_front = project(Pr)
                if not in_front:
                    continue
                if not _visible(back_depth, Wc, Hc, sx, sy, depth, bias_rel, bias_abs):
                    continue
                sample = bsample(sx, sy)
                use_back = True
                facing = facing_back

            if sample is None:
                continue
            if sample[3] < req_alpha:
                continue  # projected onto transparent background -> skip

            idx = rowbase + tx
            if facing <= score[idx]:
                continue  # a more head-on sample already won this texel
            score[idx] = facing
            o = idx * 4
            out_data[o] = int(sample[0] + 0.5) & 255
            out_data[o + 1] = int(sample[1] + 0.5) & 255
            out_data[o + 2] = int(sample[2] + 0.5) & 255
            out_data[o + 3] = 255


def bake_scene(all_meshes, groups, front_img, back_img, camera, rot_center,
               tex_size, side_mask_angle_deg, euler_order="YXZ", flip_v=True,
               occlusion_bias_rel=0.01, occlusion_bias_abs=1e-4,
               require_opaque_alpha=8, log=None):
    """Top level: build shared depth buffers, bake every material group.

    Returns ``{material_name: ImageRGBA}``. ``log`` is an optional callable.
    """
    def _log(msg):
        if log:
            log(msg)

    rc = turntable_matrix(rot_center, 180.0)
    _log("Building front depth buffer (%dx%d)..." % (camera.width, camera.height))
    front_depth = build_depth_buffer(all_meshes, camera, None, euler_order)
    _log("Building back depth buffer...")
    back_depth = build_depth_buffer(all_meshes, camera, rc, euler_order)

    results = {}
    for mat, meshes in groups.items():
        _log("Baking material group %r (%d meshes)..." % (mat, len(meshes)))
        results[mat] = bake_group(
            meshes, front_img, back_img, camera, rc, front_depth, back_depth,
            tex_size, side_mask_angle_deg, euler_order, flip_v,
            occlusion_bias_rel, occlusion_bias_abs, require_opaque_alpha,
        )
    return results

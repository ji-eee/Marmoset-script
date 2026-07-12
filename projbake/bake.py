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
from .mesh import group_by_material

INF = float("inf")


def turntable_matrix(center, angle_deg=180.0):
    """World-space rotation of ``angle_deg`` about the +Y axis through ``center``."""
    C = center
    return la.mat_mul_chain(
        la.translation(C),
        la.rot_y(angle_deg),
        la.translation((-C[0], -C[1], -C[2])),
    )


def build_depth_id(meshes, camera, extra=None, euler_order="YXZ"):
    """Rasterise all triangles into a nearest-depth buffer AND a nearest-object
    id buffer of the capture size.

    Depth is the positive camera-space distance in front of the camera; smaller =
    nearer. Perspective-correct interpolation (1/depth) is used for perspective
    cameras. The id buffer stores, per pixel, the ``bake_id`` of the nearest mesh
    (or -1 where nothing was drawn), so the baker can reject a texel when a
    DIFFERENT object is the front-most surface at its pixel (cross-object bleed).
    Returns ``(depth_array_f, id_array_i)``, each length W*H, row-major.
    """
    W, H = camera.width, camera.height
    depth = array("f", [INF]) * (W * H)
    ids = array("i", [-1]) * (W * H)
    persp = camera.mode != "orthographic"

    # Hoist the camera projection so the per-vertex project() (3x per triangle)
    # avoids the method dispatch + to_camera_space tuple. The view matrix is
    # rigid (w == 1.0), so the homogeneous divide is dropped without changing a
    # bit; arithmetic order matches camera.project exactly.
    vm = camera.view
    v0 = vm[0]; v1 = vm[1]; v2 = vm[2]; v3 = vm[3]
    v4 = vm[4]; v5 = vm[5]; v6 = vm[6]; v7 = vm[7]
    v8 = vm[8]; v9 = vm[9]; v10 = vm[10]; v11 = vm[11]
    tan_h = camera._tan_half_h
    tan_v = camera._tan_half_v
    if not persp:
        half_h = camera.ortho_scale * 0.5
        half_w = half_h * camera.aspect

    def pr(px, py, pz):
        xc = v0 * px + v1 * py + v2 * pz + v3
        yc = v4 * px + v5 * py + v6 * pz + v7
        zc = v8 * px + v9 * py + v10 * pz + v11
        depth_ = -zc
        in_front = depth_ > 1e-9
        if persp:
            dd = depth_ if in_front else 1e-9
            ndc_x = (xc / dd) / tan_h if tan_h else 0.0
            ndc_y = (yc / dd) / tan_v if tan_v else 0.0
        else:
            ndc_x = xc / half_w if half_w else 0.0
            ndc_y = yc / half_h if half_h else 0.0
        sx = (ndc_x * 0.5 + 0.5) * W
        sy = (0.5 - ndc_y * 0.5) * H
        return sx, sy, depth_, in_front

    for idx, m in enumerate(meshes):
        mid = m.bake_id if getattr(m, "bake_id", None) is not None else idx
        for _mat, (p0, p1, p2), _norms, _uvs in m.iter_world_triangles(extra, euler_order):
            x0, y0, d0, f0 = pr(p0[0], p0[1], p0[2])
            if not f0:
                continue
            x1, y1, d1, f1 = pr(p1[0], p1[1], p1[2])
            if not f1:
                continue
            x2, y2, d2, f2 = pr(p2[0], p2[1], p2[2])
            if not f2:
                continue
            _rasterize_depth(depth, ids, mid, W, H,
                             x0, y0, d0, x1, y1, d1, x2, y2, d2, persp)
    return depth, ids


def _rasterize_depth(depth, ids, mid, W, H, x0, y0, d0, x1, y1, d1, x2, y2, d2, persp):
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
                ids[idx] = mid


def _visible(depth_buf, id_buf, W, H, px, py, depth, mesh_id, bias_rel, bias_abs):
    """True if a point at (px,py,depth) belonging to ``mesh_id`` is the front-most
    surface at that pixel (not occluded by itself or another object).

    Two rejections:
      * a DIFFERENT object is nearest here -> cross-object bleed (e.g. a feather
        showing through a slot painting onto the head). Rejected via id_buf.
      * the same object has a nearer surface here -> self-occlusion. Rejected via
        the depth comparison.
    """
    ix = int(px)
    iy = int(py)
    if ix < 0 or iy < 0 or ix >= W or iy >= H:
        return False
    j = iy * W + ix
    near = depth_buf[j]
    if near == INF:
        # nothing was rasterised here; let the background-alpha test decide
        return True
    if id_buf[j] != mesh_id:
        return False  # another object is in front -> occluded
    return depth <= near + (bias_abs + bias_rel * depth)


def bake_group(meshes, front_img, back_img, camera, rc_matrix,
               front_depth, front_id, back_depth, back_id,
               tex_size, side_mask_angle_deg,
               euler_order="YXZ", flip_v=True, occlusion_bias_rel=0.01,
               occlusion_bias_abs=1e-4, require_opaque_alpha=8, unmasked=False,
               single_view=False):
    """Bake one material group into an ImageRGBA(tex_size, tex_size).

    ``front_depth``/``front_id`` (and back) are shared, whole-scene depth and
    object-id buffers so that other objects correctly occlude this group.
    ``require_opaque_alpha`` is the minimum capture alpha (0-255) for a sample to
    count as "hit the model" rather than the transparent background.

    ``unmasked=True`` is best-effort smear mode for the "full" output: no side
    masking, no occlusion test and no background rejection — every texel takes
    whichever of the front/back samples is more head-on and usable. Texels with
    no usable sample stay transparent for a later fill pass.
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

    # alpha-weighted sampling avoids dark fringes from the transparent background
    fsample = front_img.sample_bilinear_weighted
    bsample = back_img.sample_bilinear_weighted

    for m in meshes:
        mid = m.bake_id if getattr(m, "bake_id", None) is not None else -1
        for _mat, (p0, p1, p2), (n0, n1, n2), (uv0, uv1, uv2) in \
                m.iter_world_triangles(None, euler_order):
            _bake_triangle(
                out_data, score, size, flip_v, mid,
                p0, p1, p2, n0, n1, n2, uv0, uv1, uv2,
                camera, cam_pos, ortho, cam_fwd, cos_thresh,
                rc_matrix, rc3, front_depth, front_id, back_depth, back_id, Wc, Hc,
                fsample, bsample, occlusion_bias_rel, occlusion_bias_abs,
                require_opaque_alpha, unmasked, single_view,
            )
    return out


def _bake_triangle(out_data, score, size, flip_v, mesh_id,
                   p0, p1, p2, n0, n1, n2, uv0, uv1, uv2,
                   camera, cam_pos, ortho, cam_fwd, cos_thresh,
                   rc_matrix, rc3, front_depth, front_id, back_depth, back_id, Wc, Hc,
                   fsample, bsample, bias_rel, bias_abs, req_alpha, unmasked=False,
                   single_view=False):
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

    # ---- per-triangle hoist of everything the hot pixel loop needs ---------
    # camera.project() is inlined below (perspective and orthographic paths
    # preserved). The camera view matrix is rigid (last row 0,0,0,1) so
    # transform_point's homogeneous divide is by 1.0 and is dropped without
    # changing any bit; likewise the turntable matrix rc_matrix. v_normalize,
    # v_sub, transform_dir3 and _visible are inlined as local closures. Every
    # arithmetic expression and its left-to-right order is preserved, so the
    # output is bit-for-bit identical to the readable per-call version.
    sqrt = math.sqrt
    vm = camera.view
    v0 = vm[0]; v1 = vm[1]; v2 = vm[2]; v3 = vm[3]
    v4 = vm[4]; v5 = vm[5]; v6 = vm[6]; v7 = vm[7]
    v8 = vm[8]; v9 = vm[9]; v10 = vm[10]; v11 = vm[11]
    tan_h = camera._tan_half_h
    tan_v = camera._tan_half_v
    if ortho:
        half_h = camera.ortho_scale * 0.5
        half_w = half_h * camera.aspect
        ncf0 = -cam_fwd[0]; ncf1 = -cam_fwd[1]; ncf2 = -cam_fwd[2]
    rm0 = rc_matrix[0]; rm1 = rc_matrix[1]; rm2 = rc_matrix[2]; rm3 = rc_matrix[3]
    rm4 = rc_matrix[4]; rm5 = rc_matrix[5]; rm6 = rc_matrix[6]; rm7 = rc_matrix[7]
    rm8 = rc_matrix[8]; rm9 = rc_matrix[9]; rm10 = rc_matrix[10]; rm11 = rc_matrix[11]
    c0 = rc3[0]; c1 = rc3[1]; c2 = rc3[2]
    c3 = rc3[3]; c4 = rc3[4]; c5 = rc3[5]
    c6 = rc3[6]; c7 = rc3[7]; c8 = rc3[8]
    cpx = cam_pos[0]; cpy = cam_pos[1]; cpz = cam_pos[2]

    p0x = p0[0]; p0y = p0[1]; p0z = p0[2]
    p1x = p1[0]; p1y = p1[1]; p1z = p1[2]
    p2x = p2[0]; p2y = p2[1]; p2z = p2[2]
    n0x = n0[0]; n0y = n0[1]; n0z = n0[2]
    n1x = n1[0]; n1y = n1[1]; n1z = n1[2]
    n2x = n2[0]; n2y = n2[1]; n2z = n2[2]

    def proj(px, py, pz):
        """Inlined camera.project(); returns (sx, sy, depth, in_front)."""
        xc = v0 * px + v1 * py + v2 * pz + v3
        yc = v4 * px + v5 * py + v6 * pz + v7
        zc = v8 * px + v9 * py + v10 * pz + v11
        depth = -zc
        in_front = depth > 1e-9
        if ortho:
            ndc_x = xc / half_w if half_w else 0.0
            ndc_y = yc / half_h if half_h else 0.0
        else:
            dd = depth if in_front else 1e-9
            ndc_x = (xc / dd) / tan_h if tan_h else 0.0
            ndc_y = (yc / dd) / tan_v if tan_v else 0.0
        sx = (ndc_x * 0.5 + 0.5) * Wc
        sy = (0.5 - ndc_y * 0.5) * Hc
        return sx, sy, depth, in_front

    def visible(depth_buf, id_buf, sxp, syp, depthp):
        """Inlined _visible()."""
        ix = int(sxp)
        iy = int(syp)
        if ix < 0 or iy < 0 or ix >= Wc or iy >= Hc:
            return False
        j = iy * Wc + ix
        near = depth_buf[j]
        if near == INF:
            return True
        if id_buf[j] != mesh_id:
            return False
        return depthp <= near + (bias_abs + bias_rel * depthp)

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
            Px = w0 * p0x + w1 * p1x + w2 * p2x
            Py = w0 * p0y + w1 * p1y + w2 * p2y
            Pz = w0 * p0z + w1 * p1z + w2 * p2z
            Nx = w0 * n0x + w1 * n1x + w2 * n2x
            Ny = w0 * n0y + w1 * n1y + w2 * n2y
            Nz = w0 * n0z + w1 * n1z + w2 * n2z
            # N = v_normalize(N)
            nl = sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
            if nl < 1e-20:
                nx = 0.0; ny = 0.0; nz = 0.0
            else:
                ninv = 1.0 / nl
                nx = Nx * ninv; ny = Ny * ninv; nz = Nz * ninv

            if ortho:
                vcx = ncf0; vcy = ncf1; vcz = ncf2
            else:
                # vcam = v_normalize(v_sub(cam_pos, P))
                ex = cpx - Px; ey = cpy - Py; ez = cpz - Pz
                el = sqrt(ex * ex + ey * ey + ez * ez)
                if el < 1e-20:
                    vcx = 0.0; vcy = 0.0; vcz = 0.0
                else:
                    einv = 1.0 / el
                    vcx = ex * einv; vcy = ey * einv; vcz = ez * einv
            facing_front = nx * vcx + ny * vcy + nz * vcz

            use_back = False
            facing = facing_front
            if unmasked:
                # best-effort smear: whichever of front/back is more head-on and
                # yields a usable (non-empty) sample; no occlusion/background
                # rejection. Misses stay transparent for the fill pass.
                sample = None
                sx, sy, depth, in_front = proj(Px, Py, Pz)
                if in_front:
                    s = fsample(sx, sy)
                    if s is not None and s[3] >= 1.0:
                        sample = s
                if not single_view:
                    # Pr = transform_point(rc_matrix, P)
                    Prx = rm0 * Px + rm1 * Py + rm2 * Pz + rm3
                    Pry = rm4 * Px + rm5 * Py + rm6 * Pz + rm7
                    Prz = rm8 * Px + rm9 * Py + rm10 * Pz + rm11
                    # Nr = v_normalize(transform_dir3(rc3, N))
                    trx = c0 * nx + c1 * ny + c2 * nz
                    try_ = c3 * nx + c4 * ny + c5 * nz
                    trz = c6 * nx + c7 * ny + c8 * nz
                    rl = sqrt(trx * trx + try_ * try_ + trz * trz)
                    if rl < 1e-20:
                        nrx = 0.0; nry = 0.0; nrz = 0.0
                    else:
                        rinv = 1.0 / rl
                        nrx = trx * rinv; nry = try_ * rinv; nrz = trz * rinv
                    if ortho:
                        vrx = ncf0; vry = ncf1; vrz = ncf2
                    else:
                        erx = cpx - Prx; ery = cpy - Pry; erz = cpz - Prz
                        erl = sqrt(erx * erx + ery * ery + erz * erz)
                        if erl < 1e-20:
                            vrx = 0.0; vry = 0.0; vrz = 0.0
                        else:
                            erinv = 1.0 / erl
                            vrx = erx * erinv; vry = ery * erinv; vrz = erz * erinv
                    facing_back = nrx * vrx + nry * vry + nrz * vrz
                    if sample is None or facing_back > facing:
                        sxb, syb, depthb, in_front_b = proj(Prx, Pry, Prz)
                        if in_front_b:
                            s2 = bsample(sxb, syb)
                            if s2 is not None and s2[3] >= 1.0:
                                sample = s2
                                facing = facing_back
                if sample is None:
                    continue
            elif facing_front >= cos_thresh:
                sx, sy, depth, in_front = proj(Px, Py, Pz)
                if not in_front:
                    continue
                if not visible(front_depth, front_id, sx, sy, depth):
                    continue
                sample = fsample(sx, sy)
            else:
                # not facing the current view
                if single_view:
                    continue  # single-view mode has no back capture -> masked
                # try the back capture using the rotated pose
                # Pr = transform_point(rc_matrix, P)
                Prx = rm0 * Px + rm1 * Py + rm2 * Pz + rm3
                Pry = rm4 * Px + rm5 * Py + rm6 * Pz + rm7
                Prz = rm8 * Px + rm9 * Py + rm10 * Pz + rm11
                # Nr = v_normalize(transform_dir3(rc3, N))
                trx = c0 * nx + c1 * ny + c2 * nz
                try_ = c3 * nx + c4 * ny + c5 * nz
                trz = c6 * nx + c7 * ny + c8 * nz
                rl = sqrt(trx * trx + try_ * try_ + trz * trz)
                if rl < 1e-20:
                    nrx = 0.0; nry = 0.0; nrz = 0.0
                else:
                    rinv = 1.0 / rl
                    nrx = trx * rinv; nry = try_ * rinv; nrz = trz * rinv
                if ortho:
                    vrx = ncf0; vry = ncf1; vrz = ncf2
                else:
                    erx = cpx - Prx; ery = cpy - Pry; erz = cpz - Prz
                    erl = sqrt(erx * erx + ery * ery + erz * erz)
                    if erl < 1e-20:
                        vrx = 0.0; vry = 0.0; vrz = 0.0
                    else:
                        erinv = 1.0 / erl
                        vrx = erx * erinv; vry = ery * erinv; vrz = erz * erinv
                facing_back = nrx * vrx + nry * vry + nrz * vrz
                if facing_back < cos_thresh:
                    continue  # grazing / side -> masked (leave transparent)
                sx, sy, depth, in_front = proj(Prx, Pry, Prz)
                if not in_front:
                    continue
                if not visible(back_depth, back_id, sx, sy, depth):
                    continue
                sample = bsample(sx, sy)
                use_back = True
                facing = facing_back

            if sample is None:
                continue
            if not unmasked and sample[3] < req_alpha:
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


def bake_pair(meshes, front_img, back_img, camera, rc_matrix,
              front_depth, front_id, back_depth, back_id,
              tex_size, masked_side_angle_deg,
              euler_order="YXZ", flip_v=True, occlusion_bias_rel=0.01,
              occlusion_bias_abs=1e-4, require_opaque_alpha=8, single_view=False):
    """Bake the standard MASKED and FULL(unmasked) outputs in a SINGLE pass.

    The two outputs share almost all per-texel work (projection, sampling,
    facing), so producing them together instead of walking every triangle twice
    is markedly faster. The write rules for each output are applied to the same
    samples exactly as :func:`bake_group` would with ``unmasked=False`` and
    ``unmasked=True`` respectively, so the result is bit-for-bit identical to
    running bake_group twice. Returns ``(masked_img, full_img)``.
    """
    size = tex_size
    out_m = ImageRGBA(size, size)
    out_f = ImageRGBA(size, size)
    md = out_m.data
    fdt = out_f.data
    score_m = array("f", [-2.0]) * (size * size)
    score_f = array("f", [-2.0]) * (size * size)

    cos_thresh = math.cos(math.radians(masked_side_angle_deg))
    rc3 = la.mat3_from_mat4(rc_matrix)
    cam_pos = camera.position
    ortho = camera.mode == "orthographic"
    cam_fwd = camera.forward() if ortho else None
    Wc, Hc = camera.width, camera.height
    fsample = front_img.sample_bilinear_weighted
    bsample = back_img.sample_bilinear_weighted

    for m in meshes:
        mid = m.bake_id if getattr(m, "bake_id", None) is not None else -1
        for _mat, (p0, p1, p2), (n0, n1, n2), (uv0, uv1, uv2) in \
                m.iter_world_triangles(None, euler_order):
            _bake_triangle_pair(
                md, score_m, fdt, score_f, size, flip_v, mid,
                p0, p1, p2, n0, n1, n2, uv0, uv1, uv2,
                camera, cam_pos, ortho, cam_fwd, cos_thresh,
                rc_matrix, rc3, front_depth, front_id, back_depth, back_id, Wc, Hc,
                fsample, bsample, occlusion_bias_rel, occlusion_bias_abs,
                require_opaque_alpha, single_view,
            )
    return out_m, out_f


def _bake_triangle_pair(md, score_m, fdt, score_f, size, flip_v, mesh_id,
                        p0, p1, p2, n0, n1, n2, uv0, uv1, uv2,
                        camera, cam_pos, ortho, cam_fwd, cos_thresh,
                        rc_matrix, rc3, front_depth, front_id, back_depth, back_id,
                        Wc, Hc, fsample, bsample, bias_rel, bias_abs, req_alpha,
                        single_view=False):
    """Fused masked+full rasterizer for one triangle. Mirrors _bake_triangle's
    inlined math exactly; the front sample and (when either output needs it) the
    back sample are computed once and fed to both write rules."""
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

    sqrt = math.sqrt
    vm = camera.view
    v0 = vm[0]; v1 = vm[1]; v2 = vm[2]; v3 = vm[3]
    v4 = vm[4]; v5 = vm[5]; v6 = vm[6]; v7 = vm[7]
    v8 = vm[8]; v9 = vm[9]; v10 = vm[10]; v11 = vm[11]
    tan_h = camera._tan_half_h
    tan_v = camera._tan_half_v
    if ortho:
        half_h = camera.ortho_scale * 0.5
        half_w = half_h * camera.aspect
        ncf0 = -cam_fwd[0]; ncf1 = -cam_fwd[1]; ncf2 = -cam_fwd[2]
    rm0 = rc_matrix[0]; rm1 = rc_matrix[1]; rm2 = rc_matrix[2]; rm3 = rc_matrix[3]
    rm4 = rc_matrix[4]; rm5 = rc_matrix[5]; rm6 = rc_matrix[6]; rm7 = rc_matrix[7]
    rm8 = rc_matrix[8]; rm9 = rc_matrix[9]; rm10 = rc_matrix[10]; rm11 = rc_matrix[11]
    c0 = rc3[0]; c1 = rc3[1]; c2 = rc3[2]
    c3 = rc3[3]; c4 = rc3[4]; c5 = rc3[5]
    c6 = rc3[6]; c7 = rc3[7]; c8 = rc3[8]
    cpx = cam_pos[0]; cpy = cam_pos[1]; cpz = cam_pos[2]

    p0x = p0[0]; p0y = p0[1]; p0z = p0[2]
    p1x = p1[0]; p1y = p1[1]; p1z = p1[2]
    p2x = p2[0]; p2y = p2[1]; p2z = p2[2]
    n0x = n0[0]; n0y = n0[1]; n0z = n0[2]
    n1x = n1[0]; n1y = n1[1]; n1z = n1[2]
    n2x = n2[0]; n2y = n2[1]; n2z = n2[2]

    def proj(px, py, pz):
        xc = v0 * px + v1 * py + v2 * pz + v3
        yc = v4 * px + v5 * py + v6 * pz + v7
        zc = v8 * px + v9 * py + v10 * pz + v11
        depth = -zc
        in_front = depth > 1e-9
        if ortho:
            ndc_x = xc / half_w if half_w else 0.0
            ndc_y = yc / half_h if half_h else 0.0
        else:
            dd = depth if in_front else 1e-9
            ndc_x = (xc / dd) / tan_h if tan_h else 0.0
            ndc_y = (yc / dd) / tan_v if tan_v else 0.0
        sx = (ndc_x * 0.5 + 0.5) * Wc
        sy = (0.5 - ndc_y * 0.5) * Hc
        return sx, sy, depth, in_front

    def visible(depth_buf, id_buf, sxp, syp, depthp):
        ix = int(sxp)
        iy = int(syp)
        if ix < 0 or iy < 0 or ix >= Wc or iy >= Hc:
            return False
        j = iy * Wc + ix
        near = depth_buf[j]
        if near == INF:
            return True
        if id_buf[j] != mesh_id:
            return False
        return depthp <= near + (bias_abs + bias_rel * depthp)

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

            Px = w0 * p0x + w1 * p1x + w2 * p2x
            Py = w0 * p0y + w1 * p1y + w2 * p2y
            Pz = w0 * p0z + w1 * p1z + w2 * p2z
            Nx = w0 * n0x + w1 * n1x + w2 * n2x
            Ny = w0 * n0y + w1 * n1y + w2 * n2y
            Nz = w0 * n0z + w1 * n1z + w2 * n2z
            nl = sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
            if nl < 1e-20:
                nx = 0.0; ny = 0.0; nz = 0.0
            else:
                ninv = 1.0 / nl
                nx = Nx * ninv; ny = Ny * ninv; nz = Nz * ninv
            if ortho:
                vcx = ncf0; vcy = ncf1; vcz = ncf2
            else:
                ex = cpx - Px; ey = cpy - Py; ez = cpz - Pz
                el = sqrt(ex * ex + ey * ey + ez * ez)
                if el < 1e-20:
                    vcx = 0.0; vcy = 0.0; vcz = 0.0
                else:
                    einv = 1.0 / el
                    vcx = ex * einv; vcy = ey * einv; vcz = ez * einv
            facing_front = nx * vcx + ny * vcy + nz * vcz

            # front projection + sample (shared by both outputs)
            fsx, fsy, fdepth, f_in_front = proj(Px, Py, Pz)
            front_sample = fsample(fsx, fsy) if f_in_front else None

            idx = rowbase + tx

            # back pose + facing, and the back sample when either output needs it
            back_sample = None
            b_in_front = False
            bsx = 0.0; bsy = 0.0; bdepth = 0.0
            facing_back = -2.0
            if not single_view:
                Prx = rm0 * Px + rm1 * Py + rm2 * Pz + rm3
                Pry = rm4 * Px + rm5 * Py + rm6 * Pz + rm7
                Prz = rm8 * Px + rm9 * Py + rm10 * Pz + rm11
                trx = c0 * nx + c1 * ny + c2 * nz
                try_ = c3 * nx + c4 * ny + c5 * nz
                trz = c6 * nx + c7 * ny + c8 * nz
                rl = sqrt(trx * trx + try_ * try_ + trz * trz)
                if rl < 1e-20:
                    nrx = 0.0; nry = 0.0; nrz = 0.0
                else:
                    rinv = 1.0 / rl
                    nrx = trx * rinv; nry = try_ * rinv; nrz = trz * rinv
                if ortho:
                    vrx = ncf0; vry = ncf1; vrz = ncf2
                else:
                    erx = cpx - Prx; ery = cpy - Pry; erz = cpz - Prz
                    erl = sqrt(erx * erx + ery * ery + erz * erz)
                    if erl < 1e-20:
                        vrx = 0.0; vry = 0.0; vrz = 0.0
                    else:
                        erinv = 1.0 / erl
                        vrx = erx * erinv; vry = ery * erinv; vrz = erz * erinv
                facing_back = nrx * vrx + nry * vry + nrz * vrz
                front_usable_full = front_sample is not None and front_sample[3] >= 1.0
                masked_wants_back = facing_front < cos_thresh and facing_back >= cos_thresh
                full_wants_back = (not front_usable_full) or (facing_back > facing_front)
                if masked_wants_back or full_wants_back:
                    bsx, bsy, bdepth, b_in_front = proj(Prx, Pry, Prz)
                    if b_in_front:
                        back_sample = bsample(bsx, bsy)

            # ---- MASKED output (== bake_group unmasked=False) ----
            if facing_front >= cos_thresh:
                if f_in_front and facing_front > score_m[idx] and \
                        visible(front_depth, front_id, fsx, fsy, fdepth):
                    s = front_sample
                    if s is not None and s[3] >= req_alpha:
                        score_m[idx] = facing_front
                        o = idx * 4
                        md[o] = int(s[0] + 0.5) & 255
                        md[o + 1] = int(s[1] + 0.5) & 255
                        md[o + 2] = int(s[2] + 0.5) & 255
                        md[o + 3] = 255
            elif (not single_view) and facing_back >= cos_thresh:
                if b_in_front and facing_back > score_m[idx] and \
                        visible(back_depth, back_id, bsx, bsy, bdepth):
                    s = back_sample
                    if s is not None and s[3] >= req_alpha:
                        score_m[idx] = facing_back
                        o = idx * 4
                        md[o] = int(s[0] + 0.5) & 255
                        md[o + 1] = int(s[1] + 0.5) & 255
                        md[o + 2] = int(s[2] + 0.5) & 255
                        md[o + 3] = 255

            # ---- FULL output (== bake_group unmasked=True) ----
            sample = None
            facing = facing_front
            if front_sample is not None and front_sample[3] >= 1.0:
                sample = front_sample
            if not single_view:
                if sample is None or facing_back > facing:
                    if b_in_front and back_sample is not None and back_sample[3] >= 1.0:
                        sample = back_sample
                        facing = facing_back
            if sample is not None and facing > score_f[idx]:
                score_f[idx] = facing
                o = idx * 4
                fdt[o] = int(sample[0] + 0.5) & 255
                fdt[o + 1] = int(sample[1] + 0.5) & 255
                fdt[o + 2] = int(sample[2] + 0.5) & 255
                fdt[o + 3] = 255


def _standard_pair(variants):
    """If ``variants`` is exactly one masked + one unmasked variant (the case the
    plugin always uses), return ``(masked_var, full_var)`` so they can be fused
    into one pass; otherwise ``None`` (fall back to per-variant baking)."""
    if len(variants) != 2:
        return None
    masked_var = full_var = None
    for v in variants:
        if v.get("unmasked"):
            full_var = v
        else:
            masked_var = v
    if masked_var is not None and full_var is not None:
        return masked_var, full_var
    return None


def bake_variants(all_meshes, front_img, back_img, camera, rot_center,
                  tex_size, variants, euler_order="YXZ", flip_v=True,
                  occlusion_bias_rel=0.01, occlusion_bias_abs=1e-4,
                  require_opaque_alpha=8, log=None, single_view=False):
    """Build shared depth+id buffers once, then bake every material group for
    each requested variant.

    ``variants`` is a list of dicts ``{"name": str, "side_mask_angle": float}``
    with an optional ``"unmasked": True`` (see bake_group).
    ``single_view=True`` bakes only the current-view ``front_img`` (no turntable
    back capture): camera-facing texels are painted, the rest are masked. In
    that mode ``back_img``/``rot_center`` are ignored.
    Returns ``{material_name: {variant_name: ImageRGBA}}``.
    """
    def _log(msg):
        if log:
            log(msg)

    # Assign stable per-object ids used by the occlusion id-buffer, THEN build the
    # per-material views so they inherit those ids (order matters).
    for i, m in enumerate(all_meshes):
        m.bake_id = i
    groups = group_by_material(all_meshes)

    if single_view:
        # one current-view capture, no turntable; bake_group skips the back path
        # so rc/back buffers are unused placeholders.
        rc = la.IDENTITY
        back_used = front_img
        _log("Building current-view depth/id buffer (%dx%d)..."
             % (camera.width, camera.height))
        front_depth, front_id = build_depth_id(all_meshes, camera, None, euler_order)
        back_depth, back_id = front_depth, front_id
    else:
        rc = turntable_matrix(rot_center, 180.0)
        back_used = back_img
        _log("Building front depth/id buffer (%dx%d)..." % (camera.width, camera.height))
        front_depth, front_id = build_depth_id(all_meshes, camera, None, euler_order)
        _log("Building back depth/id buffer...")
        back_depth, back_id = build_depth_id(all_meshes, camera, rc, euler_order)

    pair = _standard_pair(variants)
    if pair is not None:
        masked_var, full_var = pair
        results = {}
        for mat, meshes in groups.items():
            _log("Baking %r (masked+full fused, %d meshes)..." % (mat, len(meshes)))
            m_img, f_img = bake_pair(
                meshes, front_img, back_used, camera, rc,
                front_depth, front_id, back_depth, back_id,
                tex_size, masked_var["side_mask_angle"], euler_order, flip_v,
                occlusion_bias_rel, occlusion_bias_abs, require_opaque_alpha,
                single_view,
            )
            results[mat] = {masked_var["name"]: m_img, full_var["name"]: f_img}
        return results

    results = {}
    for mat, meshes in groups.items():
        results[mat] = {}
        for var in variants:
            _log("Baking %r variant %r (%d meshes)..."
                 % (mat, var["name"], len(meshes)))
            results[mat][var["name"]] = bake_group(
                meshes, front_img, back_used, camera, rc,
                front_depth, front_id, back_depth, back_id,
                tex_size, var["side_mask_angle"], euler_order, flip_v,
                occlusion_bias_rel, occlusion_bias_abs, require_opaque_alpha,
                unmasked=var.get("unmasked", False), single_view=single_view,
            )
    return results


def bake_scene(all_meshes, front_img, back_img, camera, rot_center,
               tex_size, side_mask_angle_deg, euler_order="YXZ", flip_v=True,
               occlusion_bias_rel=0.01, occlusion_bias_abs=1e-4,
               require_opaque_alpha=8, log=None):
    """Single-variant bake. Returns ``{material: ImageRGBA}``."""
    res = bake_variants(
        all_meshes, front_img, back_img, camera, rot_center, tex_size,
        [{"name": "out", "side_mask_angle": side_mask_angle_deg}],
        euler_order, flip_v, occlusion_bias_rel, occlusion_bias_abs,
        require_opaque_alpha, log,
    )
    return {mat: v["out"] for mat, v in res.items()}


def bake_target_variants(source_meshes, target_mesh, front_img, back_img, camera,
                         rot_center, tex_size, variants, euler_order="YXZ",
                         flip_v=True, occlusion_bias_rel=0.05, occlusion_bias_abs=1e-3,
                         require_opaque_alpha=8, log=None, single_view=False):
    """Project the captures of ``source_meshes`` onto ``target_mesh``'s UVs.

    Whereas :func:`bake_variants` bakes each mesh onto its own UVs, this bakes a
    DIFFERENT target: the front/back captures are of the source geometry (e.g.
    a multi-mesh mid-poly face+hair) and the result is written into the target
    mesh's UV layout (e.g. a single low-poly mesh). When the target *is* one of
    the source meshes the two coincide and this reduces to a plain self-bake.

    Occlusion: the depth/id buffers are built from the source. All source meshes
    (and the target) share ONE bake id, so a target texel is rejected only when
    a NEARER source surface covers its pixel (self/scene occlusion) — never
    falsely rejected as "a different object is in front", which the per-object
    id scheme of bake_variants would do here. The occlusion bias defaults are
    looser than bake_variants so a low-poly target inset from the mid-poly
    surface still samples it; widen further if a low-poly cage sits deeper.

    Returns ``{material_name: {variant_name: ImageRGBA}}`` for the target's
    materials.
    """
    def _log(msg):
        if log:
            log(msg)

    SHARED_ID = 0
    for m in source_meshes:
        m.bake_id = SHARED_ID
    target_mesh.bake_id = SHARED_ID
    # per-material views of the target inherit the shared id (view() copies it)
    groups = group_by_material([target_mesh])

    if single_view:
        rc = la.IDENTITY
        back_used = front_img
        _log("Building current-view depth/id buffer from %d source mesh(es) (%dx%d)..."
             % (len(source_meshes), camera.width, camera.height))
        front_depth, front_id = build_depth_id(source_meshes, camera, None, euler_order)
        back_depth, back_id = front_depth, front_id
    else:
        rc = turntable_matrix(rot_center, 180.0)
        back_used = back_img
        _log("Building front depth/id buffer from %d source mesh(es) (%dx%d)..."
             % (len(source_meshes), camera.width, camera.height))
        front_depth, front_id = build_depth_id(source_meshes, camera, None, euler_order)
        _log("Building back depth/id buffer from source...")
        back_depth, back_id = build_depth_id(source_meshes, camera, rc, euler_order)

    pair = _standard_pair(variants)
    if pair is not None:
        masked_var, full_var = pair
        results = {}
        for mat, meshes in groups.items():
            _log("Baking target material %r (masked+full fused)..." % (mat,))
            m_img, f_img = bake_pair(
                meshes, front_img, back_used, camera, rc,
                front_depth, front_id, back_depth, back_id,
                tex_size, masked_var["side_mask_angle"], euler_order, flip_v,
                occlusion_bias_rel, occlusion_bias_abs, require_opaque_alpha,
                single_view,
            )
            results[mat] = {masked_var["name"]: m_img, full_var["name"]: f_img}
        return results

    results = {}
    for mat, meshes in groups.items():
        results[mat] = {}
        for var in variants:
            _log("Baking target material %r variant %r..." % (mat, var["name"]))
            results[mat][var["name"]] = bake_group(
                meshes, front_img, back_used, camera, rc,
                front_depth, front_id, back_depth, back_id,
                tex_size, var["side_mask_angle"], euler_order, flip_v,
                occlusion_bias_rel, occlusion_bias_abs, require_opaque_alpha,
                unmasked=var.get("unmasked", False), single_view=single_view,
            )
    return results

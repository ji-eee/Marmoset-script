"""Minimal 3D linear algebra for the projection baker (pure Python, no numpy).

Vectors are plain 3-tuples ``(x, y, z)``. Matrices are 4x4 stored row-major as a
flat tuple/list of 16 floats (row 0 first). Points transform as ``M @ [x,y,z,1]``.

Coordinate-system / Euler convention
-------------------------------------
Marmoset Toolbag uses a right-handed, **Y-up** world. ``TransformObject.rotation``
is ``[rx, ry, rz]`` in **degrees**. The exact intrinsic/extrinsic order Marmoset
applies is the single riskiest assumption in this plugin, so it is isolated in
:func:`euler_to_matrix` behind the ``order`` argument and documented in
``docs/marmoset-api-notes.md``. Everything downstream (camera view matrix and the
180-degree turntable rotation) reuses this one function, so if real testing in
Marmoset shows a mismatch, correcting the convention here fixes it everywhere.

Default: ``order="YXZ"`` meaning the composed matrix is ``Ry @ Rx @ Rz`` (so a
column vector is rotated about Z, then X, then Y). Y is the *outermost* factor,
which is why a 180-degree world-Y turntable can be applied exactly by simply
adding 180 to the Y angle (``ry += 180`` == pre-multiply by ``Ry(180)``); the
plugin's turntable and camera reuse this one convention. YXZ (yaw about world Y
outermost, then pitch, then roll) is consistent with Marmoset exposing camera
pitch/yaw limits. If real testing shows a mismatch, change ``EULER_ORDER`` in one
place (see ``docs/marmoset-api-notes.md``).
"""

import math

# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_len(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def v_normalize(a):
    L = v_len(a)
    if L < 1e-20:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / L
    return (a[0] * inv, a[1] * inv, a[2] * inv)


# ---------------------------------------------------------------------------
# 4x4 matrices (row-major, 16 floats)
# ---------------------------------------------------------------------------

IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def mat_mul(a, b):
    """Return a @ b for two row-major 4x4 matrices."""
    r = [0.0] * 16
    for i in range(4):
        ai = i * 4
        for j in range(4):
            r[ai + j] = (
                a[ai + 0] * b[0 + j]
                + a[ai + 1] * b[4 + j]
                + a[ai + 2] * b[8 + j]
                + a[ai + 3] * b[12 + j]
            )
    return tuple(r)


def mat_mul_chain(*mats):
    out = IDENTITY
    for m in mats:
        out = mat_mul(out, m)
    return out


def translation(t):
    return (
        1.0, 0.0, 0.0, t[0],
        0.0, 1.0, 0.0, t[1],
        0.0, 0.0, 1.0, t[2],
        0.0, 0.0, 0.0, 1.0,
    )


def scaling(s):
    return (
        s[0], 0.0, 0.0, 0.0,
        0.0, s[1], 0.0, 0.0,
        0.0, 0.0, s[2], 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (
        1.0, 0.0, 0.0, 0.0,
        0.0, c, -s, 0.0,
        0.0, s, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (
        c, 0.0, s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (
        c, -s, 0.0, 0.0,
        s, c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


_ROT_FUNCS = {"X": rot_x, "Y": rot_y, "Z": rot_z}


def euler_to_matrix(rx, ry, rz, order="YXZ"):
    """Build a rotation matrix from Euler angles (degrees).

    ``order`` is the *matrix multiplication* order read left-to-right. ``"YXZ"``
    produces ``Ry @ Rx @ Rz`` which, applied to a column vector, rotates about Z,
    then X, then Y (Y outermost). This single function defines the plugin's
    rotation convention; see the module docstring.
    """
    angles = {"X": rx, "Y": ry, "Z": rz}
    m = IDENTITY
    for axis in order:
        m = mat_mul(m, _ROT_FUNCS[axis](angles[axis]))
    return m


def transform_point(m, p):
    """Apply full 4x4 (including translation) to point p; returns 3-tuple."""
    x, y, z = p
    w = m[12] * x + m[13] * y + m[14] * z + m[15]
    if w == 0.0:
        w = 1.0
    return (
        (m[0] * x + m[1] * y + m[2] * z + m[3]) / w,
        (m[4] * x + m[5] * y + m[6] * z + m[7]) / w,
        (m[8] * x + m[9] * y + m[10] * z + m[11]) / w,
    )


def transform_dir(m, d):
    """Apply the rotational part (ignore translation) to a direction vector."""
    x, y, z = d
    return (
        m[0] * x + m[1] * y + m[2] * z,
        m[4] * x + m[5] * y + m[6] * z,
        m[8] * x + m[9] * y + m[10] * z,
    )


def mat3_from_mat4(m):
    return (m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10])


def transform_dir3(m3, d):
    x, y, z = d
    return (
        m3[0] * x + m3[1] * y + m3[2] * z,
        m3[3] * x + m3[4] * y + m3[5] * z,
        m3[6] * x + m3[7] * y + m3[8] * z,
    )


# ---------------------------------------------------------------------------
# Rigid transforms & inverse (for camera view matrix)
# ---------------------------------------------------------------------------


def compose_trs(position, rotation_deg, scale=(1.0, 1.0, 1.0), euler_order="YXZ"):
    """World matrix = T(position) @ R(rotation) @ S(scale).

    Note: Marmoset's ``pivot`` is handled separately by the mesh module because
    its exact semantics affect only vertex placement, not this generic compose.
    """
    return mat_mul_chain(
        translation(position),
        euler_to_matrix(rotation_deg[0], rotation_deg[1], rotation_deg[2], euler_order),
        scaling(scale),
    )


def rigid_inverse(m):
    """Inverse of a matrix that is translation @ rotation (orthonormal 3x3, no scale).

    For a rigid transform W = T(t) @ R, the inverse is R^T @ T(-t).
    """
    # rotation transpose
    r00, r01, r02 = m[0], m[4], m[8]
    r10, r11, r12 = m[1], m[5], m[9]
    r20, r21, r22 = m[2], m[6], m[10]
    tx, ty, tz = m[3], m[7], m[11]
    # -R^T @ t
    nx = -(r00 * tx + r01 * ty + r02 * tz)
    ny = -(r10 * tx + r11 * ty + r12 * tz)
    nz = -(r20 * tx + r21 * ty + r22 * tz)
    return (
        r00, r01, r02, nx,
        r10, r11, r12, ny,
        r20, r21, r22, nz,
        0.0, 0.0, 0.0, 1.0,
    )


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class Camera:
    """Pinhole (or orthographic) camera reconstructed from Marmoset parameters.

    Marmoset ``CameraObject``: ``fov`` is the *vertical* field of view in degrees,
    the camera looks down its local ``-Z`` axis, local ``+Y`` is up, local ``+X``
    is right (right-handed, Y-up world).

    ``project(world_point)`` returns ``(px, py, depth, in_front)`` where ``px/py``
    are pixel coordinates (origin top-left, matching PNG row order) and ``depth``
    is the positive camera-space distance in front of the camera (larger = farther),
    usable directly as a z-buffer key.
    """

    def __init__(self, position, rotation_deg, fov_vertical_deg,
                 width, height, mode="perspective", ortho_scale=1.0,
                 euler_order="YXZ"):
        self.position = tuple(position)
        self.rotation = tuple(rotation_deg)
        self.fov = float(fov_vertical_deg)
        self.width = int(width)
        self.height = int(height)
        self.mode = mode
        self.ortho_scale = float(ortho_scale)
        self.euler_order = euler_order

        self.aspect = float(width) / float(height) if height else 1.0
        world = compose_trs(position, rotation_deg, (1.0, 1.0, 1.0), euler_order)
        self.view = rigid_inverse(world)
        # tangents of half-fov for perspective
        self._tan_half_v = math.tan(math.radians(self.fov) * 0.5)
        self._tan_half_h = self._tan_half_v * self.aspect

    def to_camera_space(self, world_point):
        return transform_point(self.view, world_point)

    def project(self, world_point):
        """Return (px, py, depth, in_front)."""
        xc, yc, zc = self.to_camera_space(world_point)
        # camera looks down -Z: points in front have zc < 0.
        depth = -zc
        in_front = depth > 1e-9
        if self.mode == "orthographic":
            half_h = self.ortho_scale * 0.5
            half_w = half_h * self.aspect
            ndc_x = xc / half_w if half_w else 0.0
            ndc_y = yc / half_h if half_h else 0.0
        else:
            d = depth if in_front else 1e-9
            ndc_x = (xc / d) / self._tan_half_h if self._tan_half_h else 0.0
            ndc_y = (yc / d) / self._tan_half_v if self._tan_half_v else 0.0
        px = (ndc_x * 0.5 + 0.5) * self.width
        py = (0.5 - ndc_y * 0.5) * self.height  # flip Y for top-left origin
        return (px, py, depth, in_front)

    def forward(self):
        """World-space direction the camera is looking (unit)."""
        world = compose_trs(self.position, self.rotation, (1.0, 1.0, 1.0),
                            self.euler_order)
        # local -Z
        return v_normalize((-world[2], -world[6], -world[10]))

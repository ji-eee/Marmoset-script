"""Mesh geometry container and object->world transform.

Marmoset's ``Mesh`` gives flat float lists (vertices/uvs/normals) and a flat int
list of triangle indices; ``MeshObject`` gives position/rotation/scale/pivot. This
module mirrors that data as plain Python and provides world-space triangle
iteration for the baker. It has no ``mset`` dependency so it is fully testable.

Transform convention (documented in docs/marmoset-api-notes.md):
    world = T(position) @ T(pivot) @ R(rotation) @ S(scale) @ T(-pivot)
so that rotation/scale happen about ``pivot`` and changing the pivot alone (with
rotation identity) does not move the geometry. For the common case
position=0/rotation=0/scale=1 this reduces to identity regardless of pivot.
"""

from . import linalg as la


def _mat3_inverse_transpose(L):
    """Return the inverse-transpose of a 3x3 (row-major 9-tuple), for normals.

    Falls back to the input transpose if the matrix is singular.
    """
    a, b, c, d, e, f, g, h, i = L
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-20:
        return (a, d, g, b, e, h, c, f, i)  # transpose fallback
    inv = 1.0 / det
    # cofactor matrix
    ia = (e * i - f * h) * inv
    ib = (c * h - b * i) * inv
    ic = (b * f - c * e) * inv
    id_ = (f * g - d * i) * inv
    ie = (a * i - c * g) * inv
    if_ = (c * d - a * f) * inv
    ig = (d * h - e * g) * inv
    ih = (b * g - a * h) * inv
    ii = (a * e - b * d) * inv
    # inverse is [[ia,ib,ic],[id,ie,if],[ig,ih,ii]]; return its transpose
    return (ia, id_, ig, ib, ie, ih, ic, if_, ii)


class Submesh:
    """A contiguous run of the index buffer assigned to one material."""

    __slots__ = ("material", "start_index", "index_count")

    def __init__(self, material, start_index, index_count):
        self.material = material
        self.start_index = int(start_index)
        self.index_count = int(index_count)


class SceneMesh:
    """A mesh + its transform + submesh/material assignment.

    ``world_override`` (a 4x4 matrix) is used by the Marmoset plugin to pass the
    full parent-chain world transform it computed; when set it takes precedence
    over position/rotation/scale/pivot. The vertex/triangle/uv/normal arrays are
    held by reference, so per-material "views" (see :meth:`view`) are cheap.
    """

    __slots__ = ("name", "vertices", "triangles", "uvs", "normals",
                 "position", "rotation", "scale", "pivot", "submeshes",
                 "world_override")

    def __init__(self, name, vertices, triangles, uvs, normals=None,
                 position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1),
                 pivot=(0, 0, 0), submeshes=None, world_override=None):
        self.name = name
        self.vertices = vertices        # flat [x,y,z, ...]
        self.triangles = triangles      # flat [i0,i1,i2, ...] indices
        self.uvs = uvs                  # flat [u,v, ...] per vertex
        self.normals = normals          # flat [x,y,z, ...] per vertex or None
        self.position = tuple(position)
        self.rotation = tuple(rotation)
        self.scale = tuple(scale)
        self.pivot = tuple(pivot)
        self.world_override = world_override
        # default: one submesh over the whole index buffer, material None
        if submeshes:
            self.submeshes = submeshes
        else:
            self.submeshes = [Submesh(None, 0, len(triangles))]

    def view(self, submeshes):
        """Return a lightweight copy sharing the same geometry & transform but
        restricted to ``submeshes`` (used to bake one material at a time)."""
        return SceneMesh(self.name, self.vertices, self.triangles, self.uvs,
                         self.normals, self.position, self.rotation, self.scale,
                         self.pivot, submeshes=submeshes,
                         world_override=self.world_override)

    def world_matrix(self, euler_order="YXZ"):
        if self.world_override is not None:
            return self.world_override
        p = self.pivot
        return la.mat_mul_chain(
            la.translation(self.position),
            la.translation(p),
            la.euler_to_matrix(self.rotation[0], self.rotation[1],
                               self.rotation[2], euler_order),
            la.scaling(self.scale),
            la.translation((-p[0], -p[1], -p[2])),
        )

    def iter_world_triangles(self, extra=None, euler_order="YXZ"):
        """Yield triangles transformed to world space.

        ``extra`` is an optional 4x4 world-space matrix pre-multiplied onto the
        world matrix (used for the 180-degree back pose). Yields tuples:
            (material, (p0,p1,p2), (n0,n1,n2), (uv0,uv1,uv2))
        with world-space positions and unit-length world-space normals.
        """
        W = self.world_matrix(euler_order)
        if extra is not None:
            W = la.mat_mul(extra, W)
        L = la.mat3_from_mat4(W)
        Nmat = _mat3_inverse_transpose(L)

        verts = self.vertices
        uvs = self.uvs
        norms = self.normals
        tris = self.triangles

        def wpos(vi):
            b = vi * 3
            return la.transform_point(W, (verts[b], verts[b + 1], verts[b + 2]))

        def wuv(vi):
            b = vi * 2
            if uvs and b + 1 < len(uvs):
                return (uvs[b], uvs[b + 1])
            return (0.0, 0.0)

        def wnorm(vi):
            if norms and vi * 3 + 2 < len(norms):
                b = vi * 3
                return la.v_normalize(
                    la.transform_dir3(Nmat, (norms[b], norms[b + 1], norms[b + 2]))
                )
            return None

        for sm in self.submeshes:
            start = sm.start_index
            end = start + sm.index_count
            if sm.index_count < 0:
                end = len(tris)
            end = min(end, len(tris))
            i = start
            while i + 2 < end:
                a, b, c = tris[i], tris[i + 1], tris[i + 2]
                p0, p1, p2 = wpos(a), wpos(b), wpos(c)
                n0, n1, n2 = wnorm(a), wnorm(b), wnorm(c)
                if n0 is None or n1 is None or n2 is None:
                    # derive a geometric face normal
                    fn = la.v_normalize(
                        la.v_cross(la.v_sub(p1, p0), la.v_sub(p2, p0))
                    )
                    n0 = n1 = n2 = fn
                yield (sm.material, (p0, p1, p2), (n0, n1, n2),
                       (wuv(a), wuv(b), wuv(c)))
            # advance by triangles
                i += 3


def group_by_material(meshes):
    """Group meshes by material name. Returns dict {material_name: [SceneMesh...]}.

    A multi-material mesh is split into per-material *views* so that baking one
    material only rasterises that material's triangle ranges. Meshes with no
    material are grouped under the key ``None``.
    """
    groups = {}
    for m in meshes:
        by_mat = {}
        for sm in m.submeshes:
            by_mat.setdefault(sm.material, []).append(sm)
        for mat, sms in by_mat.items():
            groups.setdefault(mat, []).append(m.view(sms))
    return groups

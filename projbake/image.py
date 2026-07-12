"""ImageRGBA: a simple CPU-side 8-bit RGBA image buffer with sampling.

Row-major, top-to-bottom, 4 bytes per pixel (R,G,B,A). Backed by ``array('B')``
so it stays dependency-free and reasonably compact. Pixel (0,0) is top-left,
matching PNG scanline order and the camera projection in :mod:`projbake.linalg`.
"""

from array import array


class ImageRGBA:
    __slots__ = ("width", "height", "data")

    def __init__(self, width, height, data=None, fill=(0, 0, 0, 0)):
        self.width = int(width)
        self.height = int(height)
        n = self.width * self.height * 4
        if data is None:
            buf = array("B", bytes(fill) * (self.width * self.height))
            if len(buf) != n:  # pragma: no cover - defensive
                buf = array("B", b"\x00" * n)
            self.data = buf
        else:
            if not isinstance(data, array):
                data = array("B", data)
            if len(data) != n:
                raise ValueError(
                    "data length %d != w*h*4 (%d)" % (len(data), n)
                )
            self.data = data

    # -- pixel access -------------------------------------------------------
    def get(self, x, y):
        i = (y * self.width + x) * 4
        d = self.data
        return (d[i], d[i + 1], d[i + 2], d[i + 3])

    def set(self, x, y, rgba):
        i = (y * self.width + x) * 4
        d = self.data
        d[i] = rgba[0] & 255
        d[i + 1] = rgba[1] & 255
        d[i + 2] = rgba[2] & 255
        d[i + 3] = rgba[3] & 255

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    # -- sampling -----------------------------------------------------------
    def sample_nearest(self, px, py):
        """Nearest-neighbour sample at float pixel coords; returns RGBA or None
        if outside the image."""
        x = int(px)
        y = int(py)
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return None
        return self.get(x, y)

    def sample_bilinear(self, px, py):
        """Bilinear sample at float pixel coords (pixel centers at .5).

        Returns an (r,g,b,a) tuple of floats, or ``None`` if the sample center is
        outside the image bounds. Edge texels are clamped.
        """
        # convert to pixel-center space
        fx = px - 0.5
        fy = py - 0.5
        if fx < -0.5 or fy < -0.5 or fx > self.width - 0.5 or fy > self.height - 0.5:
            return None
        x0 = int(fx) if fx >= 0 else int(fx) - 1
        y0 = int(fy) if fy >= 0 else int(fy) - 1
        tx = fx - x0
        ty = fy - y0
        x1 = x0 + 1
        y1 = y0 + 1
        # clamp
        w, h = self.width, self.height
        cx0 = 0 if x0 < 0 else (w - 1 if x0 > w - 1 else x0)
        cx1 = 0 if x1 < 0 else (w - 1 if x1 > w - 1 else x1)
        cy0 = 0 if y0 < 0 else (h - 1 if y0 > h - 1 else y0)
        cy1 = 0 if y1 < 0 else (h - 1 if y1 > h - 1 else y1)
        p00 = self.get(cx0, cy0)
        p10 = self.get(cx1, cy0)
        p01 = self.get(cx0, cy1)
        p11 = self.get(cx1, cy1)
        out = []
        for c in range(4):
            top = p00[c] * (1 - tx) + p10[c] * tx
            bot = p01[c] * (1 - tx) + p11[c] * tx
            out.append(top * (1 - ty) + bot * ty)
        return tuple(out)

    def sample_bilinear_weighted(self, px, py):
        """Like :meth:`sample_bilinear` but the RGB channels are weighted by each
        texel's alpha, so (near-)transparent background texels don't drag the
        colour toward black at silhouette edges. The returned alpha is the plain
        bilinear alpha. Returns ``None`` if the sample center is out of bounds.

        Hot path: this runs once (or twice) per baked texel, so the four
        ``get()`` calls and the 4-iteration accumulation loop are inlined with
        direct ``data`` indexing. The arithmetic and its left-to-right order are
        preserved exactly, so results are bit-for-bit identical to the readable
        form above.
        """
        w = self.width
        h = self.height
        fx = px - 0.5
        fy = py - 0.5
        if fx < -0.5 or fy < -0.5 or fx > w - 0.5 or fy > h - 0.5:
            return None
        x0 = int(fx) if fx >= 0 else int(fx) - 1
        y0 = int(fy) if fy >= 0 else int(fy) - 1
        tx = fx - x0
        ty = fy - y0
        x1 = x0 + 1
        y1 = y0 + 1
        cx0 = 0 if x0 < 0 else (w - 1 if x0 > w - 1 else x0)
        cx1 = 0 if x1 < 0 else (w - 1 if x1 > w - 1 else x1)
        cy0 = 0 if y0 < 0 else (h - 1 if y0 > h - 1 else y0)
        cy1 = 0 if y1 < 0 else (h - 1 if y1 > h - 1 else y1)
        d = self.data
        r0 = cy0 * w
        r1 = cy1 * w
        i00 = (r0 + cx0) * 4
        i10 = (r0 + cx1) * 4
        i01 = (r1 + cx0) * 4
        i11 = (r1 + cx1) * 4
        # weight*alpha per sample, in the same order as the original loop
        wa0 = ((1 - tx) * (1 - ty)) * d[i00 + 3]
        wa1 = (tx * (1 - ty)) * d[i10 + 3]
        wa2 = ((1 - tx) * ty) * d[i01 + 3]
        wa3 = (tx * ty) * d[i11 + 3]
        a_sum = wa0 + wa1 + wa2 + wa3   # plain bilinear alpha == rgb normaliser
        aw_sum = a_sum
        r = d[i00] * wa0 + d[i10] * wa1 + d[i01] * wa2 + d[i11] * wa3
        g = d[i00 + 1] * wa0 + d[i10 + 1] * wa1 + d[i01 + 1] * wa2 + d[i11 + 1] * wa3
        b = d[i00 + 2] * wa0 + d[i10 + 2] * wa1 + d[i01 + 2] * wa2 + d[i11 + 2] * wa3
        if aw_sum > 1e-6:
            r /= aw_sum
            g /= aw_sum
            b /= aw_sum
        return (r, g, b, a_sum)

    def to_bytes(self):
        return self.data.tobytes()

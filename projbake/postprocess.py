"""Post-processing for baked textures: soft (blurred) island edges.

The bake produces hard, sometimes jagged/noisy alpha boundaries where the masked
"side" meets the painted surface. ``edge_blur`` feathers just that border band:

  * alpha is box-blurred so the opaque->transparent transition ramps smoothly;
  * colour is blurred with an ALPHA-WEIGHTED box blur (blur(rgb*a)/blur(a)) so no
    black/transparent halo bleeds in, and colour naturally extends a few pixels
    into the transparent side (acts as edge padding);
  * fully-interior texels (opaque, with an all-opaque neighbourhood) are left
    exactly as-is, so surface detail stays crisp.

Pure Python, separable prefix-sum box blur (O(W*H) per pass). No numpy. Buffers
are float32 and RGB channels are processed one at a time to keep peak memory low
even at 4096x4096.
"""

from array import array

from .image import ImageRGBA


def _box_blur(values, W, H, radius):
    """Separable box blur of a flat float32 array (length W*H), edge-clamped so
    border pixels average only the in-bounds part of the window. Prefix sums use
    Python floats (double) for accuracy; results are stored as float32."""
    if radius <= 0:
        return values
    r = radius
    tmp = array("f", bytes(4 * W * H))
    out = array("f", bytes(4 * W * H))

    for y in range(H):
        base = y * W
        pref = [0.0] * (W + 1)
        s = 0.0
        for x in range(W):
            s += values[base + x]
            pref[x + 1] = s
        for x in range(W):
            a = x - r
            b = x + r
            if a < 0:
                a = 0
            if b > W - 1:
                b = W - 1
            tmp[base + x] = (pref[b + 1] - pref[a]) / (b - a + 1)

    for x in range(W):
        pref = [0.0] * (H + 1)
        s = 0.0
        for y in range(H):
            s += tmp[y * W + x]
            pref[y + 1] = s
        for y in range(H):
            a = y - r
            b = y + r
            if a < 0:
                a = 0
            if b > H - 1:
                b = H - 1
            out[y * W + x] = (pref[b + 1] - pref[a]) / (b - a + 1)
    return out


def edge_blur(img, radius):
    """Return a new ImageRGBA with feathered/soft island edges.

    ``radius`` is the blur radius in pixels (0 -> unchanged copy).
    """
    W, H = img.width, img.height
    src = img.data
    if radius <= 0:
        return ImageRGBA(W, H, data=array("B", src))

    n = W * H
    alpha = array("f", bytes(4 * n))
    for i in range(n):
        alpha[i] = src[i * 4 + 3]
    a_blur = _box_blur(alpha, W, H, radius)   # feathered alpha + rgb normaliser

    out = array("B", bytes(4 * n))
    # per-channel alpha-weighted colour blur (one channel resident at a time)
    for c in range(3):
        chan = array("f", bytes(4 * n))
        for i in range(n):
            o = i * 4
            chan[i] = src[o + c] * src[o + 3]
        cb = _box_blur(chan, W, H, radius)
        for i in range(n):
            ab = a_blur[i]
            if ab > 1e-6:
                v = int(cb[i] / ab + 0.5)
                out[i * 4 + c] = 255 if v > 255 else v

    for i in range(n):
        o = i * 4
        orig_a = src[o + 3]
        ab = a_blur[i]
        if orig_a == 255 and ab >= 254.5:
            # deep interior: restore the original pixel exactly (crisp detail)
            out[o] = src[o]
            out[o + 1] = src[o + 1]
            out[o + 2] = src[o + 2]
            out[o + 3] = 255
            continue
        na = int(ab + 0.5)
        if na <= 0:
            out[o] = out[o + 1] = out[o + 2] = 0  # fully transparent
            out[o + 3] = 0
            continue
        out[o + 3] = 255 if na > 255 else na
    return ImageRGBA(W, H, data=out)

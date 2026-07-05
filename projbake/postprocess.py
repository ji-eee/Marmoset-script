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
from collections import deque

from .image import ImageRGBA


def _nearest_painted_map(d, W, H, alpha_threshold):
    """Multi-source BFS from every texel with alpha >= threshold. Returns a flat
    array mapping each texel index to the index of its nearest painted texel
    (itself if painted), or ``None`` if nothing is painted at all."""
    n = W * H
    src = array("i", [-1]) * n
    q = deque()
    for i in range(n):
        if d[i * 4 + 3] >= alpha_threshold:
            src[i] = i
            q.append(i)
    if not q:
        return None
    while q:
        i = q.popleft()
        s = src[i]
        x = i % W
        if x > 0 and src[i - 1] < 0:
            src[i - 1] = s
            q.append(i - 1)
        if x < W - 1 and src[i + 1] < 0:
            src[i + 1] = s
            q.append(i + 1)
        j = i - W
        if j >= 0 and src[j] < 0:
            src[j] = s
            q.append(j)
        j = i + W
        if j < n and src[j] < 0:
            src[j] = s
            q.append(j)
    return src


def fill_transparent(img, alpha_threshold=1):
    """Flood-fill every transparent texel with the colour of its nearest
    non-transparent texel, then force the whole image opaque.

    Used for the "full" output so it has NO transparent areas: texels the smear
    bake could not reach (behind-camera, off-capture, pure background) inherit
    the nearest painted colour, which also acts as infinite UV padding and kills
    background bleed at UV seams. Mutates ``img`` in place and returns it.
    A fully-transparent image is returned unchanged (nothing to fill from).
    """
    W, H = img.width, img.height
    d = img.data
    src = _nearest_painted_map(d, W, H, alpha_threshold)
    if src is None:
        return img  # nothing painted at all; leave as-is
    for i in range(W * H):
        o = i * 4
        s = src[i]
        if s != i:
            so = s * 4
            d[o] = d[so]
            d[o + 1] = d[so + 1]
            d[o + 2] = d[so + 2]
        d[o + 3] = 255
    return img


def pad_rgb(img, alpha_threshold=1):
    """UV edge padding for MASKED outputs: copy the nearest painted texel's RGB
    into every transparent texel while leaving the ALPHA channel untouched.

    The mask (transparent side regions) is preserved, but texture filtering /
    mipmapping on the model no longer blends in background colour at the alpha
    boundary — this is what made the applied _masked texture look "off" at UV
    island borders. Mutates ``img`` in place and returns it.
    """
    W, H = img.width, img.height
    d = img.data
    src = _nearest_painted_map(d, W, H, alpha_threshold)
    if src is None:
        return img  # nothing painted at all; leave as-is
    for i in range(W * H):
        s = src[i]
        if s != i:
            o = i * 4
            so = s * 4
            d[o] = d[so]
            d[o + 1] = d[so + 1]
            d[o + 2] = d[so + 2]
            # alpha intentionally unchanged: the mask stays a mask
    return img


def composite_max_alpha(base, overlay):
    """Merge ``overlay`` into ``base`` in place: for each texel keep whichever
    contributor is more opaque. Used to combine per-object bakes that share a
    material (their UV islands are normally disjoint, so this just fills each
    object's region; at any overlap the more-covered sample wins). Both images
    must be the same size.
    """
    if base.width != overlay.width or base.height != overlay.height:
        raise ValueError("composite size mismatch")
    bd = base.data
    od = overlay.data
    for i in range(0, len(bd), 4):
        if od[i + 3] > bd[i + 3]:
            bd[i] = od[i]
            bd[i + 1] = od[i + 1]
            bd[i + 2] = od[i + 2]
            bd[i + 3] = od[i + 3]
    return base


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

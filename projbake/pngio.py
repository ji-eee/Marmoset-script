"""Dependency-free PNG read/write using only the Python standard library.

Why hand-rolled: Marmoset's bundled Python has no Pillow/numpy guaranteed, and its
``mset.Image`` exposes no Python-level pixel getter. To read the captured renders
back and to write the final texture, we need a codec that works everywhere. Only
``zlib``/``struct``/``array`` (all stdlib) are used.

Supported on read: non-interlaced PNG, bit depth 8 or 16, color types 0 (gray),
2 (RGB), 3 (palette, 8-bit index), 4 (gray+alpha), 6 (RGBA). All five scanline
filters. Everything is normalised to 8-bit RGBA.
Write: 8-bit RGBA (color type 6), filter 0 (None). Compact and fast; zlib does the
heavy lifting in C.
"""

import struct
import zlib
from array import array

from .image import ImageRGBA

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(raw, width, height, bpp, stride):
    """Reverse PNG scanline filters. Returns a bytearray of height*stride bytes."""
    out = bytearray(height * stride)
    prev = bytearray(stride)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if ftype == 0:
            pass
        elif ftype == 1:  # Sub
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif ftype == 3:  # Average
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                c = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(a, prev[i], c)) & 255
        else:
            raise ValueError("Unknown PNG filter type %d" % ftype)
        base = y * stride
        out[base:base + stride] = line
        prev = line
    return out


def load_png(path):
    with open(path, "rb") as f:
        data = f.read()
    return load_png_bytes(data)


def load_png_bytes(data):
    if data[:8] != _PNG_SIG:
        raise ValueError("Not a PNG file (bad signature)")
    pos = 8
    width = height = bit_depth = color_type = interlace = 0
    idat = bytearray()
    palette = None
    trns = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        cdata = data[pos + 8:pos + 8 + length]
        pos += 12 + length  # length + type + data + crc
        if ctype == b"IHDR":
            (width, height, bit_depth, color_type, _comp, _filt, interlace) = \
                struct.unpack(">IIBBBBB", cdata)
        elif ctype == b"PLTE":
            palette = cdata
        elif ctype == b"tRNS":
            trns = cdata
        elif ctype == b"IDAT":
            idat += cdata
        elif ctype == b"IEND":
            break

    if interlace != 0:
        raise ValueError("Interlaced PNG not supported")
    if bit_depth not in (8, 16):
        # palette images with sub-byte depth are uncommon for renders
        raise ValueError("Unsupported PNG bit depth %d" % bit_depth)

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError("Unsupported PNG color type %d" % color_type)

    if color_type == 3 and (palette is None or len(palette) < 3):
        raise ValueError("Palette PNG missing/empty PLTE chunk")
    pal_entries = (len(palette) // 3) if palette else 0

    sample_bytes = bit_depth // 8
    bpp = channels * sample_bytes
    stride = width * bpp
    raw = zlib.decompress(bytes(idat))
    pixels = _unfilter(raw, width, height, bpp, stride)

    out = array("B", b"\x00" * (width * height * 4))
    # For 16-bit samples we take the big-endian high byte (offset 0 of each
    # sample), so `step` walks whole samples and pixels[soff + k*step] is the
    # 8-bit value of channel k.
    step = sample_bytes

    for y in range(height):
        srow = y * stride
        drow = y * width * 4
        for x in range(width):
            soff = srow + x * bpp
            doff = drow + x * 4
            if color_type == 6:  # RGBA
                out[doff] = pixels[soff]
                out[doff + 1] = pixels[soff + step]
                out[doff + 2] = pixels[soff + 2 * step]
                out[doff + 3] = pixels[soff + 3 * step]
            elif color_type == 2:  # RGB
                out[doff] = pixels[soff]
                out[doff + 1] = pixels[soff + step]
                out[doff + 2] = pixels[soff + 2 * step]
                out[doff + 3] = 255
            elif color_type == 0:  # gray
                g = pixels[soff]
                out[doff] = g
                out[doff + 1] = g
                out[doff + 2] = g
                out[doff + 3] = 255
            elif color_type == 4:  # gray + alpha
                g = pixels[soff]
                out[doff] = g
                out[doff + 1] = g
                out[doff + 2] = g
                out[doff + 3] = pixels[soff + step]
            elif color_type == 3:  # palette
                idx = pixels[soff]
                if idx < pal_entries:
                    pbase = idx * 3
                    out[doff] = palette[pbase]
                    out[doff + 1] = palette[pbase + 1]
                    out[doff + 2] = palette[pbase + 2]
                    out[doff + 3] = trns[idx] if (trns and idx < len(trns)) else 255
                else:
                    # index outside the palette -> opaque magenta (visible flag)
                    out[doff] = 255
                    out[doff + 2] = 255
                    out[doff + 3] = 255
    return ImageRGBA(width, height, out)


def _chunk(tag, payload):
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def save_png(path, img, compress_level=6):
    """Write an ImageRGBA to `path` as 8-bit RGBA PNG."""
    w, h = img.width, img.height
    src = img.data
    row_len = w * 4
    # prepend filter-type 0 to each scanline
    raw = bytearray(h * (row_len + 1))
    for y in range(h):
        o = y * (row_len + 1)
        raw[o] = 0
        s = y * row_len
        raw[o + 1:o + 1 + row_len] = src[s:s + row_len]
    compressed = zlib.compress(bytes(raw), compress_level)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    out = _PNG_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + \
        _chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(out)
    return path

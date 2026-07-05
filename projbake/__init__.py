"""projbake - pure-Python core for the Marmoset Capture Front/Back And Bake plugin.

This package contains NO dependency on Marmoset's ``mset`` module, so every piece
here can be unit-tested on any machine (including the developer's Mac) without
Marmoset installed. The Marmoset-specific glue lives in ``CaptureFrontBackBake.py``.

Modules:
    linalg    - vectors / 4x4 matrices / Euler->matrix / camera projection
    image     - ImageRGBA CPU buffer with bilinear sampling
    pngio     - minimal, dependency-free PNG read/write (stdlib zlib only)
    mesh      - mesh geometry container + object transform -> world space
    bake      - the reverse-projection (gather) baker with occlusion + side mask
    postprocess - baked-texture post-processing (soft/blurred island edges)
"""

__all__ = ["linalg", "image", "pngio", "mesh", "bake", "postprocess"]

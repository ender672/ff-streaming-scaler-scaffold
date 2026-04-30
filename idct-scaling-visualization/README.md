# IDCT pre-scale + Lanczos3 vs pure Lanczos3

A reproducer for the comparison gallery that demonstrates Firefox's
`image.jpeg.dct-scaling.enabled` path (libjpeg-turbo IDCT 1/2, 1/4, 1/8 followed
by Skia Lanczos3) against pure Skia Lanczos3 from a full-resolution decode.

The denom-selection logic in `../JpegIdctScale.h` is a verbatim port from
Firefox's `image/decoders/nsJPEGDecoder.cpp`. The gallery also encodes the
commit's MCU-alignment gate: per-image pages and index cards are tagged
"IDCT path" or "Lanczos only" based on whether Firefox would actually take
the IDCT path on that image.

## Prerequisites

- C++20 compiler (`clang++` or `g++`).
- `libjpeg-turbo` and `libpng` development headers.
- ImageMagick 7 (`magick` and `identify` on PATH).
- Python 3.10 or newer (uses PEP 604 union type hints).

## Build the scalers

From the parent directory:

    make downscale-skia downscale-skia-idct

`generate.py` shells out to those binaries; both must exist at
`../downscale-skia` and `../downscale-skia-idct` relative to this directory.

## Provide a JPEG corpus

Any directory of large JPEGs works. The selection filter keeps images whose
short axis is `>= 1500 px` and picks 100 of them via a deterministic stride.
Set the corpus path via env var or positional arg:

    IDCT_VIZ_SRC_DIR=/path/to/jpegs python3 generate.py
    # or
    python3 generate.py /path/to/jpegs

The original gallery was rendered against Wikimedia's [Picture of the Year
2011 finalists](https://commons.wikimedia.org/wiki/Commons:Picture_of_the_Year/2011).

To pin a different number of images set `N` in the environment (default 100):

    N=20 IDCT_VIZ_SRC_DIR=/path/to/jpegs python3 generate.py

## Output

- `index.html` — top-level gallery, badged with IDCT path / Lanczos only.
- `images/<slug>.html` — per-image page with up to six configs
  (1/2, 1/4, 1/8 each at "barely" and "heavy" Lanczos pressure) and a banner
  saying whether Firefox would take the IDCT path.
- `images/<slug>__<cfg>__{idct,pure,diff10x}.png` — render outputs.
- `thumbs/<slug>.jpg` — index thumbnails.
- `manifest.json` — per-image metadata, including MCU dims and alignment.

PNG renders, thumbs, and per-image HTML are idempotent: re-running with the
same corpus only rewrites missing files plus `index.html` and
`manifest.json`. To force a full re-render, delete `images/` and `thumbs/`.

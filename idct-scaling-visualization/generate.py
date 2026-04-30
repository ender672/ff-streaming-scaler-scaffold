#!/usr/bin/env python3
"""
Generate IDCT-vs-pure-Lanczos visualization pages for ~100 JPEGs from
poty-2011-media, plus an index page that links to each per-image page and
the existing giant-map.jpg page.

Per image, six configs are rendered, mirroring the existing giant-map page:
  1/2 barely, 1/2 heavy, 1/4 barely, 1/4 heavy, 1/8 barely, 1/8 heavy.

For each config we produce:
  <slug>__<cfg>__idct.png      IDCT pre-scale + Lanczos3
  <slug>__<cfg>__pure.png      pure Lanczos3
  <slug>__<cfg>__diff10x.png   abs diff x10

Configs whose target would be too small (< 32px on the short axis) are
skipped for that image.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Source JPEG corpus. Override with the IDCT_VIZ_SRC_DIR env var or by passing
# a positional argument on the command line. The original gallery was built
# against Wikimedia's POTY-2011 finalists, but any directory of large JPEGs
# (short axis >= 1500 px) works.
SRC_DIR = Path(os.environ.get('IDCT_VIZ_SRC_DIR', ''))
OUT_DIR = REPO / 'idct-scaling-visualization'
IMG_DIR = OUT_DIR / 'images'   # per-image PNG outputs and HTML pages
THUMB_DIR = OUT_DIR / 'thumbs' # small thumbnails for the index grid

DOWNSCALE_IDCT = REPO / 'downscale-skia-idct'
DOWNSCALE_PURE = REPO / 'downscale-skia'
MIN_FACTOR = 2.5

# Configs: (label, slug, denom, full-image ratio).
# For min_factor=2.5 the IDCT denom selection windows are:
#   1/2 selected when full ratio in [5, 10);  1/4 in [10, 20); 1/8 in [20, 40).
# "barely" sits at the lower edge of each window (Lanczos ratio ~2.5);
# "heavy"  sits just below the next window (Lanczos ratio ~5).
CONFIGS = [
    ('1/2 barely', '1-2-barely', 2,  5.00),
    ('1/2 heavy',  '1-2-heavy',  2,  9.98),
    ('1/4 barely', '1-4-barely', 4, 10.00),
    ('1/4 heavy',  '1-4-heavy',  4, 19.96),
    ('1/8 barely', '1-8-barely', 8, 20.00),
    ('1/8 heavy',  '1-8-heavy',  8, 39.92),
]

# Skip configs whose target is below this on the short axis -- below this
# the comparison gets too pixelated to read.
MIN_TARGET_PX = 32

# Concurrency for the per-config render workers. Each worker forks
# downscale-skia-idct / downscale-skia / magick. Tune to roughly the
# CPU count; the IDCT and Lanczos passes are mostly CPU-bound.
WORKERS = max(2, (os.cpu_count() or 4))


def slugify(name: str) -> str:
    """Make a filesystem- and URL-friendly slug from a filename stem."""
    s = unicodedata.normalize('NFKD', name)
    s = s.encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^A-Za-z0-9._-]+', '-', s).strip('-_.')
    return s.lower() or 'image'


def jpeg_dims(path: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(
            ['identify', '-format', '%w %h', f'{path}[0]'],
            stderr=subprocess.DEVNULL,
        )
        w, h = out.decode().split()
        return int(w), int(h)
    except Exception:
        return None


def jpeg_mcu_size(path: Path) -> tuple[int, int] | None:
    """Return MCU (width, height) in pixels for a JPEG, derived from the luma
    component's sampling factor. 4:2:0 (2x2,1x1,1x1) -> 16x16, 4:4:4 (1x1,...)
    -> 8x8, 4:2:2 (2x1,...) -> 16x8.

    This mirrors mInfo.max_h_samp_factor * DCTSIZE (and v) in nsJPEGDecoder.cpp;
    Firefox's commit gates the IDCT path on image dims being a whole multiple
    of these MCU dims."""
    try:
        out = subprocess.check_output(
            ['identify', '-format', '%[jpeg:sampling-factor]', f'{path}[0]'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        # ImageMagick prints the per-component factors comma-separated; the
        # luma component (first) sets max_h/v_samp_factor.
        first = out.split(',')[0]
        sx, sy = first.split('x')
        return int(sx) * 8, int(sy) * 8
    except Exception:
        return None


def pick_images(n: int) -> list[tuple[Path, int, int]]:
    """Pick ~n JPEGs that are large enough for the full 6-config treatment.

    Strategy: filter to W>=4000 (so the 1/8 configs produce >=100px targets),
    then take a deterministic stride through the sorted list to get a
    diverse spread of sizes / file order.
    """
    jpegs: list[tuple[Path, int, int]] = []
    for ext in ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG'):
        for p in SRC_DIR.glob(ext):
            d = jpeg_dims(p)
            if d is None:
                continue
            w, h = d
            # Need short axis large enough that 1/8 heavy target >= MIN_TARGET_PX.
            short = min(w, h)
            if short < 1500:
                continue
            jpegs.append((p, w, h))

    # Sort by filename for determinism, then take a stride to cover a range.
    jpegs.sort(key=lambda t: t[0].name.lower())
    if len(jpegs) <= n:
        return jpegs
    stride = len(jpegs) / n
    picked = [jpegs[int(i * stride)] for i in range(n)]
    return picked


def compute_targets(full_w: int, full_h: int) -> list[dict]:
    """For each config, compute integer target size that yields the desired
    IDCT denom and Lanczos pressure. Returns the configs that fit."""
    out = []
    for label, slug, denom, ratio in CONFIGS:
        tw = max(1, round(full_w / ratio))
        th = max(1, round(full_h / ratio))
        if min(tw, th) < MIN_TARGET_PX:
            continue
        # Intermediate after IDCT 1/denom (libjpeg-turbo uses ceil for partial denoms).
        inter_w = (full_w + denom - 1) // denom
        inter_h = (full_h + denom - 1) // denom
        lanczos_ratio_w = inter_w / tw
        lanczos_ratio_h = inter_h / th
        out.append({
            'label': label,
            'slug': slug,
            'denom': denom,
            'tw': tw,
            'th': th,
            'inter_w': inter_w,
            'inter_h': inter_h,
            'lanczos_ratio': max(lanczos_ratio_w, lanczos_ratio_h),
        })
    return out


def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            f'cmd failed: {cmd}\nstdout: {res.stdout.decode(errors="replace")}\n'
            f'stderr: {res.stderr.decode(errors="replace")}'
        )


def render_config(src: Path, slug: str, cfg: dict) -> None:
    out_idct = IMG_DIR / f'{slug}__{cfg["slug"]}__idct.png'
    out_pure = IMG_DIR / f'{slug}__{cfg["slug"]}__pure.png'
    out_diff = IMG_DIR / f'{slug}__{cfg["slug"]}__diff10x.png'
    if out_idct.exists() and out_pure.exists() and out_diff.exists():
        return  # idempotent

    run([
        str(DOWNSCALE_IDCT), str(src), str(out_idct),
        '--width', str(cfg['tw']), '--height', str(cfg['th']),
        '--idct-min-factor', str(MIN_FACTOR),
    ])
    run([
        str(DOWNSCALE_PURE), str(src), str(out_pure),
        '--width', str(cfg['tw']), '--height', str(cfg['th']),
    ])
    # -alpha off so the difference op doesn't run on alpha (which is 255-255=0
    # everywhere and would make the whole PNG transparent / appear black).
    run([
        'magick', str(out_idct), str(out_pure),
        '-alpha', 'off',
        '-compose', 'difference', '-composite',
        '-evaluate', 'multiply', '10',
        str(out_diff),
    ])


def make_thumb(src: Path, slug: str) -> None:
    dst = THUMB_DIR / f'{slug}.jpg'
    if dst.exists():
        return
    # 320px wide thumbnail, sRGB JPEG, modest quality.
    run([
        'magick', f'{src}[0]',
        '-auto-orient', '-strip',
        '-resize', '320x>',
        '-quality', '78',
        str(dst),
    ])


def render_image(idx: int, total: int, src: Path, w: int, h: int) -> dict:
    slug = slugify(src.stem)
    targets = compute_targets(w, h)

    mcu = jpeg_mcu_size(src)
    if mcu is None:
        # Be conservative: if we can't read the sampling factor, treat as
        # not-aligned so the page warns rather than asserting Firefox would
        # take the IDCT path.
        mcu_w, mcu_h, mcu_aligned = 0, 0, False
    else:
        mcu_w, mcu_h = mcu
        mcu_aligned = (w % mcu_w == 0) and (h % mcu_h == 0)

    print(f'[{idx+1:3d}/{total}] {src.name}  {w}x{h}  '
          f'mcu={mcu_w}x{mcu_h} aligned={mcu_aligned}  configs={len(targets)}',
          flush=True)

    make_thumb(src, slug)

    # Render configs in parallel within this image.
    with ThreadPoolExecutor(max_workers=min(len(targets), 4)) as ex:
        futs = [ex.submit(render_config, src, slug, c) for c in targets]
        for f in as_completed(futs):
            f.result()

    page = IMG_DIR / f'{slug}.html'
    write_image_page(page, src.name, w, h, slug, targets,
                     mcu_w, mcu_h, mcu_aligned)
    thumb_w, thumb_h = jpeg_dims(THUMB_DIR / f'{slug}.jpg') or (320, 240)
    return {
        'slug': slug,
        'name': src.name,
        'w': w, 'h': h,
        'thumb_w': thumb_w, 'thumb_h': thumb_h,
        'configs': len(targets),
        'mcu_w': mcu_w, 'mcu_h': mcu_h, 'mcu_aligned': mcu_aligned,
    }


def write_image_page(out: Path, src_name: str, full_w: int, full_h: int,
                     slug: str, targets: list[dict],
                     mcu_w: int, mcu_h: int, mcu_aligned: bool) -> None:
    if mcu_aligned:
        banner = (
            f'<div class="ff-banner ff-banner-aligned">'
            f'Firefox <b>would</b> take the IDCT path on this image &mdash; '
            f'{full_w} &times; {full_h} is a whole number of MCU blocks '
            f'({mcu_w} &times; {mcu_h}). The renders below reflect what would ship.'
            f'</div>'
        )
    else:
        mcu_label = (f'{mcu_w} &times; {mcu_h}' if mcu_w and mcu_h
                     else 'unknown size')
        banner = (
            f'<div class="ff-banner ff-banner-skip">'
            f'Firefox would <b>skip</b> the IDCT path on this image &mdash; '
            f'{full_w} &times; {full_h} isn\'t a whole number of MCU blocks '
            f'({mcu_label}), so trailing-MCU padding could smear into visible '
            f'pixels (<a href="https://crbug.com/890745">crbug.com/890745</a>, '
            f'<a href="https://github.com/libjpeg-turbo/libjpeg-turbo/issues/297">'
            f'libjpeg-turbo#297</a>). The IDCT renders below show what '
            f'<em>would</em> happen if the gate were lifted; in stock Firefox '
            f'with this commit, output matches the pure-Lanczos column.'
            f'</div>'
        )

    sections = []
    for cfg in targets:
        # Path is relative to IMG_DIR (this page lives in IMG_DIR), so just basenames.
        idct = f'{slug}__{cfg["slug"]}__idct.png'
        pure = f'{slug}__{cfg["slug"]}__pure.png'
        diff = f'{slug}__{cfg["slug"]}__diff10x.png'
        sections.append(f'''
<section class="config" data-cfg="{cfg["slug"]}">
  <div class="config-header">
    <span class="config-title">{cfg["label"]}</span>
    <span class="config-target">target {cfg["tw"]} &times; {cfg["th"]} &middot;
      IDCT 1/{cfg["denom"]} &rarr; {cfg["inter_w"]}&times;{cfg["inter_h"]} &middot;
      Lanczos ratio &tilde; {cfg["lanczos_ratio"]:.2f}&times;</span>
  </div>
  <div class="images">
    <div class="tile pair show-idct">
      <div class="stack">
        <img class="idct" src="{idct}" alt="IDCT">
        <img class="pure" src="{pure}" alt="pure">
      </div>
      <span class="mode-badge badge-idct">IDCT &#9679; click to flick</span>
      <span class="mode-badge badge-pure">PURE &#9679; click to flick</span>
      <div class="caption">IDCT+Lanczos &#8644; pure Lanczos @ {cfg["tw"]}&times;{cfg["th"]}</div>
    </div>
    <div class="tile">
      <img src="{diff}" alt="diff">
      <span class="label diff">abs diff &times;10</span>
      <div class="caption">pixel delta, amplified 10&times;</div>
    </div>
  </div>
</section>''')

    body = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IDCT vs Lanczos3 &mdash; {src_name}</title>
<link rel="stylesheet" href="../viz.css">
</head>
<body>

<p class="breadcrumb"><a href="../index.html">&larr; back to index</a></p>
<h1>IDCT prescale + Lanczos3 &nbsp;vs&nbsp; pure Lanczos3</h1>
<div class="subtitle">
  Input: <code>{src_name}</code> ({full_w}&times;{full_h}).
  <code>--idct-min-factor {MIN_FACTOR}</code>.
  Images render at <b>intrinsic pixel size</b> &mdash; no CSS scaling.
</div>

{banner}

<div class="toolbar">
  <button id="toggle">
    Flick all &rarr; showing
    <span id="mode" class="mode-indicator mode-idct">IDCT</span>
  </button>
  <span class="hint">
    <kbd>Space</kbd> or click the button toggles every pair at once.
    Click a single image to toggle just that pair.
    Hold <kbd>Shift</kbd> while clicking the toolbar button to swap continuously.
  </span>
</div>

{''.join(sections)}

<footer>
  Generated via <code>downscale-skia-idct</code> / <code>downscale-skia</code>;
  diffs via ImageMagick (<code>-compose difference -evaluate multiply 10</code>).
</footer>

<script src="../viz.js"></script>
</body>
</html>
'''
    out.write_text(body)


def write_index(entries: list[dict]) -> None:
    """Top-level index linking to all per-image pages."""
    cards = []
    for e in entries:
        if e.get('mcu_aligned'):
            badge = '<span class="badge badge-idct">IDCT path</span>'
        else:
            badge = '<span class="badge badge-skip">Lanczos only</span>'
        cards.append(f'''
    <a class="card" href="images/{e["slug"]}.html">
      <div class="thumb-wrap">
        <img class="thumb" src="thumbs/{e["slug"]}.jpg" alt="{e["name"]}" loading="lazy">
      </div>
      <div class="card-meta">
        <div class="card-name">{e["name"]}</div>
        <div class="card-dims">{e["w"]} &times; {e["h"]} &middot; {e["configs"]} configs &middot; {badge}</div>
      </div>
    </a>''')

    aligned_count = sum(1 for e in entries if e.get('mcu_aligned'))
    skipped_count = len(entries) - aligned_count

    body = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IDCT vs Lanczos3 &mdash; gallery ({len(entries)} images)</title>
<link rel="stylesheet" href="viz.css">
<style>
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 14px;
  }}
  .card {{
    display: block;
    background: #222; border: 1px solid #333; border-radius: 4px;
    text-decoration: none; color: inherit; overflow: hidden;
    transition: border-color 0.1s;
  }}
  .card:hover {{ border-color: #ffcc66; }}
  .thumb-wrap {{
    background: #000; aspect-ratio: 4/3; display: flex;
    align-items: center; justify-content: center; overflow: hidden;
  }}
  .thumb {{ max-width: 100%; max-height: 100%; display: block; }}
  .card-meta {{ padding: 8px 10px; font-size: 12px; }}
  .card-name {{
    color: #ddd; font-family: ui-monospace, Menlo, Consolas, monospace;
    word-break: break-all; line-height: 1.3;
    overflow: hidden; text-overflow: ellipsis; display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  }}
  .card-dims {{ color: #888; margin-top: 4px; font-family: ui-monospace, monospace; }}
  .badge {{
    display: inline-block; padding: 1px 6px; background: #2d6b3a;
    color: #fff; border-radius: 2px; font-size: 10px;
  }}
  .badge-idct {{ background: #2d6b3a; }}
  .badge-skip {{ background: #a83a2f; }}
</style>
</head>
<body>

<h1>IDCT prescale + Lanczos3 &nbsp;vs&nbsp; pure Lanczos3</h1>
<div class="subtitle">
  Visual comparison of Firefox-style IDCT pre-scaling (libjpeg-turbo's 1/2, 1/4, 1/8 IDCT)
  followed by Skia Lanczos3, against pure Skia Lanczos3 from the full-resolution decode.
  Each image page shows up to six configs covering the IDCT denom-selection windows
  (<code>--idct-min-factor {MIN_FACTOR}</code>).
  Click a thumbnail to open its page.
  <br><br>
  Firefox's commit gates the IDCT path on the JPEG having a whole number of MCUs in
  both dimensions, to avoid trailing-MCU padding smearing into visible pixels
  (<a href="https://crbug.com/890745">crbug.com/890745</a>,
  <a href="https://github.com/libjpeg-turbo/libjpeg-turbo/issues/297">libjpeg-turbo#297</a>).
  <span class="badge badge-idct">IDCT path</span> marks images Firefox would actually
  pre-scale via IDCT ({aligned_count} of {len(entries)});
  <span class="badge badge-skip">Lanczos only</span> marks the {skipped_count} where
  Firefox would skip IDCT &mdash; the renders on those pages show what the IDCT path
  <em>would</em> produce if the gate were removed, kept for curiosity.
</div>

<div class="grid">
{''.join(cards)}
</div>

<footer>{len(entries)} images &middot; pipelines: <code>downscale-skia-idct</code>, <code>downscale-skia</code>.</footer>
</body>
</html>
'''
    (OUT_DIR / 'index.html').write_text(body)


def write_assets() -> None:
    """Write shared CSS + JS used by per-image pages and the index."""
    (OUT_DIR / 'viz.css').write_text(VIZ_CSS)
    (OUT_DIR / 'viz.js').write_text(VIZ_JS)


VIZ_CSS = r'''
:root { color-scheme: dark; }
html, body { margin: 0; padding: 0; background: #1a1a1a; color: #ddd; }
body { font-family: system-ui, -apple-system, sans-serif; padding: 20px 24px 48px; }
h1 { color: #ffcc66; font-size: 22px; margin: 0 0 4px; font-weight: 600; }
.subtitle { color: #888; font-size: 13px; margin-bottom: 16px; max-width: 900px; line-height: 1.5; }
code { font-family: ui-monospace, Menlo, Consolas, monospace; color: #ffcc66; }
a { color: #9ecaff; }
.breadcrumb { font-size: 12px; margin: 0 0 12px; }

.ff-banner {
  margin: 0 0 16px; padding: 10px 14px; border-radius: 3px;
  font-size: 13px; line-height: 1.5; max-width: 900px;
}
.ff-banner b { color: #fff; }
.ff-banner-aligned {
  background: #1d3a25; color: #c8e8c8; border-left: 4px solid #2d6b3a;
}
.ff-banner-skip {
  background: #3a1d1d; color: #ffbcb0; border-left: 4px solid #a83a2f;
}

.badge {
  display: inline-block; padding: 1px 6px; background: #2d6b3a;
  color: #fff; border-radius: 2px; font-size: 10px;
}
.badge-idct { background: #2d6b3a; }
.badge-skip { background: #a83a2f; }
kbd {
  background: #333; border: 1px solid #555; border-bottom-width: 2px;
  border-radius: 3px; padding: 1px 6px; font-family: ui-monospace, monospace;
  font-size: 12px; color: #ddd;
}

.toolbar {
  position: sticky; top: 0; background: #1a1a1a;
  padding: 10px 0; border-bottom: 1px solid #333; z-index: 10;
  margin-bottom: 20px; display: flex; align-items: center; gap: 14px;
  flex-wrap: wrap;
}
.toolbar button {
  background: #2a2a2a; color: #ddd; border: 1px solid #555;
  padding: 7px 14px; font-size: 13px; cursor: pointer; font-family: inherit;
  border-radius: 3px;
}
.toolbar button:hover { background: #333; border-color: #777; }
.toolbar .hint { color: #888; font-size: 12px; }
.mode-indicator {
  display: inline-block; padding: 2px 8px; border-radius: 2px;
  font-family: ui-monospace, monospace; font-weight: bold; font-size: 12px;
  margin-left: 6px;
}
.mode-idct { background: #2d6b3a; color: #fff; }
.mode-pure { background: #2f5aa8; color: #fff; }

.config {
  margin-bottom: 36px; padding-top: 20px;
  border-top: 1px solid #333;
}
.config-header {
  display: flex; align-items: baseline; gap: 14px;
  flex-wrap: wrap; margin-bottom: 10px;
}
.config-title { font-size: 17px; font-weight: 600; color: #ffcc66; }
.config-target { color: #aaa; font-family: ui-monospace, monospace; font-size: 13px; }

.stats {
  font-family: ui-monospace, monospace; font-size: 12px; color: #bbb;
  display: grid; gap: 2px 18px;
  grid-template-columns: max-content max-content max-content max-content;
  margin-bottom: 14px;
}
.stats .row-label { color: #888; }
.stats .val-good { color: #8dca8d; }

.images { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }

.tile {
  position: relative; background: #000;
  border: 1px solid #333; padding: 0;
}
.tile img { display: block; image-rendering: pixelated; image-rendering: crisp-edges; }

.pair { cursor: pointer; user-select: none; }
.pair .stack { position: relative; }
.pair .stack img.idct { position: relative; }
.pair .stack img.pure { position: absolute; top: 0; left: 0; }
.pair.show-idct img.pure { visibility: hidden; }
.pair.show-pure img.idct { visibility: hidden; }

.label {
  position: absolute; top: 4px; left: 4px;
  background: rgba(0,0,0,0.82); padding: 2px 7px;
  color: #ffcc66; font-size: 11px;
  font-family: ui-monospace, monospace;
  pointer-events: none; border-radius: 2px;
}
.label.diff { color: #6cf; }

.pair .mode-badge {
  position: absolute; bottom: 6px; left: 6px;
  padding: 2px 6px; font-family: ui-monospace, monospace;
  font-weight: normal; font-size: 10px; letter-spacing: 0.5px;
  color: #fff; pointer-events: none; border-radius: 3px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.6);
}
.pair .badge-idct, .pair .badge-pure { display: none; }
.pair.show-idct .badge-idct { display: inline-block; background: #2d6b3a; }
.pair.show-pure .badge-pure { display: inline-block; background: #2f5aa8; }

.pair.show-idct { border-color: #2d6b3a; }
.pair.show-pure { border-color: #2f5aa8; }

.caption {
  font-size: 11px; color: #888; font-family: ui-monospace, monospace;
  padding: 4px 6px; background: #111;
}

footer { color: #666; font-size: 11px; margin-top: 48px; text-align: center; }
'''

VIZ_JS = r'''
(function () {
  // Pin each <img> in a .tile so 1 PNG pixel = 1 device pixel, regardless
  // of devicePixelRatio. The browser otherwise treats naturalWidth as CSS
  // pixels, which on HiDPI screens stretches the PNG by DPR (combined with
  // image-rendering: pixelated this looks like jagged/blocky edges, but it
  // is still scaling — and we want zero scaling for an offline-scaler
  // comparison).
  function pinIntrinsic(img) {
    const dpr = window.devicePixelRatio || 1;
    if (!img.naturalWidth) return;
    img.style.width = (img.naturalWidth / dpr) + 'px';
    img.style.height = (img.naturalHeight / dpr) + 'px';
  }
  function pinAll() {
    document.querySelectorAll('.tile img').forEach(pinIntrinsic);
  }
  document.querySelectorAll('.tile img').forEach(img => {
    if (img.complete) pinIntrinsic(img);
    else img.addEventListener('load', () => pinIntrinsic(img), { once: true });
  });
  // Re-pin on zoom / window-move-to-different-DPR-display.
  window.addEventListener('resize', pinAll);

  const pairs = Array.from(document.querySelectorAll('.pair'));
  const modeEl = document.getElementById('mode');
  const toggleBtn = document.getElementById('toggle');
  if (!modeEl || !toggleBtn) return;

  let globalMode = 'idct';
  let blinkTimer = null;

  function setGlobal(mode) {
    globalMode = mode;
    pairs.forEach(p => {
      p.classList.remove('show-idct', 'show-pure');
      p.classList.add('show-' + mode);
    });
    modeEl.textContent = mode.toUpperCase();
    modeEl.className = 'mode-indicator mode-' + mode;
  }

  function toggleGlobal() {
    setGlobal(globalMode === 'idct' ? 'pure' : 'idct');
  }

  function startBlink() {
    if (blinkTimer) return;
    blinkTimer = setInterval(toggleGlobal, 700);
    toggleBtn.textContent = toggleBtn.textContent.replace('Flick all', 'Blinking');
  }
  function stopBlink() {
    if (!blinkTimer) return;
    clearInterval(blinkTimer);
    blinkTimer = null;
    const span = modeEl;
    toggleBtn.innerHTML = 'Flick all &rarr; showing ';
    toggleBtn.appendChild(span);
  }

  toggleBtn.addEventListener('click', (e) => {
    if (e.shiftKey) {
      if (blinkTimer) stopBlink(); else startBlink();
      return;
    }
    if (blinkTimer) stopBlink();
    toggleGlobal();
  });

  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.target.matches('input, textarea, button')) {
      e.preventDefault();
      if (blinkTimer) stopBlink();
      toggleGlobal();
    }
  });

  pairs.forEach(p => {
    p.addEventListener('click', (e) => {
      const isIdct = p.classList.contains('show-idct');
      p.classList.remove('show-idct', 'show-pure');
      p.classList.add(isIdct ? 'show-pure' : 'show-idct');
      e.stopPropagation();
    });
  });
})();
'''


def main():
    global SRC_DIR
    if len(sys.argv) > 1:
        SRC_DIR = Path(sys.argv[1])
    if not SRC_DIR or str(SRC_DIR) == '.':
        sys.exit(
            'No source directory. Pass it on the command line or set '
            'IDCT_VIZ_SRC_DIR. Example:\n'
            '  IDCT_VIZ_SRC_DIR=/path/to/jpegs python3 generate.py\n'
            '  python3 generate.py /path/to/jpegs'
        )
    if not SRC_DIR.is_dir():
        sys.exit(f'Source directory does not exist: {SRC_DIR}')

    if not DOWNSCALE_IDCT.exists() or not DOWNSCALE_PURE.exists():
        sys.exit(f'Build required: missing {DOWNSCALE_IDCT} or {DOWNSCALE_PURE}')

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    n = int(os.environ.get('N', '100'))
    images = pick_images(n)
    print(f'Selected {len(images)} images.', flush=True)

    write_assets()

    entries: list[dict] = []
    # Render images in parallel at the image level so multiple cores are used.
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(render_image, i, len(images), p, w, h): (i, p)
                for i, (p, w, h) in enumerate(images)}
        for f in as_completed(futs):
            try:
                entries.append(f.result())
            except Exception as e:
                i, p = futs[f]
                print(f'FAILED {p.name}: {e}', file=sys.stderr)

    entries.sort(key=lambda e: e['name'].lower())
    write_index(entries)
    (OUT_DIR / 'manifest.json').write_text(json.dumps(entries, indent=2))
    print(f'Done. index.html written with {len(entries)} cards.')


if __name__ == '__main__':
    main()

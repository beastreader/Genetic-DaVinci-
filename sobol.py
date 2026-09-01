"""
Genetic Da Vinci — sobol.py
==========================
The search engine that "rebuilds" a target image out of a small number of
geometric shapes (circles, triangles, lines).

WHAT IT DOES
------------
Given a target image and a small shape budget, it finds the set of shapes
whose *summed* drawing is as close as possible to the target. It searches
evolutionarily / by hill-climbing:

  * each iteration it proposes a big batch of candidate shapes,
  * scores each by a *loss* = how wrong its pixels are,
  * promotes the best candidate to champion,
  * nudges (mutates) the champion and keeps the nudge only if it helped.

Two tricks make it fast:
  1. SOBOl SAMPLING  - candidate shapes are scattered across the search
     space with a Sobol sequence (a quasi-random / low-discrepancy sequence
     from scipy.stats.qmc) instead of plain random, so coverage is even.
  2. NUMBA JIT      - the hot pixel math is written in `nopython` mode and
     compiled to machine code at import time; a whole batch is scored in one
     compiled call.

run_ga (the main function) does, each step:
  1. build a small "downscaled" copy of the target to search on (fast),
     optionally splitting it into overlapping crops ("crop mode"),
  2. seed a champion shape (often guided by the current residual = where the
     reconstruction is still wrong),
  3. loop max_iterations times:
       - sample a Sobol batch of candidates,
       - score them all at once (batch JIT),
       - keep the best as champion,
       - mutate the champion by an adaptive step (1/5 success rule + EMA),
       - accept the mutation only if it lowers the loss (monotonic gate),
  4. yield the full-resolution canvas so the UI shows the image assembling.

FILE MAP
--------
  Section 1  _draw_shape_mask                legacy per-shape mask (draw_letter)
  Section 2  _cached_text_size               text-size cache
           _build_sse_ii / _rect_sum         integral image (O(1) rect sum)
  Section 3  _jit_loss / _jit_integral_loss  core loss ("fitness" of a shape)
  Section 4  _jit_fill_*                     rasterizers: draw shape into canvas
           _jit_point_in_tri                 point-in-triangle test
  Section 5  _jit_eval_*_batch               score a whole population (bbox path)
  Section 6  _jit_eval_*_scanline_partial    score a whole population (the hard,
              _batch                            fast partial-loss path)
  Section 7  _get_font / glyph helpers       letter rendering (legacy path)
  Section 8  _jit_mutate_batch              adaptive mutation of one shape
  Section 9  run_ga                         the main generator (the "brain")
           draw_letter                      small helper used by the app

Every `@njit` function is compiled by Numba. The plain-Python `*_plain`
functions sitting next to the hard blocks are READABLE REFERENCES of the
same math (slower, never called in the hot loop) so you can see what the
compiled code is actually doing.
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import time
from functools import lru_cache
from scipy.stats import qmc
import numba

_USE_CY = False  # Cython path removed — sobol engine uses Numba JIT only

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOSS_SCALE = 0.25
_N_LETTERS = len(ALPHABET)
_GENE_R, _GENE_G, _GENE_B = 0, 1, 2
_GENE_ROT, _GENE_SCALE, _GENE_X, _GENE_Y, _GENE_LET = 3, 4, 5, 6, 7
_GENE_A = 8
_N_GENES = 9
_ACCEPT_EPS = 1e-8
_BATCH = 128  # batch size for init / resample / hill
_SMALL_W, _SMALL_H = 64, 64
_SMALL_SCALE = _SMALL_W / 400.0  # assume 400 base, will recompute in run_ga

SHAPE_TYPES = ["circle", "triangle", "line"]
_N_SHAPES = len(SHAPE_TYPES)


def _draw_shape_mask(mask, shape_type, x, y, scale, thick, rot, x1, y1):
    # legacy wrapper for scale/rot mode (kept for compat)
    h, w = mask.shape
    cx, cy = x - x1, y - y1
    if shape_type == "circle":
        radius = max(1, int(scale))
        t = -1 if thick == 1 else thick
        cv2.circle(mask, (cx, cy), radius, 255, t, lineType=cv2.LINE_AA)
    elif shape_type == "triangle":
        r = max(1, int(scale))
        ang = np.deg2rad(rot)
        pts = []
        for k in range(3):
            a = ang + k * 2 * np.pi / 3
            px = int(cx + r * np.cos(a))
            py = int(cy + r * np.sin(a))
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        if thick == 1:
            cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(mask, [pts], True, 255, thick, lineType=cv2.LINE_AA)
    elif shape_type == "line":
        ang = np.deg2rad(rot)
        x2 = int(cx + scale * np.cos(ang))
        y2 = int(cy + scale * np.sin(ang))
        cv2.line(mask, (cx, cy), (x2, y2), 255, thick, lineType=cv2.LINE_AA)
    else:
        cv2.circle(mask, (cx, cy), max(1, int(scale)), 255, -1, lineType=cv2.LINE_AA)


def _draw_shape_mask_explicit(mask, shape_type, x1, y1, x2, y2, x3, y3, R, thick, patch_x1, patch_y1):
    # explicit coords mode: mask patch, shape_type, global coords, patch top-left
    # circle: (x1,y1) center, R radius
    # line: (x1,y1)-(x2,y2)
    # triangle: (x1,y1),(x2,y2),(x3,y3)
    if shape_type == "circle":
        cx, cy = x1 - patch_x1, y1 - patch_y1
        radius = max(1, int(R))
        t = -1 if thick == 1 else thick
        cv2.circle(mask, (cx, cy), radius, 255, t, lineType=cv2.LINE_AA)
    elif shape_type == "triangle":
        pts = np.array([ [x1 - patch_x1, y1 - patch_y1], [x2 - patch_x1, y2 - patch_y1], [x3 - patch_x1, y3 - patch_y1] ], dtype=np.int32)
        if thick == 1:
            cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(mask, [pts], True, 255, thick, lineType=cv2.LINE_AA)
    elif shape_type == "line":
        cv2.line(mask, (x1 - patch_x1, y1 - patch_y1), (x2 - patch_x1, y2 - patch_y1), 255, thick, lineType=cv2.LINE_AA)

@lru_cache(maxsize=4096)
def _cached_text_size(char: str, scale: float, thickness: int):
    # cache cv2.getTextSize (pure, ~30% of per-cand time)
    return cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)


def _build_sse_ii(bg, target):
    # bg/target float32 BGR 0-255, sse = sum_c (bg-target)^2
    diff = bg.astype(np.float64) - target.astype(np.float64)
    sse = np.sum(diff * diff, axis=2)
    ii = np.zeros((bg.shape[0] + 1, bg.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(sse, axis=0), axis=1)
    return ii


def _rect_sum(ii, y0, x0, y1, x1):
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]





# =====================================================================
# Section 3 — Core loss  ("fitness" = how wrong one shape is)
# ---------------------------------------------------------------------
# `_jit_loss` is the heart of the search. Given a shape's binary mask, the
# current canvas (`old`) and the target, it alpha-blends the shape's colour
# (b,g,r) over the canvas and sums the L1 (absolute) colour error vs the
# target, averaged over all H*W*3 channel values. The GA's only job is to
# make this number as small as possible.
#   * `_jit_loss`          - direct L1 error over the shape's mask
#   * `_jit_patch`         - the same blend, used to actually write a shape
#                            onto the canvas (no error computed)
#   * `_jit_integral_loss` - faster variant that reuses a precomputed SSE
#                            (sum-of-squared-error) integral image
# =====================================================================
@numba.njit(cache=True)
def _jit_loss(mask, old, target, b, g, r, alpha):
    h, w = mask.shape
    total = 0.0
    inv255 = 1.0 / 255.0
    for y in range(h):
        for w_ in range(w):
            a = mask[y, w_] * alpha * inv255  # mask 0-255 uint8, alpha=alpha_fixed/255
            blended = old[y, w_, 0] * (1.0 - a) + b * a
            total += abs(blended - target[y, w_, 0])
            blended = old[y, w_, 1] * (1.0 - a) + g * a
            total += abs(blended - target[y, w_, 1])
            blended = old[y, w_, 2] * (1.0 - a) + r * a
            total += abs(blended - target[y, w_, 2])
    return total / (h * w * 3)


@numba.njit(cache=True)
def _jit_patch(mask, old, b, g, r, alpha, out):
    h, w = mask.shape
    inv255 = 1.0 / 255.0
    for y in range(h):
        for w_ in range(w):
            a = mask[y, w_] * alpha * inv255
            out[y, w_, 0] = old[y, w_, 0] * (1.0 - a) + b * a
            out[y, w_, 1] = old[y, w_, 1] * (1.0 - a) + g * a
            out[y, w_, 2] = old[y, w_, 2] * (1.0 - a) + r * a


@numba.njit(cache=True)
def _jit_integral_loss(mask, old, target, b, g, r, alpha, bg_sse):
    h, w = mask.shape
    sum1 = 0.0
    sum2 = 0.0
    inv255 = 1.0 / 255.0
    for y in range(h):
        for x in range(w):
            k = mask[y, x] * alpha * inv255
            if k == 0.0:
                continue
            dcb = b - old[y, x, 0]
            dcg = g - old[y, x, 1]
            dcr = r - old[y, x, 2]
            db0 = old[y, x, 0] - target[y, x, 0]
            db1 = old[y, x, 1] - target[y, x, 1]
            db2 = old[y, x, 2] - target[y, x, 2]
            sum1 += k * (dcb * db0 + dcg * db1 + dcr * db2)
            sum2 += k * k * (dcb * dcb + dcg * dcg + dcr * dcr)
    total = bg_sse + 2.0 * sum1 + sum2
    return total / (h * w * 3)


def _loss_plain(mask, old, target, b, g, r, alpha):
    """NORMAL-PYTHON equivalent of `_jit_loss` (readable reference, slower).

    `mask` is 0..255; the per-pixel effective alpha is mask/255 * alpha.
    For each colour channel we alpha-blend the shape colour over the existing
    canvas and accumulate the absolute difference vs the target. The result is
    the mean L1 error over all H*W*3 channel values. (Never called in the hot
    loop - it exists only so the compiled version is easy to read.)
    """
    a = (mask / 255.0) * alpha                       # per-pixel alpha in [0, 1]
    total = 0.0
    for ch, col in ((0, b), (1, g), (2, r)):         # B, G, R channels
        blended = old[:, :, ch] * (1.0 - a) + col * a
        total += np.abs(blended - target[:, :, ch]).sum()
    h, w = mask.shape
    return total / (h * w * 3)


@numba.njit(cache=True)
def _jit_clip_color(v, lo, hi):
    if v < lo:
        return lo
    elif v > hi:
        return hi
    else:
        return int(v)


@numba.njit(cache=True)
def _jit_clip_pos(v, lo, hi):
    if v < lo:
        return lo
    elif v > hi:
        return hi
    else:
        return v


# =====================================================================
# Section 4 — Shape rasterizers (draw a shape into a uint8 mask)
# ---------------------------------------------------------------------
# These fill a binary mask (0/255) with a single shape using pure Numba loops
# instead of OpenCV. `_jit_point_in_tri` is the geometric core for triangles;
# `_jit_fill_*` / `_jit_stroke_*` rasterize fills and outlines; `_jit_draw_shape`
# dispatches to the right one. A pixel == 255 means "inside the shape".
# =====================================================================

@numba.njit(cache=True)
def _jit_fill_circle(mask, cx, cy, r):
    """Filled circle on uint8 mask at local coords (cx,cy)."""
    h, w = mask.shape
    for py in range(cy - r, cy + r + 1):
        if py < 0 or py >= h:
            continue
        for px in range(cx - r, cx + r + 1):
            if px < 0 or px >= w:
                continue
            if (px - cx) * (px - cx) + (py - cy) * (py - cy) <= r * r:
                mask[py, px] = 255


@numba.njit(cache=True)
def _jit_stroke_circle(mask, cx, cy, r, t):
    """Stroked circle (outline) on uint8 mask, thickness t."""
    h, w = mask.shape
    outer_r = r + t // 2
    inner_r = r - t // 2
    if inner_r < 0:
        inner_r = 0
    for py in range(cy - outer_r, cy + outer_r + 1):
        if py < 0 or py >= h:
            continue
        for px in range(cx - outer_r, cx + outer_r + 1):
            if px < 0 or px >= w:
                continue
            d2 = (px - cx) * (px - cx) + (py - cy) * (py - cy)
            if d2 <= outer_r * outer_r and d2 >= inner_r * inner_r:
                mask[py, px] = 255


@numba.njit(cache=True)
def _jit_draw_line_mask(mask, x0, y0, x1, y1, thick):
    """Draw thick line on uint8 mask using Bresenham + per-point fill."""
    h, w = mask.shape
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    ht = max(1, thick // 2)
    while True:
        # fill square around current point
        for py in range(y0 - ht, y0 + ht + 1):
            if py < 0 or py >= h:
                continue
            for px in range(x0 - ht, x0 + ht + 1):
                if px < 0 or px >= w:
                    continue
                mask[py, px] = 255
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


@numba.njit(cache=True)
def _jit_point_in_tri(px, py, x0, y0, x1, y1, x2, y2):
    """Check if point (px,py) is inside triangle using cross products."""
    d1 = (px - x1) * (y0 - y1) - (x0 - x1) * (py - y1)
    d2 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
    d3 = (px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _point_in_tri_plain(px, py, x0, y0, x1, y1, x2, y2):
    """NORMAL-PYTHON equivalent of `_jit_point_in_tri` (readable reference).

    A point is inside the triangle iff it is on the same side of all three
    edges. The "side" of an edge is the sign of the 2D cross product of the
    edge vector with the point-to-vertex vector. If all three cross products
    have the same sign (or are zero) the point is inside.
    """
    d1 = (px - x1) * (y0 - y1) - (x0 - x1) * (py - y1)
    d2 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
    d3 = (px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


@numba.njit(cache=True)
def _jit_fill_triangle(mask, x0, y0, x1, y1, x2, y2):
    """Filled triangle on uint8 mask."""
    h, w = mask.shape
    min_x = max(0, min(x0, min(x1, x2)))
    max_x = min(w - 1, max(x0, max(x1, x2)))
    min_y = max(0, min(y0, min(y1, y2)))
    max_y = min(h - 1, max(y0, max(y1, y2)))
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            if _jit_point_in_tri(px, py, x0, y0, x1, y1, x2, y2):
                mask[py, px] = 255


@numba.njit(cache=True)
def _jit_stroke_triangle(mask, x0, y0, x1, y1, x2, y2, thick):
    """Stroked triangle (3 lines) on uint8 mask."""
    _jit_draw_line_mask(mask, x0, y0, x1, y1, thick)
    _jit_draw_line_mask(mask, x1, y1, x2, y2, thick)
    _jit_draw_line_mask(mask, x2, y2, x0, y0, thick)


@numba.njit(cache=True)
def _jit_draw_shape(mask, shape_type, cx, cy, x1, y1, x2, y2, x3, y3, R, thick):
    """Unified JIT shape draw. circle: cx,cy,R; line: x1,y1-x2,y2; tri: x1,y1-x2,y2-x3,y3."""
    if shape_type == 0:  # circle
        if thick <= 1:
            _jit_fill_circle(mask, cx, cy, R)
        else:
            _jit_stroke_circle(mask, cx, cy, R, thick)
    elif shape_type == 1:  # triangle
        if thick <= 1:
            _jit_fill_triangle(mask, x1, y1, x2, y2, x3, y3)
        else:
            _jit_stroke_triangle(mask, x1, y1, x2, y2, x3, y3, thick)
    else:  # line
        _jit_draw_line_mask(mask, x1, y1, x2, y2, max(1, thick))


_SHAPE_INT = {"circle": 0, "triangle": 1, "line": 2}


# ── Batch JIT evaluators — evaluate _BATCH candidates in ONE call ───
# Eliminates per-candidate Python function overhead + np.zeros + np.clip

# =====================================================================
# Section 5 — Batch evaluators (score a whole population at once)
# ---------------------------------------------------------------------
# The "explore" step. Each takes arrays of N candidate shapes and returns the
# index of the best one (lowest loss). Two flavours, per shape type:
#   * `_jit_eval_*_batch`            - compute the full loss inside the
#                                      shape's bounding box (simple baseline)
#   * `_jit_eval_*_scanline_partial` - compute only the CHANGE in the running
#                                      total error over the pixels the shape
#                                      touches (the fast path, Section 6)
# The matching `_jit_draw_*_patch` writes a chosen shape onto the canvas.
# =====================================================================
@numba.njit(cache=True)
def _jit_eval_circle_batch(canvas_float, target_float,
                           cx_arr, cy_arr, R_arr, thick_arr,
                           b_arr, g_arr, r_arr, alpha_val, img_w, img_h):
    """Evaluate all circle candidates, return (best_idx, best_loss).
    Loss matches _jit_loss exactly: all bbox pixels included, non-shape pixels have a=0."""
    n = len(R_arr)
    best_loss = 1e30
    best_idx = -1
    for i in range(n):
        cx = cx_arr[i]; cy = cy_arr[i]; R = R_arr[i]; thick = thick_arr[i]
        b = b_arr[i]; g = g_arr[i]; r = r_arr[i]
        bx1 = cx - R
        if bx1 < 0: bx1 = 0
        by1 = cy - R
        if by1 < 0: by1 = 0
        bx2 = cx + R + 1
        if bx2 > img_w: bx2 = img_w
        by2 = cy + R + 1
        if by2 > img_h: by2 = img_h
        loss_val = 1e30
        if bx2 > bx1 and by2 > by1:
            ph = by2 - by1; pw = bx2 - bx1
            local_cx = cx - bx1; local_cy = cy - by1
            inner = R - thick
            if inner < 0: inner = 0
            total = 0.0
            for py in range(by1, by2):
                lpy = py - by1
                for px in range(bx1, bx2):
                    lpx = px - bx1
                    d2 = (lpx - local_cx) * (lpx - local_cx) + (lpy - local_cy) * (lpy - local_cy)
                    inside = False
                    if d2 <= R * R:
                        inside = True
                    elif thick > 1 and d2 <= (R + thick) * (R + thick):
                        if d2 >= inner * inner:
                            inside = True
                    if inside:
                        a = alpha_val
                    else:
                        a = 0.0
                    blended0 = canvas_float[py, px, 0] * (1.0 - a) + b * a
                    total += abs(blended0 - target_float[py, px, 0])
                    blended1 = canvas_float[py, px, 1] * (1.0 - a) + g * a
                    total += abs(blended1 - target_float[py, px, 1])
                    blended2 = canvas_float[py, px, 2] * (1.0 - a) + r * a
                    total += abs(blended2 - target_float[py, px, 2])
            loss_val = total / (ph * pw * 3)
        if loss_val < best_loss:
            best_loss = loss_val
            best_idx = i
    return best_idx, best_loss


@numba.njit(cache=True)
def _jit_draw_circle_patch(canvas_float, cx, cy, R, thick, b, g, r, alpha_val,
                           out, bx1, by1, bx2, by2):
    """Draw circle result into out patch (for winner only)."""
    local_cx = cx - bx1; local_cy = cy - by1
    inv255 = 1.0 / 255.0
    for py in range(by1, by2):
        lpy = py - by1
        for px in range(bx1, bx2):
            lpx = px - bx1
            d2 = (lpx - local_cx) * (lpx - local_cx) + (lpy - local_cy) * (lpy - local_cy)
            fill = False
            if d2 <= R * R:
                fill = True
            elif thick > 1 and d2 <= (R + thick) * (R + thick):
                inner = R - thick
                if inner < 0: inner = 0
                if d2 >= inner * inner:
                    fill = True
            if fill:
                a = alpha_val
                out[lpy, lpx, 0] = canvas_float[py, px, 0] * (1.0 - a) + b * a
                out[lpy, lpx, 1] = canvas_float[py, px, 1] * (1.0 - a) + g * a
                out[lpy, lpx, 2] = canvas_float[py, px, 2] * (1.0 - a) + r * a
            else:
                out[lpy, lpx, 0] = canvas_float[py, px, 0]
                out[lpy, lpx, 1] = canvas_float[py, px, 1]
                out[lpy, lpx, 2] = canvas_float[py, px, 2]


@numba.njit(cache=True)
def _jit_eval_line_batch(canvas_float, target_float,
                         x1_arr, y1_arr, x2_arr, y2_arr, thick_arr,
                         b_arr, g_arr, r_arr, alpha_val, img_w, img_h):
    """Evaluate all line candidates, return (best_idx, best_loss).
    Loss matches _jit_loss exactly: all bbox pixels included, non-shape pixels have a=0."""
    n = len(thick_arr)
    best_loss = 1e30
    best_idx = -1
    for i in range(n):
        lx1 = x1_arr[i]; ly1 = y1_arr[i]; lx2 = x2_arr[i]; ly2 = y2_arr[i]
        thick = thick_arr[i]
        b = b_arr[i]; g = g_arr[i]; r = r_arr[i]
        mn_x = lx1 if lx1 < lx2 else lx2
        mx_x = lx1 if lx1 > lx2 else lx2
        mn_y = ly1 if ly1 < ly2 else ly2
        mx_y = ly1 if ly1 > ly2 else ly2
        ht = thick // 2
        bx1 = mn_x - ht
        if bx1 < 0: bx1 = 0
        by1 = mn_y - ht
        if by1 < 0: by1 = 0
        bx2 = mx_x + ht + 1
        if bx2 > img_w: bx2 = img_w
        by2 = mx_y + ht + 1
        if by2 > img_h: by2 = img_h
        loss_val = 1e30
        if bx2 > bx1 and by2 > by1:
            ph = by2 - by1; pw = bx2 - bx1
            total = 0.0
            for py in range(by1, by2):
                for px in range(bx1, bx2):
                    dx = float(lx2 - lx1); dy = float(ly2 - ly1)
                    len2 = dx * dx + dy * dy
                    if len2 < 1.0:
                        d2 = float((px - lx1) * (px - lx1) + (py - ly1) * (py - ly1))
                    else:
                        t = ((px - lx1) * dx + (py - ly1) * dy) / len2
                        if t < 0.0: t = 0.0
                        elif t > 1.0: t = 1.0
                        proj_x = lx1 + t * dx
                        proj_y = ly1 + t * dy
                        d2 = (px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y)
                    if d2 <= ht * ht:
                        a = alpha_val
                    else:
                        a = 0.0
                    blended0 = canvas_float[py, px, 0] * (1.0 - a) + b * a
                    total += abs(blended0 - target_float[py, px, 0])
                    blended1 = canvas_float[py, px, 1] * (1.0 - a) + g * a
                    total += abs(blended1 - target_float[py, px, 1])
                    blended2 = canvas_float[py, px, 2] * (1.0 - a) + r * a
                    total += abs(blended2 - target_float[py, px, 2])
            loss_val = total / (ph * pw * 3)
        if loss_val < best_loss:
            best_loss = loss_val
            best_idx = i
    return best_idx, best_loss


@numba.njit(cache=True)
def _jit_draw_line_patch(canvas_float, lx1, ly1, lx2, ly2, thick, b, g, r, alpha_val,
                         out, bx1, by1, bx2, by2):
    """Draw line result into out patch (for winner only)."""
    ht = thick // 2
    for py in range(by1, by2):
        lpy = py - by1
        for px in range(bx1, bx2):
            lpx = px - bx1
            dx = float(lx2 - lx1); dy = float(ly2 - ly1)
            len2 = dx * dx + dy * dy
            if len2 < 1.0:
                d2 = float((px - lx1) * (px - lx1) + (py - ly1) * (py - ly1))
            else:
                t = ((px - lx1) * dx + (py - ly1) * dy) / len2
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                proj_x = lx1 + t * dx
                proj_y = ly1 + t * dy
                d2 = (px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y)
            if d2 <= ht * ht:
                a = alpha_val
                out[lpy, lpx, 0] = canvas_float[py, px, 0] * (1.0 - a) + b * a
                out[lpy, lpx, 1] = canvas_float[py, px, 1] * (1.0 - a) + g * a
                out[lpy, lpx, 2] = canvas_float[py, px, 2] * (1.0 - a) + r * a
            else:
                out[lpy, lpx, 0] = canvas_float[py, px, 0]
                out[lpy, lpx, 1] = canvas_float[py, px, 1]
                out[lpy, lpx, 2] = canvas_float[py, px, 2]


@numba.njit(cache=True)
def _jit_eval_triangle_batch(canvas_float, target_float,
                             x1_arr, y1_arr, x2_arr, y2_arr, x3_arr, y3_arr,
                             thick_arr, b_arr, g_arr, r_arr, alpha_val, img_w, img_h):
    """Evaluate all triangle candidates, return (best_idx, best_loss).
    Loss matches _jit_loss exactly: all bbox pixels included, non-shape pixels have a=0."""
    n = len(thick_arr)
    best_loss = 1e30
    best_idx = -1
    for i in range(n):
        tx1 = x1_arr[i]; ty1 = y1_arr[i]
        tx2 = x2_arr[i]; ty2 = y2_arr[i]
        tx3 = x3_arr[i]; ty3 = y3_arr[i]
        thick = thick_arr[i]
        b = b_arr[i]; g = g_arr[i]; r = r_arr[i]
        mn_x = tx1
        if tx2 < mn_x: mn_x = tx2
        if tx3 < mn_x: mn_x = tx3
        mx_x = tx1
        if tx2 > mx_x: mx_x = tx2
        if tx3 > mx_x: mx_x = tx3
        mn_y = ty1
        if ty2 < mn_y: mn_y = ty2
        if ty3 < mn_y: mn_y = ty3
        mx_y = ty1
        if ty2 > mx_y: mx_y = ty2
        if ty3 > mx_y: mx_y = ty3
        bx1 = mn_x - thick
        if bx1 < 0: bx1 = 0
        by1 = mn_y - thick
        if by1 < 0: by1 = 0
        bx2 = mx_x + thick + 1
        if bx2 > img_w: bx2 = img_w
        by2 = mx_y + thick + 1
        if by2 > img_h: by2 = img_h
        loss_val = 1e30
        if bx2 > bx1 and by2 > by1:
            ph = by2 - by1; pw = bx2 - bx1
            total = 0.0
            for py in range(by1, by2):
                for px in range(bx1, bx2):
                    d1 = (px - tx1) * (ty2 - ty1) - (tx2 - tx1) * (py - ty1)
                    d2 = (px - tx2) * (ty3 - ty2) - (tx3 - tx2) * (py - ty2)
                    d3 = (px - tx3) * (ty1 - ty3) - (tx1 - tx3) * (py - ty3)
                    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
                    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
                    inside = not (has_neg and has_pos)
                    if inside:
                        a = alpha_val
                    else:
                        a = 0.0
                    blended0 = canvas_float[py, px, 0] * (1.0 - a) + b * a
                    total += abs(blended0 - target_float[py, px, 0])
                    blended1 = canvas_float[py, px, 1] * (1.0 - a) + g * a
                    total += abs(blended1 - target_float[py, px, 1])
                    blended2 = canvas_float[py, px, 2] * (1.0 - a) + r * a
                    total += abs(blended2 - target_float[py, px, 2])
            loss_val = total / (ph * pw * 3)
        if loss_val < best_loss:
            best_loss = loss_val
            best_idx = i
    return best_idx, best_loss


@numba.njit(cache=True)
def _jit_draw_tri_patch(canvas_float, tx1, ty1, tx2, ty2, tx3, ty3, thick, b, g, r, alpha_val,
                        out, bx1, by1, bx2, by2):
    """Draw triangle result into out patch (for winner only)."""
    for py in range(by1, by2):
        lpy = py - by1
        for px in range(bx1, bx2):
            lpx = px - bx1
            d1 = (px - tx1) * (ty2 - ty1) - (tx2 - tx1) * (py - ty1)
            d2 = (px - tx2) * (ty3 - ty2) - (tx3 - tx2) * (py - ty2)
            d3 = (px - tx3) * (ty1 - ty3) - (tx1 - tx3) * (py - ty3)
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            inside = not (has_neg and has_pos)
            if inside:
                a = alpha_val
                out[lpy, lpx, 0] = canvas_float[py, px, 0] * (1.0 - a) + b * a
                out[lpy, lpx, 1] = canvas_float[py, px, 1] * (1.0 - a) + g * a
                out[lpy, lpx, 2] = canvas_float[py, px, 2] * (1.0 - a) + r * a
            else:
                out[lpy, lpx, 0] = canvas_float[py, px, 0]
                out[lpy, lpx, 1] = canvas_float[py, px, 1]
                out[lpy, lpx, 2] = canvas_float[py, px, 2]


# =====================================================================
# Section 6 — Scanline partial evaluators  (the fastest, hardest path)
# ---------------------------------------------------------------------
# Same goal as Section 5 but far cheaper. Instead of recomputing a shape's
# loss from scratch, each here computes only the DELTA in the already-known
# total error (`current_total`) over the pixels the new shape paints. The
# trick is "scanline" rasterisation: walk the shape row by row and, per row,
# only the left/right extents it actually covers (for a circle, dx = sqrt(R^2 -
# dy^2)). `delta` is then added to the running total so we never touch pixels
# the shape does not change. This is why the GA can score thousands of
# candidates per second.
# =====================================================================
@numba.njit(cache=True)
def _jit_eval_circle_scanline_partial_batch(canvas_float, target_float,
                            cx_arr, cy_arr, R_arr, thick_arr,
                            b_arr, g_arr, r_arr, alpha_val, img_w, img_h, current_total):
    n = len(R_arr)
    best_loss = 1e30
    best_idx = -1
    total_pixels = float(img_w * img_h * 3)
    for i in range(n):
        cx = int(cx_arr[i]); cy = int(cy_arr[i]); R = int(R_arr[i]); thick = int(thick_arr[i])
        b = b_arr[i]; g = g_arr[i]; rr = r_arr[i]
        if R < 1:
            R = 1
        delta = 0.0
        if thick <= 1:
            y0 = cy - R
            if y0 < 0: y0 = 0
            y1 = cy + R
            if y1 >= img_h: y1 = img_h - 1
            for y in range(y0, y1+1):
                dy = y - cy
                tmp = R*R - dy*dy
                if tmp < 0:
                    continue
                dx = int(np.sqrt(float(tmp)))
                x0 = cx - dx
                if x0 < 0: x0 = 0
                x1_ = cx + dx
                if x1_ >= img_w: x1_ = img_w - 1
                for x in range(x0, x1_+1):
                    c0 = canvas_float[y, x, 0]; c1 = canvas_float[y, x, 1]; c2 = canvas_float[y, x, 2]
                    t0 = target_float[y, x, 0]; t1 = target_float[y, x, 1]; t2 = target_float[y, x, 2]
                    old0 = c0 - t0
                    if old0 < 0: old0 = -old0
                    old1 = c1 - t1
                    if old1 < 0: old1 = -old1
                    old2 = c2 - t2
                    if old2 < 0: old2 = -old2
                    nb0 = c0 * (1.0 - alpha_val) + b * alpha_val
                    nb1 = c1 * (1.0 - alpha_val) + g * alpha_val
                    nb2 = c2 * (1.0 - alpha_val) + rr * alpha_val
                    new0 = nb0 - t0
                    if new0 < 0: new0 = -new0
                    new1 = nb1 - t1
                    if new1 < 0: new1 = -new1
                    new2 = nb2 - t2
                    if new2 < 0: new2 = -new2
                    delta += (new0 - old0) + (new1 - old1) + (new2 - old2)
        else:
            rad = R + thick
            if rad < 1: rad = 1
            inner = R - thick
            if inner < 0: inner = 0
            inner2 = inner*inner
            bx0 = cx - rad
            if bx0 < 0: bx0 = 0
            by0 = cy - rad
            if by0 < 0: by0 = 0
            bx1 = cx + rad + 1
            if bx1 > img_w: bx1 = img_w
            by1 = cy + rad + 1
            if by1 > img_h: by1 = img_h
            R2 = R*R
            rad2 = rad*rad
            for y in range(by0, by1):
                for x in range(bx0, bx1):
                    d2 = (x - cx)*(x - cx) + (y - cy)*(y - cy)
                    inside = False
                    if d2 <= R2:
                        inside = True
                    elif d2 <= rad2 and d2 >= inner2:
                        inside = True
                    if not inside:
                        continue
                    c0 = canvas_float[y, x, 0]; c1 = canvas_float[y, x, 1]; c2 = canvas_float[y, x, 2]
                    t0 = target_float[y, x, 0]; t1 = target_float[y, x, 1]; t2 = target_float[y, x, 2]
                    old0 = c0 - t0
                    if old0 < 0: old0 = -old0
                    old1 = c1 - t1
                    if old1 < 0: old1 = -old1
                    old2 = c2 - t2
                    if old2 < 0: old2 = -old2
                    nb0 = c0 * (1.0 - alpha_val) + b * alpha_val
                    nb1 = c1 * (1.0 - alpha_val) + g * alpha_val
                    nb2 = c2 * (1.0 - alpha_val) + rr * alpha_val
                    new0 = nb0 - t0
                    if new0 < 0: new0 = -new0
                    new1 = nb1 - t1
                    if new1 < 0: new1 = -new1
                    new2 = nb2 - t2
                    if new2 < 0: new2 = -new2
                    delta += (new0 - old0) + (new1 - old1) + (new2 - old2)
        cand_loss = (current_total + delta) / total_pixels
        if cand_loss < best_loss:
            best_loss = cand_loss
            best_idx = i
    return best_idx, best_loss


def _circle_scanline_delta_plain(canvas, target, cx, cy, R, thick, b, g, r, alpha):
    """NORMAL-PYTHON equivalent of `_jit_eval_circle_scanline_partial_batch`
    (one circle - readable reference for the compiled batch version).

    Returns `delta`: how the whole-image L1 error changes if we alpha-paint
    this circle (colour b,g,r) onto `canvas`. For each pixel inside the shape
    we add (error_after - error_before); pixels the shape doesn't touch
    contribute nothing, so this stays cheap. The GA adds this delta to its
    running total instead of re-scoring the whole image.
    """
    delta = 0.0
    H, W = canvas.shape[:2]
    rad = R + thick                      # outer radius of the ring
    inner = max(0, R - thick)            # inner radius of the ring
    for y in range(max(0, cy - rad), min(H, cy + rad + 1)):
        for x in range(max(0, cx - rad), min(W, cx + rad + 1)):
            d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            inside = (d2 <= R * R) or (inner * inner <= d2 <= rad * rad)
            if not inside:
                continue
            old = np.abs(canvas[y, x] - target[y, x]).sum()
            new_col = canvas[y, x] * (1.0 - alpha) + np.array([b, g, r], dtype=np.float32) * alpha
            delta += np.abs(new_col - target[y, x]).sum() - old
    return delta


@numba.njit(cache=True)
def _jit_eval_triangle_scanline_partial_batch(canvas_float, target_float,
                              x1_arr, y1_arr, x2_arr, y2_arr, x3_arr, y3_arr,
                              thick_arr, b_arr, g_arr, r_arr, alpha_val, img_w, img_h, current_total):
    n = len(thick_arr)
    best_loss = 1e30
    best_idx = -1
    total_pixels = float(img_w * img_h * 3)
    for i in range(n):
        tx1 = int(x1_arr[i]); ty1 = int(y1_arr[i])
        tx2 = int(x2_arr[i]); ty2 = int(y2_arr[i])
        tx3 = int(x3_arr[i]); ty3 = int(y3_arr[i])
        thick = int(thick_arr[i])
        b = b_arr[i]; g = g_arr[i]; rr = r_arr[i]
        delta = 0.0
        if thick <= 1:
            y_min = ty1
            if ty2 < y_min: y_min = ty2
            if ty3 < y_min: y_min = ty3
            y_max = ty1
            if ty2 > y_max: y_max = ty2
            if ty3 > y_max: y_max = ty3
            if y_min < 0: y_min = 0
            if y_max >= img_h: y_max = img_h - 1
            inv_dy12 = 0.0
            dx12 = 0.0
            if ty1 != ty2:
                inv_dy12 = 1.0 / float(ty2 - ty1)
                dx12 = float(tx2 - tx1)
            inv_dy23 = 0.0
            dx23 = 0.0
            if ty2 != ty3:
                inv_dy23 = 1.0 / float(ty3 - ty2)
                dx23 = float(tx3 - tx2)
            inv_dy31 = 0.0
            dx31 = 0.0
            if ty3 != ty1:
                inv_dy31 = 1.0 / float(ty1 - ty3)
                dx31 = float(tx1 - tx3)
            for y in range(y_min, y_max+1):
                cnt = 0
                x_a = 0.0; x_b = 0.0; x_c = 0.0
                has_a = False; has_b = False; has_c = False
                if ty1 != ty2 and y >= (ty1 if ty1 < ty2 else ty2) and y <= (ty1 if ty1 > ty2 else ty2):
                    x_int = float(tx1) + dx12 * float(y - ty1) * inv_dy12
                    x_a = x_int; has_a = True; cnt += 1
                if ty2 != ty3 and y >= (ty2 if ty2 < ty3 else ty3) and y <= (ty2 if ty2 > ty3 else ty3):
                    x_int = float(tx2) + dx23 * float(y - ty2) * inv_dy23
                    if not has_a:
                        x_a = x_int; has_a = True
                    elif not has_b:
                        x_b = x_int; has_b = True
                    else:
                        x_c = x_int; has_c = True
                    cnt += 1
                if ty3 != ty1 and y >= (ty3 if ty3 < ty1 else ty1) and y <= (ty3 if ty3 > ty1 else ty1):
                    x_int = float(tx3) + dx31 * float(y - ty3) * inv_dy31
                    if not has_a:
                        x_a = x_int; has_a = True
                    elif not has_b:
                        x_b = x_int; has_b = True
                    else:
                        x_c = x_int; has_c = True
                    cnt += 1
                if cnt < 2:
                    continue
                x_left = x_a; x_right = x_a
                if has_b:
                    if x_b < x_left: x_left = x_b
                    if x_b > x_right: x_right = x_b
                if has_c:
                    if x_c < x_left: x_left = x_c
                    if x_c > x_right: x_right = x_c
                xl = int(x_left)
                if float(xl) < x_left:
                    xl += 1
                xr = int(x_right)
                if xl < 0: xl = 0
                if xr >= img_w: xr = img_w - 1
                if xl > xr:
                    continue
                for x in range(xl, xr+1):
                    c0 = canvas_float[y, x, 0]; c1 = canvas_float[y, x, 1]; c2 = canvas_float[y, x, 2]
                    t0 = target_float[y, x, 0]; t1 = target_float[y, x, 1]; t2 = target_float[y, x, 2]
                    old0 = c0 - t0
                    if old0 < 0: old0 = -old0
                    old1 = c1 - t1
                    if old1 < 0: old1 = -old1
                    old2 = c2 - t2
                    if old2 < 0: old2 = -old2
                    nb0 = c0 * (1.0 - alpha_val) + b * alpha_val
                    nb1 = c1 * (1.0 - alpha_val) + g * alpha_val
                    nb2 = c2 * (1.0 - alpha_val) + rr * alpha_val
                    new0 = nb0 - t0
                    if new0 < 0: new0 = -new0
                    new1 = nb1 - t1
                    if new1 < 0: new1 = -new1
                    new2 = nb2 - t2
                    if new2 < 0: new2 = -new2
                    delta += (new0 - old0) + (new1 - old1) + (new2 - old2)
        else:
            mn_x = tx1
            if tx2 < mn_x: mn_x = tx2
            if tx3 < mn_x: mn_x = tx3
            mx_x = tx1
            if tx2 > mx_x: mx_x = tx2
            if tx3 > mx_x: mx_x = tx3
            mn_y = ty1
            if ty2 < mn_y: mn_y = ty2
            if ty3 < mn_y: mn_y = ty3
            mx_y = ty1
            if ty2 > mx_y: mx_y = ty2
            if ty3 > mx_y: mx_y = ty3
            bx0 = mn_x - thick
            if bx0 < 0: bx0 = 0
            by0 = mn_y - thick
            if by0 < 0: by0 = 0
            bx1 = mx_x + thick + 1
            if bx1 > img_w: bx1 = img_w
            by1 = mx_y + thick + 1
            if by1 > img_h: by1 = img_h
            for py in range(by0, by1):
                for px in range(bx0, bx1):
                    d1 = (px - tx1) * (ty2 - ty1) - (tx2 - tx1) * (py - ty1)
                    d2 = (px - tx2) * (ty3 - ty2) - (tx3 - tx2) * (py - ty2)
                    d3 = (px - tx3) * (ty1 - ty3) - (tx1 - tx3) * (py - ty3)
                    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
                    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
                    inside = not (has_neg and has_pos)
                    if not inside:
                        continue
                    c0 = canvas_float[py, px, 0]; c1 = canvas_float[py, px, 1]; c2 = canvas_float[py, px, 2]
                    t0 = target_float[py, px, 0]; t1 = target_float[py, px, 1]; t2 = target_float[py, px, 2]
                    old0 = c0 - t0
                    if old0 < 0: old0 = -old0
                    old1 = c1 - t1
                    if old1 < 0: old1 = -old1
                    old2 = c2 - t2
                    if old2 < 0: old2 = -old2
                    nb0 = c0 * (1.0 - alpha_val) + b * alpha_val
                    nb1 = c1 * (1.0 - alpha_val) + g * alpha_val
                    nb2 = c2 * (1.0 - alpha_val) + rr * alpha_val
                    new0 = nb0 - t0
                    if new0 < 0: new0 = -new0
                    new1 = nb1 - t1
                    if new1 < 0: new1 = -new1
                    new2 = nb2 - t2
                    if new2 < 0: new2 = -new2
                    delta += (new0 - old0) + (new1 - old1) + (new2 - old2)
        cand_loss = (current_total + delta) / total_pixels
        if cand_loss < best_loss:
            best_loss = cand_loss
            best_idx = i
    return best_idx, best_loss


def _triangle_scanline_delta_plain(canvas, target, x1, y1, x2, y2, x3, y3,
                                   b, g, r, alpha):
    """NORMAL-PYTHON equivalent of `_jit_eval_triangle_scanline_partial_batch`
    (one triangle - readable reference).

    Same delta idea as the circle: walk the triangle's bounding box, keep only
    the pixels inside the triangle (point-in-tri test), and for each add
    (error_after - error_before) of alpha-painting the colour.
    """
    delta = 0.0
    H, W = canvas.shape[:2]
    xs = (x1, x2, x3)
    ys = (y1, y2, y3)
    for y in range(max(0, int(min(ys)) - 1), min(H, int(max(ys)) + 2)):
        for x in range(max(0, int(min(xs)) - 1), min(W, int(max(xs)) + 2)):
            if not _point_in_tri_plain(x, y, x1, y1, x2, y2, x3, y3):
                continue
            old = np.abs(canvas[y, x] - target[y, x]).sum()
            new_col = canvas[y, x] * (1.0 - alpha) + np.array([b, g, r], dtype=np.float32) * alpha
            delta += np.abs(new_col - target[y, x]).sum() - old
    return delta


@numba.njit(cache=True)
def _jit_eval_line_partial_batch(canvas_float, target_float,
                          x1_arr, y1_arr, x2_arr, y2_arr, thick_arr,
                          b_arr, g_arr, r_arr, alpha_val, img_w, img_h, current_total):
    n = len(thick_arr)
    best_loss = 1e30
    best_idx = -1
    total_pixels = float(img_w * img_h * 3)
    for i in range(n):
        lx1 = int(x1_arr[i]); ly1 = int(y1_arr[i]); lx2 = int(x2_arr[i]); ly2 = int(y2_arr[i])
        thick = int(thick_arr[i])
        b = b_arr[i]; g = g_arr[i]; rr = r_arr[i]
        ht = thick // 2
        if ht < 1: ht = 1
        ht2 = ht*ht
        mn_x = lx1 if lx1 < lx2 else lx2
        mx_x = lx1 if lx1 > lx2 else lx2
        mn_y = ly1 if ly1 < ly2 else ly2
        mx_y = ly1 if ly1 > ly2 else ly2
        bx0 = mn_x - ht
        if bx0 < 0: bx0 = 0
        by0 = mn_y - ht
        if by0 < 0: by0 = 0
        bx1 = mx_x + ht + 1
        if bx1 > img_w: bx1 = img_w
        by1 = mx_y + ht + 1
        if by1 > img_h: by1 = img_h
        delta = 0.0
        dx = float(lx2 - lx1); dy = float(ly2 - ly1)
        len2 = dx*dx + dy*dy
        for py in range(by0, by1):
            for px in range(bx0, bx1):
                d2 = 0.0
                if len2 < 1.0:
                    d2 = float((px - lx1)*(px - lx1) + (py - ly1)*(py - ly1))
                else:
                    t = ((px - lx1)*dx + (py - ly1)*dy) / len2
                    if t < 0.0: t = 0.0
                    elif t > 1.0: t = 1.0
                    proj_x = lx1 + t*dx
                    proj_y = ly1 + t*dy
                    d2 = (px - proj_x)*(px - proj_x) + (py - proj_y)*(py - proj_y)
                if d2 > ht2:
                    continue
                c0 = canvas_float[py, px, 0]; c1 = canvas_float[py, px, 1]; c2 = canvas_float[py, px, 2]
                t0 = target_float[py, px, 0]; t1 = target_float[py, px, 1]; t2 = target_float[py, px, 2]
                old0 = c0 - t0
                if old0 < 0: old0 = -old0
                old1 = c1 - t1
                if old1 < 0: old1 = -old1
                old2 = c2 - t2
                if old2 < 0: old2 = -old2
                nb0 = c0 * (1.0 - alpha_val) + b * alpha_val
                nb1 = c1 * (1.0 - alpha_val) + g * alpha_val
                nb2 = c2 * (1.0 - alpha_val) + rr * alpha_val
                new0 = nb0 - t0
                if new0 < 0: new0 = -new0
                new1 = nb1 - t1
                if new1 < 0: new1 = -new1
                new2 = nb2 - t2
                if new2 < 0: new2 = -new2
                delta += (new0 - old0) + (new1 - old1) + (new2 - old2)
        cand_loss = (current_total + delta) / total_pixels
        if cand_loss < best_loss:
            best_loss = cand_loss
            best_idx = i
    return best_idx, best_loss


def _line_partial_delta_plain(canvas, target, x1, y1, x2, y2, thick, b, g, r, alpha):
    """NORMAL-PYTHON equivalent of `_jit_eval_line_partial_batch` (one line).

    Returns `delta`: the change in whole-image L1 error if we paint this thick
    line (colour b,g,r) onto `canvas`. Walk the segment's bounding box, keep
    the pixels within `thick` of the segment, and add (error_after -
    error_before) for each.
    """
    delta = 0.0
    H, W = canvas.shape[:2]
    for y in range(max(0, int(min(y1, y2)) - thick), min(H, int(max(y1, y2)) + thick + 1)):
        for x in range(max(0, int(min(x1, x2)) - thick), min(W, int(max(x1, x2)) + thick + 1)):
            if _dist_point_to_segment(x, y, x1, y1, x2, y2) > thick:
                continue
            old = np.abs(canvas[y, x] - target[y, x]).sum()
            new_col = canvas[y, x] * (1.0 - alpha) + np.array([b, g, r], dtype=np.float32) * alpha
            delta += np.abs(new_col - target[y, x]).sum() - old
    return delta


def _dist_point_to_segment(px, py, x1, y1, x2, y2):
    """Shortest distance from point (px,py) to segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return ((px - (x1 + t * dx)) ** 2 + (py - (y1 + t * dy)) ** 2) ** 0.5


@numba.njit(cache=True)
def _jit_clip_pos(v, lo, hi):
    if v < lo:
        return lo
    elif v > hi:
        return hi
    else:
        return int(v)


# Cross-platform font fallback: try the requested font first, then a list of
# fonts that usually exist on Windows / macOS / Linux, then Pillow's built-in
# default. This keeps the repo runnable anywhere without bundling a .ttf.
_FONT_FALLBACKS = (
    "arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttf",
    "/System/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


@lru_cache(maxsize=64)
def _get_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    """Load a TTF font, falling back across platforms. See _FONT_FALLBACKS."""
    candidates = (font_path,) if font_path else ()
    for path in candidates + _FONT_FALLBACKS:
        try:
            return ImageFont.truetype(path, font_size)
        except (OSError, IOError, TypeError):
            continue
    # Last resort: Pillow's built-in bitmap font (still usable for putText).
    return ImageFont.load_default()


@lru_cache(maxsize=64)
def _get_base_glyph_arr(font_path: str, letter: str) -> np.ndarray:
    font = _get_font(font_path, 64)
    bbox = font.getbbox(letter)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    img = Image.new("L", (w + 2, h + 2), 0)
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0], -bbox[1]), letter, font=font, fill=255)
    return np.array(img, dtype=np.float32) / 255.0


def _make_letter_mask(font_path: str, letter: str,
                      rotation: float, scale: float) -> np.ndarray:
    glyph = _get_base_glyph_arr(font_path, letter)
    h_in, w_in = glyph.shape

    if rotation != 0.0:
        center = (w_in / 2.0, h_in / 2.0)
        mat = cv2.getRotationMatrix2D(center, rotation, 1.0)
        cos_a = abs(mat[0, 0])
        sin_a = abs(mat[0, 1])
        new_w_rot = int(h_in * sin_a + w_in * cos_a)
        new_h_rot = int(h_in * cos_a + w_in * sin_a)
        mat[0, 2] += (new_w_rot / 2.0) - center[0]
        mat[1, 2] += (new_h_rot / 2.0) - center[1]
        glyph = cv2.warpAffine(glyph, mat, (new_w_rot, new_h_rot),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    target_w = max(1, int(glyph.shape[1] * scale * LOSS_SCALE))
    target_h = max(1, int(glyph.shape[0] * scale * LOSS_SCALE))
    return cv2.resize(glyph, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def draw_letter(
    base_img: Image.Image, letter: str, font_path: str, font_size: int,
    rgba: tuple[int, int, int, int], x: int, y: int,
    rotation: float = 0.0, scale: float = 1.0,
) -> tuple[int, int, int, int]:
    """Legacy PIL renderer (top-left x,y). Kept for backward compat with
    ga_hc_j_variant_a_nopolish shapes. For cv2 Hershey shapes use draw_letter_cv2."""
    r, g, b, a = rgba
    mask = _make_letter_mask(font_path, letter, rotation, scale * 5.0)
    h, w = mask.shape
    arr = np.empty((h, w, 4), dtype=np.float32)
    arr[:, :, :3] = np.array((r / 255.0, g / 255.0, b / 255.0), dtype=np.float32)
    arr[:, :, 3] = mask * (a / 255.0)
    img = Image.fromarray((np.clip(arr, 0, 1,dtype=np.uint8) * 255), "RGBA")
    display_w = max(1, int(w / LOSS_SCALE))
    display_h = max(1, int(h / LOSS_SCALE))
    img = img.resize((display_w, display_h), Image.NEAREST)
    base_img.paste(img, (x, y), img)
    return (x, y, x + img.width, y + img.height)


def draw_letter_cv2(
    base_img: Image.Image, letter: str,
    rgba: tuple[int, int, int, int], x: int, y: int,
    scale: float = 1.0, thickness: int = 2,
) -> tuple[int, int, int, int]:
    """cv2 Hershey renderer matching run_ga's putText path.
    x,y is baseline org (same as shape_data x,y). Returns blended bbox."""
    r, g, b, a = rgba
    cv_font = cv2.FONT_HERSHEY_SIMPLEX
    # work in BGR for blending
    arr = np.array(base_img.convert("RGBA"))
    # arr is RGBA, convert to BGR for cv2 blending
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    h_img, w_img = bgr.shape[:2]
    (tw, th), bl = cv2.getTextSize(letter, cv_font, scale, thickness)
    x1, y1 = max(0, x), max(0, y - th)
    x2, y2 = min(w_img, x + tw), min(h_img, y + bl)
    if x2 <= x1 or y2 <= y1:
        return (x1, y1, x2, y2)
    patch_h, patch_w = y2 - y1, x2 - x1
    mask_canvas = np.zeros((patch_h, patch_w), dtype=np.float32)
    cv2.putText(mask_canvas, letter, (x - x1, y - y1), cv_font, scale, 255, thickness, cv2.LINE_AA)
    mask = mask_canvas 
    alpha_val = a / 255.0
    effective = mask * alpha_val
    # BGR color
    color_bgr = np.array((b, g, r), dtype=np.float32)
    # blend region
    region = bgr[y1:y2, x1:x2].astype(np.float32)
    # region shape HxWx3, effective HxWx1
    eff3 = effective[:, :, None]
    blended = color_bgr * eff3 + region * (1.0 - eff3)
    bgr[y1:y2, x1:x2] = blended.astype(np.uint8)
    # convert back to RGBA PIL
    rgba_arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    # preserve original alpha where not drawn? For simplicity keep opaque
    # paste back
    out = Image.fromarray(rgba_arr, "RGBA")
    base_img.paste(out, (0, 0))
    return (x1, y1, x2, y2)


_SHAPE_INT_MAP = {"circle": 0, "triangle": 1, "line": 2}


@numba.njit(cache=True)
def _jit_mutate_batch(
    best_x1, best_y1, best_x2, best_y2, best_x3, best_y3,
    best_R, best_thickness, best_b, best_g, best_r,
    shape_type_int, img_w, img_h,
    jitter_pos, jitter_rad, jitter_thk,
    out_x1, out_y1, out_x2, out_y2, out_x3, out_y3,
    out_R, out_thick, out_b, out_g, out_r,
):
    """Generate N mutated candidates from best params. shape_type_int: 0=circle,1=tri,2=line."""
    n = len(out_x1)
    for i in range(n):
        out_x1[i] = best_x1; out_y1[i] = best_y1
        out_x2[i] = best_x2; out_y2[i] = best_y2
        out_x3[i] = best_x3; out_y3[i] = best_y3
        out_R[i] = best_R
        out_thick[i] = best_thickness
        out_b[i] = best_b; out_g[i] = best_g; out_r[i] = best_r
    for i in range(n):
        mut = np.random.randint(0, 5)
        if mut == 0:
            ch = np.random.randint(0, 3)
            nv = np.random.normal(0.0, 25.0)
            if ch == 0:
                out_r[i] = min(255.0, max(0.0, out_r[i] + nv))
            elif ch == 1:
                out_g[i] = min(255.0, max(0.0, out_g[i] + nv))
            else:
                out_b[i] = min(255.0, max(0.0, out_b[i] + nv))
        elif mut == 1:
            out_thick[i] = min(10.0, max(1.0, out_thick[i] + np.random.randint(-jitter_thk, jitter_thk + 1)))
        elif mut == 2:
            jp = jitter_pos
            dx = np.random.normal(0.0, jp)
            dy = np.random.normal(0.0, jp)
            if shape_type_int == 0:
                out_x1[i] = min(float(img_w - 1), max(0.0, out_x1[i] + dx))
                out_y1[i] = min(float(img_h - 1), max(0.0, out_y1[i] + dy))
            elif shape_type_int == 1:
                v = np.random.randint(0, 3)
                if v == 0:
                    out_x1[i] = min(float(img_w - 1), max(0.0, out_x1[i] + dx))
                    out_y1[i] = min(float(img_h - 1), max(0.0, out_y1[i] + dy))
                elif v == 1:
                    out_x2[i] = min(float(img_w - 1), max(0.0, out_x2[i] + dx))
                    out_y2[i] = min(float(img_h - 1), max(0.0, out_y2[i] + dy))
                else:
                    out_x3[i] = min(float(img_w - 1), max(0.0, out_x3[i] + dx))
                    out_y3[i] = min(float(img_h - 1), max(0.0, out_y3[i] + dy))
            else:
                v = np.random.randint(0, 2)
                if v == 0:
                    out_x1[i] = min(float(img_w - 1), max(0.0, out_x1[i] + dx))
                    out_y1[i] = min(float(img_h - 1), max(0.0, out_y1[i] + dy))
                else:
                    out_x2[i] = min(float(img_w - 1), max(0.0, out_x2[i] + dx))
                    out_y2[i] = min(float(img_h - 1), max(0.0, out_y2[i] + dy))
        elif mut == 3:
            if shape_type_int == 0:
                max_r = min(float(img_w), float(img_h)) * 0.4
                out_R[i] = min(max_r, max(3.0, out_R[i] + np.random.randint(-jitter_rad, jitter_rad + 1)))
            else:
                out_thick[i] = min(10.0, max(1.0, out_thick[i] + np.random.randint(-jitter_thk, jitter_thk + 1)))
        else:
            out_r[i] = float(np.random.randint(0, 256))
            out_g[i] = float(np.random.randint(0, 256))
            out_b[i] = float(np.random.randint(0, 256))
            out_thick[i] = float(np.random.randint(1, 11))
            if shape_type_int == 0:
                out_R[i] = float(np.random.randint(3, 306))
                out_x1[i] = float(np.random.randint(0, img_w))
                out_y1[i] = float(np.random.randint(0, img_h))
            elif shape_type_int == 1:
                out_x1[i] = float(np.random.randint(0, img_w)); out_y1[i] = float(np.random.randint(0, img_h))
                out_x2[i] = float(np.random.randint(0, img_w)); out_y2[i] = float(np.random.randint(0, img_h))
                out_x3[i] = float(np.random.randint(0, img_w)); out_y3[i] = float(np.random.randint(0, img_h))
            else:
                out_x1[i] = float(np.random.randint(0, img_w)); out_y1[i] = float(np.random.randint(0, img_h))
                out_x2[i] = float(np.random.randint(0, img_w)); out_y2[i] = float(np.random.randint(0, img_h))


def _mutate_plain(best, shape_type_int, img_w, img_h, jitter_pos, jitter_rad, jitter_thk):
    """NORMAL-PYTHON equivalent of `_jit_mutate_batch` (one child, readable).

    Copy the best shape's parameters, then apply ONE random mutation:
      0  nudge one colour channel by a small Gaussian
      1  nudge the thickness
      2  nudge a position (centre, or one random triangle/line vertex)
      3  nudge the radius (circle) / thickness (other shapes)
      4  full random "restart" of colour, thickness and position
    This is the local-search step run after every batch.
    """
    import random as _r
    p = dict(best)
    clamp = lambda v, lo, hi: min(hi, max(lo, v))
    mode = _r.randint(0, 4)
    if mode == 0:                                   # nudge one colour channel
        key = _r.choice(['b', 'g', 'r'])
        p[key] = clamp(p[key] + _r.gauss(0, 25), 0, 255)
    elif mode == 1:                                 # nudge thickness
        p['thick'] = clamp(p['thick'] + _r.randint(-jitter_thk, jitter_thk + 1), 1, 10)
    elif mode == 2:                                 # nudge a position
        dx, dy = _r.gauss(0, jitter_pos), _r.gauss(0, jitter_pos)
        keys = [1] if shape_type_int == 0 else ([1, 2, 3] if shape_type_int == 1 else [1, 2])
        k = _r.choice(keys)
        p[f'x{k}'] = clamp(p[f'x{k}'] + dx, 0, img_w - 1)
        p[f'y{k}'] = clamp(p[f'y{k}'] + dy, 0, img_h - 1)
    elif mode == 3:                                 # nudge radius / thickness
        if shape_type_int == 0:
            p['R'] = clamp(p['R'] + _r.randint(-jitter_rad, jitter_rad + 1), 3, min(img_w, img_h) * 0.4)
        else:
            p['thick'] = clamp(p['thick'] + _r.randint(-jitter_thk, jitter_thk + 1), 1, 10)
    else:                                           # 4: full random restart
        p['r'], p['g'], p['b'] = _r.randint(0, 255), _r.randint(0, 255), _r.randint(0, 255)
        p['thick'] = _r.randint(1, 11)
        keys = [1] if shape_type_int == 0 else ([1, 2, 3] if shape_type_int == 1 else [1, 2])
        for k in keys:
            p[f'x{k}'], p[f'y{k}'] = _r.randint(0, img_w), _r.randint(0, img_h)
        if shape_type_int == 0:
            p['R'] = _r.randint(3, 306)
    return p


# =====================================================================
# Section 9 — run_ga  (the "brain": ties it all together)
# ---------------------------------------------------------------------
# A generator that yields the full-resolution canvas once per shape. It builds
# a fast downscale of the target (optionally per overlapping crop), seeds a
# champion shape, then loops: Sobol-sample a batch -> score with the batch
# JIT -> keep the best -> adaptively mutate -> keep the mutation only if it
# lowers the loss (monotonic gate). See the module docstring at the top for
# the high-level flow. `draw_letter` below is a small helper used by the app.
# =====================================================================
def run_ga(target_img, font_path, font_size, max_shapes, population_size,
           img_w, img_h, mutation_factor=0.7, crossover_rate=0.9,
           seed=None, elite_pool=None, epsilon=0.15, alpha=128,
           max_generations=200, patience=8, shape_timeout=5.0,
            obl_jump=False, verbose=True, soft_restarts=0, memetic_every=0,
            es_restarts=2, loss_mode="pooled", random_pct=None, guided_pct=0.4,
            clip_scale=(3.0, 15.0), clip_color=(0, 255), shape_types=None,
            use_one_fifth=False, use_ema=False, ema_alpha=0.82, ema_tau=0.22):
    del mutation_factor, crossover_rate, elite_pool
    del epsilon, max_generations, soft_restarts, memetic_every
    del es_restarts, obl_jump
    # population_size now controls per-shape eval budget: total random perms = population_size * 30 (~1500 for 50)
    if population_size is None:
        max_evals = 400
    else:
        max_evals = int(max( _BATCH, int(population_size) * 8 ))

    alpha_fixed = int(np.clip(int(alpha), 16, 240))
    rng = np.random.default_rng(seed)
    clip_scale_lo, clip_scale_hi = clip_scale
    clip_color_lo, clip_color_hi = clip_color
    # shape types handling — only circles/triangles/lines (letters removed)
    if shape_types is None:
        shape_types = SHAPE_TYPES
    shape_types = [s for s in shape_types if s in SHAPE_TYPES]
    if not shape_types:
        shape_types = SHAPE_TYPES
    sobol_d = 14
    _sobol = qmc.Sobol(d=sobol_d, scramble=True, seed=int(rng.integers(0, 1 << 30)))
    def _next_sobol():
        return _sobol.random(n=1)[0]

    target_cv = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_cv = cv2.resize(target_cv, (img_w, img_h))
    target_float = target_cv.astype(np.float32)
    # --- downscaled canvases for fast candidate eval (Geometrize-style) ---
    _small_scale_w = _SMALL_W / float(img_w)
    _small_scale_h = _SMALL_H / float(img_h)
    _small_target_cv = cv2.resize(target_cv, (_SMALL_W, _SMALL_H), interpolation=cv2.INTER_AREA)
    _small_target_float = _small_target_cv.astype(np.float32)

    avg_color = target_cv.mean(axis=(0, 1)).astype(np.uint8)
    canvas_cv = np.full_like(target_cv, avg_color)
    canvas_float = canvas_cv.astype(np.float32)
    _small_canvas_cv = cv2.resize(canvas_cv, (_SMALL_W, _SMALL_H), interpolation=cv2.INTER_AREA)
    _small_canvas_float = _small_canvas_cv.astype(np.float32)

    current_loss = np.sum(np.abs(canvas_float - target_float)) / target_float.size

    bg_rgb = cv2.cvtColor(canvas_cv, cv2.COLOR_BGR2RGB)
    full_bg = Image.fromarray(bg_rgb).convert("RGBA")

    cv_font = cv2.FONT_HERSHEY_SIMPLEX
    fixed_shapes = []
    shape_idx = 0

    # Pre-allocate batch arrays OUTSIDE shape loop (avoid per-shape malloc)
    _i_x1 = np.empty(_BATCH, dtype=np.float64)
    _i_y1 = np.empty(_BATCH, dtype=np.float64)
    _i_x2 = np.empty(_BATCH, dtype=np.float64)
    _i_y2 = np.empty(_BATCH, dtype=np.float64)
    _i_x3 = np.empty(_BATCH, dtype=np.float64)
    _i_y3 = np.empty(_BATCH, dtype=np.float64)
    _i_R = np.empty(_BATCH, dtype=np.float64)
    _i_thick = np.empty(_BATCH, dtype=np.float64)
    _i_b = np.empty(_BATCH, dtype=np.float64)
    _i_g = np.empty(_BATCH, dtype=np.float64)
    _i_r = np.empty(_BATCH, dtype=np.float64)
    _i_type_idx = np.empty(_BATCH, dtype=np.int64)
    _i_type = [None] * _BATCH  # kept for compat but unused in hot path after int optimization
    _b_x1 = np.empty(_BATCH, dtype=np.float64)
    _b_y1 = np.empty(_BATCH, dtype=np.float64)
    _b_x2 = np.empty(_BATCH, dtype=np.float64)
    _b_y2 = np.empty(_BATCH, dtype=np.float64)
    _b_x3 = np.empty(_BATCH, dtype=np.float64)
    _b_y3 = np.empty(_BATCH, dtype=np.float64)
    _b_R = np.empty(_BATCH, dtype=np.float64)
    _b_thick = np.empty(_BATCH, dtype=np.float64)
    _b_b = np.empty(_BATCH, dtype=np.float64)
    _b_g = np.empty(_BATCH, dtype=np.float64)
    _b_r = np.empty(_BATCH, dtype=np.float64)
    # downscaled copies for fast eval
    _s_x1 = np.empty(_BATCH, dtype=np.float64)
    _s_y1 = np.empty(_BATCH, dtype=np.float64)
    _s_x2 = np.empty(_BATCH, dtype=np.float64)
    _s_y2 = np.empty(_BATCH, dtype=np.float64)
    _s_x3 = np.empty(_BATCH, dtype=np.float64)
    _s_y3 = np.empty(_BATCH, dtype=np.float64)
    _s_R = np.empty(_BATCH, dtype=np.float64)
    _s_thick = np.empty(_BATCH, dtype=np.float64)

    # hoist constant alpha (was per-shape)
    alpha_val = alpha_fixed / 255.0
    # reusable max patch buffer (worst bbox ~ 306x306x3 ~1.1MB) to avoid per-winner alloc
    _patch_buf = np.empty((400, 400, 3), dtype=np.float32)  # sliced as needed

    while max_shapes is None or shape_idx < max_shapes:
        shape_t0 = time.perf_counter()
        def evaluate_gene(c_idx, scale, thick, col, x, y, shape_type="letter", rot=0, x1=None, y1=None, x2=None, y2=None, x3=None, y3=None, R=None):
            b_col, g_col, r_col = col[2], col[1], col[0]
            if shape_type == "circle":
                radius = R if R is not None else max(1, int(scale))
                bx1 = x - radius
                if bx1 < 0: bx1 = 0
                by1 = y - radius
                if by1 < 0: by1 = 0
                bx2 = x + radius + 1
                if bx2 > img_w: bx2 = img_w
                by2 = y + radius + 1
                if by2 > img_h: by2 = img_h
                if bx2 <= bx1 or by2 <= by1:
                    return float('inf'), None, None
                mask_canvas = np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
                _jit_draw_shape(mask_canvas, 0, x - bx1, y - by1, 0, 0, 0, 0, 0, 0, radius, thick)
                local_canvas_old = canvas_float[by1:by2, bx1:bx2]
                local_target = target_float[by1:by2, bx1:bx2]
                loss_val = _jit_loss(mask_canvas, local_canvas_old, local_target, b_col, g_col, r_col, alpha_val)
                return loss_val, mask_canvas, (bx1, by1, bx2, by2)
            elif shape_type == "triangle":
                if x1 is None or x2 is None or x3 is None:
                    return float('inf'), None, None
                mn_x = x1
                if x2 < mn_x: mn_x = x2
                if x3 < mn_x: mn_x = x3
                mx_x = x1
                if x2 > mx_x: mx_x = x2
                if x3 > mx_x: mx_x = x3
                mn_y = y1
                if y2 < mn_y: mn_y = y2
                if y3 < mn_y: mn_y = y3
                mx_y = y1
                if y2 > mx_y: mx_y = y2
                if y3 > mx_y: mx_y = y3
                bx1 = mn_x - thick
                if bx1 < 0: bx1 = 0
                by1 = mn_y - thick
                if by1 < 0: by1 = 0
                bx2 = mx_x + thick + 1
                if bx2 > img_w: bx2 = img_w
                by2 = mx_y + thick + 1
                if by2 > img_h: by2 = img_h
                if bx2 <= bx1 or by2 <= by1:
                    return float('inf'), None, None
                mask_canvas = np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
                _jit_draw_shape(mask_canvas, 1, 0, 0,
                                x1 - bx1, y1 - by1, x2 - bx1, y2 - by1, x3 - bx1, y3 - by1, 0, thick)
                local_canvas_old = canvas_float[by1:by2, bx1:bx2]
                local_target = target_float[by1:by2, bx1:bx2]
                loss_val = _jit_loss(mask_canvas, local_canvas_old, local_target, b_col, g_col, r_col, alpha_val)
                return loss_val, mask_canvas, (bx1, by1, bx2, by2)
            elif shape_type == "line":
                if x1 is None or x2 is None:
                    return float('inf'), None, None
                mn_x = x1 if x1 < x2 else x2
                mx_x = x1 if x1 > x2 else x2
                mn_y = y1 if y1 < y2 else y2
                mx_y = y1 if y1 > y2 else y2
                bx1 = mn_x - thick
                if bx1 < 0: bx1 = 0
                by1 = mn_y - thick
                if by1 < 0: by1 = 0
                bx2 = mx_x + thick + 1
                if bx2 > img_w: bx2 = img_w
                by2 = mx_y + thick + 1
                if by2 > img_h: by2 = img_h
                if bx2 <= bx1 or by2 <= by1:
                    return float('inf'), None, None
                mask_canvas = np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
                _jit_draw_shape(mask_canvas, 2, 0, 0,
                                x1 - bx1, y1 - by1, x2 - bx1, y2 - by1, 0, 0, 0, thick)
                local_canvas_old = canvas_float[by1:by2, bx1:bx2]
                local_target = target_float[by1:by2, bx1:bx2]
                loss_val = _jit_loss(mask_canvas, local_canvas_old, local_target, b_col, g_col, r_col, alpha_val)
                return loss_val, mask_canvas, (bx1, by1, bx2, by2)
            else:
                char = ALPHABET[c_idx]
                (tw, th), bl = _cached_text_size(char, scale, thick)
                x1, y1 = max(0, x), max(0, y - th)
                x2, y2 = min(img_w, x + tw), min(img_h, y + bl)
                if x2 <= x1 or y2 <= y1:
                    return float('inf'), None, None
                local_canvas_old = canvas_float[y1:y2, x1:x2]
                local_target = target_float[y1:y2, x1:x2]
                patch_h, patch_w = y2 - y1, x2 - x1
                mask_canvas = np.zeros((patch_h, patch_w), dtype=np.uint8)
                cv2.putText(mask_canvas, char, (x - x1, y - y1), cv_font, scale, 255, thick, cv2.LINE_AA)
                b_col, g_col, r_col = col[2], col[1], col[0]
                loss_val = _jit_loss(mask_canvas, local_canvas_old, local_target, b_col, g_col, r_col, alpha_val)
                return loss_val, mask_canvas, (x1, y1, x2, y2)

        # residual for guided — err2 sum squared per pixel (lazy, only if guided needed)
        guided_n = 0
        total = 0.0
        cum = None
        err2 = None
        if guided_pct and guided_pct > 0:
            err2 = np.sum((canvas_float - target_float) ** 2, axis=2)
            cum = np.cumsum(err2.ravel())
            total = float(cum[-1]) if cum.size else 0.0
            guided_n = int(_BATCH * guided_pct) if total > 1e-6 else 0
        # anneal scale: big early (8→20), normal late (4→15)
        if max_shapes:
            prog = shape_idx / max_shapes
            scale_lo = 8.0 * (1 - prog) + 4.0 * prog
            scale_hi = 20.0 * (1 - prog) + 15.0 * prog
        else:
            scale_lo, scale_hi = 8.0, 20.0
        scale_range = scale_hi - scale_lo
        # --- adaptive stochastic: global vs crop, 64 vs 128 (works with max_shapes=None) ---
        if max_shapes is not None and max_shapes > 0:
            _prog_adaptive = shape_idx / float(max_shapes)
        else:
            _prog_adaptive = shape_idx / (shape_idx + 100.0) if shape_idx > 0 else 0.0
        _prog_adaptive = float(np.clip(_prog_adaptive, 0.0, 1.0))
        p_global = 1.0 - 0.85 * _prog_adaptive
        p_global = float(np.clip(p_global, 0.15, 1.0))
        is_global = bool(rng.random() < p_global)
        if is_global:
            p_128 = 0.2 + 0.5 * _prog_adaptive
            _use_small = 128 if rng.random() < p_128 else 64
            _small_w = _small_h = _use_small
            _small_scale_w = _small_w / float(img_w)
            _small_scale_h = _small_h / float(img_h)
            _small_target_cv = cv2.resize(target_cv, (_small_w, _small_h), interpolation=cv2.INTER_AREA)
            _small_target_float = _small_target_cv.astype(np.float32)
            _small_canvas_cv = cv2.resize(canvas_cv, (_small_w, _small_h), interpolation=cv2.INTER_AREA)
            _small_canvas_float = _small_canvas_cv.astype(np.float32)
            _eval_mode = "global"
            _patch_x1 = _patch_y1 = _patch_x2 = _patch_y2 = 0
            _patch_w = _patch_h = 0
        else:
            if rng.random() < (0.3 + 0.4 * _prog_adaptive):
                _patch_w = 64
            else:
                _patch_w = 96 if rng.random() < 0.5 else 128
            _patch_h = _patch_w
            _patch_w = min(_patch_w, img_w)
            _patch_h = min(_patch_h, img_h)
            if total > 1e-6:
                rv = rng.random() * total
                idx = int(np.searchsorted(cum, rv))
                idx = int(np.clip(idx, 0, img_w*img_h-1))
                pcy, pcx = divmod(idx, img_w)
            else:
                pcx = int(rng.integers(0, img_w))
                pcy = int(rng.integers(0, img_h))
            _patch_x1 = int(np.clip(pcx - _patch_w//2, 0, img_w - _patch_w))
            _patch_y1 = int(np.clip(pcy - _patch_h//2, 0, img_h - _patch_h))
            _patch_x2 = _patch_x1 + _patch_w
            _patch_y2 = _patch_y1 + _patch_h
            _eval_mode = "crop"
            _patch_target_float = target_float[_patch_y1:_patch_y2, _patch_x1:_patch_x2].copy()
            _patch_canvas_float = canvas_float[_patch_y1:_patch_y2, _patch_x1:_patch_x2].copy()
            _small_w = _small_h = 64
        # Batch init — Sobol low-discrepancy, _BATCH candidates evaluated via batch JIT
        S_init = _sobol.random(n=_BATCH)
        best_gene_loss = float('inf')
        best_patch_float = None
        best_bounds = None
        best_scale = 0.0
        best_thickness = 1
        best_color = (0, 0, 0)
        best_x = 0
        best_y = 0
        best_x1, best_y1, best_x2, best_y2, best_x3, best_y3, best_R = 0,0,0,0,0,0,10
        best_shape_type = shape_types[0]
        # --- fully vectorized pass 1: generate all candidates ---
        # shape types from Sobol dim 5 — keep as int index to avoid Python string loop
        _i_type_idx[:] = (S_init[:, 5] * len(shape_types)).astype(int) % len(shape_types)
        # legacy _i_type list kept but not used in hot path (int index is faster)
        # thickness from Sobol dim 4
        _i_thick[:] = np.clip((S_init[:, 4] * 10).astype(int) + 1, 1, 10)
        # base color from Sobol dims 0-2
        _i_b[:] = (S_init[:, 0] * 256).astype(int) % 256
        _i_g[:] = (S_init[:, 1] * 256).astype(int) % 256
        _i_r[:] = (S_init[:, 2] * 256).astype(int) % 256
        # guided sampling: residual-weighted positions + colors
        if guided_n > 0 and total > 0:
            rv = rng.random(guided_n) * total
            guided_idx = np.searchsorted(cum, rv)
            np.clip(guided_idx, 0, img_w * img_h - 1, out=guided_idx)
            g_py, g_px = divmod(guided_idx, img_w)
            g_b = target_float[g_py, g_px, 2] + rng.normal(0, 12, guided_n)
            g_g = target_float[g_py, g_px, 1] + rng.normal(0, 12, guided_n)
            g_r = target_float[g_py, g_px, 0] + rng.normal(0, 12, guided_n)
            np.clip(g_b, 0, 255, out=g_b); np.clip(g_g, 0, 255, out=g_g); np.clip(g_r, 0, 255, out=g_r)
            _i_b[:guided_n] = g_b; _i_g[:guided_n] = g_g; _i_r[:guided_n] = g_r
        # Sobol coords for all dims 6-13
        sob_x = S_init[:, 6] * img_w
        sob_y = S_init[:, 7] * img_h
        np.clip(sob_x, 0, img_w - 1, out=sob_x)
        np.clip(sob_y, 0, img_h - 1, out=sob_y)
        sob_x2 = S_init[:, 8] * img_w; sob_y2 = S_init[:, 9] * img_h
        np.clip(sob_x2, 0, img_w - 1, out=sob_x2); np.clip(sob_y2, 0, img_h - 1, out=sob_y2)
        sob_x3 = S_init[:, 10] * img_w; sob_y3 = S_init[:, 11] * img_h
        np.clip(sob_x3, 0, img_w - 1, out=sob_x3); np.clip(sob_y3, 0, img_h - 1, out=sob_y3)
        sob_x4 = S_init[:, 12] * img_w; sob_y4 = S_init[:, 13] * img_h
        np.clip(sob_x4, 0, img_w - 1, out=sob_x4); np.clip(sob_y4, 0, img_h - 1, out=sob_y4)
        # guided coords near residual pixel
        if guided_n > 0 and total > 0:
            gdx = g_px + rng.integers(-int(img_w*0.25), int(img_w*0.25)+1, guided_n)
            gdy = g_py + rng.integers(-int(img_h*0.25), int(img_h*0.25)+1, guided_n)
            np.clip(gdx, 0, img_w - 1, out=gdx); np.clip(gdy, 0, img_h - 1, out=gdy)
            gdx2 = g_px + rng.integers(-int(img_w*0.6), int(img_w*0.6)+1, guided_n)
            gdy2 = g_py + rng.integers(-int(img_h*0.6), int(img_h*0.6)+1, guided_n)
            np.clip(gdx2, 0, img_w - 1, out=gdx2); np.clip(gdy2, 0, img_h - 1, out=gdy2)
        # assign coords per shape type
        _i_x1[:] = sob_x; _i_y1[:] = sob_y
        _i_x2[:] = sob_x2; _i_y2[:] = sob_y2
        _i_x3[:] = sob_x3; _i_y3[:] = sob_y3
        _i_R[:] = np.clip(5 + S_init[:, 3] * min(img_w, img_h) * 0.4, 3, min(img_w, img_h) * 0.4)
        # override guided coords for first guided_n candidates (int index, no string loop)
        if guided_n > 0 and total > 0:
            # map shape_types index to SHAPE_INT for fast compare
            _circle_type_idx = shape_types.index("circle") if "circle" in shape_types else -1
            _tri_type_idx = shape_types.index("triangle") if "triangle" in shape_types else -1
            _line_type_idx = shape_types.index("line") if "line" in shape_types else -1
            is_circle = _i_type_idx == _circle_type_idx if _circle_type_idx>=0 else np.zeros(_BATCH, dtype=bool)
            is_tri = _i_type_idx == _tri_type_idx if _tri_type_idx>=0 else np.zeros(_BATCH, dtype=bool)
            is_line = _i_type_idx == _line_type_idx if _line_type_idx>=0 else np.zeros(_BATCH, dtype=bool)
            g_mask = np.arange(_BATCH) < guided_n
            # circles: guided jitter ±40
            cm = g_mask & is_circle
            if np.any(cm):
                cm_idx = np.where(cm)[0]
                n_cm = len(cm_idx)
                _i_x1[cm] = gdx[:n_cm]; _i_y1[cm] = gdy[:n_cm]
            # triangles: guided jitter ±100 for x1,y1
            tm = g_mask & is_tri
            if np.any(tm):
                tm_idx = np.where(tm)[0]
                n_tm = len(tm_idx)
                _i_x1[tm] = gdx2[:n_tm]; _i_y1[tm] = gdy2[:n_tm]
            # lines: guided jitter ±100 for x1,y1,x2,y2
            lm = g_mask & is_line
            if np.any(lm):
                lm_idx = np.where(lm)[0]
                n_lm = len(lm_idx)
                _i_x1[lm] = gdx2[:n_lm]; _i_y1[lm] = gdy2[:n_lm]
                gdx2b = g_px[lm_idx] + rng.integers(-int(img_w*0.6), int(img_w*0.6)+1, n_lm)
                gdy2b = g_py[lm_idx] + rng.integers(-int(img_h*0.6), int(img_h*0.6)+1, n_lm)
                np.clip(gdx2b, 0, img_w - 1, out=gdx2b); np.clip(gdy2b, 0, img_h - 1, out=gdy2b)
                _i_x2[lm] = gdx2b; _i_y2[lm] = gdy2b
        if _eval_mode == "crop":
            _i_x1[:] = np.clip(_i_x1, _patch_x1, _patch_x2-1)
            _i_y1[:] = np.clip(_i_y1, _patch_y1, _patch_y2-1)
            _i_x2[:] = np.clip(_i_x2, _patch_x1, _patch_x2-1)
            _i_y2[:] = np.clip(_i_y2, _patch_y1, _patch_y2-1)
            _i_x3[:] = np.clip(_i_x3, _patch_x1, _patch_x2-1)
            _i_y3[:] = np.clip(_i_y3, _patch_y1, _patch_y2-1)
            _i_R[:] = np.clip(_i_R, 3, min(_patch_w, _patch_h)*0.4)
            # clip+strict: for circles ensure fully inside patch (no truncated half-circles that look broken)
            if _eval_mode == "crop" and "circle" in shape_types:
                circ_mask = _i_type_idx == shape_types.index("circle")
                if np.any(circ_mask):
                    _i_x1[circ_mask] = np.clip(_i_x1[circ_mask], _patch_x1 + _i_R[circ_mask], _patch_x2 - 1 - _i_R[circ_mask])
                    _i_y1[circ_mask] = np.clip(_i_y1[circ_mask], _patch_y1 + _i_R[circ_mask], _patch_y2 - 1 - _i_R[circ_mask])
        # --- pass 2: batch evaluate by shape type (scanline + partial scoring) ---
        if _eval_mode == "global":
            _cur_total = float(np.sum(np.abs(_small_canvas_float - _small_target_float)))
            _s_x1[:] = np.clip(_i_x1 * _small_scale_w, 0, _small_w-1)
            _s_y1[:] = np.clip(_i_y1 * _small_scale_h, 0, _small_h-1)
            _s_x2[:] = np.clip(_i_x2 * _small_scale_w, 0, _small_w-1)
            _s_y2[:] = np.clip(_i_y2 * _small_scale_h, 0, _small_h-1)
            _s_x3[:] = np.clip(_i_x3 * _small_scale_w, 0, _small_w-1)
            _s_y3[:] = np.clip(_i_y3 * _small_scale_h, 0, _small_h-1)
            _s_R[:] = np.clip(_i_R * _small_scale_w, 1, min(_small_w,_small_h)*0.4)
            _s_thick[:] = np.clip(np.maximum(1, (_i_thick * _small_scale_w).astype(int)), 1, 10)
        else:
            _cur_total = float(np.sum(np.abs(_patch_canvas_float - _patch_target_float)))
            _s_x1[:] = _i_x1 - _patch_x1
            _s_y1[:] = _i_y1 - _patch_y1
            _s_x2[:] = _i_x2 - _patch_x1
            _s_y2[:] = _i_y2 - _patch_y1
            _s_x3[:] = _i_x3 - _patch_x1
            _s_y3[:] = _i_y3 - _patch_y1
            _s_R[:] = _i_R
            _s_thick[:] = _i_thick
        # int mapping via shape_idx -> SHAPE_INT without string loop
        # precompute shape_idx_to_int: shape_types[i] -> SHAPE_INT
        _shape_idx_to_int = np.array([_SHAPE_INT_MAP.get(s, 0) for s in shape_types], dtype=np.int64)
        _type_arr = _shape_idx_to_int[_i_type_idx]
        for ti, st in enumerate(shape_types):
            mask_t = _type_arr == ti
            if not np.any(mask_t):
                continue
            idx_arr = np.where(mask_t)[0]
            n = len(idx_arr)
            # extract sub-arrays (scaled for fast small eval, original for winner)
            sx1 = _i_x1[idx_arr]; sy1 = _i_y1[idx_arr]
            sx2 = _i_x2[idx_arr]; sy2 = _i_y2[idx_arr]
            sx3 = _i_x3[idx_arr]; sy3 = _i_y3[idx_arr]
            sR = _i_R[idx_arr]
            sthick = _i_thick[idx_arr]
            sb = _i_b[idx_arr]; sg = _i_g[idx_arr]; sr = _i_r[idx_arr]
            # scaled versions for small eval
            ssx1 = _s_x1[idx_arr]; ssy1 = _s_y1[idx_arr]
            ssx2 = _s_x2[idx_arr]; ssy2 = _s_y2[idx_arr]
            ssx3 = _s_x3[idx_arr]; ssy3 = _s_y3[idx_arr]
            ssR = _s_R[idx_arr]
            ssthick = _s_thick[idx_arr]
            if _eval_mode == "global":
                if st == "circle":
                    bi, bl = _jit_eval_circle_scanline_partial_batch(_small_canvas_float, _small_target_float, ssx1, ssy1, ssR, ssthick, sb, sg, sr, alpha_val, _small_w, _small_h, _cur_total)
                elif st == "line":
                    bi, bl = _jit_eval_line_partial_batch(_small_canvas_float, _small_target_float, ssx1, ssy1, ssx2, ssy2, ssthick, sb, sg, sr, alpha_val, _small_w, _small_h, _cur_total)
                else:
                    bi, bl = _jit_eval_triangle_scanline_partial_batch(_small_canvas_float, _small_target_float, ssx1, ssy1, ssx2, ssy2, ssx3, ssy3, ssthick, sb, sg, sr, alpha_val, _small_w, _small_h, _cur_total)
            else:
                if st == "circle":
                    bi, bl = _jit_eval_circle_scanline_partial_batch(_patch_canvas_float, _patch_target_float, ssx1, ssy1, ssR, ssthick, sb, sg, sr, alpha_val, _patch_w, _patch_h, _cur_total)
                elif st == "line":
                    bi, bl = _jit_eval_line_partial_batch(_patch_canvas_float, _patch_target_float, ssx1, ssy1, ssx2, ssy2, ssthick, sb, sg, sr, alpha_val, _patch_w, _patch_h, _cur_total)
                else:
                    bi, bl = _jit_eval_triangle_scanline_partial_batch(_patch_canvas_float, _patch_target_float, ssx1, ssy1, ssx2, ssy2, ssx3, ssy3, ssthick, sb, sg, sr, alpha_val, _patch_w, _patch_h, _cur_total)
            if bi < 0:
                continue
            gi = idx_arr[bi]  # global index
            # extract winner params
            wt = int(sthick[bi]); wb = int(sb[bi]); wg = int(sg[bi]); wr = int(sr[bi])
            cand_color = (wr, wg, wb)
            if st == "circle":
                wx1 = int(sx1[bi]); wy1 = int(sy1[bi]); wR = int(sR[bi])
                bx1 = wx1 - wR
                if bx1 < 0: bx1 = 0
                by1 = wy1 - wR
                if by1 < 0: by1 = 0
                bx2 = wx1 + wR + 1
                if bx2 > img_w: bx2 = img_w
                by2 = wy1 + wR + 1
                if by2 > img_h: by2 = img_h
                # strict crop: intersect with patch
                if _eval_mode == "crop":
                    bx1 = max(bx1, _patch_x1); by1 = max(by1, _patch_y1)
                    bx2 = min(bx2, _patch_x2); by2 = min(by2, _patch_y2)
                if bx2 <= bx1 or by2 <= by1:
                    continue
                cand_bounds = (bx1, by1, bx2, by2)
                patch_float = np.empty((by2 - by1, bx2 - bx1, 3), dtype=np.float32)
                _jit_draw_circle_patch(canvas_float, wx1, wy1, wR, wt, wb, wg, wr, alpha_val,
                                       patch_float, bx1, by1, bx2, by2)
            elif st == "line":
                wx1 = int(sx1[bi]); wy1 = int(sy1[bi])
                wx2 = int(sx2[bi]); wy2 = int(sy2[bi])
                ht = wt // 2
                mn_x = wx1 if wx1 < wx2 else wx2
                mx_x = wx1 if wx1 > wx2 else wx2
                mn_y = wy1 if wy1 < wy2 else wy2
                mx_y = wy1 if wy1 > wy2 else wy2
                bx1 = mn_x - ht
                if bx1 < 0: bx1 = 0
                by1 = mn_y - ht
                if by1 < 0: by1 = 0
                bx2 = mx_x + ht + 1
                if bx2 > img_w: bx2 = img_w
                by2 = mx_y + ht + 1
                if by2 > img_h: by2 = img_h
                # strict crop: intersect with patch
                if _eval_mode == "crop":
                    bx1 = max(bx1, _patch_x1); by1 = max(by1, _patch_y1)
                    bx2 = min(bx2, _patch_x2); by2 = min(by2, _patch_y2)
                if bx2 <= bx1 or by2 <= by1:
                    continue
                cand_bounds = (bx1, by1, bx2, by2)
                patch_float = np.empty((by2 - by1, bx2 - bx1, 3), dtype=np.float32)
                _jit_draw_line_patch(canvas_float, wx1, wy1, wx2, wy2, wt, wb, wg, wr, alpha_val,
                                     patch_float, bx1, by1, bx2, by2)
            else:  # triangle
                wx1 = int(sx1[bi]); wy1 = int(sy1[bi])
                wx2 = int(sx2[bi]); wy2 = int(sy2[bi])
                wx3 = int(sx3[bi]); wy3 = int(sy3[bi])
                mn_x = wx1
                if wx2 < mn_x: mn_x = wx2
                if wx3 < mn_x: mn_x = wx3
                mx_x = wx1
                if wx2 > mx_x: mx_x = wx2
                if wx3 > mx_x: mx_x = wx3
                mn_y = wy1
                if wy2 < mn_y: mn_y = wy2
                if wy3 < mn_y: mn_y = wy3
                mx_y = wy1
                if wy2 > mx_y: mx_y = wy2
                if wy3 > mx_y: mx_y = wy3
                bx1 = mn_x - wt
                if bx1 < 0: bx1 = 0
                by1 = mn_y - wt
                if by1 < 0: by1 = 0
                bx2 = mx_x + wt + 1
                if bx2 > img_w: bx2 = img_w
                by2 = mx_y + wt + 1
                if by2 > img_h: by2 = img_h
                # strict crop: intersect with patch
                if _eval_mode == "crop":
                    bx1 = max(bx1, _patch_x1); by1 = max(by1, _patch_y1)
                    bx2 = min(bx2, _patch_x2); by2 = min(by2, _patch_y2)
                if bx2 <= bx1 or by2 <= by1:
                    continue
                cand_bounds = (bx1, by1, bx2, by2)
                patch_float = np.empty((by2 - by1, bx2 - bx1, 3), dtype=np.float32)
                _jit_draw_tri_patch(canvas_float, wx1, wy1, wx2, wy2, wx3, wy3, wt, wb, wg, wr, alpha_val,
                                    patch_float, bx1, by1, bx2, by2)
            # --- best-update ---
            if bl < best_gene_loss:
                best_gene_loss = bl
                best_patch_float = patch_float
                best_bounds = cand_bounds
                best_thickness = wt
                best_color = cand_color
                best_shape_type = st
                if st == "circle":
                    best_x, best_y = wx1, wy1
                    best_x1, best_y1 = wx1, wy1
                    best_x2, best_y2 = 0, 0
                    best_x3, best_y3 = 0, 0
                    best_R = wR
                    best_scale = wR
                elif st == "triangle":
                    best_x, best_y = wx1, wy1
                    best_x1, best_y1 = wx1, wy1
                    best_x2, best_y2 = wx2, wy2
                    best_x3, best_y3 = wx3, wy3
                    best_R = 0
                    best_scale = 0
                else:  # line
                    best_x, best_y = wx1, wy1
                    best_x1, best_y1 = wx1, wy1
                    best_x2, best_y2 = wx2, wy2
                    best_x3, best_y3 = 0, 0
                    best_R = 0
                    best_scale = 0

        stall = 0
        it = 0
        # Adaptive mutation: large early, small late
        # mut_scale goes from 1.0 (early) to 0.2 (late)
        if max_shapes and max_shapes > 0:
            _prog = shape_idx / max_shapes
            mut_scale = 1.0 - 0.8 * _prog
        else:
            mut_scale = 1.0
        # Batched hill climbing: _BATCH candidates per iter, budget = population_size*12 (tuned for speed)
        # adaptive ranges scaled by mut_scale
        _jitter_pos = max(3, int(min(img_w, img_h) * 0.2 * mut_scale))   # vertex position jitter
        _jitter_rad = max(2, int(min(img_w, img_h) * 0.15 * mut_scale))    # radius jitter
        _jitter_thk = max(1, int(10 * mut_scale))    # thickness jitter
        # --- 1/5 success-rule adaptive jitter (Rechenberg) ---
        # per-shape success tracking: adapt every _adapt_every batch iters
        # NOTE: max_evals is ~3-4 iters for POP=50 (400/128), so _adapt_every must be small
        _adapt_every = 2
        _adapt_window_success = 0
        _adapt_window_total = 0
        _adapt_factor_inc = 1.22
        _adapt_factor_dec = 0.82
        _jitter_pos_min, _jitter_pos_max = 3, int(min(img_w, img_h) * 0.4)
        _jitter_rad_min, _jitter_rad_max = 2, int(min(img_w, img_h) * 0.4)
        _jitter_thk_min, _jitter_thk_max = 1, 10
        # --- EMA random-step-size ---
        # use_ema: jitter each iter is mutated around EMA; EMA moves toward successful jitters
        if use_ema:
            _ema_pos = float(_jitter_pos)
            _ema_rad = float(_jitter_rad)
            _ema_thk = float(_jitter_thk)
            # ema_alpha and ema_tau come from run_ga args (ema_alpha=0.82, ema_tau=0.22)
            _ema_alpha_local = float(ema_alpha)
            _ema_tau_local = float(ema_tau)
        # if disabled, keep windows but flag prevents adaptation
        _best_st_int = _SHAPE_INT_MAP.get(best_shape_type, 0)
        while it < max_evals:
            if shape_timeout is not None and time.perf_counter() - shape_t0 > shape_timeout:
                break
            # --- EMA random-step-size: mutate jitter around EMA before each batch ---
            if use_ema:
                # random-step EMA: jitter = EMA * exp(N(0, tau)), clipped to [min,max]
                # use rng.normal for determinism per seed (rng is default_rng from run_ga)
                _jitter_pos = int(np.clip(_ema_pos * np.exp(rng.normal(0.0, _ema_tau_local)), _jitter_pos_min, _jitter_pos_max))
                _jitter_rad = int(np.clip(_ema_rad * np.exp(rng.normal(0.0, _ema_tau_local)), _jitter_rad_min, _jitter_rad_max))
                _jitter_thk = int(np.clip(_ema_thk * np.exp(rng.normal(0.0, _ema_tau_local)), _jitter_thk_min, _jitter_thk_max))
                if _jitter_thk < 1:
                    _jitter_thk = 1
                if _jitter_pos < _jitter_pos_min:
                    _jitter_pos = _jitter_pos_min
                if _jitter_rad < _jitter_rad_min:
                    _jitter_rad = _jitter_rad_min
            # --- JIT batch mutation ---
            _jit_mutate_batch(
                best_x1, best_y1, best_x2, best_y2, best_x3, best_y3,
                best_R, best_thickness, best_color[2], best_color[1], best_color[0],
                _best_st_int, img_w, img_h,
                _jitter_pos, _jitter_rad, _jitter_thk,
                _b_x1, _b_y1, _b_x2, _b_y2, _b_x3, _b_y3,
                _b_R, _b_thick, _b_b, _b_g, _b_r,
            )
            # strict crop: keep hill candidates inside patch (clip with extent)
            if _eval_mode == "crop":
                _b_x1[:] = np.clip(_b_x1, _patch_x1, _patch_x2-1)
                _b_y1[:] = np.clip(_b_y1, _patch_y1, _patch_y2-1)
                _b_x2[:] = np.clip(_b_x2, _patch_x1, _patch_x2-1)
                _b_y2[:] = np.clip(_b_y2, _patch_y1, _patch_y2-1)
                _b_x3[:] = np.clip(_b_x3, _patch_x1, _patch_x2-1)
                _b_y3[:] = np.clip(_b_y3, _patch_y1, _patch_y2-1)
                _b_R[:] = np.clip(_b_R, 3, min(_patch_w, _patch_h)*0.4)
                # for circles ensure fully inside patch (avoid truncated half-circles that look broken)
                if best_shape_type == "circle":
                    _b_x1[:] = np.clip(_b_x1, _patch_x1 + _b_R, _patch_x2 - 1 - _b_R)
                    _b_y1[:] = np.clip(_b_y1, _patch_y1 + _b_R, _patch_y2 - 1 - _b_R)
            if _eval_mode == "global":
                _s_x1[:] = np.clip(_b_x1 * _small_scale_w, 0, _small_w-1)
                _s_y1[:] = np.clip(_b_y1 * _small_scale_h, 0, _small_h-1)
                _s_x2[:] = np.clip(_b_x2 * _small_scale_w, 0, _small_w-1)
                _s_y2[:] = np.clip(_b_y2 * _small_scale_h, 0, _small_h-1)
                _s_x3[:] = np.clip(_b_x3 * _small_scale_w, 0, _small_w-1)
                _s_y3[:] = np.clip(_b_y3 * _small_scale_h, 0, _small_h-1)
                _s_R[:] = np.clip(_b_R * _small_scale_w, 1, min(_small_w,_small_h)*0.4)
                _s_thick[:] = np.clip(np.maximum(1, (_b_thick * _small_scale_w).astype(int)), 1, 10)
            else:
                _s_x1[:] = _b_x1 - _patch_x1
                _s_y1[:] = _b_y1 - _patch_y1
                _s_x2[:] = _b_x2 - _patch_x1
                _s_y2[:] = _b_y2 - _patch_y1
                _s_x3[:] = _b_x3 - _patch_x1
                _s_y3[:] = _b_y3 - _patch_y1
                _s_R[:] = _b_R
                _s_thick[:] = _b_thick
            # --- batch evaluate via single JIT call (scanline + partial) ---
            if _eval_mode == "global":
                if best_shape_type == "circle":
                    bi, bl = _jit_eval_circle_scanline_partial_batch(_small_canvas_float, _small_target_float,
                        _s_x1, _s_y1, _s_R, _s_thick, _b_b, _b_g, _b_r, alpha_val, _small_w, _small_h, _cur_total)
                elif best_shape_type == "line":
                    bi, bl = _jit_eval_line_partial_batch(_small_canvas_float, _small_target_float,
                        _s_x1, _s_y1, _s_x2, _s_y2, _s_thick, _b_b, _b_g, _b_r, alpha_val, _small_w, _small_h, _cur_total)
                else:
                    bi, bl = _jit_eval_triangle_scanline_partial_batch(_small_canvas_float, _small_target_float,
                        _s_x1, _s_y1, _s_x2, _s_y2, _s_x3, _s_y3, _s_thick, _b_b, _b_g, _b_r, alpha_val, _small_w, _small_h, _cur_total)
            else:
                if best_shape_type == "circle":
                    bi, bl = _jit_eval_circle_scanline_partial_batch(_patch_canvas_float, _patch_target_float,
                        _s_x1, _s_y1, _s_R, _s_thick, _b_b, _b_g, _b_r, alpha_val, _patch_w, _patch_h, _cur_total)
                elif best_shape_type == "line":
                    bi, bl = _jit_eval_line_partial_batch(_patch_canvas_float, _patch_target_float,
                        _s_x1, _s_y1, _s_x2, _s_y2, _s_thick, _b_b, _b_g, _b_r, alpha_val, _patch_w, _patch_h, _cur_total)
                else:
                    bi, bl = _jit_eval_triangle_scanline_partial_batch(_patch_canvas_float, _patch_target_float,
                        _s_x1, _s_y1, _s_x2, _s_y2, _s_x3, _s_y3, _s_thick, _b_b, _b_g, _b_r, alpha_val, _patch_w, _patch_h, _cur_total)
            it += _BATCH
            if bi < 0:
                stall += 1
                if stall > 15:
                    break
                continue
            # --- extract winner and generate patch ---
            # re-read winner params from arrays
            wt = int(_b_thick[bi]); wb = int(_b_b[bi]); wg = int(_b_g[bi]); wr = int(_b_r[bi])
            wx1 = int(_b_x1[bi]); wy1 = int(_b_y1[bi])
            wx2 = int(_b_x2[bi]); wy2 = int(_b_y2[bi])
            wx3 = int(_b_x3[bi]); wy3 = int(_b_y3[bi])
            wR = int(_b_R[bi])
            cand_color = (wr, wg, wb)
            # compute bbox for winner
            if best_shape_type == "circle":
                bx1 = wx1 - wR
                if bx1 < 0: bx1 = 0
                by1 = wy1 - wR
                if by1 < 0: by1 = 0
                bx2 = wx1 + wR + 1
                if bx2 > img_w: bx2 = img_w
                by2 = wy1 + wR + 1
                if by2 > img_h: by2 = img_h
                # strict crop: intersect with patch
                if _eval_mode == "crop":
                    bx1 = max(bx1, _patch_x1); by1 = max(by1, _patch_y1)
                    bx2 = min(bx2, _patch_x2); by2 = min(by2, _patch_y2)
                    if bx1 >= bx2 or by1 >= by2:
                        stall += 1
                        if stall > 15:
                            break
                        continue
                cand_bounds = (bx1, by1, bx2, by2)
                patch_float = np.empty((by2 - by1, bx2 - bx1, 3), dtype=np.float32)
                _jit_draw_circle_patch(canvas_float, wx1, wy1, wR, wt, wb, wg, wr, alpha_val,
                                       patch_float, bx1, by1, bx2, by2)
            elif best_shape_type == "line":
                ht = wt // 2
                mn_x = wx1 if wx1 < wx2 else wx2
                mx_x = wx1 if wx1 > wx2 else wx2
                mn_y = wy1 if wy1 < wy2 else wy2
                mx_y = wy1 if wy1 > wy2 else wy2
                bx1 = mn_x - ht
                if bx1 < 0: bx1 = 0
                by1 = mn_y - ht
                if by1 < 0: by1 = 0
                bx2 = mx_x + ht + 1
                if bx2 > img_w: bx2 = img_w
                by2 = mx_y + ht + 1
                if by2 > img_h: by2 = img_h
                # strict crop: intersect with patch
                if _eval_mode == "crop":
                    bx1 = max(bx1, _patch_x1); by1 = max(by1, _patch_y1)
                    bx2 = min(bx2, _patch_x2); by2 = min(by2, _patch_y2)
                    if bx1 >= bx2 or by1 >= by2:
                        stall += 1
                        if stall > 15:
                            break
                        continue
                cand_bounds = (bx1, by1, bx2, by2)
                patch_float = np.empty((by2 - by1, bx2 - bx1, 3), dtype=np.float32)
                _jit_draw_line_patch(canvas_float, wx1, wy1, wx2, wy2, wt, wb, wg, wr, alpha_val,
                                     patch_float, bx1, by1, bx2, by2)
            else:
                mn_x = wx1
                if wx2 < mn_x: mn_x = wx2
                if wx3 < mn_x: mn_x = wx3
                mx_x = wx1
                if wx2 > mx_x: mx_x = wx2
                if wx3 > mx_x: mx_x = wx3
                mn_y = wy1
                if wy2 < mn_y: mn_y = wy2
                if wy3 < mn_y: mn_y = wy3
                mx_y = wy1
                if wy2 > mx_y: mx_y = wy2
                if wy3 > mx_y: mx_y = wy3
                bx1 = mn_x - wt
                if bx1 < 0: bx1 = 0
                by1 = mn_y - wt
                if by1 < 0: by1 = 0
                bx2 = mx_x + wt + 1
                if bx2 > img_w: bx2 = img_w
                by2 = mx_y + wt + 1
                if by2 > img_h: by2 = img_h
                # strict crop: intersect with patch
                if _eval_mode == "crop":
                    bx1 = max(bx1, _patch_x1); by1 = max(by1, _patch_y1)
                    bx2 = min(bx2, _patch_x2); by2 = min(by2, _patch_y2)
                    if bx1 >= bx2 or by1 >= by2:
                        stall += 1
                        if stall > 15:
                            break
                        continue
                cand_bounds = (bx1, by1, bx2, by2)
                patch_float = np.empty((by2 - by1, bx2 - bx1, 3), dtype=np.float32)
                _jit_draw_tri_patch(canvas_float, wx1, wy1, wx2, wy2, wx3, wy3, wt, wb, wg, wr, alpha_val,
                                    patch_float, bx1, by1, bx2, by2)
            _is_success = (bl < best_gene_loss - _ACCEPT_EPS)
            if _is_success:
                best_gene_loss = bl
                best_patch_float = patch_float
                best_bounds = cand_bounds
                best_thickness = wt
                best_color = cand_color
                best_shape_type = best_shape_type
                best_x1, best_y1 = wx1, wy1
                best_x2, best_y2 = wx2, wy2
                best_x3, best_y3 = wx3, wy3
                best_R = wR
                if best_shape_type == "circle":
                    best_x, best_y = wx1, wy1
                    best_scale = wR
                else:
                    best_x, best_y = wx1, wy1
                    best_scale = 0
                stall = 0
            else:
                stall += 1
                if stall > 15:
                    break
            # --- EMA update on successful step sizes ---
            if use_ema and _is_success:
                # EMA moves toward the jitter that produced the success
                _ema_pos = _ema_alpha_local * _ema_pos + (1.0 - _ema_alpha_local) * float(_jitter_pos)
                _ema_rad = _ema_alpha_local * _ema_rad + (1.0 - _ema_alpha_local) * float(_jitter_rad)
                _ema_thk = _ema_alpha_local * _ema_thk + (1.0 - _ema_alpha_local) * float(_jitter_thk)
                # keep EMA within bounds
                _ema_pos = float(np.clip(_ema_pos, _jitter_pos_min, _jitter_pos_max))
                _ema_rad = float(np.clip(_ema_rad, _jitter_rad_min, _jitter_rad_max))
                _ema_thk = float(np.clip(_ema_thk, _jitter_thk_min, _jitter_thk_max))
            # --- 1/5 rule adaptation (Rechenberg) ---
            if use_one_fifth:
                _adapt_window_total += 1
                if _is_success:
                    _adapt_window_success += 1
                if _adapt_window_total >= _adapt_every:
                    _succ_rate = _adapt_window_success / _adapt_window_total if _adapt_window_total else 0.0
                    if _succ_rate > 0.20:
                        # too many successes -> increase step to explore faster
                        if use_ema:
                            # scale EMA itself when EMA is active
                            _ema_pos = float(np.clip(_ema_pos * _adapt_factor_inc, _jitter_pos_min, _jitter_pos_max))
                            _ema_rad = float(np.clip(_ema_rad * _adapt_factor_inc, _jitter_rad_min, _jitter_rad_max))
                            _ema_thk = float(np.clip(_ema_thk * _adapt_factor_inc, _jitter_thk_min, _jitter_thk_max))
                        else:
                            _jitter_pos = int(min(_jitter_pos_max, max(_jitter_pos_min, _jitter_pos * _adapt_factor_inc)))
                            _jitter_rad = int(min(_jitter_rad_max, max(_jitter_rad_min, _jitter_rad * _adapt_factor_inc)))
                            _jitter_thk = int(min(_jitter_thk_max, max(_jitter_thk_min, _jitter_thk * _adapt_factor_inc)))
                            if _jitter_thk < 1: _jitter_thk = 1
                    elif _succ_rate < 0.20:
                        # too few successes -> decrease step to exploit finer
                        if use_ema:
                            _ema_pos = float(np.clip(_ema_pos * _adapt_factor_dec, _jitter_pos_min, _jitter_pos_max))
                            _ema_rad = float(np.clip(_ema_rad * _adapt_factor_dec, _jitter_rad_min, _jitter_rad_max))
                            _ema_thk = float(np.clip(_ema_thk * _adapt_factor_dec, _jitter_thk_min, _jitter_thk_max))
                        else:
                            _jitter_pos = int(min(_jitter_pos_max, max(_jitter_pos_min, _jitter_pos * _adapt_factor_dec)))
                            _jitter_rad = int(min(_jitter_rad_max, max(_jitter_rad_min, _jitter_rad * _adapt_factor_dec)))
                            _jitter_thk = int(min(_jitter_thk_max, max(_jitter_thk_min, _jitter_thk * _adapt_factor_dec)))
                            if _jitter_thk < 1: _jitter_thk = 1
                    # ensure ints and reset window (for non-EMA path)
                    if not use_ema:
                        _jitter_pos = int(_jitter_pos); _jitter_rad = int(_jitter_rad); _jitter_thk = int(_jitter_thk)
                    _adapt_window_success = 0
                    _adapt_window_total = 0

        if best_bounds is None:
            shape_idx += 1
            yield full_bg, current_loss, shape_idx, list(fixed_shapes), time.perf_counter() - shape_t0
            continue

        x1, y1, x2, y2 = best_bounds
        # strict crop: intersect with patch so output never spills outside patch
        if _eval_mode == "crop":
            nx1 = max(x1, _patch_x1); ny1 = max(y1, _patch_y1)
            nx2 = min(x2, _patch_x2); ny2 = min(y2, _patch_y2)
            if nx1 >= nx2 or ny1 >= ny2:
                shape_idx += 1
                yield full_bg, current_loss, shape_idx, list(fixed_shapes), time.perf_counter() - shape_t0
                continue
            if nx1 != x1 or ny1 != y1 or nx2 != x2 or ny2 != y2:
                # slice patch to intersected region
                best_patch_float = best_patch_float[ny1 - y1:ny1 - y1 + (ny2 - ny1), nx1 - x1:nx1 - x1 + (nx2 - nx1)]
                x1, y1, x2, y2 = nx1, ny1, nx2, ny2
        old_region_loss = np.sum(np.abs(canvas_float[y1:y2, x1:x2] - target_float[y1:y2, x1:x2]))
        new_region_loss = np.sum(np.abs(best_patch_float - target_float[y1:y2, x1:x2]))
        # monotonic gate — only commit if true global improvement
        if new_region_loss >= old_region_loss - 1e-8:
            shape_idx += 1
            if verbose:
                print(f"[hc] slot {shape_idx}: SKIP (no true improvement, new {new_region_loss:.1f} >= old {old_region_loss:.1f})", flush=True)
            yield full_bg, current_loss, shape_idx, list(fixed_shapes), time.perf_counter() - shape_t0
            continue
        
        canvas_cv[y1:y2, x1:x2] = best_patch_float.astype(np.uint8)
        canvas_float[y1:y2, x1:x2] = best_patch_float
        # keep small canvases in sync if next is global (cheap resize)
        if _eval_mode == "global":
            _small_canvas_cv = cv2.resize(canvas_cv, (_small_w, _small_h), interpolation=cv2.INTER_AREA)
            _small_canvas_float = _small_canvas_cv.astype(np.float32)
        # patch canvases are re-copied per shape, no need to sync

        loss_diff = old_region_loss - new_region_loss
        current_loss -= loss_diff / target_float.size

        r, g, b = best_color
        shape_data = {
            "r": int(r), "g": int(g), "b": int(b), "a": alpha_fixed,
            "shape_type": best_shape_type,
            "x": best_x, "y": best_y,
            "x1": int(best_x1), "y1": int(best_y1),
            "x2": int(best_x2), "y2": int(best_y2),
            "x3": int(best_x3), "y3": int(best_y3),
            "R": int(best_R),
            "thickness": int(best_thickness),
            "loss": float(current_loss),
        }
        fixed_shapes.append(shape_data)

        bg_rgb = cv2.cvtColor(canvas_cv, cv2.COLOR_BGR2RGB)
        full_bg = Image.fromarray(bg_rgb).convert("RGBA")

        shape_idx += 1
        if verbose:
            print(f"[hc] shape {shape_idx}: committed '{best_shape_type}' loss={current_loss:.5f} t={time.perf_counter() - shape_t0:.2f}s", flush=True)
        yield full_bg, float(current_loss), shape_idx, list(fixed_shapes), time.perf_counter() - shape_t0
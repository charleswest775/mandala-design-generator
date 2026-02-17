"""
Mandala & Sacred Geometry CNC Design Generator
All cut-out patterns guarantee material connectivity — no floating pieces.
Black = cut paths / material removed. White = material remains.
All dimensions in inches. Exports to DXF for plasma/laser CNC.
"""

import math
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import ezdxf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon
import numpy as np

PHI = (1 + math.sqrt(5)) / 2

PATTERN_TYPES = [
    "Petal Mandala",
    "Sacred Geometry",
    "Geometric Lattice",
    "Hybrid",
]


# ══════════════════════════════════════════════════════════════
# Geometry primitives
# ══════════════════════════════════════════════════════════════

def _circle_pts(cx, cy, r, n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=True)
    return list(zip((cx + r * np.cos(t)).tolist(),
                    (cy + r * np.sin(t)).tolist()))


def _polar_pt(r, a):
    return (r * math.cos(a), r * math.sin(a))


def _line_seg(x1, y1, x2, y2):
    return [(x1, y1), (x2, y2)]


def _arc_pts(cx, cy, r, a_start, a_end, n=60):
    t = np.linspace(a_start, a_end, n)
    return list(zip((cx + r * np.cos(t)).tolist(),
                    (cy + r * np.sin(t)).tolist()))


def _bridged_circle(cx, cy, r, n_bridges=4, bridge_gap=0.06, rot=0, n=200):
    """Circle with small gaps (bridges) so enclosed material stays connected.
    Returns list of arc segments (open polylines)."""
    arcs = []
    for i in range(n_bridges):
        a_start = rot + 2 * math.pi * i / n_bridges + bridge_gap / 2
        a_end = rot + 2 * math.pi * (i + 1) / n_bridges - bridge_gap / 2
        arcs.append(_arc_pts(cx, cy, r, a_start, a_end, n // n_bridges))
    return arcs


def _polygon_pts(cx, cy, r, n_sides, rotation=0):
    pts = []
    for i in range(n_sides + 1):
        a = rotation + 2 * math.pi * i / n_sides - math.pi / 2
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _hex_pts(cx, cy, r):
    """Regular hexagon."""
    return _polygon_pts(cx, cy, r, 6, math.pi / 6)


def _pentagon_pts(cx, cy, r):
    """Regular pentagon."""
    return _polygon_pts(cx, cy, r, 5)


def _triangle_pts(cx, cy, r):
    """Regular triangle."""
    return _polygon_pts(cx, cy, r, 3)


def _square_pts(cx, cy, r):
    """Regular square (diamond orientation)."""
    return _polygon_pts(cx, cy, r, 4)


def _octagon_pts(cx, cy, r):
    """Regular octagon."""
    return _polygon_pts(cx, cy, r, 8)


def _star_pts(cx, cy, r, n_points=5):
    """Star shape with alternating outer/inner vertices."""
    pts = []
    r_inner = r * 0.4
    for i in range(n_points * 2 + 1):
        a = -math.pi / 2 + 2 * math.pi * i / (n_points * 2)
        rv = r if i % 2 == 0 else r_inner
        pts.append((cx + rv * math.cos(a), cy + rv * math.sin(a)))
    return pts



# ══════════════════════════════════════════════════════════════
# Petal gap styles — 8 varieties
# ══════════════════════════════════════════════════════════════

def _gap_pointed(r_in, r_out, ac, ah, n=40):
    pts = []
    for i in range(n + 1):
        t = i / n
        r = r_in + t * (r_out - r_in)
        w = ah * math.sin(math.pi * t) ** 0.8
        pts.append(_polar_pt(r, ac + w))
    for i in range(n, -1, -1):
        t = i / n
        r = r_in + t * (r_out - r_in)
        w = ah * math.sin(math.pi * t) ** 0.8
        pts.append(_polar_pt(r, ac - w))
    pts.append(pts[0])
    return pts


def _gap_rounded(r_in, r_out, ac, ah, n=40):
    pts = []
    for i in range(n + 1):
        t = i / n
        r = r_in + t * (r_out - r_in)
        w = ah * (math.sin(math.pi * t * 0.85) ** 1.2)
        pts.append(_polar_pt(r, ac + w))
    for i in range(n, -1, -1):
        t = i / n
        r = r_in + t * (r_out - r_in)
        w = ah * (math.sin(math.pi * t * 0.85) ** 1.2)
        pts.append(_polar_pt(r, ac - w))
    pts.append(pts[0])
    return pts


def _gap_ogee(r_in, r_out, ac, ah, n=50):
    pts = []
    for i in range(n + 1):
        t = i / n
        r = r_in + t * (r_out - r_in)
        s = 0.5 * (1 - math.cos(2 * math.pi * t)) * math.sin(math.pi * t) ** 0.5
        w = ah * s
        pts.append(_polar_pt(r, ac + w))
    for i in range(n, -1, -1):
        t = i / n
        r = r_in + t * (r_out - r_in)
        s = 0.5 * (1 - math.cos(2 * math.pi * t)) * math.sin(math.pi * t) ** 0.5
        w = ah * s
        pts.append(_polar_pt(r, ac - w))
    pts.append(pts[0])
    return pts


def _gap_leaf(r_in, r_out, ac, ah, n=40):
    pts = []
    for i in range(n + 1):
        t = i / n
        r = r_in + t * (r_out - r_in)
        w = ah * math.sin(math.pi * t) ** PHI * (1 - 0.3 * t)
        pts.append(_polar_pt(r, ac + w))
    for i in range(n, -1, -1):
        t = i / n
        r = r_in + t * (r_out - r_in)
        w = ah * math.sin(math.pi * t) ** PHI * (1 - 0.3 * t)
        pts.append(_polar_pt(r, ac - w))
    pts.append(pts[0])
    return pts



def _gap_arrow(r_in, r_out, ac, ah, n=40):
    """Arrow/chevron gap — wide at inner, sharp point at outer."""
    pts = []
    for i in range(n + 1):
        t = i / n
        r = r_in + t * (r_out - r_in)
        w = ah * (1 - t) ** 0.7 * math.sin(math.pi * t) ** 0.3
        pts.append(_polar_pt(r, ac + w))
    for i in range(n, -1, -1):
        t = i / n
        r = r_in + t * (r_out - r_in)
        w = ah * (1 - t) ** 0.7 * math.sin(math.pi * t) ** 0.3
        pts.append(_polar_pt(r, ac - w))
    pts.append(pts[0])
    return pts



GAP_STYLES = {
    "pointed": _gap_pointed, "rounded": _gap_rounded, "ogee": _gap_ogee,
    "leaf": _gap_leaf, "arrow": _gap_arrow,
}
ALL_GAP_STYLES = list(GAP_STYLES.keys())

# Shape type options for per-layer overrides
GAP_SHAPE_NAMES = ["Pointed", "Rounded", "Ogee", "Leaf", "Arrow"]
HOLE_SHAPE_NAMES = ["Circle", "Hexagon", "Pentagon", "Triangle", "Diamond",
                    "Teardrop", "Star", "Square", "Octagon"]
LAYER_SHAPE_TYPES = GAP_SHAPE_NAMES + HOLE_SHAPE_NAMES

# Border shape options for the Outer_Border layer
BORDER_SHAPE_NAMES = ["Circle", "Square", "Triangle", "Hexagon", "Octagon", "Pentagon"]
BORDER_SHAPE_TYPES = BORDER_SHAPE_NAMES

# Map UI shape names to gap style function keys
GAP_SHAPE_MAP = {
    "Pointed": "pointed", "Rounded": "rounded", "Ogee": "ogee", "Leaf": "leaf",
    "Arrow": "arrow",
}

# Layers that should not appear in the layer editor
_NON_EDITABLE_LAYERS = set()  # All layers are editable


def _make_border_shape(size, shape_type, rotation=0):
    """Generate an outer border shape at the given size."""
    if shape_type == "Circle":
        return [_circle_pts(0, 0, size)]
    elif shape_type == "Square":
        return [_polygon_pts(0, 0, size, 4, rotation)]
    elif shape_type == "Triangle":
        return [_polygon_pts(0, 0, size, 3, rotation)]
    elif shape_type == "Hexagon":
        return [_polygon_pts(0, 0, size, 6, rotation)]
    elif shape_type == "Octagon":
        return [_polygon_pts(0, 0, size, 8, rotation)]
    elif shape_type == "Pentagon":
        return [_polygon_pts(0, 0, size, 5, rotation)]
    return None


def _make_hole_shape(cx, cy, hole_size, shape_type, facing_angle=0):
    """Generate a single hole cutout shape for lattice-type layers."""
    if shape_type == "Circle":
        return _circle_pts(cx, cy, hole_size * 0.45, 32)
    elif shape_type == "Hexagon":
        return _hex_pts(cx, cy, hole_size * 0.5)
    elif shape_type == "Pentagon":
        return _pentagon_pts(cx, cy, hole_size * 0.5)
    elif shape_type == "Triangle":
        return _triangle_pts(cx, cy, hole_size * 0.5)
    elif shape_type == "Square":
        return _square_pts(cx, cy, hole_size * 0.5)
    elif shape_type == "Octagon":
        return _octagon_pts(cx, cy, hole_size * 0.5)
    elif shape_type == "Star":
        return _star_pts(cx, cy, hole_size * 0.5, 5)
    elif shape_type == "Diamond":
        s = hole_size * 0.5
        pts = [(cx, cy + s), (cx + s * 0.6, cy),
               (cx, cy - s), (cx - s * 0.6, cy), (cx, cy + s)]
        ca, sa = math.cos(facing_angle), math.sin(facing_angle)
        return [(cx + (x - cx) * ca - (y - cy) * sa,
                 cy + (x - cx) * sa + (y - cy) * ca) for x, y in pts]
    elif shape_type == "Teardrop":
        s = hole_size * 0.5
        td = []
        for k in range(41):
            t = 2 * math.pi * k / 40
            rx = s * 0.8 * math.cos(t) * (1 + 0.3 * math.cos(t))
            ry = s * 0.5 * math.sin(t)
            td.append((rx, ry))
        td.append(td[0])
        ca, sa = math.cos(facing_angle), math.sin(facing_angle)
        return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in td]
    # Fallback: circle
    return _circle_pts(cx, cy, hole_size * 0.45, 32)


def _scale_shape_pts(pts, scale):
    """Scale shape points around their centroid."""
    if scale == 1.0 or not pts:
        return pts
    n = len(pts)
    # Exclude closing duplicate point for centroid calc
    calc_pts = pts[:-1] if (n > 2 and pts[0] == pts[-1]) else pts
    cx = sum(p[0] for p in calc_pts) / len(calc_pts)
    cy = sum(p[1] for p in calc_pts) / len(calc_pts)
    return [(cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in pts]


def _ring_gaps(r_in, r_out, n_petals, rotation, gap_frac, style_name):
    """Generate cutout gaps between petals. Each gap is a closed polyline.
    Material connectivity: gaps don't touch ring edges, leaving solid bands."""
    gap_fn = GAP_STYLES.get(style_name, _gap_pointed)
    # Inset slightly from ring edges to guarantee connecting material bands
    margin = (r_out - r_in) * 0.04
    r_in_safe = r_in + margin
    r_out_safe = r_out - margin
    gaps = []
    sector = 2 * math.pi / n_petals
    a_half = sector * gap_frac / 2
    for i in range(n_petals):
        a_center = rotation + sector * (i + 0.5)
        gaps.append(gap_fn(r_in_safe, r_out_safe, a_center, a_half))
    return gaps


# ══════════════════════════════════════════════════════════════
# Pattern 1: PETAL MANDALA (connectivity: ring bands between rings)
# ══════════════════════════════════════════════════════════════

def _gen_petal_mandala(size, petals, rot, complexity, rng):
    layers = []
    styles = ALL_GAP_STYLES.copy()
    rng.shuffle(styles)

    # Center hole (small, doesn't disconnect anything)
    layers.append(("Center", [_circle_pts(0, 0, size * 0.025)]))

    # Center flower petal gaps
    if petals >= 3:
        gaps = _ring_gaps(size * 0.04, size * 0.10, petals, rot, 0.40, "rounded")
        layers.append(("Center_Flower", gaps))

    # Petal ring definitions: (inner_frac, outer_frac, petal_mult, gap_frac, offset)
    ring_defs = []
    if complexity >= 1:
        ring_defs.append((0.11, 0.26, 1, 0.42, 0))
    if complexity >= 2:
        ring_defs.append((0.27, 0.44, 1, 0.38, 0.5))
    if complexity >= 3:
        ring_defs.append((0.45, 0.64, 2, 0.40, 0))
    if complexity >= 4:
        ring_defs.append((0.65, 0.82, 2, 0.35, 0.5))
    if complexity >= 5:
        ring_defs.append((0.83, 0.93, 3, 0.38, 0))

    for idx, (ri, ro, pm, gf, off) in enumerate(ring_defs):
        st = styles[idx % len(styles)]
        n = petals * pm
        offset_a = rot + (2 * math.pi / n * off)
        gaps = _ring_gaps(size * ri, size * ro, n, offset_a, gf, st)
        layers.append((f"Petal_Ring_{idx + 1}", gaps))

    # NO separator arcs — they would cut through connecting ring bands
    # The solid bands between rings are what hold the piece together

    # Outer border
    layers.append(("Outer_Border", [_circle_pts(0, 0, size)]))

    return layers, "filled"


# ══════════════════════════════════════════════════════════════
# Pattern 2: SACRED GEOMETRY (connectivity: bridged circles, open lines)
# ══════════════════════════════════════════════════════════════

def _flower_of_life_bridged(cx, cy, r, n_rings, rotation=0, n_bridges=6):
    """Flower of Life with bridges in every circle to prevent isolation."""
    all_arcs = []
    placed = [(cx, cy)]
    all_arcs.extend(_bridged_circle(cx, cy, r, n_bridges, 0.08, rotation))

    for ring in range(1, n_rings + 1):
        if ring == 1:
            for i in range(6):
                a = rotation + i * math.pi / 3
                px = cx + r * math.cos(a)
                py = cy + r * math.sin(a)
                all_arcs.extend(_bridged_circle(px, py, r, n_bridges, 0.08, rotation))
                placed.append((px, py))
        else:
            new_placed = []
            for px, py in placed:
                for i in range(6):
                    a = rotation + i * math.pi / 3
                    nx = px + r * math.cos(a)
                    ny = py + r * math.sin(a)
                    if math.hypot(nx - cx, ny - cy) > r * ring + 0.01:
                        continue
                    dup = any(math.hypot(nx - ex, ny - ey) < r * 0.1
                              for ex, ey in placed + new_placed)
                    if not dup:
                        new_placed.append((nx, ny))
                        all_arcs.extend(_bridged_circle(nx, ny, r, n_bridges, 0.08, rotation))
            placed.extend(new_placed)

    return all_arcs, placed


def _gen_sacred_geometry(size, petals, rot, complexity, rng):
    layers = []

    # Concentric circles — bridged so material stays connected
    frame_arcs = []
    r = size * 0.08
    while r < size * 0.99:
        frame_arcs.extend(_bridged_circle(0, 0, r, petals, 0.05, rot))
        r *= PHI
    layers.append(("Circles", frame_arcs))

    # Flower of Life — bridged circles
    fol_r = size * 0.22
    n_fol = min(complexity, 3)
    fol_arcs, fol_centers = _flower_of_life_bridged(0, 0, fol_r, n_fol, rot)
    layers.append(("Flower_of_Life", fol_arcs))

    # Metatron's Cube — open line segments (always safe for connectivity)
    if complexity >= 2 and len(fol_centers) > 1:
        metro = []
        for i in range(len(fol_centers)):
            for j in range(i + 1, len(fol_centers)):
                x1, y1 = fol_centers[i]
                x2, y2 = fol_centers[j]
                if math.hypot(x2 - x1, y2 - y1) <= fol_r * 2.1:
                    metro.append(_line_seg(x1, y1, x2, y2))
        layers.append(("Metatrons_Cube", metro))

    # Star polygon — open line segments (safe)
    if complexity >= 2:
        star = []
        star_r = size * 0.55
        skip = max(2, petals // 3)
        verts = []
        for i in range(petals):
            a = rot + 2 * math.pi * i / petals - math.pi / 2
            verts.append(_polar_pt(star_r, a))
        for i in range(petals):
            j = (i + skip) % petals
            star.append([verts[i], verts[j]])
        # Polygon outline — open segments between vertices (not closed)
        for i in range(petals):
            j = (i + 1) % petals
            star.append([verts[i], verts[j]])
        layers.append(("Star_Polygon", star))

    # Triangles — open line segments
    if complexity >= 3:
        tri = []
        for tr in [0, math.pi / 3]:
            v = [_polar_pt(size * 0.45, rot + tr + 2 * math.pi * i / 3 - math.pi / 2)
                 for i in range(3)]
            for i in range(3):
                tri.append([v[i], v[(i + 1) % 3]])
        if complexity >= 4:
            tr2 = size * 0.45 * PHI
            if tr2 < size * 0.9:
                for tr in [0, math.pi / 3]:
                    v = [_polar_pt(tr2, rot + tr + 2 * math.pi * i / 3 - math.pi / 2)
                         for i in range(3)]
                    for i in range(3):
                        tri.append([v[i], v[(i + 1) % 3]])
        layers.append(("Triangles", tri))

    # Radial rays — open lines (safe)
    rays = []
    n_rays = petals * 2
    for i in range(n_rays):
        a = rot + 2 * math.pi * i / n_rays
        rays.append(_line_seg(size * 0.05 * math.cos(a), size * 0.05 * math.sin(a),
                              size * 0.95 * math.cos(a), size * 0.95 * math.sin(a)))
    layers.append(("Radial_Lines", rays))

    # Dot holes — small circles (fall out, but tiny so acceptable as drill holes)
    if complexity >= 2:
        dots = []
        dot_r = size * 0.012
        for cx, cy in fol_centers:
            dots.append(_circle_pts(cx, cy, dot_r, 24))
        r = size * 0.08
        while r < size * 0.95:
            for i in range(n_rays):
                a = rot + 2 * math.pi * i / n_rays
                dots.append(_circle_pts(r * math.cos(a), r * math.sin(a), dot_r, 24))
            r *= PHI
        layers.append(("Dots", dots))

    # Phi spirals — open curves (safe)
    if complexity >= 4:
        spirals = []
        t = np.linspace(0, 6 * np.pi, 2000)
        for k in range(min(petals, 6)):
            a_off = rot + 2 * math.pi * k / min(petals, 6)
            r_sp = size * 0.02 * np.exp(t / (2 * np.pi) * math.log(PHI))
            mask = r_sp <= size * 0.9
            sx = (r_sp * np.cos(t + a_off)).tolist()
            sy = (r_sp * np.sin(t + a_off)).tolist()
            seg = []
            for i in range(len(mask)):
                if mask[i]:
                    seg.append((sx[i], sy[i]))
                else:
                    if len(seg) >= 2:
                        spirals.append(seg)
                    seg = []
            if len(seg) >= 2:
                spirals.append(seg)
        layers.append(("Phi_Spirals", spirals))

    # Outer boundary — bridged (NOT fully closed, so outer material stays connected)
    layers.append(("Outer_Border",
                    _bridged_circle(0, 0, size, petals, 0.03, rot)))

    return layers, "lines"


# ══════════════════════════════════════════════════════════════
# Pattern 3: GEOMETRIC LATTICE (connectivity: isolated cutout holes)
# Each cutout is a small shape surrounded by material — inherently safe.
# ══════════════════════════════════════════════════════════════

def _gen_geometric_lattice(size, petals, rot, complexity, rng):
    layers = []
    shape_types = ["circle", "hex", "diamond", "teardrop"]
    rng.shuffle(shape_types)

    # Build concentric rings of cutout holes at phi radii
    r = size * 0.06
    ring_idx = 0
    while r < size * 0.90:
        shapes = []
        n_holes = petals * (ring_idx + 1)
        # Hole size scales with ring radius but stays small enough for bridges
        max_hole = 2 * math.pi * r / n_holes * 0.35
        hole_size = min(max_hole, size * 0.04 * (1 + ring_idx * 0.2))
        shape_type = shape_types[ring_idx % len(shape_types)]

        for j in range(n_holes):
            a = rot + 2 * math.pi * j / n_holes
            cx = r * math.cos(a)
            cy = r * math.sin(a)

            if shape_type == "circle":
                shapes.append(_circle_pts(cx, cy, hole_size * 0.45, 32))
            elif shape_type == "hex":
                shapes.append(_hex_pts(cx, cy, hole_size * 0.5))
            elif shape_type == "diamond":
                s = hole_size * 0.5
                pts = [(cx, cy + s), (cx + s * 0.6, cy),
                       (cx, cy - s), (cx - s * 0.6, cy), (cx, cy + s)]
                # Rotate to face outward
                ca, sa = math.cos(a), math.sin(a)
                pts = [(cx + (x - cx) * ca - (y - cy) * sa,
                        cy + (x - cx) * sa + (y - cy) * ca)
                       for x, y in pts]
                shapes.append(pts)
            elif shape_type == "teardrop":
                s = hole_size * 0.5
                td = []
                for k in range(41):
                    t = 2 * math.pi * k / 40
                    rx = s * 0.8 * math.cos(t) * (1 + 0.3 * math.cos(t))
                    ry = s * 0.5 * math.sin(t)
                    td.append((rx, ry))
                td.append(td[0])
                ca, sa = math.cos(a), math.sin(a)
                shapes.append([(cx + x * ca - y * sa, cy + x * sa + y * ca)
                                for x, y in td])

        layers.append((f"Lattice_Ring_{ring_idx + 1}", shapes))
        r *= PHI
        ring_idx += 1

    # Center hole
    layers.append(("Center", [_circle_pts(0, 0, size * 0.025)]))

    # Radial slot cutouts between rings (small elongated holes)
    if complexity >= 3:
        slots = []
        r = size * 0.06
        radii = []
        while r < size * 0.88:
            radii.append(r)
            r *= PHI
        for ri in range(len(radii) - 1):
            r_mid = (radii[ri] + radii[ri + 1]) / 2
            slot_len = (radii[ri + 1] - radii[ri]) * 0.45
            slot_w = slot_len * 0.15
            n_slots = petals * (ri + 1)
            for j in range(n_slots):
                a = rot + 2 * math.pi * (j + 0.5) / n_slots
                cx = r_mid * math.cos(a)
                cy = r_mid * math.sin(a)
                # Elongated hole aligned radially
                pts = []
                for k in range(21):
                    t = k / 20
                    x = (t - 0.5) * slot_len
                    y = slot_w * 0.5 * math.sin(math.pi * t)
                    pts.append((x, y))
                for k in range(20, -1, -1):
                    t = k / 20
                    x = (t - 0.5) * slot_len
                    y = -slot_w * 0.5 * math.sin(math.pi * t)
                    pts.append((x, y))
                pts.append(pts[0])
                ca, sa = math.cos(a), math.sin(a)
                slots.append([(cx + x * ca - y * sa, cy + x * sa + y * ca)
                               for x, y in pts])
        layers.append(("Radial_Slots", slots))

    # Outer border
    layers.append(("Outer_Border", [_circle_pts(0, 0, size)]))

    return layers, "filled"



# ══════════════════════════════════════════════════════════════
# Pattern 6: HYBRID (petal core + sacred geometry overlay)
# ══════════════════════════════════════════════════════════════

def _gen_hybrid(size, petals, rot, complexity, rng):
    layers = []
    inner_size = size * 0.55
    styles = ALL_GAP_STYLES.copy()
    rng.shuffle(styles)

    # Center
    layers.append(("Center", [_circle_pts(0, 0, inner_size * 0.04)]))
    if petals >= 3:
        gaps = _ring_gaps(inner_size * 0.06, inner_size * 0.15,
                          petals, rot, 0.40, "rounded")
        layers.append(("Center_Flower", gaps))

    # Inner petal rings
    inner_rings = [
        (0.16, 0.38, 1, 0.42, 0),
        (0.40, 0.65, 1, 0.38, 0.5),
    ]
    if complexity >= 4:
        inner_rings.append((0.67, 0.92, 2, 0.36, 0))

    for idx, (ri, ro, pm, gf, off) in enumerate(inner_rings):
        st = styles[idx % len(styles)]
        n = petals * pm
        offset_a = rot + (2 * math.pi / n * off)
        gaps = _ring_gaps(inner_size * ri, inner_size * ro, n, offset_a, gf, st)
        layers.append((f"Inner_Petals_{idx + 1}", gaps))

    # Outer zone: bridged concentric circles
    outer_arcs = []
    r = inner_size
    while r < size * 0.98:
        outer_arcs.extend(_bridged_circle(0, 0, r, petals, 0.05, rot))
        r *= PHI ** 0.5
    layers.append(("Outer_Circles", outer_arcs))

    # Star polygon — open segments
    if complexity >= 2:
        star = []
        star_r = size * 0.78
        skip = max(2, petals // 3)
        verts = [_polar_pt(star_r, rot + 2 * math.pi * i / petals - math.pi / 2)
                 for i in range(petals)]
        for i in range(petals):
            star.append([verts[i], verts[(i + skip) % petals]])
            star.append([verts[i], verts[(i + 1) % petals]])
        layers.append(("Star_Polygon", star))

    # Radial lines outer zone
    rays = []
    n_rays = petals * 2
    for i in range(n_rays):
        a = rot + 2 * math.pi * i / n_rays
        rays.append(_line_seg(inner_size * 0.9 * math.cos(a),
                              inner_size * 0.9 * math.sin(a),
                              size * 0.95 * math.cos(a),
                              size * 0.95 * math.sin(a)))
    layers.append(("Radial_Lines", rays))

    # Dots
    if complexity >= 3:
        dots = []
        dot_r = size * 0.012
        r = inner_size
        while r < size * 0.92:
            for i in range(n_rays):
                a = rot + 2 * math.pi * i / n_rays
                dots.append(_circle_pts(r * math.cos(a), r * math.sin(a), dot_r, 20))
            r *= PHI ** 0.5
        layers.append(("Dots", dots))

    # Outer border
    layers.append(("Outer_Border", [_circle_pts(0, 0, size)]))

    return layers, "hybrid"


# ══════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════

_GENERATORS = {
    "Petal Mandala": _gen_petal_mandala,
    "Sacred Geometry": _gen_sacred_geometry,
    "Geometric Lattice": _gen_geometric_lattice,
    "Hybrid": _gen_hybrid,
}


def _apply_layer_overrides(layers, layer_overrides, rng, size, pattern_type):
    """Apply per-layer shape and scale overrides to generated layers.

    layer_overrides: {editable_layer_index: {"shape": str, "scale": float}}
    editable_layer_index is the index among editable layers only.
    """
    if not layer_overrides:
        return layers

    # Identify editable layers and their indices in the full layers list
    editable_indices = []
    for i, (name, _shapes) in enumerate(layers):
        if name not in _NON_EDITABLE_LAYERS:
            editable_indices.append(i)

    result = list(layers)
    for edit_idx, overrides in layer_overrides.items():
        if edit_idx >= len(editable_indices):
            continue
        full_idx = editable_indices[edit_idx]
        layer_name, shapes = result[full_idx]
        shape_type = overrides.get("shape", "Default")
        scale = overrides.get("scale", 1.0)

        if shape_type not in ("Default", ""):
            # Handle Outer_Border specially with border shapes
            if layer_name == "Outer_Border" and shape_type in BORDER_SHAPE_NAMES:
                border = _make_border_shape(size, shape_type)
                if border:
                    shapes = border
                # Apply scale then continue
                if scale != 1.0:
                    shapes = [_scale_shape_pts(s, scale) for s in shapes]
                result[full_idx] = (layer_name, shapes)
                continue

            # Determine if this layer uses gap-style shapes or hole-style shapes
            is_gap_layer = any(k in layer_name for k in
                               ("Petal_Ring", "Inner_Petals", "Mid_Petals"))
            is_lattice_layer = "Lattice_Ring" in layer_name

            if is_gap_layer and shape_type in GAP_SHAPE_MAP:
                # Re-generate gap shapes with the overridden style
                # We can't easily re-generate from scratch without params,
                # so we rebuild the gap style by extracting params from existing shapes
                # Instead, replace the gap function on the existing shape data
                # by regenerating with the new style
                new_shapes = _rebuild_gap_layer(shapes, shape_type, rng)
                if new_shapes:
                    shapes = new_shapes

            elif is_lattice_layer and shape_type in HOLE_SHAPE_NAMES:
                # Replace each hole with the new shape type
                new_shapes = []
                for shape in shapes:
                    if len(shape) < 3:
                        new_shapes.append(shape)
                        continue
                    # Calculate centroid and approximate size
                    calc = shape[:-1] if (len(shape) > 2 and
                            abs(shape[0][0] - shape[-1][0]) < 0.001 and
                            abs(shape[0][1] - shape[-1][1]) < 0.001) else shape
                    cx = sum(p[0] for p in calc) / len(calc)
                    cy = sum(p[1] for p in calc) / len(calc)
                    max_dist = max(math.hypot(p[0] - cx, p[1] - cy) for p in calc)
                    hole_size = max_dist * 2
                    facing_angle = math.atan2(cy, cx)
                    new_shapes.append(
                        _make_hole_shape(cx, cy, hole_size, shape_type, facing_angle))
                shapes = new_shapes

            elif shape_type in HOLE_SHAPE_NAMES:
                # Generic: replace any closed shape with hole shape
                new_shapes = []
                for shape in shapes:
                    if len(shape) < 3:
                        new_shapes.append(shape)
                        continue
                    calc = shape[:-1] if (len(shape) > 2 and
                            abs(shape[0][0] - shape[-1][0]) < 0.001 and
                            abs(shape[0][1] - shape[-1][1]) < 0.001) else shape
                    cx = sum(p[0] for p in calc) / len(calc)
                    cy = sum(p[1] for p in calc) / len(calc)
                    max_dist = max(math.hypot(p[0] - cx, p[1] - cy) for p in calc)
                    hole_size = max_dist * 2
                    facing_angle = math.atan2(cy, cx)
                    new_shapes.append(
                        _make_hole_shape(cx, cy, hole_size, shape_type, facing_angle))
                shapes = new_shapes

            elif shape_type in GAP_SHAPE_MAP:
                # Apply gap shape to non-gap layers that have closed shapes
                new_shapes = _rebuild_gap_layer(shapes, shape_type, rng)
                if new_shapes:
                    shapes = new_shapes

        # Apply scale
        if scale != 1.0:
            shapes = [_scale_shape_pts(s, scale) for s in shapes]

        result[full_idx] = (layer_name, shapes)

    return result


def _rebuild_gap_layer(shapes, shape_type, rng):
    """Rebuild gap-style shapes with a different gap style.
    Approximates by using the centroid and radial extent of existing shapes."""
    gap_fn_name = GAP_SHAPE_MAP.get(shape_type)
    if not gap_fn_name:
        return None
    gap_fn = GAP_STYLES.get(gap_fn_name, _gap_pointed)

    new_shapes = []
    for shape in shapes:
        if len(shape) < 3:
            new_shapes.append(shape)
            continue
        # Find centroid, inner/outer radius, and center angle
        calc = shape[:-1] if (len(shape) > 2 and
                abs(shape[0][0] - shape[-1][0]) < 0.001 and
                abs(shape[0][1] - shape[-1][1]) < 0.001) else shape
        cx = sum(p[0] for p in calc) / len(calc)
        cy = sum(p[1] for p in calc) / len(calc)
        dists = [math.hypot(p[0], p[1]) for p in calc]
        r_in = min(dists)
        r_out = max(dists)
        ac = math.atan2(cy, cx)
        # Estimate angular half-width
        angles = [math.atan2(p[1], p[0]) for p in calc]
        # Normalize angles relative to center angle
        diffs = []
        for a in angles:
            d = a - ac
            while d > math.pi:
                d -= 2 * math.pi
            while d < -math.pi:
                d += 2 * math.pi
            diffs.append(abs(d))
        ah = max(diffs) if diffs else 0.1
        new_shapes.append(gap_fn(r_in, r_out, ac, ah))
    return new_shapes


def generate_mandala(size, petals, rotation_deg, complexity,
                     pattern_type="Petal Mandala", seed=None,
                     layer_overrides=None):
    rng = random.Random(seed) if seed is not None else random.Random()
    rot = math.radians(rotation_deg)
    gen = _GENERATORS.get(pattern_type, _gen_petal_mandala)
    layers, render_mode = gen(size, petals, rot, complexity, rng)
    if layer_overrides:
        # Use a separate rng for overrides so base design stays stable
        override_rng = random.Random(seed + 9999 if seed is not None else 42)
        layers = _apply_layer_overrides(layers, layer_overrides, override_rng,
                                        size, pattern_type)
    return layers, render_mode


# ══════════════════════════════════════════════════════════════
# Preview rendering
# ══════════════════════════════════════════════════════════════

def plot_mandala(ax, layers, size, render_mode):
    ax.clear()
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_xlabel("inches", fontsize=8, color="#666666")
    ax.set_ylabel("inches", fontsize=8, color="#666666")
    ax.tick_params(labelsize=7, colors="#999999")
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
    margin = size * 0.06
    ax.set_xlim(-size - margin, size + margin)
    ax.set_ylim(-size - margin, size + margin)

    if render_mode == "filled":
        _render_filled(ax, layers, size)
    elif render_mode == "lines":
        _render_lines(ax, layers, size)
    else:
        _render_hybrid(ax, layers, size)


def _render_filled(ax, layers, size):
    # White background = material remains. Black shapes = material removed.
    for layer_name, shapes in layers:
        if layer_name == "Outer_Border":
            shape = shapes[0]
            xs, ys = zip(*shape)
            ax.plot(xs, ys, color="black", linewidth=0.8, zorder=9)
            continue
        for shape in shapes:
            poly = Polygon(shape, closed=True, facecolor="black",
                          edgecolor="black", linewidth=0.2, zorder=5)
            ax.add_patch(poly)


def _render_lines(ax, layers, size):
    for layer_name, shapes in layers:
        lw = 0.5
        if "Dot" in layer_name:
            lw = 0.4
        elif "Outer" in layer_name or "Circle" in layer_name:
            lw = 0.7
        elif "Star" in layer_name or "Triangle" in layer_name:
            lw = 0.9
        elif "Radial" in layer_name:
            lw = 0.3
        for shape in shapes:
            if len(shape) >= 2:
                xs, ys = zip(*shape)
                ax.plot(xs, ys, color="black", linewidth=lw,
                        solid_capstyle="round")


def _render_hybrid(ax, layers, size):
    # White background = material remains. Black = material removed.
    filled_prefixes = ("Center", "Inner_Petals")
    for name, shapes in layers:
        if any(name.startswith(p) for p in filled_prefixes):
            for shape in shapes:
                ax.add_patch(Polygon(shape, closed=True, facecolor="black",
                                     edgecolor="black", linewidth=0.2, zorder=5))
        elif name == "Outer_Border":
            shape = shapes[0]
            xs, ys = zip(*shape)
            ax.plot(xs, ys, color="black", linewidth=0.8, zorder=9)
        else:
            lw = 0.4 if "Dot" in name else 0.6
            if "Star" in name or "Triangle" in name:
                lw = 0.9
            for shape in shapes:
                if len(shape) >= 2:
                    xs, ys = zip(*shape)
                    ax.plot(xs, ys, color="black", linewidth=lw,
                            solid_capstyle="round")


# ══════════════════════════════════════════════════════════════
# DXF export
# ══════════════════════════════════════════════════════════════

DXF_COLORS = {
    "Center": 1, "Petal": 1, "Inner": 1, "Separator": 8,
    "Flower": 5, "Metatron": 3, "Star": 2, "Triangle": 4,
    "Radial": 8, "Dot": 6, "Vesica": 200, "Web": 150,
    "Phi": 3, "Circle": 5, "Outer": 7, "Lattice": 4,
    "Slot": 8, "Mid": 30,
}


def _dxf_color(name):
    for key, c in DXF_COLORS.items():
        if key.lower() in name.lower():
            return c
    return 7


def export_dxf(layers, filepath):
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 1
    msp = doc.modelspace()
    for layer_name, shapes in layers:
        color = _dxf_color(layer_name)
        try:
            doc.layers.add(layer_name, color=color)
        except ezdxf.DXFTableEntryError:
            pass
        for shape in shapes:
            if len(shape) >= 2:
                is_closed = (len(shape) >= 3 and
                             math.hypot(shape[-1][0] - shape[0][0],
                                        shape[-1][1] - shape[0][1]) < 0.001)
                msp.add_lwpolyline(shape, close=is_closed,
                                   dxfattribs={"layer": layer_name})
    doc.saveas(filepath)


# ══════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════

class MandalaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mandala & Sacred Geometry CNC Generator")
        self.root.configure(bg="#F5F5F5")
        self.root.minsize(1100, 780)
        self._pending_update = None
        self.current_layers = []
        self.current_render_mode = "filled"
        self._current_seed = random.randint(0, 2**31)
        self.var_size = tk.DoubleVar(value=12.0)
        self.var_petals = tk.IntVar(value=8)
        self.var_rotation = tk.DoubleVar(value=0.0)
        self.var_complexity = tk.IntVar(value=4)
        self.var_pattern = tk.StringVar(value="Petal Mandala")
        self.layer_overrides = {}
        self._hidden_layers = set()
        self._layer_widgets = []  # list of (shape_var, scale_var, visible_var) per editable layer
        self._build_ui()
        self._update_preview()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        bg = "#F5F5F5"
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground="#333333",
                         font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"),
                         foreground="#222222", background=bg)
        style.configure("Sub.TLabel", font=("Segoe UI", 9),
                         foreground="#666666", background=bg)
        style.configure("TScale", background=bg, troughcolor="#DDDDDD")
        style.configure("TCombobox", font=("Segoe UI", 10))
        style.configure("Export.TButton", font=("Segoe UI", 11, "bold"),
                         foreground="#FFFFFF", background="#2D8A4E")
        style.map("Export.TButton",
                   background=[("active", "#236B3C"), ("pressed", "#1B5230")])
        style.configure("Random.TButton", font=("Segoe UI", 11, "bold"),
                         foreground="#FFFFFF", background="#3B6FB6")
        style.map("Random.TButton",
                   background=[("active", "#2E5A96"), ("pressed", "#224478")])

        style.configure("LayerHeader.TLabel", font=("Segoe UI", 11, "bold"),
                         foreground="#222222", background=bg)
        style.configure("LayerName.TLabel", font=("Segoe UI", 8),
                         foreground="#444444", background=bg)
        style.configure("ScaleVal.TLabel", font=("Segoe UI", 8),
                         foreground="#666666", background=bg)

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # Left panel: scrollable
        left_outer = ttk.Frame(main, width=310)
        left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)
        left_outer.pack_propagate(False)

        left_canvas = tk.Canvas(left_outer, bg="#F5F5F5", highlightthickness=0,
                                width=290)
        left_scrollbar = ttk.Scrollbar(left_outer, orient="vertical",
                                        command=left_canvas.yview)
        left = ttk.Frame(left_canvas, width=290)
        left.bind("<Configure>",
                  lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left, anchor="nw", width=290)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        ttk.Label(left, text="CNC Design Generator",
                  style="Header.TLabel").pack(anchor="w", pady=(0, 2))
        ttk.Label(left, text="All material stays connected",
                  style="Sub.TLabel").pack(anchor="w", pady=(0, 8))

        pat_frame = ttk.Frame(left)
        pat_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(pat_frame, text="Pattern Type:").pack(anchor="w")
        pat_combo = ttk.Combobox(pat_frame, textvariable=self.var_pattern,
                                  values=PATTERN_TYPES, state="readonly",
                                  font=("Segoe UI", 10))
        pat_combo.pack(fill=tk.X)
        pat_combo.bind("<<ComboboxSelected>>",
                       lambda e: self._on_pattern_change())

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=4)

        self._add_slider(left, "Radius (in)", self.var_size, 2, 36, 0.5)
        self._add_slider(left, "Symmetry", self.var_petals, 3, 20, 1)
        self._add_slider(left, "Rotation (\u00b0)", self.var_rotation, 0, 360, 1)
        self._add_slider(left, "Layers", self.var_complexity, 1, 5, 1)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=6)

        self.dim_label = ttk.Label(left, text="", style="Sub.TLabel")
        self.dim_label.pack(anchor="w", pady=(0, 2))
        self.info_label = ttk.Label(left, text="", style="Sub.TLabel")
        self.info_label.pack(anchor="w", pady=(0, 2))
        ttk.Label(left, text=f"\u03C6 = {PHI:.6f}  (golden ratio)",
                  style="Sub.TLabel").pack(anchor="w", pady=(0, 10))

        rand_btn = ttk.Button(left, text="New Random Design",
                               style="Random.TButton", command=self._randomize)
        rand_btn.pack(fill=tk.X, pady=(0, 8), ipady=6)
        export_btn = ttk.Button(left, text="Export DXF",
                                style="Export.TButton", command=self._export)
        export_btn.pack(fill=tk.X, ipady=6)
        png_btn = ttk.Button(left, text="Export PNG",
                             style="Export.TButton", command=self._export_png)
        png_btn.pack(fill=tk.X, pady=(4, 0), ipady=6)

        # Layer Editor section
        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=8)
        ttk.Label(left, text="Layer Editor",
                  style="LayerHeader.TLabel").pack(anchor="w", pady=(0, 4))
        self._layer_editor_frame = ttk.Frame(left)
        self._layer_editor_frame.pack(fill=tk.X, pady=(0, 8))

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.fig, self.ax = plt.subplots(figsize=(7, 7), facecolor="white")
        self.fig.subplots_adjust(left=0.06, right=0.97, top=0.97, bottom=0.05)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()
        toolbar.pack(fill=tk.X)

    def _add_slider(self, parent, label, variable, from_, to, resolution):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 4))
        val_label = ttk.Label(frame, text=f"{label}: {variable.get()}")
        val_label.pack(anchor="w")
        slider_row = ttk.Frame(frame)
        slider_row.pack(fill=tk.X)
        minus_btn = tk.Button(
            slider_row, text="\u2212", width=2, relief="flat",
            bg="#DDDDDD", activebackground="#BBBBBB", font=("Segoe UI", 9, "bold"),
            bd=0, cursor="hand2",
            command=lambda: self._step_slider(variable, -resolution, from_, to,
                                              val_label, label, resolution))
        minus_btn.pack(side=tk.LEFT, padx=(0, 2))
        slider = ttk.Scale(slider_row, from_=from_, to=to, variable=variable,
                           orient="horizontal",
                           command=lambda _v, lbl=val_label, l=label, v=variable:
                           self._on_slider(lbl, l, v, resolution))
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        plus_btn = tk.Button(
            slider_row, text="+", width=2, relief="flat",
            bg="#DDDDDD", activebackground="#BBBBBB", font=("Segoe UI", 9, "bold"),
            bd=0, cursor="hand2",
            command=lambda: self._step_slider(variable, resolution, from_, to,
                                              val_label, label, resolution))
        plus_btn.pack(side=tk.LEFT, padx=(2, 0))

    def _step_slider(self, variable, step, from_, to, val_label, label, resolution):
        raw = variable.get() + step
        clamped = max(from_, min(to, raw))
        snapped = round(clamped / resolution) * resolution
        if isinstance(variable, tk.IntVar):
            snapped = int(snapped)
        variable.set(snapped)
        val_label.config(text=f"{label}: {snapped}")
        self._schedule_update()

    def _on_slider(self, lbl, name, var, res):
        raw = var.get()
        snapped = round(raw / res) * res
        if isinstance(var, tk.IntVar):
            snapped = int(snapped)
        lbl.config(text=f"{name}: {snapped}")
        self._schedule_update()

    def _on_pattern_change(self):
        """Reset layer overrides when pattern type changes."""
        self.layer_overrides = {}
        self._schedule_update()

    def _schedule_update(self):
        if self._pending_update is not None:
            self.root.after_cancel(self._pending_update)
        self._pending_update = self.root.after(120, self._update_preview)

    def _get_params(self):
        return {
            "size": round(self.var_size.get() * 2) / 2,
            "petals": int(round(self.var_petals.get())),
            "rotation_deg": round(self.var_rotation.get()),
            "complexity": int(round(self.var_complexity.get())),
            "pattern_type": self.var_pattern.get(),
        }

    def _collect_overrides(self):
        """Read current override values from UI widgets into layer_overrides."""
        self.layer_overrides = {}
        self._hidden_layers = set()
        for idx, (shape_var, scale_var, visible_var) in enumerate(self._layer_widgets):
            shape = shape_var.get()
            scale = scale_var.get()
            visible = visible_var.get()
            self.layer_overrides[idx] = {"shape": shape, "scale": scale}
            if not visible:
                self._hidden_layers.add(idx)

    def _update_preview(self):
        self._pending_update = None
        params = self._get_params()
        # Collect overrides from UI before generating
        self._collect_overrides()
        overrides = self.layer_overrides if self.layer_overrides else None
        self.current_layers, self.current_render_mode = generate_mandala(
            **params, seed=self._current_seed, layer_overrides=overrides)
        # Filter out hidden editable layers for rendering
        if self._hidden_layers:
            editable_idx = 0
            visible_layers = []
            for name, shapes in self.current_layers:
                if name in _NON_EDITABLE_LAYERS:
                    visible_layers.append((name, shapes))
                else:
                    if editable_idx not in self._hidden_layers:
                        visible_layers.append((name, shapes))
                    editable_idx += 1
            render_layers = visible_layers
        else:
            render_layers = self.current_layers
        plot_mandala(self.ax, render_layers, params["size"],
                     self.current_render_mode)
        diameter = params["size"] * 2
        n_paths = sum(len(s) for _, s in self.current_layers)
        self.dim_label.config(
            text=f"Diameter: {diameter:.1f}\"  ({diameter * 25.4:.0f} mm)")
        self.info_label.config(
            text=f"Cut paths: {n_paths}  |  Layers: {len(self.current_layers)}")
        self.canvas.draw_idle()
        self._rebuild_layer_editor()

    def _rebuild_layer_editor(self):
        """Rebuild the layer editor UI to match current layers."""
        # Find editable layers
        editable = []
        for i, (name, _shapes) in enumerate(self.current_layers):
            if name not in _NON_EDITABLE_LAYERS:
                editable.append((i, name))

        # Check if layer structure changed
        current_names = [name for _, name in editable]
        widget_count = len(self._layer_widgets)
        if widget_count == len(current_names):
            # Layer count same — just update labels if needed
            return

        # Save current override values before rebuilding
        saved_overrides = {}
        for idx, (shape_var, scale_var, visible_var) in enumerate(self._layer_widgets):
            s = shape_var.get()
            sc = scale_var.get()
            vis = visible_var.get()
            saved_overrides[idx] = (s, sc, vis)

        # Clear old widgets
        for child in self._layer_editor_frame.winfo_children():
            child.destroy()
        self._layer_widgets = []

        if not editable:
            ttk.Label(self._layer_editor_frame, text="No editable layers",
                      style="Sub.TLabel").pack(anchor="w")
            return

        # Column headers
        header = ttk.Frame(self._layer_editor_frame)
        header.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(header, text="#", style="ScaleVal.TLabel", width=3).pack(
            side=tk.LEFT)
        ttk.Label(header, text="Shape", style="ScaleVal.TLabel", width=12).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Label(header, text="Scale", style="ScaleVal.TLabel", width=10).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Label(header, text="Visible", style="ScaleVal.TLabel").pack(
            side=tk.LEFT, padx=(6, 0))

        for edit_idx, (full_idx, name) in enumerate(editable):
            row = ttk.Frame(self._layer_editor_frame)
            row.pack(fill=tk.X, pady=(0, 3))

            # Layer number label
            ttk.Label(row, text=f"{edit_idx + 1}", style="LayerName.TLabel",
                      width=3).pack(side=tk.LEFT)

            # Shape combobox — use border shapes for Outer_Border
            shape_values = (BORDER_SHAPE_TYPES if name == "Outer_Border"
                            else LAYER_SHAPE_TYPES)
            default_shape = "Circle" if name == "Outer_Border" else shape_values[0]
            shape_var = tk.StringVar(value=default_shape)
            shape_combo = ttk.Combobox(row, textvariable=shape_var,
                                        values=shape_values,
                                        state="readonly", width=10,
                                        font=("Segoe UI", 8))
            shape_combo.pack(side=tk.LEFT, padx=(0, 4))
            shape_combo.bind("<<ComboboxSelected>>",
                             lambda e: self._schedule_update())

            # Scale +/- with numeric entry
            scale_var = tk.DoubleVar(value=1.0)
            scale_minus = tk.Button(
                row, text="\u2212", width=2, relief="flat",
                bg="#DDDDDD", activebackground="#BBBBBB",
                font=("Segoe UI", 8, "bold"), bd=0, cursor="hand2",
                command=lambda sv=scale_var: self._step_scale(sv, -0.05))
            scale_minus.pack(side=tk.LEFT, padx=(0, 1))
            scale_entry = tk.Entry(
                row, textvariable=scale_var, width=5, justify="center",
                font=("Segoe UI", 8), relief="solid", bd=1)
            scale_entry.pack(side=tk.LEFT)
            scale_entry.bind("<Return>", lambda e: self._on_scale_entry(scale_var))
            scale_entry.bind("<FocusOut>", lambda e: self._on_scale_entry(scale_var))
            scale_plus = tk.Button(
                row, text="+", width=2, relief="flat",
                bg="#DDDDDD", activebackground="#BBBBBB",
                font=("Segoe UI", 8, "bold"), bd=0, cursor="hand2",
                command=lambda sv=scale_var: self._step_scale(sv, 0.05))
            scale_plus.pack(side=tk.LEFT, padx=(1, 0))

            # Toggle switch for visibility
            visible_var = tk.BooleanVar(value=True)
            toggle_btn = tk.Button(
                row, text="ON", width=4, relief="flat",
                bg="#2D8A4E", fg="white", activebackground="#236B3C",
                activeforeground="white", font=("Segoe UI", 7, "bold"),
                bd=0, highlightthickness=0, cursor="hand2",
                command=lambda vv=visible_var: self._toggle_layer(vv))
            toggle_btn.pack(side=tk.LEFT, padx=(6, 0), ipady=1)
            # Store ref so we can update appearance
            visible_var._toggle_btn = toggle_btn

            # Restore saved overrides if they exist for this index
            if edit_idx in saved_overrides:
                s, sc, vis = saved_overrides[edit_idx]
                shape_var.set(s)
                scale_var.set(round(sc, 2))
                visible_var.set(vis)
                if not vis:
                    toggle_btn.config(text="OFF", bg="#999999",
                                      activebackground="#777777")

            self._layer_widgets.append((shape_var, scale_var, visible_var))

    def _toggle_layer(self, visible_var):
        """Toggle layer visibility and update the button appearance."""
        new_val = not visible_var.get()
        visible_var.set(new_val)
        btn = visible_var._toggle_btn
        if new_val:
            btn.config(text="ON", bg="#2D8A4E", activebackground="#236B3C")
        else:
            btn.config(text="OFF", bg="#999999", activebackground="#777777")
        self._schedule_update()

    def _step_scale(self, scale_var, step):
        val = scale_var.get() + step
        val = max(0.05, round(val * 20) / 20)
        scale_var.set(round(val, 2))
        self._schedule_update()

    def _on_scale_entry(self, scale_var):
        try:
            val = float(scale_var.get())
            val = max(0.05, round(val * 20) / 20)
            scale_var.set(round(val, 2))
        except (ValueError, tk.TclError):
            scale_var.set(1.0)
        self._schedule_update()

    def _randomize(self):
        self._current_seed = random.randint(0, 2**31)
        self.var_petals.set(random.randint(5, 16))
        self.var_rotation.set(random.randint(0, 360))
        self.var_complexity.set(random.randint(3, 5))
        self.layer_overrides = {}
        self._layer_widgets = []
        self._update_preview()

    def _export(self):
        params = self._get_params()
        diameter = params["size"] * 2
        ptype = params["pattern_type"].lower().replace(" ", "_")
        default_name = f"mandala_{ptype}_{diameter:.0f}in_{params['petals']}s"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".dxf", initialfile=default_name,
            filetypes=[("DXF Files", "*.dxf"), ("All Files", "*.*")],
            title="Export as DXF (inches)")
        if not filepath:
            return
        try:
            # Filter hidden layers for export
            self._collect_overrides()
            if self._hidden_layers:
                editable_idx = 0
                export_layers = []
                for name, shapes in self.current_layers:
                    if name in _NON_EDITABLE_LAYERS:
                        export_layers.append((name, shapes))
                    else:
                        if editable_idx not in self._hidden_layers:
                            export_layers.append((name, shapes))
                        editable_idx += 1
            else:
                export_layers = self.current_layers
            export_dxf(export_layers, filepath)
            n_paths = sum(len(s) for _, s in export_layers)
            messagebox.showinfo("Export Complete",
                f"Saved: {filepath}\n\nPattern: {params['pattern_type']}\n"
                f"Diameter: {diameter:.1f}\"\nCut paths: {n_paths}\nUnits: inches")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _export_png(self):
        params = self._get_params()
        diameter = params["size"] * 2
        ptype = params["pattern_type"].lower().replace(" ", "_")
        default_name = f"mandala_{ptype}_{diameter:.0f}in_{params['petals']}s"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=default_name,
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")],
            title="Export as PNG")
        if not filepath:
            return
        try:
            self.fig.savefig(filepath, dpi=300, bbox_inches="tight",
                             facecolor="white", edgecolor="none")
            messagebox.showinfo("Export Complete", f"Saved: {filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))


def main():
    root = tk.Tk()
    MandalaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

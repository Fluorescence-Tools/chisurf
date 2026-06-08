"""Cartoon geometry builders for Chimol.

Architecture
------------
The cartoon pipeline matches PyMOL's ``RepCartoon`` design:

1. **Sampler** — smooth a CA backbone path with Catmull-Rom splines and
   propagate up-vectors (:func:`_sample_path`, :func:`_propagate_ups`).
2. **Segmenter** — split the smoothed path into contiguous blocks of the
   same secondary-structure type (helix, strand, loop).
3. **Shape + Extruder** — for each block, build the appropriate cross-section
   shape (oval for helices, rectangle for strands, circle for loops) and
   extrude it along the path, adding an arrowhead at the C-terminus of
   strands.

The public entry point :func:`_generate_cartoon_tube_arrays` dispatches
per SS block and merges the results.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math

import numpy as np


# ---------------------------------------------------------------------------
# Sampler  (unchanged from Chimol original)
# ---------------------------------------------------------------------------

def _catmull_rom(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float,
    tension: float = 0.0,
) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    tau = float(np.clip(tension, 0.0, 1.0))
    m1 = (1.0 - tau) * 0.5 * (p2 - p0)
    m2 = (1.0 - tau) * 0.5 * (p3 - p1)
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2


def _sample_path(
    coords: np.ndarray,
    colors: Optional[np.ndarray],
    subdivisions: int = 5,
    tension: float = 0.0,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return arr, colors
    n = arr.shape[0]
    subdivs = max(int(subdivisions), 1)
    if subdivs <= 1:
        return arr, colors
    out_pos: list[np.ndarray] = []
    out_col: Optional[list[np.ndarray]] = None
    col_arr = None
    if colors is not None:
        col_arr = np.asarray(colors, dtype=float)
        if col_arr.shape[0] == n:
            out_col = []
        else:
            col_arr = None
    for i in range(n - 1):
        p0 = arr[i - 1] if i > 0 else arr[i]
        p1 = arr[i]
        p2 = arr[i + 1]
        p3 = arr[i + 2] if (i + 2) < n else arr[i + 1]
        if col_arr is not None:
            c0 = col_arr[i - 1] if i > 0 else col_arr[i]
            c1 = col_arr[i]
            c2 = col_arr[i + 1]
            c3 = col_arr[i + 2] if (i + 2) < n else col_arr[i + 1]
        for j in range(subdivs):
            if i > 0 and j == 0:
                continue
            t = float(j) / float(subdivs)
            pos = _catmull_rom(p0, p1, p2, p3, t, tension=tension)
            out_pos.append(pos)
            if out_col is not None and col_arr is not None:
                col = _catmull_rom(c0, c1, c2, c3, t, tension=tension)
                out_col.append(col)
    out_pos.append(arr[-1])
    if out_col is not None and col_arr is not None:
        out_col.append(col_arr[-1])
    pos_arr = np.asarray(out_pos, dtype=float)
    col_out_arr = np.asarray(out_col, dtype=float) if out_col is not None else None
    return pos_arr, col_out_arr


def _propagate_ups(
    path: np.ndarray,
    ups_hint: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(path, dtype=float)
    m = pts.shape[0]
    tangents = np.zeros((m, 3), dtype=float)
    ups = np.zeros_like(tangents)
    hint = None
    if ups_hint is not None:
        try:
            hint = np.asarray(ups_hint, dtype=float)
            if hint.shape[0] != m:
                hint = None
        except Exception:
            hint = None
    prev_up = None
    for i in range(m):
        if i == 0:
            t = pts[i + 1] - pts[i]
        elif i == m - 1:
            t = pts[i] - pts[i - 1]
        else:
            t = pts[i + 1] - pts[i - 1]
        tn = float(np.linalg.norm(t))
        if tn <= 0.0:
            t = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            t = t / tn
        tangents[i] = t
        candidate = hint[i] if hint is not None else None
        up_vec = _project_perpendicular(candidate, t)
        if up_vec is None and prev_up is not None:
            up_vec = _project_perpendicular(prev_up, t)
        if up_vec is None:
            fallback = _default_up_from_tangent(t)
            up_vec = _project_perpendicular(fallback, t)
        if up_vec is None:
            up_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        if prev_up is not None and float(np.dot(prev_up, up_vec)) < 0.0:
            up_vec = -up_vec
        ups[i] = up_vec
        prev_up = up_vec
    return tangents, ups


def _project_perpendicular(
    vec: Optional[np.ndarray], direction: np.ndarray
) -> Optional[np.ndarray]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (3,):
        return None
    proj = arr - np.dot(arr, direction) * direction
    norm = float(np.linalg.norm(proj))
    if norm <= 1e-8:
        return None
    return proj / norm


def _default_up_from_tangent(tangent: np.ndarray) -> np.ndarray:
    axis_candidates = (
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
    )
    for axis in axis_candidates:
        proj = axis - np.dot(axis, tangent) * tangent
        if float(np.linalg.norm(proj)) > 1e-6:
            return proj
    return np.array([0.0, 1.0, 0.0], dtype=float)


def _default_side_from_up(up_vec: np.ndarray) -> np.ndarray:
    axis = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(axis, up_vec))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0], dtype=float)
    side = np.cross(up_vec, axis)
    sn = float(np.linalg.norm(side))
    if sn <= 0.0:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return side / sn


# ---------------------------------------------------------------------------
# Frame basis  (3x3 orthonormal frame at each path point)
# ---------------------------------------------------------------------------

def _build_frames(
    tangents: np.ndarray,
    up_vectors: np.ndarray,
) -> np.ndarray:
    """Build 3x3 orthonormal frames along the path.

    Each frame has columns ``[side, up, tangent]`` so that transforming a
    shape vertex ``(sx, sy, sz=0)`` gives::

        point + sx * side + sy * up

    Returns
    -------
    frames : (M, 3, 3)  orthonormal basis matrices.
    """
    m = tangents.shape[0]
    frames = np.zeros((m, 3, 3), dtype=float)
    prev_side = None
    for i in range(m):
        t = tangents[i]
        up = up_vectors[i]
        side = np.cross(t, up)
        sn = float(np.linalg.norm(side))
        if sn <= 0.0:
            side = _default_side_from_up(up)
        else:
            side = side / sn
        if prev_side is not None and float(np.dot(prev_side, side)) < 0.0:
            side = -side
        # Re-orthogonalize up
        up = np.cross(side, t)
        frames[i, :, 0] = side
        frames[i, :, 1] = up
        frames[i, :, 2] = t
        prev_side = side
    return frames


# ---------------------------------------------------------------------------
# Shape constructors  (cross-section profiles)
# ---------------------------------------------------------------------------

def _make_circle_shape(
    n_verts: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (shape_vertices, shape_normals) for a circle.

    PyMOL analog: ``ExtrudeCircle``

    Returns
    -------
    verts : (n_verts, 3)  `(0, side, up)` offsets — built-in z=0.
    norms : (n_verts, 3)  radial normals.
    """
    angles = np.linspace(0.0, 2.0 * math.pi, n_verts, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    verts = np.column_stack([np.zeros_like(cos_a), cos_a * radius, sin_a * radius])
    norms = np.column_stack([np.zeros_like(cos_a), cos_a, sin_a])
    return verts, norms


def _make_oval_shape(
    n_verts: int,
    width: float,
    length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (shape_vertices, shape_normals) for an oval.

    PyMOL analog: ``ExtrudeOval``

    Parameters
    ----------
    width : float
        Extent in the side (x) direction.
    length : float
        Extent in the up (y) direction.
    """
    angles = np.linspace(0.0, 2.0 * math.pi, n_verts, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    verts = np.column_stack([np.zeros_like(cos_a), cos_a * width, sin_a * length])
    norms = np.column_stack([
        np.zeros_like(cos_a),
        cos_a * length,
        sin_a * width,
    ])
    norm_n = np.linalg.norm(norms[:, 1:], axis=1)
    nonzero = norm_n > 0.0
    norms[nonzero, 1] /= norm_n[nonzero]
    norms[nonzero, 2] /= norm_n[nonzero]
    return verts, norms


def _make_rectangle_shape(
    width: float,
    length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (shape_vertices, shape_normals) for a flat ribbon.

    PyMOL analog: ``ExtrudeRectangle`` (mode=0, 8 vertices).

    The rectangle has 8 vertices — 4 corners, each split into two
    vertices with different normals (one per adjacent face).
    """
    c = float(math.cos(math.pi / 4))
    s = float(math.sin(math.pi / 4))

    vdata = np.array([
        [0.0,  c * width, -s * length],
        [0.0,  c * width,  s * length],
        [0.0,  c * width,  s * length],
        [0.0, -c * width,  s * length],
        [0.0, -c * width,  s * length],
        [0.0, -c * width, -s * length],
        [0.0, -c * width, -s * length],
        [0.0,  c * width, -s * length],
    ])

    ndata = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
    ])

    return vdata, ndata


# ---------------------------------------------------------------------------
# Generic extruder
# ---------------------------------------------------------------------------

def _extrude_shape(
    path: np.ndarray,
    frames: np.ndarray,
    shape_verts: np.ndarray,
    shape_norms: np.ndarray,
    colors: Optional[np.ndarray],
    *,
    cap_ends: bool = True,
    cap_first: bool = True,
    cap_last: bool = True,
    vert_scale: Optional[np.ndarray] = None,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Extrude a 2-D cross-section shape along a 3-D path.

    PyMOL analog: ``ExtrudeCGOSurfaceTube`` / ``ExtrudeCGOSurfacePolygon``.

    Parameters
    ----------
    path : (M, 3)
    frames : (M, 3, 3)  orthonormal bases ``[side, up, tangent]``.
    shape_verts : (S, 3)  cross-section vertices in (z, side, up) order, z=0.
    shape_norms : (S, 3)  cross-section normals in same order.
    colors : (M, 4) or None
    cap_ends : bool
        Add flat end caps.
    vert_scale : (M,) or None
        Per-point scale factor for the shape (used for putty/arrow).
    """
    m = path.shape[0]
    s = shape_verts.shape[0]
    if m < 2 or s < 2:
        return None

    total_verts = m * s + int(cap_first) + int(cap_last)

    verts = np.zeros((total_verts, 3), dtype=float)
    norms = np.zeros_like(verts)
    cols_arr: Optional[np.ndarray] = None
    if colors is not None and colors.shape[0] >= m:
        cols_arr = np.zeros((total_verts, 4), dtype=float)

    # Transform shape at each path point
    for i in range(m):
        base = i * s
        side = frames[i, :, 0]  # (3,)
        up = frames[i, :, 1]    # (3,)
        scale = float(vert_scale[i]) if vert_scale is not None else 1.0
        # tv[j] = shape_verts[j, 1] * side + shape_verts[j, 2] * up
        tv = (shape_verts[:, 1:2] * side[None, :] +
              shape_verts[:, 2:3] * up[None, :]) * scale
        tn = (shape_norms[:, 1:2] * side[None, :] +
              shape_norms[:, 2:3] * up[None, :])
        tn_norm = np.linalg.norm(tn, axis=1, keepdims=True)
        tn_mask = tn_norm[:, 0] > 1e-10
        tn[tn_mask] /= tn_norm[tn_mask]
        verts[base:base + s] = path[i:i+1] + tv
        norms[base:base + s] = tn
        if cols_arr is not None and colors is not None and i < colors.shape[0]:
            cols_arr[base:base + s] = colors[i]

    # Faces  (triangle strips between rings)
    face_list: list[list[int]] = []
    for i in range(m - 1):
        i0 = i * s
        i1 = (i + 1) * s
        for j in range(s):
            k0 = i0 + j
            k1 = i0 + (j + 1) % s
            k2 = i1 + j
            k3 = i1 + (j + 1) % s
            face_list.append([k0, k2, k1])
            face_list.append([k1, k2, k3])

    # End caps (triangle fans)
    next_offset = m * s

    def _add_cap(ring_idx: int, reverse: bool):
        nonlocal next_offset
        base = ring_idx * s
        center = verts[base:base + s].mean(axis=0)
        # Place center vertex
        verts[next_offset] = center
        if reverse:
            cap_normal = -frames[ring_idx, :, 2]
        else:
            cap_normal = frames[ring_idx, :, 2]
        norms[next_offset] = cap_normal
        if cols_arr is not None:
            ci = min(ring_idx, colors.shape[0] - 1) if colors is not None else 0
            cols_arr[next_offset] = colors[ci] if colors is not None else np.ones(4)
        center_idx = next_offset
        next_offset += 1
        # Fan triangles: center -> v0 -> v1
        for j in range(1, s - 1):
            v0 = base + (0 if reverse else j)
            v1 = base + (s - j if reverse else j + 1)
            if reverse:
                face_list.append([center_idx, v1, v0])
            else:
                face_list.append([center_idx, v0, v1])

    if cap_first:
        _add_cap(0, reverse=True)
    if cap_last:
        _add_cap(m - 1, reverse=False)

    if not face_list:
        return None

    faces_arr = np.asarray(face_list, dtype=np.int32)
    return verts, norms, faces_arr, cols_arr


def _extrude_arrowhead(
    path: np.ndarray,
    frames: np.ndarray,
    shape_verts: np.ndarray,
    shape_norms: np.ndarray,
    colors: Optional[np.ndarray],
    arrow_sampling: int,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Extrude a strand with an arrowhead at the C-terminus.

    PyMOL analog: ``ExtrudeCGOSurfaceStrand``.

    The first ``m - arrow_sampling`` path points use the normal rectangle
    shape; the last ``arrow_sampling`` points expand in width to form
    the arrowhead, and a flat back-face triangle strip closes the tip.
    """
    m = path.shape[0]
    s = shape_verts.shape[0]
    if m < 2 or s < 2 or arrow_sampling < 1:
        return None

    subN = m - arrow_sampling  # first index of arrow region
    if subN < 0:
        subN = 0

    # We'll produce the body and arrowhead separately and merge.
    # Scale factors: body→1.0 at front, arrow→expanding
    scale = np.ones(m, dtype=float)
    for i in range(subN, m):
        frac = float(m - 1 - i) / max(float(arrow_sampling), 1.0)
        scale[i] = 1.0 + 0.5 * frac  # expand up to 1.5x at tip

    # Body part: normal extrusion with no caps
    body = _extrude_shape(
        path, frames, shape_verts, shape_norms,
        colors, cap_ends=False, vert_scale=scale,
    )
    if body is None:
        return None

    verts_b, norms_b, faces_b, cols_b = body

    # Arrowhead flat back face at subN (the "cut" plane)
    # We need to add a triangle fan at the subN ring to close the arrow
    frame_sub = frames[subN]
    base_sub = subN * s
    center_sub = verts_b[base_sub:base_sub + s].mean(axis=0)

    # Extend arrays
    extra_verts_per_side = s // 2 - 1  # triangle fan verts
    extra_total = 1 + 2 * extra_verts_per_side  # center + two sides fan

    verts_out = np.zeros((verts_b.shape[0] + extra_total, 3), dtype=float)
    verts_out[:verts_b.shape[0]] = verts_b
    norms_out = np.zeros_like(verts_out)
    norms_out[:norms_b.shape[0]] = norms_b
    cols_out = None
    if cols_b is not None:
        cols_out = np.zeros((cols_b.shape[0] + extra_total, 4), dtype=float)
        cols_out[:cols_b.shape[0]] = cols_b

    face_list = faces_b.tolist()
    off = verts_b.shape[0]

    # Back face normal = -tangent at subN
    back_normal = -frame_sub[:, 2]

    # Vertex colors at subN ring
    c0 = colors[subN] if colors is not None and subN < colors.shape[0] else np.ones(4)
    c1 = colors[subN] if colors is not None and subN < colors.shape[0] else np.ones(4)

    # Side 1: use shape_verts indices with positive side component
    side1_idx = [j for j in range(s) if shape_verts[j, 1] >= 0]
    if len(side1_idx) >= 2:
        verts_out[off] = center_sub
        norms_out[off] = back_normal
        if cols_out is not None:
            cols_out[off] = c0
        center_idx = off
        off += 1
        for jj in range(1, len(side1_idx) - 1):
            v0 = base_sub + side1_idx[0]
            v1 = base_sub + side1_idx[jj]
            v2 = base_sub + side1_idx[jj + 1]
            face_list.append([center_idx, v0, v1])
            face_list.append([center_idx, v1, v2])

    # Side 2: shape_verts indices with negative side component
    side2_idx = [j for j in range(s) if shape_verts[j, 1] < 0]
    if len(side2_idx) >= 2:
        verts_out[off] = center_sub
        norms_out[off] = back_normal
        if cols_out is not None:
            cols_out[off] = c1
        center_idx = off
        off += 1
        for jj in range(1, len(side2_idx) - 1):
            v0 = base_sub + side2_idx[0]
            v1 = base_sub + side2_idx[jj]
            v2 = base_sub + side2_idx[jj + 1]
            face_list.append([center_idx, v0, v1])
            face_list.append([center_idx, v1, v2])

    faces_arr = np.asarray(face_list, dtype=np.int32)

    # Trim unused storage
    if off < verts_out.shape[0]:
        verts_out = verts_out[:off]
        norms_out = norms_out[:off]
        if cols_out is not None:
            cols_out = cols_out[:off]

    return verts_out, norms_out, faces_arr, cols_out


# ---------------------------------------------------------------------------
# SS-based segmenter
# ---------------------------------------------------------------------------

def _segment_ss(
    ss_codes: Optional[np.ndarray],
) -> list[dict]:
    """Split the residue indices into contiguous SS blocks.

    Returns a list of dicts with ``start``, ``end`` (residue indices) and
    ``ss_type`` (``'H'``, ``'E'``, ``'C'``).
    """
    if ss_codes is None or ss_codes.size < 1:
        return []

    def _ss_type(c) -> str:
        c = str(c).strip().upper()[:1]
        return "H" if c == "H" else ("E" if c == "E" else "C")

    types = np.array([_ss_type(c) for c in ss_codes])
    n = len(types)
    segments = []
    start = 0
    while start < n:
        ss_t = types[start]
        end = start + 1
        while end < n and types[end] == ss_t:
            end += 1
        segments.append({"start": start, "end": end, "ss_type": ss_t})
        start = end
    return segments


def _residue_to_path_index(
    res_idx: int,
    n_residues: int,
    n_path: int,
) -> int:
    """Map a residue index to the closest path index."""
    if n_residues <= 1:
        return 0
    return int(round(float(res_idx) * (n_path - 1) / (n_residues - 1)))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _generate_cartoon_tube_arrays(
    coords: np.ndarray,
    colors: Optional[np.ndarray],
    trace_ups: Optional[np.ndarray] = None,
    base_radius: float = 0.5,
    segments_circle: int = 14,
    subdivisions: int = 6,
    *,
    style: str = "tube",
    ss_codes: Optional[np.ndarray] = None,
    config: Optional[dict] = None,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Return (vertices, normals, faces, colors) for the cartoon mesh.

    When ``style='ribbon'`` (the default), the cartoon uses PyMOL-style
    per-secondary-structure shapes: oval helices, rectangular strands
    with arrowheads, and round loop tubes.

    When ``style='tube'``, a uniform tube is used for all residues.
    """
    style_l = str(style or "tube").lower()

    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None

    n = arr.shape[0]
    col_arr: Optional[np.ndarray] = None
    if colors is not None:
        col_arr = np.asarray(colors, dtype=float)
        if col_arr.shape[0] != n:
            col_arr = None

    cfg = config or {}
    coordinate_scale = float(cfg.get("coordinate_scale", 1.0))
    subdivisions = int(cfg.get("cartoon_sampling", cfg.get("subdivisions", subdivisions)))
    segments_circle = int(cfg.get("tube_quality", cfg.get("segments_circle", segments_circle)))

    # -- Stage 1: Sample path --
    tension = float(cfg.get("spline_tension", 0.0))
    try:
        path, path_colors = _sample_path(
            arr, col_arr, subdivisions=subdivisions, tension=tension
        )
    except Exception:
        path = arr
        path_colors = col_arr

    m = path.shape[0]
    if m < 2:
        return None

    # -- Densify up-vectors --
    ups_path: Optional[np.ndarray] = None
    if trace_ups is not None:
        try:
            ups_arr = np.asarray(trace_ups, dtype=float)
            if ups_arr.shape[0] == n:
                ups_path, _ = _sample_path(
                    ups_arr, None, subdivisions=subdivisions, tension=tension
                )
        except Exception:
            ups_path = None

    # -- Build frames --
    tangents, up_vectors = _propagate_ups(path, ups_path)
    frames = _build_frames(tangents, up_vectors)

    if style_l == "tube":
        # Uniform tube
        segments = max(int(segments_circle), 6)
        tube_radius = float(cfg.get("tube_radius", base_radius)) * coordinate_scale
        sv, sn = _make_circle_shape(segments, tube_radius)
        return _extrude_shape(
            path, frames, sv, sn, path_colors, cap_ends=True,
        )

    # -- Ribbon / PyMOL-style automatic --
    n_segments = max(int(cfg.get("profile_segments", 10)), 6)
    oval_quality = max(int(cfg.get("oval_quality", n_segments)), 6)
    loop_quality = max(int(cfg.get("loop_quality", 8)), 4)

    oval_width = float(cfg.get("oval_width", 0.25)) * coordinate_scale
    oval_length = float(cfg.get("oval_length", 1.35)) * coordinate_scale
    rect_width = float(cfg.get("rect_width", 0.4)) * coordinate_scale
    rect_length = float(cfg.get("rect_length", 1.4)) * coordinate_scale
    loop_radius = float(cfg.get("loop_radius", 0.2)) * coordinate_scale
    arrow_sampling_residues = max(int(cfg.get("arrow_sampling", 2)), 1)

    # Also support legacy ss_shapes config for backward compatibility. Prefer
    # modern fixed scene-unit keys when present so cartoon thickness does not
    # grow with molecule radius.
    ss_shapes = cfg.get("ss_shapes") or {}
    modern_keys = {"oval_width", "oval_length", "rect_width", "rect_length", "loop_radius"}
    if ss_shapes and not modern_keys.intersection(cfg):
        h_cfg = ss_shapes.get("helix", {})
        oval_width = float(h_cfg.get("width", 0.25)) * coordinate_scale
        oval_length = float(h_cfg.get("thickness", 1.35)) * coordinate_scale
        s_cfg = ss_shapes.get("strand", {})
        rect_width = float(s_cfg.get("width", 0.4)) * coordinate_scale
        rect_length = float(s_cfg.get("thickness", 1.4)) * coordinate_scale
        c_cfg = ss_shapes.get("coil", {})
        loop_radius = float(c_cfg.get("width", 0.2)) * coordinate_scale

    # -- Build shapes --
    oval_sv, oval_sn = _make_oval_shape(oval_quality, oval_width, oval_length)
    rect_sv, rect_sn = _make_rectangle_shape(rect_width, rect_length)
    loop_sv, loop_sn = _make_circle_shape(loop_quality, loop_radius)

    # -- Segment by SS (or fallback to uniform tube) --
    segments = _segment_ss(ss_codes)

    if not segments:
        # No SS data — use a uniform tube
        n_seg = max(int(segments_circle), 6)
        tube_radius = float(cfg.get("tube_radius", base_radius)) * coordinate_scale
        sv, sn = _make_circle_shape(n_seg, tube_radius)
        return _extrude_shape(
            path, frames, sv, sn, path_colors, cap_ends=True,
        )

    # -- Extrude each segment --
    all_verts: list[np.ndarray] = []
    all_norms: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    vert_offset = 0

    for seg in segments:
        ss_t = seg["ss_type"]
        s_res = seg["start"]
        e_res = seg["end"]
        if e_res - s_res < 1:
            continue
        s_path = _residue_to_path_index(s_res, n, m)
        # Match PyMOL's sampling across cartoon-type boundaries: a block
        # covers the segment up to the next residue anchor, otherwise the
        # interpolated samples between SS blocks are dropped and gaps appear.
        e_anchor = min(e_res, n - 1)
        e_path = _residue_to_path_index(e_anchor, n, m) + 1
        if e_path - s_path < 2:
            continue

        seg_path = path[s_path:e_path]
        seg_frames = frames[s_path:e_path]
        seg_colors = path_colors[s_path:e_path] if path_colors is not None else None

        if ss_t == "E":
            # Strand: rectangle + arrowhead
            arrow_samp = min(arrow_sampling_residues, (e_res - s_res) // 2)
            arrow_samp = max(arrow_samp, 1)
            result = _extrude_arrowhead(
                seg_path, seg_frames, rect_sv, rect_sn,
                seg_colors, arrow_samp,
            )
        elif ss_t == "H":
            # Helix: oval
            result = _extrude_shape(
                seg_path, seg_frames, oval_sv, oval_sn,
                seg_colors, cap_first=(s_res == 0), cap_last=(e_res >= n),
            )
        else:
            # Loop: tube
            result = _extrude_shape(
                seg_path, seg_frames, loop_sv, loop_sn,
                seg_colors, cap_first=(s_res == 0), cap_last=(e_res >= n),
            )

        if result is not None:
            v, nrm, f, c = result
            f = f + vert_offset
            all_verts.append(v)
            all_norms.append(nrm)
            all_faces.append(f)
            if c is not None:
                all_cols.append(c)
            vert_offset += v.shape[0]

    if not all_verts:
        return None

    out_verts = np.concatenate(all_verts, axis=0)
    out_norms = np.concatenate(all_norms, axis=0)
    out_faces = np.concatenate(all_faces, axis=0)
    out_cols = np.concatenate(all_cols, axis=0) if all_cols else None

    return out_verts, out_norms, out_faces, out_cols


# ---------------------------------------------------------------------------
# Trace path (simple smoothed line)
# ---------------------------------------------------------------------------

def _generate_trace_arrays(
    coords: np.ndarray,
    colors: Optional[np.ndarray],
    subdivisions: int = 5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return _sample_path(coords, colors, subdivisions=subdivisions)


# ---------------------------------------------------------------------------
# Up-vector builder
# ---------------------------------------------------------------------------

def _build_trace_ups(
    atoms: np.ndarray,
    res_ids: Optional[np.ndarray],
    ca_coords: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if res_ids is None or ca_coords is None:
        return None
    if not isinstance(atoms, np.ndarray):
        return None
    fields = set(atoms.dtype.fields or {})
    if not {"res_id", "atom_name", "xyz"}.issubset(fields):
        return None
    try:
        atom_res_id = np.asarray(atoms["res_id"])
        atom_names = np.char.strip(atoms["atom_name"].astype(str))
        atom_xyz = np.asarray(atoms["xyz"], dtype=float)
    except Exception:
        return None
    n = len(res_ids)
    ups = np.zeros((n, 3), dtype=float)
    for i, rid in enumerate(res_ids):
        mask = atom_res_id == rid
        if not np.any(mask):
            ups[i] = np.array([0.0, 0.0, 1.0], dtype=float)
            continue
        names = atom_names[mask]
        coords = atom_xyz[mask]
        idx_c = np.where(names == "C")[0]
        idx_o = np.where(names == "O")[0]
        if idx_c.size and idx_o.size:
            c = coords[idx_c[0]]
            o = coords[idx_o[0]]
            up_vec = c - o
        else:
            up_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        norm = float(np.linalg.norm(up_vec))
        if norm > 0.0:
            up_vec = up_vec / norm
        ups[i] = up_vec
    for i in range(1, n):
        if float(np.dot(ups[i - 1], ups[i])) < 0.0:
            ups[i] = -ups[i]
    return ups


def _newell_normal(coords: np.ndarray) -> np.ndarray:
    """Calculate the normal vector of a polygon using Newell's method.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 3) representing the coordinates of the polygon.

    Returns
    -------
    np.ndarray
        A unit normal vector of shape (3,).
    """
    normal = np.zeros(3, dtype=float)
    n = len(coords)
    for i in range(n):
        curr = coords[i]
        nxt = coords[(i + 1) % n]
        normal[0] += (curr[1] - nxt[1]) * (curr[2] + nxt[2])
        normal[1] += (curr[2] - nxt[2]) * (curr[0] + nxt[0])
        normal[2] += (curr[0] - nxt[0]) * (curr[1] + nxt[1])
    norm = float(np.linalg.norm(normal))
    if norm > 1e-6:
        return normal / norm
    return np.array([0.0, 0.0, 1.0], dtype=float)


def _generate_prism_mesh(
    coords: np.ndarray,
    thickness: float,
    color: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a 3D prism mesh from a flat polygon.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the polygon vertices, shape (N, 3).
    thickness : float
        Thickness of the extruded prism.
    color : np.ndarray
        RGBA color vector, shape (4,).

    Returns
    -------
    tuple of np.ndarray
        Tuple of (vertices, normals, faces, colors).
    """
    n = len(coords)
    normal = _newell_normal(coords)

    half_thick = thickness / 2.0
    verts_top = coords + normal * half_thick
    verts_bottom = coords - normal * half_thick

    center = coords.mean(axis=0)
    center_top = center + normal * half_thick
    center_bottom = center - normal * half_thick

    verts = []
    norms = []
    faces = []

    # 1. Top cap
    idx_center_top = 0
    verts.append(center_top)
    norms.append(normal)
    for v in verts_top:
        verts.append(v)
        norms.append(normal)
    for i in range(1, n):
        faces.append([idx_center_top, i, i + 1])
    faces.append([idx_center_top, n, 1])

    # 2. Bottom cap
    idx_center_bottom = len(verts)
    verts.append(center_bottom)
    norms.append(-normal)
    for v in verts_bottom:
        verts.append(v)
        norms.append(-normal)
    for i in range(1, n):
        faces.append([idx_center_bottom, idx_center_bottom + i + 1, idx_center_bottom + i])
    faces.append([idx_center_bottom, idx_center_bottom + 1, idx_center_bottom + n])

    # 3. Side faces
    for i in range(n):
        i_next = (i + 1) % n
        v_top_curr = verts_top[i]
        v_top_next = verts_top[i_next]

        edge = v_top_next - v_top_curr
        side_norm = np.cross(edge, normal)
        sn = float(np.linalg.norm(side_norm))
        if sn > 1e-6:
            side_norm /= sn
        else:
            side_norm = np.array([0.0, 1.0, 0.0], dtype=float)

        idx_vt_curr = len(verts)
        verts.append(v_top_curr)
        norms.append(side_norm)

        idx_vt_next = len(verts)
        verts.append(v_top_next)
        norms.append(side_norm)

        idx_vb_curr = len(verts)
        verts.append(verts_bottom[i])
        norms.append(side_norm)

        idx_vb_next = len(verts)
        verts.append(verts_bottom[i_next])
        norms.append(side_norm)

        faces.append([idx_vb_curr, idx_vt_curr, idx_vt_next])
        faces.append([idx_vb_curr, idx_vt_next, idx_vb_next])

    verts_arr = np.asarray(verts, dtype=float)
    norms_arr = np.asarray(norms, dtype=float)
    faces_arr = np.asarray(faces, dtype=np.int32)
    colors_arr = np.tile(color, (len(verts), 1))

    return verts_arr, norms_arr, faces_arr, colors_arr


def _generate_cylinder(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
    color: np.ndarray,
    segments: int = 8,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Generate a capped cylinder connecting two points.

    Parameters
    ----------
    p1 : np.ndarray
        Start coordinate, shape (3,).
    p2 : np.ndarray
        End coordinate, shape (3,).
    radius : float
        Cylinder radius.
    color : np.ndarray
        RGBA color vector, shape (4,).
    segments : int, optional
        Number of circle segments for the cylinder.

    Returns
    -------
    tuple of np.ndarray, or None
        Tuple of (vertices, normals, faces, colors).
    """
    d = p2 - p1
    dn = float(np.linalg.norm(d))
    if dn <= 1e-6:
        return None
    direction = d / dn

    up = _default_up_from_tangent(direction)
    side = np.cross(direction, up)
    sn = float(np.linalg.norm(side))
    if sn > 0.0:
        side /= sn
    else:
        side = np.array([0.0, 1.0, 0.0], dtype=float)
    up = np.cross(side, direction)

    frame = np.column_stack([side, up, direction])
    frames = np.array([frame, frame])
    path = np.array([p1, p2])

    sv, sn_shape = _make_circle_shape(segments, radius)
    colors = np.array([color, color])

    res = _extrude_shape(path, frames, sv, sn_shape, colors, cap_ends=True)
    return res


def _generate_nucleic_cartoon_arrays(
    atoms: np.ndarray,
    coords_all: np.ndarray,
    res_ids: np.ndarray,
    chain_ids: np.ndarray,
    colors: Optional[np.ndarray],
    config: Optional[dict] = None,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Generate ladders and rings for nucleic acids (DNA/RNA).

    Parameters
    ----------
    atoms : np.ndarray
        Structured array of all atoms in the structure.
    coords_all : np.ndarray
        Scaled and centered coordinates of all atoms, shape (N, 3).
    res_ids : np.ndarray
        Residue IDs along the backbone trace.
    chain_ids : np.ndarray
        Chain IDs along the backbone trace.
    colors : np.ndarray or None
        RGBA colors for each residue in the trace.
    config : dict, optional
        Display config dictionary.

    Returns
    -------
    tuple of np.ndarray, or None
        Tuple of (vertices, normals, faces, colors).
    """
    if atoms is None or coords_all is None or res_ids is None or len(res_ids) == 0:
        return None

    cfg = config or {}
    coordinate_scale = float(cfg.get("coordinate_scale", 1.0))
    ladder_radius = float(cfg.get("ladder_radius", 0.15)) * coordinate_scale
    ring_thickness = float(cfg.get("ring_thickness", 0.25)) * coordinate_scale
    backbone_radius = float(cfg.get("backbone_radius", 0.1)) * coordinate_scale
    backbone_quality = int(cfg.get("backbone_quality", 18))

    fields = set(atoms.dtype.fields or {})
    if not {"res_id", "atom_name"}.issubset(fields):
        return None

    atom_res_ids = np.asarray(atoms["res_id"])
    try:
        atom_names = np.char.strip(atoms["atom_name"].astype(str))
    except Exception:
        atom_names = np.array([str(n).strip() for n in atoms["atom_name"]])

    try:
        atom_names_u = np.char.upper(atom_names)
    except Exception:
        atom_names_u = np.array([str(t).upper() for t in atom_names])

    if "chain" in fields:
        try:
            atom_chains = np.char.strip(atoms["chain"].astype(str))
        except Exception:
            atom_chains = np.array([str(c).strip() for c in atoms["chain"]])
    else:
        atom_chains = np.zeros(len(atoms), dtype=object)

    all_verts = []
    all_norms = []
    all_faces = []
    all_colors = []
    vert_offset = 0

    def add_mesh(v, n, f, c):
        nonlocal vert_offset
        if v is not None and len(v) > 0:
            all_verts.append(v)
            all_norms.append(n)
            all_faces.append(f + vert_offset)
            if c is not None:
                all_colors.append(c)
            vert_offset += len(v)

    pyr_ring_names = ["N1", "C2", "N3", "C4", "C5", "C6"]
    pur_ring6_names = ["N1", "C2", "N3", "C4", "C5", "C6"]
    pur_ring5_names = ["C4", "C5", "N7", "C8", "N9"]
    
    # Collect backbone coordinates (P or C1') for backbone trace
    backbone_coords = []
    backbone_colors = []

    for i, rid in enumerate(res_ids):
        chain_id = chain_ids[i] if chain_ids is not None else ""

        mask = (atom_res_ids == rid) & (atom_chains == chain_id)
        if not np.any(mask):
            continue

        res_atom_names = atom_names_u[mask]
        res_coords = coords_all[mask]

        atom_to_coord = {}
        for name, coord in zip(res_atom_names, res_coords):
            atom_to_coord[name] = coord

        # Use C1' or C1* for backbone to match ladder connection point
        # Fall back to P if sugar atoms are not available
        backbone_atom_name = None
        for cand in ["C1'", "C1*", "P"]:  # Prefer C1', then C1*, then P
            if cand in atom_to_coord:
                backbone_atom_name = cand
                break
        
        if backbone_atom_name is None:
            continue
        
        # Get backbone atom coordinate
        backbone_coord = atom_to_coord[backbone_atom_name]
        res_color = colors[i] if colors is not None else np.array([1.0, 1.0, 1.0, 1.0])
        
        # Collect backbone coordinates (P or C1')
        backbone_coords.append(backbone_coord.copy())
        backbone_colors.append(res_color.copy())
        
        # For ladder and base rings, we need C1' or C1*
        c1_name = None
        for cand in ["C1'", "C1*", backbone_atom_name]:  # Try C1', then C1*, then whatever we have
            if cand in atom_to_coord:
                c1_name = cand
                break

        if not c1_name:
            # If we have a backbone atom but no C1', just add backbone and skip ladder/rings
            continue

        c1_coord = atom_to_coord[c1_name]

        is_purine = False
        is_pyrimidine = False
        base_anchor_name = None

        if "N9" in atom_to_coord:
            is_purine = True
            base_anchor_name = "N9"
        elif "N1" in atom_to_coord:
            is_pyrimidine = True
            base_anchor_name = "N1"

        if not base_anchor_name:
            continue

        base_anchor_coord = atom_to_coord[base_anchor_name]

        ladder_mesh = _generate_cylinder(c1_coord, base_anchor_coord, ladder_radius, res_color)
        if ladder_mesh is not None:
            add_mesh(*ladder_mesh)

        if is_purine:
            if all(n in atom_to_coord for n in pur_ring6_names):
                coords6 = np.array([atom_to_coord[n] for n in pur_ring6_names])
                r6_mesh = _generate_prism_mesh(coords6, ring_thickness, res_color)
                add_mesh(*r6_mesh)
            if all(n in atom_to_coord for n in pur_ring5_names):
                coords5 = np.array([atom_to_coord[n] for n in pur_ring5_names])
                r5_mesh = _generate_prism_mesh(coords5, ring_thickness, res_color)
                add_mesh(*r5_mesh)
        elif is_pyrimidine:
            if all(n in atom_to_coord for n in pyr_ring_names):
                coords6 = np.array([atom_to_coord[n] for n in pyr_ring_names])
                r6_mesh = _generate_prism_mesh(coords6, ring_thickness, res_color)
                add_mesh(*r6_mesh)

    # Generate backbone tube connecting C1' atoms
    if len(backbone_coords) >= 2:
        bb_coords = np.array(backbone_coords, dtype=float)
        bb_colors = np.array(backbone_colors, dtype=float) if backbone_colors else None
        
        # Sample the backbone path for smoothness
        bb_smooth, bb_colors_smooth = _sample_path(bb_coords, bb_colors)
        
        # Generate tube arrays for backbone
        backbone_arrays = _generate_cartoon_tube_arrays(
            bb_smooth,
            bb_colors_smooth,
            None,  # No trace_ups for backbone
            base_radius=backbone_radius,
            style="tube",
            ss_codes=None,
            config={"coordinate_scale": coordinate_scale, "tube_quality": backbone_quality},
        )
        
        if backbone_arrays is not None:
            bb_verts, bb_norms, bb_faces, bb_cols = backbone_arrays
            add_mesh(bb_verts, bb_norms, bb_faces, bb_cols)
    
    if not all_verts:
        return None

    out_verts = np.concatenate(all_verts, axis=0)
    out_norms = np.concatenate(all_norms, axis=0)
    out_faces = np.concatenate(all_faces, axis=0)
    out_colors = np.concatenate(all_colors, axis=0) if all_colors else None

    return out_verts, out_norms, out_faces, out_colors


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------

_smooth_backbone = _sample_path
_extrude_sweep = _extrude_shape
_build_profile = lambda *a, **kw: {}


__all__ = [
    "_build_trace_ups",
    "_generate_cartoon_tube_arrays",
    "_generate_nucleic_cartoon_arrays",
    "_sample_path",
    "_generate_trace_arrays",
    "_build_profile",
    "_extrude_sweep",
    "_propagate_ups",
]

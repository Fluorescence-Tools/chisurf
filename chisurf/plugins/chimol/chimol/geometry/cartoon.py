"""Cartoon geometry builders for Chimol."""

from __future__ import annotations

from typing import Optional, Tuple

import math

import numpy as np


def _generate_cartoon_ribbon_arrays(
    path: np.ndarray,
    path_colors: Optional[np.ndarray],
    ups_path: Optional[np.ndarray],
    base_radius: float,
    subdivisions: int,
    ss_codes: Optional[np.ndarray],
    config: Optional[dict],
    n_samples: int,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Build ribbon vertex/normal/color arrays for a backbone path."""
    m = path.shape[0]
    if m < 2:
        return None

    cfg = config or {}
    profile_segments = max(int(cfg.get("profile_segments", 20)), 4)
    ribbon_scale = float(cfg.get("ribbon_thickness_scale", 0.45))

    widths_res, thickness_res, profile_power_res = _compute_ss_profile_arrays(
        ss_codes,
        cfg,
        max(int(n_samples), 2),
        base_radius,
        ribbon_scale,
    )

    if (
        widths_res is None
        or thickness_res is None
        or profile_power_res is None
    ):
        widths_res = np.full(path.shape[0], base_radius, dtype=float)
        thickness_res = np.full(path.shape[0], base_radius * ribbon_scale, dtype=float)
        profile_power_res = np.ones(path.shape[0], dtype=float) * 2.0

    width_path = _subdivide_attribute_array(widths_res, subdivisions)
    thickness_path = _subdivide_attribute_array(thickness_res, subdivisions)
    profile_power_path = _subdivide_attribute_array(profile_power_res, subdivisions)

    if width_path.shape[0] != m:
        width_path = _resample_attribute_array(width_path, m)
    if thickness_path.shape[0] != m:
        thickness_path = _resample_attribute_array(thickness_path, m)
    if profile_power_path.shape[0] != m:
        profile_power_path = _resample_attribute_array(profile_power_path, m)

    strand_mask_path: Optional[np.ndarray] = None
    helix_mask_path: Optional[np.ndarray] = None
    try:
        if ss_codes is not None:
            codes_arr = np.asarray(ss_codes)
            strand_mask_res = np.array(
                [str(c).strip().upper().startswith("E") for c in codes_arr],
                dtype=float,
            )
            helix_mask_res = np.array(
                [str(c).strip().upper().startswith("H") for c in codes_arr],
                dtype=float,
            )
            strand_mask_path = _subdivide_attribute_array(strand_mask_res, subdivisions)
            helix_mask_path = _subdivide_attribute_array(helix_mask_res, subdivisions)
            if strand_mask_path.shape[0] != m:
                strand_mask_path = _resample_attribute_array(strand_mask_path, m)
            if helix_mask_path.shape[0] != m:
                helix_mask_path = _resample_attribute_array(helix_mask_path, m)
    except Exception:
        strand_mask_path = None
        helix_mask_path = None

    two_pi = 2.0 * math.pi
    angles = np.linspace(0.0, two_pi, profile_segments, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    verts = np.zeros((m * profile_segments, 3), dtype=float)
    norms = np.zeros_like(verts)
    cols: Optional[np.ndarray]
    if path_colors is not None:
        cols = np.zeros((m * profile_segments, 4), dtype=float)
    else:
        cols = None

    tangents, up_vectors = _build_parallel_transport_frames(path, ups_path)
    prev_side = None

    for i in range(m):
        p = path[i]
        t = tangents[i]
        up_vec = up_vectors[i]

        side = np.cross(t, up_vec)
        sn = float(np.linalg.norm(side))
        if sn <= 0.0:
            side = _default_side_from_up(up_vec)
        else:
            side = side / sn

        if prev_side is not None and float(np.dot(prev_side, side)) < 0.0:
            side = -side
        prev_side = side

        width = max(float(width_path[i]), 1e-4)
        thickness = max(float(thickness_path[i]), 1e-4)
        power = max(float(profile_power_path[i]), 1e-3)
        exponent = 2.0 / max(power, 1e-3)
        base_index = i * profile_segments

        is_strand = (
            bool(strand_mask_path[i] >= 0.5)
            if strand_mask_path is not None and i < strand_mask_path.shape[0]
            else False
        )
        is_helix = (
            bool(helix_mask_path[i] >= 0.5)
            if helix_mask_path is not None and i < helix_mask_path.shape[0]
            else False
        )

        # Rotate helix cross-section 90° around the tangent to match NGL-style ribbon
        if is_strand:
            side_eff = up_vec
            up_eff = -side
        elif is_helix:
            side_eff = up_vec
            up_eff = side
        else:
            side_eff = side
            up_eff = up_vec

        for j in range(profile_segments):
            cx = math.copysign(abs(cos_a[j]) ** exponent, cos_a[j])
            cy = math.copysign(abs(sin_a[j]) ** exponent, sin_a[j])
            offset = (width * 0.5 * cx) * side_eff + (thickness * 0.5 * cy) * up_eff
            verts[base_index + j] = p + offset
            norm_vec = offset
            norm_len = float(np.linalg.norm(norm_vec))
            if norm_len > 0.0:
                norms[base_index + j] = norm_vec / norm_len
            else:
                norms[base_index + j] = up_vec
            if cols is not None and path_colors is not None:
                cols[base_index + j] = path_colors[i]

    faces = []
    for i in range(m - 1):
        i0 = i * profile_segments
        i1 = (i + 1) * profile_segments
        for j in range(profile_segments):
            k0 = i0 + j
            k1 = i0 + (j + 1) % profile_segments
            k2 = i1 + j
            k3 = i1 + (j + 1) % profile_segments
            faces.append((k0, k2, k1))
            faces.append((k1, k2, k3))

    if not faces:
        return None

    faces_arr = np.asarray(faces, dtype=np.int32)
    return verts, norms, faces_arr, cols


def _catmull_rom(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float,
    tension: float = 0.0,
) -> np.ndarray:
    """Cardinal (Catmull-Rom) spline with adjustable tension.

    ``tension`` in ``[0, 1]`` reduces overshoot and keeps strands straighter;
    ``0`` is the classic Catmull-Rom, ``1`` moves toward straight segments.
    """

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



def _subdivide_attribute_array(values: np.ndarray, subdivisions: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    n = vals.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    subdivisions = max(int(subdivisions), 1)
    if subdivisions <= 1 or n == 1:
        return vals.copy()

    out: list[np.ndarray] = []
    for i in range(n - 1):
        v0 = vals[i]
        v1 = vals[i + 1]
        for j in range(subdivisions):
            t = float(j) / float(subdivisions)
            out.append((1.0 - t) * v0 + t * v1)
    out.append(vals[-1])
    return np.asarray(out, dtype=float)


def _resample_attribute_array(values: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.zeros(0, dtype=float)
    src = np.asarray(values, dtype=float)
    if src.size == 0:
        return np.zeros(target_len, dtype=float)
    x_src = np.linspace(0.0, 1.0, src.size)
    x_dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_dst, x_src, src)


def _compute_ss_profile_arrays(
    ss_codes: Optional[np.ndarray],
    config: dict,
    n_points: int,
    base_radius: float,
    base_thickness_scale: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if n_points <= 0:
        return None, None, None

    codes = None
    if ss_codes is not None:
        try:
            codes = np.asarray(ss_codes)
        except Exception:
            codes = None

    shapes_cfg = (config.get("ss_shapes") or {}) if config is not None else {}

    def _shape_for(code: Optional[str]) -> dict:
        key = str(code).strip().upper()[:1] if code is not None else "C"
        if key == "H":
            lookup = "helix"
        elif key == "E":
            lookup = "strand"
        else:
            lookup = "coil"
        return shapes_cfg.get(lookup, shapes_cfg.get("coil", {}))

    widths = np.zeros(n_points, dtype=float)
    thicknesses = np.zeros(n_points, dtype=float)
    strand_mask = np.zeros(n_points, dtype=bool)
    strand_arrow_scale = np.ones(n_points, dtype=float)
    profile_power = np.ones(n_points, dtype=float)

    for i in range(n_points):
        code = codes[i] if (codes is not None and i < codes.shape[0]) else None
        shape = _shape_for(code)
        width_scale = float(shape.get("width", 1.0))
        thick_scale = float(shape.get("thickness", 1.0))
        widths[i] = base_radius * width_scale
        thicknesses[i] = base_radius * base_thickness_scale * thick_scale
        profile_power[i] = float(shape.get("profile_power", 2.0))
        if str(code).strip().upper()[:1] == "E":
            strand_mask[i] = True
            strand_arrow_scale[i] = float(shape.get("arrow_scale", 1.0))

    if np.any(strand_mask):
        arrow_len = max(int(config.get("arrow_tip_residues", 0)), 0) if config else 0
        if arrow_len > 0:
            start = 0
            n = n_points
            while start < n:
                if not strand_mask[start]:
                    start += 1
                    continue
                end = start
                while end + 1 < n and strand_mask[end + 1]:
                    end += 1
                block_len = end - start + 1
                tip = min(block_len, arrow_len)
                if tip > 0:
                    arrow_scale = float(np.max(strand_arrow_scale[start : end + 1]))
                    if not math.isclose(arrow_scale, 1.0):
                        for offset in range(tip):
                            idx = end - tip + 1 + offset
                            frac = float(offset + 1) / float(tip)
                            widths[idx] *= 1.0 + (arrow_scale - 1.0) * frac
                            thicknesses[idx] *= max(0.2, 1.0 - 0.4 * frac)
                start = end + 1

    return widths, thicknesses, profile_power


def _build_parallel_transport_frames(
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


def _project_perpendicular(vec: Optional[np.ndarray], direction: np.ndarray) -> Optional[np.ndarray]:
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



def _smooth_backbone(
    coords: np.ndarray,
    colors: Optional[np.ndarray],
    subdivisions: int = 5,
    tension: float = 0.0,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Return a slightly densified backbone polyline and matching colors.

    A small number of linear subdivisions between consecutive CA atoms
    produces a smoother-looking cartoon without heavy spline machinery.
    """

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

    # Append final endpoint explicitly
    out_pos.append(arr[-1])
    if out_col is not None and col_arr is not None:
        out_col.append(col_arr[-1])

    pos_arr = np.asarray(out_pos, dtype=float)
    col_out_arr = np.asarray(out_col, dtype=float) if out_col is not None else None
    return pos_arr, col_out_arr

def _build_trace_ups(
    atoms: np.ndarray,
    res_ids: Optional[np.ndarray],
    ca_coords: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Estimate backbone up-vectors per CA from full atom data.

    Uses C-O vectors within each residue as an approximate "up" direction
    similar to pyball. Returns an array of shape ``(N, 3)`` aligned to
    ``res_ids`` / ``ca_coords`` or ``None`` if not available.
    """

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

    # Make neighbouring up-vectors point in a consistent direction
    for i in range(1, n):
        if float(np.dot(ups[i - 1], ups[i])) < 0.0:
            ups[i] = -ups[i]

    return ups


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
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Return (vertices, normals, faces, colors) for the cartoon mesh."""

    style_l = str(style or "tube").lower()

    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None

    n = arr.shape[0]
    col_arr: Optional[np.ndarray]
    if colors is not None:
        col_arr = np.asarray(colors, dtype=float)
        if col_arr.shape[0] != n:
            col_arr = None
    else:
        col_arr = None

    tension = 0.0
    if config is not None:
        try:
            tension = float(config.get("spline_tension", 0.0))
        except Exception:
            tension = 0.0

    try:
        path, path_colors = _smooth_backbone(
            arr, col_arr, subdivisions=subdivisions, tension=tension
        )
    except Exception:
        path = arr
        path_colors = col_arr

    m = path.shape[0]
    if m < 2:
        return None

    # Densify/up-sample up-vectors if provided
    ups_path: Optional[np.ndarray] = None
    if trace_ups is not None:
        try:
            ups_arr = np.asarray(trace_ups, dtype=float)
            if ups_arr.shape[0] == n:
                ups_path, _ = _smooth_backbone(
                    ups_arr, None, subdivisions=subdivisions, tension=tension
                )
        except Exception:
            ups_path = None

    if style_l == "ribbon":
        return _generate_cartoon_ribbon_arrays(
            path,
            path_colors,
            ups_path,
            base_radius,
            subdivisions,
            ss_codes,
            config,
            n,
        )

    two_pi = 2.0 * math.pi
    segments_circle = max(int(segments_circle), 6)
    angles = np.linspace(0.0, two_pi, segments_circle, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    verts = np.zeros((m * segments_circle, 3), dtype=float)
    norms = np.zeros_like(verts)
    cols: Optional[np.ndarray]
    if path_colors is not None:
        cols = np.zeros((m * segments_circle, 4), dtype=float)
    else:
        cols = None

    for i in range(m):
        p = path[i]
        if i == 0:
            t = path[i + 1] - path[i]
        elif i == m - 1:
            t = path[i] - path[i - 1]
        else:
            t = path[i + 1] - path[i - 1]

        tn = float(np.linalg.norm(t))
        if tn <= 0.0:
            t = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            t = t / tn

        if ups_path is not None and ups_path.shape[0] == m:
            up_vec = ups_path[i]
        else:
            up_vec = np.array([0.0, 0.0, 1.0], dtype=float)

        # Remove component parallel to tangent
        up_vec = up_vec - np.dot(up_vec, t) * t
        un = float(np.linalg.norm(up_vec))
        if un <= 0.0:
            if abs(t[2]) < 0.9:
                up_vec = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                up_vec = np.array([1.0, 0.0, 0.0], dtype=float)
            up_vec = up_vec - np.dot(up_vec, t) * t
            un = float(np.linalg.norm(up_vec))
        if un > 0.0:
            up_vec = up_vec / un

        side = np.cross(t, up_vec)
        sn = float(np.linalg.norm(side))
        if sn > 0.0:
            side = side / sn
        else:
            side = np.array([1.0, 0.0, 0.0], dtype=float)

        radius = float(base_radius)
        base_index = i * segments_circle
        for j in range(segments_circle):
            nx = cos_a[j] * up_vec[0] + sin_a[j] * side[0]
            ny = cos_a[j] * up_vec[1] + sin_a[j] * side[1]
            nz = cos_a[j] * up_vec[2] + sin_a[j] * side[2]
            normal = np.array([nx, ny, nz], dtype=float)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm > 0.0:
                normal /= normal_norm
            norms[base_index + j] = normal
            verts[base_index + j] = p + radius * normal
            if cols is not None and path_colors is not None:
                cols[base_index + j] = path_colors[i]

    faces = []
    for i in range(m - 1):
        i0 = i * segments_circle
        i1 = (i + 1) * segments_circle
        for j in range(segments_circle):
            k0 = i0 + j
            k1 = i0 + (j + 1) % segments_circle
            k2 = i1 + j
            k3 = i1 + (j + 1) % segments_circle
            faces.append((k0, k2, k1))
            faces.append((k1, k2, k3))

    if not faces:
        return None

    faces_arr = np.asarray(faces, dtype=np.int32)
    return verts, norms, faces_arr, cols


def _generate_trace_arrays(
    coords: np.ndarray,
    colors: Optional[np.ndarray],
    subdivisions: int = 5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return smoothed trace coordinates and colors for the CA backbone."""

    return _smooth_backbone(coords, colors, subdivisions=subdivisions)


__all__ = [
    "_build_trace_ups",
    "_generate_cartoon_tube_arrays",
    "_generate_cartoon_ribbon_arrays",
    "_smooth_backbone",
    "_generate_trace_arrays",
]


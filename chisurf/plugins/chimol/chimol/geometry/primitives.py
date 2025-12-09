from __future__ import annotations

from typing import Optional, Tuple

import math

import numpy as np


def _compute_center_radius(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (center, radius) bounding sphere approximation."""

    arr = np.asarray(xyz, dtype=float)
    center = arr.mean(axis=0)
    # Use max distance from center as radius
    diffs = arr - center
    dist2 = np.sum(diffs * diffs, axis=1)
    radius = float(math.sqrt(float(dist2.max()))) if dist2.size else 1.0
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    return center, radius


def _build_sphere_mesh(radius: float) -> Optional[dict[str, np.ndarray]]:
    """Return a procedural sphere approximation with normals."""

    r = float(radius)
    if not np.isfinite(r) or r <= 0.0:
        return None

    rows = 10
    cols = 20
    phi = np.linspace(0.0, np.pi, rows)
    theta = np.linspace(0.0, 2.0 * np.pi, cols, endpoint=False)
    phi, theta = np.meshgrid(phi, theta, indexing="ij")
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x = r * sin_phi * cos_theta
    y = r * sin_phi * sin_theta
    z = r * cos_phi
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    normals = np.stack([sin_phi * cos_theta, sin_phi * sin_theta, cos_phi], axis=-1)
    normals = normals.reshape(-1, 3)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norms, 1e-8, None)

    faces = []
    for i in range(rows - 1):
        for j in range(cols):
            k0 = i * cols + j
            k1 = i * cols + (j + 1) % cols
            k2 = (i + 1) * cols + j
            k3 = (i + 1) * cols + (j + 1) % cols
            faces.append([k0, k2, k1])
            faces.append([k1, k2, k3])
    faces_arr = np.asarray(faces, dtype=np.int32)
    return {"vertices": vertices, "normals": normals, "faces": faces_arr}


_CYLINDER_TEMPLATE_CACHE: dict[int, dict[str, np.ndarray]] = {}


def _get_cylinder_template(segments_circle: int = 12) -> Optional[dict[str, np.ndarray]]:
    """Return cached unit-length cylinder template aligned with the +Z axis."""

    seg = max(3, int(segments_circle))
    template = _CYLINDER_TEMPLATE_CACHE.get(seg)
    if template is not None:
        return template

    angles = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False, dtype=float)
    cos = np.cos(angles)
    sin = np.sin(angles)

    bottom = np.column_stack((cos, sin, np.zeros(seg, dtype=float)))
    top = np.column_stack((cos, sin, np.ones(seg, dtype=float)))

    vertices = np.vstack((bottom, top)).astype(np.float32, copy=False)
    normals = np.vstack(
        (
            np.column_stack((cos, sin, np.zeros(seg, dtype=float))),
            np.column_stack((cos, sin, np.zeros(seg, dtype=float))),
        )
    ).astype(np.float32, copy=False)

    faces: list[list[int]] = []
    for i in range(seg):
        j = (i + 1) % seg
        faces.append([i, j, seg + i])
        faces.append([seg + i, j, seg + j])

    faces_arr = np.asarray(faces, dtype=np.int32)
    template = {
        "vertices": vertices,
        "normals": normals,
        "faces": faces_arr,
        "z": vertices[:, 2].astype(np.float32, copy=False),
    }
    _CYLINDER_TEMPLATE_CACHE[seg] = template
    return template


def _rotation_from_z(direction: np.ndarray) -> np.ndarray:
    """Return rotation matrix that aligns the +Z axis with ``direction``."""

    dir_vec = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(dir_vec))
    if norm <= 1e-8 or not np.isfinite(norm):
        return np.eye(3, dtype=float)
    dir_unit = dir_vec / norm
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    c = float(np.dot(z_axis, dir_unit))
    if c >= 0.9999:
        return np.eye(3, dtype=float)
    if c <= -0.9999:
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=float,
        )

    axis = np.cross(z_axis, dir_unit)
    sin_theta = float(np.linalg.norm(axis))
    if sin_theta <= 1e-8 or not np.isfinite(sin_theta):
        return np.eye(3, dtype=float)
    axis_unit = axis / sin_theta

    kx, ky, kz = axis_unit
    K = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=float,
    )

    rot = np.eye(3, dtype=float) + sin_theta * K + (1.0 - c) * (K @ K)
    return rot


def _build_stick_mesh(
    bonds: np.ndarray,
    atom_positions: np.ndarray,
    atom_colors: Optional[np.ndarray],
    radius: float,
    segments_circle: int = 12,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build a mesh for cylindrical sticks between bonded atom pairs."""

    template = _get_cylinder_template(segments_circle)
    if template is None:
        return None

    pts = np.asarray(atom_positions, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None

    bonds_arr = np.asarray(bonds, dtype=int)
    if bonds_arr.ndim != 2 or bonds_arr.shape[1] != 2:
        return None

    colors_arr: Optional[np.ndarray]
    if atom_colors is not None:
        try:
            colors_arr = np.asarray(atom_colors, dtype=float)
            if colors_arr.shape[0] != pts.shape[0]:
                colors_arr = None
        except Exception:
            colors_arr = None
    else:
        colors_arr = None

    base_color = np.array([0.8, 0.8, 0.8, 1.0], dtype=float)
    base_vertices = np.asarray(template["vertices"], dtype=float)
    base_normals = np.asarray(template["normals"], dtype=float)
    base_faces = np.asarray(template["faces"], dtype=np.int32)
    base_z = np.asarray(template.get("z", base_vertices[:, 2]), dtype=float)

    verts_list: list[np.ndarray] = []
    norms_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    idx_offset = 0

    radius_val = float(radius)
    if not np.isfinite(radius_val) or radius_val <= 0.0:
        return None

    verts_per_cyl = base_vertices.shape[0]

    for pair in bonds_arr:
        i0 = int(pair[0])
        i1 = int(pair[1])
        if (
            i0 < 0
            or i1 < 0
            or i0 >= pts.shape[0]
            or i1 >= pts.shape[0]
            or i0 == i1
        ):
            continue

        start = pts[i0]
        end = pts[i1]
        vec = end - start
        length = float(np.linalg.norm(vec))
        if not np.isfinite(length) or length <= 1e-5:
            continue

        rot = _rotation_from_z(vec)

        verts_local = base_vertices.copy()
        verts_local[:, :2] *= radius_val
        verts_local[:, 2] *= length
        verts_world = verts_local @ rot.T + start
        verts_list.append(verts_world)

        normals_world = base_normals @ rot.T
        norms_list.append(normals_world)

        if colors_arr is not None:
            c0 = colors_arr[i0]
            c1 = colors_arr[i1]
        else:
            c0 = base_color
            c1 = base_color

        z = base_z.reshape(-1, 1)
        col = c0 * (1.0 - z) + c1 * z
        col[:, 3] = 1.0
        cols_list.append(col)

        faces_list.append(base_faces + idx_offset)
        idx_offset += verts_per_cyl

    if not verts_list:
        return None

    positions = np.vstack(verts_list).astype(np.float32, copy=False)
    normals = np.vstack(norms_list).astype(np.float32, copy=False)
    colors = np.vstack(cols_list).astype(np.float32, copy=False)
    faces = np.vstack(faces_list).astype(np.int32, copy=False)

    return positions, normals, faces, colors


__all__ = [
    "_compute_center_radius",
    "_build_sphere_mesh",
    "_get_cylinder_template",
    "_rotation_from_z",
    "_build_stick_mesh",
]


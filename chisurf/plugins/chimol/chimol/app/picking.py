from __future__ import annotations

from typing import Optional

import math

import numpy as np
try:
    import numba as nb
    _HAVE_NUMBA = True
except Exception:
    nb = None
    _HAVE_NUMBA = False

from qtpy import QtGui, QtCore


if _HAVE_NUMBA and nb is not None:
    @nb.jit(nopython=True, nogil=True, cache=True)
    def _pick_from_ray_nb(
        pts: np.ndarray,
        cam: np.ndarray,
        ray_dir: np.ndarray,
        thresh2: float,
    ) -> int:
        n = pts.shape[0]
        best_i = -1
        best_d2 = thresh2
        for i in range(n):
            x0 = pts[i, 0]
            y0 = pts[i, 1]
            z0 = pts[i, 2]
            vx = x0 - cam[0]
            vy = y0 - cam[1]
            vz = z0 - cam[2]
            proj = vx * ray_dir[0] + vy * ray_dir[1] + vz * ray_dir[2]
            if proj <= 0.0:
                continue
            cx = cam[0] + proj * ray_dir[0]
            cy = cam[1] + proj * ray_dir[1]
            cz = cam[2] + proj * ray_dir[2]
            dx = x0 - cx
            dy = y0 - cy
            dz = z0 - cz
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i


def _project_points_to_screen(
    coords: np.ndarray,
    view,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None

    try:
        cam_pos = view.cameraPosition()
        center = view.opts.get("center", None)
    except Exception:
        return None

    if center is None:
        return None

    try:
        cam = np.array([cam_pos.x(), cam_pos.y(), cam_pos.z()], dtype=float)
    except Exception:
        cam = np.asarray(cam_pos, dtype=float)
    cen = np.asarray(center, dtype=float)

    view_dir = cen - cam
    norm = float(np.linalg.norm(view_dir))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    view_dir /= norm

    up0 = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(up0, view_dir))) > 0.99:
        up0 = np.array([0.0, 1.0, 0.0], dtype=float)
    right = np.cross(view_dir, up0)
    rnorm = float(np.linalg.norm(right))
    if not np.isfinite(rnorm) or rnorm <= 0.0:
        return None
    right /= rnorm
    up = np.cross(right, view_dir)
    unorm = float(np.linalg.norm(up))
    if not np.isfinite(unorm) or unorm <= 0.0:
        return None
    up /= unorm

    w = max(1.0, float(view.width()))
    h = max(1.0, float(view.height()))

    fov_deg = float(view.opts.get("fov", 60.0))
    fov = math.radians(fov_deg)
    dist = float(np.linalg.norm(cen - cam))
    if not np.isfinite(dist) or dist <= 0.0:
        dist = 1.0
    half_h = math.tan(fov / 2.0) * dist
    aspect = w / h if h > 0 else 1.0
    half_w = half_h * aspect
    if half_w == 0.0 or half_h == 0.0:
        return None

    v = pts - cam
    z_c = np.dot(v, view_dir)
    mask = np.isfinite(z_c) & (z_c > 0.0)
    if not np.any(mask):
        return None

    x_c = np.dot(v, right)
    y_c = np.dot(v, up)
    scale = np.zeros_like(z_c)
    scale[mask] = dist / z_c[mask]

    x_plane = x_c * scale
    y_plane = y_c * scale

    with np.errstate(invalid="ignore", divide="ignore"):
        nx = x_plane / half_w
        ny = y_plane / half_h

    sx = 0.5 * w * (1.0 + nx)
    sy = 0.5 * h * (1.0 - ny)

    return sx, sy, mask


def pick_residue_from_click(
    coords: np.ndarray,
    view,
    ev: QtGui.QMouseEvent,  # type: ignore[name-defined]
    radius_px: float,
) -> Optional[int]:
    """Return index of the residue closest to the click in screen space."""

    projected = _project_points_to_screen(coords, view)
    if projected is None:
        return None

    sx, sy, mask = projected
    if not np.any(mask):
        return None

    try:
        pos = ev.pos()
        click_x = float(pos.x())
        click_y = float(pos.y())
    except Exception:
        click_x = float(ev.x())
        click_y = float(ev.y())

    valid_idx = np.nonzero(mask)[0]
    dx = sx[mask] - click_x
    dy = sy[mask] - click_y
    dist2 = dx * dx + dy * dy
    if dist2.size == 0:
        return None

    i_local = int(np.argmin(dist2))
    if i_local < 0 or i_local >= dist2.shape[0]:
        return None

    try:
        radius2 = float(radius_px) ** 2
    except Exception:
        radius2 = 64.0  # default 8px squared

    if not np.isfinite(dist2[i_local]) or dist2[i_local] > radius2:
        return None

    return int(valid_idx[i_local])


def pick_residues_in_rect(
    coords: np.ndarray,
    view,
    rect,
) -> np.ndarray:
    try:
        if isinstance(rect, QtCore.QRect):
            x0 = float(rect.left())
            x1 = float(rect.right())
            y0 = float(rect.top())
            y1 = float(rect.bottom())
        else:
            x0 = float(getattr(rect, "left", lambda: 0)())
            x1 = float(getattr(rect, "right", lambda: 0)())
            y0 = float(getattr(rect, "top", lambda: 0)())
            y1 = float(getattr(rect, "bottom", lambda: 0)())
    except Exception:
        return np.zeros(0, dtype=int)

    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)

    projected = _project_points_to_screen(coords, view)
    if projected is None:
        return np.zeros(0, dtype=int)

    sx, sy, mask = projected
    if not np.any(mask):
        return np.zeros(0, dtype=int)

    sel_mask = (
        mask
        & (sx >= xmin)
        & (sx <= xmax)
        & (sy >= ymin)
        & (sy <= ymax)
    )

    if not np.any(sel_mask):
        return np.zeros(0, dtype=int)

    try:
        arr = np.asarray(np.nonzero(sel_mask)[0], dtype=int)
    except Exception:
        arr = np.zeros(0, dtype=int)
    return arr


__all__ = ["pick_residue_from_click", "pick_residues_in_rect"]

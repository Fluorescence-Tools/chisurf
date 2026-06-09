from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import numba as _nb
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    _nb = None  # type: ignore


@dataclass
class Sphere:
    center: np.ndarray  # (3,) world position
    radius: float
    color: np.ndarray  # (3,) RGB in [0, 1]


@dataclass
class RayCamera:
    origin: np.ndarray      # (3,)
    forward: np.ndarray     # (3,) unit vector
    up: np.ndarray          # (3,) unit vector
    fov_degrees: float = 45.0
    far_clip: float = 200.0


_FOV_DEG_DEFAULT = 45.0


def _camera_from_view_state(view: List[float]) -> RayCamera:
    vals = [float(v) for v in view]
    elevation = vals[10]
    azimuth = vals[11]
    distance = max(abs(vals[9]), 1.0)
    target = np.array(vals[12:15], dtype=float)
    far_clip = vals[16] if len(vals) > 16 else 200.0

    theta = math.radians(azimuth)
    phi = math.radians(elevation)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    sin_p = math.sin(phi)
    cos_p = math.cos(phi)

    origin = target + np.array(
        [sin_t * sin_p * distance, cos_t * sin_p * distance, cos_p * distance],
        dtype=float,
    )

    forward = target - origin
    fnorm = np.linalg.norm(forward)
    if fnorm > 1e-9:
        forward /= fnorm
    else:
        forward = np.array([0.0, 0.0, 1.0], dtype=float)

    right = np.array([cos_t, -sin_t, 0.0], dtype=float)
    rn = np.linalg.norm(right)
    if rn > 1e-9:
        right /= rn

    cam_up = np.array([sin_t * cos_p, cos_t * cos_p, -sin_p], dtype=float)
    un = np.linalg.norm(cam_up)
    if un > 1e-9:
        cam_up /= un

    return RayCamera(origin=origin, forward=forward, up=cam_up,
                     fov_degrees=_FOV_DEG_DEFAULT, far_clip=far_clip)


# ------------------------------------------------------------------ #
# Numba-accelerated kernel
# ------------------------------------------------------------------ #

if _HAVE_NUMBA:
    _JIT_SPECDICT = {
        "nopython": True,
        "fastmath": True,
        "cache": True,
        "parallel": True,
    }

    @_nb.njit(fastmath=True, cache=True)
    def _jit_mt_intersect(
        rox: float, roy: float, roz: float,
        dx: float, dy: float, dz: float,
        v0x: float, v0y: float, v0z: float,
        v1x: float, v1y: float, v1z: float,
        v2x: float, v2y: float, v2z: float,
    ) -> float:
        """Moller-Trumbore ray-triangle intersection; returns t or -1."""
        e1x = v1x - v0x
        e1y = v1y - v0y
        e1z = v1z - v0z
        e2x = v2x - v0x
        e2y = v2y - v0y
        e2z = v2z - v0z

        pvecx = dy * e2z - dz * e2y
        pvecy = dz * e2x - dx * e2z
        pvecz = dx * e2y - dy * e2x

        det = e1x * pvecx + e1y * pvecy + e1z * pvecz
        if abs(det) < 1e-12:
            return -1.0
        inv_det = 1.0 / det

        tx = rox - v0x
        ty = roy - v0y
        tz = roz - v0z

        u = (tx * pvecx + ty * pvecy + tz * pvecz) * inv_det
        if u < 0.0 or u > 1.0:
            return -1.0

        qvecx = ty * e1z - tz * e1y
        qvecy = tz * e1x - tx * e1z
        qvecz = tx * e1y - ty * e1x

        v = (dx * qvecx + dy * qvecy + dz * qvecz) * inv_det
        if v < 0.0 or u + v > 1.0:
            return -1.0

        t = (e2x * qvecx + e2y * qvecy + e2z * qvecz) * inv_det
        if t < 1e-6:
            return -1.0
        return t

    @_nb.njit(**_JIT_SPECDICT)
    def _jit_trace(
        centers: np.ndarray,
        radii: np.ndarray,
        col_rgb: np.ndarray,
        tri_vertices: np.ndarray,
        tri_vnormals: np.ndarray,
        tri_colors: np.ndarray,
        n_triangles: int,
        cam_origin: np.ndarray,
        cam_forward: np.ndarray,
        cam_up: np.ndarray,
        fov_radians: float,
        light_dirs: np.ndarray,
        width: int,
        height: int,
        ssaa: int,
        bg_r: int,
        bg_g: int,
        bg_b: int,
        ambient: float,
        diffuse: float,
        specular: float,
        shininess: float,
        direct_spec: float,
        direct_spec_power: float,
        reflect_power: float,
        legacy_lighting: float,
        shadow_enabled: int,
        shadow_fudge: float,
        shadow_decay_factor: float,
        shadow_decay_range: float,
        depth_cue_enabled: int,
        fog_start: float,
        fog_intensity: float,
        far_clip: float,
    ) -> np.ndarray:
        """JIT-compiled ray tracing kernel with sphere + triangle support."""
        n_spheres: int = centers.shape[0]
        rw: int = int(width * ssaa)
        rh: int = int(height * ssaa)
        n_lights: int = light_dirs.shape[0]

        right = np.cross(cam_forward, cam_up)
        rn = _jit_length(right)
        if rn < 1e-9:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right[0] /= rn
            right[1] /= rn
            right[2] /= rn
        fw = cam_forward.copy()
        up = np.cross(right, fw)
        un = _jit_length(up)
        up[0] /= un
        up[1] /= un
        up[2] /= un

        half_h = math.tan(fov_radians * 0.5)
        aspect = float(rw) / float(max(rh, 1))
        half_w = half_h * aspect

        out = np.zeros((height, width, 3), dtype=np.float64)
        accum = np.zeros((height, width), dtype=np.float64)

        bg_f = float(bg_r) / 255.0
        bg_g_f = float(bg_g) / 255.0
        bg_b_f = float(bg_b) / 255.0

        ldirs = np.zeros((n_lights, 3), dtype=np.float64)
        for li in range(n_lights):
            ld = light_dirs[li]
            ln = math.sqrt(ld[0]*ld[0] + ld[1]*ld[1] + ld[2]*ld[2])
            if ln < 1e-9:
                ln = 1.0
            ldirs[li, 0] = ld[0] / ln
            ldirs[li, 1] = ld[1] / ln
            ldirs[li, 2] = ld[2] / ln

        spec_per_light = 1.0 / pow(float(max(n_lights - 1, 1)), 0.6)
        legacy = max(0.0, min(1.0, legacy_lighting))

        for py in _nb.prange(rh):
            for px in range(rw):
                u = (float(px) + 0.5) / float(max(rw - 1, 1)) - 0.5
                v = 0.5 - (float(py) + 0.5) / float(max(rh - 1, 1))

                dir_x = fw[0] + right[0] * u * 2.0 * half_w + up[0] * v * 2.0 * half_h
                dir_y = fw[1] + right[1] * u * 2.0 * half_w + up[1] * v * 2.0 * half_h
                dir_z = fw[2] + right[2] * u * 2.0 * half_w + up[2] * v * 2.0 * half_h
                dlen = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
                if dlen < 1e-9:
                    dlen = 1.0
                dir_x /= dlen
                dir_y /= dlen
                dir_z /= dlen

                rox, roy, roz = cam_origin[0], cam_origin[1], cam_origin[2]

                best_t = np.inf
                best_id = -1
                best_is_tri = False
                best_nx = 0.0
                best_ny = 0.0
                best_nz = 0.0
                best_cr = 0.0
                best_cg = 0.0
                best_cb = 0.0

                # ---- trace spheres ------
                for si in range(n_spheres):
                    r = radii[si]
                    if r <= 0.0:
                        continue
                    cx = centers[si, 0]
                    cy = centers[si, 1]
                    cz = centers[si, 2]
                    ocx = rox - cx
                    ocy = roy - cy
                    ocz = roz - cz
                    b = 2.0 * (ocx * dir_x + ocy * dir_y + ocz * dir_z)
                    c = ocx * ocx + ocy * ocy + ocz * ocz - r * r
                    disc = b * b - 4.0 * c
                    if disc < 0.0:
                        continue
                    sqrt_d = math.sqrt(disc)
                    t1 = (-b - sqrt_d) * 0.5
                    t2 = (-b + sqrt_d) * 0.5
                    if t1 > 1e-6:
                        t_hit = t1
                    elif t2 > 1e-6:
                        t_hit = t2
                    else:
                        continue
                    if t_hit < best_t:
                        best_t = t_hit
                        best_id = si
                        best_is_tri = False

                # ---- trace triangles ------
                for ti in range(n_triangles):
                    v0x = tri_vertices[ti, 0, 0]
                    v0y = tri_vertices[ti, 0, 1]
                    v0z = tri_vertices[ti, 0, 2]
                    v1x = tri_vertices[ti, 1, 0]
                    v1y = tri_vertices[ti, 1, 1]
                    v1z = tri_vertices[ti, 1, 2]
                    v2x = tri_vertices[ti, 2, 0]
                    v2y = tri_vertices[ti, 2, 1]
                    v2z = tri_vertices[ti, 2, 2]

                    t = _jit_mt_intersect(
                        rox, roy, roz, dir_x, dir_y, dir_z,
                        v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z,
                    )
                    if t < 0.0:
                        continue
                    if t < best_t:
                        best_t = t
                        best_id = ti
                        best_is_tri = True
                        # Store triangle color
                        best_cr = tri_colors[ti, 0]
                        best_cg = tri_colors[ti, 1]
                        best_cb = tri_colors[ti, 2]

                if best_id < 0:
                    continue

                # ---- hit point & normal ------
                hx = rox + dir_x * best_t
                hy = roy + dir_y * best_t
                hz = roz + dir_z * best_t

                if best_is_tri:
                    # Interpolate vertex normal via barycentric coords
                    ti = best_id
                    v0x = tri_vertices[ti, 0, 0]
                    v0y = tri_vertices[ti, 0, 1]
                    v0z = tri_vertices[ti, 0, 2]
                    v1x = tri_vertices[ti, 1, 0]
                    v1y = tri_vertices[ti, 1, 1]
                    v1z = tri_vertices[ti, 1, 2]
                    v2x = tri_vertices[ti, 2, 0]
                    v2y = tri_vertices[ti, 2, 1]
                    v2z = tri_vertices[ti, 2, 2]

                    # Compute barycentric coords of hit point
                    e1x = v1x - v0x
                    e1y = v1y - v0y
                    e1z = v1z - v0z
                    e2x = v2x - v0x
                    e2y = v2y - v0y
                    e2z = v2z - v0z
                    ppx = hx - v0x
                    ppy = hy - v0y
                    ppz = hz - v0z
                    d00 = e1x*e1x + e1y*e1y + e1z*e1z
                    d01 = e1x*e2x + e1y*e2y + e1z*e2z
                    d11 = e2x*e2x + e2y*e2y + e2z*e2z
                    d20 = ppx*e1x + ppy*e1y + ppz*e1z
                    d21 = ppx*e2x + ppy*e2y + ppz*e2z
                    denom = d00 * d11 - d01 * d01
                    if abs(denom) > 1e-12:
                        u_bc = (d11 * d20 - d01 * d21) / denom
                        v_bc = (d00 * d21 - d01 * d20) / denom
                    else:
                        u_bc = 0.0
                        v_bc = 0.0
                    w_bc = 1.0 - u_bc - v_bc

                    n0x = tri_vnormals[ti, 0, 0]
                    n0y = tri_vnormals[ti, 0, 1]
                    n0z = tri_vnormals[ti, 0, 2]
                    n1x = tri_vnormals[ti, 1, 0]
                    n1y = tri_vnormals[ti, 1, 1]
                    n1z = tri_vnormals[ti, 1, 2]
                    n2x = tri_vnormals[ti, 2, 0]
                    n2y = tri_vnormals[ti, 2, 1]
                    n2z = tri_vnormals[ti, 2, 2]

                    nx = w_bc * n0x + u_bc * n1x + v_bc * n2x
                    ny = w_bc * n0y + u_bc * n1y + v_bc * n2y
                    nz = w_bc * n0z + u_bc * n1z + v_bc * n2z
                    nl = math.sqrt(nx*nx + ny*ny + nz*nz)
                    if nl < 1e-9:
                        nl = 1.0
                    nx /= nl
                    ny /= nl
                    nz /= nl

                    cr = best_cr
                    cg = best_cg
                    cb = best_cb
                else:
                    nx = hx - centers[best_id, 0]
                    ny = hy - centers[best_id, 1]
                    nz = hz - centers[best_id, 2]
                    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
                    if nl < 1e-9:
                        nl = 1.0
                    nx /= nl
                    ny /= nl
                    nz /= nl
                    cr = col_rgb[best_id, 0]
                    cg = col_rgb[best_id, 1]
                    cb = col_rgb[best_id, 2]

                vx = cam_origin[0] - hx
                vy = cam_origin[1] - hy
                vz = cam_origin[2] - hz
                vl = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-9
                vx /= vl
                vy /= vl
                vz /= vl

                reflect_sum = 0.0
                spec_sum = 0.0

                for li in range(n_lights):
                    lx = ldirs[li, 0]
                    ly = ldirs[li, 1]
                    lz = ldirs[li, 2]

                    if shadow_enabled:
                        lit = _jit_shadow_soft(
                            hx + lx * shadow_fudge,
                            hy + ly * shadow_fudge,
                            hz + lz * shadow_fudge,
                            lx, ly, lz,
                            centers, radii, best_id, n_spheres,
                            shadow_decay_factor, shadow_decay_range,
                        )
                    else:
                        lit = 1.0

                    n_dot_l = nx * lx + ny * ly + nz * lz
                    if n_dot_l < 0.0:
                        n_dot_l = 0.0
                    if n_dot_l > 1.0:
                        n_dot_l = 1.0

                    if lit > 0.0 and n_dot_l > 0.0:
                        reflect_sum += lit * pow(n_dot_l, reflect_power)

                    if lit > 0.0 and n_dot_l > 0.0:
                        hnx = lx + vx
                        hny = ly + vy
                        hnz = lz + vz
                        hn = math.sqrt(hnx*hnx + hny*hny + hnz*hnz)
                        if hn > 1e-9:
                            hnx /= hn
                            hny /= hn
                            hnz /= hn
                            n_dot_h = nx*hnx + ny*hny + nz*hnz
                            if n_dot_h < 0.0:
                                n_dot_h = 0.0
                            if n_dot_h > 1.0:
                                n_dot_h = 1.0
                            spec_sum += lit * pow(n_dot_h, shininess)

                reflect_norm = reflect_sum / float(max(n_lights, 1))

                n_dot_v = nx * vx + ny * vy + nz * vz
                if n_dot_v < 0.0:
                    n_dot_v = 0.0
                if n_dot_v > 1.0:
                    n_dot_v = 1.0
                direct_cmp = pow(n_dot_v, direct_spec_power)

                if legacy > 0.0:
                    n_dot_l0 = nx * ldirs[0, 0] + ny * ldirs[0, 1] + nz * ldirs[0, 2]
                    if n_dot_l0 < 0.0:
                        n_dot_l0 = 0.0
                    legacy_bright = ambient + diffuse * n_dot_l0
                else:
                    legacy_bright = 0.0

                bright = ambient + diffuse * reflect_norm
                if legacy > 0.0:
                    bright = bright * (1.0 - legacy) + legacy_bright * legacy
                if bright < 0.0:
                    bright = 0.0
                if bright > 1.0:
                    bright = 1.0

                excess = direct_spec * direct_cmp + specular * spec_sum * spec_per_light
                if excess < 0.0:
                    excess = 0.0
                if excess > 1.0:
                    excess = 1.0

                cr_out = cr * bright + excess
                cg_out = cg * bright + excess
                cb_out = cb * bright + excess

                if depth_cue_enabled and far_clip > 0.0:
                    nd = best_t / far_clip
                    if nd > fog_start:
                        ffact = (nd - fog_start) / (1.0 - fog_start) * fog_intensity
                        if ffact > 1.0:
                            ffact = 1.0
                        if ffact > 0.0:
                            cr_out = cr_out * (1.0 - ffact) + bg_f * ffact
                            cg_out = cg_out * (1.0 - ffact) + bg_g_f * ffact
                            cb_out = cb_out * (1.0 - ffact) + bg_b_f * ffact

                oy = py // ssaa
                ox = px // ssaa
                out[oy, ox, 0] += cr_out
                out[oy, ox, 1] += cg_out
                out[oy, ox, 2] += cb_out
                accum[oy, ox] += 1.0

        img = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                w = accum[y, x]
                if w < 0.5:
                    r = bg_r
                    g = bg_g
                    b = bg_b
                else:
                    inv = 1.0 / w
                    r = int(out[y, x, 0] * inv * 255.0)
                    g = int(out[y, x, 1] * inv * 255.0)
                    b = int(out[y, x, 2] * inv * 255.0)
                    if r > 255: r = 255
                    if g > 255: g = 255
                    if b > 255: b = 255
                    if r < 0: r = 0
                    if g < 0: g = 0
                    if b < 0: b = 0
                img[y, x, 0] = np.uint8(r)
                img[y, x, 1] = np.uint8(g)
                img[y, x, 2] = np.uint8(b)
        return img

    @_nb.njit(fastmath=True, cache=True)
    def _jit_length(v: np.ndarray) -> float:
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    @_nb.njit(fastmath=True, cache=True)
    def _jit_shadow_soft(
        hx: float, hy: float, hz: float,
        lx: float, ly: float, lz: float,
        centers: np.ndarray,
        radii: np.ndarray,
        skip_idx: int,
        n_spheres: int,
        decay_factor: float,
        decay_range: float,
    ) -> float:
        for si in range(n_spheres):
            if si == skip_idx:
                continue
            r = radii[si]
            if r <= 0.0:
                continue
            cx = centers[si, 0]
            cy = centers[si, 1]
            cz = centers[si, 2]
            ocx = hx - cx
            ocy = hy - cy
            ocz = hz - cz
            b = 2.0 * (ocx * lx + ocy * ly + ocz * lz)
            c = ocx * ocx + ocy * ocy + ocz * ocz - r * r
            disc = b * b - 4.0 * c
            if disc < 0.0:
                continue
            sqrt_d = math.sqrt(disc)
            t = (-b - sqrt_d) * 0.5
            if t > 1e-6:
                if decay_factor > 0.0:
                    d = t - decay_range
                    if d <= 0.0:
                        return 1.0
                    occlusion = 1.0 - math.exp(-d * decay_factor)
                    if occlusion >= 1.0:
                        return 0.0
                    return 1.0 - occlusion
                else:
                    return 0.0
        return 1.0


# ------------------------------------------------------------------ #
# Public API (falls back if Numba unavailable)
# ------------------------------------------------------------------ #


def trace(
    spheres: List[Sphere],
    camera: RayCamera,
    light_directions: np.ndarray,
    width: int = 800,
    height: int = 600,
    *,
    background: Tuple[int, int, int] = (25, 25, 25),
    ambient: float = 0.14,
    diffuse: float = 0.45,
    specular: float = 0.25,
    shininess: float = 40.0,
    ssaa: int = 2,
    direct_specular: float = 0.30,
    direct_specular_power: float = 55.0,
    reflect_power: float = 1.0,
    legacy_lighting: float = 0.0,
    shadow: bool = True,
    shadow_fudge: float = 0.001,
    shadow_decay_factor: float = 0.2,
    shadow_decay_range: float = 1.8,
    gamma: float = 2.2,
    depth_cue: bool = True,
    fog_start: float = 0.45,
    fog_intensity: float = 1.0,
    color_blend: bool = True,
    color_blend_red: float = 0.17,
    color_blend_green: float = 0.25,
    color_blend_blue: float = 0.14,
    # Triangle mesh data (optional)
    tri_vertices: np.ndarray | None = None,
    tri_vnormals: np.ndarray | None = None,
    tri_colors: np.ndarray | None = None,
) -> np.ndarray:
    """Return an (H, W, 3) uint8 raytraced image of spheres and triangles.

    Parameters
    ----------
    spheres : list of Sphere
        Spheres to render.
    camera : RayCamera
        Camera viewpoint.
    light_directions : np.ndarray
        Direction vectors of light sources, shape (N, 3).
    width, height : int
        Output image dimensions.
    tri_vertices : np.ndarray or None
        Triangle vertices, shape (T, 3, 3).
    tri_vnormals : np.ndarray or None
        Per-vertex normals, shape (T, 3, 3).
    tri_colors : np.ndarray or None
        Per-triangle RGB colors, shape (T, 3).

    All other parameters are as in :func:`render_scene`.
    """
    bg_r, bg_g, bg_b = background
    if gamma > 0.0 and gamma != 1.0:
        bg_norm = np.array([bg_r, bg_g, bg_b], dtype=float) / 255.0
        inp = bg_norm.mean()
        if inp > 1e-6:
            sig = pow(float(inp), float(gamma)) / float(inp)
            linear_bg = np.clip(bg_norm * sig * 255.0, 0.0, 255.0)
            bg_r = int(round(linear_bg[0]))
            bg_g = int(round(linear_bg[1]))
            bg_b = int(round(linear_bg[2]))

    has_any = bool(spheres)
    n_tri = 0
    if tri_vertices is not None:
        n_tri = tri_vertices.shape[0]
        has_any = has_any or (n_tri > 0)

    if not has_any:
        img = np.full((height, width, 3), [bg_r, bg_g, bg_b], dtype=np.uint8)
        if color_blend:
            img = _apply_color_blend(img, color_blend_red, color_blend_green, color_blend_blue, _gamma=gamma)
        return img

    n = len(spheres)
    centers = np.zeros((n, 3), dtype=np.float64)
    radii_arr = np.zeros(n, dtype=np.float64)
    colors_arr = np.zeros((n, 3), dtype=np.float64)
    for i, s in enumerate(spheres):
        centers[i] = s.center
        radii_arr[i] = float(s.radius)
        colors_arr[i] = np.clip(s.color, 0.0, 1.0)

    tverts = np.zeros((n_tri, 3, 3), dtype=np.float64)
    tnorms = np.zeros((n_tri, 3, 3), dtype=np.float64)
    tcols = np.zeros((n_tri, 3), dtype=np.float64)
    if n_tri > 0:
        tverts[:] = tri_vertices.astype(np.float64)
        tnorms[:] = tri_vnormals.astype(np.float64) if tri_vnormals is not None else 0.0
        tcols[:] = tri_colors.astype(np.float64) if tri_colors is not None else 0.5

    lds = np.asarray(light_directions, dtype=np.float64)
    if lds.ndim == 1:
        lds = lds.reshape(1, 3)
    elif lds.ndim != 2 or lds.shape[1] != 3:
        lds = lds.reshape(-1, 3)
    for i in range(lds.shape[0]):
        ln = np.linalg.norm(lds[i])
        if ln > 1e-9:
            lds[i] /= ln
        else:
            lds[i] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    fov_rad = math.radians(camera.fov_degrees)

    if _HAVE_NUMBA:
        try:
            img = _jit_trace(
                centers, radii_arr, colors_arr,
                tverts, tnorms, tcols, n_tri,
                camera.origin.astype(np.float64).copy(),
                camera.forward.astype(np.float64).copy(),
                camera.up.astype(np.float64).copy(),
                float(fov_rad),
                lds,
                int(width), int(height), int(ssaa),
                int(bg_r), int(bg_g), int(bg_b),
                float(ambient), float(diffuse), float(specular), float(shininess),
                float(direct_specular), float(direct_specular_power),
                float(reflect_power), float(legacy_lighting),
                int(shadow), float(shadow_fudge),
                float(shadow_decay_factor), float(shadow_decay_range),
                int(depth_cue), float(fog_start), float(fog_intensity),
                float(camera.far_clip),
            )
        except Exception:
            img = _trace_numpy(
                centers, radii_arr, colors_arr,
                tverts, tnorms, tcols, n_tri,
                camera, lds, fov_rad,
                width, height, ssaa,
                background, ambient, diffuse, specular, shininess,
                direct_specular, direct_specular_power, reflect_power, legacy_lighting,
                shadow, shadow_fudge, shadow_decay_factor, shadow_decay_range,
                depth_cue, fog_start, fog_intensity,
                bg_r, bg_g, bg_b,
            )
    else:
        img = _trace_numpy(
            centers, radii_arr, colors_arr,
            tverts, tnorms, tcols, n_tri,
            camera, lds, fov_rad,
            width, height, ssaa,
            background, ambient, diffuse, specular, shininess,
            direct_specular, direct_specular_power, reflect_power, legacy_lighting,
            shadow, shadow_fudge, shadow_decay_factor, shadow_decay_range,
            depth_cue, fog_start, fog_intensity,
            bg_r, bg_g, bg_b,
        )

    if color_blend:
        img = _apply_color_blend(img, color_blend_red, color_blend_green, color_blend_blue,
                                 _gamma=gamma)

    return img


def _apply_color_blend(
    img: np.ndarray,
    red_blend: float,
    green_blend: float,
    blue_blend: float,
    _gamma: float = 2.2,
) -> np.ndarray:
    out = img.astype(np.float64)
    r_part = red_blend * out[:, :, 0]
    g_part = green_blend * out[:, :, 1]
    b_part = blue_blend * out[:, :, 2]
    r_min = np.maximum(g_part, b_part)
    g_min = np.maximum(r_part, b_part)
    b_min = np.maximum(g_part, r_part)
    out[:, :, 0] = np.maximum(out[:, :, 0], r_min)
    out[:, :, 1] = np.maximum(out[:, :, 1], g_min)
    out[:, :, 2] = np.maximum(out[:, :, 2], b_min)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _trace_numpy(
    centers: np.ndarray,
    radii: np.ndarray,
    colors: np.ndarray,
    tri_vertices: np.ndarray,
    tri_vnormals: np.ndarray,
    tri_colors: np.ndarray,
    n_tri: int,
    camera: RayCamera,
    light_dirs: np.ndarray,
    fov_rad: float,
    width: int,
    height: int,
    ssaa: int,
    background: tuple,
    ambient: float,
    diffuse: float,
    specular: float,
    shininess: float,
    direct_specular: float,
    direct_specular_power: float,
    reflect_power: float,
    legacy_lighting: float,
    shadow: bool,
    shadow_fudge: float,
    shadow_decay_factor: float,
    shadow_decay_range: float,
    depth_cue: bool,
    fog_start: float,
    fog_intensity: float,
    bg_r: int, bg_g: int, bg_b: int,
) -> np.ndarray:
    """Pure NumPy fallback for environments without Numba."""
    rw, rh = width * ssaa, height * ssaa
    origins, dirs = _build_ray_grid_numpy(camera, rw, rh, fov_rad)

    n_spheres = centers.shape[0]
    best_t = np.full(origins.shape[:2], np.inf, dtype=np.float64)
    best_id = np.full(origins.shape[:2], -1, dtype=np.int32)
    best_is_tri = np.zeros(origins.shape[:2], dtype=np.bool_)

    # Sphere intersection
    for i in range(n_spheres):
        r = radii[i]
        if r <= 0.0:
            continue
        oc = origins - centers[i]
        b = 2.0 * np.sum(oc * dirs, axis=-1)
        c = np.sum(oc * oc, axis=-1) - r * r
        disc = b * b - 4.0 * c
        hit = disc >= 0.0
        if not np.any(hit):
            continue
        sqrt_d = np.sqrt(np.maximum(disc[hit], 0.0))
        t1 = (-b[hit] - sqrt_d) * 0.5
        t2 = (-b[hit] + sqrt_d) * 0.5
        t = np.where(t1 > 1e-6, t1, np.where(t2 > 1e-6, t2, np.inf))
        better = (t < best_t[hit]) & np.isfinite(t)
        if not np.any(better):
            continue
        rows, cols = np.where(hit)
        good = better.nonzero()[0]
        idx = rows[good], cols[good]
        best_t[idx] = t[good]
        best_id[idx] = i
        best_is_tri[idx] = False

    # Triangle intersection
    for ti in range(n_tri):
        v0 = tri_vertices[ti, 0]
        v1 = tri_vertices[ti, 1]
        v2 = tri_vertices[ti, 2]
        e1 = v1 - v0
        e2 = v2 - v0

        pvec = np.cross(dirs, e2)
        det = np.sum(e1 * pvec, axis=-1)
        valid = np.abs(det) > 1e-12
        if not np.any(valid):
            continue
        inv_det = 1.0 / np.where(valid, det, 1.0)
        tvec = origins - v0
        u = np.sum(tvec * pvec, axis=-1) * inv_det
        qvec = np.cross(tvec, e1)
        v = np.sum(dirs * qvec, axis=-1) * inv_det
        t = np.sum(e2 * qvec, axis=-1) * inv_det

        hit = valid & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-6) & (t < best_t)
        if not np.any(hit):
            continue
        best_t[hit] = t[hit]
        best_id[hit] = ti
        best_is_tri[hit] = True

    hit_mask = best_id >= 0
    bg_rgb = np.array([bg_r, bg_g, bg_b], dtype=np.float64)
    img = np.full((rh, rw, 3), bg_rgb, dtype=np.float64)

    if not hit_mask.any():
        base = _downsample_numpy(img, height, width, ssaa)
        if ssaa > 1:
            base = np.clip(base, 0, 255).astype(np.uint8)
        return base

    positions = origins + dirs * best_t[..., np.newaxis]
    hit_pos = positions[hit_mask]

    hit_normals = np.zeros_like(hit_pos)
    hit_colors = np.zeros((hit_pos.shape[0], 3), dtype=np.float64)

    for i in range(n_spheres):
        m = (best_id == i) & (~best_is_tri) & hit_mask
        if not np.any(m):
            continue
        n = hit_pos[m] - centers[i]
        nl = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        hit_normals[m] = n / nl
        hit_colors[m] = colors[i]

    for ti in range(n_tri):
        m = (best_id == ti) & best_is_tri & hit_mask
        if not np.any(m):
            continue
        v0 = tri_vertices[ti, 0]
        v1 = tri_vertices[ti, 1]
        v2 = tri_vertices[ti, 2]
        n0 = tri_vnormals[ti, 0]
        n1 = tri_vnormals[ti, 1]
        n2 = tri_vnormals[ti, 2]

        hp = hit_pos[m]
        e1 = v1 - v0
        e2 = v2 - v0
        d00 = np.dot(e1, e1)
        d01 = np.dot(e1, e2)
        d11 = np.dot(e2, e2)
        denom = d00 * d11 - d01 * d01
        pp = hp - v0
        d20 = np.sum(pp * e1, axis=-1)
        d21 = np.sum(pp * e2, axis=-1)
        if abs(denom) > 1e-12:
            u_bc = (d11 * d20 - d01 * d21) / denom
            v_bc = (d00 * d21 - d01 * d20) / denom
        else:
            u_bc = 0.0
            v_bc = 0.0
        w_bc = 1.0 - u_bc - v_bc

        n = (w_bc[:, np.newaxis] * n0 + u_bc[:, np.newaxis] * n1 + v_bc[:, np.newaxis] * n2)
        nl = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        hit_normals[m] = n / nl
        hit_colors[m] = tri_colors[ti]

    view_v = camera.origin - hit_pos
    view_v = view_v / (np.linalg.norm(view_v, axis=-1, keepdims=True) + 1e-9)

    n_lights = light_dirs.shape[0]
    spec_per_light = 1.0 / pow(max(n_lights - 1, 1), 0.6)
    legacy = np.clip(legacy_lighting, 0.0, 1.0)
    bg_f = np.array([bg_r / 255.0, bg_g / 255.0, bg_b / 255.0], dtype=np.float64)

    bright_arr = np.full(hit_mask.shape, ambient, dtype=np.float64)
    excess_arr = np.zeros(hit_mask.shape, dtype=np.float64)

    for li in range(n_lights):
        ld = light_dirs[li]
        if shadow:
            occluded = _any_hit_numpy_with_decay(
                positions, ld, centers, radii, hit_mask,
                shadow_fudge, shadow_decay_factor, shadow_decay_range,
            )
            lit = np.where(occluded, 0.0, 1.0)
        else:
            lit = np.ones(hit_mask.shape, dtype=np.float64)

        n_dot_l = np.clip(np.sum(hit_normals * ld, axis=-1), 0.0, 1.0)

        diff_weight = lit[hit_mask] * (n_dot_l ** reflect_power)
        bright_arr[hit_mask] += diffuse * diff_weight / n_lights

        half = ld + view_v
        hn = np.linalg.norm(half, axis=-1, keepdims=True) + 1e-9
        half = half / hn
        n_dot_h = np.clip(np.sum(hit_normals * half, axis=-1), 0.0, 1.0)
        spec_weight = lit[hit_mask] * (n_dot_h ** shininess)
        excess_arr[hit_mask] += specular * spec_weight * spec_per_light

    n_dot_v_arr = np.clip(np.sum(hit_normals * view_v, axis=-1), 0.0, 1.0)
    direct_cmp = n_dot_v_arr ** direct_specular_power
    excess_arr[hit_mask] += direct_specular * direct_cmp

    if legacy > 0.0:
        n_dot_l0 = np.clip(np.sum(hit_normals * light_dirs[0], axis=-1), 0.0, 1.0)
        legacy_bright = ambient + diffuse * n_dot_l0
        bright_arr[hit_mask] = (
            bright_arr[hit_mask] * (1.0 - legacy) + legacy_bright * legacy
        )

    bright_arr = np.clip(bright_arr, 0.0, 1.0)
    excess_arr = np.clip(excess_arr, 0.0, 1.0)

    img[hit_mask] = (hit_colors * bright_arr[hit_mask, np.newaxis] + excess_arr[hit_mask, np.newaxis]) * 255.0

    if depth_cue and camera.far_clip > 0.0:
        nd = best_t / camera.far_clip
        fog = np.zeros_like(nd)
        mask_fog = (nd > fog_start) & hit_mask
        if mask_fog.any():
            fog[mask_fog] = np.clip(
                (nd[mask_fog] - fog_start) / (1.0 - fog_start) * fog_intensity,
                0.0, 1.0,
            )
        for c in range(3):
            img[hit_mask, c] = (
                img[hit_mask, c] * (1.0 - fog[hit_mask])
                + bg_rgb[c] * fog[hit_mask]
            )

    base = _downsample_numpy(np.clip(img, 0.0, 255.0), height, width, ssaa)
    return base


def _build_ray_grid_numpy(camera, rw, rh, fov_rad):
    aspect = rw / max(rh, 1)
    half_h = math.tan(fov_rad * 0.5)
    half_w = half_h * aspect
    right = np.cross(camera.forward, camera.up)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, camera.forward)
    up = up / (np.linalg.norm(up) + 1e-9)
    y = np.linspace(half_h, -half_h, rh)
    x = np.linspace(-half_w, half_w, rw)
    gy, gx = np.meshgrid(y, x, indexing="ij")
    dirs = camera.forward + right * gx[..., np.newaxis] + up * gy[..., np.newaxis]
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.tile(camera.origin.reshape(1, 1, 3), (rh, rw, 1))
    return origins, dirs


def _any_hit_numpy_with_decay(
    origins, light_dir, centers, radii, mask,
    fudge, decay_factor, decay_range,
):
    h, w = mask.shape
    ld = light_dir / (np.linalg.norm(light_dir) + 1e-9)
    result = np.zeros((h, w), dtype=bool)
    offset_origins = origins + ld * fudge
    for i in range(centers.shape[0]):
        r = radii[i]
        if r <= 0.0:
            continue
        oc = offset_origins - centers[i]
        b = 2.0 * np.sum(oc * ld, axis=-1)
        c = np.sum(oc * oc, axis=-1) - r * r
        disc = b * b - 4.0 * c
        hit = (disc >= 0) & mask
        if not np.any(hit):
            continue
        sqrt_d = np.sqrt(np.maximum(disc[hit], 0.0))
        t = (-b[hit] - sqrt_d) * 0.5
        hit_t = t > 1e-6
        if not np.any(hit_t):
            continue
        rows, cols = np.where(hit)
        if decay_factor > 0.0:
            d = t - decay_range
            soft = d <= 0.0
            good = hit_t & ~soft
            result[rows[good], cols[good]] = True
        else:
            good = hit_t
            result[rows[good], cols[good]] = True
    return result


def _downsample_numpy(img_float, height, width, ssaa):
    pooled = img_float.reshape(height, ssaa, width, ssaa, 3).mean(axis=(1, 3))
    return np.clip(pooled, 0, 255).astype(np.uint8)


def render_scene(
    scene,
    camera: RayCamera,
    light_directions: np.ndarray,
    width: int,
    height: int,
    background: tuple = (25, 25, 25),
    ambient: float = 0.14,
    diffuse: float = 0.45,
    specular: float = 0.25,
    shininess: float = 40.0,
    depth_cue: bool = True,
    fog_start: float = 0.45,
    fog_intensity: float = 1.0,
    **kwargs
) -> np.ndarray:
    """Render a Scene object by extracting all renderable geometry.

    Handles ``points`` geometry (spheres) and ``mesh`` geometry (triangles).
    The result is passed to :func:`trace` with both sphere and triangle data.
    """
    spheres: list[Sphere] = []
    tri_vertices_list: list[np.ndarray] = []
    tri_vnormals_list: list[np.ndarray] = []
    tri_colors_list: list[np.ndarray] = []

    for obj in scene.objects:
        geom = obj.geometry
        if geom.kind == "points":
            positions = np.asarray(geom.positions, dtype=float)
            colors = np.asarray(geom.colors, dtype=float) if geom.colors is not None else None
            radii_arr = np.asarray(geom.radii, dtype=float) if geom.radii is not None else None
            meta_radius = geom.meta.get("radius", 0.5) if isinstance(geom.meta, dict) else 0.5
            for i in range(positions.shape[0]):
                r = float(radii_arr[i]) if radii_arr is not None else float(meta_radius)
                c = np.clip(colors[i, :3], 0.0, 1.0) if colors is not None else np.array([0.8, 0.8, 0.8])
                spheres.append(Sphere(center=positions[i], radius=r, color=c))

        elif geom.kind == "mesh":
            verts = np.asarray(geom.positions, dtype=float)
            norms = np.asarray(geom.normals, dtype=float) if geom.normals is not None else None
            cols = np.asarray(geom.colors, dtype=float) if geom.colors is not None else None
            idx = np.asarray(geom.indices, dtype=np.int32) if geom.indices is not None else None

            if idx is not None and idx.shape[1] == 3:
                # Indexed triangle mesh
                t = np.stack([verts[idx[:, 0]], verts[idx[:, 1]], verts[idx[:, 2]]], axis=1)
                if norms is not None and norms.shape == verts.shape:
                    tn = np.stack([norms[idx[:, 0]], norms[idx[:, 1]], norms[idx[:, 2]]], axis=1)
                elif norms is not None:
                    tn = np.tile(norms, (idx.shape[0], 1, 1))
                else:
                    tn = np.zeros_like(t)
                if cols is not None and cols.shape[0] == verts.shape[0]:
                    tc = cols[idx[:, 0], :3] if cols.ndim == 2 else cols[idx[:, 0]]
                elif cols is not None:
                    tc = cols[:3] if cols.ndim == 1 else cols[0, :3]
                else:
                    tc = np.full((idx.shape[0], 3), 0.8)
                tri_vertices_list.append(t)
                tri_vnormals_list.append(tn)
                tri_colors_list.append(tc)

            else:
                # Non-indexed triangles (N*3 vertices in triangle order)
                n_tris = verts.shape[0] // 3
                t = verts.reshape(n_tris, 3, 3)
                if norms is not None:
                    tn = norms.reshape(n_tris, 3, 3)
                else:
                    tn = np.zeros((n_tris, 3, 3), dtype=float)
                if cols is not None:
                    tc = cols.reshape(n_tris, 3, 3)[:, 0, :3]
                else:
                    tc = np.full((n_tris, 3), 0.8)
                tri_vertices_list.append(t)
                tri_vnormals_list.append(tn)
                tri_colors_list.append(tc)

    # Concatenate all triangle data
    if tri_vertices_list:
        all_verts = np.concatenate(tri_vertices_list, axis=0)
        all_norms = np.concatenate(tri_vnormals_list, axis=0)
        all_cols = np.concatenate(tri_colors_list, axis=0)
    else:
        all_verts = np.zeros((0, 3, 3), dtype=float)
        all_norms = np.zeros((0, 3, 3), dtype=float)
        all_cols = np.zeros((0, 3), dtype=float)

    return trace(
        spheres=spheres,
        camera=camera,
        light_directions=light_directions,
        width=width,
        height=height,
        background=background,
        ambient=ambient,
        diffuse=diffuse,
        specular=specular,
        shininess=shininess,
        depth_cue=depth_cue,
        fog_start=fog_start,
        fog_intensity=fog_intensity,
        tri_vertices=all_verts,
        tri_vnormals=all_norms,
        tri_colors=all_cols,
        **kwargs
    )

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

    # forward points FROM target TOWARD camera (up-vector convention)
    up_vec = np.array([sin_t * sin_p, -cos_t * sin_p, cos_p], dtype=float)
    up_vec /= np.linalg.norm(up_vec)
    origin = target + up_vec * distance

    # The ray tracer forward is the REVERSE: from camera toward target
    forward = target - origin
    fnorm = np.linalg.norm(forward)
    if fnorm > 1e-9:
        forward /= fnorm
    else:
        forward = np.array([0.0, 0.0, 1.0], dtype=float)

    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    cam_up /= np.linalg.norm(cam_up)

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

    @_nb.njit(**_JIT_SPECDICT)
    def _jit_trace(
        centers: np.ndarray,
        radii: np.ndarray,
        col_rgb: np.ndarray,
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
        """JIT-compiled ray tracing kernel with multi-light, shadows, depth cue.

        Parameters are all plain numpy arrays / scalars so Numba can infer
        concrete types at compile time.
        """
        n_spheres: int = centers.shape[0]
        rw: int = int(width * ssaa)
        rh: int = int(height * ssaa)
        n_lights: int = light_dirs.shape[0]

        # ---- build camera basis --------------------------------------------
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

        # Allocate output image (float32 accumulator for downscale)
        out = np.zeros((height, width, 3), dtype=np.float64)
        accum = np.zeros((height, width), dtype=np.float64)

        bg_f = float(bg_r) / 255.0
        bg_g_f = float(bg_g) / 255.0
        bg_b_f = float(bg_b) / 255.0

        # Pre-fetch light directions into local arrays for faster access
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

        # ---- per-pixel (ssaa) loop -----------------------------------------
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

                # ---- trace spheres ------
                best_t = np.inf
                best_id = -1
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

                if best_id < 0:
                    continue  # background (accumulated as zero)

                # ---- hit point & normal ------
                hx = rox + dir_x * best_t
                hy = roy + dir_y * best_t
                hz = roz + dir_z * best_t

                nx = hx - centers[best_id, 0]
                ny = hy - centers[best_id, 1]
                nz = hz - centers[best_id, 2]
                nl = math.sqrt(nx * nx + ny * ny + nz * nz)
                if nl < 1e-9:
                    nl = 1.0
                nx /= nl
                ny /= nl
                nz /= nl

                # ---- view direction ------
                vx = cam_origin[0] - hx
                vy = cam_origin[1] - hy
                vz = cam_origin[2] - hz
                vl = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-9
                vx /= vl
                vy /= vl
                vz /= vl

                # ---- lighting ------
                reflect_sum = 0.0
                spec_sum = 0.0

                for li in range(n_lights):
                    lx = ldirs[li, 0]
                    ly = ldirs[li, 1]
                    lz = ldirs[li, 2]

                    # Shadow test
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

                    # Diffuse (reflect)
                    if lit > 0.0 and n_dot_l > 0.0:
                        reflect_sum += lit * pow(n_dot_l, reflect_power)

                    # Specular (Blinn-Phong half-vector)
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

                # Normalize reflect by number of lights
                reflect_norm = reflect_sum / float(max(n_lights, 1))

                # Direct head-on specular
                n_dot_v = nx * vx + ny * vy + nz * vz
                if n_dot_v < 0.0:
                    n_dot_v = 0.0
                if n_dot_v > 1.0:
                    n_dot_v = 1.0
                direct_cmp = pow(n_dot_v, direct_spec_power)

                # Legacy / new blend
                if legacy > 0.0:
                    n_dot_l0 = nx * ldirs[0, 0] + ny * ldirs[0, 1] + nz * ldirs[0, 2]
                    if n_dot_l0 < 0.0:
                        n_dot_l0 = 0.0
                    legacy_bright = ambient + diffuse * n_dot_l0
                else:
                    legacy_bright = 0.0

                # Brightness
                bright = ambient + diffuse * reflect_norm
                if legacy > 0.0:
                    bright = bright * (1.0 - legacy) + legacy_bright * legacy

                if bright < 0.0:
                    bright = 0.0
                if bright > 1.0:
                    bright = 1.0

                # Specular (excess)
                excess = direct_spec * direct_cmp + specular * spec_sum * spec_per_light
                if excess < 0.0:
                    excess = 0.0
                if excess > 1.0:
                    excess = 1.0

                cr = col_rgb[best_id, 0] * bright + excess
                cg = col_rgb[best_id, 1] * bright + excess
                cb = col_rgb[best_id, 2] * bright + excess

                # ---- depth cueing ------
                if depth_cue_enabled and far_clip > 0.0:
                    nd = best_t / far_clip
                    if nd > fog_start:
                        ffact = (nd - fog_start) / (1.0 - fog_start) * fog_intensity
                        if ffact > 1.0:
                            ffact = 1.0
                        if ffact > 0.0:
                            cr = cr * (1.0 - ffact) + bg_f * ffact
                            cg = cg * (1.0 - ffact) + bg_g_f * ffact
                            cb = cb * (1.0 - ffact) + bg_b_f * ffact

                oy = py // ssaa
                ox = px // ssaa
                out[oy, ox, 0] += cr
                out[oy, ox, 1] += cg
                out[oy, ox, 2] += cb
                accum[oy, ox] += 1.0

        # ---- downsample (alpha-premultiplied weighted average) ----
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
                    if r > 255:
                        r = 255
                    if g > 255:
                        g = 255
                    if b > 255:
                        b = 255
                    if r < 0:
                        r = 0
                    if g < 0:
                        g = 0
                    if b < 0:
                        b = 0
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
        """Return lit factor in [0, 1] for the given point and light direction.

        1.0 = fully lit, 0.0 = fully shadowed.
        When decay_factor > 0, shadows soften with distance from occluder.
        """
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
) -> np.ndarray:
    """Return an (H, W, 3) uint8 raytraced image of the spheres.

    Features:
    - Multi-light Phong shading with Blinn-Phong half-vector specular
    - Direct (head-on) specular highlight
    - Soft shadows with distance-based decay
    - Depth cueing (fog) for far objects
    - Gamma-corrected background and output
    - Per-channel color blend post-filter
    - Configurable super-sampling anti-aliasing

    Uses a Numba JIT kernel when Numba is available (typically 10-50×
    faster); falls back to the vectorized NumPy path otherwise.
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

    if not spheres:
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

    # Normalize light directions
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
            camera, lds, fov_rad,
            width, height, ssaa,
            background, ambient, diffuse, specular, shininess,
            direct_specular, direct_specular_power, reflect_power, legacy_lighting,
            shadow, shadow_fudge, shadow_decay_factor, shadow_decay_range,
            depth_cue, fog_start, fog_intensity,
            bg_r, bg_g, bg_b,
        )

    # ---- Color blend post-filter ----
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
    """Per-channel minimum blending post-filter (PyMOL style)."""
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
    best_t, best_id = _intersect_all_spheres_numpy(origins, dirs, centers, radii)
    hit_mask = best_id >= 0

    bg_rgb = np.array([bg_r, bg_g, bg_b], dtype=np.float64)
    img = np.full((rh, rw, 3), bg_rgb, dtype=np.float64)

    if not hit_mask.any():
        base = _downsample_numpy(img, height, width, ssaa)
        return base

    positions = origins + dirs * best_t[..., np.newaxis]
    hit_pos = positions[hit_mask]
    hit_centers = centers[best_id[hit_mask]]
    hit_colors = colors[best_id[hit_mask]]

    normals = hit_pos - hit_centers
    nlen = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9
    normals = normals / nlen

    view_v = camera.origin - hit_pos
    view_v = view_v / (np.linalg.norm(view_v, axis=-1, keepdims=True) + 1e-9)

    n_lights = light_dirs.shape[0]
    spec_per_light = 1.0 / pow(max(n_lights - 1, 1), 0.6)
    legacy = np.clip(legacy_lighting, 0.0, 1.0)
    bg_f = np.array([bg_r / 255.0, bg_g / 255.0, bg_b / 255.0], dtype=np.float64)

    # Accumulate lighting per hit pixel
    bright_arr = np.full(hit_mask.shape, ambient, dtype=np.float64)
    excess_arr = np.zeros(hit_mask.shape, dtype=np.float64)

    for li in range(n_lights):
        ld = light_dirs[li]

        # Shadow
        if shadow:
            occluded = _any_hit_numpy_with_decay(positions, ld, centers, radii, hit_mask,
                                                  shadow_fudge, shadow_decay_factor,
                                                  shadow_decay_range)
            lit = np.where(occluded, 0.0, 1.0)
        else:
            lit = np.ones(hit_mask.shape, dtype=np.float64)

        n_dot_l = np.clip(np.sum(normals * ld, axis=-1), 0.0, 1.0)

        # Diffuse
        diff_weight = lit[hit_mask] * (n_dot_l ** reflect_power)
        bright_arr[hit_mask] += diffuse * diff_weight / n_lights

        # Specular (Blinn-Phong)
        half = ld + view_v
        hn = np.linalg.norm(half, axis=-1, keepdims=True) + 1e-9
        half = half / hn
        n_dot_h = np.clip(np.sum(normals * half, axis=-1), 0.0, 1.0)
        spec_weight = lit[hit_mask] * (n_dot_h ** shininess)
        excess_arr[hit_mask] += specular * spec_weight * spec_per_light

    # Direct head-on specular
    n_dot_v = np.clip(np.sum(normals * view_v, axis=-1), 0.0, 1.0)
    direct_cmp = n_dot_v ** direct_specular_power
    excess_arr[hit_mask] += direct_specular * direct_cmp

    # Legacy blend
    if legacy > 0.0:
        n_dot_l0 = np.clip(np.sum(normals * light_dirs[0], axis=-1), 0.0, 1.0)
        legacy_bright = ambient + diffuse * n_dot_l0
        bright_arr[hit_mask] = (
            bright_arr[hit_mask] * (1.0 - legacy) + legacy_bright * legacy
        )

    bright_arr = np.clip(bright_arr, 0.0, 1.0)
    excess_arr = np.clip(excess_arr, 0.0, 1.0)

    # Apply to pixels
    for i in range(centers.shape[0]):
        m = best_id == i
        if not np.any(m):
            continue
        col = colors[i]
        img[m] = (col * bright_arr[m, np.newaxis] + excess_arr[m, np.newaxis]) * 255.0

    # Depth cueing
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


def _intersect_all_spheres_numpy(origins, dirs, centers, radii):
    best_t = np.full(origins.shape[:2], np.inf, dtype=np.float64)
    best_id = np.full(origins.shape[:2], -1, dtype=np.int32)
    for i in range(centers.shape[0]):
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
        best_t[rows[good], cols[good]] = t[good]
        best_id[rows[good], cols[good]] = i
    return best_t, best_id


def _any_hit_numpy_with_decay(
    origins, light_dir, centers, radii, mask,
    fudge, decay_factor, decay_range,
):
    """Return boolean array: True if point is fully shadowed (decay gives < 0.99 lit)."""
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

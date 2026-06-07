from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import BaseCmd


class ExportMixin(BaseCmd):
    """Image and data export commands."""

    def _mixin_commands(self):
        return {
            "png": self._cmd_png,
            "ray": self._cmd_ray,
        }

    def _cmd_png(self, args: List[str]) -> None:
        """Save the current live OpenGL viewport as a PNG file."""

        joined = " ".join(args).strip()
        if not joined:
            self._emit_error("Usage: png filename [, width [, height [, dpi [, ray]]]]")
            return

        parts = [part.strip() for part in joined.split(",") if part.strip()]
        filename = parts[0]
        width = self._parse_optional_int(parts, 1)
        height = self._parse_optional_int(parts, 2)
        ray = self._parse_optional_bool(parts, 4)

        if ray:
            self._cmd_ray([str(width or 0), str(height or 0)])

        path = Path(filename).expanduser()
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        try:
            ok = bool(viewer.save_png(path, width=width, height=height))
        except AttributeError:
            ok = self._save_png_from_renderer(viewer, path, width=width, height=height)
        except Exception as exc:
            self._emit_error(f"Failed to write PNG {path}: {exc}")
            return

        if ok:
            self._emit_message(f"Wrote PNG: {path}")
        else:
            self._emit_error(f"Failed to write PNG {path}")

    def _cmd_ray(self, args: List[str]) -> None:
        """Ray-trace the current scene.

        PyMOL syntax: ray [width, [height]] or ray filename, width, height
        Default output: chimol_ray_YYYYMMDD_HHMMSS.png
        """

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        default_w, default_h = 800, 600
        width = default_w
        height = default_h

        joined = " ".join(args).strip()
        output_path: Optional[str] = None

        if joined:
            if "," in joined:
                parts = [p.strip() for p in joined.split(",")]
                output_path = parts[0] if parts[0] else None
                nums = []
                for p in parts[1:]:
                    try:
                        nums.append(int(float(p.strip())))
                    except (ValueError, TypeError):
                        continue
            else:
                nums = []
                for tok in joined.split():
                    try:
                        nums.append(int(float(tok)))
                    except (ValueError, TypeError):
                        continue
        else:
            nums = []

        if len(nums) >= 2 and nums[0] > 0 and nums[1] > 0:
            width, height = nums[0], nums[1]
        elif len(nums) >= 1 and nums[0] > 0:
            width = nums[0]
            height = max(1, int(width * 0.75))

        try:
            sphere_data = getattr(viewer, "get_atom_sphere_data", None)
            view_state_func = getattr(viewer, "get_ray_view_state", None)
        except Exception:
            sphere_data = None
            view_state_func = None

        if sphere_data is None or view_state_func is None:
            self._emit_message("ray: viewer does not support ray tracing")
            return

        try:
            positions, colors_rgb, radii = sphere_data()
            view = view_state_func()
        except Exception as exc:
            self._emit_error(f"ray: failed to extract scene data: {exc}")
            return

        if positions.shape[0] == 0:
            self._emit_message("ray: no atom spheres visible; nothing to trace")
            return

        from ..renderer.raytracer import Sphere, RayCamera, _camera_from_view_state, trace
        from ..config import _DISPLAY_CONFIG

        camera = _camera_from_view_state(view)

        spheres = []
        for i in range(positions.shape[0]):
            spheres.append(Sphere(
                center=positions[i],
                radius=float(radii[i]) if i < len(radii) else 1.0,
                color=np.clip(colors_rgb[i], 0.0, 1.0),
            ))

        try:
            dist = np.linalg.norm(positions - camera.origin, axis=1)
            safe_far = float(np.nanmax(dist + radii)) * 1.2
            if np.isfinite(safe_far) and safe_far > camera.far_clip:
                camera.far_clip = safe_far
        except Exception:
            pass

        ray_cfg = _DISPLAY_CONFIG.get("ray", {})
        light_cfg = _DISPLAY_CONFIG.get("lighting", {})

        # Legacy single light direction (fallback)
        legacy_light_dir = np.asarray(
            light_cfg.get("light_direction", [0.0, 0.0, 1.0]),
            dtype=float,
        )
        # New multi-light from ray config
        light_dirs = ray_cfg.get("light_directions", [[0.0, 0.0, 1.0]])
        light_dirs_arr = np.asarray(light_dirs, dtype=float)
        if light_dirs_arr.ndim == 1:
            light_dirs_arr = light_dirs_arr.reshape(1, 3)

        ambient = float(ray_cfg.get("ambient", 0.14))
        diffuse = float(ray_cfg.get("diffuse", 0.45))
        specular = float(ray_cfg.get("specular", 0.25))
        shininess = float(ray_cfg.get("shininess", 40.0))
        direct_specular = float(ray_cfg.get("direct_specular", 0.30))
        direct_specular_power = float(ray_cfg.get("direct_specular_power", 55.0))
        reflect_power = float(ray_cfg.get("reflect_power", 1.0))
        legacy_lighting = float(ray_cfg.get("legacy_lighting", 0.0))
        ssaa_val = int(ray_cfg.get("antialias", 2))
        gamma = float(ray_cfg.get("gamma", 2.2))
        shadow_enabled = bool(ray_cfg.get("shadow", True))
        shadow_fudge = float(ray_cfg.get("shadow_fudge", 0.001))
        shadow_decay_factor = float(ray_cfg.get("shadow_decay_factor", 0.2))
        shadow_decay_range = float(ray_cfg.get("shadow_decay_range", 1.8))
        depth_cue = bool(ray_cfg.get("depth_cue", True))
        fog_start = float(ray_cfg.get("fog_start", 0.45))
        fog_intensity = float(ray_cfg.get("fog_intensity", 1.0))
        color_blend = bool(ray_cfg.get("color_blend", True))
        color_blend_red = float(ray_cfg.get("color_blend_red", 0.17))
        color_blend_green = float(ray_cfg.get("color_blend_green", 0.25))
        color_blend_blue = float(ray_cfg.get("color_blend_blue", 0.14))

        bg_color = _DISPLAY_CONFIG.get("background", "k")
        bg_rgb = self._parse_background(bg_color)

        self._emit_message(f"ray: tracing {len(spheres)} spheres at {width}x{height} ...")

        try:
            image = trace(
                spheres=spheres,
                camera=camera,
                light_directions=light_dirs_arr,
                width=width,
                height=height,
                background=bg_rgb,
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
                shininess=shininess,
                ssaa=ssaa_val,
                direct_specular=direct_specular,
                direct_specular_power=direct_specular_power,
                reflect_power=reflect_power,
                legacy_lighting=legacy_lighting,
                shadow=shadow_enabled,
                shadow_fudge=shadow_fudge,
                shadow_decay_factor=shadow_decay_factor,
                shadow_decay_range=shadow_decay_range,
                gamma=gamma,
                depth_cue=depth_cue,
                fog_start=fog_start,
                fog_intensity=fog_intensity,
                color_blend=color_blend,
                color_blend_red=color_blend_red,
                color_blend_green=color_blend_green,
                color_blend_blue=color_blend_blue,
            )
        except Exception as exc:
            self._emit_error(f"ray: trace failed: {exc}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path:
            out_path = Path(output_path).expanduser()
        else:
            out_path = Path(f"chimol_ray_{timestamp}.png")
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")

        try:
            from PIL import Image
            img = Image.fromarray(image)
            img.save(str(out_path), "PNG")
            self._emit_message(f"ray: wrote {out_path} ({width}x{height})")
        except Exception as exc:
            self._emit_error(f"ray: failed to save image: {exc}")

        try:
            if window is not None:
                window._update_sequence_view()
        except Exception:
            pass

    def _parse_background(self, spec) -> tuple:
        if isinstance(spec, str):
            spec = spec.strip().lower()
            if spec in ("k", "black"):
                return (0, 0, 0)
            if spec in ("w", "white"):
                return (255, 255, 255)
        try:
            items = np.asarray(spec, dtype=float).flatten()
            if len(items) == 3:
                return tuple(int(max(0, min(255, x * 255))) for x in items)
        except Exception:
            pass
        return (0, 0, 0)

    def _save_png_from_renderer(
        self,
        viewer,
        path: Path,
        *,
        width: Optional[int],
        height: Optional[int],
    ) -> bool:
        renderer = getattr(viewer, "_renderer", None)
        if renderer is None:
            return False
        widget = renderer.widget() if hasattr(renderer, "widget") else renderer
        grab = getattr(widget, "grabFramebuffer", None)
        if not callable(grab):
            return False
        if width or height:
            self._emit_message(
                "png: width/height currently use the live viewport; offscreen sizing is not implemented"
            )
        image = grab()
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        return bool(image.save(str(path), "PNG"))

    def _parse_optional_int(self, parts: List[str], index: int) -> Optional[int]:
        if index >= len(parts):
            return None
        text = parts[index].strip()
        if not text:
            return None
        if "=" in text:
            _, text = text.split("=", 1)
            text = text.strip()
        try:
            value = int(float(text))
        except Exception:
            return None
        return value if value > 0 else None

    def _parse_optional_bool(self, parts: List[str], index: int) -> bool:
        if index >= len(parts):
            return False
        text = parts[index].strip().lower()
        if "=" in text:
            _, text = text.split("=", 1)
            text = text.strip().lower()
        return text in ("1", "true", "yes", "on")

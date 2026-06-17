from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional
import tempfile
import time

import numpy as np

from qtpy import QtCore, QtWidgets

from .base import BaseCmd


class RayRenderThread(QtCore.QThread):
    """Background thread for Chimol ray-tracing.

    Runs a render callable that returns an (H, W, 3) uint8 image and emits
    the result (or an error message) back to the GUI thread.
    """

    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        render_func: Callable[[], np.ndarray],
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._render_func = render_func

    def run(self) -> None:
        try:
            image = self._render_func()
            self.finished.emit(image)
        except Exception as exc:
            self.error.emit(str(exc))


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

        explicit_size = False
        if len(nums) >= 2 and nums[0] > 0 and nums[1] > 0:
            width, height = nums[0], nums[1]
            explicit_size = True
        elif len(nums) >= 1 and nums[0] > 0:
            width = nums[0]
            height = max(1, int(width * 0.75))
            explicit_size = True

        if not explicit_size:
            try:
                renderer = getattr(viewer, "_renderer", None)
                widget = renderer.widget() if renderer is not None and hasattr(renderer, "widget") else None
                if widget is not None and widget.width() > 0 and widget.height() > 0:
                    width, height = int(widget.width()), int(widget.height())
            except Exception:
                pass

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path:
            out_path = Path(output_path).expanduser()
        else:
            out_path = Path(tempfile.gettempdir()) / f"chimol_ray_{timestamp}.png"
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")

        grab_current = None
        if callable(grab_current):
            try:
                image = grab_current(width=width, height=height)
            except Exception:
                image = None
            if image is not None:
                try:
                    parent = out_path.parent
                    parent.mkdir(parents=True, exist_ok=True)
                    if image.save(str(out_path), "PNG"):
                        show_overlay = getattr(viewer, "show_ray_overlay", None)
                        if callable(show_overlay):
                            show_overlay(image)
                        self._emit_message(f"ray: wrote {out_path} ({width}x{height})")
                        return
                except Exception as exc:
                    self._emit_error(f"ray: failed to save image: {exc}")
                    return

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

        from ..renderer.raytracer import Sphere, RayCamera, _camera_from_view_state, render_scene, trace
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

        scene_func = getattr(viewer, "get_current_scene", None)
        use_scene_path = False
        scene = None
        if callable(scene_func):
            try:
                scene = scene_func()
            except Exception:
                scene = None
            if scene is not None and getattr(scene, "objects", None):
                use_scene_path = True

        progress = np.zeros(1, dtype=np.int64)
        cancel = np.zeros(1, dtype=np.int64)
        total_rows = height * max(1, ssaa_val)

        if use_scene_path:
            self._emit_message(f"ray: rendering current scene at {width}x{height} ...")

            def _render_scene() -> np.ndarray:
                return render_scene(
                    scene=scene,
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
                    progress=progress,
                    cancel=cancel,
                )

            render_func = _render_scene
        else:
            self._emit_message(f"ray: tracing {len(spheres)} spheres at {width}x{height} ...")

            def _render_trace() -> np.ndarray:
                return trace(
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
                    progress=progress,
                    cancel=cancel,
                )

            render_func = _render_trace

        self._render_ray_async(
            render_func=render_func,
            progress=progress,
            cancel=cancel,
            total_rows=total_rows,
            out_path=out_path,
            width=width,
            height=height,
            viewer=viewer,
            window=window,
        )

    def _render_ray_async(
        self,
        render_func: Callable[[], np.ndarray],
        progress: np.ndarray,
        cancel: np.ndarray,
        total_rows: int,
        out_path: Path,
        width: int,
        height: int,
        viewer: object,
        window: Optional[object],
    ) -> None:
        """Run *render_func* in a background thread and show a progress dialog.

        The dialog is modal to the Chimol window so the scene cannot be
        modified while rendering, but the application event loop stays alive.
        It displays a progress bar, an ETA, and a Cancel button.  When no real
        GUI window is available (e.g. tests), rendering falls back to the
        synchronous path.
        """
        parent = window if isinstance(window, QtWidgets.QWidget) else None
        if parent is None:
            # Headless / test context: run synchronously without a dialog.
            try:
                image = render_func()
            except Exception as exc:
                self._emit_error(f"ray: {exc}")
                return
            self._finish_ray(image, out_path, width, height, viewer, window)
            return

        dialog = QtWidgets.QProgressDialog(
            "Ray tracing...", "Cancel", 0, total_rows, parent,
        )
        dialog.setWindowTitle("Rendering")
        dialog.setWindowModality(QtCore.Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumSize(360, 100)

        start_time = time.time()
        timer = QtCore.QTimer(parent)
        timer.timeout.connect(
            lambda: self._update_ray_progress(
                dialog, progress, cancel, total_rows, start_time,
            )
        )
        timer.start(100)

        thread = RayRenderThread(render_func, parent=parent)
        thread.finished.connect(
            lambda image: self._on_ray_finished(
                image, cancel, out_path, width, height, viewer, window, dialog, timer, thread,
            )
        )
        thread.error.connect(
            lambda msg: self._on_ray_error(msg, dialog, timer, thread)
        )
        dialog.canceled.connect(lambda: self._on_ray_cancel(cancel, dialog))
        thread.start()

    def _update_ray_progress(
        self,
        dialog: QtWidgets.QProgressDialog,
        progress: np.ndarray,
        cancel: np.ndarray,
        total_rows: int,
        start_time: float,
    ) -> None:
        """Poll the shared progress counter and update the dialog + ETA.

        When the render thread reports real progress (NumPy path), we use it
        directly.  When it does not (Numba path — atomic-free to keep the
        hot loop fast), we fall back to a time-based estimate so the user
        still sees a moving bar and an ETA.
        """
        try:
            current = int(progress[0])
        except Exception:
            current = 0

        elapsed = time.time() - start_time
        if cancel[0] != 0:
            text = "Cancelling..."
            fraction = min(0.99, dialog.value() / total_rows) if total_rows else 0
        elif current > 0:
            current = min(current, total_rows)
            dialog.setValue(current)
            fraction = current / total_rows if total_rows > 0 else 0.0
            if fraction > 0.02:
                eta = elapsed / fraction - elapsed
                text = (
                    f"Ray tracing... {int(fraction * 100)}%  "
                    f"ETA: {max(0, int(eta))}s  (elapsed: {int(elapsed)}s)"
                )
            else:
                text = f"Ray tracing... {int(fraction * 100)}%  (elapsed: {int(elapsed)}s)"
        else:
            # Time-based estimate: assume ~150k SSAA-rows per second on a
            # modest machine; clamp to 99% so we never claim completion.
            est_total = max(1, total_rows) / 150_000.0
            fraction = min(0.99, elapsed / est_total) if est_total > 0 else 0
            dialog.setValue(int(fraction * total_rows))
            text = f"Ray tracing...  (elapsed: {int(elapsed)}s)"
        dialog.setLabelText(text)

    def _on_ray_cancel(
        self,
        cancel: np.ndarray,
        dialog: QtWidgets.QProgressDialog,
    ) -> None:
        """Signal the background thread to stop rendering."""
        cancel[0] = 1
        dialog.setLabelText("Cancelling...")

    def _finish_ray(
        self,
        image: object,
        out_path: Path,
        width: int,
        height: int,
        viewer: object,
        window: Optional[object],
    ) -> None:
        """Save the ray-traced image, show the overlay and emit a message."""
        try:
            if image is None:
                self._emit_error("ray: render returned no image")
                return
            from PIL import Image
            img = Image.fromarray(image)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_path), "PNG")
            show_overlay = getattr(viewer, "show_ray_overlay", None)
            if callable(show_overlay):
                try:
                    from qtpy import QtGui
                    qimg = QtGui.QImage(
                        image.data,
                        image.shape[1],
                        image.shape[0],
                        image.strides[0],
                        QtGui.QImage.Format_RGB888,
                    ).copy()
                    show_overlay(qimg)
                except Exception:
                    pass
            self._emit_message(f"ray: wrote {out_path} ({width}x{height})")
        except Exception as exc:
            self._emit_error(f"ray: failed to save image: {exc}")
        finally:
            try:
                if window is not None:
                    window._update_sequence_view()
            except Exception:
                pass

    def _on_ray_finished(
        self,
        image: object,
        cancel: np.ndarray,
        out_path: Path,
        width: int,
        height: int,
        viewer: object,
        window: Optional[object],
        dialog: QtWidgets.QProgressDialog,
        timer: QtCore.QTimer,
        thread: RayRenderThread,
    ) -> None:
        timer.stop()
        dialog.close()
        if cancel[0] != 0:
            self._emit_message("ray: cancelled")
        else:
            self._finish_ray(image, out_path, width, height, viewer, window)
        thread.deleteLater()
        dialog.deleteLater()
        timer.deleteLater()

    def _on_ray_error(
        self,
        msg: str,
        dialog: QtWidgets.QProgressDialog,
        timer: QtCore.QTimer,
        thread: RayRenderThread,
    ) -> None:
        timer.stop()
        dialog.close()
        self._emit_error(f"ray: {msg}")
        thread.deleteLater()
        dialog.deleteLater()
        timer.deleteLater()

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

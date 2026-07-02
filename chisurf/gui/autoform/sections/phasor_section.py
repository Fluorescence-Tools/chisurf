"""AutoForm ``phasor`` section: 2-D phasor density + universal semicircle + overlays.

Renders the per-pixel phasor cloud as a calibrated 2-D density image (g on the
horizontal axis, s on the vertical) with the universal semicircle overlaid, so
the plot is correctly oriented and never clips the cloud. The model's ``target``
returns the density array (axis 0 = g, axis 1 = s); the extent is read from the
model's ``PHASOR_G_RANGE`` / ``PHASOR_S_RANGE`` (fallback to the circle bounds).

The ``target`` is optional: with no density (e.g. the FRET/phasor *calculator*),
the section is a pure phasor plot. An optional ``overlays`` option names a model
method returning reference-geometry polylines/markers as
``[{"kind": "curve"|"scatter", "x": [...], "y": [...], "labels": [...],
"style": {...}}, ...]`` (as produced by the phasor toolkit's overlay helpers /
the ``phasor.overlays`` RPC method), drawn on top of the density and semicircle.

An optional per-frame **movie** is enabled with ``movie: true`` + a ``frames``
option naming a model method that returns a ``(n_frames, g_bins, s_bins)`` density
stack: the section then shows play/loop/stop/fps controls and a frame slider, and
animates the phasor density (semicircle + overlays stay fixed). ``movie_fps`` sets
the initial speed.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from .registry import register_section

logger = logging.getLogger(__name__)


@register_section("phasor")
class PhasorSectionWidget(QtWidgets.QWidget):
    """AutoForm section drawing a phasor density map + universal semicircle."""

    AUTOFORM_REFRESH = True
    _autoform_expanding = True

    def __init__(self, model, target: str, **options):
        super().__init__()
        self._model = model
        self._target = target
        self._overlays_source = options.get("overlays")
        self._overlay_items: list = []
        self._g_range = tuple(getattr(model, "PHASOR_G_RANGE", (-0.1, 1.1)))
        self._s_range = tuple(getattr(model, "PHASOR_S_RANGE", (-0.05, 0.7)))
        # optional per-frame movie: ``frames`` names a model method returning a
        # (n_frames, g_bins, s_bins) density stack; the density ``target`` (if any)
        # is used for the static frame-0 fallback.
        self._frames_source = options.get("frames")
        self._movie = bool(options.get("movie")) and bool(self._frames_source)
        self._movie_fps = int(options.get("movie_fps", 10))
        self._frames = None  # cached (n, g, s) density stack
        self._frame = 0
        self._playing = False
        self._play_timer = None
        self._play_btn = self._loop_btn = self._stop_btn = self._fps_spin = self._slider = None

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if self._movie:
            layout.addLayout(self._build_movie_bar())

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "g")
        self._plot.setLabel("left", "s")
        self._plot.setAspectLocked(True)
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        if options.get("title"):
            self._plot.setTitle(str(options["title"]))
        self._image = pg.ImageItem()
        self._plot.addItem(self._image)
        # Universal semicircle: centre (0.5, 0), radius 0.5.
        theta = np.linspace(0.0, np.pi, 256)
        self._plot.plot(
            0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta),
            pen=pg.mkPen((255, 255, 255, 200), width=1.5),
        )
        self._plot.setXRange(0.0, 1.0)
        self._plot.setYRange(0.0, 0.6)
        layout.addWidget(self._plot, 1)
        self.refresh()

    def refresh(self) -> None:
        """Re-render the density (if any) and the reference-geometry overlays."""
        if self._movie:
            self._reload_frames()
        self._refresh_density()
        self._refresh_overlays()

    def _current_density(self):
        """Return the 2-D density to draw: the current movie frame, or the static target."""
        if self._movie:
            stack = self._frames
            if stack is not None and stack.ndim == 3 and 0 <= self._frame < stack.shape[0]:
                return stack[self._frame]
            return None
        source = getattr(self._model, self._target, None) if self._target else None
        if not callable(source):
            return None
        try:
            return source()
        except Exception as exc:  # pragma: no cover - model-defined source
            logger.warning("PhasorSection: source %r failed: %s", self._target, exc)
            return None

    def _refresh_density(self) -> None:
        density = self._current_density()
        if density is None:
            self._image.clear()
            return
        density = np.asarray(density, dtype=float)
        self._image.setImage(density, autoLevels=True)
        g0, g1 = self._g_range
        s0, s1 = self._s_range
        self._image.setRect(QtCore.QRectF(g0, s0, g1 - g0, s1 - s0))

    # ── per-frame movie ────────────────────────────────────────────────
    def _build_movie_bar(self) -> QtWidgets.QHBoxLayout:
        """Build the play/loop/stop/fps + frame-slider bar for the phasor movie."""
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 0)
        self._play_btn = QtWidgets.QToolButton()
        self._play_btn.setText("▶")
        self._play_btn.setToolTip("Play the per-frame phasor plot")
        self._play_btn.clicked.connect(self._toggle_play)
        bar.addWidget(self._play_btn)
        self._loop_btn = QtWidgets.QToolButton()
        self._loop_btn.setText("🔁")
        self._loop_btn.setCheckable(True)
        self._loop_btn.setChecked(True)
        self._loop_btn.setToolTip("Loop playback (wrap around at the end)")
        bar.addWidget(self._loop_btn)
        self._stop_btn = QtWidgets.QToolButton()
        self._stop_btn.setText("⏹")
        self._stop_btn.setToolTip("Stop and return to the first frame")
        self._stop_btn.clicked.connect(self._on_stop)
        bar.addWidget(self._stop_btn)
        self._fps_spin = QtWidgets.QSpinBox()
        self._fps_spin.setRange(1, 120)
        self._fps_spin.setValue(self._movie_fps)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setToolTip("Playback speed (frames per second)")
        self._fps_spin.valueChanged.connect(self._on_fps)
        bar.addWidget(self._fps_spin)
        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setToolTip("Frame")
        self._slider.valueChanged.connect(self._on_slider)
        bar.addWidget(self._slider, 1)
        self._set_movie_enabled(False)
        return bar

    def _reload_frames(self) -> None:
        """Re-read the per-frame density stack and sync the slider range."""
        src = getattr(self._model, self._frames_source, None) if self._frames_source else None
        try:
            stack = np.asarray(src(), dtype=float) if callable(src) else None
        except Exception as exc:  # pragma: no cover - model-defined source
            logger.warning("PhasorSection: frames %r failed: %s", self._frames_source, exc)
            stack = None
        self._frames = stack if (stack is not None and stack.ndim == 3) else None
        n = self._frames.shape[0] if self._frames is not None else 0
        self._set_movie_enabled(n > 1)
        if self._slider is not None:
            self._slider.blockSignals(True)
            self._slider.setMaximum(max(0, n - 1))
            if self._frame >= n:
                self._frame = 0
            self._slider.setValue(self._frame)
            self._slider.blockSignals(False)

    def _set_movie_enabled(self, on: bool) -> None:
        for w in (self._play_btn, self._loop_btn, self._stop_btn, self._fps_spin, self._slider):
            if w is not None:
                w.setEnabled(bool(on))
        if not on:
            self._stop_play()

    def _on_slider(self, value: int) -> None:
        self._frame = int(value)
        self._refresh_density()

    def _toggle_play(self) -> None:
        self._stop_play() if self._playing else self._start_play()

    def _on_stop(self) -> None:
        self._stop_play()
        self._frame = 0
        if self._slider is not None:
            self._slider.setValue(0)

    def _start_play(self) -> None:
        fps = int(self._fps_spin.value()) if self._fps_spin else self._movie_fps
        if fps <= 0 or self._frames is None:
            return
        if self._play_timer is None:
            self._play_timer = QtCore.QTimer(self)
            self._play_timer.timeout.connect(self._advance_frame)
        self._play_timer.start(int(1000 / max(fps, 1)))
        self._playing = True
        if self._play_btn is not None:
            self._play_btn.setText("⏸")

    def _advance_frame(self) -> None:
        if self._frames is None:
            return
        n = int(self._frames.shape[0])
        if n <= 1:
            return
        idx = self._frame + 1
        loop = self._loop_btn.isChecked() if self._loop_btn is not None else True
        if idx >= n:
            if not loop:
                self._stop_play()
                return
            idx = 0
        if self._slider is not None:
            self._slider.setValue(idx)  # drives _on_slider -> redraw
        else:
            self._frame = idx
            self._refresh_density()

    def _stop_play(self) -> None:
        if self._play_timer is not None:
            self._play_timer.stop()
        self._playing = False
        if self._play_btn is not None:
            self._play_btn.setText("▶")

    def _on_fps(self, value: int) -> None:
        if self._playing:
            self._start_play()  # restart at the new rate

    def _refresh_overlays(self) -> None:
        for item in self._overlay_items:
            self._plot.removeItem(item)
        self._overlay_items = []
        source = (
            getattr(self._model, self._overlays_source, None)
            if self._overlays_source
            else None
        )
        if not callable(source):
            return
        try:
            overlays = source() or []
        except Exception as exc:  # pragma: no cover - model-defined source
            logger.warning("PhasorSection: overlays %r failed: %s", self._overlays_source, exc)
            return
        for ov in overlays:
            self._draw_overlay(ov)

    def _draw_overlay(self, ov: dict) -> None:
        style = ov.get("style", {}) or {}
        color = style.get("color", "w")
        pen = pg.mkPen(
            color,
            width=style.get("width", 1),
            style=QtCore.Qt.DashLine if style.get("dash") else QtCore.Qt.SolidLine,
        )
        x, y = ov.get("x", []), ov.get("y", [])
        if ov.get("kind") == "scatter":
            item = pg.ScatterPlotItem(
                x=x, y=y, pen=pg.mkPen(color), brush=pg.mkBrush(color),
                size=style.get("size", 8), symbol=style.get("symbol", "o"),
            )
            self._plot.addItem(item)
            self._overlay_items.append(item)
            for xi, yi, label in zip(x, y, ov.get("labels", [])):
                text = pg.TextItem(str(label), color=color, anchor=(0, 1))
                text.setPos(float(xi), float(yi))
                self._plot.addItem(text)
                self._overlay_items.append(text)
        else:
            item = self._plot.plot(x, y, pen=pen)
            self._overlay_items.append(item)

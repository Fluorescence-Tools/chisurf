"""Custom AutoForm sections for the TTTR Audifier tool.

The interactive parts that do not map to declarative controls are registered
here: the detector-definition page + file loader (``audifier_setup``), the
dynamic per-detector / per-channel mixing rows (``audifier_mix``) and the
waterfall image + audio transport (``audifier_waterfall``). The many audio and
waterfall parameters are plain built-in ``value``/``choice``/``toggle`` sections
in ``audifier.view.json``. The widgets own Qt concerns and drive the Qt-free
:class:`~..view_model.AudifierViewModel`. Imported (registered) by ``gui.tool``.
"""

from __future__ import annotations

import logging

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.autoform.sections.registry import register_section

logger = logging.getLogger(__name__)


def _tool_button(text: str, tooltip: str, slot) -> QtWidgets.QToolButton:
    """Build a configured ``QToolButton`` (emoji label + tooltip) in one call."""
    btn = QtWidgets.QToolButton()
    btn.setText(text)
    btn.setToolTip(tooltip)
    btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    btn.clicked.connect(slot)
    return btn


def _format_time(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


# ---------------------------------------------------------------------------
# audifier_setup — file loader + detector definition page
# ---------------------------------------------------------------------------


@register_section("audifier_setup")
def audifier_setup(model, target=None, **options):
    """AutoForm factory for the file loader + detector-definition page."""
    return _SetupSection(model)


class _SetupSection(QtWidgets.QWidget):
    """Load a TTTR file and define detectors, then push them into the model."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self.setAcceptDrops(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(_tool_button("📂 Load TTTR", "Load a TTTR file.", self._load))
        self._file_lbl = QtWidgets.QLabel("No file loaded")
        self._file_lbl.setStyleSheet("color: #888;")
        bar.addWidget(self._file_lbl, 1)
        bar.addWidget(
            _tool_button(
                "🔄 Update channels", "Rebuild channels/detectors from the setup.", self._update
            )
        )
        layout.addLayout(bar)

        from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage

        self._page = DetectorWizardPage(
            show_help=False,
            show_setups_file=True,
            show_setup_selection=True,
            show_tttr_reading=True,
            show_tables=True,
            show_add_inputs=True,
        )
        layout.addWidget(self._page, 1)

        self._model.add_observer(self._on_model_event)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if urls:
            self._load_path(urls[0].toLocalFile())
            event.acceptProposedAction()

    def _on_model_event(self, event: str) -> None:
        if event == "loaded":
            self._file_lbl.setText(self._model.input_file or "No file loaded")

    def _load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load TTTR File")
        if path:
            self._load_path(path)

    def _load_path(self, path: str) -> None:
        try:
            self._model.load(path)
            self._update()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load: {exc}")

    def _update(self) -> None:
        try:
            self._model.set_detectors_from_settings(self._page.get_settings())
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to read setup: {exc}")


# ---------------------------------------------------------------------------
# audifier_mix — dynamic detector + channel controls
# ---------------------------------------------------------------------------


@register_section("audifier_mix")
def audifier_mix(model, target=None, **options):
    """AutoForm factory for the per-detector and per-channel mixing rows."""
    return _MixSection(model)


class _MixSection(QtWidgets.QWidget):
    """Rebuildable detector (enable/color) and channel (chord/pitch/gain) rows."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        layout.addWidget(self._header("Detectors"))
        self._det_box = QtWidgets.QVBoxLayout()
        self._det_box.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._det_box)

        layout.addWidget(self._header("Channel notes"))
        self._ch_box = QtWidgets.QVBoxLayout()
        self._ch_box.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._ch_box)
        layout.addStretch(1)

        self._model.add_observer(self._on_model_event)
        self._rebuild()

    @staticmethod
    def _header(text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("font-weight: bold; padding: 2px;")
        return lbl

    def _on_model_event(self, event: str) -> None:
        if event == "channels":
            self._rebuild()

    @staticmethod
    def _clear(box: QtWidgets.QVBoxLayout) -> None:
        while box.count():
            item = box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def _rebuild(self) -> None:
        self._clear(self._det_box)
        self._clear(self._ch_box)
        for i, det in enumerate(self._model.detectors):
            self._det_box.addWidget(self._detector_row(i, det))
        for ch in self._model.channels:
            self._ch_box.addWidget(self._channel_row(ch))

    # ── detector row ────────────────────────────────────────────────────
    def _detector_row(self, index: int, det: dict) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        rl = QtWidgets.QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        chk = QtWidgets.QCheckBox(det["name"])
        chk.setChecked(bool(det["enabled"]))
        chk.toggled.connect(lambda v, i=index: self._model.set_detector(i, enabled=v))
        rl.addWidget(chk)
        color_btn = QtWidgets.QToolButton()
        color_btn.setText("🎨 Color")
        color_btn.setToolTip("Pick the detector's waterfall colour.")
        self._apply_btn_color(color_btn, det["color"])
        color_btn.clicked.connect(lambda _=False, i=index, b=color_btn: self._pick_color(i, b))
        rl.addWidget(color_btn)
        rl.addStretch(1)
        return row

    @staticmethod
    def _apply_btn_color(btn: QtWidgets.QToolButton, rgb) -> None:
        r, g, b = (int(c * 255) for c in rgb)
        btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: white; font-weight: bold;")

    def _pick_color(self, index: int, btn: QtWidgets.QToolButton) -> None:
        r, g, b = self._model.detectors[index]["color"]
        initial = QtGui.QColor.fromRgbF(r, g, b)
        color = QtWidgets.QColorDialog.getColor(initial, self)
        if color.isValid():
            rgb = color.getRgbF()[:3]
            self._model.set_detector(index, color=rgb)
            self._apply_btn_color(btn, rgb)

    # ── channel row ─────────────────────────────────────────────────────
    def _channel_row(self, ch: int) -> QtWidgets.QWidget:
        cfg = self._model.channel_configs.get(ch)
        row = QtWidgets.QWidget()
        rl = QtWidgets.QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        chk = QtWidgets.QCheckBox(f"Ch {ch}")
        chk.setChecked(self._model.channel_enabled.get(ch, True))
        chk.toggled.connect(lambda v, c=ch: self._model.set_channel(c, enabled=v))
        rl.addWidget(chk)

        rl.addWidget(QtWidgets.QLabel("Chord:"))
        combo = QtWidgets.QComboBox()
        combo.addItems(self._model.CHORD_TYPES)
        if cfg is not None and cfg.chord_type in self._model.CHORD_TYPES:
            combo.setCurrentText(cfg.chord_type)
        combo.currentTextChanged.connect(lambda t, c=ch: self._model.set_channel(c, chord_type=t))
        rl.addWidget(combo)

        rl.addWidget(QtWidgets.QLabel("Pitch:"))
        pitch = QtWidgets.QDoubleSpinBox()
        pitch.setRange(-24, 24)
        pitch.setValue(cfg.pitch_semitones if cfg else 0.0)
        pitch.valueChanged.connect(lambda v, c=ch: self._model.set_channel(c, pitch_semitones=v))
        rl.addWidget(pitch)

        rl.addWidget(QtWidgets.QLabel("Gain:"))
        gain = QtWidgets.QDoubleSpinBox()
        gain.setRange(0, 10)
        gain.setValue(cfg.gain if cfg else 1.0)
        gain.valueChanged.connect(lambda v, c=ch: self._model.set_channel(c, gain=v))
        rl.addWidget(gain)
        rl.addStretch(1)
        return row


# ---------------------------------------------------------------------------
# audifier_transport — waterfall update + audio transport
# ---------------------------------------------------------------------------
# The waterfall image itself is the general AutoForm ``waterfall`` section
# (source: ``waterfall_payload``); this bar only recomputes it and drives audio.


@register_section("audifier_transport")
def audifier_transport(model, target=None, **options):
    """AutoForm factory for the waterfall-update + audio transport bar."""
    return _TransportSection(model)


class _TransportSection(QtWidgets.QWidget):
    """Update-waterfall + play/pause/stop/revert/save-WAV controls."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model

        from ..sound_playback import SoundPlayer

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(_tool_button("🔄 Update", "Recompute the waterfall.", self._update))
        bar.addWidget(_tool_button("▶️ Play", "Synthesize and play audio.", self._play))
        self._btn_pause = _tool_button("⏸️ Pause", "Pause playback.", self._pause)
        self._btn_stop = _tool_button("⏹️ Stop", "Stop playback.", self._stop)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        bar.addWidget(self._btn_pause)
        bar.addWidget(self._btn_stop)
        bar.addWidget(_tool_button("⏪ Revert", "Revert to start.", self._revert))
        bar.addWidget(_tool_button("💾 WAV", "Save audio as WAV.", self._save_wav))
        self._pos = QtWidgets.QLabel("00:00 / 00:00")
        bar.addWidget(self._pos)
        bar.addStretch(1)
        layout.addLayout(bar)

        self._info = QtWidgets.QLabel("")
        self._info.setStyleSheet("color: #888;")
        layout.addWidget(self._info)

        self._player = SoundPlayer(self)
        self._player.position_changed.connect(self._on_position)
        self._player.state_changed.connect(self._on_state)
        self._player.error_occurred.connect(
            lambda msg: QtWidgets.QMessageBox.warning(self, "Playback Error", msg)
        )

    # ── waterfall ───────────────────────────────────────────────────────
    def _update(self) -> None:
        if self._model.data is None:
            return
        try:
            payload = self._model.compute_waterfall()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to update: {exc}")
            return
        self._info.setText(payload["info"] if payload else "No detectors enabled")
        self._model.notify("waterfall")

    # ── transport ───────────────────────────────────────────────────────
    def _play(self) -> None:
        reason = self._model.can_render()
        if reason is not None:
            QtWidgets.QMessageBox.warning(self, "Error", reason)
            return
        try:
            wav, _duration = self._model.build_audio()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to play: {exc}")
            return
        if self._player.load_audio(wav, self._model.sample_rate):
            self._player.play()

    def _pause(self) -> None:
        self._player.pause()

    def _stop(self) -> None:
        self._player.stop()

    def _revert(self) -> None:
        self._player.revert()

    def _save_wav(self) -> None:
        reason = self._model.can_render()
        if reason is not None:
            QtWidgets.QMessageBox.warning(self, "Error", reason)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save WAV File", "", "WAV Files (*.wav)"
        )
        if not path:
            return
        try:
            self._model.save_wav(path)
            QtWidgets.QMessageBox.information(self, "Success", f"WAV saved to {path}")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save: {exc}")

    def _on_position(self, current: float, duration: float) -> None:
        self._pos.setText(f"{_format_time(current)} / {_format_time(duration)}")
        payload = self._model.waterfall_payload()
        n_macro = payload["n_macro_bins"] if payload else 0
        if n_macro > 0 and duration > 0:
            self._model.waterfall_position = (current / duration) * n_macro
            self._model.notify("position")

    def _on_state(self, state: str) -> None:
        playing = state == "playing"
        self._btn_pause.setEnabled(playing)
        self._btn_stop.setEnabled(playing)
        if not playing:
            self._model.waterfall_position = None
            self._model.notify("position")


__all__ = ["audifier_setup", "audifier_mix", "audifier_transport"]

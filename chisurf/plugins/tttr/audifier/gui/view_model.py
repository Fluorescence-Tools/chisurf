"""Qt-free view-model backing the TTTR Audifier tool.

:class:`AudifierViewModel` holds the audio- and waterfall-synthesis parameters
that AutoForm binds its controls to, the loaded :class:`~..core.TTTRData`, and
the detector/channel mixing state, and orchestrates the (Qt-free) synthesis and
waterfall computations in :mod:`..core` / :mod:`..lifetime_analysis`. Colors are
kept as ``(r, g, b)`` float tuples (no ``QColor``) so the whole model is free of
Qt and unit-testable headlessly. The detector page, per-channel/detector
controls, the waterfall image and the audio transport are Qt custom sections that
edit this model.

Mirrors the other TTTR view-models (ALEX, Count Rate).
"""

from __future__ import annotations

import colorsys
import dataclasses
import logging
import pathlib
from collections.abc import Callable

import numpy as np

from ..core import (
    CHORD_TYPE_NAMES,
    ChannelConfig,
    compute_microtime_waterfall,
    load_tttr_with_tttrlib,
    make_default_channel_cfg,
    tttr_to_wav,
)

logger = logging.getLogger(__name__)

_VIEW_JSON = pathlib.Path(__file__).parent / "audifier.view.json"

try:
    from ..lifetime_analysis import compute_lifetime_waterfall
except Exception:  # pragma: no cover
    compute_lifetime_waterfall = None


class AudifierViewModel:
    """State + logic for the TTTR Audifier tool (no Qt)."""

    CHORD_TYPES = list(CHORD_TYPE_NAMES)

    def view_spec(self):
        """Resolve AutoForm's view spec from the authored ``audifier.view.json``."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_VIEW_JSON)

    def __init__(self) -> None:
        # ── AutoForm-bound audio parameters ─────────────────────────────
        self.bin_width = 0.02
        self.env_mode = "log"
        self.sample_rate = 44100
        self.master_gain = 0.8
        self.env_floor = 0.0
        self.env_scale = 1.0
        self.attack_frames = 2
        self.release_frames = 6

        # ── AutoForm-bound waterfall parameters ─────────────────────────
        self.waterfall_mode = "microtime"
        self.wf_bin_width = 0.05
        self.wf_micro_bins = 256
        self.wf_log = True
        self.lt_bin_width = 0.05
        self.lt_tau_min = 0.5
        self.lt_tau_max = 15.0
        self.lt_n_tau = 100
        self.lt_reg = 1e-2
        self.lt_log_tau = True
        self.lt_log_amp = True

        # ── runtime state (single source of truth for the mixing UI) ────
        self.input_file = ""
        self.data = None
        self.channels: list[int] = []
        #: detector dicts ``{name, channels, color(r,g,b), enabled}``.
        self.detectors: list[dict] = []
        #: per-channel synthesis config.
        self.channel_configs: dict[int, ChannelConfig] = {}
        #: per-channel enable flag for playback/export.
        self.channel_enabled: dict[int, bool] = {}
        #: last computed waterfall payload (read by the AutoForm ``waterfall`` section).
        self._waterfall_payload: dict | None = None
        #: current macro-bin position for the waterfall indicator during playback.
        self.waterfall_position: float | None = None
        self._observers: list[Callable[[str], None]] = []

    # ── observer hook ──────────────────────────────────────────────────
    def add_observer(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to be called with an event name on every change."""
        self._observers.append(cb)

    def notify(self, event: str = "changed") -> None:
        """Notify observers that state changed."""
        for cb in list(self._observers):
            try:
                cb(event)
            except Exception:
                logger.debug("audifier observer failed", exc_info=True)

    def update(self) -> None:
        """AutoForm hook after a bound field changes (no-op; sections pull state)."""

    # ── AutoForm options sources ────────────────────────────────────────
    def chord_options(self) -> list[str]:
        """Available chord-quality names for a channel."""
        return self.CHORD_TYPES

    @property
    def has_data(self) -> bool:
        """Whether a TTTR file is loaded."""
        return self.data is not None

    # ── loading ─────────────────────────────────────────────────────────
    def load(self, path: str) -> None:
        """Load a TTTR file and refresh derived state."""
        self.data = load_tttr_with_tttrlib(path)
        self.input_file = path
        self.notify("loaded")

    # ── detector / channel mixing state ─────────────────────────────────
    def set_detectors_from_settings(self, settings: dict) -> None:
        """Rebuild detectors + channels from a DetectorWizardPage settings dict."""
        dets = settings.get("detectors", {})
        self.detectors = []
        for i, (name, det) in enumerate(dets.items()):
            hue = ((i * 60) % 360) / 360.0
            self.detectors.append(
                {
                    "name": name,
                    "channels": list(det.get("chs", [])),
                    "color": colorsys.hsv_to_rgb(hue, 0.78, 1.0),
                    "enabled": True,
                }
            )
        self.channels = sorted({ch for d in self.detectors for ch in d["channels"]})
        defaults = make_default_channel_cfg(self.channels) if self.channels else {}
        self.channel_configs = dict(defaults)
        self.channel_enabled = {ch: True for ch in self.channels}
        self.notify("channels")

    def set_detector(self, index: int, *, enabled: bool | None = None, color=None) -> None:
        """Update the enable flag / color of detector *index*."""
        if not (0 <= index < len(self.detectors)):
            return
        if enabled is not None:
            self.detectors[index]["enabled"] = bool(enabled)
        if color is not None:
            self.detectors[index]["color"] = tuple(color)

    def set_channel(
        self,
        ch: int,
        *,
        chord_type: str | None = None,
        pitch_semitones: float | None = None,
        gain: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Update the synthesis config / enable flag of channel *ch*."""
        cfg = self.channel_configs.get(ch)
        if cfg is None:
            return
        changes = {}
        if chord_type is not None:
            changes["chord_type"] = chord_type
        if pitch_semitones is not None:
            changes["pitch_semitones"] = float(pitch_semitones)
        if gain is not None:
            changes["gain"] = float(gain)
        if changes:
            self.channel_configs[ch] = dataclasses.replace(cfg, **changes)
        if enabled is not None:
            self.channel_enabled[ch] = bool(enabled)

    def selected_channels(self) -> list[int]:
        """Channels currently enabled for playback / export."""
        return [ch for ch in self.channels if self.channel_enabled.get(ch, False)]

    def selected_configs(self) -> dict[int, ChannelConfig]:
        """``{channel: ChannelConfig}`` for the enabled channels."""
        return {
            ch: self.channel_configs[ch]
            for ch in self.selected_channels()
            if ch in self.channel_configs
        }

    # ── waterfall ───────────────────────────────────────────────────────
    def waterfall_payload(self) -> dict | None:
        """Return the last computed waterfall payload (AutoForm ``waterfall`` source)."""
        return self._waterfall_payload

    def compute_waterfall(self) -> dict | None:
        """Compute the RGB waterfall for the enabled detectors and current mode.

        The result is cached and returned; it is a *new* dict each call so the
        ``waterfall`` section knows to re-upload the image.
        """
        payload = None
        if self.data is not None:
            enabled = [d for d in self.detectors if d.get("enabled")]
            if enabled:
                payload = (
                    self._lifetime_waterfall(enabled)
                    if self.waterfall_mode == "lifetime"
                    else self._microtime_waterfall(enabled)
                )
        self._waterfall_payload = payload
        return payload

    def _microtime_waterfall(self, dets: list[dict]) -> dict:
        w0, macro_t_s, micro_centers = compute_microtime_waterfall(
            data=self.data,
            channels=dets[0]["channels"],
            macro_bin_width_s=self.wf_bin_width,
            n_micro_bins=self.wf_micro_bins,
        )
        n_macro, n_micro = w0.shape
        rgb = np.zeros((n_macro, n_micro, 3), dtype=np.float32)
        total = np.zeros((n_macro, n_micro), dtype=np.float32)
        for det in dets:
            w_det, _, _ = compute_microtime_waterfall(
                data=self.data,
                channels=det["channels"],
                macro_bin_width_s=self.wf_bin_width,
                n_micro_bins=self.wf_micro_bins,
            )
            if self.wf_log:
                w_det = np.log1p(w_det)
            total += w_det
            r, g, b = det["color"]
            rgb[:, :, 0] += w_det * r
            rgb[:, :, 1] += w_det * g
            rgb[:, :, 2] += w_det * b
        mask = total > 0
        rgb[mask] /= total[mask, np.newaxis]
        return {
            "rgb_data": rgb.transpose(1, 0, 2),
            "macro_t_s": macro_t_s,
            "micro_centers": micro_centers,
            "n_macro_bins": n_macro,
            "n_micro_bins": n_micro,
            "title": "Microtime Waterfall",
            "x_label": None,
            "log_x": False,
            "info": f"Microtime Waterfall: {n_macro} macro bins, {n_micro} micro bins",
        }

    def _lifetime_waterfall(self, dets: list[dict]) -> dict | None:
        if compute_lifetime_waterfall is None:
            raise RuntimeError("Lifetime analysis not available.")
        chans = [(ch, det) for det in dets for ch in det["channels"]]
        if not chans:
            return None
        per_channel = {}
        for ch, det in chans:
            try:
                a, macro_t_s, tau = compute_lifetime_waterfall(
                    data=self.data,
                    channel=ch,
                    macro_bin_width_s=self.lt_bin_width,
                    micro_gate=None,
                    tau_min=self.lt_tau_min * 1e-9,
                    tau_max=self.lt_tau_max * 1e-9,
                    n_tau=self.lt_n_tau,
                    lam=self.lt_reg,
                )
                per_channel[ch] = (a, macro_t_s, tau, det)
            except Exception:
                logger.warning("lifetime waterfall failed for channel %s", ch, exc_info=True)
        if not per_channel:
            return None
        a_ref, macro_t_s, tau, _ = next(iter(per_channel.values()))
        n_macro, n_tau = a_ref.shape
        rgb = np.zeros((n_macro, n_tau, 3), dtype=np.float32)
        total = np.zeros((n_macro, n_tau), dtype=np.float32)
        for a, _, _, det in per_channel.values():
            a_plot = np.log1p(a) if self.lt_log_amp else a.copy()
            total += a_plot
            r, g, b = det["color"]
            rgb[:, :, 0] += a_plot * r
            rgb[:, :, 1] += a_plot * g
            rgb[:, :, 2] += a_plot * b
        mask = total > 0
        rgb[mask] /= total[mask, np.newaxis]
        return {
            "rgb_data": rgb.transpose(1, 0, 2),
            "macro_t_s": macro_t_s,
            "micro_centers": tau,
            "n_macro_bins": n_macro,
            "n_micro_bins": n_tau,
            "title": "Lifetime Waterfall",
            "x_label": "Lifetime τ (s)",
            "log_x": bool(self.lt_log_tau),
            "info": f"Lifetime Waterfall: {n_macro} macro bins, {n_tau} lifetime points",
        }

    # ── audio ───────────────────────────────────────────────────────────
    def can_render(self) -> str | None:
        """Return ``None`` when audio can be rendered, else a reason string."""
        if self.data is None:
            return "No data loaded."
        if not self.selected_channels():
            return "No channels selected."
        if not self.selected_configs():
            return "No channels configured."
        return None

    def build_audio(self):
        """Synthesize audio for the enabled channels; ``(wav_data, duration)``."""
        from ..sound_playback import create_tttr_audio

        return create_tttr_audio(
            data=self.data,
            channels=self.selected_channels(),
            channel_cfg=self.selected_configs(),
            bin_width_s=self.bin_width,
            sample_rate=self.sample_rate,
            env_mode=self.env_mode,
            master_gain=self.master_gain,
        )

    def save_wav(self, path: str) -> None:
        """Render audio for the enabled channels and write it to *path* as WAV."""
        tttr_to_wav(
            data=self.data,
            out_wav_path=path,
            channels=self.selected_channels(),
            channel_cfg=self.selected_configs(),
            bin_width_s=self.bin_width,
            sample_rate=self.sample_rate,
            env_mode=self.env_mode,
            env_floor=self.env_floor,
            env_scale=self.env_scale,
            attack_frames=self.attack_frames,
            release_frames=self.release_frames,
            master_gain=self.master_gain,
        )


__all__ = ["AudifierViewModel"]

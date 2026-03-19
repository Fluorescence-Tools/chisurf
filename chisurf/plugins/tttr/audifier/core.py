"""
TTTR Audifier Core Module

Core logic for loading TTTR files and converting to audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import math
import wave
import struct

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class TTTRData:
    """
    Minimal in-memory representation of a TTTR photon stream.

    routing: integer channel per photon (aka routing channel)
    macro_ticks: integer macrotime ticks per photon
    micro_bins: integer microtime bin per photon
    macro_time_unit_s: seconds per macro tick
    micro_time_unit_s: seconds per microtime bin (optional; only for plotting labels)
    """
    routing: np.ndarray          # shape (N,), int
    macro_ticks: np.ndarray      # shape (N,), int
    micro_bins: np.ndarray       # shape (N,), int
    macro_time_unit_s: float
    micro_time_unit_s: Optional[float] = None


@dataclass(frozen=True)
class ChannelConfig:
    """
    Per routing-channel audio mapping & gating.

    note_hz: Base oscillator frequency for the channel (e.g., 261.63 for C4)
    pitch_semitones: Fine adjustment in semitones (+/-)
    micro_min: Inclusive microtime bin lower bound
    micro_max: Exclusive microtime bin upper bound
    gain: Linear gain multiplier for this channel
    """
    note_hz: float
    pitch_semitones: float = 0.0
    micro_min: int = 0
    micro_max: int = 2**31 - 1
    gain: float = 1.0


# -----------------------------
# TTTR loading (tttrlib)
# -----------------------------

def load_tttr_with_tttrlib(path: str) -> TTTRData:
    """
    Load a TTTR file using tttrlib.

    Adapted from ChiSurf usage.
    """
    try:
        import tttrlib  # type: ignore
    except Exception as e:
        raise ImportError(
            "tttrlib not available. Install or pass arrays via TTTRData."
        ) from e

    tttr = tttrlib.TTTR(path)

    # Get routing channel
    routing = np.asarray(tttr.routing_channels, dtype=np.int32)
    # Get macro time
    macro = np.asarray(tttr.macro_times, dtype=np.int64)
    # Get micro time
    micro = np.asarray(tttr.micro_times, dtype=np.int32)

    # Macro time unit
    macro_unit_s = getattr(tttr.header, 'macro_time_resolution', None)
    if macro_unit_s is None:
        macro_unit_s = getattr(tttr.header, 'macro_time_unit', None)
    if macro_unit_s is None:
        macro_unit_s = getattr(tttr.header, 'macro_time_calibration', None)
    if macro_unit_s is None:
        macro_unit_s = 1e-9  # default 1 ns
    else:
        if callable(macro_unit_s):
            macro_unit_s = macro_unit_s()
        macro_unit_s = float(macro_unit_s)

    # Micro time unit (optional)
    micro_unit_s = getattr(tttr.header, 'micro_time_resolution', None)
    if micro_unit_s is None:
        micro_unit_s = getattr(tttr.header, 'micro_time_unit', None)
    if micro_unit_s is None:
        micro_unit_s = getattr(tttr.header, 'micro_time_calibration', None)
    if micro_unit_s is not None:
        if callable(micro_unit_s):
            micro_unit_s = micro_unit_s()
        micro_unit_s = float(micro_unit_s)

    return TTTRData(
        routing=routing,
        macro_ticks=macro,
        micro_bins=micro,
        macro_time_unit_s=macro_unit_s,
        micro_time_unit_s=micro_unit_s,
    )


# -----------------------------
# Audio synthesis helpers
# -----------------------------

def _semitones_to_ratio(semitones: float) -> float:
    return 2.0 ** (semitones / 12.0)


def _write_wav_mono(path: str, y: np.ndarray, sample_rate: int) -> None:
    """
    Write mono float waveform [-1, 1] to 16-bit PCM WAV.
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, -1.0, 1.0)

    pcm = (y * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _smooth_envelope(x: np.ndarray, attack_samps: int, release_samps: int) -> np.ndarray:
    """
    Simple attack/release smoothing for an envelope sampled per audio frame.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)

    a = 1.0 - math.exp(-1.0 / max(1, attack_samps))
    r = 1.0 - math.exp(-1.0 / max(1, release_samps))

    for i in range(len(x)):
        if i == 0:
            y[i] = x[i]
        else:
            if x[i] > y[i - 1]:
                y[i] = y[i - 1] + a * (x[i] - y[i - 1])
            else:
                y[i] = y[i - 1] + r * (x[i] - y[i - 1])
    return y


# -----------------------------
# Core: binning + mapping photons → sound
# -----------------------------

def bin_photons(
    data: TTTRData,
    bin_width_s: float,
    channels: Iterable[int],
    channel_cfg: Dict[int, ChannelConfig],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin photons into macrotime bins, returning:
    - bin_edges_ticks (int64) length (B+1)
    - counts_per_bin_per_channel (float64) shape (B, C)

    Microtime gating is applied per channel according to channel_cfg.
    """
    channels = list(channels)
    if len(channels) == 0:
        raise ValueError("No channels selected.")

    macro = data.macro_ticks
    routing = data.routing
    micro = data.micro_bins

    t0 = int(macro.min())
    t1 = int(macro.max()) + 1

    ticks_per_bin = max(1, int(round(bin_width_s / data.macro_time_unit_s)))
    n_bins = int(math.ceil((t1 - t0) / ticks_per_bin))

    edges = (t0 + np.arange(n_bins + 1, dtype=np.int64) * ticks_per_bin)

    # Assign each photon to a macro bin index
    bin_idx = ((macro - t0) // ticks_per_bin).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    counts = np.zeros((n_bins, len(channels)), dtype=np.float64)

    # For each channel, gate microtime and count per bin
    for j, ch in enumerate(channels):
        cfg = channel_cfg[ch]
        mask = (routing == ch) & (micro >= cfg.micro_min) & (micro < cfg.micro_max)
        if not np.any(mask):
            continue
        b = bin_idx[mask]
        # fast bincount into bins
        counts[:, j] = np.bincount(b, minlength=n_bins).astype(np.float64)

    return edges, counts


def counts_to_envelopes(
    counts: np.ndarray,
    mode: str = "log",
    floor: float = 0.0,
    scale: float = 1.0,
    attack_frames: int = 2,
    release_frames: int = 6,
) -> np.ndarray:
    """
    Drop-in replacement.

    Key change: enforce SILENCE for baseline by robust baseline subtraction + thresholding.
    This eliminates the 'hum' when there is background in every bin.

    - Baseline: median(counts)
    - Noise: MAD (median absolute deviation) scaled to sigma
    - Threshold: baseline + k*sigma  (k chosen internally)
    """
    x = np.asarray(counts, dtype=np.float64)
    x = np.maximum(x, 0.0)

    n_frames, n_ch = x.shape
    env_out = np.zeros_like(x, dtype=np.float64)

    # Internal knobs (kept internal so signature stays identical)
    k_sigma = 4.0          # higher => more silence, only strong bursts pass
    min_thresh = 1.0       # at least 1 photon above baseline
    use_robust_ref = 99.0  # percentile for normalization

    for j in range(n_ch):
        c = x[:, j]
        if np.all(c <= 0):
            continue

        # Robust baseline + noise estimate
        baseline = np.median(c)
        mad = np.median(np.abs(c - baseline))
        sigma = 1.4826 * mad  # MAD -> sigma for Gaussian-like noise

        thresh = max(baseline + k_sigma * sigma, baseline + min_thresh)

        # Enforce true silence below threshold (this kills humming)
        c2 = c - thresh
        c2[c2 < 0] = 0.0

        # Compress dynamic range AFTER thresholding
        if mode == "linear":
            v = c2
        elif mode == "sqrt":
            v = np.sqrt(c2)
        elif mode == "log":
            v = np.log1p(c2)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Robust normalization (avoid one extreme burst flattening everything else)
        v_pos = v[v > 0]
        if v_pos.size == 0:
            continue
        ref = np.percentile(v_pos, use_robust_ref)
        ref = max(ref, 1e-12)

        env = np.clip(v / ref, 0.0, 1.0)

        # Respect your existing floor/scale
        env = np.maximum(env - floor, 0.0) * scale

        # Apply attack/release smoothing
        env = _smooth_envelope(env, attack_frames, release_frames)

        env_out[:, j] = np.clip(env, 0.0, 1.0)

    return env_out


def synthesize_audio_from_envelopes(
    envelopes: np.ndarray,
    frame_width_s: float,
    sample_rate: int,
    channels: List[int],
    channel_cfg: Dict[int, ChannelConfig],
    master_gain: float = 0.8,
) -> np.ndarray:
    """
    Render *non-continuous* audio: per-frame pings modulated by envelopes.
    Drop-in replacement for the original continuous sine mix.

    envelopes: shape (n_frames, n_channels) in [0,1]
    Produces silence when envelope == 0 and louder pings when envelope is high.
    """
    envelopes = np.asarray(envelopes, dtype=np.float64)
    if envelopes.ndim != 2:
        raise ValueError("envelopes must be 2D: (n_frames, n_channels)")

    n_frames = envelopes.shape[0]
    frame_samps = int(round(frame_width_s * sample_rate))
    if frame_samps <= 0:
        raise ValueError("frame_width_s too small for given sample_rate")

    total_samps = n_frames * frame_samps
    t = np.arange(total_samps, dtype=np.float64) / sample_rate

    # --- Per-frame ping shape (attack + exponential decay) ---
    # Make sure the ping is well-contained within the frame so you do NOT hear a continuous tone.
    # Decay scales with frame length to remain perceptually stable.
    attack_s = min(0.001, 0.25 * frame_width_s)                   # up to 1 ms
    decay_s  = min(0.010, 0.85 * frame_width_s)                   # up to 10 ms, but within frame
    attack_n = max(1, int(round(attack_s * sample_rate)))

    t_frame = np.arange(frame_samps, dtype=np.float64) / sample_rate
    ping_env = np.exp(-t_frame / max(decay_s, 1e-6))
    # linear attack ramp
    ping_env[:attack_n] *= np.linspace(0.0, 1.0, attack_n, endpoint=False)

    # repeat ping envelope per frame
    ping_env_rep = np.tile(ping_env, n_frames)[:total_samps]

    y = np.zeros(total_samps, dtype=np.float64)

    # --- Render each channel as a pitched ping train ---
    for j, ch in enumerate(channels):
        cfg = channel_cfg[ch]
        freq = cfg.note_hz * _semitones_to_ratio(cfg.pitch_semitones)

        # per-frame envelope expanded to samples
        a = np.repeat(envelopes[:, j], frame_samps)[:total_samps]

        # HARD silence for truly empty bins
        # (envelopes already should be 0 for empty bins, but guard numeric noise)
        a[a < 1e-6] = 0.0

        # random initial phase avoids "phasiness" when mixing channels
        phase0 = np.random.rand() * 2.0 * math.pi
        osc = np.sin(2.0 * math.pi * freq * t + phase0)

        # Apply per-frame ping envelope so sound is transient, not continuous
        sig = cfg.gain * a * ping_env_rep * osc

        # Optional: add a small amount of "click" (wideband) proportional to amplitude
        # This improves audibility of fast intensity changes without changing interface.
        # Keep it subtle.
        click_mix = 0.12
        if click_mix > 0:
            noise = np.random.randn(total_samps).astype(np.float64)
            sig = (1.0 - click_mix) * sig + click_mix * (cfg.gain * a * ping_env_rep * noise)

        y += sig

    # --- Normalize / prevent clipping ---
    peak = np.max(np.abs(y)) if total_samps > 0 else 1.0
    if peak > 0:
        y = (master_gain / max(1.0, peak)) * y

    return y.astype(np.float32)


def tttr_to_wav(
    data: TTTRData,
    out_wav_path: str,
    channels: List[int],
    channel_cfg: Dict[int, ChannelConfig],
    bin_width_s: float = 0.02,   # 20 ms “audio frame”
    sample_rate: int = 44100,
    env_mode: str = "log",
    env_floor: float = 0.0,
    env_scale: float = 1.0,
    attack_frames: int = 2,
    release_frames: int = 6,
    master_gain: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    End-to-end: bin photons → envelopes → synthesize → write WAV.

    Returns:
    - wav (float32) mono waveform
    - bin_edges_ticks (int64)
    - envelopes (float64) shape (n_bins, n_channels)
    """
    edges, counts = bin_photons(
        data=data,
        bin_width_s=bin_width_s,
        channels=channels,
        channel_cfg=channel_cfg,
    )
    env = counts_to_envelopes(
        counts=counts,
        mode=env_mode,
        floor=env_floor,
        scale=env_scale,
        attack_frames=attack_frames,
        release_frames=release_frames,
    )
    wav = synthesize_audio_from_envelopes(
        envelopes=env,
        frame_width_s=bin_width_s,
        sample_rate=sample_rate,
        channels=channels,
        channel_cfg=channel_cfg,
        master_gain=master_gain,
    )
    _write_wav_mono(out_wav_path, wav, sample_rate)
    return wav, edges, env


# -----------------------------
# Waterfall plot (macro bins × microtime histogram)
# -----------------------------

def compute_microtime_waterfall(
    data: TTTRData,
    channel: Optional[int] = None,
    macro_bin_width_s: float = 0.05,
    micro_bins_range: Optional[Tuple[int, int]] = None,
    n_micro_bins: int = 256,
    micro_gate: Optional[Tuple[int, int]] = None,
    channels: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Waterfall matrix W with shape (n_macro_bins, n_micro_bins_plot)
    where each row is a microtime histogram for a macrotime window.

    channel:
      - None => include all routing channels
      - int  => include only that routing channel

    micro_bins_range:
      - (min_bin, max_bin) plotted range; if None, inferred from data

    micro_gate:
      - optional gate (min_bin, max_bin) applied before plotting
    """
    macro = data.macro_ticks
    micro = data.micro_bins
    routing = data.routing

    mask = np.ones_like(macro, dtype=bool)
    if channel is not None:
        mask &= (routing == channel)

    if channels is not None:
        mask &= np.isin(routing, channels)

    if micro_gate is not None:
        g0, g1 = micro_gate
        mask &= (micro >= g0) & (micro < g1)

    macro = macro[mask]
    micro = micro[mask]

    if macro.size == 0:
        raise ValueError("No photons in the selected subset (channel/gate).")

    t0 = int(macro.min())
    t1 = int(macro.max()) + 1
    ticks_per_bin = max(1, int(round(macro_bin_width_s / data.macro_time_unit_s)))
    n_macro = int(math.ceil((t1 - t0) / ticks_per_bin))
    macro_edges = t0 + np.arange(n_macro + 1, dtype=np.int64) * ticks_per_bin

    if micro_bins_range is None:
        m0 = int(micro.min())
        m1 = int(micro.max()) + 1
    else:
        m0, m1 = micro_bins_range

    # Plot bins along microtime axis
    micro_edges = np.linspace(m0, m1, n_micro_bins + 1, dtype=np.float64)

    # Assign each photon to macro bin and micro bin
    macro_idx = np.clip(((macro - t0) // ticks_per_bin).astype(np.int64), 0, n_macro - 1)
    micro_idx = np.clip(np.digitize(micro, micro_edges) - 1, 0, n_micro_bins - 1)

    W = np.zeros((n_macro, n_micro_bins), dtype=np.float64)

    # Efficient accumulation
    flat = macro_idx * n_micro_bins + micro_idx
    acc = np.bincount(flat, minlength=n_macro * n_micro_bins).astype(np.float64)
    W[:, :] = acc.reshape(n_macro, n_micro_bins)

    # Convert macro edges to seconds for labeling
    macro_t_s = (macro_edges - macro_edges[0]) * data.macro_time_unit_s
    micro_centers = 0.5 * (micro_edges[:-1] + micro_edges[1:])

    return W, macro_t_s, micro_centers


def plot_waterfall(
    W: np.ndarray,
    macro_t_s: np.ndarray,
    micro_centers: np.ndarray,
    title: str = "Microtime waterfall",
    log_scale: bool = True,
) -> None:
    """
    Render waterfall as an image (time on y-axis, microtime on x-axis).
    """
    M = W.copy()
    if log_scale:
        M = np.log1p(M)

    plt.figure()
    # y-axis is macro bin index; label with seconds at edges
    extent = [micro_centers[0], micro_centers[-1], macro_t_s[-1], macro_t_s[0]]
    plt.imshow(M, aspect="auto", extent=extent)
    plt.xlabel("Microtime (bins or scaled units)")
    plt.ylabel("Macrotime (s)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Convenience: “select channels to listen to”
# -----------------------------

DEFAULT_NOTE_MAP_HZ = {
    # Simple “distinct notes” palette (C major-ish). Customize as needed.
    0: 261.63,  # C4
    1: 329.63,  # E4
    2: 392.00,  # G4
    3: 466.16,  # Bb4
    4: 523.25,  # C5
    5: 659.26,  # E5
}


def make_default_channel_cfg(
    channels: Iterable[int],
    micro_defaults: Tuple[int, int] = (0, 4096),
    pitch_adjust_semitones: Optional[Dict[int, float]] = None,
    micro_gates: Optional[Dict[int, Tuple[int, int]]] = None,
    gains: Optional[Dict[int, float]] = None,
    base_note_map_hz: Optional[Dict[int, float]] = None,
) -> Dict[int, ChannelConfig]:
    """
    Create a per-channel config dict quickly.
    """
    base_note_map_hz = dict(DEFAULT_NOTE_MAP_HZ if base_note_map_hz is None else base_note_map_hz)
    pitch_adjust_semitones = pitch_adjust_semitones or {}
    micro_gates = micro_gates or {}
    gains = gains or {}

    cfg: Dict[int, ChannelConfig] = {}
    for ch in channels:
        note = base_note_map_hz.get(ch, 261.63 * _semitones_to_ratio((ch % 12) * 2.0))
        g0, g1 = micro_gates.get(ch, micro_defaults)
        cfg[ch] = ChannelConfig(
            note_hz=float(note),
            pitch_semitones=float(pitch_adjust_semitones.get(ch, 0.0)),
            micro_min=int(g0),
            micro_max=int(g1),
            gain=float(gains.get(ch, 1.0)),
        )
    return cfg

"""tttrlib-backed streaming simulator.

Drop-in replacement for ``BurbulatorSimulator``: exposes
``simulate_photons_streaming(params, data_queue, stop_event)`` producing BH SPC-132
``uint32`` record batches on the queue (consumed by ``SimulationDevice.read_fifo``),
and optionally writes ``m###.spc`` files. No Qt; uses only the ``core.algorithms``
tttrlib path.
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np

from .algorithms import generate_spc132_uint32, tttrlib_available

logger = logging.getLogger(__name__)


def _make_json_serializable(data):
    """Convert NumPy values to JSON-compatible Python objects.

    Parameters
    ----------
    data : object
        Value to convert recursively.

    Returns
    -------
    object
        JSON-serializable value.
    """
    if isinstance(data, dict):
        return {k: _make_json_serializable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_make_json_serializable(x) for x in data]
    if hasattr(data, "tolist"):
        return data.tolist()
    if hasattr(data, "item"):
        return data.item()
    return data


def spc132_file_bytes(words: np.ndarray, macro_time_clock: int = 100) -> bytes:
    """Return a BH SPC-132 file payload for encoded record words.

    Parameters
    ----------
    words : numpy.ndarray
        Encoded SPC words.
    macro_time_clock : int, optional
        Macro-time clock value encoded into the four-byte file header.

    Returns
    -------
    bytes
        SPC-132 header followed by encoded records.
    """
    mtc = int(macro_time_clock) & 0xFFFFFF
    header = bytes((mtc & 0xFF, (mtc >> 8) & 0xFF, (mtc >> 16) & 0xFF, 0x80))
    return header + words.astype(np.uint32, copy=False).tobytes()


class TttrlibSimulator:
    """Streaming photon generator backed by the tttrlib photon simulator."""

    def __init__(self):
        """Initialize the tttrlib streaming simulator."""
        self.available = tttrlib_available()
        self.generation_thread = None
        if self.available:
            logger.info("SIMULATION: using tttrlib photon simulator backend")
        else:
            logger.warning("SIMULATION: tttrlib photon simulator not available")

    def is_available(self) -> bool:
        """Return whether this backend can run in the current environment.

        Returns
        -------
        bool
            ``True`` when ``tttrlib`` exposes the simulator API.
        """
        return self.available

    def simulate_photons(self, params) -> np.ndarray:
        """Generate all encoded SPC words synchronously.

        Parameters
        ----------
        params : dict
            Acquisition simulation parameter dictionary.

        Returns
        -------
        numpy.ndarray
            Encoded SPC words.
        """
        if not self.available:
            raise RuntimeError("tttrlib photon simulator is not available")
        return generate_spc132_uint32(params)

    def simulate_photons_streaming(self, params, data_queue, stop_event) -> bool:
        """Start a background tttrlib generation thread.

        Parameters
        ----------
        params : dict
            Acquisition simulation parameter dictionary.
        data_queue : queue.Queue
            Queue receiving ``numpy.uint32`` arrays and a final ``None`` marker.
        stop_event : threading.Event
            Event used to request cancellation.

        Returns
        -------
        bool
            ``True`` if the thread was started.
        """
        if not self.available:
            logger.error("SIMULATION: tttrlib backend unavailable")
            return False
        self.generation_thread = threading.Thread(
            target=self._generate, args=(params, data_queue, stop_event), daemon=True)
        self.generation_thread.start()
        return True

    def _generate(self, params, data_queue, stop_event):
        """Generate, optionally write, and stream encoded photons.

        Parameters
        ----------
        params : dict
            Acquisition simulation parameter dictionary.
        data_queue : queue.Queue
            Queue receiving generated chunks.
        stop_event : threading.Event
            Cancellation event.
        """
        try:
            words = self.simulate_photons(params)
            logger.info("SIMULATION: tttrlib generated %d SPC words", len(words))
            self._write_output(params, words)

            batch = int(params.get("stream_words_per_batch", 10000))
            batch = max(1, batch)
            for i in range(0, len(words), batch):
                if stop_event.is_set():
                    break
                data_queue.put(words[i:i + batch], timeout=5.0)
            data_queue.put(None)  # signal completion
            logger.info("SIMULATION: tttrlib streaming completed")
        except Exception as e:  # pragma: no cover - defensive
            logger.error("SIMULATION: tttrlib generation error: %s", e)
            import traceback
            logger.debug(traceback.format_exc())
            try:
                data_queue.put(None)
            except Exception:
                pass

    @staticmethod
    def _write_output(params, words):
        """Write split SPC files and a JSON config, if output is configured.

        Parameters
        ----------
        params : dict
            Acquisition simulation parameter dictionary.
        words : numpy.ndarray
            Encoded SPC words.
        """
        out = params.get("spc_output_path")
        if not out:
            return
        try:
            os.makedirs(out, exist_ok=True)
            photons_per_file = max(1, int(params.get("N_ph_per_file", 100000)))
            words_per_photon = max(1, int(params.get("spc_words_per_photon", 2)))
            words_per_file = photons_per_file * words_per_photon
            macro_time_clock = int(params.get("macro_time_clock_header", 100))
            file_index = 0
            for start in range(0, len(words), words_per_file):
                chunk = words[start:start + words_per_file]
                filename = os.path.join(out, f"m{file_index:03d}.spc")
                with open(filename, "wb") as f:
                    f.write(spc132_file_bytes(chunk, macro_time_clock=macro_time_clock))
                file_index += 1
            with open(os.path.join(out, "simulation_config.json"), "w", encoding="utf-8") as f:
                import json

                json.dump(_make_json_serializable(params), f, indent=2)
            logger.info("SIMULATION: wrote %d SPC file(s) + config to %s", file_index, out)
        except Exception as e:  # pragma: no cover
            logger.error("SIMULATION: failed writing SPC output to %s: %s", out, e)

"""tttrlib-backed simulated TCSPC acquisition device."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from datetime import datetime

import numpy as np
from qtpy.QtCore import Signal

from .core.streaming import TttrlibSimulator
from ..abc import TCSPCDeviceABC

logger = logging.getLogger(__name__)


def make_json_serializable(data):
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
        return {k: make_json_serializable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [make_json_serializable(x) for x in data]
    if hasattr(data, "tolist"):
        return data.tolist()
    if hasattr(data, "item"):
        return data.item()
    return data


class SimulationDevice(TCSPCDeviceABC):
    """Virtual TCSPC device that streams tttrlib photon-simulator output."""

    message_logged = Signal(str)

    def __init__(self):
        """Initialize the virtual acquisition device."""
        super().__init__()
        self.simulator = TttrlibSimulator()
        self.initialized = True
        self.measurement_running = False
        self.available_cards = [0]
        self.active_cards = [0]
        self.device_type = "SIMULATION"

        self.data_queue = queue.Queue(maxsize=50)
        self.stop_event = threading.Event()
        self.generation_thread = None
        self.current_buffer = np.array([], dtype=np.uint32)
        self.buffer_index = 0

        self.simulation_params = {
            "N_species": 1,
            "M": [50.0],
            "D": [3.0],
            "N_channels": 2,
            "q": [50.0, 50.0],
            "q_bg": [0.001, 0.001],
            "k_rad": [0.0],
            "k_nrad": [0.0],
            "box_xy": 2.0,
            "box_z": 4.0,
            "focus_type": 0,
            "focus_param": [0.3, 2.0],
            "dt": 0.01,
            "N_ph_max": 1_000_000,
            "N_ph_per_file": 100_000,
            "pulsed_exc": 0,
            "ch_conversion": [8, 0, 9, 1, 10, 2],
            "N_tac_channels": 4096,
            "tac_dt": 0.004069,
            "laser_period": 13.596,
            "rmt1seed": 12345,
            "rmt2seed": 54321,
            "spc_output_path": "",
        }

    def log_message(self, message):
        """Log a message and emit it for the acquisition GUI.

        Parameters
        ----------
        message : str
            Message text.
        """
        logger.info("SIMULATION: %s", message)
        self.message_logged.emit(str(message))

    def detect_cards(self, simulation=False):
        """Return available virtual card numbers.

        Parameters
        ----------
        simulation : bool, optional
            Ignored compatibility flag.

        Returns
        -------
        list of int
            Virtual card identifiers.
        """
        return list(self.available_cards)

    def set_active_cards(self, card_numbers):
        """Set active virtual card numbers.

        Parameters
        ----------
        card_numbers : iterable
            Card identifiers selected by the acquisition GUI.
        """
        self.active_cards = list(card_numbers)

    def get_active_cards(self):
        """Return active virtual card numbers.

        Returns
        -------
        list of int
            Active card identifiers.
        """
        return list(self.active_cards)

    def initialize(self, simulation=True):
        """Initialize the virtual device.

        Parameters
        ----------
        simulation : bool, optional
            Ignored compatibility flag.

        Returns
        -------
        bool
            ``True`` when the device is ready.
        """
        self.initialized = True
        if self.simulator.is_available():
            self.log_message("Simulation device initialized with tttrlib backend")
        else:
            self.log_message("Simulation device initialized, but tttrlib simulator is unavailable")
        return True

    def start_measurement(self):
        """Start photon generation and queue streaming.

        Returns
        -------
        bool
            ``True`` when streaming starts successfully.
        """
        if not self.initialized:
            self.log_message("Simulation device not initialized")
            return False
        if not self.simulator.is_available():
            self.log_message("ERROR: tttrlib photon simulator is not available")
            return False

        try:
            self._ensure_output_path()
            self._write_simulation_info()
            self._clear_queue()
            self.stop_event.clear()
            self.current_buffer = np.array([], dtype=np.uint32)
            self.buffer_index = 0
            self.measurement_running = True

            success = self.simulator.simulate_photons_streaming(
                self.simulation_params,
                self.data_queue,
                self.stop_event,
            )
            self.generation_thread = getattr(self.simulator, "generation_thread", None)
            if not success:
                self.measurement_running = False
                self.log_message("ERROR: Failed to start tttrlib simulation")
                return False

            self.log_message(
                f"Started tttrlib streaming simulation: "
                f"{int(self.simulation_params.get('N_ph_max', 0))} photons"
            )
            return True
        except Exception as exc:
            self.measurement_running = False
            self.log_message(f"Error starting simulation: {exc}")
            logger.exception("SIMULATION: start_measurement failed")
            return False

    def _ensure_output_path(self):
        """Create the configured output folder or a timestamped default."""
        spc_output_path = str(self.simulation_params.get("spc_output_path", "") or "").strip()
        if not spc_output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            spc_output_path = f"simulation_output_{timestamp}"
            self.simulation_params["spc_output_path"] = spc_output_path
        os.makedirs(spc_output_path, exist_ok=True)

    def _write_simulation_info(self):
        """Write human-readable simulation metadata next to generated SPC files."""
        out = self.simulation_params.get("spc_output_path")
        if not out:
            return
        try:
            config_file_path = os.path.join(out, "simulation_config.json")
            with open(config_file_path, "w", encoding="utf-8") as handle:
                json.dump(make_json_serializable(self.simulation_params), handle, indent=2)

            info_file_path = os.path.join(out, "simulation_info.txt")
            with open(info_file_path, "w", encoding="utf-8") as handle:
                handle.write("Simulation Settings\n")
                handle.write("==================\n\n")
                handle.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                handle.write("Backend: tttrlib photon simulator\n\n")
                for key, value in self.simulation_params.items():
                    handle.write(f"{key}: {value}\n")
        except Exception as exc:
            logger.warning("SIMULATION: failed to write simulation metadata: %s", exc)

    def _clear_queue(self):
        """Remove pending chunks from the stream queue."""
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

    def stop_measurement(self):
        """Stop simulated measurement.

        Returns
        -------
        bool
            ``True`` after the stop request is processed.
        """
        self.measurement_running = False
        self.stop_event.set()

        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=3.0)
            if self.generation_thread.is_alive():
                self.log_message("Background generation thread did not stop gracefully")

        self._clear_queue()
        self.log_message("Simulation measurement stopped")
        return True

    def read_fifo(self, max_words=32768):
        """Read encoded photon records from the streaming queue.

        Parameters
        ----------
        max_words : int, optional
            Maximum number of 16-bit FIFO words requested by the acquisition GUI.

        Returns
        -------
        numpy.ndarray
            ``uint32`` SPC words.
        """
        if not self.initialized or not self.measurement_running:
            return np.array([], dtype=np.uint32)

        max_records = max(1, int(max_words) // 2)
        if len(self.current_buffer) > self.buffer_index:
            return self._read_from_current_buffer(max_records)

        try:
            data = self.data_queue.get(timeout=0.01)
        except queue.Empty:
            return np.array([], dtype=np.uint32)

        if data is None:
            self.measurement_running = False
            self.log_message("Streaming generation completed")
            return np.array([], dtype=np.uint32)

        self.current_buffer = np.asarray(data, dtype=np.uint32)
        self.buffer_index = 0
        return self._read_from_current_buffer(max_records)

    def _read_from_current_buffer(self, max_records):
        """Read at most ``max_records`` from the active buffer.

        Parameters
        ----------
        max_records : int
            Maximum number of ``uint32`` records.

        Returns
        -------
        numpy.ndarray
            Slice of the current buffer.
        """
        remaining = len(self.current_buffer) - self.buffer_index
        to_return = min(remaining, max_records)
        chunk = self.current_buffer[self.buffer_index:self.buffer_index + to_return]
        self.buffer_index += to_return
        if self.buffer_index >= len(self.current_buffer):
            self.current_buffer = np.array([], dtype=np.uint32)
            self.buffer_index = 0
        return chunk

    def get_fifo_usage(self):
        """Return queue fill as a virtual FIFO usage percentage.

        Returns
        -------
        dict
            Mapping of card number to percentage usage.
        """
        if not self.initialized:
            return {0: -1}
        if not self.measurement_running:
            return {0: 0.0}
        max_size = self.data_queue.maxsize if self.data_queue.maxsize > 0 else 50
        return {0: min(100.0, (self.data_queue.qsize() / max_size) * 100.0)}

    def close(self):
        """Close the virtual device."""
        if self.measurement_running:
            self.stop_measurement()
        self.initialized = False
        self.log_message("Simulation device closed")


BurbulatorSimulator = TttrlibSimulator

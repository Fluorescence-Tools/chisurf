"""BrickMic TCSPC device wrapper.

This module implements a minimal TCSPCDeviceABC wrapper around a
Measurement Computing (MCC) counter used in the BrickMic spectrometer.

The goal is to expose BrickMic as a new TCSPC device type ("BRICKMIC")
for the single-molecule acquisition plugin and to stream data as
BH SPC-130 style 32-bit records, so that the rest of the pipeline can
reuse the existing BH/SPC processing code. In addition, the wrapper can
optionally write BH-SPC132 compatible `.spc` files (Becker & Hickl
TTTR format with a 4-byte header) during acquisition.

The current implementation focuses on a single USB MCC counter board
with 32-bit counters and uses a fixed configuration that matches the
original BrickMic prototype:

- 2 input channels (start_ctr=0, end_ctr=1)
- 1 MHz acquisition rate
- Buffer size of 2e6 samples per channel

These parameters can be made user-configurable in a future iteration.
"""

from __future__ import annotations

import os
import time
import threading
import queue
import logging
from typing import List

import numpy as np
from qtpy.QtCore import Signal

from ..abc import TCSPCDeviceABC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional MCC / mcculw imports
# ---------------------------------------------------------------------------

try:  # pragma: no cover - hardware specific
    from mcculw import ul
    from mcculw.enums import (
        ScanOptions,
        CounterMode,
        FunctionType,
        InterfaceType,
        CounterTickSize,
        CounterEdgeDetection,
        CounterDebounceTime,
        ErrorCode,
        Status,
    )
    from mcculw.ul import ULError

    MCC_AVAILABLE = True
except Exception:  # pragma: no cover - allow running without hardware
    ul = None  # type: ignore
    ULError = Exception  # type: ignore
    ScanOptions = CounterMode = FunctionType = InterfaceType = None  # type: ignore
    CounterTickSize = CounterEdgeDetection = CounterDebounceTime = None  # type: ignore
    Status = None  # type: ignore
    MCC_AVAILABLE = False


# ---------------------------------------------------------------------------
# BrickMic hardware configuration defaults
# ---------------------------------------------------------------------------

BRICKMIC_CHANNELS = 2
BRICKMIC_START_CTR = 0
BRICKMIC_END_CTR = BRICKMIC_START_CTR + BRICKMIC_CHANNELS - 1
BRICKMIC_ACQUISITION_RATE = 1_000_000  # Hz
BRICKMIC_BUFFER_SIZE = 2_000_000  # samples per channel


class BrickMicDevice(TCSPCDeviceABC):
    """BrickMic TCSPC device wrapper implementing :class:`TCSPCDeviceABC`.

    The device exposes a queue-based streaming interface similar to the
    simulation device: a background thread watches the MCC ring buffer,
    converts new counts into BH-132 compatible 32-bit records, and
    pushes them into a queue consumed by :meth:`read_fifo`.

    In parallel, the same records can be written to BH-SPC132 files using
    a minimal header compatible with :func:`chisurf.fio.fluorescence.tttr.bh132_photons`.
    """

    # Qt signal must be a class attribute
    message_logged = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self.device_type = "BRICKMIC"
        self.initialized: bool = False
        self.measurement_running: bool = False

        # Hardware acquisition parameters (user-adjustable via setup dialog)
        self.channels: int = BRICKMIC_CHANNELS
        self.start_ctr: int = BRICKMIC_START_CTR
        self.end_ctr: int = BRICKMIC_END_CTR
        self.acquisition_rate: int = BRICKMIC_ACQUISITION_RATE
        self.buffer_size: int = BRICKMIC_BUFFER_SIZE  # samples per channel

        # MCC state
        self.board_num: int = 0
        self.memhandle = None
        self._np_buffer: np.ndarray | None = None
        self._last_idx: int = 0
        self._global_sample_index: int = 0

        # Streaming queue for BH 32-bit words (SPC-130 records)
        self.data_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)
        self.current_buffer: np.ndarray = np.array([], dtype=np.uint32)
        self.buffer_index: int = 0
        self.stop_event = threading.Event()
        self.reader_thread: threading.Thread | None = None

        # SPC file writing configuration
        self.spc_output_path: str = ""
        self.N_ph_per_file: int = 100_000
        self._file_write_buffer: List[bytes] = []
        self._file_index: int = 0
        self._file_photons: int = 0

    # ------------------------------------------------------------------
    # Helper properties
    # ------------------------------------------------------------------

    @property
    def macrotime_clock(self) -> float:
        """Return macrotime resolution in **seconds** per BH macro bin.

        For BrickMic we treat macro bins as coarse time bins derived from
        the acquisition rate. The exact value is not critical as long as
        it is self-consistent between online plots and SPC headers.
        """

        # One macro bin per hardware sample for now
        return 1.0 / float(self.acquisition_rate)

    # ------------------------------------------------------------------
    # TCSPCDeviceABC interface
    # ------------------------------------------------------------------

    def log_message(self, message: str) -> None:
        logger.info("BrickMic: %s", message)
        self.message_logged.emit(message)

    def detect_cards(self, simulation: bool = False):  # type: ignore[override]
        """Detect BrickMic-compatible MCC devices.

        For now we assume a single USB board; this returns ``[0]`` when
        the MCC stack is available and at least one device is detected.
        """

        if not MCC_AVAILABLE:
            self.log_message("mcculw not available; BrickMic hardware disabled")
            return []

        try:
            ul.ignore_instacal()
            devices = ul.get_daq_device_inventory(InterfaceType.USB)
            if not devices:
                self.log_message("No MCC USB counter devices found")
                return []
            return [0]
        except Exception as e:  # pragma: no cover - hardware specific
            self.log_message(f"Error detecting BrickMic device: {e}")
            return []

    def set_active_cards(self, card_numbers):  # type: ignore[override]
        """BrickMic uses a single logical counter device; ignore selection."""
        # Present only for API compatibility; nothing to do.
        return True

    def get_active_cards(self):  # type: ignore[override]
        return [0] if self.initialized else []

    def initialize(self, simulation: bool = True) -> bool:  # type: ignore[override]
        """Initialize the BrickMic device and allocate the ring buffer."""

        if not MCC_AVAILABLE:
            self.log_message("mcculw not available - cannot initialize BrickMic")
            self.initialized = False
            return False

        try:
            ul.ignore_instacal()
            devices = ul.get_daq_device_inventory(InterfaceType.USB)
            if not devices:
                self.log_message("No MCC USB counter devices found")
                self.initialized = False
                return False

            # Bind first detected device to board 0
            self.board_num = 0
            ul.create_daq_device(self.board_num, devices[0])

            total_size = int(self.buffer_size) * int(self.channels)
            self.memhandle = ul.win_buf_alloc_32(total_size)
            if not self.memhandle:
                self.log_message("BrickMic: could not allocate ring buffer")
                self.initialized = False
                return False

            # Configure all counter channels in CLEAR_ON_READ mode
            for chan in range(int(self.start_ctr), int(self.end_ctr) + 1):
                ul.c_config_scan(
                    self.board_num,
                    chan,
                    CounterMode.CLEAR_ON_READ,
                    CounterDebounceTime.DEBOUNCE_NONE,
                    0,
                    CounterEdgeDetection.RISING_EDGE,
                    CounterTickSize.TICK20PT83ns,
                    chan,
                )

            # Create a NumPy view onto the MCC buffer
            import ctypes
            from ctypes import POINTER, c_ulong

            buf_ptr = ctypes.cast(self.memhandle, POINTER(c_ulong))
            self._np_buffer = np.ctypeslib.as_array(buf_ptr, shape=(total_size,))

            self._last_idx = 0
            self._global_sample_index = 0

            self.initialized = True
            self.log_message("BrickMic device initialized")
            return True
        except ULError as e:  # pragma: no cover - hardware specific
            self.log_message(f"BrickMic ULError during initialization: {e}")
            self.initialized = False
            return False
        except Exception as e:  # pragma: no cover
            self.log_message(f"BrickMic initialization error: {e}")
            self.initialized = False
            return False

    def start_measurement(self) -> bool:  # type: ignore[override]
        """Start continuous counter scan and background reader thread."""

        if not self.initialized or not MCC_AVAILABLE:
            self.log_message("BrickMic device not initialized or MCC unavailable")
            return False

        try:
            total_size = int(self.buffer_size) * int(self.channels)
            scanrate = ul.c_in_scan(
                self.board_num,
                int(self.start_ctr),
                int(self.end_ctr),
                total_size,
                int(self.acquisition_rate),
                self.memhandle,
                ScanOptions.BACKGROUND
                | ScanOptions.CONTINUOUS
                | ScanOptions.CTR32BIT,
            )
            self.log_message(f"BrickMic: scan started at {scanrate} Hz")

            # Reset streaming state
            self.data_queue = queue.Queue(maxsize=50)
            self.current_buffer = np.array([], dtype=np.uint32)
            self.buffer_index = 0
            self.stop_event.clear()

            # Reset SPC file state
            self._file_write_buffer = []
            self._file_index = 0
            self._file_photons = 0

            self.measurement_running = True
            self.reader_thread = threading.Thread(
                target=self._background_reader,
                name="BrickMicReader",
                daemon=True,
            )
            self.reader_thread.start()
            return True
        except ULError as e:  # pragma: no cover - hardware specific
            self.log_message(f"BrickMic ULError when starting scan: {e}")
            self.measurement_running = False
            return False
        except Exception as e:  # pragma: no cover
            self.log_message(f"BrickMic error when starting scan: {e}")
            self.measurement_running = False
            return False

    def stop_measurement(self) -> bool:  # type: ignore[override]
        """Stop the continuous scan and the background reader."""

        if not self.initialized or not self.measurement_running:
            return False

        try:
            self.stop_event.set()
            try:
                ul.stop_background(self.board_num, FunctionType.CTRFUNCTION)
            except Exception:
                pass

            if self.reader_thread is not None and self.reader_thread.is_alive():
                self.reader_thread.join(timeout=2.0)

            self.measurement_running = False
            self.log_message("BrickMic measurement stopped")
            return True
        except Exception as e:  # pragma: no cover
            self.log_message(f"Error stopping BrickMic measurement: {e}")
            self.measurement_running = False
            return False

    def read_fifo(self, max_words: int = 32768) -> np.ndarray:  # type: ignore[override]
        """Return up to ``max_words`` 16-bit words as 32-bit BH records.

        The acquisition thread expects 32-bit words where each word
        corresponds to a BH SPC-130 record. Here we interpret
        ``max_words`` as a limit on 16-bit words, so we return at most
        ``max_words // 2`` 32-bit records.
        """

        import queue as queue_module

        if not self.initialized or not self.measurement_running:
            return np.array([], dtype=np.uint32)

        max_photons = max_words // 2

        # First serve any leftover data in the current buffer
        if len(self.current_buffer) > self.buffer_index:
            remaining = len(self.current_buffer) - self.buffer_index
            to_return = min(remaining, max_photons)
            chunk = self.current_buffer[
                self.buffer_index : self.buffer_index + to_return
            ]
            self.buffer_index += to_return
            if self.buffer_index >= len(self.current_buffer):
                self.current_buffer = np.array([], dtype=np.uint32)
                self.buffer_index = 0
            return chunk

        # Otherwise pull next batch from the queue
        try:
            data = self.data_queue.get(timeout=0.01)
        except queue_module.Empty:
            return np.array([], dtype=np.uint32)

        if data is None:
            # Sentinel from background thread indicating completion
            self.measurement_running = False
            return np.array([], dtype=np.uint32)

        self.current_buffer = np.asarray(data, dtype=np.uint32)
        self.buffer_index = 0

        to_return = min(len(self.current_buffer), max_photons)
        chunk = self.current_buffer[:to_return]
        self.buffer_index = to_return
        return chunk

    def get_fifo_usage(self):  # type: ignore[override]
        """Approximate FIFO usage from queue occupancy (0–100%)."""

        if not self.initialized:
            return {0: -1.0}
        if not self.measurement_running:
            return {0: 0.0}

        try:
            qsize = self.data_queue.qsize()
            maxsize = self.data_queue.maxsize or 50
            usage = min(100.0, (qsize / float(maxsize)) * 100.0)
            return {0: usage}
        except Exception:
            return {0: 0.0}

    def close(self) -> None:  # type: ignore[override]
        """Close the device and free MCC resources."""

        try:
            if self.measurement_running:
                self.stop_measurement()

            if MCC_AVAILABLE and self.memhandle is not None:
                try:
                    ul.win_buf_free(self.memhandle)
                except Exception:
                    pass
                self.memhandle = None

            self._np_buffer = None
            self.initialized = False
            self.log_message("BrickMic device closed")
        except Exception as e:  # pragma: no cover
            self.log_message(f"Error closing BrickMic device: {e}")

    # ------------------------------------------------------------------
    # Background reader and SPC writing
    # ------------------------------------------------------------------

    def _background_reader(self) -> None:
        """Watch MCC ring buffer, generate BH records, and write SPC files."""

        if self._np_buffer is None:
            self.log_message("BrickMic: background reader started without buffer")
            self.measurement_running = False
            try:
                self.data_queue.put(None, timeout=1.0)
            except Exception:
                pass
            return

        try:
            last_idx = self._last_idx
            total_size = int(self.buffer_size) * int(self.channels)
            global_sample = self._global_sample_index

            while not self.stop_event.is_set():
                try:
                    cur_status, cur_idx = ul.get_status(
                        self.board_num, FunctionType.CTRFUNCTION
                    )
                except Exception as e:  # pragma: no cover
                    self.log_message(f"BrickMic get_status error: {e}")
                    break

                if Status is not None and cur_status != Status.RUNNING:
                    # Hardware stopped; end acquisition
                    break

                num_new = cur_idx - last_idx
                if num_new < 0:
                    num_new += total_size
                if num_new == 0:
                    time.sleep(0.002)
                    continue

                records: List[int] = []

                for offset in range(num_new):
                    idx = (last_idx + offset) % total_size
                    count = int(self._np_buffer[idx])
                    if count <= 0:
                        global_sample += 1
                        continue

                    # Map sample index to (macro bin, routing channel)
                    channel = idx % max(1, int(self.channels))
                    macro_bin = global_sample & 0x0FFF  # 12-bit macro counter

                    for _ in range(count):
                        rec = self._encode_bh132_record(
                            macro_bin=macro_bin,
                            channel=channel,
                            tac=0,
                        )
                        records.append(rec)

                    global_sample += 1

                last_idx = cur_idx
                self._last_idx = last_idx
                self._global_sample_index = global_sample

                if not records:
                    continue

                chunk_arr = np.asarray(records, dtype=np.uint32)

                # Push to acquisition queue (blocks if full)
                if not self.stop_event.is_set():
                    try:
                        self.data_queue.put(chunk_arr, timeout=5.0)
                    except queue.Full:  # pragma: no cover
                        self.log_message("BrickMic data queue full; dropping chunk")

                # Accumulate for SPC writing
                if self.spc_output_path:
                    spc_bytes = chunk_arr.tobytes()
                    self._file_write_buffer.append(spc_bytes)
                    self._file_photons += chunk_arr.size
                    if self._file_photons >= self.N_ph_per_file:
                        self._flush_spc_file()

            # Flush remaining SPC data
            if self.spc_output_path and self._file_write_buffer:
                self._flush_spc_file()

        except Exception as e:  # pragma: no cover
            self.log_message(f"BrickMic background reader error: {e}")
        finally:
            # Signal completion
            try:
                self.data_queue.put(None, timeout=1.0)
            except Exception:
                pass
            self.measurement_running = False

    @staticmethod
    def _encode_bh132_record(macro_bin: int, channel: int, tac: int) -> int:
        """Encode a single BH132 photon record as a 32-bit little-endian word.

        This follows the layout expected by :func:`bh132_photons`:

        - ``mt12``: 12-bit macro time in ``b0`` and low nibble of ``b1``
        - ``can``: 4-bit routing channel in high nibble of ``b1``
        - ``tac``: 12-bit micro time in low nibble of ``b3`` and ``b2``
        - ``inv`` and ``mtov`` bits (bits 7 and 6 of ``b3``) are zero for
          normal photon records.
        """

        mt12 = int(macro_bin) & 0x0FFF
        ch4 = int(channel) & 0x0F
        tac12 = int(tac) & 0x0FFF

        b0 = mt12 & 0xFF
        b1 = ((ch4 & 0x0F) << 4) | ((mt12 >> 8) & 0x0F)
        b2 = tac12 & 0xFF
        # inv = 0, mtov = 0, high 4 bits of TAC in low nibble
        b3 = (tac12 >> 8) & 0x0F

        return (b3 << 24) | (b2 << 16) | (b1 << 8) | b0

    def _flush_spc_file(self) -> None:
        """Write accumulated SPC-130 records as a BH-SPC132 file."""

        if not self.spc_output_path or not self._file_write_buffer:
            return

        try:
            os.makedirs(self.spc_output_path, exist_ok=True)
            filename = os.path.join(
                self.spc_output_path, f"m{self._file_index:03d}.spc"
            )
            spc_bytes_concat = b"".join(self._file_write_buffer)
            header = self._make_spc132_header()
            with open(filename, "wb") as f:
                f.write(header)
                f.write(spc_bytes_concat)
            self.log_message(
                f"BrickMic wrote SPC file {filename} with {self._file_photons} photons"
            )
            self._file_index += 1
            self._file_write_buffer = []
            self._file_photons = 0
        except Exception as e:  # pragma: no cover
            self.log_message(f"BrickMic failed to write SPC file: {e}")

    @staticmethod
    def _make_spc132_header() -> bytes:
        """Return a minimal BH-SPC132 header (4 bytes).

        The header encodes the macrotime clock in the first 3 bytes as a
        24-bit integer; :func:`bh123_header` interprets this as
        ``MTCLK = value / 10`` in nanoseconds. We choose a value
        consistent with the BrickMic acquisition rate, but for intensity
        work the exact scale is not critical.
        """

        # Macrotime clock in ns: one bin per hardware sample
        mtclk_ns = 1.0e9 / float(BRICKMIC_ACQUISITION_RATE if BRICKMIC_ACQUISITION_RATE > 0 else 1_000_000)
        mtclk_int = int(max(1, min(round(mtclk_ns * 10.0), (1 << 24) - 1)))
        return bytes(
            (
                mtclk_int & 0xFF,
                (mtclk_int >> 8) & 0xFF,
                (mtclk_int >> 16) & 0xFF,
                0x80,  # invalid bit set to 1 as in simulation wrapper
            )
        )

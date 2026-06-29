"""burbulator_dll_wrapper
=========================

ctypes wrapper for the Burbulator C++ shared library (smdif_ov3 / data2spc132_tac).

The library is compiled from ``src/csrc/burbulator/`` via CMake and installed
into this directory during the package build.
"""

from __future__ import annotations

import ctypes as ct
import logging
import math
import os
import platform
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ctypes array helpers – avoid repetitive type construction
# ---------------------------------------------------------------------------

def _c_double(v: float) -> ct.c_double:
    return ct.c_double(v)


def _c_int(v: int) -> ct.c_int:
    return ct.c_int(v)


def _c_ulong(v: int) -> ct.c_ulong:
    return ct.c_ulong(v)


def _c_array(ctype, values):
    """Construct a ctypes array from an iterable of Python values."""
    seq = list(values)
    return (ctype * len(seq))(*seq)


def _c_uint_array(values):
    return _c_array(ct.c_uint, (int(v) for v in values))


def _c_ulong_array(values):
    return _c_array(ct.c_ulong, (int(v) for v in values))


def _c_double_array(values):
    return _c_array(ct.c_double, (float(v) for v in values))


def _c_short_array(values):
    return _c_array(ct.c_short, (int(v) for v in values))


# ---------------------------------------------------------------------------
# RNG state  (Mersenne Twister, 624 words + position)
# ---------------------------------------------------------------------------

_MT_STATE_SIZE = 624


class _RngState:
    """Opaque container for one Mersenne Twister generator state."""

    __slots__ = ("state", "left", "_pending_seed")

    def __init__(self, seed: int = 12345):
        self.state = _c_ulong_array([0] * _MT_STATE_SIZE)
        self.left = ct.c_int(0)
        self._pending_seed: Optional[int] = seed

    def consume_seed(self) -> int:
        """Return the seed value or -1 (meaning 'restore from state')."""
        if self._pending_seed is not None:
            s = self._pending_seed
            self._pending_seed = None
            return s
        return -1


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

def _find_library() -> Optional[str]:
    """Locate ``burbulator`` shared library on the current platform."""
    module_dir = os.path.dirname(os.path.abspath(__file__))

    if sys.platform == "win32":
        names = ["burbulator.dll", "burbulator_x64.dll"]
    elif sys.platform == "darwin":
        names = ["libburbulator.dylib", "burbulator.dylib"]
    else:  # Linux / other POSIX
        names = ["libburbulator.so", "burbulator.so"]

    for name in names:
        path = os.path.join(module_dir, name)
        if os.path.isfile(path):
            logger.info("Burbulator library found: %s", path)
            return path

    # Fall-back: try LD_LIBRARY_PATH / DYLD_LIBRARY_PATH / PATH
    try:
        if sys.platform == "win32":
            return ct.util.find_library("burbulator") or ct.util.find_library("burbulator_x64")
        elif sys.platform == "darwin":
            return ct.util.find_library("burbulator")
        else:
            return ct.util.find_library("burbulator")
    except Exception:
        pass

    logger.warning("Burbulator shared library not found in %s", module_dir)
    return None


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class BurbulatorError(RuntimeError):
    """Raised when the underlying C++ library reports a failure."""


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class BurbulatorDLL:
    """ctypes wrapper for the Burbulator C++ simulation library.

    Exposes ``simulate_ov3`` and ``convert_to_spc132`` as Python methods.

    Parameters
    ----------
    dll_path:
        Explicit path to the shared library.  When *None* the wrapper
        searches the package directory automatically.
    """

    def __init__(self, dll_path: Optional[str] = None) -> None:
        self.path: str = dll_path or _find_library() or ""
        if not self.path or not os.path.isfile(self.path):
            raise BurbulatorError(
                f"Burbulator library not found.  Searched in: "
                f"{os.path.dirname(os.path.abspath(__file__))}"
            )

        if platform.system() == "Windows":
            self._dll: ct.CDLL = ct.windll.LoadLibrary(self.path)  # type: ignore[attr-defined]
        else:
            self._dll: ct.CDLL = ct.CDLL(self.path)

        self._setup_prototypes()
        logger.info("BurbulatorDLL loaded from %s", self.path)

    # -- prototypes --------------------------------------------------------

    def _setup_prototypes(self) -> None:
        """Declare argument / return types for every exported function."""

        dll = self._dll

        # ---- smdif_ov3_c (extern "C" wrapper) ----------------------------
        dll.smdif_ov3_c.argtypes = [
            ct.c_int,                     # N_species
            ct.POINTER(ct.c_double),      # M
            ct.POINTER(ct.c_double),      # D
            ct.c_int,                     # N_channels
            ct.POINTER(ct.c_double),      # q
            ct.POINTER(ct.c_double),      # q_bg
            ct.POINTER(ct.c_double),      # k_rad
            ct.POINTER(ct.c_double),      # k_nrad
            ct.c_double,                  # box_xy
            ct.c_double,                  # box_z
            ct.c_int,                     # focus_type
            ct.POINTER(ct.c_double),      # focus_param
            ct.c_double,                  # dt
            ct.c_ulong,                   # N_ph_max
            ct.POINTER(ct.c_ulong),       # data_T  (output)
            ct.POINTER(ct.c_double),      # data_t  (output)
            ct.POINTER(ct.c_short),       # data_N  (output)
            ct.POINTER(ct.c_short),       # data_species (output)
            ct.POINTER(ct.c_int),         # data_molecule (output)
            ct.POINTER(ct.c_ulong),       # T0  (in/out)
            ct.POINTER(ct.c_int),         # N_molecules (in/out)
            ct.POINTER(ct.c_double),      # x
            ct.POINTER(ct.c_double),      # y
            ct.POINTER(ct.c_double),      # z
            ct.POINTER(ct.c_short),       # species
            ct.c_int,                     # rmt1seed
            ct.POINTER(ct.c_ulong),       # rmt1state
            ct.POINTER(ct.c_int),         # rmt1left
            ct.c_int,                     # rmt2seed
            ct.POINTER(ct.c_ulong),       # rmt2state
            ct.POINTER(ct.c_int),         # rmt2left
        ]
        dll.smdif_ov3_c.restype = ct.c_ulong

        # ---- data2spc132_tac_c (extern "C" wrapper) ----------------------
        dll.data2spc132_tac_c.argtypes = [
            ct.c_int,                     # pulsed_exc
            ct.c_int,                     # N_channels
            ct.POINTER(ct.c_ulong),       # data_T
            ct.POINTER(ct.c_double),      # data_t
            ct.POINTER(ct.c_short),       # data_N
            ct.POINTER(ct.c_short),       # data_species
            ct.POINTER(ct.c_int),         # data_molecule
            ct.c_ulong,                   # N_photons
            ct.c_double,                  # tw
            ct.POINTER(ct.c_ushort),      # ch_conversion
            ct.c_int,                     # N_tac_channels
            ct.c_double,                  # tac_dt
            ct.c_double,                  # laser_period
            ct.POINTER(ct.c_double),      # F
            ct.POINTER(ct.c_int),         # lookup
            ct.POINTER(ct.c_char),        # spc_data  (output)
            ct.POINTER(ct.c_ulong),       # MT_ov  (in/out)
            ct.POINTER(ct.c_ulong),       # i      (in/out)
            ct.POINTER(ct.c_ulong),       # rmt2state
            ct.POINTER(ct.c_int),         # rmt2left
        ]
        dll.data2spc132_tac_c.restype = ct.c_int

    # -- simulate_ov3 ------------------------------------------------------

    def simulate_ov3(
        self,
        *,
        Nspecies: int,
        M: Sequence[float],
        D: Sequence[float],
        Nchannels: int,
        q: Sequence[float],
        q_bg: Sequence[float],
        k_rad: Sequence[float],
        k_nrad: Sequence[float],
        box_xy: float,
        box_z: float,
        focus_type: int,
        focus_param: Sequence[float],
        dt: float,
        N_ph_max: int,
        rmt1seed: int = 12345,
        rmt2seed: int = 54321,
    ) -> Dict:
        """Run the Burbulator ``smdif_ov3`` simulation.

        Returns
        -------
        dict
            Keys: ``N_ph``, ``T0``, ``Nmolecules``, ``data_T``, ``data_t``,
            ``data_N``, ``data_species``, ``data_molecule``.
        """
        # ---- validate / pad inputs ---------------------------------------
        if Nspecies <= 0:
            raise ValueError("Nspecies must be positive")
        if N_ph_max <= 0:
            raise ValueError("N_ph_max must be positive")

        n_rates = Nspecies * Nspecies
        n_q = Nspecies * Nchannels

        _M = _c_double_array([float(v) for v in M][:Nspecies])
        _D = _c_double_array([float(v) for v in D][:Nspecies])
        _q = _c_double_array([float(v) for v in q][:n_q] + [0.0] * max(0, n_q - len(q)))
        _q_bg = _c_double_array([float(v) for v in q_bg][:Nchannels] + [0.0] * max(0, Nchannels - len(q_bg)))
        _k_rad = _c_double_array([float(v) for v in k_rad][:n_rates] + [0.0] * max(0, n_rates - len(k_rad)))
        _k_nrad = _c_double_array([float(v) for v in k_nrad][:n_rates] + [0.0] * max(0, n_rates - len(k_nrad)))
        _focus_param = _c_double_array([float(v) for v in focus_param][:6] + [0.0] * max(0, 6 - len(focus_param)))

        # ---- allocate output buffers -------------------------------------
        _data_T = _c_ulong_array([0]) if N_ph_max <= 0 else _c_ulong_array([0] * (N_ph_max * 2))
        _data_t = _c_double_array([0.0] * (N_ph_max * 2))
        _data_N = _c_short_array([0] * (N_ph_max * 2))
        _data_species = _c_short_array([0] * (N_ph_max * 2))
        _data_molecule = _c_array(ct.c_int, [0] * (N_ph_max * 2))

        sum_M = int(sum(max(1, int(math.ceil(m))) for m in M))
        array_size = sum_M * 2 + 50
        _x = _c_double_array([0.0] * array_size)
        _y = _c_double_array([0.0] * array_size)
        _z = _c_double_array([0.0] * array_size)
        _species = _c_short_array([0] * array_size)

        _T0 = ct.c_ulong(0)
        _Nmolecules = ct.c_int(0)

        # ---- RNG state ---------------------------------------------------
        rng1 = _RngState(rmt1seed)
        rng2 = _RngState(rmt2seed)

        # ---- call --------------------------------------------------------
        result_n_ph = self._dll.smdif_ov3_c(
            _c_int(Nspecies),
            _M, _D,
            _c_int(Nchannels),
            _q, _q_bg, _k_rad, _k_nrad,
            _c_double(box_xy), _c_double(box_z),
            _c_int(focus_type),
            _focus_param,
            _c_double(dt),
            _c_ulong(N_ph_max),
            _data_T, _data_t, _data_N, _data_species, _data_molecule,
            ct.byref(_T0), ct.byref(_Nmolecules),
            _x, _y, _z, _species,
            _c_int(rng1.consume_seed()),
            rng1.state, ct.byref(rng1.left),
            _c_int(rng2.consume_seed()),
            rng2.state, ct.byref(rng2.left),
        )

        n_ph = int(result_n_ph)

        # ---- pack result -------------------------------------------------
        def _as_np(ctype_arr, dtype, length):
            if length <= 0:
                return np.array([], dtype=dtype)
            buf = (ctype_arr._type_ * length).from_address(ct.addressof(ctype_arr))
            return np.frombuffer(buf, dtype=dtype).copy()

        result: Dict = {
            "N_ph": n_ph,
            "T0": int(_T0.value),
            "Nmolecules": int(_Nmolecules.value),
        }

        if n_ph > 0:
            result["data_T"] = _as_np(_data_T, np.uint32, n_ph)
            result["data_t"] = _as_np(_data_t, np.float64, n_ph)
            result["data_N"] = _as_np(_data_N, np.int16, n_ph)
            result["data_species"] = _as_np(_data_species, np.int16, n_ph)
            result["data_molecule"] = _as_np(_data_molecule, np.int32, n_ph)
        else:
            result["data_T"] = np.array([], dtype=np.uint32)
            result["data_t"] = np.array([], dtype=np.float64)
            result["data_N"] = np.array([], dtype=np.int16)
            result["data_species"] = np.array([], dtype=np.int16)
            result["data_molecule"] = np.array([], dtype=np.int32)

        return result

    # -- convert_to_spc132 -------------------------------------------------

    def convert_to_spc132(
        self,
        *,
        pulsed_exc: int,
        Nchannels: int,
        data_T: Sequence[int],
        data_t: Sequence[float],
        data_N: Sequence[int],
        data_species: Sequence[int],
        data_molecule: Sequence[int],
        tw: float,
        ch_conversion: Sequence[int],
        N_tac_channels: int,
        tac_dt: float,
        laser_period: float,
        F: Optional[Sequence[float]] = None,
        lookup: Optional[Sequence[int]] = None,
        N_photons: Optional[int] = None,
        spc_data_bytes_per_photon: int = 8,
    ) -> Tuple[bytes, int, int]:
        """Convert simulated photons to BH SPC-132 / SPC-130 TAC records.

        Returns
        -------
        (spc_bytes, MT_ov, n_bytes)
            Raw SPC-132 bytes, final macro-time-overflow counter, and number
            of bytes written.
        """
        if N_photons is None:
            N_photons = min(
                len(data_T), len(data_t), len(data_N),
                len(data_species), len(data_molecule),
            )
        if N_photons <= 0:
            return b"", 0, 0

        _data_T = _c_ulong_array(data_T[:N_photons])
        _data_t = _c_double_array(data_t[:N_photons])
        _data_N = _c_short_array(data_N[:N_photons])
        _data_species = _c_short_array(data_species[:N_photons])
        _data_molecule = _c_array(ct.c_int, (int(v) for v in data_molecule[:N_photons]))
        _ch_conv = _c_array(ct.c_ushort, (int(v) for v in ch_conversion))

        if F is not None:
            _F = _c_double_array(F)
        else:
            _F = ct.POINTER(ct.c_double)()  # NULL

        if lookup is not None:
            _lookup = _c_array(ct.c_int, (int(v) for v in lookup))
        else:
            _lookup = ct.POINTER(ct.c_int)()

        spc_data_size = int(N_photons * spc_data_bytes_per_photon + 64)
        _spc_data = ct.create_string_buffer(spc_data_size)

        _MT_ov = ct.c_ulong(0)
        _i = ct.c_ulong(0)

        # RNG for TAC  (pulsed excitation)
        rng = _RngState(54321)

        ret = self._dll.data2spc132_tac_c(
            _c_int(pulsed_exc),
            _c_int(Nchannels),
            _data_T, _data_t, _data_N, _data_species, _data_molecule,
            _c_ulong(N_photons),
            _c_double(tw),
            _ch_conv,
            _c_int(N_tac_channels),
            _c_double(tac_dt),
            _c_double(laser_period),
            _F, _lookup,
            _spc_data,
            ct.byref(_MT_ov), ct.byref(_i),
            rng.state, ct.byref(rng.left),
        )

        if ret != 1:
            raise BurbulatorError(
                f"data2spc132_tac returned {ret} (expected 1)"
            )

        n_bytes = int(_i.value)
        return _spc_data.raw[:n_bytes], int(_MT_ov.value), n_bytes

    # -- write_spc132_file -------------------------------------------------

    @staticmethod
    def write_spc132_file(
        path: str,
        spc_bytes: bytes,
        macro_time_clock: int = 100,
    ) -> None:
        """Write a BH SPC-132 file (4-byte header + records) to disk."""
        if macro_time_clock <= 0 or macro_time_clock >= (1 << 24):
            raise ValueError("macro_time_clock must be in range (0, 2^24)")

        header = bytes((
            macro_time_clock & 0xFF,
            (macro_time_clock >> 8) & 0xFF,
            (macro_time_clock >> 16) & 0xFF,
            0x80,  # invalid=1, unused=0
        ))

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(header)
            f.write(spc_bytes)

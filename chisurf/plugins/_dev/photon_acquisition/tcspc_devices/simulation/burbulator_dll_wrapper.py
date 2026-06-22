"""burbulator_dll_wrapper
=========================

High-level wrapper around the Burbulator simulation / conversion DLL.

This module provides a documented, type-safe interface for calling the
C++ functions exported by the Burbulator DLLs:

- ``smdif_ov3``: single-molecule diffusion and photon generation
- ``data2spc132_tac``: conversion of simulated photons to Becker & Hickl
  SPC-132 format (SPC-130 TTTR records + TAC microtime)

The goal is to hide the raw ``ctypes`` details behind a small, well
structured Python API that can be reused by tests and the simulation
backend without duplicating boilerplate.

The implementation closely follows the original C# reference
implementation (``RunOpenVDll.cs``) and the C++ conversion code in
``data2spc_tac.cpp``.
"""

from __future__ import annotations

import os
import sys
import ctypes as ct
from typing import Iterable, Optional, Sequence, Tuple

try:  # NumPy is optional but recommended
    import numpy as np
    NDArray = np.ndarray
except Exception:  # pragma: no cover - fallback when NumPy is not available
    np = None  # type: ignore
    NDArray = None  # type: ignore


class BurbulatorError(RuntimeError):
    """Raised when a call into the Burbulator DLL fails.

    This exception is used for errors reported by the underlying C++ code
    (non-zero return codes) or for misconfiguration on the Python side
    (e.g. inconsistent array lengths).
    """


class BurbulatorDLL:
    """Wrapper for the Burbulator simulation / SPC conversion DLL.

    Parameters
    ----------
    dll_path:
        Optional explicit path to the DLL. If omitted, the wrapper
        searches for ``burbulator_x64.dll`` / ``burbulator.dll`` in

        1. The current simulation plugin directory.
        2. ``E:/dev/tttrlib/playground/Burbulator`` (original playground).

    Notes
    -----
    The wrapper sets up ``ctypes`` prototypes for

    * ``smdif_ov3`` – single-molecule diffusion simulator
    * ``data2spc132_tac`` – converter to BH SPC-132/130 TAC format

    and exposes higher-level methods :meth:`simulate_ov3` and
    :meth:`convert_to_spc132` that operate on NumPy arrays (when
    available) or on pure Python sequences as a fallback.
    """

    # ------------------------------------------------------------------
    # Construction and DLL loading
    # ------------------------------------------------------------------

    def __init__(self, dll_path: Optional[str] = None) -> None:
        self.path: str = dll_path or self._find_default_dll_path()
        if not self.path or not os.path.exists(self.path):
            raise FileNotFoundError(_format_missing_library_message(self.path))

        # Load the shared library in a cross-platform way
        # - Windows: use windll (stdcall)
        # - Other OS (Linux/macOS): use CDLL
        try:
            if os.name == "nt":
                self.dll = ct.windll.LoadLibrary(self.path)
            else:
                self.dll = ct.CDLL(self.path)
        except OSError as e:
            # Provide a helpful hint about how to build/install the library
            raise BurbulatorError(_format_missing_library_message(self.path)) from e
        self._setup_prototypes()

    # ------------------------------------------------------------------
    # Public high-level API
    # ------------------------------------------------------------------

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
    ) -> dict:
        """Run the Burbulator ``smdif_ov3`` simulation.

        This is a direct wrapper around the C++ function
        ``smdif_ov3`` as declared in ``RunOpenVDll.cs``. It simulates
        diffusion and photon emission for a set of species and channels
        and returns the raw per-photon data arrays.

        Parameters
        ----------
        Nspecies:
            Number of fluorescent species in the simulation.
        M:
            Sequence of length ``Nspecies`` with the initial number of
            molecules per species.
        D:
            Sequence of length ``Nspecies`` with diffusion coefficients
            (µm²/µs). Values are passed directly to the DLL.
        Nchannels:
            Number of detection channels.
        q:
            Sequence of length ``Nspecies * Nchannels`` with quantum
            yields (brightness) per species and channel.
        q_bg:
            Background quantum yield per species and channel. Same size
            convention as ``q``.
        k_rad, k_nrad:
            Radiative and non-radiative rate constants. Must have length
            ``Nspecies * Nspecies``; if the provided sequences have a
            different length they are padded with zeros.
        box_xy, box_z:
            Size of the simulation box in µm.
        focus_type:
            Integer focus type identifier (matches the C++/C# code).
        focus_param:
            Focus parameters (typically a 2-element list) passed
            directly to the DLL.
        dt:
            Simulation time step, in the same units as used by the DLL
            (typically microseconds).
        N_ph_max:
            Maximum number of photons to generate in this call.
        rmt1seed, rmt2seed:
            Seeds for the two internal Mersenne Twister RNGs.

        Returns
        -------
        dict
            A dictionary with the following keys:

            ``N_ph`` : int
                Number of photons actually generated.
            ``T0`` : int
                Start time (macro time offset) returned by the DLL.
            ``Nmolecules`` : int
                Number of molecules used/allocated by the simulator.
            ``data_T`` : ndarray or list of uint32
                Macrotime window indices for each photon.
            ``data_t`` : ndarray or list of float64
                Intra-window arrival times for each photon.
            ``data_N`` : ndarray or list of int16
                Detection channel index for each photon.
            ``data_species`` : ndarray or list of int16
                Emitting species index for each photon.
            ``data_molecule`` : ndarray or list of int32
                Emitting molecule identifier for each photon.

        Notes
        -----
        The arrays are truncated to ``N_ph`` elements before being
        returned. If NumPy is available, the arrays are returned as
        ``numpy.ndarray`` instances; otherwise they are returned as
        Python lists.
        """

        # Debug: print all parameters passed to the DLL, for troubleshooting
        try:
            print("BURBULATOR DLL simulate_ov3 parameters:")
            print(f"  Nspecies      = {Nspecies}")
            print(f"  M             = {list(M)}")
            print(f"  D             = {list(D)}")
            print(f"  Nchannels     = {Nchannels}")
            print(f"  q             = {list(q)}")
            print(f"  q_bg          = {list(q_bg)}")
            print(f"  k_rad         = {list(k_rad)}")
            print(f"  k_nrad        = {list(k_nrad)}")
            print(f"  box_xy        = {box_xy}")
            print(f"  box_z         = {box_z}")
            print(f"  focus_type    = {focus_type}")
            print(f"  focus_param   = {list(focus_param)}")
            print(f"  dt            = {dt}")
            print(f"  N_ph_max      = {N_ph_max}")
            print(f"  rmt1seed      = {rmt1seed}")
            print(f"  rmt2seed      = {rmt2seed}")
        except Exception:
            # Debug printing must never break the simulation
            pass

        if Nspecies <= 0:
            raise ValueError("Nspecies must be positive")
        if len(M) < Nspecies or len(D) < Nspecies:
            raise ValueError("M and D must have at least Nspecies elements")
        if N_ph_max <= 0:
            raise ValueError("N_ph_max must be positive")

        # Ensure rate constant arrays have correct size
        n_rates = Nspecies * Nspecies
        if len(k_rad) != n_rates:
            k_rad = list(k_rad)[:n_rates] + [0.0] * max(0, n_rates - len(k_rad))
        if len(k_nrad) != n_rates:
            k_nrad = list(k_nrad)[:n_rates] + [0.0] * max(0, n_rates - len(k_nrad))

        # Convert Python sequences to ctypes arrays
        M_arr = _to_double_array(M)
        D_arr = _to_double_array(D)
        q_arr = _to_double_array(q)
        q_bg_arr = _to_double_array(q_bg)
        k_rad_arr = _to_double_array(k_rad)
        k_nrad_arr = _to_double_array(k_nrad)
        focus_param_arr = _to_double_array(focus_param)

        # Array sizes follow the C# reference implementation
        sum_M = sum(int(float(m)) for m in M)
        array_size = sum_M * 2 + 50

        # Output arrays (factor 2 for safety, as done in the C# code)
        data_T = (ct.c_uint * (N_ph_max * 2))()
        data_t = (ct.c_double * (N_ph_max * 2))()
        data_N = (ct.c_short * (N_ph_max * 2))()
        data_species = (ct.c_short * (N_ph_max * 2))()
        data_molecule = (ct.c_int * (N_ph_max * 2))()

        # Molecule state arrays
        x = (ct.c_double * array_size)()
        y = (ct.c_double * array_size)()
        z = (ct.c_double * array_size)()
        species = (ct.c_short * array_size)()

        # RNG state
        T0 = ct.c_uint(0)
        Nmolecules = ct.c_int(0)

        rmt1state = (ct.c_uint * 624)(0)
        rmt1left = ct.c_int(0)
        rmt2state = (ct.c_uint * 624)(0)
        rmt2left = ct.c_int(0)

        # Call the DLL
        N_ph = self.dll.smdif_ov3(
            int(Nspecies),
            M_arr,
            D_arr,
            int(Nchannels),
            q_arr,
            q_bg_arr,
            k_rad_arr,
            k_nrad_arr,
            float(box_xy),
            float(box_z),
            int(focus_type),
            focus_param_arr,
            float(dt),
            int(N_ph_max),
            data_T,
            data_t,
            data_N,
            data_species,
            data_molecule,
            ct.byref(T0),
            ct.byref(Nmolecules),
            x,
            y,
            z,
            species,
            int(rmt1seed),
            rmt1state,
            ct.byref(rmt1left),
            int(rmt2seed),
            rmt2state,
            ct.byref(rmt2left),
        )

        if N_ph < 0:
            raise BurbulatorError(f"smdif_ov3 returned error code {N_ph}")

        # Convert outputs to NumPy arrays or lists and truncate to N_ph
        n = int(N_ph)
        result = {
            "N_ph": n,
            "T0": int(T0.value),
            "Nmolecules": int(Nmolecules.value),
        }

        def slice_and_convert(arr, ctype, dtype):
            if n == 0:
                if np is not None:
                    return np.array([], dtype=dtype)
                return []
            if np is not None:
                return np.array([ctype(arr[i]).value for i in range(n)], dtype=dtype)
            return [ctype(arr[i]).value for i in range(n)]

        result["data_T"] = slice_and_convert(data_T, ct.c_uint, "uint32")
        result["data_t"] = slice_and_convert(data_t, ct.c_double, "float64")
        result["data_N"] = slice_and_convert(data_N, ct.c_short, "int16")
        result["data_species"] = slice_and_convert(data_species, ct.c_short, "int16")
        result["data_molecule"] = slice_and_convert(data_molecule, ct.c_int, "int32")

        return result

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

        This is a high-level wrapper around the C++ function
        ``data2spc132_tac`` from ``data2spc_tac.cpp``.

        Parameters
        ----------
        pulsed_exc:
            Excitation mode flag; ``0`` for CW, non-zero for pulsed
            excitation.
        Nchannels:
            Number of detection channels.
        data_T, data_t, data_N, data_species, data_molecule:
            Per-photon arrays as returned by :meth:`simulate_ov3`. All
            sequences must have at least ``N_photons`` elements.
        tw:
            Original time window, passed as ``tw`` to the C++ function.
            In the SM acquisition code this is typically the same as
            ``dt`` used during simulation.
        ch_conversion:
            Channel conversion table mapping simulation channels to BH
            SPC routing channels. The C# reference uses a 6-element
            table ``(8, 0, 9, 1, 10, 2)``.
        N_tac_channels:
            Number of TAC channels (e.g. ``4096``).
        tac_dt:
            TAC bin width in nanoseconds (e.g. ``0.004069``).
        laser_period:
            Laser repetition period in nanoseconds (e.g. ``13.596``).
        F, lookup:
            IRF / lookup tables used for pulsed excitation. For CW
            excitation (``pulsed_exc == 0``) these are not used and may
            be left as ``None``.
        N_photons:
            Number of photons to convert. If ``None``, the minimum length
            of the input sequences is used.
        spc_data_bytes_per_photon:
            Size of the output buffer per photon. The default of ``8``
            bytes per photon matches the convention used in the test and
            simulation code and leaves room for overflow records.

        Returns
        -------
        spc_bytes : bytes
            Raw SPC-130 records (overflow + photon records) exactly as
            produced by the C++ function, without any SPC-132 header.
        MT_ov : int
            Total number of macrotime overflows reported by the DLL.
        spc_i : int
            Number of bytes actually written into ``spc_bytes``.

        Notes
        -----
        To produce a valid BH SPC-132 file compatible with tttrlib you
        typically prepend a 4-byte header as created by
        :meth:`write_spc132_file`.
        """
        # Debug: print all parameters passed to the SPC conversion DLL
        try:
            print("BURBULATOR DLL convert_to_spc132 parameters:")
            print(f"  pulsed_exc      = {pulsed_exc}")
            print(f"  Nchannels       = {Nchannels}")
            print(f"  N_photons       = {N_photons if N_photons is not None else 'min(len(data_*))'}")
            print(f"  tw              = {tw}")
            print(f"  N_tac_channels  = {N_tac_channels}")
            print(f"  tac_dt          = {tac_dt}")
            print(f"  laser_period    = {laser_period}")
            print(f"  spc_bytes_per_ph= {spc_data_bytes_per_photon}")
            print(f"  len(data_T)     = {len(list(data_T))}")
            print(f"  len(data_t)     = {len(list(data_t))}")
            print(f"  len(data_N)     = {len(list(data_N))}")
            print(f"  len(data_species)= {len(list(data_species))}")
            print(f"  len(data_molecule)= {len(list(data_molecule))}")
            print(f"  ch_conversion   = {list(ch_conversion)}")
            if F is not None:
                print(f"  len(F)          = {len(list(F))}")
            else:
                print("  F               = None")
            if lookup is not None:
                print(f"  len(lookup)     = {len(list(lookup))}")
            else:
                print("  lookup          = None")
        except Exception:
            # Never let debug printing break conversion
            pass

        # Determine number of photons to convert
        if N_photons is None:
            N_photons = min(len(data_T), len(data_t), len(data_N), len(data_species), len(data_molecule))
        if N_photons <= 0:
            return b"", 0, 0

        # CW excitation does not use F/lookup; for pulsed they must be provided
        if pulsed_exc and (F is None or lookup is None):
            raise ValueError("For pulsed excitation, F and lookup must be provided")

        # Convert inputs to ctypes arrays
        data_T_arr = _to_uint_array(data_T)
        data_t_arr = _to_double_array(data_t)
        data_N_arr = _to_short_array(data_N)
        data_species_arr = _to_short_array(data_species)
        data_molecule_arr = _to_int_array(data_molecule)

        ch_conv_arr = _to_ushort_array(ch_conversion)

        if F is not None:
            F_arr = _to_double_array(F)
        else:
            F_arr = ct.cast(None, ct.POINTER(ct.c_double))  # type: ignore
        if lookup is not None:
            lookup_arr = _to_int_array(lookup)
        else:
            lookup_arr = ct.cast(None, ct.POINTER(ct.c_int))  # type: ignore

        # Output buffer and state variables
        spc_data_size = int(N_photons * spc_data_bytes_per_photon)
        spc_data = (ct.c_byte * spc_data_size)()
        MT_ov = ct.c_ulong(0)
        spc_i = ct.c_ulong(0)

        # Emission RNG state
        rmt2state = (ct.c_uint * 624)(0)
        rmt2left = ct.c_int(0)

        result = self.dll.data2spc132_tac(
            int(pulsed_exc),
            int(Nchannels),
            data_T_arr,
            data_t_arr,
            data_N_arr,
            data_species_arr,
            data_molecule_arr,
            ct.c_uint(int(N_photons)),
            float(tw),
            ch_conv_arr,
            int(N_tac_channels),
            float(tac_dt),
            float(laser_period),
            F_arr,
            lookup_arr,
            spc_data,
            ct.byref(MT_ov),
            ct.byref(spc_i),
            rmt2state,
            ct.byref(rmt2left),
        )

        if result != 1:
            raise BurbulatorError(f"data2spc132_tac returned error code {result}")

        n_bytes = int(spc_i.value)
        spc_bytes = bytes((int(b) & 0xFF) for b in spc_data[:n_bytes])
        return spc_bytes, int(MT_ov.value), n_bytes

    def write_spc132_file(
        self,
        path: str,
        spc_bytes: bytes,
        macro_time_clock: int = 100,
    ) -> None:
        """Write a BH SPC-132 file (header + records) to disk.

        Parameters
        ----------
        path:
            Destination file path (typically ending in ``.spc``).
        spc_bytes:
            Raw SPC-130 records as returned by :meth:`convert_to_spc132`.
        macro_time_clock:
            Value for the ``macro_time_clock`` field in the
            ``bh_spc132_header_t`` header, in units of 0.1 ns. tttrlib
            interprets the macrotime resolution as ``macro_time_clock /
            10e9`` (seconds).

        Notes
        -----
        The header layout follows tttrlib's ``bh_spc132_header_t``:

        .. code-block:: c

           typedef union bh_spc132_header{
               uint32_t allbits;
               struct{
                   unsigned macro_time_clock :24;
                   unsigned unused           :7;
                   bool     invalid          :1;
               } bits;
           } bh_spc132_header_t;

        The last bit (``invalid``) is set to 1, matching tttrlib's
        built-in writer.
        """

        if macro_time_clock <= 0 or macro_time_clock >= (1 << 24):
            raise ValueError("macro_time_clock must be in the range (0, 2^24)")

        header = bytes(
            (
                macro_time_clock & 0xFF,
                (macro_time_clock >> 8) & 0xFF,
                (macro_time_clock >> 16) & 0xFF,
                0x80,  # invalid = 1, unused = 0
            )
        )

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(header)
            f.write(spc_bytes)

    # ------------------------------------------------------------------
    # Internal helpers: DLL search and prototypes
    # ------------------------------------------------------------------

    @staticmethod
    def _find_default_dll_path() -> str:
        """Locate the Burbulator DLL in standard search locations.

        The search order is:

        1. The current simulation package directory
        2. (Windows only) ``E:/dev/tttrlib/playground/Burbulator`` as a
           development fallback

        Returns
        -------
        str
            Absolute path to the first DLL found. If none is found, the
            returned path may be non-existent; the constructor will then
            raise :class:`FileNotFoundError`.
        """

        module_dir = os.path.dirname(os.path.abspath(__file__))
        is_64bit = sys.maxsize > 2 ** 32

        # Preferred library names per platform
        if os.name == "nt":
            dll_names = [
                "burbulator_x64.dll",
                "burbulator.dll",
            ] if is_64bit else [
                "burbulator.dll",
                "burbulator_x64.dll",
            ]
        elif sys.platform == "darwin":
            dll_names = [
                "libburbulator.dylib",
                "burbulator.dylib",
                "libburbulator.so",
            ]
        else:
            # Generic Unix (Linux, etc.)
            dll_names = [
                "libburbulator.so",
                "burbulator.so",
                "libburbulator.dylib",
            ]

        # 1) Simulation package directory
        for name in dll_names:
            candidate = os.path.join(module_dir, name)
            if os.path.exists(candidate):
                return candidate

        # 2) Original playground directory (Windows development fallback only)
        if os.name == "nt":
            burbulator_dir = r"E:/dev/tttrlib/playground/Burbulator"
            for name in dll_names:
                candidate = os.path.join(burbulator_dir, name)
                if os.path.exists(candidate):
                    return candidate

        # Fall back to first name in the simulation directory
        return os.path.join(module_dir, dll_names[0])

    def _setup_prototypes(self) -> None:
        """Configure ``ctypes`` prototypes for DLL functions.

        The prototypes are set to match the C# delegate declarations in
        ``RunOpenVDll.cs`` and the usage in ``test_dll_debug.py``.
        """

        # smdif_ov3
        self.dll.smdif_ov3.argtypes = [
            ct.c_int,  # Nspecies
            ct.POINTER(ct.c_double),  # M
            ct.POINTER(ct.c_double),  # D
            ct.c_int,  # Nchannels
            ct.POINTER(ct.c_double),  # q
            ct.POINTER(ct.c_double),  # q_bg
            ct.POINTER(ct.c_double),  # k_rad
            ct.POINTER(ct.c_double),  # k_nrad
            ct.c_double,  # box_xy
            ct.c_double,  # box_z
            ct.c_int,  # focus_type
            ct.POINTER(ct.c_double),  # focus_param
            ct.c_double,  # dt
            ct.c_int,  # N_ph_max
            ct.POINTER(ct.c_uint),  # data_T
            ct.POINTER(ct.c_double),  # data_t
            ct.POINTER(ct.c_short),  # data_N
            ct.POINTER(ct.c_short),  # data_species
            ct.POINTER(ct.c_int),  # data_molecule
            ct.POINTER(ct.c_uint),  # T0
            ct.POINTER(ct.c_int),  # Nmolecules
            ct.POINTER(ct.c_double),  # x
            ct.POINTER(ct.c_double),  # y
            ct.POINTER(ct.c_double),  # z
            ct.POINTER(ct.c_short),  # species
            ct.c_int,  # rmt1seed
            ct.POINTER(ct.c_uint),  # rmt1state
            ct.POINTER(ct.c_int),  # rmt1left
            ct.c_int,  # rmt2seed
            ct.POINTER(ct.c_uint),  # rmt2state
            ct.POINTER(ct.c_int),  # rmt2left
        ]
        self.dll.smdif_ov3.restype = ct.c_int

        # data2spc132_tac
        self.dll.data2spc132_tac.argtypes = [
            ct.c_int,  # pulsed_exc
            ct.c_int,  # N_channels
            ct.POINTER(ct.c_uint),  # data_T
            ct.POINTER(ct.c_double),  # data_t
            ct.POINTER(ct.c_short),  # data_N
            ct.POINTER(ct.c_short),  # data_species
            ct.POINTER(ct.c_int),  # data_molecule
            ct.c_uint,  # N_photons
            ct.c_double,  # tw
            ct.POINTER(ct.c_ushort),  # ch_conversion
            ct.c_int,  # N_tac_channels
            ct.c_double,  # tac_dt
            ct.c_double,  # laser_period
            ct.POINTER(ct.c_double),  # F
            ct.POINTER(ct.c_int),  # lookup
            ct.POINTER(ct.c_byte),  # spc_data
            ct.POINTER(ct.c_ulong),  # MT_ov
            ct.POINTER(ct.c_ulong),  # spc_i
            ct.POINTER(ct.c_uint),  # rmt2state
            ct.POINTER(ct.c_int),  # rmt2left
        ]
        self.dll.data2spc132_tac.restype = ct.c_int


# ----------------------------------------------------------------------
# Helper functions for converting Python sequences to ctypes arrays
# ----------------------------------------------------------------------


def _to_double_array(values: Sequence[float]) -> ct.Array:
    seq = list(values)
    length = len(seq)
    if length == 0:
        raise ValueError("Sequence must not be empty")
    arr_type = ct.c_double * length
    return arr_type(*[float(v) for v in seq])


def _to_uint_array(values: Sequence[int]) -> ct.Array:
    seq = list(values)
    length = len(seq)
    if length == 0:
        raise ValueError("Sequence must not be empty")
    arr_type = ct.c_uint * length
    return arr_type(*[int(v) for v in seq])


def _to_int_array(values: Sequence[int]) -> ct.Array:
    seq = list(values)
    length = len(seq)
    if length == 0:
        raise ValueError("Sequence must not be empty")
    arr_type = ct.c_int * length
    return arr_type(*[int(v) for v in seq])


def _to_short_array(values: Sequence[int]) -> ct.Array:
    seq = list(values)
    length = len(seq)
    if length == 0:
        raise ValueError("Sequence must not be empty")
    arr_type = ct.c_short * length
    return arr_type(*[int(v) for v in seq])


def _to_ushort_array(values: Sequence[int]) -> ct.Array:
    seq = list(values)
    length = len(seq)
    if length == 0:
        raise ValueError("Sequence must not be empty")
    arr_type = ct.c_ushort * length
    return arr_type(*[int(v) for v in seq])


def _format_missing_library_message(path: str) -> str:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    build_doc = os.path.join(module_dir, "BUILD_BURBULATOR.md")
    return (
        "Burbulator shared library not found or could not be loaded.\n"
        f"Tried path: {path!r}\n"
        "Make sure the library is built in this simulation directory. "
        "You can build it with CMake as described in BUILD_BURBULATOR.md "
        f"(expected to be located at: {build_doc})."
    )

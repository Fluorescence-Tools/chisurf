"""Qt-free simulation core: generate TCSPC photon streams via the tttrlib photon
simulator (replaces the bundled Burbulator DLL)."""
from .algorithms import build_engine, generate_spc132_uint32, tttrlib_available

__all__ = ["build_engine", "generate_spc132_uint32", "tttrlib_available"]

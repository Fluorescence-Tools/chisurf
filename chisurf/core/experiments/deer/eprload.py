from __future__ import annotations

"""Pure-numpy Bruker BES3T (.DSC/.DTA) EPR/DEER loader.

Returns plain NumPy arrays so it can be used inside the ChiSurf ``DeerReader``
without pulling in extra packages. Only the 1D DEER use case (single X abscissa)
is needed; higher-dimensional datasets are squeezed.

Reference: Bruker BES3T format, version 1.2 (Xepr >= 2.1).
"""

import os
import re

import numpy as np


def load_bes3t(dsc_path: str, dta_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a Bruker BES3T dataset from its ``.DSC``/``.DTA`` file pair.

    Parameters
    ----------
    dsc_path, dta_path : str
        Paths to the descriptor and binary data files.

    Returns
    -------
    (t, V, attrs)
        ``t`` is the abscissa in microseconds, ``V`` the (possibly complex)
        signal, and ``attrs`` a dict of parsed metadata.
    """
    with open(dsc_path, "rb") as f:
        dsc = f.read()
    with open(dta_path, "rb") as f:
        dta = f.read()
    return _bes3t(dsc, dta)


def find_companion(path: str) -> tuple[str | None, str | None]:
    """Given a ``.DSC`` or ``.DTA`` path, return the ``(dsc, dta)`` pair."""
    root, ext = os.path.splitext(path)
    ext = ext.upper()
    if ext == ".DSC":
        dta = root + (".DTA" if os.path.exists(root + ".DTA") else ".dta")
        return path, dta
    if ext == ".DTA":
        dsc = root + (".DSC" if os.path.exists(root + ".DSC") else ".dsc")
        return dsc, path
    return None, None


def _bes3t(DSC: bytes, DTA: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    """Parse BES3T descriptor + binary data into ``(t_us, V, attrs)``."""
    parameters = _read_dsc(DSC)
    par = parameters["DESC"]

    nx = int(par["XPTS"]) if "XPTS" in par else None
    if nx is None:
        raise ValueError("No XPTS in DSC file.")

    byteorder = ">" if par.get("BSEQ", "BIG") == "BIG" else "<"

    fmt_map = {"C": "int8", "S": "int16", "I": "int32", "F": "float32", "D": "float64"}
    irfmt = par.get("IRFMT")
    if irfmt not in fmt_map:
        raise ValueError(f"Unsupported/absent IRFMT in .DSC file: {irfmt!r}")
    dt_spc = np.dtype(fmt_map[irfmt])
    dt_data = dt_spc
    dt_spc = dt_spc.newbyteorder(byteorder)

    # X abscissa (linear axis assumed for DEER traces).
    xtyp = par.get("XTYP", "IDX")
    if xtyp == "IDX":
        xmin = float(par["XMIN"])
        xwid = float(par["XWID"])
        npts = int(par["XPTS"])
        t = np.linspace(xmin, xmin + xwid, npts)
    else:
        # Nonlinear/companion axes are uncommon for DEER; fall back to indices.
        t = np.arange(nx, dtype=float)

    ikkf = par.get("IKKF", "REAL")
    raw = np.frombuffer(DTA, dtype=dt_spc)
    if ikkf == "CPLX":
        data = np.copy(raw.astype(dt_data).view(np.complex128))
    else:
        data = np.copy(raw.astype(float))

    data = np.atleast_1d(np.squeeze(data))
    # ns -> µs (BES3T stores time abscissa in ns for pulse experiments).
    t = t / 1e3

    attrs: dict = {"title": par.get("TITL", "").strip("'")}
    attrs.update(_extract_key_parameters(parameters))
    return t.astype(float), data, attrs


def _read_dsc(DSC_file: bytes | str) -> dict:
    """Parse a Bruker BES3T ``.DSC`` file into a nested parameter dict."""
    if isinstance(DSC_file, bytes):
        DSC_file = DSC_file.decode("utf-8", errors="ignore")

    lines = [ln for ln in DSC_file.splitlines() if ln and not ln.startswith("*")]
    lines = [ln.rstrip("\r\n") for ln in lines]

    merged: list[str] = []
    val = ""
    for line in lines:
        val = "".join([val, line])
        if val.endswith("\\"):
            val = val.strip("\\")
        else:
            merged.append(val)
            val = ""
    lines = merged

    params: dict = {}
    section = None
    device = None
    re_section = re.compile(r"#(\w+)\W+(\d+.\d+)")
    re_device = re.compile(r"\.DVC\W+(\w+),\W+(\d+\.\d+)")
    re_kv = re.compile(r"(\w+)\W+(.*?)'?$")

    for line in lines:
        if "MANIPULATION HISTORY LAYER" in line:
            break
        mo = re_section.search(line)
        if mo:
            section = mo.group(1)
            params.setdefault(section, {})
            device = None
            continue
        mo = re_device.search(line)
        if mo:
            device = mo.group(1)
            params.setdefault(section, {})[device] = {}
            continue
        mo = re_kv.search(line)
        if not mo or section is None:
            continue
        key, value = mo.group(1), mo.group(2)
        if device:
            params[section][device][key] = value
        else:
            params[section][key] = value

    if "DESC" not in params:
        raise ValueError("Missing DESC section in .DSC file.")
    return params


def _extract_key_parameters(parameters: dict) -> dict:
    """Flatten and extract a few useful acquisition parameters."""
    flat: dict = {}
    for section, content in parameters.items():
        if section == "DSL":
            for device, dev_content in content.items():
                if isinstance(dev_content, dict):
                    flat.update(dev_content)
        elif isinstance(content, dict):
            flat.update(content)

    out: dict = {}
    if "FrequencyMon" in flat:
        out["freq_ghz"] = _num(flat["FrequencyMon"])
    if "CenterField" in flat:
        out["center_field"] = _num(flat["CenterField"])
    return out


def _num(s: str) -> float | None:
    """Parse the leading number out of a Bruker value string."""
    m = re.match(r"([-\d.]+)", str(s).strip())
    return float(m.group(1)) if m else None

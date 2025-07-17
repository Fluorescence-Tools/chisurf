from __future__ import annotations

import os
import struct
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import chisurf.data
import chisurf.fio
import chisurf.fluorescence


class PQResReader:
    """
    Reader for PicoQuant SymPhoTime .pqres result files (PTU-style header).
    Parses metadata tags and decodes values into native Python types.
    Provides NumPy and pandas integration for curve access.
    """

    tyEmpty8 = 0xFFFF0008
    tyBool8 = 0x00000008
    tyInt8 = 0x10000008
    tyBitSet64 = 0x11000008
    tyColor8 = 0x12000008
    tyFloat8 = 0x20000008
    tyTDateTime = 0x21000008
    tyFloat8Array = 0x2001FFFF
    tyAnsiString = 0x4001FFFF
    tyWideString = 0x4002FFFF
    tyBinaryBlob = 0xFFFFFFFF

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tags = {}
        self.data_offset = None
        self._read_header()

    def _read_header(self):
        with open(self.filepath, 'rb') as f:
            magic = f.read(8)
            if magic[:7] != b'PQRESLT':
                raise ValueError("Invalid magic. Not a PQRES file.")
            version = f.read(8).rstrip(b'\x00').decode('ascii', errors='ignore')
            self.tags['Version'] = version

            while True:
                tag_data = f.read(48)
                if len(tag_data) < 48:
                    break
                ident_raw, idx, typ, value = struct.unpack('<32s i I Q', tag_data)
                ident = ident_raw.split(b'\x00')[0].decode('ascii', errors='ignore')
                if ident == "Header_End":
                    break
                self.tags[ident] = self._interpret_tag(f, typ, value)

            self.data_offset = f.tell()
            self.tags['_data_offset'] = self.data_offset

    def _interpret_tag(self, f, typ, value):
        if typ == self.tyEmpty8:
            return None
        elif typ == self.tyBool8:
            return bool(value)
        elif typ in (self.tyInt8, self.tyBitSet64, self.tyColor8):
            return int(value)
        elif typ == self.tyFloat8:
            return struct.unpack('<d', struct.pack('<Q', value))[0]
        elif typ == self.tyTDateTime:
            dt = struct.unpack('<d', struct.pack('<Q', value))[0]
            return (dt - 25569) * 86400
        elif typ == self.tyFloat8Array:
            count = value // 8
            data = f.read(count * 8)
            return np.frombuffer(data, dtype='<f8', count=count)
        elif typ == self.tyAnsiString:
            data = f.read(value)
            return data.rstrip(b'\x00').decode('ascii', errors='ignore')
        elif typ == self.tyWideString:
            data = f.read(value)
            return data.decode('utf-16le', errors='ignore').rstrip('\x00')
        elif typ == self.tyBinaryBlob:
            f.seek(value, 1)
            return f"<{value} bytes blob>"
        else:
            return f"<Unsupported type 0x{typ:X}>"

    def get_tag(self, name, default=None):
        return self.tags.get(name, default)

    def list_tags(self):
        return [k for k in self.tags if not k.startswith('_')]

    def read_raw_data(self):
        with open(self.filepath, 'rb') as f:
            f.seek(self.data_offset)
            return f.read()

    def get_curves(self) -> Dict[str, Dict[str, Any]]:
        curves = {}
        for k in self.tags:
            if k.endswith("X"):
                base = k[:-1]
                x = self.tags.get(f"{base}X")
                y = self.tags.get(f"{base}Y")
                if isinstance(x, (np.ndarray, list, tuple)) and isinstance(y, (np.ndarray, list, tuple)) and len(x) == len(y):
                    curves[base] = {
                        "label": base,
                        "X": np.array(x),
                        "Y": np.array(y),
                        "StdDev": np.array(self.tags.get(f"{base}StdDevY", [])),
                        "Weight": np.array(self.tags.get(f"{base}WeightY", [])),
                    }
        return curves

    def get_curves_as_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Returns:
            Dict[str, pd.DataFrame]: Each key is a curve name, value is a DataFrame with columns:
            'X', 'Y', and optionally 'StdDev', 'Weight'
        """
        df_dict = {}
        for name, curve in self.get_curves().items():
            data = {
                "X": curve["X"],
                "Y": curve["Y"]
            }
            if curve["StdDev"].size:
                data["StdDev"] = curve["StdDev"]
            if curve["Weight"].size:
                data["Weight"] = curve["Weight"]
            df_dict[name] = pd.DataFrame(data)
        return df_dict

    def __repr__(self):
        lines = [f"<PQResReader: {self.filepath}>",
                 f"Version: {self.tags.get('Version')}",
                 f"Data offset: {self.data_offset}",
                 f"Tags ({len(self.list_tags())}):"]
        for k in self.list_tags():
            v = self.tags[k]
            v_str = repr(v)
            lines.append(f"  {k}: {v_str[:100]}{'...' if len(v_str) > 100 else ''}")
        return "\n".join(lines)


def read_pqres_fcs(filename: str, data_reader: chisurf.experiments.reader.ExperimentReader = None, 
                  experiment: chisurf.experiments.experiment.Experiment = None, **kwargs) -> chisurf.data.DataCurveGroup:
    """
    Read a PicoQuant SymPhoTime .pqres FCS result file and return a DataCurveGroup.

    Parameters
    ----------
    filename : str
        Path to the .pqres file
    data_reader : chisurf.experiments.reader.ExperimentReader, optional
        Data reader to use for reading the file
    experiment : chisurf.experiments.experiment.Experiment, optional
        Experiment to associate with the data
    **kwargs
        Additional keyword arguments to pass to the DataCurve constructor

    Returns
    -------
    chisurf.data.DataCurveGroup
        A DataCurveGroup containing all curves in the .pqres file
    """
    reader = PQResReader(filename)
    curves = reader.get_curves()

    # Create a DataCurveGroup to hold all curves
    curve_group = chisurf.data.DataCurveGroup([])

    # Add each curve to the group
    for name, curve_data in curves.items():
        x = curve_data["X"]
        y = curve_data["Y"]

        # Use StdDev as error if available, otherwise use None
        ex = np.zeros_like(x)
        ey = curve_data["StdDev"] if curve_data["StdDev"].size else np.ones_like(y)

        # Create a DataCurve for this curve
        data_curve = chisurf.data.DataCurve(
            x=x, 
            y=y, 
            ex=ex, 
            ey=ey,
            filename=filename,
            data_reader=data_reader,
            experiment=experiment,
            name=name,
            load_filename_on_init=False,
            **kwargs
        )

        # Add the curve to the group
        curve_group.append(data_curve)

    return curve_group


def read_pqres_tcspc(filename: str, data_reader: chisurf.experiments.reader.ExperimentReader = None, 
                    experiment: chisurf.experiments.experiment.Experiment = None, 
                    dt: float = 1.0, rebin: typing.Tuple[int, int] = (1, 1), **kwargs) -> chisurf.data.DataCurveGroup:
    """
    Read a PicoQuant SymPhoTime .pqres TCSPC result file and return a DataCurveGroup.
    For TCSPC data, the x values are multiplied by 1e9 to convert from seconds to nanoseconds.
    Applies binning according to the rebin parameter.

    Parameters
    ----------
    filename : str
        Path to the .pqres file
    data_reader : chisurf.experiments.reader.ExperimentReader, optional
        Data reader to use for reading the file
    experiment : chisurf.experiments.experiment.Experiment, optional
        Experiment to associate with the data
    dt : float, optional
        Time resolution in nanoseconds, by default 1.0
    rebin : tuple of int, optional
        Rebinning factors for x and y axes, by default (1, 1)
    **kwargs
        Additional keyword arguments to pass to the DataCurve constructor

    Returns
    -------
    chisurf.data.DataCurveGroup
        A DataCurveGroup containing all curves in the .pqres file
    """
    # Import here to avoid circular imports
    import chisurf.data
    from chisurf import typing

    # Load data
    rebin_x, rebin_y = rebin

    reader = PQResReader(filename)
    curves = reader.get_curves()

    # Create a DataCurveGroup to hold all curves
    curve_group = chisurf.data.DataCurveGroup([])

    # Add each curve to the group
    for name, curve_data in curves.items():
        # Convert x values from seconds to nanoseconds (multiply by 1e9)
        x = curve_data["X"] * 1e9
        y = curve_data["Y"]

        # Apply rebinning if needed
        if rebin_y > 1:
            # Calculate new length after rebinning
            new_length = len(y) // rebin_y
            # Reshape and sum along the rebinning axis
            y = y[:new_length * rebin_y].reshape(-1, rebin_y).sum(axis=1)
            # Adjust time axis
            x = x[:new_length * rebin_y:rebin_y]

        # Use zeros for x error
        ex = np.zeros_like(x)
        # For TCSPC data, use Poisson error (sqrt(y)) as this is appropriate for photon counting
        ey = np.sqrt(y)

        # Create a DataCurve for this curve
        data_curve = chisurf.data.DataCurve(
            x=x, 
            y=y, 
            ex=ex, 
            ey=ey,
            filename=filename,
            data_reader=data_reader,
            experiment=experiment,
            name=name,
            load_filename_on_init=False,
            **kwargs
        )

        # Add the curve to the group
        curve_group.append(data_curve)

    return curve_group

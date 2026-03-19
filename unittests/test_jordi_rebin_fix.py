import os
import tempfile
import importlib.util
from pathlib import Path

import numpy as np


def test_vv_vh_rebin_reshape_groups_are_stable():
    module_path = Path(__file__).resolve().parents[1] / "chisurf" / "fio" / "jordi.py"
    spec = importlib.util.spec_from_file_location("jordi_local", module_path)
    jordi = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(jordi)

    n_points = 2048
    x = np.linspace(0.0, 10.0, n_points)
    vv = np.exp(-x / 3.0) * 1000.0
    vh = np.exp(-x / 3.0) * 600.0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        filename = tmp.name

    try:
        jordi.write_jordi(filename, vv=vv, vh=vh, g_factor=1.0)
        data, _meta = jordi.read_jordi(filename, split=True, return_metadata=True)

        vv_read = np.asarray(data["VV"])
        vh_read = np.asarray(data["VH"])
        y = np.vstack([vv_read, vh_read])

        assert y.shape == (2, n_points)

        for rebin_y in (1, 2, 4, 8):
            new_channels = n_points // rebin_y
            y_rebinned = y.reshape([2, new_channels, rebin_y]).sum(axis=2)
            assert y_rebinned.shape == (2, new_channels)
    finally:
        os.unlink(filename)

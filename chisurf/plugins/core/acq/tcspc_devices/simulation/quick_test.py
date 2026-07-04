#!/usr/bin/env python3
"""Quick smoke test for the tttrlib-backed acquisition simulator."""

from __future__ import annotations

import tempfile
from pathlib import Path

from qtpy.QtWidgets import QApplication

from chisurf.plugins.core.acq.tcspc_devices.simulation.wrapper import SimulationDevice


def main() -> int:
    """Run a tiny end-to-end simulation smoke test.

    Returns
    -------
    int
        Process exit code.
    """
    QApplication.instance() or QApplication([])
    output = Path(tempfile.mkdtemp(prefix="chisurf-sim-"))
    device = SimulationDevice()
    device.simulation_params.update({
        "N_ph_max": 16,
        "N_ph_per_file": 8,
        "M": [1.0],
        "D": [1.0],
        "q": [5.0, 5.0],
        "q_bg": [0.0, 0.0],
        "spc_output_path": str(output),
    })
    if not device.start_measurement():
        print("Failed to start tttrlib simulation")
        return 1
    if device.generation_thread:
        device.generation_thread.join(timeout=30)

    n_words = 0
    while device.measurement_running:
        n_words += len(device.read_fifo(1024))
    device.close()

    files = sorted(output.glob("m*.spc"))
    print(f"Generated {n_words} FIFO words in {len(files)} SPC file(s): {output}")
    return 0 if n_words > 0 and files else 1


if __name__ == "__main__":
    raise SystemExit(main())

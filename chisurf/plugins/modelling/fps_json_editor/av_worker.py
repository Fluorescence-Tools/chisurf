"""Worker thread for non-blocking Accessible Volume calculations."""

from __future__ import annotations

import numpy as np
from qtpy import QtCore

from chisurf.plugins.modelling.fret import av


class AVWorker(QtCore.QThread):
    """Non-blocking AV computation worker.

    Emits result_ready when done, error on failure.
    Never raises — always emits one of the two signals.
    """

    # Signals:
    # - result_ready: n_points, volume_A3, mean_x, mean_y, mean_z, coords (N, 4)
    # - error: error message string
    result_ready = QtCore.Signal(int, float, float, float, float, object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        atoms_xyzr: np.ndarray,
        source_xyz: np.ndarray,
        linker_length: float,
        linker_width: float,
        radii: tuple[float, float, float],
        disc_step: float = 1.5,
        pdb_path: str | None = None,
        source_info: dict | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.atoms_xyzr = atoms_xyzr
        self.source_xyz = source_xyz
        self.linker_length = linker_length
        self.linker_width = linker_width
        self.radii = radii
        self.disc_step = disc_step
        self.pdb_path = pdb_path
        self.source_info = source_info

    def run(self) -> None:
        """Run AV computation and emit result or error signal."""
        try:
            accessible_volume = av.compute_av(
                atoms=self.atoms_xyzr,
                source_xyz=self.source_xyz,
                linker_length=self.linker_length,
                linker_width=self.linker_width,
                radii=self.radii,
                disc_step=self.disc_step,
                pdb_path=self.pdb_path,
                source_info=self.source_info,
            )
            n_points = accessible_volume.n_points
            volume = n_points * (accessible_volume.grid_step ** 3)
            mean_xyz = accessible_volume.mean_position
            coords = accessible_volume.points
            self.result_ready.emit(
                n_points,
                float(volume),
                float(mean_xyz[0]),
                float(mean_xyz[1]),
                float(mean_xyz[2]),
                coords
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(f"{str(e)}\n{tb}")

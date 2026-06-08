"""3D viewer for visualizing accessible volumes and restraints.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from qtpy import QtCore, QtWidgets

import chisurf.core.structure
from chisurf.plugins.chimol.chimol.renderer.view import MolView


class AVViewer3D(QtWidgets.QWidget):
    """A 3D widget for visualizing structures, accessible volumes, and FRET restraints."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initialize the viewer embedding MolView."""
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.mol_view = MolView(self)
        layout.addWidget(self.mol_view)

    def show_structure(self, pdb_path: str) -> None:
        """Load and display the PDB structure in the 3D viewer.

        Parameters
        ----------
        pdb_path : str
            Path to the PDB file.
        """
        try:
            struct = chisurf.core.structure.Structure(pdb_path)
            self.mol_view.set_structure(struct)
        except Exception as e:
            print(f"AVViewer3D: Failed to load structure {pdb_path}: {e}")

    def show_av(self, key: str, coords: np.ndarray, color: tuple = (0.0, 1.0, 0.5, 0.5)) -> None:
        """Display the accessible volume point cloud.

        Parameters
        ----------
        key : str
            Unique identifier for this AV.
        coords : np.ndarray
            (N, 3) or (N, 4) coordinate array.
        color : tuple, optional
            RGBA color tuple.
        """
        self.mol_view.add_point_overlay(
            key,
            coords[:, :3],
            color=color,
            size_scale=0.015,
            min_size=1.0,
            alpha=color[3] if len(color) > 3 else 0.5
        )

    def show_mean_position(self, key: str, center: np.ndarray, color: tuple = (1.0, 0.8, 0.2, 0.9), label: Optional[str] = None) -> None:
        """Display the mean position as a 3D sphere.

        Parameters
        ----------
        key : str
            Unique identifier.
        center : np.ndarray
            (3,) coordinates.
        color : tuple, optional
            RGBA color.
        label : str, optional
            Text label.
        """
        self.mol_view.add_sphere(center, radius=1.5, color=color, label=label, key=key)

    def show_restraints(self, restraints_coords: List[Tuple[np.ndarray, np.ndarray]], color: tuple = (1.0, 0.0, 0.0, 0.8)) -> None:
        """Draw lines representing distance restraints.

        Parameters
        ----------
        restraints_coords : List[Tuple[np.ndarray, np.ndarray]]
            List of (start_xyz, end_xyz) coordinate pairs.
        color : tuple, optional
            RGBA color.
        """
        if self.mol_view._measurements is None:
            self.mol_view._measurements = {}
            
        # Clear old restraints
        for k in list(self.mol_view._measurements.keys()):
            if k.startswith("restraint_"):
                del self.mol_view._measurements[k]

        for i, (start, end) in enumerate(restraints_coords):
            self.mol_view._measurements[f"restraint_{i}"] = {
                "kind": "distance",
                "positions": np.array([start, end], dtype=float),
                "color": color,
                "label": ""
            }
        self.mol_view._update_view()

    def clear_av(self) -> None:
        """Clear all point overlays and spheres."""
        self.mol_view.clear_point_overlays()
        if self.mol_view._measurements:
            for k in list(self.mol_view._measurements.keys()):
                if k.startswith("restraint_"):
                    del self.mol_view._measurements[k]
        self.mol_view._update_view()

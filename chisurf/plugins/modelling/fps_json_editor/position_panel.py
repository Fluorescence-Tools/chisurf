"""Panel for configuring labeling positions and viewing Accessible Volumes in 3D."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from qtpy import QtCore, QtWidgets

import chisurf.core.structure
import chisurf.gui.widgets
import chisurf.gui.widgets.fluorescence.av
import chisurf.gui.widgets.general
import chisurf.gui.widgets.pdb
from chisurf.plugins.chimol.chimol.renderer.view import MolView
from chisurf.plugins.modelling.fret import av

from .av_worker import AVWorker

logger = logging.getLogger("chisurf.plugins.modelling.fret")


class PositionPanel(QtWidgets.QWidget):
    """A widget for selecting labeling positions on a PDB structure.

    Integrates standard PDB selectors, AV properties, a live 3D viewer (MolView)
    for rendering computed Accessible Volumes, and a position list.

    Attributes
    ----------
    position_added : QtCore.Signal
        Emitted when a new labeling position is added. Passes (name, params).
    position_removed : QtCore.Signal
        Emitted when a labeling position is removed. Passes (name).
    """

    position_added = QtCore.Signal(str, dict)
    position_removed = QtCore.Signal(str)

    def _show_status(self, msg: str, level: str = "info"):
        if level == "info":
            logger.info(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
        win = self.window()
        if win and hasattr(win, "statusBar") and win.statusBar() is not None:
            win.statusBar().showMessage(msg, 5000)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize the PositionPanel with layout and sub-widgets."""
        super().__init__(parent)
        self._client = client or self._make_default_client()
        self._structure: chisurf.core.structure.Structure | None = None
        self._pdb_path: str | None = None
        self._av_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        self._last_av_points: np.ndarray | None = None
        self._last_mean_xyz: np.ndarray | None = None
        self._last_pos_name: str | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        # Use a horizontal splitter to separate parameters/list and the 3D viewer
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        # Left panel: controls and list
        self.left_pane = QtWidgets.QWidget(self.splitter)
        left_layout = QtWidgets.QVBoxLayout(self.left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # PDB loading row
        pdb_layout = QtWidgets.QHBoxLayout()
        pdb_layout.addWidget(QtWidgets.QLabel("Reference PDB:"))
        self.pdb_filename_edit = QtWidgets.QLineEdit()
        self.pdb_filename_edit.setReadOnly(True)
        pdb_layout.addWidget(self.pdb_filename_edit, stretch=1)
        self.load_pdb_btn = QtWidgets.QPushButton("...")
        self.load_pdb_btn.setFixedWidth(30)
        self.load_pdb_btn.clicked.connect(self.onLoadReferencePDB)
        pdb_layout.addWidget(self.load_pdb_btn)
        left_layout.addLayout(pdb_layout)

        fetch_layout = QtWidgets.QHBoxLayout()
        fetch_layout.addWidget(QtWidgets.QLabel("PDB ID:"))
        self.pdb_id_edit = QtWidgets.QLineEdit()
        self.pdb_id_edit.setPlaceholderText("e.g. 148L")
        self.pdb_id_edit.returnPressed.connect(self.onFetchPDBById)
        fetch_layout.addWidget(self.pdb_id_edit, stretch=1)
        self.fetch_pdb_btn = QtWidgets.QPushButton("Fetch PDB")
        self.fetch_pdb_btn.clicked.connect(self.onFetchPDBById)
        fetch_layout.addWidget(self.fetch_pdb_btn)
        left_layout.addLayout(fetch_layout)

        # Position name and Body ID row
        pos_name_layout = QtWidgets.QGridLayout()
        pos_name_layout.addWidget(QtWidgets.QLabel("Position Name:"), 0, 0)
        self.position_name_edit = QtWidgets.QLineEdit()
        pos_name_layout.addWidget(self.position_name_edit, 0, 1)

        pos_name_layout.addWidget(QtWidgets.QLabel("Body ID:"), 1, 0)
        self.body_id_spin = QtWidgets.QSpinBox()
        self.body_id_spin.setRange(0, 99)
        self.body_id_spin.setValue(0)
        self.body_id_spin.setToolTip("Body index for multi-body docking.")
        pos_name_layout.addWidget(self.body_id_spin, 1, 1)
        left_layout.addLayout(pos_name_layout)

        optional_layout = QtWidgets.QHBoxLayout()
        optional_layout.addWidget(QtWidgets.QLabel("Optional field:"))
        self.optional_field_edit = QtWidgets.QLineEdit()
        self.optional_field_edit.setPlaceholderText("key=value, e.g. dye_name=ATTO647N")
        self.optional_field_edit.setToolTip(
            "Optional position metadata. Leave empty to keep the existing fps.json schema."
        )
        optional_layout.addWidget(self.optional_field_edit, stretch=1)
        left_layout.addLayout(optional_layout)

        # PDB Atom Selector
        self.atom_select = chisurf.gui.widgets.pdb.PDBSelector()
        left_layout.addWidget(self.atom_select)

        # Simulation type and AV Properties
        sim_layout = QtWidgets.QHBoxLayout()
        sim_layout.addWidget(QtWidgets.QLabel("Simulation type:"))
        self.simulation_type_combo = QtWidgets.QComboBox()
        self.simulation_type_combo.addItems(["AV1", "AV0", "AV3"])
        self.simulation_type_combo.currentTextChanged.connect(self.onSimulationTypeChanged)
        sim_layout.addWidget(self.simulation_type_combo)
        left_layout.addLayout(sim_layout)

        self.av_properties = chisurf.gui.widgets.fluorescence.av.AVProperties()
        left_layout.addWidget(self.av_properties)

        # AV Preview and Show in 3D buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.preview_av_btn = QtWidgets.QPushButton("Preview AV")
        self.preview_av_btn.clicked.connect(self.onPreviewAV)
        self.show_3d_btn = QtWidgets.QPushButton("Show in 3D")
        self.show_3d_btn.setEnabled(False)
        self.show_3d_btn.clicked.connect(self.onShowIn3D)
        btn_layout.addWidget(self.preview_av_btn)
        btn_layout.addWidget(self.show_3d_btn)
        left_layout.addLayout(btn_layout)

        # Progress indicator and preview details
        self.av_preview_label = QtWidgets.QLabel("AV: Not computed")
        self.av_preview_label.setWordWrap(True)
        left_layout.addWidget(self.av_preview_label)

        self.av_progress_bar = QtWidgets.QProgressBar()
        self.av_progress_bar.setRange(0, 0)
        self.av_progress_bar.setVisible(False)
        left_layout.addWidget(self.av_progress_bar)

        # Add / Remove Position buttons
        self.add_position_btn = QtWidgets.QPushButton("Add position")
        self.add_position_btn.clicked.connect(self.onAddPosition)
        left_layout.addWidget(self.add_position_btn)

        # Position list
        left_layout.addWidget(QtWidgets.QLabel("Positions List (double-click to delete):"))
        self.positions_list = QtWidgets.QListWidget()
        self.positions_list.itemSelectionChanged.connect(self.onPositionSelectionChanged)
        self.positions_list.doubleClicked.connect(self.onPositionsListDoubleClicked)
        left_layout.addWidget(self.positions_list)

        # Right panel: 3D viewer (MolView)
        self.mol_view_3d = MolView(self.splitter)
        self.mol_view_3d.setMinimumWidth(300)
        self.mol_view_3d.show()  # Always display even when no structure is loaded

        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.mol_view_3d)
        self.splitter.setSizes([350, 400])

        self.mol_view_3d.atomSelectionChanged.connect(
            self.on_chimol_atom_selection_changed
        )

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.splitter)

        self._av_worker: AVWorker | None = None

    @staticmethod
    def _make_default_client():
        """Create a default FpsJsonEditorClient with local in-process services."""
        from .gui.communication import FpsJsonEditorClient
        return FpsJsonEditorClient()

    def onSimulationTypeChanged(self, text: str) -> None:
        """Handle changes to the simulation type dropdown."""
        self.av_properties.av_type = text

    def onLoadReferencePDB(self) -> None:
        """Open a file dialog to load a reference PDB structure."""
        filename = chisurf.gui.widgets.get_filename(
            'Open PDB-File',
            'PDB-Files (*.pdb);;PDB-GZ (*.pdb.gz)'
        )
        if filename:
            self.load_structure(filename)

    def onFetchPDBById(self) -> None:
        """Fetch a reference PDB structure from RCSB by four-character ID."""
        pdb_id = self.pdb_id_edit.text().strip()
        if not pdb_id:
            self._show_status("Please enter a four-character PDB ID.", "warning")
            return

        self.fetch_pdb_btn.setEnabled(False)
        self.pdb_id_edit.setEnabled(False)
        self._show_status(f"Fetching PDB structure '{pdb_id}'...", "info")
        try:
            result = self._client.fetch_pdb(pdb_id) if self._client is not None else None
            if result is None:
                raise RuntimeError("PDB fetch client is not configured")
            self.load_structure(result["path"])
            self._show_status(f"Fetched PDB structure '{pdb_id}'.", "info")
        except Exception as exc:
            self._show_status(f"Failed to fetch PDB structure '{pdb_id}': {exc}", "error")
        finally:
            self.fetch_pdb_btn.setEnabled(True)
            self.pdb_id_edit.setEnabled(True)

    def load_structure(self, path: str) -> None:
        """Load a structure from path into the panel and selectors.

        Parameters
        ----------
        path : str
            Absolute path to the PDB file(s), optionally comma-separated.
        """
        try:
            self._pdb_path = path
            self.pdb_filename_edit.setText(path)

            # Disable non-standard residues filter globally so DNA/RNA residues are kept
            import chisurf as cs
            if hasattr(cs.core.settings, "structure_data"):
                cs.core.settings.structure_data.setdefault("IMP", {})["filter_non_standard_residues"] = False

            # Split path by comma to support multiple PDB files
            paths = [p.strip() for p in path.split(",") if p.strip()]
            if len(paths) == 1:
                self._structure = chisurf.core.structure.Structure(paths[0])
            else:
                atoms_list = []
                for p in paths:
                    s = chisurf.core.structure.Structure(p)
                    if s.atoms is not None and len(s.atoms) > 0:
                        atoms_list.append(s.atoms)
                if atoms_list:
                    combined_atoms = np.concatenate(atoms_list)
                    # Make atom_id sequential
                    combined_atoms['atom_id'] = np.arange(1, len(combined_atoms) + 1)

                    self._structure = chisurf.core.structure.Structure()
                    self._structure.atoms = combined_atoms
                    self._structure.filename = path
                else:
                    self._structure = chisurf.core.structure.Structure()

            self.atom_select.atoms = self._structure.atoms
            self.mol_view_3d.set_structure(self._structure)
            self.mol_view_3d.show()
        except Exception as e:
            self._show_status(f"Error loading reference PDB structure: {str(e)}", "error")


    def onPreviewAV(self) -> None:
        """Trigger background computation of the accessible volume (AV) for the active selection."""
        if not self._pdb_path:
            self._show_status("Please load a PDB structure first.", "warning")
            return

        try:
            # Load structure with VdW radii
            atoms_xyzr = av.load_structure_with_vdw(self._pdb_path)

            atom_idx = self.atom_select.atom_number
            if atom_idx is None or atom_idx < 0 or atom_idx >= atoms_xyzr.shape[0]:
                self._show_status("Please select a valid attachment atom.", "warning")
                return

            source_xyz = atoms_xyzr[atom_idx, :3]

            linker_length = float(self.av_properties.linker_length)
            linker_width = float(self.av_properties.linker_width)
            r1 = float(self.av_properties.radius_1)
            r2 = float(self.av_properties.radius_2)
            r3 = float(self.av_properties.radius_3)
            disc_step = float(self.av_properties.resolution)

            # Start non-blocking QThread worker
            self.av_preview_label.setText("AV: Computing...")
            self.av_progress_bar.setVisible(True)
            self.preview_av_btn.setEnabled(False)
            self.show_3d_btn.setEnabled(False)

            source_info = {
                "chain_identifier": str(self.atom_select.chain_id),
                "residue_seq_number": int(self.atom_select.residue_id),
                "atom_name": str(self.atom_select.atom_name),
            }

            self._av_worker = AVWorker(
                atoms_xyzr=atoms_xyzr,
                source_xyz=source_xyz,
                linker_length=linker_length,
                linker_width=linker_width,
                radii=(r1, r2, r3),
                disc_step=disc_step,
                pdb_path=self._pdb_path,
                source_info=source_info,
            )
            self._av_worker.result_ready.connect(self.onAVComputationFinished)
            self._av_worker.error.connect(self.onAVComputationError)
            self._av_worker.start()

        except Exception as e:
            self.av_preview_label.setText("AV: Calculation failed to start")
            self.av_progress_bar.setVisible(False)
            self.preview_av_btn.setEnabled(True)
            self._show_status(f"Failed to start AV calculation: {str(e)}", "error")

    def onAVComputationFinished(
        self, n_points: int, volume: float, mx: float, my: float, mz: float, coords: np.ndarray
    ) -> None:
        """Handle background AV computation success."""
        self.av_progress_bar.setVisible(False)
        self.preview_av_btn.setEnabled(True)
        self.show_3d_btn.setEnabled(True)

        self.av_preview_label.setText(
            f"AV preview result:\nPoints: {n_points}\nVolume: {volume:.1f} Å³\nMean: ({mx:.2f}, {my:.2f}, {mz:.2f})"
        )

        self._last_av_points = coords
        self._last_mean_xyz = np.array([mx, my, mz])
        self._last_pos_name = self.position_name_edit.text().strip() or "Position"

        # Cache the result
        self._av_cache[self._last_pos_name] = (self._last_av_points, self._last_mean_xyz)

        # Render immediately
        self.onShowIn3D()

    def onAVComputationError(self, err_msg: str) -> None:
        """Handle background AV computation failure."""
        self.av_progress_bar.setVisible(False)
        self.preview_av_btn.setEnabled(True)
        self.av_preview_label.setText("AV: Calculation failed.")
        self._show_status(f"AV Worker Error: {err_msg}", "error")

    def onShowIn3D(self) -> None:
        """Load the computed AV points and mean position into the MolView viewer."""
        if self._last_av_points is None or self._last_mean_xyz is None:
            return

        if self.mol_view_3d.isHidden():
            self.mol_view_3d.show()
            self.splitter.setSizes([350, 400])

        self.mol_view_3d.clear_point_overlays()

        # Display AV points as transparent green point overlay
        # coordinates are columns 0, 1, 2 of coords
        self.mol_view_3d.add_point_overlay(
            "av",
            self._last_av_points[:, :3],
            color=(0.0, 1.0, 0.5, 0.5),
            size_scale=0.015,
            min_size=1.0,
            alpha=0.5
        )

        # Display mean position as single sphere
        self.mol_view_3d.add_sphere(
            self._last_mean_xyz,
            radius=1.5,
            color=(1.0, 0.8, 0.2, 0.9),
            label=self._last_pos_name,
            key="mean_position"
        )

    def _optional_field(self) -> tuple[str, str] | None:
        """Return an optional key/value metadata field from the UI."""
        text = self.optional_field_edit.text().strip()
        if not text:
            return None
        if "=" not in text:
            raise ValueError("Optional field must use key=value syntax")
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("Optional field key must not be empty")
        return key, value

    def onAddPosition(self) -> None:
        """Extract configurations and emit position_added signal."""
        name = self.position_name_edit.text().strip()
        if not name:
            self._show_status("Please provide a position name.", "warning")
            return

        try:
            allowed_sphere_radius = float(chisurf.core.settings.fps.get('allowed_sphere_radius', 1.5))
            params = {
                "atom_name": str(self.atom_select.atom_name),
                "chain_identifier": str(self.atom_select.chain_id),
                "residue_seq_number": int(self.atom_select.residue_id),
                "residue_name": str(self.atom_select.residue_name),
                "attachment_atom_index": int(self.atom_select.atom_number),
                "allowed_sphere_radius": allowed_sphere_radius,
                "anchor_atoms": "",
                "chain_weighting": False,
                "contact_volume_thickness": 0,
                "contact_volume_trapped_fraction": -1,
                "simulation_type": str(self.simulation_type_combo.currentText()),
                "linker_length": float(self.av_properties.linker_length),
                "linker_width": float(self.av_properties.linker_width),
                "min_sphere_volume_fraction": 0,
                "radius1": float(self.av_properties.radius_1),
                "radius2": float(self.av_properties.radius_2),
                "radius3": float(self.av_properties.radius_3),
                "simulation_grid_resolution": float(self.av_properties.resolution),
                "strip_mask": "",
                "body_id": self.body_id_spin.value()
            }
            optional_field = self._optional_field()
            if optional_field is not None:
                key, value = optional_field
                params[key] = value
            self.position_added.emit(name, params)
        except Exception as e:
            self._show_status(f"Failed to add position: {str(e)}", "error")

    def onPositionsListDoubleClicked(self, model_index: QtCore.QModelIndex) -> None:
        """Double click removes the position from the data model and list."""
        item = self.positions_list.item(model_index.row())
        if item is None:
            return
        name = item.text()
        reply = QtWidgets.QMessageBox.question(
            self, "Remove Position?",
            f"Are you sure you want to remove position '{name}' and all distances referencing it?",
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.position_removed.emit(name)
            if name in self._av_cache:
                del self._av_cache[name]

    def onPositionSelectionChanged(self) -> None:
        """Update interface fields and 3D preview when active item changes."""
        items = self.positions_list.selectedItems()
        if not items:
            return
        name = items[0].text()
        self.position_name_edit.setText(name)

        # Retrieve cached AV visualization if available
        if name in self._av_cache:
            self._last_av_points, self._last_mean_xyz = self._av_cache[name]
            self._last_pos_name = name
            self.show_3d_btn.setEnabled(True)
            self.onShowIn3D()
        else:
            self._last_av_points = None
            self._last_mean_xyz = None
            self._last_pos_name = None
            self.show_3d_btn.setEnabled(False)
            self.mol_view_3d.clear_point_overlays()

    def update_positions(self, positions: dict[str, dict[str, Any]]) -> None:
        """Update the QListWidget with current positions.

        Parameters
        ----------
        positions : dict
            Dictionary of positions from FpsJsonModel.
        """
        self.positions_list.clear()
        self.positions_list.addItems(list(positions.keys()))

    def on_chimol_atom_selection_changed(self, selected_atom_indices):
        """Handle atom selection from the chimol view.

        Parameters
        ----------
        selected_atom_indices : list of int
            List of selected atom indices.
        """
        if not selected_atom_indices:
            return

        atom_index = selected_atom_indices[0]
        self.atom_select.set_atom_index(atom_index)

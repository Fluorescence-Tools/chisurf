from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import mdtraj as md
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

import chisurf
import chisurf.gui.widgets

from .olga_greedy import select_informative_pairs
from .traj_utils import (
    build_atom_pairs,
    build_sites,
    candidate_pairs_from_fps_json,
    compute_efficiencies,
    compute_efficiencies_from_fps_av,
    compute_efficiencies_per_pair_r0,
    load_fps_json,
    parse_residue_ranges,
    positions_from_fps_json,
    rmsd_matrix,
)


class FRETPairSelectionWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Optimal FRET Pair Selection")
        self.resize(1100, 700)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.traj_edit = QtWidgets.QLineEdit()
        self.traj_browse = QtWidgets.QPushButton("Browse")
        traj_row = QtWidgets.QHBoxLayout()
        traj_row.addWidget(self.traj_edit, 1)
        traj_row.addWidget(self.traj_browse)
        form.addRow("Trajectory", traj_row)

        self.top_edit = QtWidgets.QLineEdit()
        self.top_browse = QtWidgets.QPushButton("Browse")
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.top_edit, 1)
        top_row.addWidget(self.top_browse)
        form.addRow("Topology (optional)", top_row)

        self.fps_edit = QtWidgets.QLineEdit()
        self.fps_browse = QtWidgets.QPushButton("Browse")
        fps_row = QtWidgets.QHBoxLayout()
        fps_row.addWidget(self.fps_edit, 1)
        fps_row.addWidget(self.fps_browse)
        form.addRow("FPS JSON (optional)", fps_row)

        self.use_av_checkbox = QtWidgets.QCheckBox()
        self.use_av_checkbox.setChecked(True)
        form.addRow("Use AV backend (fps.json)", self.use_av_checkbox)

        self.av_samples_spin = QtWidgets.QSpinBox()
        self.av_samples_spin.setRange(100, 1000000)
        self.av_samples_spin.setValue(5000)
        form.addRow("AV samples", self.av_samples_spin)

        self.av_max_points_spin = QtWidgets.QSpinBox()
        self.av_max_points_spin.setRange(1000, 500000)
        self.av_max_points_spin.setValue(20000)
        form.addRow("Max AV points", self.av_max_points_spin)

        self.chain_edit = QtWidgets.QLineEdit("A")
        form.addRow("Chain", self.chain_edit)

        self.residues_edit = QtWidgets.QLineEdit("35-50,85-95,115-120")
        form.addRow("Residues (PDB resSeq)", self.residues_edit)

        self.atom_edit = QtWidgets.QComboBox()
        self.atom_edit.addItems(["CB", "CA"])
        form.addRow("Attachment atom", self.atom_edit)

        self.r0_spin = QtWidgets.QDoubleSpinBox()
        self.r0_spin.setRange(1.0, 200.0)
        self.r0_spin.setValue(52.0)
        self.r0_spin.setDecimals(2)
        form.addRow("R0 (Å)", self.r0_spin)

        self.err_spin = QtWidgets.QDoubleSpinBox()
        self.err_spin.setRange(0.0001, 1.0)
        self.err_spin.setValue(0.06)
        self.err_spin.setDecimals(4)
        form.addRow("Expected E error", self.err_spin)

        self.max_pairs_spin = QtWidgets.QSpinBox()
        self.max_pairs_spin.setRange(1, 999)
        self.max_pairs_spin.setValue(10)
        form.addRow("Max pairs", self.max_pairs_spin)

        self.unique_only = QtWidgets.QCheckBox()
        self.unique_only.setChecked(True)
        form.addRow("Unique pairs only", self.unique_only)

        self.stride_spin = QtWidgets.QSpinBox()
        self.stride_spin.setRange(1, 1000000)
        self.stride_spin.setValue(1)
        form.addRow("Stride", self.stride_spin)

        self.rmsd_sel_edit = QtWidgets.QLineEdit("name CA")
        form.addRow("RMSD atom selection", self.rmsd_sel_edit)

        self.run_btn = QtWidgets.QPushButton("Run selection")
        layout.addWidget(self.run_btn)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, 1)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["#", "Pair", "<<RMSD>> (Å)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.table)

        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Pairs added")
        self.plot.setLabel("left", "<<RMSD>> (Å)")
        self.plot.showGrid(x=True, y=True)
        plot_layout.addWidget(self.plot)
        splitter.addWidget(plot_container)
        splitter.setSizes([650, 450])

        btn_row = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_row)
        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.setEnabled(False)
        btn_row.addStretch(1)
        btn_row.addWidget(self.export_btn)

        self._last_pair_names = []
        self._last_selected = None
        self._last_decay = None

        self.traj_browse.clicked.connect(self._browse_traj)
        self.top_browse.clicked.connect(self._browse_top)
        self.fps_browse.clicked.connect(self._browse_fps)
        self.run_btn.clicked.connect(self._run)
        self.export_btn.clicked.connect(self._export)

    def _browse_traj(self):
        files = chisurf.gui.widgets.open_files("Open Trajectory", "All files (*)")
        if files:
            self.traj_edit.setText(str(files[0]))

    def _browse_top(self):
        files = chisurf.gui.widgets.open_files("Open Topology", "PDB files (*.pdb);;All files (*)")
        if files:
            self.top_edit.setText(str(files[0]))

    def _browse_fps(self):
        files = chisurf.gui.widgets.open_files("Open FPS JSON", "JSON-Files (*.fps.json);;All files (*)")
        if files:
            self.fps_edit.setText(str(files[0]))

    def _load_traj(self) -> md.Trajectory:
        traj_path = Path(self.traj_edit.text().strip())
        if not traj_path.exists():
            raise ValueError("Trajectory file does not exist")

        top_text = self.top_edit.text().strip()
        top_path = Path(top_text) if top_text else None

        stride = int(self.stride_spin.value())

        if top_path and top_path.exists():
            return md.load(str(traj_path), top=str(top_path), stride=stride)
        return md.load(str(traj_path), stride=stride)

    def _run(self):
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.run_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.table.setRowCount(0)
            self.plot.clear()

            traj = self._load_traj()

            fps_path_text = self.fps_edit.text().strip()
            if fps_path_text:
                fps_path = Path(fps_path_text)
            else:
                fps_path = None

            default_r0 = float(self.r0_spin.value())
            if fps_path is not None and fps_path.exists():
                fps = load_fps_json(fps_path)
                positions = positions_from_fps_json(traj, fps)
                if len(positions) < 2:
                    raise ValueError("fps.json did not yield at least 2 valid labeling positions")

                atom_pairs, pair_names, r0s, pair_position_names = candidate_pairs_from_fps_json(
                    positions=positions,
                    fps=fps,
                    default_r0_angstrom=default_r0,
                )

                if atom_pairs.shape[0] == 0:
                    raise ValueError("No candidate pairs found in fps.json")

                if self.use_av_checkbox.isChecked():
                    try:
                        effs = compute_efficiencies_from_fps_av(
                            traj=traj,
                            fps=fps,
                            pair_position_names=pair_position_names,
                            r0s_angstrom=r0s,
                            n_samples=int(self.av_samples_spin.value()),
                            max_points_per_av=int(self.av_max_points_spin.value()),
                        )
                    except Exception as av_exc:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "AV backend unavailable",
                            f"Failed to compute AV-based efficiencies (falling back to attachment-atom distances).\n\n{av_exc}",
                        )
                        effs = compute_efficiencies_per_pair_r0(traj, atom_pairs, r0s)
                else:
                    effs = compute_efficiencies_per_pair_r0(traj, atom_pairs, r0s)

            else:
                chain = self.chain_edit.text().strip()
                residue_numbers = parse_residue_ranges(self.residues_edit.text())
                if not residue_numbers:
                    raise ValueError("No residues specified")

                atom_name = str(self.atom_edit.currentText())
                fallback = "CA" if atom_name == "CB" else None

                sites = build_sites(
                    traj,
                    chain_spec=chain,
                    residue_numbers=residue_numbers,
                    atom_name=atom_name,
                    fallback_atom_name=fallback,
                )
                if len(sites) < 2:
                    raise ValueError("Need at least 2 valid residues with an attachment atom")

                atom_pairs, pair_names = build_atom_pairs(sites)

                effs = compute_efficiencies(traj, atom_pairs, r0_angstrom=default_r0)

            rmsd_sel = self.rmsd_sel_edit.text().strip() or None
            rmsds = rmsd_matrix(traj, atom_selection=rmsd_sel)

            max_pairs = int(self.max_pairs_spin.value())
            err = float(self.err_spin.value())
            unique_only = bool(self.unique_only.isChecked())

            selected, decay = select_informative_pairs(
                effs=effs,
                rmsds=rmsds,
                err=err,
                max_pairs=max_pairs,
                unique_only=unique_only,
            )

            self._last_pair_names = pair_names
            self._last_selected = selected
            self._last_decay = decay

            self._populate_results(pair_names, selected, decay)
            self.export_btn.setEnabled(True)

        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "FRET pair selection failed", str(exc))
        finally:
            self.run_btn.setEnabled(True)
            QtWidgets.QApplication.restoreOverrideCursor()

    def _populate_results(self, pair_names, selected, decay):
        self.table.setRowCount(int(selected.shape[0]))
        for i in range(int(selected.shape[0])):
            idx = int(selected[i])
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(pair_names[idx]))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{float(decay[i]):.2f}"))

        x = np.arange(1, len(decay) + 1, dtype=float)
        self.plot.plot(x, decay.astype(float), pen=pg.mkPen(width=2))

    def _export(self):
        if self._last_selected is None or self._last_decay is None:
            return

        out = chisurf.gui.widgets.save_file(description="Export pair selection", file_type="Text files (*.txt);;All files (*)")
        if not out:
            return

        path = Path(out)
        lines = ["#\tPair_added\t<<RMSD>>/A\n"]
        for i in range(int(self._last_selected.shape[0])):
            idx = int(self._last_selected[i])
            pair = self._last_pair_names[idx]
            lines.append(f"{i+1}\t{pair}\t{float(self._last_decay[i]):.2f}\n")

        path.write_text("".join(lines), encoding="utf-8")


if __name__ == "plugin":
    window = FRETPairSelectionWindow()
    window.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = FRETPairSelectionWindow()
    w.show()
    sys.exit(app.exec_())

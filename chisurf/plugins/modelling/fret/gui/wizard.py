"""Wizard user interface for the FRET modeling plugin.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, Optional
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

import chisurf
from chisurf.plugins.chimol.chimol.renderer.view import MolView
from chisurf.plugins.modelling.fps_json_editor.label_structure import LabelStructure
from ..core import av, docking, engine, io, results, sampling, screening, evaluate, pair_selection
from ..core.io import read_fps_json


class _DockWidget(QtWidgets.QWidget):
    """Docking tab widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_dock_results = []
        self._original_structures = []

        # Splitter to allow left controls and right 3D viewer
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        self.left_pane = QtWidgets.QWidget(self.splitter)
        left_layout = QtWidgets.QVBoxLayout(self.left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # File selection
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("fps.json:"), 0, 0)
        self.fps_path = QtWidgets.QLineEdit()
        self.fps_btn = QtWidgets.QPushButton("Browse...")
        self.fps_btn.clicked.connect(self._browse_fps)
        grid.addWidget(self.fps_path, 0, 1)
        grid.addWidget(self.fps_btn, 0, 2)

        grid.addWidget(QtWidgets.QLabel("PDB:"), 1, 0)
        self.pdb_path = QtWidgets.QLineEdit()
        self.pdb_path.textChanged.connect(self._update_initial_viewer)
        self.pdb_btn = QtWidgets.QPushButton("Browse...")
        self.pdb_btn.clicked.connect(self._browse_pdb)
        grid.addWidget(self.pdb_path, 1, 1)
        grid.addWidget(self.pdb_btn, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Output dir:"), 2, 0)
        self.out_path = QtWidgets.QLineEdit("dock_out")
        self.out_btn = QtWidgets.QPushButton("Browse...")
        self.out_btn.clicked.connect(self._browse_out)
        grid.addWidget(self.out_path, 2, 1)
        grid.addWidget(self.out_btn, 2, 2)
        left_layout.addLayout(grid)

        # Parameters
        params_group = QtWidgets.QGroupBox("Parameters")
        pgrid = QtWidgets.QGridLayout()
        pgrid.addWidget(QtWidgets.QLabel("Max iterations:"), 0, 0)
        self.max_iter = QtWidgets.QSpinBox()
        self.max_iter.setRange(100, 1_000_000)
        self.max_iter.setValue(50000)
        pgrid.addWidget(self.max_iter, 0, 1)
        pgrid.addWidget(QtWidgets.QLabel("Max force:"), 0, 2)
        self.max_force = QtWidgets.QDoubleSpinBox()
        self.max_force.setRange(0.1, 10000)
        self.max_force.setValue(100.0)
        pgrid.addWidget(self.max_force, 0, 3)
        pgrid.addWidget(QtWidgets.QLabel("Trials:"), 1, 0)
        self.n_trials = QtWidgets.QSpinBox()
        self.n_trials.setRange(1, 100)
        self.n_trials.setValue(3)
        pgrid.addWidget(self.n_trials, 1, 1)
        pgrid.addWidget(QtWidgets.QLabel("k_clash:"), 1, 2)
        self.k_clash = QtWidgets.QDoubleSpinBox()
        self.k_clash.setRange(0.01, 1000)
        self.k_clash.setValue(10.0)
        pgrid.addWidget(self.k_clash, 1, 3)
        params_group.setLayout(pgrid)
        left_layout.addWidget(params_group)

        # Run
        self.run_btn = QtWidgets.QPushButton("Run Docking")
        self.run_btn.clicked.connect(self._run)
        left_layout.addWidget(self.run_btn)

        # Results control (List Widget on the left)
        self.res_group = QtWidgets.QGroupBox("Docking Results")
        res_layout = QtWidgets.QVBoxLayout()
        self.trial_list = QtWidgets.QListWidget()
        self.trial_list.itemSelectionChanged.connect(self._on_trial_list_selection_changed)
        res_layout.addWidget(self.trial_list)
        self.res_group.setLayout(res_layout)
        self.res_group.hide()
        left_layout.addWidget(self.res_group)

        # Progress
        self.progress = QtWidgets.QProgressBar()

        # Log
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)

        # Right panel: 3D viewer (MolView)
        self.mol_view_3d = MolView(self.splitter)
        self.mol_view_3d.setMinimumWidth(300)
        self.mol_view_3d.hide()  # Collapsed by default

        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.mol_view_3d)
        self.splitter.setSizes([500, 500])

        # Main layout with splitter on top, and progress + log at the bottom
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.splitter, stretch=1)
        main_layout.addWidget(self.progress)
        main_layout.addWidget(self.log)

    def _browse_fps(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select fps.json", "", "fps.json (*.fps.json *.json)"
        )
        if fn:
            self.fps_path.setText(fn)

    def _browse_pdb(self):
        fns, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select PDB Files", "", "PDB (*.pdb *.ent)"
        )
        if fns:
            self.pdb_path.setText(", ".join(fns))

    def _browse_out(self):
        dn = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if dn:
            self.out_path.setText(dn)

    def _update_initial_viewer(self):
        pdb_path = self.pdb_path.text()
        if not pdb_path:
            self.mol_view_3d.hide()
            return

        try:
            # Disable non-standard residues filter globally so DNA/RNA residues are kept
            import chisurf as cs
            if hasattr(cs.core.settings, "structure_data"):
                cs.core.settings.structure_data.setdefault("IMP", {})["filter_non_standard_residues"] = False

            pdb_paths = [p.strip() for p in pdb_path.split(",") if p.strip()]
            self._original_structures = []
            atoms_list = []
            for p in pdb_paths:
                if os.path.exists(p):
                    s = chisurf.core.structure.Structure(p)
                    com = s.xyz.mean(axis=0) if s.xyz.shape[0] > 0 else np.zeros(3)
                    self._original_structures.append((s, s.xyz.copy(), com))
                    atoms_list.append(s.atoms)

            if atoms_list:
                combined_atoms = np.concatenate(atoms_list)
                combined_atoms['atom_id'] = np.arange(1, len(combined_atoms) + 1)

                s_combined = chisurf.core.structure.Structure()
                s_combined.atoms = combined_atoms

                self.mol_view_3d.set_structure(s_combined)
                self.mol_view_3d.show()
                self.splitter.setSizes([500, 500])
            else:
                self.mol_view_3d.hide()
        except Exception as e:
            print(f"Error loading initial structures in docking viewer: {e}")

    @QtCore.Slot()
    def _update_trial_list(self):
        self.trial_list.clear()
        if not self._last_dock_results:
            self.res_group.hide()
            return

        for i, r in enumerate(self._last_dock_results):
            status = "Converged" if r.converged else "No Conv"
            self.trial_list.addItem(f"Trial {i + 1} ({status}, Energy: {r.energy:.2f})")

        self.res_group.show()
        self.trial_list.setCurrentRow(0)
        self.load_trial_in_viewer(0)

    def _on_trial_list_selection_changed(self):
        selected_items = self.trial_list.selectedItems()
        if not selected_items:
            return
        index = self.trial_list.row(selected_items[0])
        if index >= 0:
            self.load_trial_in_viewer(index)

    def load_trial_in_viewer(self, trial_idx: int):
        if not self._last_dock_results or not self._original_structures:
            return

        sr = self._last_dock_results[trial_idx]
        combined_atoms_list = []

        for bi, (s_orig, orig_xyz, com) in enumerate(self._original_structures):
            if bi < len(sr.translations) and bi < len(sr.rotations):
                trans = sr.translations[bi]
                rot = sr.rotations[bi]
                # Reconstruct transformed coordinates
                transformed_xyz = (orig_xyz - com) @ rot.T + trans

                # Make a copy of s_orig.atoms
                atoms_copy = np.copy(s_orig.atoms)
                atoms_copy['xyz'] = transformed_xyz
                combined_atoms_list.append(atoms_copy)
            else:
                combined_atoms_list.append(s_orig.atoms)

        if combined_atoms_list:
            combined_atoms = np.concatenate(combined_atoms_list)
            combined_atoms['atom_id'] = np.arange(1, len(combined_atoms) + 1)

            s_combined = chisurf.core.structure.Structure()
            s_combined.atoms = combined_atoms

            self.mol_view_3d.set_structure(s_combined)
            self.mol_view_3d.show()

    def _log(self, msg: str):
        self.log.append(msg)
        QtWidgets.QApplication.processEvents()

    def _run(self):
        fps_path = self.fps_path.text()
        pdb_path = self.pdb_path.text()
        out_dir = self.out_path.text()
        if not fps_path or not pdb_path:
            self._log("Please select fps.json and PDB files")
            return

        os.makedirs(out_dir, exist_ok=True)
        self.run_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self._update_initial_viewer()

        def worker():
            try:
                positions, dists, _, _ = read_fps_json(fps_path)
                self._log(f"Loaded {len(positions)} positions, {len(dists)} distances")

                params = engine.SpringParameters(
                    max_iterations=self.max_iter.value(),
                    max_force=self.max_force.value(),
                    k_clash=self.k_clash.value(),
                )

                self._log("Starting docking...")
                dock_results, av_dict, body_list = docking.run_docking(
                    pdb_path, positions, dists,
                    params=params,
                    n_trials=self.n_trials.value(),
                )
                self._log(f"Docking done. Converged: "
                          f"{sum(1 for r in dock_results if r.converged)}/{len(dock_results)}")

                # Write results
                atoms_per_body = [b.atoms_local for b in body_list]
                results.write_docking_results_pdb(
                    dock_results, atoms_per_body, out_dir
                )
                summary = [
                    {
                        "trial": i,
                        "converged": sr.converged,
                        "iterations": sr.iterations,
                        "energy": sr.energy,
                        "clash_energy": sr.clash_energy,
                        "restraint_energy": sr.restraint_energy,
                    }
                    for i, sr in enumerate(dock_results)
                ]
                with open(os.path.join(out_dir, "summary.json"), "w") as f:
                    json.dump(summary, f, indent=2)
                self._log(f"Results written to {out_dir}/")

                self._last_dock_results = dock_results
                QtCore.QMetaObject.invokeMethod(
                    self, "_update_trial_list", QtCore.Qt.QueuedConnection
                )
            except Exception as e:
                self._log(f"Error: {e}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                self.progress.setRange(0, 100)
                self.progress.setValue(100)
                self.run_btn.setEnabled(True)

        threading.Thread(target=worker, daemon=True).start()



class _ScreenWidget(QtWidgets.QWidget):
    """Screening tab widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("fps.json:"), 0, 0)
        self.fps_path = QtWidgets.QLineEdit()
        self.fps_btn = QtWidgets.QPushButton("Browse...")
        self.fps_btn.clicked.connect(self._browse_fps)
        grid.addWidget(self.fps_path, 0, 1)
        grid.addWidget(self.fps_btn, 0, 2)

        grid.addWidget(QtWidgets.QLabel("PDB dir:"), 1, 0)
        self.pdb_dir = QtWidgets.QLineEdit()
        self.pdb_btn = QtWidgets.QPushButton("Browse...")
        self.pdb_btn.clicked.connect(self._browse_pdb_dir)
        grid.addWidget(self.pdb_dir, 1, 1)
        grid.addWidget(self.pdb_btn, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Output CSV:"), 2, 0)
        self.out_path = QtWidgets.QLineEdit("screening.csv")
        grid.addWidget(self.out_path, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Threads:"), 3, 0)
        self.n_threads = QtWidgets.QSpinBox()
        self.n_threads.setRange(1, 64)
        self.n_threads.setValue(4)
        grid.addWidget(self.n_threads, 3, 1)
        layout.addLayout(grid)

        self.run_btn = QtWidgets.QPushButton("Run Screening")
        self.run_btn.clicked.connect(self._run)
        layout.addWidget(self.run_btn)

        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["Rank", "Filename", "chi2", "chi2_red", "N_valid", "Viol1s", "Viol3s"]
        )
        layout.addWidget(self.table)

    def _browse_fps(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select fps.json", "", "fps.json (*.fps.json *.json)"
        )
        if fn:
            self.fps_path.setText(fn)

    def _browse_pdb_dir(self):
        dn = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select PDB directory"
        )
        if dn:
            self.pdb_dir.setText(dn)

    def _run(self):
        fps_path = self.fps_path.text()
        pdb_dir = self.pdb_dir.text()
        out_path = self.out_path.text()
        n_threads = self.n_threads.value()
        if not fps_path or not pdb_dir:
            QtWidgets.QMessageBox.warning(self, "Error", "Select fps.json and PDB dir")
            return

        self.run_btn.setEnabled(False)
        self.progress.setRange(0, 0)

        def worker():
            try:
                positions, dists, _, _ = read_fps_json(fps_path)
                scr_results = screening.screen_structure_library(
                    pdb_dir, positions, dists, n_threads=n_threads
                )
                results.write_screening_results_csv(scr_results, out_path)
                QtCore.QMetaObject.invokeMethod(
                    self, "_update_table", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(list, scr_results),
                )
            except Exception as e:
                import traceback
                QtCore.QMetaObject.invokeMethod(
                    self, "_log_error", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, traceback.format_exc()),
                )
            finally:
                QtCore.QMetaObject.invokeMethod(
                    self.progress, "setRange",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(int, 0), QtCore.Q_ARG(int, 100),
                )
                QtCore.QMetaObject.invokeMethod(
                    self.progress, "setValue",
                    QtCore.Qt.QueuedConnection, QtCore.Q_ARG(int, 100),
                )
                QtCore.QMetaObject.invokeMethod(
                    self.run_btn, "setEnabled",
                    QtCore.Qt.QueuedConnection, QtCore.Q_ARG(bool, True),
                )

        threading.Thread(target=worker, daemon=True).start()

    @QtCore.Slot(list)
    def _update_table(self, scr_results):
        self.table.setRowCount(len(scr_results))
        for i, r in enumerate(scr_results):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(r.filename))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{r.chi2:.4f}"))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{r.reduced_chi2:.4f}"))
            self.table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(r.n_valid)))
            self.table.setItem(i, 5, QtWidgets.QTableWidgetItem(str(r.n_violations_1sigma)))
            self.table.setItem(i, 6, QtWidgets.QTableWidgetItem(str(r.n_violations_3sigma)))
        self.table.resizeColumnsToContents()

    @QtCore.Slot(str)
    def _log_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Error", msg)


class _EvaluatorWidget(QtWidgets.QWidget):
    """Evaluators tab widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        # File selection
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("fps.json:"), 0, 0)
        self.fps_path = QtWidgets.QLineEdit()
        self.fps_btn = QtWidgets.QPushButton("Browse...")
        self.fps_btn.clicked.connect(self._browse_fps)
        grid.addWidget(self.fps_path, 0, 1)
        grid.addWidget(self.fps_btn, 0, 2)

        grid.addWidget(QtWidgets.QLabel("Input Type:"), 1, 0)
        self.input_type = QtWidgets.QComboBox()
        self.input_type.addItems(["Single PDB File", "PDB Directory", "MDTraj Trajectory"])
        self.input_type.currentTextChanged.connect(self._on_type_changed)
        grid.addWidget(self.input_type, 1, 1, 1, 2)

        self.input_label = QtWidgets.QLabel("PDB File:")
        grid.addWidget(self.input_label, 2, 0)
        self.input_path = QtWidgets.QLineEdit()
        self.input_btn = QtWidgets.QPushButton("Browse...")
        self.input_btn.clicked.connect(self._browse_input)
        grid.addWidget(self.input_path, 2, 1)
        grid.addWidget(self.input_btn, 2, 2)

        # Trajectory path (only for MDTraj Trajectory)
        self.traj_label = QtWidgets.QLabel("Trajectory (XTC/DCD):")
        self.traj_label.hide()
        self.traj_path = QtWidgets.QLineEdit()
        self.traj_btn = QtWidgets.QPushButton("Browse...")
        self.traj_btn.clicked.connect(self._browse_traj)
        self.traj_path.hide()
        self.traj_btn.hide()
        grid.addWidget(self.traj_label, 3, 0)
        grid.addWidget(self.traj_path, 3, 1)
        grid.addWidget(self.traj_btn, 3, 2)

        grid.addWidget(QtWidgets.QLabel("Output CSV:"), 4, 0)
        self.out_path = QtWidgets.QLineEdit("evaluation_results.csv")
        self.out_btn = QtWidgets.QPushButton("Browse...")
        self.out_btn.clicked.connect(self._browse_out)
        grid.addWidget(self.out_path, 4, 1)
        grid.addWidget(self.out_btn, 4, 2)

        # Active backend choice
        grid.addWidget(QtWidgets.QLabel("AV Backend:"), 5, 0)
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(["auto", "labellib", "imp-bff"])
        grid.addWidget(self.backend_combo, 5, 1, 1, 2)

        layout.addLayout(grid)

        # Run button
        self.run_btn = QtWidgets.QPushButton("Run Evaluation")
        self.run_btn.clicked.connect(self._run)
        layout.addWidget(self.run_btn)

        # Progress bar
        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)

        # Log output
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def _browse_fps(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select fps.json", "", "fps.json (*.fps.json *.json)"
        )
        if fn:
            self.fps_path.setText(fn)

    def _on_type_changed(self, text):
        if text == "Single PDB File":
            self.input_label.setText("PDB File:")
            self.traj_label.hide()
            self.traj_path.hide()
            self.traj_btn.hide()
        elif text == "PDB Directory":
            self.input_label.setText("PDB Directory:")
            self.traj_label.hide()
            self.traj_path.hide()
            self.traj_btn.hide()
        elif text == "MDTraj Trajectory":
            self.input_label.setText("Topology PDB:")
            self.traj_label.show()
            self.traj_path.show()
            self.traj_btn.show()

    def _browse_input(self):
        t = self.input_type.currentText()
        if t == "PDB Directory":
            dn = QtWidgets.QFileDialog.getExistingDirectory(self, "Select PDB directory")
            if dn:
                self.input_path.setText(dn)
        else:
            fn, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select PDB", "", "PDB (*.pdb *.ent)"
            )
            if fn:
                self.input_path.setText(fn)

    def _browse_traj(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Trajectory File", "", "Trajectory (*.xtc *.dcd *.trr *.nc)"
        )
        if fn:
            self.traj_path.setText(fn)

    def _browse_out(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select Output CSV", "evaluation_results.csv", "CSV Files (*.csv)"
        )
        if fn:
            self.out_path.setText(fn)

    def _log(self, msg: str):
        self.log.append(msg)
        QtWidgets.QApplication.processEvents()

    def _run(self):
        fps = self.fps_path.text()
        input_p = self.input_path.text()
        out = self.out_path.text()
        input_t = self.input_type.currentText()
        backend = self.backend_combo.currentText()

        if not fps or not input_p or not out:
            self._log("Please fill in all file paths.")
            return

        self.run_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.log.clear()

        def worker():
            try:
                av.select_backend(backend)
                self._log(f"Reading {fps}...")
                positions, dists, _, _ = io.read_fps_json(fps)
                evaluators = io.read_evaluators_json(fps)
                if not evaluators:
                    self._log("No evaluators defined in fps.json.")
                    return

                self._log(f"Loaded {len(evaluators)} evaluators.")
                
                if input_t == "Single PDB File":
                    self._log(f"Evaluating structure {input_p}...")
                    res = evaluate.evaluate_structure(input_p, positions, evaluators)
                    storage = evaluate.EvaluationStorage()
                    storage.add_frame(os.path.basename(input_p), res)
                    storage.to_csv(out)
                    self._log(f"Done. Wrote results to {out}")
                    for name, r in res.items():
                        self._log(f"  {name}: {r.value:.4f} {r.unit}")
                
                elif input_t == "PDB Directory":
                    self._log(f"Evaluating directory {input_p}...")
                    storage = evaluate.evaluate_directory(input_p, positions, evaluators)
                    storage.to_csv(out)
                    self._log(f"Done. Wrote results to {out}")
                
                elif input_t == "MDTraj Trajectory":
                    traj = self.traj_path.text()
                    if not traj:
                        self._log("Trajectory file is required.")
                        return
                    self._log(f"Evaluating trajectory {traj} with topology {input_p}...")
                    storage = evaluate.evaluate_trajectory(input_p, traj, positions, evaluators)
                    storage.to_csv(out)
                    self._log(f"Done. Wrote results to {out}")

            except Exception as e:
                self._log(f"Error during evaluation: {e}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                QtCore.QMetaObject.invokeMethod(self.progress, "setRange", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(int, 0), QtCore.Q_ARG(int, 100))
                QtCore.QMetaObject.invokeMethod(self.progress, "setValue", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(int, 100))
                QtCore.QMetaObject.invokeMethod(self.run_btn, "setEnabled", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(bool, True))

        threading.Thread(target=worker, daemon=True).start()


class _PairSelectWidget(QtWidgets.QWidget):
    """Pair selection tab widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        # File selection
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("fps.json:"), 0, 0)
        self.fps_path = QtWidgets.QLineEdit()
        self.fps_btn = QtWidgets.QPushButton("Browse...")
        self.fps_btn.clicked.connect(self._browse_fps)
        grid.addWidget(self.fps_path, 0, 1)
        grid.addWidget(self.fps_btn, 0, 2)

        grid.addWidget(QtWidgets.QLabel("PDB Directory:"), 1, 0)
        self.pdb_dir = QtWidgets.QLineEdit()
        self.pdb_btn = QtWidgets.QPushButton("Browse...")
        self.pdb_btn.clicked.connect(self._browse_pdb_dir)
        grid.addWidget(self.pdb_dir, 1, 1)
        grid.addWidget(self.pdb_btn, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Output Report:"), 2, 0)
        self.out_path = QtWidgets.QLineEdit("pair_selection_report.txt")
        self.out_btn = QtWidgets.QPushButton("Browse...")
        self.out_btn.clicked.connect(self._browse_out)
        grid.addWidget(self.out_path, 2, 1)
        grid.addWidget(self.out_btn, 2, 2)

        grid.addWidget(QtWidgets.QLabel("Max Pairs:"), 3, 0)
        self.max_pairs = QtWidgets.QSpinBox()
        self.max_pairs.setRange(1, 50)
        self.max_pairs.setValue(3)
        grid.addWidget(self.max_pairs, 3, 1, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Error (err, Å):"), 4, 0)
        self.err_spin = QtWidgets.QDoubleSpinBox()
        self.err_spin.setRange(0.1, 50.0)
        self.err_spin.setValue(5.0)
        grid.addWidget(self.err_spin, 4, 1, 1, 2)

        grid.addWidget(QtWidgets.QLabel("AV Backend:"), 5, 0)
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(["auto", "labellib", "imp-bff"])
        grid.addWidget(self.backend_combo, 5, 1, 1, 2)

        layout.addLayout(grid)

        # Run button
        self.run_btn = QtWidgets.QPushButton("Run Pair Selection")
        self.run_btn.clicked.connect(self._run)
        layout.addWidget(self.run_btn)

        # Progress bar
        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)

        # Log output
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def _browse_fps(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select fps.json", "", "fps.json (*.fps.json *.json)"
        )
        if fn:
            self.fps_path.setText(fn)

    def _browse_pdb_dir(self):
        dn = QtWidgets.QFileDialog.getExistingDirectory(self, "Select PDB directory")
        if dn:
            self.pdb_dir.setText(dn)

    def _browse_out(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select Output Report", "pair_selection_report.txt", "Text Files (*.txt)"
        )
        if fn:
            self.out_path.setText(fn)

    def _log(self, msg: str):
        self.log.append(msg)
        QtWidgets.QApplication.processEvents()

    def _run(self):
        fps = self.fps_path.text()
        pdb_dir = self.pdb_dir.text()
        out = self.out_path.text()
        max_pairs = self.max_pairs.value()
        err = self.err_spin.value()
        backend = self.backend_combo.currentText()

        if not fps or not pdb_dir or not out:
            self._log("Please select fps.json, PDB dir, and report output paths.")
            return

        self.run_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.log.clear()

        def worker():
            try:
                av.select_backend(backend)
                self._log(f"Reading {fps}...")
                positions, distances, _, _ = io.read_fps_json(fps)

                # 1. Compute RMSD matrix
                self._log("Computing RMSD matrix...")
                rmsds, filenames = pair_selection.compute_rmsd_matrix_from_pdb_dir(pdb_dir)

                # 2. Compute FRET efficiencies
                self._log("Computing FRET efficiencies...")
                effs, pair_names = pair_selection.compute_efficiency_matrix_from_evaluators(
                    pdb_dir, positions, distances
                )

                # 3. Pre-process NaN values
                self._log("Pre-processing NaN values...")
                effs_clean, rmsds_clean, valid_indices = pair_selection.preprocess_efficiency_matrix(
                    effs, rmsds
                )
                clean_pair_names = [pair_names[i] for i in valid_indices]

                # 4. Run greedy selection
                self._log(f"Selecting up to {max_pairs} informative pairs...")
                selected_indices, decay = pair_selection.select_informative_pairs(
                    effs_clean, rmsds_clean, err=err, max_pairs=max_pairs
                )
                selected_pair_names = [clean_pair_names[i] for i in selected_indices]

                # 5. Write decay report
                self._log("Writing report...")
                pair_selection.write_pair_selection_report(
                    selected_pair_names, decay, out, rmsds.mean()
                )
                self._log(f"Done! Report saved to {out}")
                self._log("\nSelected Pairs:")
                for idx, name in enumerate(selected_pair_names):
                    self._log(f"  {idx + 1}. {name}")

            except Exception as e:
                self._log(f"Error during pair selection: {e}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                QtCore.QMetaObject.invokeMethod(self.progress, "setRange", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(int, 0), QtCore.Q_ARG(int, 100))
                QtCore.QMetaObject.invokeMethod(self.progress, "setValue", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(int, 100))
                QtCore.QMetaObject.invokeMethod(self.run_btn, "setEnabled", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(bool, True))

        threading.Thread(target=worker, daemon=True).start()


class FretDockWizard(QtWidgets.QMainWindow):
    """Main docking & screening wizard window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FRET Docking & Screening")
        self.resize(1000, 750)

        self.tabs = QtWidgets.QTabWidget()
        self.label_structure = LabelStructure()
        self.dock_widget = _DockWidget()
        self.screen_widget = _ScreenWidget()
        self.evaluator_widget = _EvaluatorWidget()
        self.pair_select_widget = _PairSelectWidget()

        self.tabs.addTab(self.label_structure, "Edit fps.json")
        self.tabs.addTab(self.dock_widget, "Docking")
        self.tabs.addTab(self.screen_widget, "Screening")
        self.tabs.addTab(self.evaluator_widget, "Evaluators")
        self.tabs.addTab(self.pair_select_widget, "Pair Selection")
        self.setCentralWidget(self.tabs)

        # Create Menus
        mbar = self.menuBar()
        mbar.setNativeMenuBar(False)
        file_menu = mbar.addMenu("&File")

        load_proj_action = file_menu.addAction("Load Project...")
        load_proj_action.triggered.connect(self.onLoadProject)
        save_proj_action = file_menu.addAction("Save Project...")
        save_proj_action.triggered.connect(self.onSaveProject)

        file_menu.addSeparator()
        examples_menu = file_menu.addMenu("Load Example Project")
        
        load_hiv_rt_action = examples_menu.addAction("FPS (HIV:RT complex)")
        load_hiv_rt_action.triggered.connect(self.onLoadExampleHivRt)
        
        load_olga_action = examples_menu.addAction("Olga (md traj screening)")
        load_olga_action.triggered.connect(self.onLoadExampleOlga)

        file_menu.addSeparator()
        close_action = file_menu.addAction("Close")
        close_action.triggered.connect(self.close)

    def save_project(self, project_dir: str):
        """Save the FRET modeling wizard state as a ChiSurf project.

        Parameters
        ----------
        project_dir : str
            Path to target directory.
        """
        from chisurf.core.project import Project
        from ..core.io import write_fps_json
        import shutil

        os.makedirs(project_dir, exist_ok=True)

        # Copy and update reference PDB
        ref_pdb_path = self.label_structure.position_panel._pdb_path
        new_ref_pdbs = []
        if ref_pdb_path:
            for p in [x.strip() for x in ref_pdb_path.split(",") if x.strip()]:
                if os.path.exists(p):
                    basename = os.path.basename(p)
                    dest = os.path.join(project_dir, basename)
                    if os.path.abspath(p) != os.path.abspath(dest):
                        try:
                            shutil.copy(p, dest)
                        except Exception as e:
                            print(f"Error copying reference PDB: {e}")
                    new_ref_pdbs.append(dest)
                else:
                    new_ref_pdbs.append(p)
            self.label_structure.position_panel._pdb_path = ",".join(new_ref_pdbs)
            self.label_structure.position_panel.pdb_filename_edit.setText(",".join(new_ref_pdbs))

        # Copy and update docking PDBs
        dock_pdbs = self.dock_widget.pdb_path.text()
        new_dock_pdbs = []
        if dock_pdbs:
            for p in [x.strip() for x in dock_pdbs.split(",") if x.strip()]:
                if os.path.exists(p):
                    basename = os.path.basename(p)
                    dest = os.path.join(project_dir, basename)
                    if os.path.abspath(p) != os.path.abspath(dest):
                        try:
                            shutil.copy(p, dest)
                        except Exception as e:
                            print(f"Error copying docking PDB: {e}")
                    new_dock_pdbs.append(dest)
                else:
                    new_dock_pdbs.append(p)
            self.dock_widget.pdb_path.setText(", ".join(new_dock_pdbs))

        def to_rel(path_str: str, base_dir: str) -> str:
            if not path_str:
                return ""
            paths = [p.strip() for p in path_str.split(",") if p.strip()]
            rel_paths = []
            for p in paths:
                if os.path.isabs(p):
                    try:
                        r = os.path.relpath(p, base_dir)
                        if not r.startswith(".."):
                            rel_paths.append(r)
                        else:
                            rel_paths.append(p)
                    except Exception:
                        rel_paths.append(p)
                else:
                    rel_paths.append(p)
            return ",".join(rel_paths)

        def to_rel_single(path_str: str, base_dir: str) -> str:
            if not path_str:
                return ""
            if os.path.isabs(path_str):
                try:
                    r = os.path.relpath(path_str, base_dir)
                    if not r.startswith(".."):
                        return r
                except Exception:
                    pass
            return path_str

        # Write fps_json file
        fps_filename = "project.fps.json"
        fps_payload = self.label_structure.fps_json_payload
        write_fps_json(os.path.join(project_dir, fps_filename), 
                       fps_payload.get("Positions", {}),
                       fps_payload.get("Distances", {}),
                       fps_payload.get("χ²", {}),
                       {k: v for k, v in fps_payload.items() if k not in ("Positions", "Distances", "χ²")})

        ui_fret = {
            "fps_json": fps_payload,
            "fps_json_file": fps_filename,
            "reference_pdb": to_rel(self.label_structure.position_panel._pdb_path, project_dir),
            "docking": {
                "fps_path": to_rel_single(self.dock_widget.fps_path.text(), project_dir),
                "pdb_path": to_rel(self.dock_widget.pdb_path.text(), project_dir),
                "out_path": to_rel_single(self.dock_widget.out_path.text(), project_dir),
                "max_iter": self.dock_widget.max_iter.value(),
                "max_force": self.dock_widget.max_force.value(),
                "n_trials": self.dock_widget.n_trials.value(),
                "k_clash": self.dock_widget.k_clash.value(),
            },
            "screening": {
                "fps_path": to_rel_single(self.screen_widget.fps_path.text(), project_dir),
                "pdb_dir": to_rel_single(self.screen_widget.pdb_dir.text(), project_dir),
                "out_path": to_rel_single(self.screen_widget.out_path.text(), project_dir),
                "n_threads": self.screen_widget.n_threads.value(),
            },
            "evaluator": {
                "fps_path": to_rel_single(self.evaluator_widget.fps_path.text(), project_dir),
                "input_type": self.evaluator_widget.input_type.currentText(),
                "input_path": to_rel_single(self.evaluator_widget.input_path.text(), project_dir),
                "traj_path": to_rel_single(self.evaluator_widget.traj_path.text(), project_dir),
                "out_path": to_rel_single(self.evaluator_widget.out_path.text(), project_dir),
                "backend": self.evaluator_widget.backend_combo.currentText(),
            },
            "pair_select": {
                "fps_path": to_rel_single(self.pair_select_widget.fps_path.text(), project_dir),
                "pdb_dir": to_rel_single(self.pair_select_widget.pdb_dir.text(), project_dir),
                "out_path": to_rel_single(self.pair_select_widget.out_path.text(), project_dir),
                "max_pairs": self.pair_select_widget.max_pairs.value(),
                "err_spin": self.pair_select_widget.err_spin.value(),
                "backend": self.pair_select_widget.backend_combo.currentText(),
            }
        }

        project = Project(
            name=os.path.basename(project_dir) or "fret_project",
            description="FRET Docking and Screening Project",
            ui_state={"fret_dock_wizard": ui_fret}
        )
        project.save(project_dir)

    def load_project(self, project_dir: str):
        """Load FRET modeling wizard state from a ChiSurf project.

        Parameters
        ----------
        project_dir : str
            Path to source directory.
        """
        from chisurf.core.project import Project

        def to_abs(path_str: str, base_dir: str) -> str:
            if not path_str:
                return ""
            paths = [p.strip() for p in path_str.split(",") if p.strip()]
            abs_paths = []
            for p in paths:
                if not os.path.isabs(p):
                    abs_paths.append(os.path.abspath(os.path.join(base_dir, p)))
                else:
                    abs_paths.append(p)
            return ",".join(abs_paths)

        def to_abs_single(path_str: str, base_dir: str) -> str:
            if not path_str:
                return ""
            if not os.path.isabs(path_str):
                return os.path.abspath(os.path.join(base_dir, path_str))
            return path_str

        project = Project.load(project_dir)
        ui_fret = project.ui_state.get("fret_dock_wizard", {})

        # Load fps.json
        fps_payload = ui_fret.get("fps_json")
        fps_file = ui_fret.get("fps_json_file")
        if fps_file:
            fps_abs_file = os.path.join(project_dir, fps_file)
            if os.path.exists(fps_abs_file):
                try:
                    positions, distances, score_sets, extra = read_fps_json(fps_abs_file)
                    fps_payload = {}
                    if extra:
                        fps_payload.update(extra)
                    fps_payload["Positions"] = positions
                    fps_payload["Distances"] = distances
                    if score_sets:
                        fps_payload["χ²"] = score_sets
                except Exception as e:
                    print(f"Error loading {fps_file}: {e}")

        if fps_payload:
            self.label_structure.fps_json_payload = fps_payload

        # Load reference PDB structure
        ref_pdb = ui_fret.get("reference_pdb")
        if ref_pdb:
            ref_pdb_abs = to_abs(ref_pdb, project_dir)
            paths = [p.strip() for p in ref_pdb_abs.split(",") if p.strip()]
            if any(os.path.exists(p) for p in paths):
                try:
                    self.label_structure.position_panel.load_structure(ref_pdb_abs)
                except Exception as e:
                    print(f"Error loading reference PDB {ref_pdb_abs}: {e}")

        # Docking settings
        d_state = ui_fret.get("docking", {})
        self.dock_widget.fps_path.setText(to_abs_single(d_state.get("fps_path", ""), project_dir))
        self.dock_widget.pdb_path.setText(to_abs(d_state.get("pdb_path", ""), project_dir))
        self.dock_widget.out_path.setText(to_abs_single(d_state.get("out_path", ""), project_dir))
        if "max_iter" in d_state:
            self.dock_widget.max_iter.setValue(d_state["max_iter"])
        if "max_force" in d_state:
            self.dock_widget.max_force.setValue(d_state["max_force"])
        if "n_trials" in d_state:
            self.dock_widget.n_trials.setValue(d_state["n_trials"])
        if "k_clash" in d_state:
            self.dock_widget.k_clash.setValue(d_state["k_clash"])

        # Screening settings
        s_state = ui_fret.get("screening", {})
        self.screen_widget.fps_path.setText(to_abs_single(s_state.get("fps_path", ""), project_dir))
        self.screen_widget.pdb_dir.setText(to_abs_single(s_state.get("pdb_dir", ""), project_dir))
        self.screen_widget.out_path.setText(to_abs_single(s_state.get("out_path", ""), project_dir))
        if "n_threads" in s_state:
            self.screen_widget.n_threads.setValue(s_state["n_threads"])

        # Evaluator settings
        e_state = ui_fret.get("evaluator", {})
        self.evaluator_widget.fps_path.setText(to_abs_single(e_state.get("fps_path", ""), project_dir))
        if "input_type" in e_state:
            self.evaluator_widget.input_type.setCurrentText(e_state["input_type"])
        self.evaluator_widget.input_path.setText(to_abs_single(e_state.get("input_path", ""), project_dir))
        self.evaluator_widget.traj_path.setText(to_abs_single(e_state.get("traj_path", ""), project_dir))
        self.evaluator_widget.out_path.setText(to_abs_single(e_state.get("out_path", ""), project_dir))
        if "backend" in e_state:
            self.evaluator_widget.backend_combo.setCurrentText(e_state["backend"])

        # Pair select settings
        ps_state = ui_fret.get("pair_select", {})
        self.pair_select_widget.fps_path.setText(to_abs_single(ps_state.get("fps_path", ""), project_dir))
        self.pair_select_widget.pdb_dir.setText(to_abs_single(ps_state.get("pdb_dir", ""), project_dir))
        self.pair_select_widget.out_path.setText(to_abs_single(ps_state.get("out_path", ""), project_dir))
        if "max_pairs" in ps_state:
            self.pair_select_widget.max_pairs.setValue(ps_state["max_pairs"])
        if "err_spin" in ps_state:
            self.pair_select_widget.err_spin.setValue(ps_state["err_spin"])
        if "backend" in ps_state:
            self.pair_select_widget.backend_combo.setCurrentText(ps_state["backend"])

    def onSaveProject(self):
        dn = QtWidgets.QFileDialog.getExistingDirectory(self, "Select project directory to save")
        if dn:
            try:
                self.save_project(dn)
                QtWidgets.QMessageBox.information(self, "Success", f"Project saved to {dn}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save project: {e}")

    def onLoadProject(self):
        dn = QtWidgets.QFileDialog.getExistingDirectory(self, "Select project directory to load")
        if dn:
            try:
                self.load_project(dn)
                QtWidgets.QMessageBox.information(self, "Success", f"Project loaded from {dn}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load project: {e}")

    def onLoadExampleHivRt(self):
        base_dir = os.path.dirname(__file__)
        example_dir = os.path.abspath(os.path.join(base_dir, "..", "examples", "fps_hiv_rt"))
        try:
            self.load_project(example_dir)
            QtWidgets.QMessageBox.information(self, "Success", "Loaded FPS (HIV:RT complex) example project!")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load example project: {e}")

    def onLoadExampleOlga(self):
        base_dir = os.path.dirname(__file__)
        example_dir = os.path.abspath(os.path.join(base_dir, "..", "examples", "olga_t4l"))
        try:
            self.load_project(example_dir)
            QtWidgets.QMessageBox.information(self, "Success", "Loaded Olga (md traj screening) example project!")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load example project: {e}")


if __name__ == "plugin":
    window = FretDockWizard()
    window.show()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = FretDockWizard()
    w.show()
    sys.exit(app.exec_())

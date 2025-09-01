#!/usr/bin/env python3
"""
PyQt5 GUI for PTU Processing (using external CLI)

This application provides a graphical interface to the PTU processing CLI
(sm_image_mle.py or ptu_processor.py). Users can drag-and-drop PTU files,
check which to process, select an IRF file, adjust all CLI parameters via GUI
controls on a separate tab, and click "Process" to run the CLI in separate
subprocesses. Output logs (including the exact command-line call) appear in a
monospaced, tighter-spaced text pane in real time. A progress bar shows overall
progress across selected files. A new "Browser" tab displays processed molecules:
PTU file, molecule ID, molecule image, and fitted decay plot.
"""

import sys
import shlex
import subprocess
from pathlib import Path
from typing import List, Tuple

from chisurf.gui import QtCore, QtGui, QtWidgets


class FileListWidget(QtWidgets.QListWidget):
    """
    QListWidget subclass that accepts drag-and-drop of files.
    Each dropped file becomes an item with a checkbox, checked by default.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                if local_path:
                    p = Path(local_path)
                    if p.exists() and p.is_file() and p.suffix.lower() == ".ptu":
                        item = QtWidgets.QListWidgetItem(str(p))
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                        item.setCheckState(QtCore.Qt.Checked)  # Checked by default
                        self.addItem(item)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def selected_files(self) -> List[Path]:
        """
        Return a list of Path objects for items whose checkboxes are checked.
        """
        files: List[Path] = []
        for index in range(self.count()):
            item = self.item(index)
            if item.checkState() == QtCore.Qt.Checked:
                files.append(Path(item.text()))
        return files


class Worker(QtCore.QThread):
    """
    QThread worker that runs the PTU CLI in separate subprocesses.

    It emits:
      - message(str): for log lines and command display
      - progress(int): percentage (0-100)
      - finished(): when done
    """
    message = QtCore.Signal(str)
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()

    def __init__(
        self,
        files: List[Path],
        irf_file: Path,
        detector_chs: List[int],
        micro_time_range: Tuple[int, int],
        micro_time_binning: int,
        normalize_counts: int,
        threshold: float,
        minlength: int,
        shift_sp: float,
        shift_ss: float,
        irf_threshold_fraction: float,
        fit_initial_values: Tuple[float, float, float, float],
        fit_fixed_flags: Tuple[int, int, int, int],
        l1: float,
        l2: float,
        twoi_star: bool,
        bifl_scatter: bool,
        # ← New segmentation parameters:
        seg_sigma: float,
        seg_threshold: float,
        peak_footprint_size: int,
        cli_script: Path
    ):
        super().__init__()
        self.files = files
        self.irf_file = irf_file
        self.detector_chs = detector_chs
        self.micro_time_range = micro_time_range
        self.micro_time_binning = micro_time_binning
        self.normalize_counts = normalize_counts
        self.threshold = threshold
        self.minlength = minlength
        self.shift_sp = shift_sp
        self.shift_ss = shift_ss
        self.irf_threshold_fraction = irf_threshold_fraction
        self.fit_initial_values = fit_initial_values
        self.fit_fixed_flags = fit_fixed_flags
        self.l1 = l1
        self.l2 = l2
        self.twoi_star = twoi_star
        self.bifl_scatter = bifl_scatter
        # ← Store new segmentation parameters:
        self.seg_sigma = seg_sigma
        self.seg_threshold = seg_threshold
        self.peak_footprint_size = peak_footprint_size

        self.cli_script = cli_script

    def run(self):
        # Verify CLI script exists
        if not self.cli_script.exists():
            self.message.emit(f"Error: CLI script not found at {self.cli_script}\n")
            self.finished.emit()
            return

        all_output_paths = []
        total_files = len(self.files)
        for idx, ptu_path in enumerate(self.files):
            parent_dir = ptu_path.parent.resolve()
            pattern = ptu_path.name

            self.message.emit(f"\n=== Processing {pattern} ===\n")

            # Build command line arguments
            args = [
                sys.executable,
                str(self.cli_script),
                "--ptu-pattern", pattern,
                "--irf-file", str(self.irf_file),
            ]
            # detector channels: repeated flags
            for ch in self.detector_chs:
                args.extend(["--detector-chs", str(ch)])

            # micro-time range (nargs=2)
            args.extend([
                "--micro-time-range",
                str(self.micro_time_range[0]),
                str(self.micro_time_range[1])
            ])
            # micro-time binning
            args.extend(["--micro-time-binning", str(self.micro_time_binning)])
            # normalize counts
            args.extend(["--normalize-counts", str(self.normalize_counts)])
            # threshold
            args.extend(["--threshold", str(self.threshold)])
            # minlength
            args.extend(["--minlength", str(self.minlength)])
            # shifts
            args.extend(["--shift-sp", str(self.shift_sp)])
            args.extend(["--shift-ss", str(self.shift_ss)])
            # IRF threshold fraction
            args.extend(["--irf-threshold-fraction", str(self.irf_threshold_fraction)])
            # fit initial values (nargs=4)
            args.append("--fit-initial-values")
            for val in self.fit_initial_values:
                args.append(str(val))
            # fit fixed flags (nargs=4)
            args.append("--fit-fixed-flags")
            for flag in self.fit_fixed_flags:
                args.append(str(flag))
            # l1, l2
            args.extend(["--l1", str(self.l1)])
            args.extend(["--l2", str(self.l2)])
            # twoi-star / bifl-scatter
            if self.twoi_star:
                args.append("--twoi-star")
            else:
                args.append("--no-twoi-star")
            if self.bifl_scatter:
                args.append("--bifl-scatter")
            else:
                args.append("--no-bifl-scatter")

            # ─────── New segmentation flags ───────
            args.extend(["--seg-sigma", str(self.seg_sigma)])
            args.extend(["--seg-threshold", str(self.seg_threshold)])
            args.extend(["--peak-footprint-size", str(self.peak_footprint_size)])
            # ────────────────────────────────────────

            # ─────── Modified: write combined TSV as <ptu_basename>.tsv in parent folder ───────
            output_file = parent_dir / f"{ptu_path.stem}_analysis/molecule_data.tsv"
            all_output_paths.append(output_file)  # ← Track output files
            args.extend(["--output-file", str(output_file)])
            # ─────────────────────────────────────────────────────────────────────────────────────

            # output-dir = parent folder
            args.extend(["--output-dir", str(parent_dir)])

            # Display the exact command-line call
            cmd_str = shlex.join(args)
            self.message.emit(f"Running command:\n{cmd_str}\n\n")

            # Launch subprocess with UTF-8 decoding and replace errors
            try:
                process = subprocess.Popen(
                    args,
                    cwd=str(parent_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1
                )
            except Exception as e:
                self.message.emit(f"Failed to start process for {pattern}: {e}\n")
                # Update progress even on failure
                percent = int(((idx + 1) / total_files) * 100)
                self.progress.emit(percent)
                continue

            # Read stdout in real time
            if process.stdout:
                for line in process.stdout:
                    self.message.emit(line)
                process.wait()
                self.message.emit(f"Process exited with code {process.returncode}\n")
            else:
                self.message.emit("No output captured.\n")

            # Update progress
            percent = int(((idx + 1) / total_files) * 100)
            self.progress.emit(percent)

        self.message.emit("\n=== All tasks completed ===\n")

        # ───── Combine all individual TSVs ─────
        try:
            import pandas as pd

            dfs = []
            for path in all_output_paths:
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    dfs.append(df)
                else:
                    self.message.emit(f"Warning: TSV not found: {path}\n")

            if dfs:
                joint_df = pd.concat(dfs, ignore_index=True)
                joint_path = Path(self.files[0].parent) / "joint_output.tsv"
                joint_df.to_csv(joint_path, sep="\t", index=False)
                self.message.emit(f"\nJoint output written to: {joint_path}\n")
            else:
                self.message.emit("No valid TSVs found to join.\n")

        except Exception as e:
            self.message.emit(f"Failed to combine TSV files: {e}\n")
        # ───────────────────────────────────────

        self.finished.emit()


class BrowserWidget(QtWidgets.QWidget):
    """
    A widget to browse processed molecules via a spinbox.
    Shows PTU filename, Molecule ID, file paths, and displays images.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        # Main vertical layout with zero spacing/margins
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        form = QtWidgets.QFormLayout()
        form.setSpacing(0)
        form.setContentsMargins(0, 0, 0, 0)

        self.ptu_line = QtWidgets.QLineEdit()
        self.ptu_line.setReadOnly(True)
        form.addRow("PTU File:", self.ptu_line)
        self.mol_line = QtWidgets.QLineEdit()
        self.mol_line.setReadOnly(True)
        form.addRow("Molecule ID:", self.mol_line)
        self.img_path_line = QtWidgets.QLineEdit()
        self.img_path_line.setReadOnly(True)
        form.addRow("Molecule Image:", self.img_path_line)
        self.decay_path_line = QtWidgets.QLineEdit()
        self.decay_path_line.setReadOnly(True)
        form.addRow("Decay Plot:", self.decay_path_line)
        layout.addLayout(form)

        # Spinbox to select index
        spin_layout = QtWidgets.QHBoxLayout()
        spin_layout.setSpacing(0)
        spin_layout.setContentsMargins(0, 0, 0, 0)
        spin_layout.addWidget(QtWidgets.QLabel("Index:"))
        self.spin = QtWidgets.QSpinBox()
        self.spin.setMinimum(0)
        self.spin.valueChanged.connect(self.update_display)
        spin_layout.addWidget(self.spin)
        spin_layout.addStretch()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_list)
        spin_layout.addWidget(self.refresh_btn)
        layout.addLayout(spin_layout)

        # Image display
        img_layout = QtWidgets.QHBoxLayout()
        img_layout.setSpacing(0)
        img_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_img = QtWidgets.QLabel()
        self.lbl_img.setAlignment(QtCore.Qt.AlignCenter)
        img_layout.addWidget(self.lbl_img)
        self.lbl_decay = QtWidgets.QLabel()
        self.lbl_decay.setAlignment(QtCore.Qt.AlignCenter)
        img_layout.addWidget(self.lbl_decay)
        layout.addLayout(img_layout)

        self.entries: List[Tuple[Path, str, Path, Path]] = []  # (ptu, mol_id, img, decay)

    def load_list(self):
        self.entries.clear()
        files = self.parent.file_list.selected_files()
        for ptu in files:
            base = ptu.stem
            analysis = ptu.parent / f"{base}_analysis"
            img_dir = analysis / "molecule_images"
            decay_dir = analysis / "jordi_images"
            if not analysis.exists():
                continue
            for img in sorted(img_dir.glob("molecule_*.png")):
                mol_id = img.stem.split("_")[1]
                decay = decay_dir / f"molecule_{mol_id}_jordi.png"
                self.entries.append((ptu, mol_id, img, decay))
        self.spin.setMaximum(max(len(self.entries)-1, 0))
        self.spin.setValue(0)
        self.update_display(0)

    def update_display(self, idx: int):
        if not self.entries:
            self.ptu_line.clear()
            self.mol_line.clear()
            self.img_path_line.clear()
            self.decay_path_line.clear()
            self.lbl_img.clear()
            self.lbl_decay.clear()
            return
        idx = max(0, min(idx, len(self.entries)-1))
        ptu, mol_id, img, decay = self.entries[idx]
        self.ptu_line.setText(str(ptu.name))
        self.mol_line.setText(mol_id)
        self.img_path_line.setText(str(img))
        self.decay_path_line.setText(str(decay))
        pix = QtGui.QPixmap(str(img))
        if not pix.isNull():
            pix = pix.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
            self.lbl_img.setPixmap(pix)
        else:
            self.lbl_img.clear()
        pix2 = QtGui.QPixmap(str(decay))
        if not pix2.isNull():
            pix2 = pix2.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
            self.lbl_decay.setPixmap(pix2)
        else:
            self.lbl_decay.clear()


class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window with three tabs:
      - Files: for input PTU list, IRF selection, progress bar, log output, process button
      - Settings: for all CLI parameters (including the three new segmentation fields)
      - Browser: view processed molecule images and decay plots
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PTU Processor GUI")
        self.resize(980, 600)
        self._central = QtWidgets.QWidget()
        self.setCentralWidget(self._central)

        # Main vertical layout with zero spacing/margins
        main_layout = QtWidgets.QVBoxLayout(self._central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget
        self.tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tabs)

        # --- Tab 1: Files ---
        self.tab_files = QtWidgets.QWidget()
        files_layout = QtWidgets.QVBoxLayout(self.tab_files)
        files_layout.setSpacing(0)
        files_layout.setContentsMargins(0, 0, 0, 0)

        self.file_list = FileListWidget()
        self.file_list.setFixedHeight(150)
        files_layout.addWidget(QtWidgets.QLabel("Drag & drop .ptu files below:"))
        files_layout.addWidget(self.file_list)

        irf_layout = QtWidgets.QHBoxLayout()
        irf_layout.setSpacing(0)
        irf_layout.setContentsMargins(0, 0, 0, 0)
        irf_label = QtWidgets.QLabel("IRF file:")
        self.irf_lineedit = QtWidgets.QLineEdit()
        self.irf_browse_btn = QtWidgets.QPushButton("Browse")
        self.irf_browse_btn.clicked.connect(self.browse_irf)
        irf_layout.addWidget(irf_label)
        irf_layout.addWidget(self.irf_lineedit)
        irf_layout.addWidget(self.irf_browse_btn)
        files_layout.addLayout(irf_layout)

        out_label = QtWidgets.QLabel("(Each PTU’s output directory will be its parent folder)")
        files_layout.addWidget(out_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        files_layout.addWidget(self.progress_bar)

        self.process_btn = QtWidgets.QPushButton("Process")
        self.process_btn.clicked.connect(self.start_processing)
        files_layout.addWidget(self.process_btn)

        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        font = QtGui.QFont("Courier New")
        font.setStyleHint(QtGui.QFont.Monospace)
        font.setPointSize(10)
        self.log_text.setFont(font)
        # Remove any extra viewport margins
        self.log_text.setViewportMargins(0, 0, 0, 0)
        files_layout.addWidget(QtWidgets.QLabel("Log Output:"))
        files_layout.addWidget(self.log_text)

        self.tabs.addTab(self.tab_files, "Files")

        # --- Tab 2: Settings ---
        self.tab_settings = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(self.tab_settings)
        settings_layout.setSpacing(0)
        settings_layout.setContentsMargins(0, 0, 0, 0)

        params_group = QtWidgets.QGroupBox("Parameters")
        params_layout = QtWidgets.QFormLayout(params_group)
        params_layout.setSpacing(0)
        params_layout.setContentsMargins(0, 0, 0, 0)

        # Detector channels
        self.detector_lineedit = QtWidgets.QLineEdit("2 0")
        params_layout.addRow("Detector channels (e.g. '2 0'):", self.detector_lineedit)

        # Micro-time range
        self.mtr_start_spin = QtWidgets.QSpinBox()
        self.mtr_start_spin.setRange(0, 10000)
        self.mtr_start_spin.setValue(0)
        self.mtr_stop_spin = QtWidgets.QSpinBox()
        self.mtr_stop_spin.setRange(0, 10000)
        self.mtr_stop_spin.setValue(256)
        mtr_layout = QtWidgets.QHBoxLayout()
        mtr_layout.setSpacing(0)
        mtr_layout.setContentsMargins(0, 0, 0, 0)
        mtr_layout.addWidget(QtWidgets.QLabel("Start:"))
        mtr_layout.addWidget(self.mtr_start_spin)
        mtr_layout.addSpacing(10)
        mtr_layout.addWidget(QtWidgets.QLabel("Stop:"))
        mtr_layout.addWidget(self.mtr_stop_spin)
        params_layout.addRow("Micro-time range:", mtr_layout)

        # Micro-time binning
        self.mtb_spin = QtWidgets.QSpinBox()
        self.mtb_spin.setRange(1, 10000)
        self.mtb_spin.setValue(32)
        params_layout.addRow("Micro-time binning:", self.mtb_spin)

        # Normalize counts
        self.norm_combo = QtWidgets.QComboBox()
        self.norm_combo.addItems([
            "0: no normalization",
            "1: average rate",
            "2: per-channel",
            "3: by acquisition time"
        ])
        params_layout.addRow("Normalize counts:", self.norm_combo)

        # Jordi threshold
        self.thresh_spin = QtWidgets.QDoubleSpinBox()
        self.thresh_spin.setRange(-1.0, 1.0)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setValue(-1.0)
        params_layout.addRow("Threshold (fraction):", self.thresh_spin)

        # Min length
        self.minlength_spin = QtWidgets.QSpinBox()
        self.minlength_spin.setRange(-1, 100000)
        self.minlength_spin.setValue(-1)
        params_layout.addRow("Min length (histogram):", self.minlength_spin)

        # Shift SP
        self.shift_sp_spin = QtWidgets.QDoubleSpinBox()
        self.shift_sp_spin.setRange(-1000.0, 1000.0)
        self.shift_sp_spin.setSingleStep(0.1)
        self.shift_sp_spin.setValue(0.0)
        params_layout.addRow("Shift SP (bins):", self.shift_sp_spin)

        # Shift SS
        self.shift_ss_spin = QtWidgets.QDoubleSpinBox()
        self.shift_ss_spin.setRange(-1000.0, 1000.0)
        self.shift_ss_spin.setSingleStep(0.1)
        self.shift_ss_spin.setValue(0.0)
        params_layout.addRow("Shift SS (bins):", self.shift_ss_spin)

        # IRF threshold fraction
        self.irf_thresh_spin = QtWidgets.QDoubleSpinBox()
        self.irf_thresh_spin.setRange(0.0, 1.0)
        self.irf_thresh_spin.setSingleStep(0.01)
        self.irf_thresh_spin.setValue(0.08)
        params_layout.addRow("IRF threshold fraction:", self.irf_thresh_spin)

        # Fit initial values (τ, γ, ρ, scale)
        fit_init_layout = QtWidgets.QHBoxLayout()
        fit_init_layout.setSpacing(0)
        fit_init_layout.setContentsMargins(0, 0, 0, 0)
        self.fit_init_spins = []
        for default in [1.0, 0.0, 0.38, 1.0]:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1000.0, 1000.0)
            spin.setSingleStep(0.01)
            spin.setValue(default)
            fit_init_layout.addWidget(spin)
            self.fit_init_spins.append(spin)
        params_layout.addRow("Fit initial values (τ, γ, ρ, scale):", fit_init_layout)

        # Fit fixed flags
        fit_flag_layout = QtWidgets.QHBoxLayout()
        fit_flag_layout.setSpacing(0)
        fit_flag_layout.setContentsMargins(0, 0, 0, 0)
        self.fit_flag_checks = []
        for i in range(4):
            chk = QtWidgets.QCheckBox(f"Fix {i}")
            if i == 2:
                chk.setChecked(True)
            fit_flag_layout.addWidget(chk)
            self.fit_flag_checks.append(chk)
        params_layout.addRow("Fit fixed flags:", fit_flag_layout)

        # l1 parameter
        self.l1_spin = QtWidgets.QDoubleSpinBox()
        self.l1_spin.setRange(0.0, 100.0)
        self.l1_spin.setSingleStep(0.01)
        self.l1_spin.setValue(0.04)
        params_layout.addRow("Fit parameter l1:", self.l1_spin)

        # l2 parameter
        self.l2_spin = QtWidgets.QDoubleSpinBox()
        self.l2_spin.setRange(0.0, 100.0)
        self.l2_spin.setSingleStep(0.01)
        self.l2_spin.setValue(0.04)
        params_layout.addRow("Fit parameter l2:", self.l2_spin)

        # twoI* checkbox
        self.twoistar_check = QtWidgets.QCheckBox("Enable twoI* flag")
        self.twoistar_check.setChecked(True)
        params_layout.addRow(self.twoistar_check)

        # bifl-scatter checkbox
        self.biflscat_check = QtWidgets.QCheckBox("Enable bifl-scatter flag")
        self.biflscat_check.setChecked(False)
        params_layout.addRow(self.biflscat_check)

        # ───── New Segmentation Parameters ─────

        # seg-sigma
        self.seg_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.seg_sigma_spin.setRange(0.0, 100.0)
        self.seg_sigma_spin.setSingleStep(0.1)
        self.seg_sigma_spin.setValue(1.0)
        params_layout.addRow("Segmentation σ (seg-sigma):", self.seg_sigma_spin)

        # seg-threshold
        self.seg_thresh_spin = QtWidgets.QDoubleSpinBox()
        self.seg_thresh_spin.setRange(-1.0, 65535.0)
        self.seg_thresh_spin.setSingleStep(0.1)
        self.seg_thresh_spin.setValue(-1.0)
        params_layout.addRow(
            "Segmentation threshold (seg-threshold):",
            self.seg_thresh_spin
        )

        # peak-footprint-size
        self.peak_footprint_spin = QtWidgets.QSpinBox()
        self.peak_footprint_spin.setRange(1, 100)
        self.peak_footprint_spin.setValue(6)
        params_layout.addRow(
            "Peak footprint size (peak-footprint-size):",
            self.peak_footprint_spin
        )

        # ─────────────────────────────────────────

        settings_layout.addWidget(params_group)
        settings_layout.addStretch()
        self.tabs.addTab(self.tab_settings, "Settings")

        # --- Tab 3: Browser ---
        self.tab_browser = BrowserWidget(self)
        self.tabs.addTab(self.tab_browser, "Browser")

        self.worker = None

        # Find CLI script (ptu_processor.py or sm_image_mle.py)
        candidates = ["ptu_processor.py", "sm_image_mle.py"]
        found = None
        for name in candidates:
            path = Path(__file__).parent / name
            if path.exists():
                found = path
                break
        self.cli_script = found
        if not self.cli_script:
            QtWidgets.QMessageBox.critical(
                self, "Error", "Cannot find CLI script (ptu_processor.py or sm_image_mle.py) in this folder."
            )
            self.process_btn.setEnabled(False)

    def browse_irf(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select IRF PTU File", str(Path.cwd()), "PTU Files (*.ptu);;All Files (*)"
        )
        if path:
            self.irf_lineedit.setText(path)

    def append_log(self, text: str):
        self.log_text.appendPlainText(text)

    def update_progress(self, val: int):
        self.progress_bar.setValue(val)

    def start_processing(self):
        self.log_text.clear()
        self.progress_bar.setValue(0)

        files = self.file_list.selected_files()
        if not files:
            QtWidgets.QMessageBox.warning(self, "No Files Selected", "Please check at least one .ptu file.")
            return

        irf_path_str = self.irf_lineedit.text().strip()
        if not irf_path_str:
            QtWidgets.QMessageBox.warning(self, "No IRF File", "Please select an IRF file.")
            return
        irf_path = Path(irf_path_str)
        if not irf_path.exists() or not irf_path.is_file():
            QtWidgets.QMessageBox.warning(self, "Invalid IRF File", "Please select a valid IRF .ptu file.")
            return

        det_text = self.detector_lineedit.text().strip()
        try:
            detector_chs = [int(x) for x in det_text.split()]
            if not detector_chs:
                raise ValueError("No channels provided")
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Invalid Detector Channels", "Enter space-separated integers, e.g. '2 0'.")
            return

        mtr_start = self.mtr_start_spin.value()
        mtr_stop = self.mtr_stop_spin.value()
        micro_time_range = (mtr_start, mtr_stop)

        micro_time_binning = self.mtb_spin.value()
        normalize_counts = self.norm_combo.currentIndex()
        threshold = self.thresh_spin.value()
        minlength = self.minlength_spin.value()
        shift_sp = self.shift_sp_spin.value()
        shift_ss = self.shift_ss_spin.value()
        irf_threshold_fraction = self.irf_thresh_spin.value()
        fit_initial_values = tuple(spin.value() for spin in self.fit_init_spins)
        fit_fixed_flags = tuple(1 if chk.isChecked() else 0 for chk in self.fit_flag_checks)
        l1 = self.l1_spin.value()
        l2 = self.l2_spin.value()
        twoi_star = self.twoistar_check.isChecked()
        bifl_scatter = self.biflscat_check.isChecked()

        # ───── Read new segmentation values ─────
        seg_sigma = self.seg_sigma_spin.value()
        seg_threshold = self.seg_thresh_spin.value()
        peak_footprint_size = self.peak_footprint_spin.value()
        # ────────────────────────────────────────

        self.process_btn.setEnabled(False)

        # Instantiate Worker with all parameters (including segmentation)
        self.worker = Worker(
            files=files,
            irf_file=irf_path,
            detector_chs=detector_chs,
            micro_time_range=micro_time_range,
            micro_time_binning=micro_time_binning,
            normalize_counts=normalize_counts,
            threshold=threshold,
            minlength=minlength,
            shift_sp=shift_sp,
            shift_ss=shift_ss,
            irf_threshold_fraction=irf_threshold_fraction,
            fit_initial_values=fit_initial_values,
            fit_fixed_flags=fit_fixed_flags,
            l1=l1,
            l2=l2,
            twoi_star=twoi_star,
            bifl_scatter=bifl_scatter,
            # Pass segmentation values here:
            seg_sigma=seg_sigma,
            seg_threshold=seg_threshold,
            peak_footprint_size=peak_footprint_size,
            cli_script=self.cli_script
        )
        self.worker.message.connect(self.append_log)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self):
        self.process_btn.setEnabled(True)
        self.append_log("Worker thread finished.\n")
        self.progress_bar.setValue(100)
        # Automatically refresh browser after processing
        self.tab_browser.load_list()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


if __name__ == "plugin":
    window = MainWindow()
    window.show()

name = "Imaging:Single-Molecule MLE"

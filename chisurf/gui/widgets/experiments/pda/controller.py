import pathlib
import tttrlib

import chisurf.gui
from chisurf.experiments import reader

from chisurf.gui import QtWidgets, QtGui, QtCore
import chisurf
from chisurf import logging
from chisurf.macros import core_data as core_data_macros
from chisurf.experiments.pda import PdaReader

# Reuse the setups loader from the DetectorWizard
from chisurf.gui.widgets.wizard.tttr_channel_definition import load_detector_setups


class PdaTTTRWidget(
    QtWidgets.QWidget,
    reader.ExperimentReaderController
):

    class DropFileList(QtWidgets.QListWidget):
        def __init__(self, parent=None, accept_exts=None, dir_resolver=None):
            super().__init__(parent)
            self.setAcceptDrops(True)
            self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
            self.setDefaultDropAction(QtCore.Qt.CopyAction)
            self.setDropIndicatorShown(True)
            # accepted extensions in lowercase with dot prefix
            self.accept_exts = set(accept_exts or [])
            # optional callable(pathlib.Path) -> List[str] to expand directories
            self.dir_resolver = dir_resolver

        def _maybe_add_path(self, file_path: str):
            p = pathlib.Path(file_path)
            # If a directory is dropped and a resolver is provided, try to expand it
            if p.is_dir() and callable(self.dir_resolver):
                try:
                    expanded = self.dir_resolver(p)
                    for fp in expanded or []:
                        self._maybe_add_path(fp)
                except Exception:
                    pass
                return
            # Accept only files with allowed extensions
            if not p.is_file():
                return
            ext = p.suffix.lower()
            if self.accept_exts and ext not in self.accept_exts:
                return
            # Prevent duplicates
            for i in range(self.count()):
                if self.item(i).text() == str(p):
                    return
            item = QtWidgets.QListWidgetItem(str(p))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.addItem(item)

        def dragEnterEvent(self, event):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
            else:
                event.ignore()

        def dragMoveEvent(self, event):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
            else:
                event.ignore()

        def dropEvent(self, event):
            if event.mimeData().hasUrls():
                for url in event.mimeData().urls():
                    file_path = url.toLocalFile()
                    if file_path:
                        self._maybe_add_path(file_path)
                event.acceptProposedAction()

                # If the parent PDA widget has an "Auto load" checkbox
                # enabled, automatically trigger loading of the newly
                # dropped files, reusing the same routine as the "Load
                # dropped files" button.
                try:
                    owner = self.parent()
                    # Walk up a few levels in case the list is nested in
                    # intermediate layouts/containers.
                    steps = 0
                    while owner is not None and not hasattr(owner, "_on_load_dropped_files_clicked") and steps < 4:
                        owner = owner.parent()
                        steps += 1
                    if owner is not None and getattr(owner, "checkBox", None) is not None and owner.checkBox.isChecked():
                        owner._on_load_dropped_files_clicked()
                except Exception:
                    # Auto-load on drop is best-effort and must not break
                    # normal dragging behavior.
                    pass
            else:
                event.ignore()

        def remove_selected(self):
            for item in self.selectedItems():
                self.takeItem(self.row(item))

        def clear_all(self):
            self.clear()

        def contextMenuEvent(self, event):
            menu = QtWidgets.QMenu(self)
            act_delete = menu.addAction("Delete selected")
            act_clear = menu.addAction("Clear all")
            chosen = menu.exec_(event.globalPos())
            if chosen == act_delete:
                self.remove_selected()
            elif chosen == act_clear:
                self.clear_all()

    @chisurf.gui.decorators.init_with_ui("pda_tttr.ui")
    def __init__(self, *args, **kwargs):
        # super().__init__(parent=parent)
        # Internal cache for current setup data
        self._setup_name = None
        self._windows = {}
        self._detectors = {}
        self._tttr_reading = {}

        # Wire UI
        if hasattr(self, 'label_10'):
            self.label_10.setText("Setup")

        # Add file drop area into verticalLayout
        self._init_filedrop_area()

        # Action trigger on parameter changes
        self.actionParametersChanged.triggered.connect(self.onParametersChanged)

        # Populate reading routine combo with supported containers
        self.comboBox.clear()
        self.comboBox.insertItems(0, tttrlib.TTTR.get_supported_container_names())

        # Populate setups and connect signals
        self._load_available_setups_into_combobox()
        self.comboBox_setup.currentTextChanged.connect(self._on_setup_changed)
        self.comboBox_det1.currentTextChanged.connect(self._on_detector_combo_changed)
        self.comboBox_det2.currentTextChanged.connect(self._on_detector_combo_changed)

        # Initialize from current selections
        self._apply_current_setup_and_detectors()
        self.onParametersChanged()

    def _init_filedrop_area(self):
        # Accepted extensions: BID and TTTR families (lowercase)
        bid_exts = {'.bid', '.bur', '.bst'}
        tttr_exts = {'.ptu', '.ht3', '.spc'}
        self._accepted_exts = set(e.lower() for e in bid_exts | tttr_exts)
        self._tttr_exts = tttr_exts
        # Label
        self.drop_label = QtWidgets.QLabel("drop files or analysis folders here")
        self.drop_label.setAlignment(QtCore.Qt.AlignCenter)
        # A subtle frame to indicate dropping area (no heavy styling)
        self.drop_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.drop_label.setToolTip("Drag and drop TTTR files, BID/BUR/BST files, or burst analysis folders (bi4_bur). Checked files are used.")
        # List widget for files
        self.file_list = PdaTTTRWidget.DropFileList(parent=self, accept_exts=self._accepted_exts, dir_resolver=self._expand_burst_folder)
        # Notify parameter changes when items toggled
        self.file_list.itemChanged.connect(lambda _: self.actionParametersChanged.trigger())
        # Also notify when rows are inserted/removed (e.g., via context menu)
        try:
            self.file_list.model().rowsInserted.connect(lambda *args: self.actionParametersChanged.trigger())
            self.file_list.model().rowsRemoved.connect(lambda *args: self.actionParametersChanged.trigger())
        except Exception:
            pass
        # Insert into the bottom verticalLayout defined in the .ui
        if hasattr(self, 'verticalLayout') and isinstance(self.verticalLayout, QtWidgets.QVBoxLayout):
            self.verticalLayout.addWidget(self.drop_label)
            self.verticalLayout.addWidget(self.file_list)
            # Add small buttons row below the drop list
            btn_row = QtWidgets.QHBoxLayout()
            self.load_button = QtWidgets.QPushButton("Load dropped files")
            self.load_button.setToolTip("Read checked TTTR files using current PDA settings")
            self.load_button.clicked.connect(self._on_load_dropped_files_clicked)
            btn_row.addWidget(self.load_button)
            self.clear_button = QtWidgets.QPushButton("Clear")
            self.clear_button.setToolTip("Clear all dropped files from the list")
            self.clear_button.clicked.connect(lambda: (self.file_list.clear(), self.actionParametersChanged.trigger()))
            btn_row.addWidget(self.clear_button)
            btn_row.addStretch(1)
            self.verticalLayout.addLayout(btn_row)
        # Internal accessor for used files
        self._get_used_files = lambda: [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == QtCore.Qt.Checked
        ]

    def _compute_burst_slices_for_files(self, tttr_files):
        """
        For each TTTR file, try to locate a nearby bi4_bur/bur folder with .bur files.
        Build a mapping file -> list of (start, stop_exclusive) photon indices based on rows
        where First File == Last File == file name. Bursts spanning multiple files are ignored.
        """
        try:
            import pandas as pd
        except Exception:
            return {}
        slices = {}
        # Helper: locate a bur directory near a given TTTR file
        def find_bur_dir(tttr_path: pathlib.Path):
            # search up to 3 levels for a dir that contains 'bi4_bur' or 'bur'
            for up in [tttr_path.parent, tttr_path.parent.parent, tttr_path.parent.parent.parent]:
                if up is None:
                    continue
                if (up / 'bi4_bur').is_dir():
                    return up / 'bi4_bur'
                if (up / 'bur').is_dir():
                    return up / 'bur'
                # also handle the case where current folder is bi4_bur/bur
                if up.name.lower() in ('bi4_bur', 'bur'):
                    return up
            return None
        # Aggregate all BUR files per potential root
        bur_cache = {}
        for f in tttr_files:
            p = pathlib.Path(f)
            bur_dir = find_bur_dir(p)
            if bur_dir is None:
                continue
            key = str(bur_dir)
            if key not in bur_cache:
                bur_cache[key] = list(sorted(bur_dir.glob('*.bur')))
        if not bur_cache:
            return {}
        # Build filename-based slice lists
        tttr_names = {pathlib.Path(f).name: f for f in tttr_files}
        for bur_dir, bur_list in bur_cache.items():
            for bur_path in bur_list:
                try:
                    df = pd.read_csv(bur_path, sep='\t')
                except Exception:
                    continue
                # Normalize columns to lower
                cols_map = {c.lower(): c for c in df.columns}
                req = ['first photon','last photon','first file','last file']
                if not all(c in cols_map for c in req):
                    continue
                fp_col = cols_map['first photon']
                lp_col = cols_map['last photon']
                ff_col = cols_map['first file']
                lf_col = cols_map['last file']
                for _, row in df.iterrows():
                    try:
                        first_file = str(row[ff_col]).strip()
                        last_file = str(row[lf_col]).strip()
                        if not first_file or first_file != last_file:
                            continue
                        # match exact name (as listed in BUR) to a TTTR file name
                        if first_file not in tttr_names:
                            # try to match by stem + any extension (already chosen files have fixed ext)
                            # if BUR contains bare stem, extend check
                            for name, fullpath in list(tttr_names.items()):
                                if pathlib.Path(name).stem == pathlib.Path(first_file).stem:
                                    first_file = name
                                    break
                            if first_file not in tttr_names:
                                continue
                        # parse indices (floats in BUR -> ints)
                        a = int(float(row[fp_col]))
                        b_inc = int(float(row[lp_col]))
                        # convert to python slice convention: stop exclusive
                        b = b_inc + 1
                        slices.setdefault(tttr_names[first_file], []).append((a, b))
                    except Exception:
                        continue
        return self._merge_intervals(slices)

    def _resolve_tttr_and_slices_from_bur(self, bur_files, progress_callback=None):
        """
        From a list of BUR file paths, resolve the corresponding TTTR files and
        build a burst_slices mapping limited strictly to these selected BUR tables.

        Optimized: index TTTR files once per root and do O(1) lookups per BUR row.
        Returns: (tttr_files_list, burst_slices_dict)
        """
        try:
            import pandas as pd
        except Exception:
            return [], {}

        # Collect candidate roots to search (avoid duplicates)
        roots = []
        roots_seen = set()
        bur_to_roots = {}
        for bur in bur_files:
            try:
                bur_path = pathlib.Path(bur)
                if not bur_path.exists():
                    continue
                if bur_path.parent.name.lower() in ('bi4_bur', 'bur'):
                    root_hint = bur_path.parent.parent
                else:
                    root_hint = bur_path.parent
                roots_this_bur = []
                for r in [root_hint, root_hint.parent if root_hint else None, root_hint.parent.parent if root_hint and root_hint.parent else None]:
                    if r and r.exists():
                        sr = str(r)
                        roots_this_bur.append(r)
                        if sr not in roots_seen:
                            roots.append(r)
                            roots_seen.add(sr)
                bur_to_roots[bur] = roots_this_bur
            except Exception:
                continue

        # Index TTTR files under all roots once
        from collections import defaultdict
        name_to_paths = defaultdict(list)
        stem_to_paths = defaultdict(list)
        for root in roots:
            try:
                for ext in self._tttr_exts:
                    for p in root.rglob(f"*{ext}"):
                        if not p.is_file():
                            continue
                        name_to_paths[p.name].append(p)
                        stem_to_paths[p.stem].append(p)
            except Exception:
                continue

        tttr_files_order = []
        tttr_files_set = set()
        burst_slices = {}

        def add_tttr_file(p: pathlib.Path):
            sp = str(p)
            if sp not in tttr_files_set:
                tttr_files_set.add(sp)
                tttr_files_order.append(sp)

        total = len(bur_files) if isinstance(bur_files, (list, tuple)) else 0
        for idx, bur in enumerate(bur_files, start=1):
            try:
                if callable(progress_callback):
                    try:
                        progress_callback(idx, total, bur)
                    except Exception:
                        pass
                bur_path = pathlib.Path(bur)
                if not bur_path.is_file():
                    continue
                # Read BUR file (case-insensitive columns)
                try:
                    df = pd.read_csv(bur_path, sep='\t')
                except Exception:
                    continue
                cols_map = {c.lower(): c for c in df.columns}
                req = ['first photon','last photon','first file','last file']
                if not all(c in cols_map for c in req):
                    continue
                fp_col = cols_map['first photon']
                lp_col = cols_map['last photon']
                ff_col = cols_map['first file']
                lf_col = cols_map['last file']
                # Filter rows: First File == Last File and non-empty
                df = df.dropna(subset=[ff_col, lf_col])
                same_file = df[ff_col].astype(str).str.strip() == df[lf_col].astype(str).str.strip()
                df = df.loc[same_file]
                if df.empty:
                    continue
                local_roots = set(bur_to_roots.get(bur, []))

                def choose_path(name_str: str):
                    # Prefer candidates within local roots
                    cands = name_to_paths.get(name_str)
                    if cands:
                        for cand in cands:
                            if cand.parent in local_roots:
                                return cand
                        return cands[0]
                    stem = pathlib.Path(name_str).stem
                    cands = stem_to_paths.get(stem)
                    if cands:
                        for cand in cands:
                            if cand.parent in local_roots:
                                return cand
                        return cands[0]
                    return None

                for _, row in df.iterrows():
                    try:
                        first_file = str(row[ff_col]).strip()
                        if not first_file:
                            continue
                        resolved = choose_path(first_file)
                        if resolved is None:
                            continue
                        # slice indices (inclusive->exclusive)
                        a = int(float(row[fp_col]))
                        b = int(float(row[lp_col])) + 1
                        key = str(resolved)
                        burst_slices.setdefault(key, []).append((a, b))
                        add_tttr_file(resolved)
                    except Exception:
                        continue
            except Exception:
                continue
        return tttr_files_order, self._merge_intervals(burst_slices)

    def _expand_burst_folder(self, base_dir: pathlib.Path):
        """
        Given a dropped folder, detect burst analysis structure and return
        contained .bur files (so the user can select which burst tables to use).
        Previously this returned TTTR files; now we list the BUR files themselves.
        """
        try:
            # Identify the folder that holds .bur files
            if (base_dir / 'bi4_bur').is_dir():
                bur_dir = base_dir / 'bi4_bur'
            elif base_dir.name.lower() in ('bi4_bur', 'bur'):
                bur_dir = base_dir
            elif (base_dir / 'bur').is_dir():
                bur_dir = base_dir / 'bur'
            else:
                return []
            # Return absolute paths to .bur files found
            return [str(p) for p in sorted(bur_dir.glob('*.bur')) if p.is_file()]
        except Exception:
            return []

    def _merge_intervals(self, mapping):
        """Merge overlapping/adjacent [start, stop) intervals per key in a dict."""
        merged = {}
        for k, ivals in (mapping or {}).items():
            if not ivals:
                continue
            ivals_sorted = sorted(ivals, key=lambda x: (int(x[0]), int(x[1])))
            out = []
            cs, ce = ivals_sorted[0]
            for s, e in ivals_sorted[1:]:
                if s <= ce:  # overlap or directly adjacent (exclusive stop)
                    ce = max(ce, e)
                else:
                    out.append((cs, ce))
                    cs, ce = s, e
            out.append((cs, ce))
            merged[k] = out
        return merged

    # ---- Setup handling ----
    def _load_available_setups_into_combobox(self):
        setups = load_detector_setups()
        names = list(setups.get("setups", {}).keys())
        self.comboBox_setup.blockSignals(True)
        self.comboBox_setup.clear()
        if names:
            self.comboBox_setup.addItems(names)
            # Select last used if present
            last_used = setups.get("last_used")
            if last_used and last_used in names:
                self.comboBox_setup.setCurrentText(last_used)
            else:
                self.comboBox_setup.setCurrentIndex(0)
        else:
            # Placeholder when no setups are available
            self.comboBox_setup.addItem("No setups available")
        self.comboBox_setup.blockSignals(False)

    def _on_setup_changed(self, name: str):
        # Load selected setup and apply
        setups = load_detector_setups()
        data = setups.get("setups", {}).get(name)
        if not data:
            # Clear caches if no valid setup
            self._setup_name = None
            self._windows, self._detectors, self._tttr_reading = {}, {}, {}
            return
        self._setup_name = name
        self._windows = data.get("windows", {}) or {}
        self._detectors = data.get("detectors", {}) or {}
        self._tttr_reading = data.get("tttr_reading", {}) or {}
        # Update UI elements based on setup
        self._populate_detector_combos()
        self._apply_tttr_reading_to_combo()
        # Trigger global parameter update
        self.actionParametersChanged.trigger()

    def _apply_current_setup_and_detectors(self):
        # Attempt to apply currently selected setup if any
        current = self.comboBox_setup.currentText()
        if current:
            self._on_setup_changed(current)
        else:
            self._populate_detector_combos()

    # ---- Detector selection -> channels and microtime ----
    def _populate_detector_combos(self):
        # Fill both detector selection combos with available detector names
        names = list(self._detectors.keys())
        self.comboBox_det1.blockSignals(True)
        self.comboBox_det2.blockSignals(True)
        self.comboBox_det1.clear()
        self.comboBox_det2.clear()
        if names:
            self.comboBox_det1.addItems(names)
            self.comboBox_det2.addItems(names)
            # Prefer conventional defaults if available
            if "green" in names:
                self.comboBox_det1.setCurrentText("green")
            else:
                self.comboBox_det1.setCurrentIndex(0)
            # Choose a different detector for det2 if possible, prefer red
            if "red" in names:
                self.comboBox_det2.setCurrentText("red")
            else:
                self.comboBox_det2.setCurrentIndex(min(1, len(names)-1))
        else:
            self.comboBox_det1.addItem("")
            self.comboBox_det2.addItem("")
        self.comboBox_det1.blockSignals(False)
        self.comboBox_det2.blockSignals(False)
        # After populating, sync all dependent fields
        self._update_from_detector_selection()

    def _on_detector_combo_changed(self, _):
        self._update_from_detector_selection()
        # Trigger parameter update to propagate change
        self.actionParametersChanged.trigger()

    def _format_window_value(self, win_val) -> str:
        # Accept tuple/list/int, mirror logic from PhotonFilter.update_pie_windows
        if isinstance(win_val, (list, tuple)):
            # Single [start, end]
            if len(win_val) == 2 and all(isinstance(x, int) for x in win_val):
                return f"{win_val[0]}-{win_val[1]}"
            parts = []
            for i in win_val:
                if isinstance(i, (list, tuple)) and len(i) >= 2:
                    parts.append(f"{i[0]}-{i[1]}")
                elif isinstance(i, int):
                    parts.append(f"{i}-{i}")
            return ";".join(parts)
        elif isinstance(win_val, int):
            return f"{win_val}-{win_val}"
        return ""

    def _update_from_detector_selection(self):
        # Map selected detector names to routing channels and microtime ranges
        name1 = self.comboBox_det1.currentText()
        name2 = self.comboBox_det2.currentText()
        det1 = self._detectors.get(name1, {}) or {}
        det2 = self._detectors.get(name2, {}) or {}
        # Update channels
        chs1 = ", ".join(str(i) for i in det1.get("chs", []))
        chs2 = ", ".join(str(i) for i in det2.get("chs", []))
        self.lineEdit.setText(chs1)
        self.lineEdit_4.setText(chs2)
        # Update micro-time ranges
        self.lineEdit_2.setText(self._format_window_value(det1.get("micro_time_ranges", [])))
        self.lineEdit_3.setText(self._format_window_value(det2.get("micro_time_ranges", [])))

    # ---- Detectors -> routing channels ----
    def _populate_channels_from_detectors(self):
        # Choose two detectors (prefer green/red), and apply their channel lists
        if not self._detectors:
            return
        names = list(self._detectors.keys())
        def pick(name_pref: str, fallback_idx: int):
            if name_pref in self._detectors:
                return name_pref
            return names[fallback_idx] if fallback_idx < len(names) else names[0]
        d1_name = pick("green", 0)
        # Avoid same detector twice
        if "red" in self._detectors:
            d2_name = "red"
        else:
            d2_name = names[1] if len(names) > 1 else names[0]
        chs1 = ", ".join(str(i) for i in self._detectors[d1_name].get("chs", []))
        chs2 = ", ".join(str(i) for i in self._detectors[d2_name].get("chs", []))
        self.lineEdit.setText(chs1)
        self.lineEdit_4.setText(chs2)

    # ---- TTTR reading routine ----
    def _apply_tttr_reading_to_combo(self):
        if not self._tttr_reading:
            return
        file_type = self._tttr_reading.get("file_type")
        if not file_type:
            return
        # Ensure reading routine combo contains supported names and select the one from setup
        supported = tttrlib.TTTR.get_supported_container_names()
        # Refresh items to the supported list only once (they are set in __init__), just set selection if present
        if file_type in supported:
            self.comboBox.setCurrentText(file_type)

    def onParametersChanged(self):
        # Parse channels
        ch0_text = self.lineEdit.text().strip()
        ch1_text = self.lineEdit_4.text().strip()
        ch0 = [int(k) for k in ch0_text.split(',')] if ch0_text else []
        ch1 = [int(k) for k in ch1_text.split(',')] if ch1_text else []
        # Parse microtime ranges
        def parse_mtr(s: str):
            s = s.strip()
            if not s:
                return []
            parts = []
            for seg in s.split(';'):
                seg = seg.strip()
                if not seg:
                    continue
                ab = [int(j) for j in seg.split('-')]
                if len(ab) >= 2:
                    parts.append([ab[0], ab[1]])
            return parts
        mt0 = parse_mtr(self.lineEdit_2.text())
        mt1 = parse_mtr(self.lineEdit_3.text())
        micro_time_ranges = [mt0, mt1]
        channels = [ch0, ch1]
        minimum_number_of_photons = self.spinBox.value()
        maximum_number_of_photons = self.spinBox_2.value()
        minimum_time_window_length = self.doubleSpinBox.value()

        reading_routine = self.comboBox.currentText()
        chisurf.run(
            "\n".join(
                [
                    f"cs.current_setup.reading_routine = '{reading_routine}'",
                    f"cs.current_setup.channels = {channels}",
                    f"cs.current_setup.micro_time_ranges = {micro_time_ranges}",
                    f"cs.current_setup.minimum_number_of_photons = {minimum_number_of_photons}",
                    f"cs.current_setup.maximum_number_of_photons = {maximum_number_of_photons}",
                    f"cs.current_setup.minimum_time_window_length = {minimum_time_window_length / 1000.0}"
                ]
            )
        )

    def get_filename(self) -> pathlib.Path:
        return chisurf.gui.widgets.open_files(
            description='HT3/PTU/SPC file',
            file_type='All files (*.*)',
            working_path=None
        )

    def _on_load_dropped_files_clicked(self):
        try:
            files = self._get_used_files() if hasattr(self, '_get_used_files') else []
            if not files:
                logging.warning("PDA: No dropped files are checked to load.")
                QtWidgets.QMessageBox.information(self, "No files", "No dropped files are checked to load.")
                return
            # Split files by type: bur and tttr
            bur_files = [f for f in files if pathlib.Path(f).suffix.lower() == '.bur']
            tttr_files = [f for f in files if pathlib.Path(f).suffix.lower() in self._tttr_exts]
            if not bur_files and not tttr_files:
                logging.warning("PDA: Dropped items contain neither BUR nor TTTR files to load.")
                QtWidgets.QMessageBox.warning(
                    self, "No files", "Please drop .bur burst files or TTTR files "
                                      "(e.g., .ptu, .ht3, .spc, .sdt, .t3r, .t2r, .phu, .phd) to load.")
                return

            # Gather current PDA parameters from UI
            # channels
            ch0_text = self.lineEdit.text().strip()
            ch1_text = self.lineEdit_4.text().strip()
            ch0 = [int(k) for k in ch0_text.split(',')] if ch0_text else []
            ch1 = [int(k) for k in ch1_text.split(',')] if ch1_text else []
            # microtime ranges
            def parse_mtr(s: str):
                s = s.strip()
                if not s:
                    return []
                parts = []
                for seg in s.split(';'):
                    seg = seg.strip()
                    if not seg:
                        continue
                    ab = [int(j) for j in seg.split('-')]
                    if len(ab) >= 2:
                        parts.append((ab[0], ab[1]))
                return parts
            mt0 = parse_mtr(self.lineEdit_2.text())
            mt1 = parse_mtr(self.lineEdit_3.text())
            micro_time_ranges = [mt0, mt1]
            maximum_number_of_photons = int(self.spinBox_2.value())
            minimum_number_of_photons = int(self.spinBox.value())
            minimum_time_window_length = float(self.doubleSpinBox.value()) / 1000.0  # UI is ms, reader expects seconds
            reading_routine = self.comboBox.currentText()

            # Create a PdaReader instance with current settings
            logging.info(f"PDA: Preparing to load {len(tttr_files)} TTTR file(s) with routine '{reading_routine}'.")
            logging.debug({
                'channels': (ch0, ch1),
                'micro_time_ranges': micro_time_ranges,
                'max_photons': maximum_number_of_photons,
                'min_photons': minimum_number_of_photons,
                'min_time_window_s': minimum_time_window_length
            })
            pda_reader = PdaReader(
                channels=(ch0, ch1),
                micro_time_ranges=micro_time_ranges,
                reading_routine=reading_routine,
                maximum_number_of_photons=maximum_number_of_photons,
                minimum_number_of_photons=minimum_number_of_photons,
                minimum_time_window_length=minimum_time_window_length
            )
            # Attach the correct experiment to the reader so get_data can set d.experiment
            try:
                pda_reader.experiment = chisurf.experiments.types.get('pda') or chisurf.cs.current_experiment
            except Exception:
                # Fallback to current experiment if types lookup fails
                pda_reader.experiment = getattr(chisurf.cs, 'current_experiment', None)
            # Optionally link controller
            try:
                pda_reader.controller = self
            except Exception:
                pass

            # Resolve TTTR files and burst slices
            if bur_files:
                logging.info(f"PDA: Resolving TTTR files and slices from {len(bur_files)} selected BUR file(s).")
                # Progress dialog for resolving BUR files
                progress = QtWidgets.QProgressDialog("Resolving BUR files...", "Cancel", 0, len(bur_files), self)
                try:
                    progress.setWindowModality(QtCore.Qt.WindowModal)
                except Exception:
                    pass
                progress.setWindowTitle("PDA Loading")
                progress.setAutoClose(True)
                progress.setAutoReset(True)

                def _progress_cb(i, total, current):
                    try:
                        progress.setLabelText(f"Resolving: {pathlib.Path(current).name} ({i}/{total})")
                        progress.setValue(i)
                        QtWidgets.QApplication.processEvents()
                    except Exception:
                        pass
                    if progress.wasCanceled():
                        raise RuntimeError("Operation canceled by user")

                try:
                    tttr_files_resolved, burst_slices = self._resolve_tttr_and_slices_from_bur(bur_files, progress_callback=_progress_cb)
                finally:
                    try:
                        progress.close()
                    except Exception:
                        pass
                tttr_files = tttr_files_resolved
                if not tttr_files:
                    logging.warning("PDA: No TTTR files could be resolved from selected BUR files.")
                    QtWidgets.QMessageBox.warning(self, "No TTTR files found", "Could not resolve any TTTR files from the selected BUR files.")
                    return
            else:
                # Try to derive burst slices from nearby BUR files if present
                # show indeterminate progress during slice computation
                progress2 = QtWidgets.QProgressDialog("Searching for nearby BUR files...", "Cancel", 0, 0, self)
                progress2.setWindowTitle("PDA Loading")
                progress2.setAutoClose(True)
                progress2.setAutoReset(True)
                progress2.show()
                QtWidgets.QApplication.processEvents()
                try:
                    burst_slices = self._compute_burst_slices_for_files(tttr_files)
                finally:
                    try:
                        progress2.close()
                    except Exception:
                        pass

            if burst_slices:
                try:
                    total_slices = sum(len(v) for v in burst_slices.values())
                except Exception:
                    total_slices = 0
                logging.info(f"PDA: Applying burst slicing from BUR files: {total_slices} slices across {len(burst_slices)} file(s).")
                logging.debug({'burst_slices_keys': list(burst_slices.keys())})

            # Use macro to add dataset into ChiSurf
            filenames_arg = "|".join(tttr_files)
            logging.debug({'tttr_files': tttr_files})
            # Show an indeterminate progress during actual data loading
            progress3 = QtWidgets.QProgressDialog("Loading TTTR data and computing histograms...", "Cancel", 0, 0, self)
            progress3.setWindowTitle("PDA Loading")
            progress3.setAutoClose(True)
            progress3.setAutoReset(True)
            progress3.show()
            QtWidgets.QApplication.processEvents()
            try:
                if burst_slices:
                    core_data_macros.add_dataset(experiment_reader=pda_reader, filename=filenames_arg, burst_slices=burst_slices)
                else:
                    core_data_macros.add_dataset(experiment_reader=pda_reader, filename=filenames_arg)
            finally:
                try:
                    progress3.close()
                except Exception:
                    pass

            logging.info(f"PDA: Loaded {len(tttr_files)} TTTR file(s).")
            try:
                if getattr(self, "checkBox", None) is not None and self.checkBox.isChecked():
                    if hasattr(self, "file_list"):
                        self.file_list.clear()
                    self.actionParametersChanged.trigger()
            except Exception:
                logging.warning("PDA: Auto-clear after load failed.")
        except Exception as e:
            # Show error message and log warning
            logging.warning(f"PDA: Failed to load files: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load files: {e}")

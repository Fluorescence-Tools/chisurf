import pathlib
import tttrlib

import chisurf.gui
from chisurf.experiments import reader

from chisurf.gui import QtWidgets, QtGui, QtCore
import chisurf
from chisurf import logging
from chisurf.macros import core_data as core_data_macros
from chisurf.experiments.pda import PdaReader
from chisurf.gui.widgets.progress import EnhancedProgressDialog

# Reuse the setups loader from the DetectorWizard
from chisurf.gui.widgets.wizard.tttr_channel_definition import load_detector_setups

_TTTR_INDEX_CACHE = {}


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
            """Safely add a dropped path or expand a directory into files.

            All errors are logged and swallowed so that a bad path or
            unexpected directory layout cannot crash the GUI.
            """
            try:
                p = pathlib.Path(file_path)
            except Exception:
                logging.warning("PDA: Invalid path dropped: %r", file_path)
                return

            # If a directory is dropped, add the folder itself as one entry.
            # BUR file discovery is deferred to load time to keep UI load low.
            if p.is_dir():
                sp = str(p)
                for i in range(self.count()):
                    if self.item(i).text() == sp:
                        return
                try:
                    item = QtWidgets.QListWidgetItem(sp)
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    item.setCheckState(QtCore.Qt.Checked)
                    self.addItem(item)
                except Exception:
                    logging.warning(
                        "PDA: Failed to add dropped folder to list: %s", sp, exc_info=True
                    )
                return

            # Accept only files with allowed extensions
            if not p.is_file():
                return
            ext = p.suffix.lower()
            if self.accept_exts and ext not in self.accept_exts:
                return
            # Prevent duplicates
            sp = str(p)
            for i in range(self.count()):
                if self.item(i).text() == sp:
                    return
            try:
                item = QtWidgets.QListWidgetItem(sp)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)
                self.addItem(item)
            except Exception:
                logging.warning(
                    "PDA: Failed to add dropped file to list: %s", sp, exc_info=True
                )

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
            try:
                if event.mimeData().hasUrls():
                    for url in event.mimeData().urls():
                        try:
                            file_path = url.toLocalFile()
                        except Exception:
                            continue
                        if file_path:
                            try:
                                self._maybe_add_path(file_path)
                            except Exception:
                                logging.warning(
                                    "PDA: Error handling dropped path: %r", file_path, exc_info=True
                                )
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
                            # Schedule the heavy loading routine on the event
                            # loop to avoid running it re-entrantly inside the
                            # dropEvent handler, which can destabilize
                            # QListWidget / drag-and-drop internals when many
                            # items are dropped at once.
                            QtCore.QTimer.singleShot(0, owner._on_load_dropped_files_clicked)
                    except Exception:
                        # Auto-load on drop is best-effort and must not break
                        # normal dragging behavior.
                        logging.warning(
                            "PDA: Auto-load on drop failed; ignoring.", exc_info=True
                        )
                else:
                    event.ignore()
            except Exception:
                logging.warning(
                    "PDA: Unexpected error in DropFileList.dropEvent; ignoring drop.",
                    exc_info=True
                )
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

    def _get_tttr_supported_exts(self):
        exts = set()
        try:
            if hasattr(tttrlib, "get_supported_filetypes"):
                for e in tttrlib.get_supported_filetypes():
                    s = str(e).strip().lower()
                    if not s:
                        continue
                    if not s.startswith('.'):
                        s = '.' + s
                    exts.add(s)
        except Exception:
            exts = set()
        if not exts:
            exts = {'.ptu', '.ht3', '.spc', '.h5', '.hdf5'}
        return exts

    def _init_filedrop_area(self):
        # Accepted extensions: BID and TTTR families (lowercase)
        bid_exts = {'.bid', '.bur', '.bst'}
        tttr_exts = set(e.lower() for e in self._get_tttr_supported_exts())
        self._accepted_exts = set(e.lower() for e in bid_exts | tttr_exts)
        self._tttr_exts = tttr_exts
        # Label describing the drop area (the list widget below)
        self.drop_label = QtWidgets.QLabel("drop files or analysis folders below")
        self.drop_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
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
                    required = {'first photon', 'last photon', 'first file', 'last file'}
                    df = pd.read_csv(
                        bur_path,
                        sep='\t',
                        usecols=lambda c: c.lower() in required
                    )
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
                                logging.debug(
                                    "PDA: TTTR file referenced in BUR not found among selected TTTR files: %s (BUR: %s)",
                                    first_file, str(bur_path)
                                )
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

        # For each BUR file, determine the most likely TTTR folder(s) once.
        # We follow the convention that BUR tables live in a "bi4_bur"/"bur" folder
        # somewhere below the TTTR data. In practice, TTTR files are often stored in
        # the parent or grand-parent folder (optionally in an 'hdf5' subfolder).
        #
        # We therefore, for each "analysis root" (one level above bi4_bur/bur),
        # build a short ordered list of candidate TTTR roots by walking up at most
        # three levels and, at each level, preferring a sibling 'hdf5' directory if
        # present, then the directory itself. These roots are cached per analysis
        # root so we do not recompute them for every BUR file.
        bur_to_roots = {}
        root_hint_cache = {}
        for bur in bur_files:
            try:
                bur_path = pathlib.Path(bur)
                if not bur_path.exists():
                    continue
                if bur_path.parent.name.lower() in ('bi4_bur', 'bur'):
                    root_hint = bur_path.parent.parent
                else:
                    root_hint = bur_path.parent

                if root_hint is None:
                    bur_to_roots[bur] = []
                    continue

                cache_key = str(root_hint)
                cached_roots = root_hint_cache.get(cache_key)
                if cached_roots is not None:
                    bur_to_roots[bur] = cached_roots
                    continue

                roots_this_bur = []
                seen = set()

                # Walk up to three levels: analysis root, its parent, and grand-parent.
                candidates = []
                try:
                    candidates.append(root_hint)
                    parent1 = root_hint.parent
                    if parent1 is not None:
                        candidates.append(parent1)
                        parent2 = parent1.parent
                        if parent2 is not None:
                            candidates.append(parent2)
                except Exception:
                    pass

                for base in candidates:
                    if base is None:
                        continue
                    try:
                        if not base.exists():
                            continue
                    except Exception:
                        continue

                    # Prefer sibling 'hdf5' directory if present (typical MFD HDF5 layout)
                    try:
                        hdf5_dir = base / 'hdf5'
                        if hdf5_dir.is_dir():
                            key_hdf5 = str(hdf5_dir)
                            if key_hdf5 not in seen:
                                roots_this_bur.append(hdf5_dir)
                                seen.add(key_hdf5)
                    except Exception:
                        pass

                    key_base = str(base)
                    if key_base not in seen:
                        roots_this_bur.append(base)
                        seen.add(key_base)

                root_hint_cache[cache_key] = roots_this_bur
                bur_to_roots[bur] = roots_this_bur
            except Exception:
                continue

        tttr_files_order = []
        tttr_files_set = set()
        burst_slices = {}

        # Cache of the preferred TTTR directory per analysis root. Once one
        # BUR file from a given analysis root has been successfully mapped to
        # a TTTR file, all subsequent BUR files from the same analysis root
        # reuse that TTTR directory instead of walking multiple candidate
        # roots again.
        analysis_preferred_root = {}

        def add_tttr_file(p: pathlib.Path):
            sp = str(p)
            if sp not in tttr_files_set:
                tttr_files_set.add(sp)
                tttr_files_order.append(sp)

        def _resolve_name_in_roots(name_str: str, roots_for_bur):
            """Resolve a TTTR filename relative to a small set of roots.

            The function tries, in order:
            - absolute path in the filesystem
            - direct join (root / name_str) for each candidate root
            - same stem with any known TTTR extension in each root
            """
            if not name_str:
                return None

            # Absolute path reference
            try:
                candidate = pathlib.Path(name_str)
                if candidate.is_absolute() and candidate.is_file():
                    return candidate
            except Exception:
                candidate = None

            # Relative name: search only the nearby roots, no recursion
            stem = pathlib.Path(name_str).stem
            for root in roots_for_bur or []:
                try:
                    # Direct join, preserving any subdirectories encoded in name_str
                    cand = (root / name_str)
                    if cand.is_file():
                        return cand
                except Exception:
                    cand = None

                # Fallback: same stem with any supported TTTR extension in this folder
                try:
                    for ext in self._tttr_exts:
                        alt = (root / f"{stem}{ext}")
                        if alt.is_file():
                            return alt
                except Exception:
                    continue
            return None

        total = len(bur_files) if isinstance(bur_files, (list, tuple)) else 0
        for idx, bur in enumerate(bur_files, start=1):
            try:
                if callable(progress_callback):
                    try:
                        # Allow the callback to request early abort by returning False
                        if progress_callback(idx, total, bur) is False:
                            break
                    except Exception:
                        # Ignore callback errors; do not abort resolving
                        pass

                bur_path = pathlib.Path(bur)
                if not bur_path.is_file():
                    continue

                # Read BUR file (case-insensitive columns)
                try:
                    required = {'first photon', 'last photon', 'first file', 'last file'}
                    df = pd.read_csv(
                        bur_path,
                        sep='\t',
                        usecols=lambda c: c.lower() in required
                    )
                except Exception:
                    continue

                cols_map = {c.lower(): c for c in df.columns}
                req = ['first photon', 'last photon', 'first file', 'last file']
                if not all(c in cols_map for c in req):
                    continue

                fp_col = cols_map['first photon']
                lp_col = cols_map['last photon']
                ff_col = cols_map['first file']
                lf_col = cols_map['last file']

                # Filter rows: First File == Last File and non-empty
                df = df.dropna(subset=[ff_col, lf_col])
                if df.empty:
                    continue

                file_series = df[ff_col].astype(str).str.strip()
                same_file = file_series == df[lf_col].astype(str).str.strip()
                df = df.loc[same_file]
                file_series = file_series.loc[df.index]
                if df.empty:
                    continue

                # Pre-compute integer photon index ranges once per BUR
                try:
                    start_series = df[fp_col].astype(float).astype('int64')
                    stop_series = df[lp_col].astype(float).astype('int64') + 1
                except Exception:
                    continue
                df = df.copy()
                df['_pda_start'] = start_series
                df['_pda_stop'] = stop_series

                # Determine analysis root for this BUR (one level above bi4_bur/bur
                # if present, otherwise the BUR's parent directory).
                analysis_root = None
                try:
                    if bur_path.parent.name.lower() in ('bi4_bur', 'bur'):
                        analysis_root = bur_path.parent.parent
                    else:
                        analysis_root = bur_path.parent
                except Exception:
                    analysis_root = None
                analysis_key = str(analysis_root) if analysis_root is not None else None

                roots_for_bur = bur_to_roots.get(bur, [])
                if analysis_key is not None and analysis_key in analysis_preferred_root:
                    roots_for_bur = [analysis_preferred_root[analysis_key]]
                if not roots_for_bur:
                    logging.warning(
                        "PDA: No TTTR search roots found for BUR file %s; skipping its bursts.",
                        str(bur_path),
                    )
                    continue

                # First try the simple 1:1 mapping: `<bur_stem>.bur` -> `<bur_stem>.<ext>`
                # in the TTTR folder(s). This matches the typical MFD layout where each
                # BUR file belongs to exactly one TTTR file with the same stem.
                bur_tttr = None
                for root in roots_for_bur:
                    try:
                        for ext in self._tttr_exts:
                            cand = root / f"{bur_path.stem}{ext}"
                            if cand.is_file():
                                bur_tttr = cand
                                break
                    except Exception:
                        continue
                    if bur_tttr is not None:
                        break

                if bur_tttr is not None:
                    if analysis_key is not None and analysis_key not in analysis_preferred_root:
                        try:
                            analysis_preferred_root[analysis_key] = bur_tttr.parent
                        except Exception:
                            pass
                    key = str(bur_tttr)
                    try:
                        starts = df['_pda_start'].tolist()
                        stops = df['_pda_stop'].tolist()
                    except Exception:
                        try:
                            starts = start_series.tolist()
                            stops = stop_series.tolist()
                        except Exception:
                            starts = stops = []
                    if starts:
                        burst_slices.setdefault(key, []).extend(zip(starts, stops))
                        try:
                            add_tttr_file(bur_tttr)
                        except Exception:
                            pass
                    # Done with this BUR file; proceed to the next one.
                    continue

                # Fallback: resolve TTTR paths from the "First File" column contents.
                # This covers less common layouts where a BUR file may reference
                # multiple TTTR files.
                resolved_cache = {}
                missing_logged = set()
                for name_str in file_series.unique():
                    if not name_str:
                        resolved_cache[name_str] = None
                        continue
                    # Some external tools encode file indices like "0", "1", ... in
                    # the First File column. Treat these as non-resolvable identifiers
                    # instead of literal filenames.
                    if str(name_str).strip().isdigit():
                        resolved_cache[name_str] = None
                        continue
                    try:
                        resolved_path = _resolve_name_in_roots(name_str, roots_for_bur)
                    except Exception:
                        resolved_path = None
                    resolved_cache[name_str] = str(resolved_path) if resolved_path is not None else None
                    # As soon as we find a real TTTR file for this analysis root,
                    # remember its parent directory as the preferred TTTR folder so
                    # subsequent BUR files from the same root do not need to re-walk
                    # all candidate ancestors.
                    if (
                        resolved_path is not None
                        and analysis_key is not None
                        and analysis_key not in analysis_preferred_root
                    ):
                        try:
                            analysis_preferred_root[analysis_key] = resolved_path.parent
                        except Exception:
                            pass

                # Map each row to its resolved TTTR file key (string path or None)
                resolved_keys = file_series.map(lambda s: resolved_cache.get(s))
                mask_valid = resolved_keys.notna()
                if not mask_valid.any():
                    # All referenced TTTRs missing for this BUR (ignoring pure indices)
                    for missing_name, key in resolved_cache.items():
                        if key is None and missing_name and not str(missing_name).strip().isdigit():
                            logging.warning(
                                "PDA: TTTR file '%s' referenced in BUR '%s' could not be found; skipping its bursts.",
                                missing_name,
                                str(bur_path),
                            )
                    continue

                valid_df = df.loc[mask_valid]
                valid_keys = resolved_keys.loc[mask_valid]

                # Group by resolved TTTR file and aggregate slices vectorized per group
                for key, group in valid_df.groupby(valid_keys):
                    if not key:
                        continue
                    try:
                        starts = group['_pda_start'].tolist()
                        stops = group['_pda_stop'].tolist()
                    except Exception:
                        continue
                    if not starts:
                        continue
                    burst_slices.setdefault(key, []).extend(zip(starts, stops))
                    try:
                        add_tttr_file(pathlib.Path(key))
                    except Exception:
                        # If the path string is malformed, keep slices but skip list entry
                        pass
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
            # Do not call resolve() here; on Windows this may convert mapped-drive
            # paths (e.g. "P:\\...") into UNC network paths ("\\\\server\\share\\..."),
            # which is undesirable for logging and user-visible paths.
            if not base_dir.exists() or not base_dir.is_dir():
                return []
            # Prefer .bur files directly in the folder; if none, search recursively
            bur_files = [p for p in base_dir.glob('*.bur') if p.is_file()]
            if not bur_files:
                bur_files = [p for p in base_dir.rglob('*.bur') if p.is_file()]
            return [str(p) for p in sorted(bur_files)]
        except Exception:
            logging.warning(
                "PDA: Error expanding burst folder: %s", str(base_dir), exc_info=True
            )
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

            progress_dialog = None
            try:
                parent = getattr(chisurf, 'cs', None)
            except Exception:
                parent = None
            if not isinstance(parent, QtWidgets.QWidget):
                try:
                    parent = self.window()
                except Exception:
                    parent = self
            try:
                progress_dialog = EnhancedProgressDialog(
                    title="Loading PDA data",
                    label_text="Preparing dropped files...",
                    min_value=0,
                    max_value=0,
                    parent=parent,
                )
                progress_dialog.setWindowModality(QtCore.Qt.ApplicationModal)
                progress_dialog.setMinimumDuration(0)
                progress_dialog.setAutoClose(False)
                progress_dialog.setAutoReset(False)
                progress_dialog.show()
                progress_dialog.update_progress(0, "Preparing dropped files...")
            except Exception:
                progress_dialog = None

            try:
                try:
                    cs = getattr(chisurf, 'cs', None)
                except Exception:
                    cs = None
                progress_bar = getattr(cs, 'progress_bar', None)
                status_label = getattr(cs, 'status_label', None)
                progress_backup = None
                status_backup = None
                if progress_bar is not None:
                    try:
                        progress_backup = (
                            progress_bar.minimum(),
                            progress_bar.maximum(),
                            progress_bar.value()
                        )
                    except Exception:
                        progress_backup = None
                if status_label is not None:
                    try:
                        status_backup = status_label.text()
                    except Exception:
                        status_backup = None

                bur_files = []
                tttr_files = []
                for f in files:
                    try:
                        p = pathlib.Path(f)
                    except Exception:
                        logging.warning("PDA: Invalid dropped path in load: %r", f)
                        continue
                    try:
                        if p.is_dir():
                            try:
                                expanded = self._expand_burst_folder(p)
                            except Exception:
                                logging.warning(
                                    "PDA: Error expanding burst folder during load: %s", str(p), exc_info=True
                                )
                                expanded = []
                            bur_files.extend(expanded or [])
                        else:
                            suffix = p.suffix.lower()
                            if suffix == '.bur':
                                bur_files.append(str(p))
                            elif suffix in self._tttr_exts:
                                tttr_files.append(str(p))
                    except Exception:
                        logging.warning(
                            "PDA: Error classifying dropped path during load: %r", f, exc_info=True
                        )
                        continue

                bur_files = sorted(set(bur_files))
                tttr_files = sorted(set(tttr_files))

                max_bur_files = 1024
                max_tttr_files = 1024
                if len(bur_files) > max_bur_files:
                    logging.warning(
                        "PDA: Too many BUR files selected (%d); aborting load.",
                        len(bur_files)
                    )
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Too many BUR files",
                        f"You selected {len(bur_files)} BUR files. "
                        f"For stability, please process them in smaller batches (<= {max_bur_files} at once)."
                    )
                    return
                if len(tttr_files) > max_tttr_files:
                    logging.warning(
                        "PDA: Too many TTTR files selected (%d); limiting to first %d.",
                        len(tttr_files), max_tttr_files
                    )
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Too many files",
                        f"You selected {len(tttr_files)} TTTR files. "
                        f"For stability, only the first {max_tttr_files} will be loaded.\n\n"
                        "Consider using BUR tables or smaller batches if you need to process more files."
                    )
                    tttr_files = tttr_files[:max_tttr_files]
                if not bur_files and not tttr_files:
                    logging.warning("PDA: Dropped items contain neither BUR nor TTTR files to load.")
                    QtWidgets.QMessageBox.warning(
                        self, "No files", "Please drop .bur burst files or TTTR files "
                                          "(e.g., .ptu, .ht3, .spc, .sdt, .t3r, .t2r, .phu, .phd) to load.")
                    return

                ch0_text = self.lineEdit.text().strip()
                ch1_text = self.lineEdit_4.text().strip()
                ch0 = [int(k) for k in ch0_text.split(',')] if ch0_text else []
                ch1 = [int(k) for k in ch1_text.split(',')] if ch1_text else []

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
                try:
                    pda_reader.controller = self
                except Exception:
                    pass

                if bur_files:
                    logging.info(f"PDA: Resolving TTTR files and slices from {len(bur_files)} selected BUR file(s).")
                    if progress_bar is not None:
                        try:
                            progress_bar.setMinimum(0)
                            progress_bar.setMaximum(max(1, len(bur_files)))
                            progress_bar.setValue(0)
                        except Exception:
                            pass
                    if status_label is not None:
                        try:
                            status_label.setText("Resolving BUR files...")
                        except Exception:
                            pass
                    if progress_dialog is not None:
                        try:
                            progress_dialog.setRange(0, max(1, len(bur_files)))
                            progress_dialog.update_progress(0, "Resolving BUR files...")
                        except Exception:
                            pass

                    def _progress_cb(i, total, current):
                        """Progress callback for BUR resolving.

                        Returns False when the user presses Cancel on the
                        EnhancedProgressDialog so that the resolver can
                        abort early.
                        """
                        try:
                            if status_label is not None:
                                status_label.setText(f"Resolving: {pathlib.Path(current).name} ({i}/{total})")
                            if progress_bar is not None:
                                try:
                                    if total:
                                        progress_bar.setMaximum(max(1, total))
                                except Exception:
                                    pass
                                try:
                                    progress_bar.setValue(i)
                                except Exception:
                                    pass
                            if progress_dialog is not None:
                                try:
                                    progress_dialog.update_progress(
                                        i,
                                        f"Resolving: {pathlib.Path(current).name} ({i}/{total})"
                                    )
                                except Exception:
                                    pass
                                # Honor Cancel so long-running BUR resolving can be aborted.
                                try:
                                    if progress_dialog.wasCanceled():
                                        return False
                                except Exception:
                                    pass
                            QtWidgets.QApplication.processEvents()
                        except Exception:
                            pass
                        return True

                    tttr_files_resolved, burst_slices = self._resolve_tttr_and_slices_from_bur(
                        bur_files,
                        progress_callback=_progress_cb
                    )
                    tttr_files = tttr_files_resolved
                    if len(tttr_files) > max_tttr_files:
                        logging.warning(
                            "PDA: Too many TTTR files resolved from BUR tables (%d); limiting to first %d.",
                            len(tttr_files), max_tttr_files
                        )
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Too many files",
                            f"Burst tables reference {len(tttr_files)} TTTR files. "
                            f"For stability, only the first {max_tttr_files} will be loaded."
                        )
                        tttr_files = tttr_files[:max_tttr_files]
                        if burst_slices:
                            keep = set(tttr_files)
                            burst_slices = {k: v for k, v in burst_slices.items() if k in keep}
                    if not tttr_files:
                        logging.warning("PDA: No TTTR files could be resolved from selected BUR files.")
                        QtWidgets.QMessageBox.warning(self, "No TTTR files found", "Could not resolve any TTTR files from the selected BUR files.")
                        return
                else:
                    if progress_bar is not None:
                        try:
                            progress_bar.setMinimum(0)
                            progress_bar.setMaximum(0)
                            progress_bar.setValue(0)
                        except Exception:
                            pass
                    if status_label is not None:
                        try:
                            status_label.setText("Searching for nearby BUR files...")
                        except Exception:
                            pass
                    if progress_dialog is not None:
                        try:
                            progress_dialog.setRange(0, 0)
                            progress_dialog.update_progress(0, "Searching for nearby BUR files...")
                        except Exception:
                            pass
                    QtWidgets.QApplication.processEvents()
                    burst_slices = self._compute_burst_slices_for_files(tttr_files)

                if burst_slices:
                    try:
                        total_slices = sum(len(v) for v in burst_slices.values())
                    except Exception:
                        total_slices = 0
                    logging.info(f"PDA: Applying burst slicing from BUR files: {total_slices} slices across {len(burst_slices)} file(s).")
                    logging.debug({'burst_slices_keys': list(burst_slices.keys())})

                filenames_arg = "|".join(tttr_files)
                logging.debug({'tttr_files': tttr_files})
                if progress_bar is not None:
                    try:
                        progress_bar.setMinimum(0)
                        progress_bar.setMaximum(0)
                        progress_bar.setValue(0)
                    except Exception:
                        pass
                if status_label is not None:
                    try:
                        status_label.setText("Loading TTTR data and computing histograms...")
                    except Exception:
                        pass
                if progress_dialog is not None:
                    try:
                        progress_dialog.setRange(0, 0)
                        progress_dialog.update_progress(0, "Loading TTTR data and computing histograms...")
                    except Exception:
                        pass
                QtWidgets.QApplication.processEvents()
                if burst_slices:
                    core_data_macros.add_dataset(experiment_reader=pda_reader, filename=filenames_arg, burst_slices=burst_slices)
                else:
                    core_data_macros.add_dataset(experiment_reader=pda_reader, filename=filenames_arg)

                logging.info(f"PDA: Loaded {len(tttr_files)} TTTR file(s).")
                try:
                    if getattr(self, "checkBox", None) is not None and self.checkBox.isChecked():
                        if hasattr(self, "file_list"):
                            self.file_list.clear()
                        self.actionParametersChanged.trigger()
                except Exception:
                    logging.warning("PDA: Auto-clear after load failed.")
            finally:
                if progress_dialog is not None:
                    try:
                        progress_dialog.finish(final_text=None, auto_close=True, close_delay_ms=0)
                    except Exception:
                        try:
                            progress_dialog.finalize(force_auto_close=True)
                        except Exception:
                            pass
                if progress_bar is not None and progress_backup is not None:
                    try:
                        mn, mx, val = progress_backup
                        progress_bar.setMinimum(mn)
                        progress_bar.setMaximum(mx)
                        progress_bar.setValue(val)
                    except Exception:
                        pass
                if status_label is not None and status_backup is not None:
                    try:
                        status_label.setText(status_backup)
                    except Exception:
                        pass
        except Exception as e:
            # Show error message and log warning
            logging.warning(f"PDA: Failed to load files: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load files: {e}")

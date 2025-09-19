"""
BID → Analysis Converter

This plugin reads burst ID (BID) files containing start/stop photon indices
and generates a burstwise analysis folder next to the corresponding TTTR data.

Workflow per BID file:
- Infer the TTTR file from the BID file name (same stem, common TTTR extensions)
- Load TTTR via tttrlib
- Compute burst summary using chisurf.fio.fluorescence.burst.generate_burst_dataframe
- Write a BUR file to analysis/bi4_bur/<stem>.bur
- Update/create analysis/Info/*.mti with total measurement time

The plugin exposes a simple GUI file dialog when launched from the Plugins menu.
It can also be called programmatically via convert_bid_file(pathlike).
"""
from __future__ import annotations

from typing import Iterable, List, Tuple, Optional, Dict, Set
import os
import pathlib
import time
import json
import numpy as np
import zipfile
import pandas as pd

import chisurf
from chisurf import logging
import chisurf.fio as io

try:
    import tttrlib
except Exception:  # pragma: no cover
    tttrlib = None  # type: ignore

# Qt is only required when launched as a GUI plugin
try:  # pragma: no cover - optional at runtime
    from qtpy import QtWidgets, QtCore
except Exception:  # pragma: no cover
    QtWidgets = None  # type: ignore
    QtCore = None  # type: ignore

from chisurf.fio.fluorescence.burst import (
    generate_burst_dataframe,
    write_dataframe_to_bur,
    write_mti_summary,
)
from chisurf.fluorescence.burst.utils import create_array_with_ones
# Detector setup wizard page for defining detectors/windows like the Trace Browser
try:
    from chisurf.gui.widgets.wizard.tttr_channel_definition import DetectorWizardPage, DetectorWizard  # type: ignore
except Exception:
    DetectorWizardPage = None  # type: ignore
    DetectorWizard = None # type: ignore


# Plugin name in menu
name = "Single-Molecule: BID→Analysis"

# Optional icon exposed for the plugin manager/UI
icon = None
try:
    # Import lazily and guard for headless environments
    from pathlib import Path as _Path
    _plugin_dir = _Path(__file__).parent
    _png = _plugin_dir / "icon.png"
    _svg = _plugin_dir / "icon.svg"
    try:
        from PyQt5.QtGui import QIcon as _QIcon  # type: ignore
    except Exception:
        _QIcon = None  # type: ignore
    if '_QIcon' in globals() and _QIcon is not None:
        if _png.exists():
            icon = _QIcon(str(_png))
        elif _svg.exists():
            icon = _QIcon(str(_svg))
except Exception:
    # Never fail import due to icon issues
    icon = None

# Supported TTTR extensions mapped to tttrlib file types
_TTTR_EXT2TYPE: Dict[str, str] = {
    ".ptu": "PTU",
    ".phu": "PHU",
    ".ht3": "HT3",
    ".ht2": "HT2",
    ".pt3": "PT3",
    ".t3r": "T3R",
    ".spc": "SPC",   # may require HDF pathway depending on source
    ".h5": "HDF5",
    ".hdf5": "HDF5",
}


def _find_tttr_by_stem(start_dir: pathlib.Path, bid_stem: str) -> Optional[pathlib.Path]:
    """Find TTTR file matching a BID stem by searching up the directory tree.

    Searches up to 4 levels up from start_dir.

    More permissive matching:
    - exact stem match (stem.ext)
    - files that start with the BID stem (stem*ext)
    - BID stem starts with the TTTR stem (to allow BID suffixes like "_state1")
    Preference order: exact > candidate.stem startswith(BID) > BID startswith(candidate.stem) > substring fallback.
    """
    def _score(candidate: pathlib.Path) -> tuple:
        cstem = candidate.stem
        if cstem == bid_stem:
            return (0, -len(cstem))
        if cstem.startswith(bid_stem):
            return (1, -len(cstem))
        if bid_stem.startswith(cstem):
            # Prefer longer candidate stems when BID has extra suffix
            return (2, -len(cstem))
        if cstem in bid_stem:
            return (3, -len(cstem))
        return (9, len(cstem))  # non-match

    current_dir = start_dir
    for _ in range(4):  # Search current dir and 3 parents
        if not current_dir.exists():
            # Go to the next parent if the current one doesn't exist
            if current_dir.parent == current_dir:
                break  # Reached root
            current_dir = current_dir.parent
            continue

        # Collect all TTTR files in this base folder
        files: List[pathlib.Path] = []
        for ext in _TTTR_EXT2TYPE.keys():
            files.extend(current_dir.glob(f"*{ext}"))

        if files:
            # Score and select best match in this directory
            scored = [(_score(p), p) for p in files]
            matched = [(s, p) for s, p in scored if s[0] < 9]
            if matched:
                # Sort by score, then by secondary metric (longer stem preferred via negative length), then by filename length
                matched.sort(key=lambda sp: (sp[0][0], sp[0][1], len(sp[1].name)))
                return matched[0][1]

        # Move to the parent directory for the next iteration
        if current_dir.parent == current_dir:
            break  # Reached the root of the filesystem
        current_dir = current_dir.parent

    return None


def _load_tttr(tttr_path: pathlib.Path) -> tttrlib.TTTR:
    if tttrlib is None:
        raise RuntimeError("tttrlib is not available; cannot load TTTR files")
    ext = tttr_path.suffix.lower()
    ftype = _TTTR_EXT2TYPE.get(ext)
    if ftype is None:
        raise ValueError(f"Unsupported TTTR file extension: {ext}")
    return tttrlib.TTTR(str(tttr_path), ftype)


def _default_windows_detectors(tttr: "tttrlib.TTTR") -> Tuple[dict, dict]:
    """Create simple default windows and detectors covering the full range.

    - windows: one window 'All' covering full micro-time range
    - detectors: one entry per routing channel, full micro-time range
    """
    micro = tttr.micro_times
    rout = tttr.routing_channel
    max_micro = int(micro.max()) if len(micro) else 0
    # Half-open intervals [start, end) used in generate_burst_dataframe
    windows = {"All": (0, max_micro + 1)}

    unique_chs = np.unique(rout) if len(rout) else np.array([0], dtype=int)
    detectors = {}
    for ch in unique_chs.tolist():
        detectors[f"ch{int(ch)}"] = {
            "chs": [int(ch)],
            "micro_time_ranges": [(0, max_micro + 1)],
        }
    return windows, detectors


def _read_bid_start_stop(bid_path: pathlib.Path) -> List[Tuple[int, int]]:
    """Read BID file assumed to contain two integer columns: start stop (inclusive or exclusive stop).

    The file is interpreted as start and stop indices. If stop <= start, the row is ignored later.
    """
    arr = np.loadtxt(str(bid_path), dtype=np.int64)
    if arr.ndim == 1 and arr.size >= 2:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return []
    # Normalize to list of tuples
    start_stop = [(int(s), int(e)) for s, e in arr[:, :2]]
    return start_stop


def _get_unique_folder_path(base_path: pathlib.Path) -> pathlib.Path:
    """Return a unique folder path by appending -NNN if needed (like the wizard)."""
    def name_taken(p: pathlib.Path) -> bool:
        return p.exists()

    if not name_taken(base_path):
        return base_path
    # try suffixes -001, -002, ... -999
    for i in range(1, 1000):
        s = f"{base_path.name}-{i:03d}"
        p = base_path.parent / s
        if not name_taken(p):
            return p
    # fallback timestamp
    ts = time.strftime("%Y%m%d-%H%M%S")
    return base_path.parent / f"{base_path.name}-{ts}"


def _prepare_output_dir(base_dir: pathlib.Path, target_folder_name: str, unique_folder: bool) -> pathlib.Path:
    logging.debug(f"_prepare_output_dir called with: base_dir={base_dir}, target_folder_name={target_folder_name}, unique_folder={unique_folder}")
    out_dir = base_dir / target_folder_name
    logging.debug(f"Initial out_dir: {out_dir}")
    if unique_folder:
        out_dir = _get_unique_folder_path(out_dir)
        logging.debug(f"Unique out_dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Ensured output directory exists: {out_dir}")
    return out_dir


def _infer_windows_detectors(tttr: "tttrlib.TTTR", windows: Optional[dict], detectors: Optional[dict]) -> Tuple[dict, dict]:
    if windows is None or detectors is None:
        w, d = _default_windows_detectors(tttr)
        if windows is None:
            windows = w
        if detectors is None:
            detectors = d
    return windows, detectors


def _write_info_files(output_dir: pathlib.Path, files: List[pathlib.Path], detectors: dict, windows: dict, selected_setup: Optional[str] = None, bid_files: Optional[List[pathlib.Path]] = None) -> None:
    info_dir = output_dir / 'Info'
    info_dir.mkdir(parents=True, exist_ok=True)

    # Derive channels and microtime ranges from detectors
    channels = []
    microtime_ranges = []
    for det in detectors.values():
        chs = det.get("chs", [])
        channels.extend(int(x) for x in chs)
        mts = det.get("micro_time_ranges", [])
        for r in mts:
            if r not in microtime_ranges:
                microtime_ranges.append(r)
    params = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'selected_setup': (selected_setup or 'BID'),
        'channels': sorted(set(channels)),
        'decay_coarse': None,
        'microtime_ranges': microtime_ranges,
        'files': [p.name for p in files],
        'filter_mode': 'bid',
        'filter_active': False,
        'use_gap_fill': False,
        'max_gap': 0,
    }
    with open(info_dir / 'photon_selection_parameters.json', 'w') as f:
        json.dump(params, f, indent=4)
    with open(info_dir / 'datetime.txt', 'w') as f:
        now = time.strftime('%Y-%m-%d'), time.strftime('%H:%M:%S')
        f.write(f"Date: {now[0]}\nTime: {now[1]}\n")

    if bid_files:
        with open(info_dir / 'burst_files.txt', 'w') as f:
            for p in bid_files:
                f.write(f"{p.name}\n")


def _write_sl5(output_dir: pathlib.Path, tttr_path: pathlib.Path, filetype: str, selected_mask: np.ndarray) -> None:
    sl5_dir = output_dir / 'sl5'
    sl5_dir.mkdir(parents=True, exist_ok=True)
    data = {
        'filename': os.path.relpath(tttr_path, output_dir),
        'filetype': filetype,
        'count_rate_filter': {},
        'delta_macro_time_filter': {},
        'filter': io.compress_numpy_array(selected_mask.astype(np.uint8))
    }
    out_file = sl5_dir / f"{tttr_path.stem}.json.gz"
    with io.open_maybe_zipped(out_file, 'w') as f:
        f.write(json.dumps(data))


def _write_hdf5_combined(output_dir: pathlib.Path, dataframes: List[pd.DataFrame]) -> None:
    import pandas as pd
    import numpy as np

    if not dataframes:
        return
    combined = pd.concat(dataframes, ignore_index=True)
    # drop empty cols
    combined = combined.dropna(axis=1, how='all')

    # downcast
    for c in combined.select_dtypes(include=["integer"]).columns:
        if (combined[c] >= 0).all():
            combined[c] = pd.to_numeric(combined[c], downcast="unsigned")
        else:
            combined[c] = pd.to_numeric(combined[c], downcast="integer")
    for c in combined.select_dtypes(include=["floating"]).columns:
        combined[c] = combined[c].astype(np.float32)

    # encode objects as categorical
    cat_map = {}
    obj_cols = combined.select_dtypes(include=["object"]).columns
    must_encode = {"Source File", "First File", "Last File"} & set(obj_cols)
    for col in obj_cols:
        nunique = combined[col].nunique(dropna=False)
        if (col in must_encode) or (nunique <= 0.5 * len(combined)):
            cat = pd.Categorical(combined[col], ordered=False)
            cat_map[col] = cat.categories.tolist()
            combined[col] = cat.codes.astype(np.int32)
        else:
            cat = pd.Categorical(combined[col], ordered=False)
            cat_map[col] = cat.categories.tolist()
            combined[col] = cat.codes.astype(np.int32)

    # choose compression
    complib = "bzip2"
    try:
        pd.HDFStore(str(output_dir / '___tmp__.h5'), mode='w', complib=complib).close()
        (output_dir / '___tmp__.h5').unlink(missing_ok=True)
    except Exception:
        complib = "blosc"
        try:
            pd.HDFStore(str(output_dir / '___tmp__.h5'), mode='w', complib=complib).close()
            (output_dir / '___tmp__.h5').unlink(missing_ok=True)
        except Exception:
            complib = "zlib"

    hdf5_dir = output_dir / 'hdf5'
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    h5_file = hdf5_dir / f"burst_data_{timestamp}.h5"
    with pd.HDFStore(h5_file, mode='w', complib=complib, complevel=9) as store:
        store.put('results', combined, format='fixed', index=False)
        st = store.get_storer('results')
        st.attrs.category_map = json.dumps(cat_map)


def zip_output_folder(output_folder: pathlib.Path) -> Optional[pathlib.Path]:
    """Zip the output folder to a unique .zip next to it; return zip path."""
    def unique_zip_path(base: pathlib.Path) -> pathlib.Path:
        p = base.with_suffix('.zip')
        if not p.exists():
            return p
        for i in range(1, 1000):
            p2 = base.parent / f"{base.name}-{i:03d}.zip"
            if not p2.exists():
                return p2
        ts = time.strftime('%Y%m%d-%H%M%S')
        return base.parent / f"{base.name}-{ts}.zip"

    output_folder = pathlib.Path(output_folder)
    zip_path = unique_zip_path(output_folder)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_folder):
            root_path = pathlib.Path(root)
            for f in files:
                fp = root_path / f
                arc = fp.relative_to(output_folder)
                zf.write(str(fp), arcname=str(arc))
    return zip_path


def _per_file_process(bid_path: pathlib.Path, output_dir: pathlib.Path, windows: Optional[dict], detectors: Optional[dict], output_types: Set[str], bid_index: int = 1) -> Tuple[pathlib.Path, 'tttrlib.TTTR', dict, dict, 'pd.DataFrame', np.ndarray]:
    import pandas as pd
    bid_path = pathlib.Path(bid_path)
    stem = bid_path.stem
    tttr_path = _find_tttr_by_stem(bid_path.parent, stem)
    if tttr_path is None:
        raise FileNotFoundError(f"Could not locate TTTR file for BID '{bid_path.name}' (stem='{stem}')")
    tttr = _load_tttr(tttr_path)

    windows, detectors = _infer_windows_detectors(tttr, windows, detectors)

    start_stop = _read_bid_start_stop(bid_path)
    df = generate_burst_dataframe(
        start_stop=start_stop,
        filename=str(tttr_path),
        tttr=tttr,
        windows=windows,
        detectors=detectors,
        include_interleaved_zeros=("bur" in output_types),
    )

    # Tag bursts with BID origin information
    try:
        df["BID File"] = bid_path.name
        df["BID Index"] = int(bid_index)
    except Exception:
        # Be tolerant if df is None or immutable
        pass

    # BUR output
    bur_path = None
    if "bur" in output_types:
        bur_dir = output_dir / 'bi4_bur'
        bur_dir.mkdir(parents=True, exist_ok=True)
        bur_path = bur_dir / f"{tttr_path.stem}.bur"
        # If a BUR for this TTTR already exists, append new bursts instead of overwriting
        try:
            include_zeros = True  # we used include_interleaved_zeros when creating df
            if bur_path.exists():
                try:
                    existing_df = pd.read_csv(bur_path, sep='\t')
                except Exception:
                    existing_df = None
                # Avoid duplicate leading zero-row when appending: drop first row of the new df
                df_to_append = df.iloc[1:].copy() if include_zeros and len(df) > 0 else df
                if existing_df is not None:
                    combined = pd.concat([existing_df, df_to_append], ignore_index=True, sort=False)
                    combined.to_csv(bur_path, sep='\t', index=False)
                else:
                    write_dataframe_to_bur(df_to_append, str(bur_path))
            else:
                write_dataframe_to_bur(df, str(bur_path))
        except Exception as _append_exc:
            # Fallback to simple write if anything went wrong during append logic
            write_dataframe_to_bur(df, str(bur_path))
        # MTI summary
        try:
            max_macro_time_s = float(tttr.macro_times[-1]) * float(tttr.header.macro_time_resolution)
        except Exception:
            max_macro_time_s = 0.0
        write_mti_summary(tttr_path, output_dir, max_macro_time_s, append=True)

    # SL5 output
    selected = None
    if "sl5" in output_types:
        n = len(tttr)
        selected = create_array_with_ones(np.array(start_stop, dtype=int), n).astype(np.uint8)
        _write_sl5(output_dir, tttr_path, _TTTR_EXT2TYPE.get(tttr_path.suffix.lower(), ""), selected)

    # For HDF5 collection
    if df is not None:
        df_copy = df.copy()
        df_copy['Source File'] = str(tttr_path)
    else:
        df_copy = df

    return bur_path, tttr, detectors, windows, df_copy, selected


def convert_bid_file(
    bid_path: os.PathLike | str,
    analysis_folder: os.PathLike | str | None = None,
    *,
    output_types: Optional[Set[str]] = None,
    windows: Optional[dict] = None,
    detectors: Optional[dict] = None,
    target_path: str = 'analysis',
    unique_folder: bool = False,
    zip_output: bool = False,
    remove_folder: bool = False,
    selected_setup: Optional[str] = None,
    bid_index: int = 1,
) -> pathlib.Path:
    """Convert a single BID file into an analysis folder.

    By default writes BUR + MTI. Optional outputs: SL5 and combined HDF5.

    Returns the path to the produced BUR file (if BUR was requested), else the output directory.
    """
    bid_path = pathlib.Path(bid_path)
    if not bid_path.exists():
        raise FileNotFoundError(bid_path)

    # Determine output directory
    # If analysis_folder provided, use as-is (no unique suffix unless unique_folder=True)
    # else create under TTTR parent / target_path
    # We need tttr_path to decide default; locate it first without loading
    stem = bid_path.stem
    tttr_path = _find_tttr_by_stem(bid_path.parent, stem)
    if tttr_path is None:
        raise FileNotFoundError(f"Could not locate TTTR file for BID '{bid_path.name}' (stem='{stem}')")
    output_dir = _prepare_output_dir(tttr_path.parent, bid_path.stem, unique_folder)

    # Default outputs
    if output_types is None:
        output_types = {"bur"}

    # Process this single file
    bur_path, tttr, dets, wins, df, selected = _per_file_process(bid_path, output_dir, windows, detectors, output_types, bid_index=bid_index)

    # Info files
    _write_info_files(output_dir, [tttr_path], dets, wins, selected_setup=selected_setup)

    # Combined HDF5: for single file, just write immediately
    if "hdf5" in output_types and df is not None:
        _write_hdf5_combined(output_dir, [df])

    # Zip handling
    if zip_output:
        zip_name = zip_output_folder(output_dir)  # define below
        if remove_folder and zip_name:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

    return bur_path if ("bur" in output_types and bur_path is not None) else output_dir


def convert_many(
    bid_paths: Iterable[os.PathLike | str],
    analysis_folder: os.PathLike | str | None = None,
    *,
    output_types: Optional[Set[str]] = None,
    windows: Optional[dict] = None,
    detectors: Optional[dict] = None,
    target_path: str = 'analysis',
    unique_folder: bool = True,
    zip_output: bool = False,
    remove_folder: bool = False,
    selected_setup: Optional[str] = None,
) -> List[pathlib.Path]:
    bid_paths = [pathlib.Path(p) for p in bid_paths]
    if not bid_paths:
        return []

    # Resolve output directory using first file's TTTR
    first_tttr = _find_tttr_by_stem(bid_paths[0].parent, bid_paths[0].stem)
    if first_tttr is None:
        raise FileNotFoundError(f"Could not locate TTTR file for BID '{bid_paths[0].name}'")
    output_dir = _prepare_output_dir(first_tttr, analysis_folder, target_path, unique_folder)

    if output_types is None:
        output_types = {"bur"}

    bur_paths: List[pathlib.Path] = []
    dfs_for_hdf5: List[pd.DataFrame] = []
    tttr_paths_for_info: List[pathlib.Path] = []

    for i, p in enumerate(bid_paths, start=1):
        try:
            bur_path, tttr, dets, wins, df, selected = _per_file_process(p, output_dir, windows, detectors, output_types, bid_index=i)
            if bur_path is not None:
                bur_paths.append(bur_path)
            if df is not None:
                dfs_for_hdf5.append(df)
            # track tttr path for info listing
            tttr_paths_for_info.append(_find_tttr_by_stem(p.parent, p.stem))
        except Exception as exc:
            logging.error(f"Failed to convert {p}: {exc}")

    # Write info once including all files
    if tttr_paths_for_info:
        # Use the last computed detectors/windows if provided; else default values using first tttr
        if detectors is None or windows is None:
            tmp_tttr = _load_tttr(first_tttr)
            w, d = _infer_windows_detectors(tmp_tttr, windows, detectors)
        else:
            w, d = windows, detectors
        _write_info_files(output_dir, [pathlib.Path(tp) for tp in tttr_paths_for_info if tp], d, w, selected_setup=selected_setup)

    if "hdf5" in output_types and dfs_for_hdf5:
        _write_hdf5_combined(output_dir, dfs_for_hdf5)

    if zip_output:
        zip_name = zip_output_folder(output_dir)
        if remove_folder and zip_name:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

    return bur_paths


# --- Simple GUI with drag-and-drop list ---
if QtWidgets is not None:
    class BidToAnalysisGUI(QtWidgets.QDialog):
        BID_EXTS = {".bid", ".bst", ".txt"}

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("BID  Analysis Converter")
            self.resize(900, 520)
            self.setAcceptDrops(True)
            self._setup_ui()

        def _setup_ui(self):
            layout = QtWidgets.QVBoxLayout(self)

            # Tabbed interface: Setup and Files
            self.tabs = QtWidgets.QTabWidget(self)
            layout.addWidget(self.tabs)

            # Setup tab
            setup_widget = QtWidgets.QWidget(self)
            setup_layout = QtWidgets.QVBoxLayout(setup_widget)
            if DetectorWizardPage is not None:
                try:
                    self.detector_page = DetectorWizardPage(
                        show_help=False,
                        show_setups_file=True,
                        show_setup_selection=True,
                        show_tttr_reading=True,
                        show_tables=True,
                        show_add_inputs=True,
                    )
                except Exception:
                    self.detector_page = None
            else:
                self.detector_page = None
            if self.detector_page is not None:
                setup_layout.addWidget(self.detector_page)
            else:
                lbl = QtWidgets.QLabel("Detector setup UI not available. Default auto-detectors will be used.", setup_widget)
                lbl.setWordWrap(True)
                setup_layout.addWidget(lbl)
            self.tabs.addTab(setup_widget, "Setup")

            # Files tab (existing UI)
            files_widget = QtWidgets.QWidget(self)
            files_layout = QtWidgets.QVBoxLayout(files_widget)

            self.table = QtWidgets.QTableWidget(0, 4, files_widget)
            self.table.setHorizontalHeaderLabels(["BID file", "TTTR file", "Output folder", "Status"])
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            try:
                self.table.horizontalHeader().setStretchLastSection(True)
            except Exception:
                pass
            files_layout.addWidget(self.table)

            btns = QtWidgets.QHBoxLayout()
            self.btnAdd = QtWidgets.QPushButton("Add Files", files_widget)
            self.btnClear = QtWidgets.QPushButton("Clear", files_widget)
            self.btnProcess = QtWidgets.QPushButton("Process", files_widget)
            btns.addWidget(self.btnAdd)
            btns.addStretch(1)
            btns.addWidget(self.btnClear)
            btns.addWidget(self.btnProcess)
            files_layout.addLayout(btns)

            self.log = QtWidgets.QTextEdit(files_widget)
            self.log.setReadOnly(True)
            self.log.setPlaceholderText("Logs will appear here")
            files_layout.addWidget(self.log)

            self.tabs.addTab(files_widget, "Files")

            # Connect buttons
            self.btnAdd.clicked.connect(self.on_add_files)
            self.btnClear.clicked.connect(self.on_clear)
            self.btnProcess.clicked.connect(self.process_all)

        # --- Drag & Drop ---
        def dragEnterEvent(self, e):
            if getattr(QtCore, "Qt", None) is None:
                e.ignore()
                return
            if e.mimeData().hasUrls():
                for url in e.mimeData().urls():
                    p = pathlib.Path(url.toLocalFile())
                    if p.suffix.lower() in self.BID_EXTS:
                        e.acceptProposedAction()
                        return
            e.ignore()

        def dropEvent(self, e):
            paths = []
            for url in e.mimeData().urls():
                p = pathlib.Path(url.toLocalFile())
                if p.suffix.lower() in self.BID_EXTS and p.exists():
                    paths.append(p)
            if paths:
                self.add_files(paths)

        # --- UI actions ---
        def on_add_files(self):
            dlg = QtWidgets.QFileDialog(self, "Select BID files")
            dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
            dlg.setNameFilters(["BID files (*.bid *.bst *.txt)", "All files (*.*)"])
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                paths = [pathlib.Path(p) for p in dlg.selectedFiles()]
                self.add_files(paths)

        def on_clear(self):
            self.table.setRowCount(0)
            self.log.clear()

        def add_files(self, paths: List[pathlib.Path]):
            for p in paths:
                if p.suffix.lower() not in self.BID_EXTS:
                    continue
                self._add_row(p)

        def _add_row(self, bid: pathlib.Path):
            # prevent duplicates
            for r in range(self.table.rowCount()):
                item = self.table.item(r, 0)
                if item and item.data(QtCore.Qt.UserRole) == str(bid):
                    return
            row = self.table.rowCount()
            self.table.insertRow(row)
            bid_item = QtWidgets.QTableWidgetItem(str(bid))
            bid_item.setData(QtCore.Qt.UserRole, str(bid))
            self.table.setItem(row, 0, bid_item)
            tttr = _find_tttr_by_stem(bid.parent, bid.stem)
            tttr_text = str(tttr) if tttr else "TTTR not found"
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(tttr_text))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(""))
            self._set_status(row, "Ready" if tttr else "Missing TTTR", error=(tttr is None))

        def _set_status(self, row: int, text: str, error: bool=False):
            it = self.table.item(row, 3)
            if it is None:
                it = QtWidgets.QTableWidgetItem("")
                self.table.setItem(row, 3, it)
            it.setText(text)

        def log_info(self, msg: str):
            try:
                logging.info(msg)
            except Exception:
                pass
            self.log.append(msg)
            self.log.ensureCursorVisible()

        def process_all(self):
            n = self.table.rowCount()
            if n == 0:
                return

            # Read detector/windows from setup page if available
            setup_windows = None
            setup_detectors = None
            setup_name = None
            if getattr(self, 'detector_page', None) is not None:
                try:
                    settings = self.detector_page.get_settings()
                    setup_windows = settings.get('windows') or None
                    setup_detectors = settings.get('detectors') or None
                    setup_name = getattr(self.detector_page, 'current_setup_name', None)
                except Exception:
                    setup_windows = None
                    setup_detectors = None
                    setup_name = None

            ok = 0
            for row in range(n):
                bid_item = self.table.item(row, 0)
                tttr_item = self.table.item(row, 1)
                if not bid_item:
                    continue
                bid = pathlib.Path(bid_item.data(QtCore.Qt.UserRole))
                tttr_path = pathlib.Path(tttr_item.text()) if tttr_item and tttr_item.text() and tttr_item.text() != "TTTR not found" else None
                if tttr_path is None or not tttr_path.exists():
                    self._set_status(row, "TTTR not found", error=True)
                    continue
                self._set_status(row, "Processing", error=False)
                QtWidgets.QApplication.processEvents()
                # try:
                bur_path = convert_bid_file(
                    bid,
                    windows=setup_windows,
                    detectors=setup_detectors,
                    selected_setup=setup_name,
                    bid_index=row + 1,
                )
                output_folder = None
                if isinstance(bur_path, pathlib.Path):
                    if bur_path.suffix.lower() == ".bur":
                        output_folder = bur_path.parent.parent
                    else:
                        output_folder = bur_path
                if output_folder:
                    self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(output_folder)))
                self._set_status(row, "Done", error=False)
                self.log_info(f"Processed {bid.name} -> {tttr_path.name}")
                ok += 1
            # except Exception as exc:
            #     self._set_status(row, f"Error: {exc}", error=True)
            #     self.log_info(f"Error processing {bid.name}: {exc}")
            self.log_info(f"Finished: {ok}/{n} files processed.")

# --- GUI entry ---
if __name__ == "plugin":  # launched from Plugins menu
    if QtWidgets is None:
        raise RuntimeError("Qt is not available; cannot run GUI plugin")
    app = QtWidgets.QApplication.instance()
    parent = None if app is None else app.activeWindow()
    window = BidToAnalysisGUI(parent)
    window.show()

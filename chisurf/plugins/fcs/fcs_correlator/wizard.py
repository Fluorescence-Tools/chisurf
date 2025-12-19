import sys
import pathlib
from typing import List, Optional
import typing

from chisurf.gui import QtWidgets, QtGui, QtCore

import chisurf
import chisurf.gui
import chisurf.gui.widgets.wizard
import chisurf.gui.widgets
import chisurf.gui.widgets.parameter_editor

import chisurf.data
import chisurf.experiments
import chisurf.curve
import chisurf.fitting

import chisurf.macros
import chisurf.settings

import tttrlib


class FileListWidget(QtWidgets.QListWidget):
    """
    Minimal file list widget for tttr files:
    - Accepts file/folder drops
    - Maintains a unique, sorted list of paths
    - Shows per-item checkboxes (default: checked)
    - Lets user optionally select rows (UI convenience), but processing is based on checkboxes.
    """
    def __init__(self, parent=None, file_added_callback=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.file_added_callback = file_added_callback
        # Allow the file list to grow vertically and fill available space
        sp = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(sp)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        paths = []
        for url in event.mimeData().urls():
            p = pathlib.Path(url.toLocalFile())
            if p.exists():
                paths.append(str(p))
        self.add_files(paths)
        event.acceptProposedAction()
        if self.file_added_callback:
            self.file_added_callback()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        menu = QtWidgets.QMenu(self)
        act_check_all = menu.addAction("Check all")
        act_uncheck_all = menu.addAction("Uncheck all")
        menu.addSeparator()
        act_remove = menu.addAction("Remove selected")
        act_clear = menu.addAction("Clear all")
        chosen = menu.exec_(event.globalPos())
        if chosen == act_check_all:
            for i in range(self.count()):
                it = self.item(i)
                it.setCheckState(QtCore.Qt.Checked)
        elif chosen == act_uncheck_all:
            for i in range(self.count()):
                it = self.item(i)
                it.setCheckState(QtCore.Qt.Unchecked)
        elif chosen == act_remove:
            for it in self.selectedItems():
                self.takeItem(self.row(it))
        elif chosen == act_clear:
            self.clear()

    def add_files(self, file_paths: typing.List[str]):
        if not file_paths:
            return
        # Build set of existing
        existing = {self.item(i).text() for i in range(self.count())}
        new_items = []
        for fp in file_paths:
            if fp not in existing:
                new_items.append(fp)
                existing.add(fp)
        new_items.sort()
        self.blockSignals(True)
        for fp in new_items:
            item = QtWidgets.QListWidgetItem(fp)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            item.setCheckState(QtCore.Qt.Checked)
            self.addItem(item)
        self.blockSignals(False)

    def all_files(self) -> typing.List[str]:
        return [self.item(i).text() for i in range(self.count())]

    def checked_files(self) -> typing.List[str]:
        files = []
        for i in range(self.count()):
            it = self.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                files.append(it.text())
        return files


class FileAndStepsPage(QtWidgets.QWizardPage):
    """
    Page 1: file list + step selection (Photon filter, FCS merger).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Files and Steps")

        # UI
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.file_list = FileListWidget(self, file_added_callback=self._files_or_checks_changed)
        form.addRow("Files:", self.file_list)

        checks = QtWidgets.QHBoxLayout()
        self.cb_photon_filter = QtWidgets.QCheckBox("Count rate/burst filter")
        self.cb_fcs_merger = QtWidgets.QCheckBox("FCS merger")
        # Default: disable photon/count-rate filter, enable FCS merger
        self.cb_photon_filter.setChecked(False)
        self.cb_fcs_merger.setChecked(True)
        checks.addWidget(self.cb_photon_filter)
        checks.addWidget(self.cb_fcs_merger)
        form.addRow("Steps:", QtWidgets.QWidget())
        form.itemAt(form.rowCount()-1, QtWidgets.QFormLayout.FieldRole).widget().setLayout(checks)

        layout.addLayout(form, 1)
        # Initialize step availability based on current file list
        try:
            self._update_step_availability()
        except Exception:
            pass

        # Connections
        self.cb_photon_filter.toggled.connect(self._files_or_checks_changed)
        self.cb_fcs_merger.toggled.connect(self._files_or_checks_changed)
        self.file_list.itemSelectionChanged.connect(self._files_or_checks_changed)
        try:
            self.file_list.itemChanged.connect(self._files_or_checks_changed)
        except Exception:
            pass

    # QWizardPage API
    def isComplete(self) -> bool:
        return self.file_list.count() > 0

    def nextId(self) -> int:
        wiz = self.wizard()
        if getattr(wiz, 'filter_page_id', None) is None:
            return -1
        return wiz.filter_page_id if self.cb_photon_filter.isChecked() else wiz.correlator_page_id

    def _emit_complete(self):
        self.completeChanged.emit()

    def _files_or_checks_changed(self):
        # Update step availability based on selected/checked files and emit completion change
        try:
            self._update_step_availability()
        finally:
            self._emit_complete()

    def _update_step_availability(self):
        """
        Disable the 'Count rate/burst filter' step if any checked file is a .bst Burst-ID file.
        Re-enable it otherwise.
        """
        try:
            files = self.checked_files
        except Exception:
            files = []
        has_bst = False
        for f in files:
            try:
                if pathlib.Path(f).suffix.lower() == '.bst':
                    has_bst = True
                    break
            except Exception:
                continue
        if has_bst:
            # Turn off and disable the photon filter step in presence of BST files
            try:
                self.cb_photon_filter.setChecked(False)
            except Exception:
                pass
            try:
                self.cb_photon_filter.setEnabled(False)
                self.cb_photon_filter.setToolTip("Disabled when Burst-ID (.bst) files are selected.")
            except Exception:
                pass
        else:
            try:
                self.cb_photon_filter.setEnabled(True)
                self.cb_photon_filter.setToolTip("")
            except Exception:
                pass

    @property
    def files(self) -> List[str]:
        return self.file_list.all_files()

    @property
    def selected_files(self) -> List[str]:
        # Backward-compat alias; prefer checked_files
        return self.file_list.checked_files()

    @property
    def checked_files(self) -> List[str]:
        return self.file_list.checked_files()


class CorrelatorPage(QtWidgets.QWizardPage):
    """
    Wrapper page that embeds WizardTTTRCorrelator instead of inheriting from it.
    This avoids plugin subclassing issues with UI decorators while preserving
    the same external API used by the wizard flow.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Correlator")
        # Embed the original correlator widget
        self.inner = chisurf.gui.widgets.wizard.WizardTTTRCorrelator()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.inner)
        # Expose commonly used child widgets for compatibility
        self.lineEdit = self.inner.lineEdit
        self.lineEdit_2 = self.inner.lineEdit_2
        self.lineEdit_3 = self.inner.lineEdit_3
        self.lineEdit_6 = self.inner.lineEdit_6
        self.lineEdit_7 = self.inner.lineEdit_7

    def nextId(self) -> int:
        wiz = self.wizard()
        if getattr(wiz, 'fcs_merger_page_id', None) is None:
            return -1
        return wiz.fcs_merger_page_id if wiz.file_page.cb_fcs_merger.isChecked() else -1

    # Delegate methods/properties used by the wizard
    @property
    def is_correlated(self) -> bool:
        return self.inner.is_correlated

    @property
    def analysis_folder(self):
        return self.inner.analysis_folder

    @property
    def output_path(self):
        return self.inner.output_path

    def correlate_data(self):
        return self.inner.correlate_data()

    def open_analysis_folder(self, folder: pathlib.Path = None):
        return self.inner.open_analysis_folder(folder)

    def load_tttr_files(self, filenames: List[str], filetype: Optional[str] = None):
        # Delegate to the inner correlator’s robust loader to ensure filenames are tracked
        # and the analysis folder is set appropriately.
        try:
            self.inner.load_tttr_files(filenames, filetype)
        except Exception:
            # Fallback: open each file with extension-aware type resolution
            self.inner.settings.setdefault('tttr_filenames', [])
            self.inner.settings['tttr_filenames'] = list(filenames)
            def _open_tttr_for_ui(p: pathlib.Path, global_type):
                p_str = p.as_posix()
                ext = p.suffix.lower()
                try:
                    if ext == '.spc':
                        try:
                            ft_int = tttrlib.inferTTTRFileType(p_str)
                            if ft_int is not None and ft_int >= 0:
                                return tttrlib.TTTR(p_str, ft_int)
                        except Exception:
                            pass
                        try:
                            return tttrlib.TTTR(p_str, 'SPC')
                        except Exception:
                            return tttrlib.TTTR(p_str)
                    if isinstance(global_type, str) and global_type.strip():
                        try:
                            return tttrlib.TTTR(p_str, global_type)
                        except Exception:
                            pass
                    try:
                        ft_int = tttrlib.inferTTTRFileType(p_str)
                        if ft_int is not None and ft_int >= 0:
                            return tttrlib.TTTR(p_str, ft_int)
                    except Exception:
                        pass
                    return tttrlib.TTTR(p_str)
                except Exception:
                    return None
            tttr_obj = None
            for fn in filenames:
                p = pathlib.Path(fn)
                if not p.exists() or not p.is_file():
                    continue
                tt = _open_tttr_for_ui(p, filetype)
                if tt is None:
                    continue
                if tttr_obj is None:
                    tttr_obj = tt
                else:
                    tttr_obj.append(tt)
            self.inner.tttr = tttr_obj
        # Update output path suggestion after loading
        try:
            self.inner.update_output_path()
        except Exception:
            pass


class ChisurfFCSWizard(QtWidgets.QWizard):

    def _on_detector_setup_changed(self, *_):
        """When the user changes the detector/setup selection, clear the file list
        to prevent processing files with a mismatched setup. Also clear any
        preloaded photon filter files/state for safety.
        """
        try:
            # Clear file list on the Files page
            try:
                self.file_page.file_list.clear()
                # Update step availability/completion state
                self.file_page._files_or_checks_changed()
            except Exception:
                pass
            # Clear photon filter page state (files/tttr objects) if present
            try:
                if hasattr(self.photon_select, 'onClearFiles'):
                    self.photon_select.onClearFiles()
            except Exception:
                pass
        except Exception:
            # Safety: never raise from a UI signal handler
            pass

    def _sync_photon_filter_setup(self):
        """Ensure photon filter has detector/window definitions from detector page."""
        try:
            settings = self.detector_page.get_settings()
            dets = settings.get('detectors', {}) or {}
            wins = settings.get('windows', {}) or {}
            # Sanitize empty names
            dets = {k: v for k, v in dets.items() if isinstance(k, str) and k.strip()}
            wins = {k: v for k, v in wins.items() if isinstance(k, str) and k.strip()}
            # Refill detectors and windows with signals blocked to avoid transient '' updates
            cb2 = getattr(self.photon_select, 'comboBox_2', None)
            cb3 = getattr(self.photon_select, 'comboBox_3', None)
            prev2 = cb2.blockSignals(True) if cb2 is not None else None
            prev3 = cb3.blockSignals(True) if cb3 is not None else None
            try:
                if cb2 is not None:
                    cb2.clear()
                if cb3 is not None:
                    cb3.clear()
                self.photon_select.fill_detectors(dets)
                self.photon_select.fill_pie_windows(wins)
                # Ensure a defined selection
                if cb2 is not None and cb2.count() > 0:
                    cb2.setCurrentIndex(0)
                if cb3 is not None and cb3.count() > 0:
                    cb3.setCurrentIndex(0)
            except Exception:
                pass
            finally:
                try:
                    if cb2 is not None:
                        cb2.blockSignals(prev2 if isinstance(prev2, bool) else False)
                    if cb3 is not None:
                        cb3.blockSignals(prev3 if isinstance(prev3, bool) else False)
                except Exception:
                    pass
            # Manually update dependent fields after safe selection
            try:
                self.photon_select.update_detectors()
            except Exception:
                pass
            try:
                self.photon_select.update_pie_windows()
            except Exception:
                pass
            # Try to align setup name in photon filter if available
            setup_name = getattr(self.detector_page, 'current_setup_name', None)
            if setup_name:
                try:
                    idx = self.photon_select.comboBox.findText(setup_name)
                    if idx >= 0:
                        self.photon_select.comboBox.setCurrentIndex(idx)
                except Exception:
                    pass
        except Exception:
            pass

    def _sync_photon_filter_files(self):
        """
        Ensure self.photon_select is loaded with the files currently selected
        in the Files page (or all files if none selected). This is safe to call
        whenever the Photon Filter page is about to be shown.
        """
        try:
            # Only act if Photon filter step is enabled
            if not self.file_page.cb_photon_filter.isChecked():
                return

            files = self.file_page.checked_files
            # Expand directories to allowed TTTR files (container names) and pass through .bst
            allowed_extensions = {
                f".{ext.lower()}" if not ext.startswith('.') else ext.lower()
                for ext in tttrlib.TTTR.get_supported_container_names()
            }
            expanded_files: List[str] = []
            for p_str in files:
                p = pathlib.Path(p_str).resolve()
                if p.is_dir():
                    for child in p.iterdir():
                        if child.is_file() and (child.suffix.lower() in allowed_extensions or child.suffix.lower() == '.bst'):
                            expanded_files.append(str(child.resolve()))
                else:
                    expanded_files.append(str(p))

            # Preload dropped files into photon filter page using detector page's filetype
            filetype = self.detector_page.filetype

            # Resolve any .bst entries to their underlying TTTR file so that the
            # photon filter (which expects raw TTTR files) can open them.
            def _resolve_bst_to_tttr(path: pathlib.Path) -> Optional[pathlib.Path]:
                try:
                    base_with_ext = path.name[:-4]
                    candidates = [path.parent]
                    if path.parent.parent:
                        candidates.append(path.parent.parent)
                    if path.parent.parent.parent:
                        candidates.append(path.parent.parent.parent)
                    if path.parent.parent.parent.parent:
                        candidates.append(path.parent.parent.parent.parent)
                    for folder in candidates:
                        cand = folder / base_with_ext
                        if cand.exists() and cand.is_file():
                            return cand.resolve()
                except Exception:
                    pass
                return None

            actual_files: List[str] = []
            for fn in expanded_files:
                p = pathlib.Path(fn).resolve()
                if p.suffix.lower() == '.bst':
                    tttr_resolved = _resolve_bst_to_tttr(p)
                    if tttr_resolved is not None:
                        actual_files.append(str(tttr_resolved))
                else:
                    actual_files.append(str(p))

            # Load TTTR objects similar to PhotonFilter's after_file_drop, with extension-aware type resolution
            self.photon_select.tttr_objects = dict()
            def _open_tttr_for_ui(p: pathlib.Path, global_type):
                p_str = str(p)
                ext = p.suffix.lower()
                # Always prefer inference for .spc to avoid forcing wrong reader (e.g., PTU on SPC)
                try:
                    if ext == '.spc':
                        try:
                            ft_int = tttrlib.inferTTTRFileType(p_str)
                            if ft_int is not None and ft_int >= 0:
                                return tttrlib.TTTR(p_str, ft_int)
                        except Exception:
                            pass
                        # As a fallback, try the string 'SPC' if supported, else auto
                        try:
                            return tttrlib.TTTR(p_str, 'SPC')
                        except Exception:
                            return tttrlib.TTTR(p_str)
                    # For other files: use provided type if any; otherwise try inference, then auto
                    if isinstance(global_type, str) and global_type.strip():
                        try:
                            return tttrlib.TTTR(p_str, global_type)
                        except Exception:
                            # Fallback to inference if the provided type fails
                            pass
                    try:
                        ft_int = tttrlib.inferTTTRFileType(p_str)
                        if ft_int is not None and ft_int >= 0:
                            return tttrlib.TTTR(p_str, ft_int)
                    except Exception:
                        pass
                    return tttrlib.TTTR(p_str)
                except Exception as e:
                    raise e

            for fn in actual_files:
                p = pathlib.Path(fn).resolve()
                p_str = str(p)
                if p_str in self.photon_select.tttr_objects:
                    continue
                if not p.exists() or not p.is_file():
                    continue
                try:
                    self.photon_select.tttr_objects[p_str] = _open_tttr_for_ui(p, filetype)
                except Exception:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Error Loading File",
                        f"Failed to load file '{p.name}' with the selected setup. Please check the file type."
                    )
                    self.photon_select.onClearFiles()
                    return

            self.photon_select.settings['tttr_filenames'] = list(actual_files)
            n_files = len(actual_files)
            self.photon_select.spinBox_4.setMaximum(n_files - 1 if n_files > 0 else 0)
            if n_files > 0:
                # Set current file index to the last one, assign path to lineEdit and load
                self.photon_select.spinBox_4.setValue(n_files - 1)
                self.photon_select.lineEdit.setText(expanded_files[0])
                try:
                    self.photon_select.read_tttr()
                except Exception:
                    pass

            # Disable photon filter's own file list controls while used in the wizard
            try:
                self.photon_select.lineEdit.setEnabled(False)
                self.photon_select.lineEdit.setDragEnabled(False)
                self.photon_select.lineEdit.setAcceptDrops(False)
            except Exception:
                pass
            try:
                self.photon_select.toolButton_6.setEnabled(False)
            except Exception:
                pass
        except Exception:
            # Don't block navigation due to sync errors
            pass

    def _on_current_id_changed(self, new_id: int):
        # Whenever the filter page becomes active, ensure setup and files are synced
        if getattr(self, 'filter_page_id', None) is not None and new_id == self.filter_page_id:
            self._sync_photon_filter_setup()
            self._sync_photon_filter_files()
        # When navigating to the correlator page, ensure the correlator is set up
        if getattr(self, 'correlator_page_id', None) is not None and new_id == self.correlator_page_id:
            try:
                files = self.file_page.checked_files
                allowed_extensions = {
                    f".{ext.lower()}" if not ext.startswith('.') else ext.lower()
                    for ext in tttrlib.TTTR.get_supported_container_names()
                }
                expanded_files: List[str] = []
                for p_str in files:
                    p = pathlib.Path(p_str).resolve()
                    if p.is_dir():
                        for child in p.iterdir():
                            if child.is_file() and child.suffix.lower() in allowed_extensions:
                                expanded_files.append(str(child.resolve()))
                    else:
                        expanded_files.append(str(p))
                # Set default analysis folder
                if expanded_files:
                    parent = pathlib.Path(expanded_files[0]).resolve().parent
                    self.correlator_page.lineEdit_3.setText(parent.as_posix())
                # If Photon filter is disabled, we are in direct TTTR mode: don't write chunk files
                try:
                    if not self.file_page.cb_photon_filter.isChecked():
                        self.correlator_page.inner.save_chunks_to_disk = False
                except Exception:
                    pass
                # Load into correlator in direct TTTR mode
                try:
                    if not self.file_page.cb_photon_filter.isChecked():
                        filetype = self.detector_page.filetype
                        self.correlator_page.load_tttr_files(expanded_files, filetype)
                except Exception:
                    pass
                # Always populate correlator combos from Detector page definitions
                try:
                    if hasattr(self.correlator_page, 'inner') and hasattr(self.correlator_page.inner, 'apply_detector_setup_from_page'):
                        self.correlator_page.inner.apply_detector_setup_from_page(self.detector_page)
                except Exception:
                    pass
                # Fallback: if combos absent or empty, prefill channel line edits from detectors
                try:
                    settings = self.detector_page.get_settings()
                    dets = settings.get('detectors', {}) or {}
                    det_names = [k for k in dets.keys() if isinstance(k, str) and k.strip()]
                    if det_names:
                        a = dets[det_names[0]].get('chs', [])
                        b = dets[det_names[1]].get('chs', a) if len(det_names) > 1 else a
                        # Only apply fallback if edits are empty
                        if not str(self.correlator_page.lineEdit.text()).strip():
                            self.correlator_page.lineEdit.setText(','.join(map(str, a)))
                        if not str(self.correlator_page.lineEdit_2.text()).strip():
                            self.correlator_page.lineEdit_2.setText(','.join(map(str, b)))
                except Exception:
                    pass
            except Exception:
                pass

    def page_actions(self):
        cur = self.currentPage()

        # After finishing detector setup page: propagate settings to subsequent pages
        if cur is self.detector_page:
            try:
                settings = self.detector_page.get_settings()
                dets = settings.get('detectors', {}) or {}
                wins = settings.get('windows', {}) or {}
                setup_name = getattr(self.detector_page, 'current_setup_name', None)

                # Update Photon Filter page from detector definition (sanitize and block signals)
                try:
                    self.photon_select.comboBox.clear()
                except Exception:
                    pass
                # Sanitize empty names
                dets = {k: v for k, v in dets.items() if isinstance(k, str) and k.strip()}
                wins = {k: v for k, v in wins.items() if isinstance(k, str) and k.strip()}
                cb2 = getattr(self.photon_select, 'comboBox_2', None)
                cb3 = getattr(self.photon_select, 'comboBox_3', None)
                prev2 = cb2.blockSignals(True) if cb2 is not None else None
                prev3 = cb3.blockSignals(True) if cb3 is not None else None
                try:
                    if cb2 is not None:
                        cb2.clear()
                    if cb3 is not None:
                        cb3.clear()
                    self.photon_select.fill_detectors(dets)
                    self.photon_select.fill_pie_windows(wins)
                    if cb2 is not None and cb2.count() > 0:
                        cb2.setCurrentIndex(0)
                    if cb3 is not None and cb3.count() > 0:
                        cb3.setCurrentIndex(0)
                except Exception:
                    pass
                finally:
                    try:
                        if cb2 is not None:
                            cb2.blockSignals(prev2 if isinstance(prev2, bool) else False)
                        if cb3 is not None:
                            cb3.blockSignals(prev3 if isinstance(prev3, bool) else False)
                    except Exception:
                        pass
                try:
                    self.photon_select.update_detectors()
                except Exception:
                    pass
                try:
                    self.photon_select.update_pie_windows()
                except Exception:
                    pass

                # Select setup in photon filter combo if available
                if setup_name:
                    try:
                        idx = self.photon_select.comboBox.findText(setup_name)
                        if idx >= 0:
                            self.photon_select.comboBox.setCurrentIndex(idx)
                    except Exception:
                        pass

                # Prefill correlator channels and microtime ranges (legacy fallback)
                try:
                    det_names = list(dets.keys())
                    if det_names:
                        a = dets[det_names[0]].get('chs', [])
                        b = dets[det_names[1]].get('chs', a) if len(det_names) > 1 else a
                        self.correlator_page.lineEdit.setText(','.join(map(str, a)))
                        self.correlator_page.lineEdit_2.setText(','.join(map(str, b)))
                except Exception:
                    pass
                try:
                    if isinstance(wins, dict) and wins:
                        ranges = []
                        for v in wins.values():
                            if isinstance(v, (list, tuple)):
                                if len(v) >= 2 and all(isinstance(x, int) for x in v[:2]):
                                    ranges.append(f"{v[0]}-{v[1]}")
                                else:
                                    for i in v:
                                        if isinstance(i, (list, tuple)) and len(i) >= 2:
                                            ranges.append(f"{i[0]}-{i[1]}")
                        s = ";".join(ranges)
                        self.correlator_page.lineEdit_6.setText(s)
                        self.correlator_page.lineEdit_7.setText(s)
                except Exception:
                    pass

                # Populate correlator combos from DetectorWizardPage
                try:
                    if hasattr(self.correlator_page, 'inner') and hasattr(self.correlator_page.inner, 'apply_detector_setup_from_page'):
                        self.correlator_page.inner.apply_detector_setup_from_page(self.detector_page)
                except Exception:
                    pass
            except Exception:
                pass

        # After files/steps page: load files accordingly
        elif cur is self.file_page:
            files = self.file_page.checked_files

            # Expand directories to allowed TTTR files (container names)
            allowed_extensions = {
                f".{ext.lower()}" if not ext.startswith('.') else ext.lower()
                for ext in tttrlib.TTTR.get_supported_container_names()
            }
            expanded_files: List[str] = []
            for p_str in files:
                p = pathlib.Path(p_str).resolve()
                if p.is_dir():
                    for child in p.iterdir():
                        if child.is_file() and child.suffix.lower() in allowed_extensions:
                            expanded_files.append(str(child.resolve()))
                else:
                    expanded_files.append(str(p))

            if self.file_page.cb_photon_filter.isChecked():
                # Preload dropped files into photon filter page using detector page's filetype
                filetype = self.detector_page.filetype

                # Load TTTR objects similar to PhotonFilter's after_file_drop, with extension-aware type resolution
                self.photon_select.tttr_objects = dict()
                def _open_tttr_for_ui(p: pathlib.Path, global_type):
                    p_str = str(p)
                    ext = p.suffix.lower()
                    try:
                        if ext == '.spc':
                            try:
                                ft_int = tttrlib.inferTTTRFileType(p_str)
                                if ft_int is not None and ft_int >= 0:
                                    return tttrlib.TTTR(p_str, ft_int)
                            except Exception:
                                pass
                            try:
                                return tttrlib.TTTR(p_str, 'SPC')
                            except Exception:
                                return tttrlib.TTTR(p_str)
                        if isinstance(global_type, str) and global_type.strip():
                            try:
                                return tttrlib.TTTR(p_str, global_type)
                            except Exception:
                                pass
                        try:
                            ft_int = tttrlib.inferTTTRFileType(p_str)
                            if ft_int is not None and ft_int >= 0:
                                return tttrlib.TTTR(p_str, ft_int)
                        except Exception:
                            pass
                        return tttrlib.TTTR(p_str)
                    except Exception as e:
                        raise e

                for fn in expanded_files:
                    p = pathlib.Path(fn).resolve()
                    p_str = str(p)
                    if p_str in self.photon_select.tttr_objects:
                        continue
                    if not p.exists() or not p.is_file():
                        continue
                    try:
                        self.photon_select.tttr_objects[p_str] = _open_tttr_for_ui(p, filetype)
                    except Exception:
                        QtWidgets.QMessageBox.critical(
                            self,
                            "Error Loading File",
                            f"Failed to load file '{p.name}' with the selected setup. Please check the file type."
                        )
                        self.photon_select.onClearFiles()
                        return

                self.photon_select.settings['tttr_filenames'] = list(expanded_files)
                n_files = len(expanded_files)
                self.photon_select.spinBox_4.setMaximum(n_files - 1 if n_files > 0 else 0)
                if n_files > 0:
                    self.photon_select.spinBox_4.setValue(n_files - 1)
                    self.photon_select.lineEdit.setText(expanded_files[0])
                    try:
                        self.photon_select.read_tttr()
                    except Exception:
                        pass
                # Disable photon filter's own file list controls while used in the wizard
                try:
                    # Prevent editing or dropping new files into the line edit
                    self.photon_select.lineEdit.setEnabled(False)
                    self.photon_select.lineEdit.setDragEnabled(False)
                    self.photon_select.lineEdit.setAcceptDrops(False)
                except Exception:
                    pass
                try:
                    # Disable the Clear Files button to keep the list in sync with the wizard
                    self.photon_select.toolButton_6.setEnabled(False)
                except Exception:
                    pass
            else:
                # Directly prepare correlator
                filetype = self.detector_page.filetype
                if expanded_files:
                    parent = pathlib.Path(expanded_files[0]).resolve().parent
                    self.correlator_page.lineEdit_3.setText(parent.as_posix())
                self.correlator_page.load_tttr_files(expanded_files, filetype)

        elif cur is self.correlator_page:
            if self.file_page.cb_photon_filter.isChecked():
                # Try to save selections (if any) and open the SL5 analysis folder
                try:
                    self.photon_select.save_selection()
                except Exception:
                    pass

                # Derive output parent folder from photon filter or from the TTTR files
                output_parent = None
                try:
                    p = self.photon_select.parent_directories[0]
                    output_parent = p
                except Exception:
                    output_parent = None

                # Build expanded file list from the Files page (checked items only)
                files = self.file_page.checked_files
                allowed_extensions = {
                    f".{ext.lower()}" if not ext.startswith('.') else ext.lower()
                    for ext in tttrlib.TTTR.get_supported_container_names()
                }
                expanded_files: List[str] = []
                for p_str in files:
                    pth = pathlib.Path(p_str).resolve()
                    if pth.is_dir():
                        for child in pth.iterdir():
                            if child.is_file() and (child.suffix.lower() in allowed_extensions or child.suffix.lower() == '.bst'):
                                expanded_files.append(str(child.resolve()))
                    else:
                        expanded_files.append(str(pth))

                if output_parent is None and expanded_files:
                    output_parent = pathlib.Path(expanded_files[0]).resolve().parent

                if output_parent is not None:
                    self.correlator_page.lineEdit_3.setText(output_parent.as_posix())
                    self.correlator_page.open_analysis_folder()

                # Fallback: if no SL5 selections were loaded, correlate all TTTR files without filtering
                try:
                    tttr_missing = (self.correlator_page.inner.tttr is None)
                except Exception:
                    tttr_missing = True
                if tttr_missing and expanded_files:
                    filetype = self.detector_page.filetype
                    self.correlator_page.load_tttr_files(expanded_files, filetype)

        elif cur is self.fcs_merger:
            # Prepare correlations for merger
            if not self.correlator_page.is_correlated:
                # Control writing of chunk files based on Photon filter usage
                try:
                    self.correlator_page.inner.save_chunks_to_disk = self.file_page.cb_photon_filter.isChecked()
                except Exception:
                    pass
                self.correlator_page.correlate_data()
            correlation_folder = self.correlator_page.analysis_folder / self.correlator_page.output_path
            self.fcs_merger.lineEdit.setText(correlation_folder.as_posix())
            if self.file_page.cb_photon_filter.isChecked():
                # File-based workflow (reads chnk-*.json.gz)
                self.fcs_merger.open_correlation_folder()
            else:
                # In-memory workflow (no chunk files written)
                try:
                    self.fcs_merger.set_correlations(self.correlator_page.inner.correlations, correlation_folder)
                except Exception:
                    # Fallback: try reading from disk if available
                    self.fcs_merger.open_correlation_folder()

    def onFinish(self):
        print("Correlation Wizard Finished")
        if self.file_page.cb_fcs_merger.isChecked():
            print("saving merged correlation")
            self.fcs_merger.save_mean_correlation()
            self.fcs_merger.add_to_chisurf()
        else:
            # Requirement: when FCS merger is disabled, save all individual (per-chunk) FCS curves
            try:
                # Ensure chunk files will be written
                self.correlator_page.inner.save_chunks_to_disk = True
            except Exception:
                pass
            # If not yet correlated, run correlation (this will save per-chunk curves)
            if not self.correlator_page.is_correlated:
                self.correlator_page.correlate_data()
            else:
                # Already correlated in-memory: write chunk files now
                try:
                    # Ensure default analysis folder is set (in case it's empty)
                    if hasattr(self.correlator_page.inner, 'ensure_analysis_folder_default'):
                        self.correlator_page.inner.ensure_analysis_folder_default()
                except Exception:
                    pass
                try:
                    self.correlator_page.inner.save_correlations()
                except Exception:
                    pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set window size
        self.resize(800, 600)

        self.setWizardStyle(QtWidgets.QWizard.ModernStyle)

        # Pages
        self.detector_page = chisurf.gui.widgets.wizard.DetectorWizardPage(parent=self)
        self.detector_page_id = self.addPage(self.detector_page)

        self.file_page = FileAndStepsPage(self)
        self.file_page_id = self.addPage(self.file_page)

        self.photon_select = chisurf.gui.widgets.wizard.WizardTTTRPhotonFilter(
            windows={},
            detectors={},
            show_dT=True,
            show_burst=False,
            show_mcs=True,
            show_decay=False,
            show_filter=False
        )
        self.photon_select.toolButton_5.clicked.connect(self.photon_select.completeChanged.emit)
        self.filter_page_id = self.addPage(self.photon_select)

        self.correlator_page = CorrelatorPage()
        self.correlator_page_id = self.addPage(self.correlator_page)

        self.fcs_merger = chisurf.gui.widgets.wizard.WizardFcsMerger()
        self.fcs_merger_page_id = self.addPage(self.fcs_merger)

        # React to changes of detector/setup selection by clearing files for safety
        try:
            self.detector_page.setup_combo.currentIndexChanged.connect(self._on_detector_setup_changed)
        except Exception:
            pass

        # Navigation hooks
        self.currentIdChanged.connect(self._on_current_id_changed)
        self.button(QtWidgets.QWizard.NextButton).clicked.connect(self.page_actions)
        self.button(QtWidgets.QWizard.FinishButton).clicked.connect(self.onFinish)


if __name__ == "plugin":
    wizard = ChisurfFCSWizard()
    wizard.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wizard = ChisurfFCSWizard()
    wizard.show()
    sys.exit(app.exec_())

import pathlib
from typing import List, Optional
import typing

from chisurf.gui import QtWidgets, QtGui, QtCore

import chisurf as cs
import chisurf.gui
import chisurf.gui.widgets.wizard


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
        self.inner = cs.gui.widgets.wizard.WizardTTTRCorrelator()
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


if __name__ == "plugin":
    from chisurf.plugins.fcs.fcs_correlator.tool import FcsCorrelatorTool
    tool = FcsCorrelatorTool()
    tool.show()

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    from chisurf.plugins.fcs.fcs_correlator.tool import FcsCorrelatorTool
    tool = FcsCorrelatorTool()
    tool.show()
    sys.exit(app.exec_())

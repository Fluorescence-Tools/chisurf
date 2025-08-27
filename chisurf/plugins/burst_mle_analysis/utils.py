import collections.abc
import json
import typing
from pathlib import Path
from typing import Dict, Callable, Iterator

import numpy as np
import tttrlib
from qtpy import QtWidgets, QtCore, QtGui


class LazyTTTRDict(collections.abc.MutableMapping):
    """
    A dict-like that maps a key (file‐stem) → TTTR object,
    but only calls tttrlib.TTTR(path, file_type) on first access.
    """
    def __init__(
        self,
        path_map: Dict[str, Path],
        file_type_getter: Callable[[], str]
    ):
        """
        Parameters
        ----------
        path_map : Dict[str, Path]
            Maps file‐stem (no extension) → full Path to .ptu/.ht3/etc.
        file_type_getter : () → str
            A zero‐argument callable returning current TTTR file‐type (e.g. self.tttr_file_type).
        """
        self._paths = path_map
        self._cache: Dict[str, tttrlib.TTTR] = {}
        self._file_type_getter = file_type_getter
        self._warning_shown = False

    def __getitem__(self, key: str) -> tttrlib.TTTR:
        if key not in self._paths:
            raise KeyError(f"No TTTR path for key {key!r}")
        if key not in self._cache:
            path = self._paths[key]
            # Check if _file_type_getter is None
            if self._file_type_getter is None:
                if not self._warning_shown:
                    # Show a warning if a QApplication exists; otherwise, print to console.
                    if QtWidgets.QApplication.instance() is not None:
                        QtWidgets.QMessageBox.warning(
                            None,
                            "Warning",
                            "The file type getter is None. This may cause issues with TTTR file loading."
                        )
                    else:
                        print("Warning: The file type getter is None. This may cause issues with TTTR file loading.")
                    self._warning_shown = True
                # Use a default file type or try to infer it
                file_type = tttrlib.inferTTTRFileType(str(path))
            else:
                file_type = self._file_type_getter()
                # Fallback to inference if getter returned None/Auto/empty
                if not file_type or (isinstance(file_type, str) and file_type.lower() == "auto"):
                    file_type = tttrlib.inferTTTRFileType(str(path))
            # instantiate on first use
            self._cache[key] = tttrlib.TTTR(str(path), file_type)
        return self._cache[key]

    def __setitem__(self, key: str, value: tttrlib.TTTR):
        # allow manual override if you really want
        self._cache[key] = value

    def __delitem__(self, key: str):
        self._paths.pop(key, None)
        self._cache.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def add_path(self, key: str, path: Path):
        """
        Register a new TTTR file to be loaded on demand.
        """
        self._paths[key] = path

    def clear(self):
        self._paths.clear()
        self._cache.clear()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Convert to list (or you could serialize differently)
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return super().default(obj)


class FileListWidget(QtWidgets.QListWidget):
    """
    A QListWidget subclass that accepts file drops and maintains a list of file paths.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    file_added_callback : callable, optional
        Function to call when files are added.
    process_on_drop : bool, optional
        Whether to process files immediately on drop.
    """

    def __init__(self, parent=None, file_added_callback=None, process_on_drop=False):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.file_added_callback = file_added_callback
        self.process_on_drop = process_on_drop
        # Allow the file list to grow vertically and fill available space
        sp = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(sp)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """
        Handle drag enter events to accept file URLs.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        """
        Handle drag move events to accept file URLs.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        """
        Handle drop events, extract file paths, and add them to the list.
        """
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        file_paths: typing.List[str] = []
        for url in event.mimeData().urls():
            local = Path(url.toLocalFile())
            if local.is_file():
                file_paths.append(str(local))
            elif local.is_dir():
                bursts = list(local.glob('**/*.bur'))
                if bursts:
                    file_paths.extend(str(f) for f in bursts)
                else:
                    for ext in tttrlib.get_supported_filetypes():
                        file_paths.extend(str(f) for f in local.glob(f'**/*{ext}'))
        file_paths.sort()
        self.blockSignals(True)
        for fp in file_paths:
            self.add_file(fp)
        self.blockSignals(False)
        if self.file_added_callback:
            self.file_added_callback()
        event.acceptProposedAction()

    def add_file(self, file_path: str):
        """
        Add a file path to the list as a checkable item.

        Parameters
        ----------
        file_path : str
            Path of the file to add.
        """
        item = QtWidgets.QListWidgetItem(file_path, self)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        self.addItem(item)

    def get_selected_files(self) -> typing.List[Path]:
        """
        Get the list of currently selected (checked) files.

        Returns
        -------
        List[Path]
            Paths of selected files.
        """
        return [Path(self.item(i).text()) for i in range(self.count())
                if self.item(i).checkState() == QtCore.Qt.Checked]

from __future__ import annotations
from chisurf import typing

import fnmatch
import numbers
import os
import pathlib
import re
import time

from chisurf.gui import QtGui, QtWidgets, QtCore
from io import BytesIO

import chisurf.core.fio
import chisurf.core.settings
import chisurf.core.curve
import chisurf.core.base
import chisurf as cs
def get_widgets_in_layout(
        layout: QtWidgets.QLayout
):
    """Returns a list of all widgets within a layout
    """
    return (layout.itemAt(i) for i in range(layout.count()))


def clear_layout(layout: QtWidgets.QLayout):
    """Clears all widgets within a layout
    """
    while layout.count():
        child = layout.takeAt(0)
        if child.widget() is not None:
            child.widget().deleteLater()
        elif child.layout() is not None:
            clear_layout(child.layout())


def hide_items_in_layout(
        layout: QtWidgets.QLayout
):
    """Hides all items within a Qt-layout
    """
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if type(item) == QtWidgets.QWidgetItem:
            item.widget().hide()


class MyMessageBox(QtWidgets.QMessageBox):

    def __init__(
            self,
            label: str = None,
            info: str = "",
            details: str = None,
            show_fortune: bool = cs.core.settings.cs_settings['fortune']
    ):
        super().__init__()
        self.setSizeGripEnabled(True)
        self.setIcon(QtWidgets.QMessageBox.Information)

        # Set title and center the popup
        if label is not None:
            self.setWindowTitle(label)

        # Use HTML for nicer formatting of info
        formatted_info = f"<b>{info}</b>" if info else ""

        # Add fortune message (if enabled) with a better look
        if show_fortune:
            try:
                fortune = cs.gui.widgets.fortune.get_fortune()
                if fortune:  # Only add fortune if it's not empty
                    fortune_html = f"<br><i>{fortune}</i><br><br>"  # Italicized fortune text, with spacing
                    self.setInformativeText(formatted_info + fortune_html)
                else:
                    self.setInformativeText(formatted_info)
            except Exception:
                # If there's any error getting the fortune, just show the info
                self.setInformativeText(formatted_info)
        else:
            self.setInformativeText(formatted_info)

        # Set detailed text (e.g., error trace) if provided
        if details is not None:
            self.setDetailedText(details)

        # Center the window
        self.center()

        # Adjust the look and feel
        self.setStyleSheet("""
            QMessageBox {
                font-size: 14px;
            }
            QTextEdit {
                font-family: "Courier New", Courier, monospace;
                font-size: 12px;
            }
        """)

        # Show the popup window
        self.exec_()

    def event(self, e) -> bool:
        result = super().event(e)

        # Optimize the size and policy for the text edit (error trace) field
        text_edit = self.findChild(QtWidgets.QTextEdit)
        if text_edit is not None:
            text_edit.setMinimumHeight(100)
            text_edit.setMaximumHeight(500)
            text_edit.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )

        return result

    def center(self):
        # Center the popup on the current screen
        screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor().pos())
        fg = self.frameGeometry()
        fg.moveCenter(screen.geometry().center())
        self.move(fg.topLeft())


class FileList(QtWidgets.QListWidget):

    @property
    def filenames(self) -> typing.List[str]:
        fn = list()
        for row in range(self.count()):
            item = self.item(row)
            fn.append(str(item.text()))
        return fn

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                s = str(url.toLocalFile())
                url.setScheme("")
                if s.endswith(self.filename_ending) or \
                        s.endswith(self.filename_ending+'.gz') or \
                        s.endswith(self.filename_ending+'.bz2') or \
                        s.endswith(self.filename_ending+'.zip'):
                    self.addItem(s)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def __init__(
            self,
            accept_drops: bool = True,
            filename_ending: str = "*",
            icon: QtGui.QIcon = None
    ):
        """
        :param accept_drops: if True accepts files that are dropped into the list
        :param kwargs:
        """
        super().__init__()
        self.filename_ending = filename_ending
        self.drag_item = None
        self.drag_row = None

        if accept_drops:
            self.setAcceptDrops(True)
            self.setDragDropMode(
                QtWidgets.QAbstractItemView.InternalMove
            )

        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/list-add.png")

        self.setWindowIcon(icon)


def table_font() -> QtGui.QFont:
    """Return the globally configured table font."""
    gui_settings = cs.core.settings.gui
    table_settings = gui_settings.get("table", {})
    font_family = table_settings.get("font_family")
    font_size = int(table_settings.get("font_size", 10))
    font = QtGui.QFont(font_family) if font_family else QtGui.QFont()
    font.setPointSize(max(1, font_size))
    font.setBold(bool(table_settings.get("font_bold", False)))
    return font


def table_row_height() -> int:
    """Return the globally configured compact table row height."""
    row_height = cs.core.settings.gui.get("table", {}).get("row_height", 18)
    return max(12, int(row_height))


def apply_compact_table_style(table) -> None:
    """Apply compact styling to a table-like widget."""
    table.setFont(table_font())
    table.setAlternatingRowColors(True)
    table.setSortingEnabled(False)

    if hasattr(table, "verticalHeader"):
        table.verticalHeader().setDefaultSectionSize(table_row_height())
        table.verticalHeader().setVisible(False)

    if hasattr(table, "horizontalHeader"):
        header = table.horizontalHeader()
        header.setDefaultSectionSize(table_header_height())
        header.setStretchLastSection(True)
        header.setSectionsClickable(False)
        header.setHighlightSections(False)


def table_header_height() -> int:
    """Return the globally configured compact table header height."""
    header_height = cs.core.settings.gui.get("table", {}).get("header_height", table_row_height())
    return max(12, int(header_height))


class LogListWidget(QtWidgets.QTableWidget):
    """Multi-column log table with basic filtering and clipboard support."""

    _LOG_PATTERN = re.compile(
        r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})?)\s+-\s+"
        r"(?P<level>[A-Z]+)\s+-\s+(?P<message>.*)$"
    )

    def __init__(self, parent=None):
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(["Time", "Level", "Origin", "Message"])
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        apply_compact_table_style(self)
        self.horizontalHeader().setStretchLastSection(False)
        self.setColumnWidth(0, 65)
        self.setColumnWidth(1, 50)
        self.setColumnWidth(2, 130)
        self.setColumnWidth(3, 220)

    def addItem(self, text, record=None):  # type: ignore[override]
        """Add one log row to the table."""
        full_time, time_text, level, source, message = self._parse_log_entry(str(text), record)
        row = self.rowCount()
        self.insertRow(row)
        self._set_item(row, 0, time_text, full_time)
        self.item(row, 0).setData(QtCore.Qt.UserRole, full_time)
        self._set_item(row, 1, level)
        self._set_item(row, 2, source, source)
        self._set_item(row, 3, message, message)
        self._style_level(row, level)
        self.scrollToBottom()

    def count(self):  # type: ignore[override]
        """Return the number of log rows."""
        return self.rowCount()

    def item(self, row, column=3):  # type: ignore[override]
        """Return the message item for a row by default."""
        return super().item(row, column)

    def row_text(self, row):
        """Return the full text of a log row."""
        parts = []
        for column in range(self.columnCount()):
            item = self.item(row, column)
            if item is None:
                continue
            parts.append(str(item.data(QtCore.Qt.UserRole) or item.text()))
        return " | ".join(parts)

    def keyPressEvent(self, event):  # type: ignore[override]
        """Copy selected log rows with Ctrl+C."""
        if event.key() == QtCore.Qt.Key_C and event.modifiers() & QtCore.Qt.ControlModifier:
            self.copy_selected_items()
        else:
            super().keyPressEvent(event)

    def copy_selected_items(self):
        """Copy selected log rows to the clipboard as tab-separated text."""
        selected_items = self.selectedItems()
        if not selected_items:
            return

        rows = {}
        for item in selected_items:
            rows.setdefault(item.row(), {})[item.column()] = item.text()

        lines = []
        for row_index in sorted(rows):
            values = []
            for column in range(self.columnCount()):
                item = self.item(row_index, column)
                if column in {0, 2, 3} and item is not None:
                    values.append(str(item.data(QtCore.Qt.UserRole) or item.text()))
                elif item is not None:
                    values.append(item.text())
                else:
                    values.append("")
            lines.append("\t".join(values))

        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        cs.logging.info(f"Copied {len(rows)} log entries to clipboard")

    def reset_row_styles(self, row):
        """Reset foreground, background, and font styles for a log row."""
        for column in range(self.columnCount()):
            item = self.item(row, column)
            if item is None:
                continue
            item.setBackground(QtGui.QBrush())
            item.setFont(table_font())

    def _set_item(self, row, column, text, tooltip=None):
        visible_text = self._visible_text(str(text), column)
        item = QtWidgets.QTableWidgetItem(visible_text)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        if tooltip is not None:
            item.setToolTip(str(tooltip))
            item.setData(QtCore.Qt.UserRole, str(tooltip))
        self.setItem(row, column, item)

    def _visible_text(self, text: str, column: int) -> str:
        if column not in {2, 3}:
            return text
        max_len = 34 if column == 2 else 90
        if len(text) <= max_len:
            return text
        return f"{text[: max_len - 1]}…"

    def _style_level(self, row, level):
        color_map = {
            "DEBUG": QtGui.QColor(90, 120, 160),
            "INFO": QtGui.QColor(30, 130, 30),
            "WARNING": QtGui.QColor(190, 130, 0),
            "ERROR": QtGui.QColor(200, 50, 50),
            "CRITICAL": QtGui.QColor(150, 0, 150),
        }
        item = self.item(row, 1)
        if item is not None:
            color = color_map.get(level.upper(), QtGui.QColor(0, 0, 0))
            item.setForeground(QtGui.QBrush(color))

    def _parse_log_entry(self, text, record=None):
        full_time = ""
        level = ""
        source = ""
        message = text

        if record is not None:
            full_time = getattr(record, "asctime", "")
            level = getattr(record, "levelname", "")
            source = getattr(record, "pathname", "") or getattr(record, "module", "") or getattr(record, "name", "")

        match = self._LOG_PATTERN.match(text)
        if match:
            full_time = match.group("time")
            level = match.group("level")
            message = match.group("message")

        return full_time, self._display_time(full_time), level, source, message

    @staticmethod
    def _display_time(full_time: str) -> str:
        if not full_time:
            return ""
        for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return time.strftime("%H:%M:%S", time.strptime(full_time, fmt))
            except ValueError:
                continue
        return full_time



class EnterAwarePlainTextEdit(QtWidgets.QPlainTextEdit):
    """
    A QPlainTextEdit that emits a signal when Enter is pressed (without Shift).
    Also supports history navigation with Up/Down arrow keys.
    """
    sendRequested = QtCore.Signal()
    historyPrevRequested = QtCore.Signal()
    historyNextRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.sendRequested.emit()
            return

        if event.key() == QtCore.Qt.Key_Up:
            self.historyPrevRequested.emit()
            return

        if event.key() == QtCore.Qt.Key_Down:
            self.historyNextRequested.emit()
            return

        super().keyPressEvent(event)


def get_filename(
        description: str = '',
        file_type: str = 'All files (*.*)',
        working_path: pathlib.Path = None
) -> pathlib.Path:
    """Open a file within a working path. If no path is specified the last
    path is used. After using this function the current working path of the
    running program (ChiSurf) is updated according to the folder of the opened
    file.

    :param working_path:
    :param description:
    :param file_type:
    :return:
    """
    if working_path is None:
        if cs.working_path is None:
            cs.working_path = pathlib.Path.home()
        working_path = cs.working_path
    filename_str, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        description,
        str(working_path.absolute()),
        file_type
    )
    filename = pathlib.Path(filename_str)
    try:
        if filename_str:
            cs.working_path = filename.parent
    except Exception:
        pass
    return filename


def open_files(
        description: str = '',
        file_type: str = 'All files (*.*)',
        working_path: pathlib.Path = None
):
    """Open a file within a working path. If no path is specified the last
    path is used. After using this function the current working path of the
    running program (ChiSurf) is updated according to the folder of the opened
    file.

    :param working_path: Base path to open the dialog in. If None, uses the current working path.
    :param description: Dialog title or description.
    :param file_type: File filter to display.
    :return: List of selected filenames, or empty list if cancel is clicked or an error occurs.
    """
    if working_path is None:
        working_path = cs.working_path
    filenames = QtWidgets.QFileDialog.getOpenFileNames(
        None,
        description,
        str(working_path.absolute()),
        file_type
    )[0]
    try:
        # Only update the working path if at least one file was selected
        if filenames:
            # Use parent() to get the directory containing the file
            cs.working_path = pathlib.Path(filenames[0]).parent
    except Exception as e:
        # Log the error but don't show it to the user
        import logging
        logging.error(f"Error in open_files: {e}")

    return filenames

def save_file(
        description: str = '',
        file_type: str = 'All files (*.*)',
        working_path: pathlib.Path = None
) -> str:
    """Opens a file save dialog. If cancel is clicked, returns None.

    Updates the current working path of the program (ChiSurf) based on the folder
    of the saved file.

    :param description: Dialog title or description.
    :param file_type: File filter to display.
    :param working_path: Base path to open the dialog in.
    :return: The selected filename, or None if cancel is clicked.
    """
    if isinstance(working_path, str):
        working_path = cs.working_path / working_path
    if working_path is None:
        working_path = cs.working_path

    filename, _ = QtWidgets.QFileDialog.getSaveFileName(
        None,
        caption=description,
        directory=str(working_path.absolute()),
        filter=file_type
    )

    # If cancel is clicked, filename will be an empty string.
    if not filename:
        return None

    # Update the working path to the directory containing the saved file.
    cs.working_path = pathlib.Path(filename).parent
    return filename


def get_directory(
        filename_ending: str = None,
        get_files: bool = False,
        directory: pathlib.Path = None,
        caption: str = None
) -> typing.Tuple[pathlib.Path, typing.List[str]]:
    """Opens a new window where you can choose a directory. The current
    working path is updated to this directory.

    It either returns the directory or the files within the directory (if
    get_files is True). The returned files can be filtered for the filename
    ending using the kwarg filename_ending.

    :return: directory str
    """
    fn_ending = filename_ending
    if directory is None:
        directory = cs.working_path
    caption_text = caption or "Select Directory"
    if isinstance(directory, pathlib.Path):
        directory_str = QtWidgets.QFileDialog.getExistingDirectory(None, caption_text, str(directory.absolute()))
    else:
        directory_str = QtWidgets.QFileDialog.getExistingDirectory(None, caption_text)
    # If cancel is clicked, return None and do not update working path
    if not directory_str:
        return None, []
    directory = pathlib.Path(directory_str)
    cs.working_path = directory
    if not get_files:
        return directory, []
    else:
        filenames = [str(directory / s) for s in os.listdir(directory)]
        if fn_ending is not None:
            filenames = fnmatch.filter(filenames, fn_ending)
        return directory, filenames


def make_widget_from_yaml(
        variable_dictionary,
        name: str = ''
):
    """
    >>> import numbers
    >>> import pyqtgraph as pg
    >>> import collections
    >>> import yaml
    >>> d = yaml.safe_load(open("./test_session.yaml"))['datasets']
    >>> od =dict(sorted(d.items()))
    >>> w = make_widget_from_yaml(od, 'test')
    >>> w.show()
    :param variable_dictionary: 
    :param name: 
    :return: 
    """
    
    import pyqtgraph as pg

    def make_group(
            d,
            name: str = ''
    ):
        g = QtWidgets.QGroupBox()
        g.setTitle(str(name))
        layout = QtWidgets.QFormLayout()
        g.setLayout(layout)

        for row, key in enumerate(d):
            label = QtWidgets.QLabel(str(key))
            value = d[key]
            if isinstance(value, dict):
                wd = make_group(value, '')
                layout.addRow(str(key), wd)
            else:
                if isinstance(value, bool):
                    wd = QtWidgets.QCheckBox()
                    wd.setChecked(value)
                elif isinstance(value, numbers.Real):
                    wd = pg.SpinBox(value=value)
                else:
                    wd = QtWidgets.QLineEdit()
                    wd.setText(str(value))
                layout.addRow(label, wd)
        return g

    return make_group(variable_dictionary, name)


def tex2svg(
        formula: str,
        fontsize: int = 12,
        dpi: int = 300
):
    """Render TeX formula to SVG.
    Args:
        formula (str): TeX formula.
        fontsize (int, optional): Font size.
        dpi (int, optional): DPI.
    Returns:
        str: SVG render.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, r'${}$'.format(formula), fontsize=fontsize)

    output = BytesIO()
    fig.savefig(
        output,
        dpi=dpi,
        transparent=True,
        format='svg',
        bbox_inches='tight',
        pad_inches=0.0,
        frameon=False
    )
    plt.close(fig)

    output.seek(0)
    return output.read()


def get_subtree_nodes(tree_widget_item):
    """Returns all QTreeWidgetItems in the subtree rooted at the given node."""
    nodes = []
    nodes.append(tree_widget_item)
    for i in range(tree_widget_item.childCount()):
        nodes.extend(get_subtree_nodes(tree_widget_item.child(i)))
    return nodes


def get_all_items(tree_widget):
    """Returns all QTreeWidgetItems in the given QTreeWidget."""
    all_items = []
    for i in range(tree_widget.topLevelItemCount()):
        top_item = tree_widget.topLevelItem(i)
        all_items.extend(get_subtree_nodes(top_item))
    return all_items


class Controller(QtWidgets.QWidget, cs.core.base.Base):
    """
    Used by FittingControllerWidget
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()

    def to_dict(
            self,
            remove_protected: bool = False,
            copy_values: bool = True,
            convert_values_to_elementary: bool = False
    ):
        d = super().to_dict(
            remove_protected=remove_protected,
            copy_values=copy_values,
            convert_values_to_elementary=convert_values_to_elementary
        )
        d.update(
            {
                'type': 'controller',
                'class': self.__class__.__name__
            }
        )
        return d


class View(
    QtWidgets.QWidget,
    cs.core.base.Base
):
    """
    Used by Plot
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()

    def update(self, *args, **kwargs) -> None:
        super().update()

    def to_dict(
            self,
            remove_protected: bool = False,
            copy_values: bool = True,
            convert_values_to_elementary: bool = False
    ):
        d = super().to_dict(
            remove_protected=remove_protected,
            copy_values=copy_values,
            convert_values_to_elementary=convert_values_to_elementary
        )
        d.update(
            {
                'type': 'view',
                'class': self.__class__.__name__
            }
        )
        return d

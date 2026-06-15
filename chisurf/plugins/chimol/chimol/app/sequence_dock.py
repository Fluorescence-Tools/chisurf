from __future__ import annotations

from typing import Tuple

from qtpy import QtWidgets, QtCore, QtGui

from ..config import _DISPLAY_CONFIG


class SequenceDock(QtCore.QObject):
    """Sequence dock with label, numbers row and residue list.

    This helper only builds the widgets; all logic and signal wiring is
    managed by the owning main window.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        *,
        margins: Tuple[int, int, int, int],
        spacing: int,
    ) -> None:
        super().__init__(parent)

        seq_cfg = _DISPLAY_CONFIG.get("sequence", {})
        font_family = str(seq_cfg.get("font_family", "Courier New"))
        try:
            font_size = int(seq_cfg.get("font_size", 9))
        except Exception:
            font_size = 9
        try:
            number_height = int(seq_cfg.get("number_height", 16))
        except Exception:
            number_height = 16
        try:
            residue_height = int(seq_cfg.get("residue_height", 20))
        except Exception:
            residue_height = 20

        font_bold = bool(seq_cfg.get("font_bold", True))
        number_font_bold = bool(seq_cfg.get("number_font_bold", True))

        # Number row font (stored for caller convenience)
        number_font = QtGui.QFont(font_family)
        number_font.setPointSize(font_size)
        number_font.setBold(number_font_bold)
        self.sequence_number_font = number_font
        self.sequence_number_bold_font = QtGui.QFont(number_font)
        self.sequence_number_bold_font.setBold(True)

        # Top label for active molecule / visibility toggle
        self.seq_label = QtWidgets.QToolButton(parent)
        self.seq_label.setText("No molecule")
        self.seq_label.setCheckable(True)
        self.seq_label.setChecked(True)
        self.seq_label.setMinimumWidth(120)

        # Sequence numbers label + list
        self.seq_numbers_label = QtWidgets.QToolButton(parent)
        self.seq_numbers_label.setText("Seq nbr")
        self.seq_numbers_label.setMinimumWidth(120)
        self.seq_numbers_label.setEnabled(False)
        self.seq_numbers_label.setCheckable(False)

        self.seq_numbers_list = QtWidgets.QListWidget(parent)
        self.seq_numbers_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.seq_numbers_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.seq_numbers_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.seq_numbers_list.setFlow(QtWidgets.QListView.LeftToRight)
        self.seq_numbers_list.setWrapping(False)
        self.seq_numbers_list.setUniformItemSizes(True)
        self.seq_numbers_list.setFixedHeight(max(10, number_height))
        self.seq_numbers_list.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.seq_numbers_list.setFont(number_font)
        self.seq_numbers_list.setFocusPolicy(QtCore.Qt.NoFocus)
        self.seq_numbers_list.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel
        )

        # Shared horizontal scrollbar
        self.seq_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal, parent)
        self.seq_scrollbar.setObjectName("ChimolSequenceScrollBar")
        self.seq_scrollbar.setEnabled(False)
        self.seq_scrollbar.setMaximumHeight(18)

        # Main sequence list
        self.seq_list = QtWidgets.QListWidget(parent)
        self.seq_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        font = QtGui.QFont(font_family)
        font.setPointSize(font_size)
        font.setBold(font_bold)
        self.seq_list.setFont(font)
        self.sequence_font = font
        self.seq_list.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.seq_list.setFlow(QtWidgets.QListView.LeftToRight)
        self.seq_list.setWrapping(False)
        self.seq_list.setUniformItemSizes(True)
        self.seq_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.seq_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.seq_list.setFixedHeight(max(12, residue_height))
        self.seq_list.setStyleSheet(
            "QListWidget::item:selected{background: transparent; color: inherit;}"
            "QListWidget::item:selected:!active{background: transparent; color: inherit;}"
        )
        self.seq_list.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)

        # Layout rows
        seq_row = QtWidgets.QWidget(parent)
        seq_row_layout = QtWidgets.QHBoxLayout(seq_row)
        seq_row_layout.setContentsMargins(0, 0, 0, 0)
        seq_row_layout.setSpacing(6)
        seq_row_layout.addWidget(self.seq_label, 0, QtCore.Qt.AlignVCenter)
        seq_row_layout.addWidget(self.seq_list, 1)

        sequence_widget = QtWidgets.QWidget(parent)
        sequence_layout = QtWidgets.QVBoxLayout(sequence_widget)
        left_m, top_m, right_m, bottom_m = margins
        sequence_layout.setContentsMargins(left_m, 0, right_m, 0)
        sequence_layout.setSpacing(0)

        numbers_row = QtWidgets.QWidget(parent)
        numbers_row_layout = QtWidgets.QHBoxLayout(numbers_row)
        numbers_row_layout.setContentsMargins(0, 0, 0, 0)
        numbers_row_layout.setSpacing(6)
        numbers_row_layout.addWidget(self.seq_numbers_label, 0, QtCore.Qt.AlignVCenter)
        numbers_row_layout.addWidget(self.seq_numbers_list, 1)

        sequence_layout.addWidget(self.seq_scrollbar)
        sequence_layout.addWidget(numbers_row)
        sequence_layout.addWidget(seq_row)

        self.extra_seq_container = QtWidgets.QWidget(parent)
        self.extra_seq_layout = QtWidgets.QVBoxLayout(self.extra_seq_container)
        self.extra_seq_layout.setContentsMargins(0, 0, 0, 0)
        self.extra_seq_layout.setSpacing(0)
        sequence_layout.addWidget(self.extra_seq_container)

        self._widget = sequence_widget
        sequence_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum,
        )

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._widget

    # ------------------------------------------------------------------
    # Helpers reused by the main window
    # ------------------------------------------------------------------

    @staticmethod
    def sequence_config() -> dict:
        try:
            return _DISPLAY_CONFIG.get("sequence", {}) or {}
        except Exception:
            return {}

    @staticmethod
    def color_from_rgba(values, default):
        try:
            seq = list(values)
        except Exception:
            seq = list(default)
        if len(seq) < 4:
            seq = list(default)
        r, g, b, a = seq[:4]
        try:
            color = QtGui.QColor.fromRgbF(float(r), float(g), float(b), float(a))
        except Exception:
            color = QtGui.QColor.fromRgbF(*default)
        return color

    @staticmethod
    def default_sequence_palette(ss_code: str) -> tuple[QtGui.QColor, QtGui.QColor]:
        code = (ss_code or "C").upper()
        try:
            colors_cfg = _DISPLAY_CONFIG.get("colors", {}) or {}
            ss_cfg = colors_cfg.get("secondary_structure", {}) or {}
        except Exception:
            ss_cfg = {}

        mapping = {"H": "helix", "E": "strand", "C": "coil"}
        key = mapping.get(code, "coil")

        default_map = {
            "helix": [0.3, 0.3, 0.9, 1.0],
            "strand": [0.9, 0.3, 0.3, 1.0],
            "coil": [0.9, 0.9, 0.7, 1.0],
        }
        default_rgba = default_map.get(key, default_map["coil"])
        rgba = ss_cfg.get(key, default_rgba)

        bg = SequenceDock.color_from_rgba(rgba, default_rgba)

        # Choose a readable foreground color based on background luminance.
        try:
            r, g, b, a = list(rgba)[:4]
            lum = 0.299 * float(r) + 0.587 * float(g) + 0.114 * float(b)
        except Exception:
            lum = 0.5
        if lum < 0.5:
            fg = QtGui.QColor(255, 255, 255)
        else:
            fg = QtGui.QColor(0, 0, 0)
        return bg, fg

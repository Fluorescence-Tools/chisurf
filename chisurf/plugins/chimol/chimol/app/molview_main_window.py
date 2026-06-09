"""Protein structure viewer (Chimol) plugin."""

import shutil

import chisurf as cs
from pathlib import Path
from typing import Optional, Any, Sequence

import numpy as np

from qtpy import QtWidgets, QtCore, QtGui

try:  # moview can run inside or outside cs
    import chisurf.core.settings as _cs_settings
    from chisurf.gui.widgets.general import open_files as _cs_open_files
    from chisurf.core.structure import Structure as _ChiSurfStructure
except Exception:  # pragma: no cover - standalone moview
    cs = None  # type: ignore[assignment]
    _cs_settings = None
    _cs_open_files = None
    _ChiSurfStructure = None

from ..config import _DISPLAY_CONFIG
from ..colors import _SEQ_COLOR_ROLE, _OBJECT_ID_ROLE
from ..io import (
    open_structure_files,
    load_structure_payload,
    load_trajectory_frames,
    MdtrajNotAvailableError,
    load_mrc_as_points,
    load_rmf_frames,
    load_rmf_full,
    RmfHierarchyNode,
    RmfNotAvailableError,
)
from ..renderer.view import MolView
from ..analysis import (
    build_residue_alignment,
    assign_ss_c3_from_atoms,
    assign_ss_c3_from_file,
)
from .command_dock import CommandDock
from .timeline_panel import TimelineDock
from .controls_panel import ControlsDock
from .state_control_panel import StateControlDock
from .objects_panel import ObjectsDock
from .sequence_dock import SequenceDock
from .hierarchy_panel import HierarchyDock
from .config_editor import MolViewConfigEditor
from ..cmd import cmd as _cmd


# Plugin name as it appears in the Plugins menu
# The "Structure:" prefix groups it with other structure tools.
name = "Structure:Chimol (protein viewer)"


_SEQ_INDEX_ROLE = QtCore.Qt.UserRole + 150


class MolViewPluginWindow(QtWidgets.QMainWindow):
    """Chimol main window wiring together viewer, docks, and toolbar."""

    def __init__(
        self,
        parent=None,
        *,
        button_overrides: Optional[dict[str, dict[str, Any]]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Chimol - Protein Viewer")

        self._object_store: dict[str, dict[str, Any]] = {}
        self._block_object_list_signals = False
        self._active_object_id: Optional[str] = None
        self._scroll_targets: list[QtWidgets.QScrollBar] = []
        self._scroll_updating = False
        self._scroll_master: Optional[QtWidgets.QScrollBar] = None
        self._content_scrollbar: Optional[QtWidgets.QScrollBar] = None

        layout_cfg = _DISPLAY_CONFIG.get("layout", {})
        margins = layout_cfg.get("root_margins", [4, 4, 4, 4])
        if not isinstance(margins, (list, tuple)) or len(margins) != 4:
            margins = [4, 4, 4, 4]
        try:
            l, t, r, b = (int(m) for m in margins)
        except Exception:
            l, t, r, b = 4, 4, 4, 4
        spacing = layout_cfg.get("root_spacing", 4)
        try:
            spacing = int(spacing)
        except Exception:
            spacing = 4
        dock_margins = (l, t, r, b)

        self.viewer = MolView(self)
        self.setCentralWidget(self.viewer)
        self.controls = ControlsDock(
            self,
            margins=dock_margins,
            spacing=spacing,
            button_overrides=button_overrides,
        )

        # ------------------------------------------------------------------
        # Toolbar (all buttons as tool buttons)
        # ------------------------------------------------------------------
        self.controls_dock = self.controls.dock_widget
        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.controls_dock)

        self.button_open = self.controls.button_open
        self.button_plane = self.controls.button_plane
        self.button_color = self.controls.button_color
        self.button_color_ss = self.controls.button_color_ss
        self.button_color_sequence = self.controls.button_color_sequence
        self.button_rep_cartoon = self.controls.button_rep_cartoon
        self.button_rep_atoms = self.controls.button_rep_atoms
        self.button_rep_sticks = self.controls.button_rep_sticks
        self.button_rep_trace = self.controls.button_rep_trace
        self.button_rep_dots = self.controls.button_rep_dots
        self.button_rep_metaballs = self.controls.button_rep_metaballs
        self.button_surface = self.controls.button_surface
        self.button_info = self.controls.button_info
        self.button_display_cfg = self.controls.button_display_cfg

        # ------------------------------------------------------------------
        # Object list (per-molecule visibility and activation)
        # ------------------------------------------------------------------
        self.objects = ObjectsDock(
            self,
            margins=dock_margins,
            spacing=spacing,
        )
        self.objects_dock = self.objects.dock_widget
        self.object_list = self.objects.object_list
        self.object_list.itemSelectionChanged.connect(self.on_object_selection_changed)
        self.object_list.itemChanged.connect(self.on_object_item_changed)
        self.object_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.object_list.customContextMenuRequested.connect(
            self._on_object_list_context_menu
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.objects_dock)

        self.hierarchy = HierarchyDock(self)
        self.hierarchy_dock = self.hierarchy
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.hierarchy_dock)
        self.tabifyDockWidget(self.objects_dock, self.hierarchy_dock)
        self.objects_dock.raise_()

        try:
            self.resizeDocks(
                [self.objects_dock],
                [220],
                QtCore.Qt.Horizontal,
            )
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Sequence viewer (per-residue selection)
        # ------------------------------------------------------------------
        self.sequence = SequenceDock(
            self,
            margins=dock_margins,
            spacing=spacing,
        )
        self.sequence_dock = self.sequence.dock_widget
        self.seq_label = self.sequence.seq_label
        self.seq_numbers_label = self.sequence.seq_numbers_label
        self.seq_numbers_list = self.sequence.seq_numbers_list
        self.seq_scrollbar = self.sequence.seq_scrollbar
        self.seq_list = self.sequence.seq_list
        self._extra_seq_container = self.sequence.extra_seq_container
        self._extra_seq_layout = self.sequence.extra_seq_layout
        self._sequence_number_font = self.sequence.sequence_number_font
        self._sequence_number_bold_font = self.sequence.sequence_number_bold_font
        self._sequence_font = self.sequence.sequence_font
        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.sequence_dock)

        self.command_panel = CommandDock(
            self,
            margins=dock_margins,
            spacing=spacing,
        )
        self.addDockWidget(
            QtCore.Qt.TopDockWidgetArea, self.command_panel.dock_widget
        )
        
        self.timeline = TimelineDock(
            self,
            self.viewer,
            _cmd,
            margins=dock_margins,
            spacing=spacing,
        )
        self.addDockWidget(
            QtCore.Qt.BottomDockWidgetArea, self.timeline.dock_widget
        )

        self.state_control = StateControlDock(
            self,
            self.viewer,
            _cmd,
            margins=dock_margins,
            spacing=spacing,
        )
        self.addDockWidget(
            QtCore.Qt.RightDockWidgetArea, self.state_control.dock_widget
        )
        try:
            # Stack Controls, Sequence, Command at the Top
            self.splitDockWidget(
                self.controls_dock,
                self.sequence_dock,
                QtCore.Qt.Vertical,
            )
            self.splitDockWidget(
                self.sequence_dock,
                self.command_panel.dock_widget,
                QtCore.Qt.Vertical,
            )
            # Put state control below/tabbed with objects on the right
            self.splitDockWidget(
                self.objects_dock,
                self.state_control.dock_widget,
                QtCore.Qt.Vertical
            )
        except Exception:
            pass

        try:
            self.resizeDocks(
                [self.controls_dock, self.sequence_dock, self.command_panel.dock_widget],
                [40, 200, 60],
                QtCore.Qt.Vertical,
            )
            self.resizeDocks(
                [self.objects_dock, self.state_control.dock_widget],
                [400, 300],
                QtCore.Qt.Vertical
            )
        except Exception:
            pass

        self._sequence_visible = True
        self._sequence_rows: dict[str, dict[str, Any]] = {}
        self._sequence_alignment_axis: Optional[np.ndarray] = None
        self._sequence_alignment_maps: dict[str, Optional[np.ndarray]] = {}
        try:
            self.seq_label.toggled.connect(self.on_seq_label_toggled)
        except Exception:
            pass

        self._reset_scroll_targets()

        # ------------------------------------------------------------------
        # Viewer + system-info panel (split horizontally)
        # ------------------------------------------------------------------

        # Keep sequence selection and 3D picking in sync.
        try:
            self.viewer.objectResidueSelectionChanged.connect(
                self.on_viewer_residue_selection_changed
            )
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Connections
        # ------------------------------------------------------------------
        self.button_open.clicked.connect(self.on_open_structure)
        self.button_plane.toggled.connect(self.viewer.set_plane_visible)
        self.button_color.toggled.connect(self.on_color_aa_toggled)
        self.button_color_ss.toggled.connect(self.on_color_ss_toggled)
        self.button_color_sequence.toggled.connect(self.on_color_sequence_toggled)
        self.button_rep_cartoon.toggled.connect(self.on_rep_cartoon_toggled)
        self.button_rep_atoms.toggled.connect(self.on_rep_atoms_toggled)
        self.button_rep_sticks.toggled.connect(self.viewer.set_sticks_visible)
        self.button_rep_trace.toggled.connect(self.viewer.set_trace_visible)
        self.button_rep_dots.toggled.connect(self.viewer.set_dots_visible)
        self.button_surface.toggled.connect(self.viewer.set_surface_visible)
        self.button_rep_metaballs.toggled.connect(self.viewer.set_metaballs_visible)
        self.button_info.toggled.connect(self.on_toggle_info_panel)
        self.button_display_cfg.clicked.connect(self.on_open_display_config)

        # Synchronize initial color-mode toggles with the viewer's default.
        try:
            mode = getattr(self.viewer, "_color_mode", "single") or "single"
        except Exception:
            mode = "single"
        try:
            self.button_color.blockSignals(True)
            self.button_color_ss.blockSignals(True)
            self.button_color_sequence.blockSignals(True)
            self.button_color.setChecked(mode == "by_residue")
            self.button_color_ss.setChecked(mode == "by_secondary_structure")
            self.button_color_sequence.setChecked(mode == "by_sequence")
        except Exception:
            pass
        finally:
            try:
                self.button_color.blockSignals(False)
                self.button_color_ss.blockSignals(False)
                self.button_color_sequence.blockSignals(False)
            except Exception:
                pass

        # Update system-info text when sequence selection changes.
        self.seq_list.itemSelectionChanged.connect(self.on_sequence_selection_changed)

        self._default_object_name_counter = 0

        try:
            self.viewer.set_system_info_visible(self.button_info.isChecked())
        except Exception:
            pass

        try:
            _cmd.set_window(self)
            _cmd.set_message_callback(self.command_panel.append_message)
            _cmd.set_error_callback(self.command_panel.append_error)
            self.command_panel.commandEntered.connect(self._on_command_entered)
        except Exception:
            pass

        # Initialize panels with placeholder content.
        self._update_sequence_view()
        self._update_system_info()

        # Restore window state (geometry/layout)
        try:
            from chisurf.gui.misc_helpers import restore_plugin_window_state
            restore_plugin_window_state(self, "chimol")
        except Exception:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle main window close events.

        Saves the window state and geometry before accepting the close event.

        Parameters
        ----------
        event : QtGui.QCloseEvent
            The Qt close event object.
        """
        try:
            from chisurf.gui.misc_helpers import save_plugin_window_state
            save_plugin_window_state(self, "chimol")
        except Exception:
            pass
        event.accept()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------


    def _on_command_entered(self, line: str) -> None:
        try:
            _cmd.do(line)
        except Exception:
            pass

    def on_open_structure(self):
        """Open a structure file and display it in the viewer."""

        filenames = open_structure_files(
            self,
            opener=_cs_open_files,
            description="Open structure file",
            file_type=(
                "Structure / map files (*.pdb *.ent *.gro *.cif *.mmcif *.mrc *.map *.ccp4 *.mrc.gz *.map.gz *.ccp4.gz);;"
                "All files (*.*)"
            ),
        )
        if not filenames:
            return

        loaded_any = False
        for path in filenames:
            try:
                self._load_structure_from_path(Path(path))
                loaded_any = True
            except Exception as e:
                try:
                    cs.logging.warning(
                        "MolViewPluginWindow.on_open_structure: failed to load '%s': %s",
                        path,
                        e,
                    )
                except Exception:
                    pass
                try:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Failed to load structure",
                        f"Could not load structure from:\n{path}\n\n{e}",
                    )
                except Exception:
                    pass

        if loaded_any:
            current_id = self.viewer.get_active_object_id()
            self._select_object_in_ui(current_id)

    def on_toggle_info_panel(self, checked: bool) -> None:
        try:
            self.viewer.set_system_info_visible(bool(checked))
        except Exception:
            return

        # When turning the info panel on, refresh the system summary so the
        # overlay is immediately populated for the currently loaded structure.
        if checked:
            try:
                self._update_system_info()
            except Exception:
                pass

    def on_rep_cartoon_toggled(self, checked: bool) -> None:
        """Toggle cartoon representation for selected or all residues."""

        if self.viewer is None:
            return
        idx = self._selected_residue_indices()
        if not idx:
            try:
                self.viewer.set_cartoon_visible(bool(checked))
            except Exception:
                pass
            return
        try:
            self.viewer.set_residue_representation(idx, cartoon=bool(checked))
        except Exception:
            pass

    def on_rep_atoms_toggled(self, checked: bool) -> None:
        """Toggle atoms/balls representation for selected or all residues."""

        if self.viewer is None:
            return
        idx = self._selected_residue_indices()
        if not idx:
            try:
                self.viewer.set_atoms_visible_all(bool(checked))
            except Exception:
                pass
            return
        try:
            self.viewer.set_residue_representation(idx, ball=bool(checked))
        except Exception:
            pass

    def on_color_aa_toggled(self, checked: bool) -> None:
        """Toggle coloring by amino-acid type in the 3D view."""

        # Turning AA-coloring on should turn SS-coloring off to avoid
        # conflicting modes.
        if checked and self.button_color_ss.isChecked():
            self.button_color_ss.blockSignals(True)
            self.button_color_ss.setChecked(False)
            self.button_color_ss.blockSignals(False)
        if checked and self.button_color_sequence.isChecked():
            self.button_color_sequence.blockSignals(True)
            self.button_color_sequence.setChecked(False)
            self.button_color_sequence.blockSignals(False)

        try:
            self.viewer.set_color_mode("by_residue" if checked else "single")
            self._update_sequence_view()
        except Exception:
            pass

    def on_color_ss_toggled(self, checked: bool) -> None:
        """Toggle coloring by secondary structure in both sequence and 3D."""

        # Turning SS-coloring on disables plain AA-coloring.
        if checked and self.button_color.isChecked():
            self.button_color.blockSignals(True)
            self.button_color.setChecked(False)
            self.button_color.blockSignals(False)
        if checked and self.button_color_sequence.isChecked():
            self.button_color_sequence.blockSignals(True)
            self.button_color_sequence.setChecked(False)
            self.button_color_sequence.blockSignals(False)

        if not checked:
            # When disabling SS coloring, fall back to AA or single.
            try:
                if self.button_color_sequence.isChecked():
                    mode = "by_sequence"
                elif self.button_color.isChecked():
                    mode = "by_residue"
                else:
                    mode = "single"
                self.viewer.set_color_mode(mode)
                self._update_sequence_view()
            except Exception:
                pass
            return

        # Ensure we have secondary-structure codes; _get_secondary_structure_codes
        # will cache them the first time it's called.
        active_id = self.viewer.get_active_object_id()
        if active_id is None:
            return

        seq_codes, res_names = self.viewer.get_sequence_arrays(active_id)
        n = 0
        if seq_codes is not None:
            n = len(seq_codes)
        elif res_names is not None:
            n = len(res_names)
        ss_codes = self._get_secondary_structure_codes(active_id, n) if n > 0 else None

        if ss_codes is None:
            return

        try:
            self.viewer.set_secondary_structure_codes(ss_codes)
            self.viewer.set_color_mode("by_secondary_structure")
            self._update_sequence_view()
        except Exception:
            pass

    def on_color_sequence_toggled(self, checked: bool) -> None:
        """Toggle coloring by sequence index gradient."""

        if checked:
            if self.button_color.isChecked():
                self.button_color.blockSignals(True)
                self.button_color.setChecked(False)
                self.button_color.blockSignals(False)
            if self.button_color_ss.isChecked():
                self.button_color_ss.blockSignals(True)
                self.button_color_ss.setChecked(False)
                self.button_color_ss.blockSignals(False)
            try:
                self.viewer.set_color_mode("by_sequence")
                self._update_sequence_view()
            except Exception:
                pass
            return

        # When disabling the sequence gradient, respect other toggles.
        try:
            if self.button_color_ss.isChecked():
                self.viewer.set_color_mode("by_secondary_structure")
            elif self.button_color.isChecked():
                self.viewer.set_color_mode("by_residue")
            else:
                self.viewer.set_color_mode("single")
            self._update_sequence_view()
        except Exception:
            pass

    def on_open_display_config(self) -> None:
        """Open the Chimol display configuration in a built-in editor."""

        try:
            from chisurf.plugins.chimol.chimol.config import (
                get_user_display_config_path,
                get_package_display_config_path,
            )
            json_path = get_user_display_config_path()
            if json_path is None or not json_path.is_file():
                pkg_path = get_package_display_config_path()
                if pkg_path.is_file():
                    if json_path is None:
                        json_path = pkg_path
                    else:
                        json_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(pkg_path, json_path)
        except Exception:
            json_path = Path(__file__).with_name("chimol_display.json")

        dlg = MolViewConfigEditor(self, json_path=json_path, viewer=self.viewer)
        dlg.exec_()

    def _update_sequence_view(self, object_id: Optional[str] = None) -> None:
        active_id = object_id or self.viewer.get_active_object_id()

        seq_cfg = self.sequence.sequence_config()
        seq_view_enabled = bool(seq_cfg.get("seq_view", True))
        self.sequence_dock.setVisible(seq_view_enabled)
        if not seq_view_enabled:
            return

        self.seq_numbers_list.clear()
        self.seq_numbers_list.setEnabled(False)
        self.seq_list.clear()
        self._clear_extra_sequence_rows()

        if active_id is None:
            self._sequence_visible = True
            self._sequence_alignment_axis = None
            self._sequence_alignment_maps = {}
            try:
                self.seq_label.blockSignals(True)
                self.seq_label.setText("No molecule selected")
                self.seq_label.setChecked(True)
                self.seq_label.blockSignals(False)
            except Exception:
                pass
            self.seq_list.addItem("(no molecule selected)")
            self.seq_list.setEnabled(False)
            self.seq_numbers_list.addItem("(no molecule selected)")
            self.seq_numbers_list.setEnabled(False)
            self._reset_scroll_targets()
            return

        if not self._object_store:
            self._sequence_alignment_axis = None
            self._sequence_alignment_maps = {}
            self.seq_list.addItem("(no molecules loaded)")
            self.seq_list.setEnabled(False)
            self.seq_numbers_list.addItem("(no molecules loaded)")
            self.seq_numbers_list.setEnabled(False)
            self._reset_scroll_targets()
            return

        seq_data: dict[str, tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
        lengths: dict[str, int] = {}
        residue_numbers_map: dict[str, Optional[np.ndarray]] = {}
        residue_colors_map: dict[str, Optional[np.ndarray]] = {}
        for obj_id in self._object_store.keys():
            seq_codes, res_names = self.viewer.get_sequence_arrays(obj_id)
            seq_data[obj_id] = (seq_codes, res_names)
            lengths[obj_id] = self._sequence_length(seq_codes, res_names)
            try:
                residue_numbers_map[obj_id] = self.viewer.get_residue_numbers(obj_id)
            except Exception:
                residue_numbers_map[obj_id] = None
            try:
                residue_colors_map[obj_id] = self.viewer.get_residue_colors(obj_id)
            except Exception:
                residue_colors_map[obj_id] = None

        # Build a shared residue-number axis across all loaded molecules. When
        # PDB residue ids are available, this aligns sequences by residue
        # number and exposes explicit gaps; otherwise it falls back to a
        # simple 1..N index axis matching the longest sequence.
        gap_mode = int(seq_cfg.get("seq_view_gap_mode", 1))
        try:
            if gap_mode > 0:
                axis, maps = build_residue_alignment(residue_numbers_map, lengths)
                if axis is not None:
                    axis, maps = self._collapse_alignment_gaps(axis, maps)
            else:
                axis = None
                maps = {}
        except Exception:
            axis = None
            maps = {}

        axis_len = 0
        if axis is not None:
            try:
                axis_arr = np.asarray(axis)
                if axis_arr.ndim == 1:
                    axis_len = int(axis_arr.shape[0])
            except Exception:
                axis_len = 0

        if axis_len > 0:
            max_len = axis_len
            try:
                self._sequence_alignment_axis = np.asarray(axis, dtype=int)
            except Exception:
                self._sequence_alignment_axis = None
            try:
                self._sequence_alignment_maps = {
                    str(k): (np.asarray(v) if v is not None else None)
                    for k, v in maps.items()
                }
            except Exception:
                self._sequence_alignment_maps = {}
        else:
            max_len = max(lengths.values()) if lengths else 0
            self._sequence_alignment_axis = None
            self._sequence_alignment_maps = {str(k): None for k in self._object_store.keys()}

        entry = self._object_store.get(active_id)
        label_text = str(active_id)
        visible = True
        if entry is not None:
            label_text = entry.get("name", active_id)
            visible = bool(entry.get("visible", True))

        self._sequence_visible = visible
        try:
            self.seq_label.blockSignals(True)
            self.seq_label.setText(label_text)
            self.seq_label.setChecked(visible)
            self.seq_label.blockSignals(False)
        except Exception:
            pass

        if max_len <= 0:
            self._sequence_alignment_axis = None
            self._sequence_alignment_maps = {}
            placeholder = "(no sequence information)"
            self.seq_list.addItem(placeholder)
            self.seq_list.setEnabled(False)
            self.seq_numbers_list.addItem(placeholder)
            self.seq_numbers_list.setEnabled(False)
            self._reset_scroll_targets()
            return

        ss_codes_map: dict[str, Optional[Sequence[str]]] = {}
        for obj_id, length in lengths.items():
            if length > 0:
                ss_codes_map[obj_id] = self._get_secondary_structure_codes(obj_id, length)
            else:
                ss_codes_map[obj_id] = None

        active_seq_codes, active_res_names = seq_data.get(active_id, (None, None))
        active_len = lengths.get(active_id, 0)
        active_ss = ss_codes_map.get(active_id)

        active_empty_text = "(no sequence information)"
        if (active_seq_codes is not None and len(active_seq_codes) > 0) or (
            active_res_names is not None and len(active_res_names) > 0
        ):
            active_empty_text = ""

        active_items = self._build_sequence_items(
            seq_codes=active_seq_codes,
            res_names=active_res_names,
            ss_codes=active_ss,
            max_len=max_len,
            enable_selection=active_len > 0,
            empty_text=active_empty_text or "(no residues)",
            res_numbers=residue_numbers_map.get(active_id),
            residue_colors=residue_colors_map.get(active_id),
            index_map=self._sequence_alignment_maps.get(active_id),
        )
        for item in active_items:
            self.seq_list.addItem(item)

        self.seq_list.setEnabled(active_len > 0)
        self._apply_sequence_selection_styles(set())

        for obj_id, obj_entry in self._object_store.items():
            if obj_id == active_id:
                continue
            seq_codes, res_names = seq_data.get(obj_id, (None, None))
            ss_codes = ss_codes_map.get(obj_id)
            length = lengths.get(obj_id, 0)
            self._add_sequence_row_for_object(
                object_id=obj_id,
                entry=obj_entry,
                seq_codes=seq_codes,
                res_names=res_names,
                ss_codes=ss_codes,
                length=length,
                max_len=max_len,
                res_numbers=residue_numbers_map.get(obj_id),
                residue_colors=residue_colors_map.get(obj_id),
            )

        seq_cfg = self.sequence.sequence_config()
        number_step = max(1, int(seq_cfg.get("seq_view_label_spacing", seq_cfg.get("number_step", 5))))

        residue_numbers = None
        try:
            if hasattr(self, "viewer") and self.viewer is not None:
                residue_numbers = self.viewer.get_residue_numbers(active_id)
        except Exception:
            residue_numbers = None

        # When a global alignment axis is available, use it for the sequence
        # number row so the labels reflect the shared PDB residue numbers.
        axis_nums = None
        try:
            axis_arr = getattr(self, "_sequence_alignment_axis", None)
            if axis_arr is not None:
                axis_arr = np.asarray(axis_arr)
                if axis_arr.ndim == 1 and axis_arr.shape[0] == max_len:
                    axis_nums = axis_arr
        except Exception:
            axis_nums = None

        if axis_nums is not None:
            numbers_for_row = axis_nums
        else:
            numbers_for_row = residue_numbers

        self._populate_sequence_numbers(
            max_len,
            number_step,
            active_length=active_len,
            residue_numbers=numbers_for_row,
        )
        enabled = max_len > 0
        self.seq_numbers_list.setEnabled(enabled)
        try:
            self.seq_numbers_label.setEnabled(enabled)
        except Exception:
            pass

        self._reset_scroll_targets()

    def _collapse_alignment_gaps(
        self,
        axis: np.ndarray,
        maps: dict[str, Optional[np.ndarray]],
        max_consecutive_gaps: int = 9,
    ) -> tuple[np.ndarray, dict[str, Optional[np.ndarray]]]:
        """Collapse large empty stretches of residue numbers into a 9-column block.

        Parameters
        ----------
        axis : np.ndarray
            1D array representing the global residue numbers alignment axis.
        maps : dict
            Mapping from object ID to 1D array of sequence indices.
        max_consecutive_gaps : int, optional
            Threshold above which a run of consecutive gaps is collapsed.

        Returns
        -------
        new_axis : np.ndarray
            The collapsed residue number axis.
        new_maps : dict
            The collapsed sequence index maps.
        """
        if axis is None or len(axis) == 0:
            return axis, maps

        # Identify objects that have a non-None map of length equal to axis
        valid_keys = [k for k, v in maps.items() if v is not None and len(v) == len(axis)]
        if not valid_keys:
            return axis, maps

        is_empty = np.ones(len(axis), dtype=bool)
        for k in valid_keys:
            is_empty &= (np.asarray(maps[k]) == -1)

        new_axis_list = []
        new_maps_lists = {k: [] for k in maps.keys()}

        i = 0
        N = len(axis)
        while i < N:
            is_run = False
            if is_empty[i]:
                run_end = i
                while run_end + 1 < N and is_empty[run_end + 1]:
                    run_end += 1
                run_len = run_end - i + 1
                if run_len > max_consecutive_gaps:
                    is_run = True

            if is_run:
                # Collapse the run from i to run_end into exactly 9 columns:
                # [-1, -1, -1, -2, -2, -2, -1, -1, -1]
                # For axis, we keep the boundary residue numbers for the non-dot parts
                for offset in range(9):
                    if offset < 3:
                        orig_idx = i + offset
                        axis_val = axis[orig_idx]
                        map_val = -1
                    elif offset < 6:
                        axis_val = -1  # Skip number display
                        map_val = -2  # Render '.' (dot)
                    else:
                        orig_idx = run_end - (8 - offset)
                        axis_val = axis[orig_idx]
                        map_val = -1

                    new_axis_list.append(axis_val)
                    for k in maps.keys():
                        if k in valid_keys:
                            new_maps_lists[k].append(map_val)
                        else:
                            new_maps_lists[k].append(-1)
                i = run_end + 1
            else:
                new_axis_list.append(axis[i])
                for k, v in maps.items():
                    if v is not None and i < len(v):
                        new_maps_lists[k].append(v[i])
                    else:
                        new_maps_lists[k].append(-1)
                i += 1

        new_axis = np.array(new_axis_list, dtype=int)
        new_maps = {}
        for k in maps.keys():
            if maps[k] is not None:
                new_maps[k] = np.array(new_maps_lists[k], dtype=int)
            else:
                new_maps[k] = None

        return new_axis, new_maps

    def _clear_extra_sequence_rows(self) -> None:
        self._sequence_rows.clear()
        layout = getattr(self, "_extra_seq_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _add_sequence_row_for_object(
        self,
        object_id: str,
        entry: dict[str, Any],
        *,
        seq_codes: Optional[Sequence[object]],
        res_names: Optional[Sequence[object]],
        ss_codes: Optional[Sequence[object]],
        length: int,
        max_len: int,
        res_numbers: Optional[np.ndarray] = None,
        residue_colors: Optional[np.ndarray] = None,
    ) -> None:
        container = getattr(self, "_extra_seq_container", None)
        layout = getattr(self, "_extra_seq_layout", None)
        if container is None or layout is None:
            return

        row_widget = QtWidgets.QWidget(container)
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        button = QtWidgets.QToolButton(row_widget)
        button.setMinimumWidth(120)
        label_text = entry.get("name", object_id)
        button.setText(str(label_text))
        button.setCheckable(True)
        visible = bool(entry.get("visible", True))
        button.setChecked(visible)

        seq_list = QtWidgets.QListWidget(row_widget)
        seq_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        seq_list.setFlow(QtWidgets.QListView.LeftToRight)
        seq_list.setWrapping(False)
        seq_list.setUniformItemSizes(True)
        seq_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        seq_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        seq_list.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        try:
            seq_font = getattr(self, "_sequence_font", None)
            if isinstance(seq_font, QtGui.QFont):
                seq_list.setFont(seq_font)
            seq_list.setFixedHeight(self.seq_list.height())
            seq_list.setStyleSheet(self.seq_list.styleSheet())
        except Exception:
            pass

        row_layout.addWidget(button, 0, QtCore.Qt.AlignVCenter)
        row_layout.addWidget(seq_list, 1)
        layout.addWidget(row_widget)

        self._sequence_rows[object_id] = {"button": button, "list": seq_list}

        extra_items = self._build_sequence_items(
            seq_codes=seq_codes,
            res_names=res_names,
            ss_codes=ss_codes,
            max_len=max_len,
            enable_selection=False,
            empty_text="(no sequence information)" if length <= 0 else "",
            res_numbers=res_numbers,
            residue_colors=residue_colors,
            index_map=self._sequence_alignment_maps.get(object_id),
        )
        for item in extra_items:
            seq_list.addItem(item)

        seq_list.setEnabled(bool(visible))
        self._update_sequence_row_colors(object_id, visible)

        button.toggled.connect(
            lambda checked, oid=object_id: self.on_seq_row_toggled(oid, checked)
        )

    def _update_sequence_row_colors(self, object_id: str, visible: bool) -> None:
        row_info = self._sequence_rows.get(str(object_id)) if hasattr(self, "_sequence_rows") else None
        if not isinstance(row_info, dict):
            return
        lst = row_info.get("list")
        if not isinstance(lst, QtWidgets.QListWidget):
            return
        if lst.count() == 0:
            return

        gray_bg = QtGui.QColor(200, 200, 200)
        gray_fg = QtGui.QColor(140, 140, 140)

        for i in range(lst.count()):
            item = lst.item(i)
            if item is None:
                continue
            palette = item.data(_SEQ_COLOR_ROLE)
            if isinstance(palette, tuple) and len(palette) == 2:
                base_bg, base_fg = palette
            else:
                base_bg = QtGui.QColor(220, 220, 200)
                base_fg = QtGui.QColor(0, 0, 0)

            if not visible:
                item.setBackground(QtGui.QBrush(gray_bg))
                item.setForeground(QtGui.QBrush(gray_fg))
            else:
                item.setBackground(QtGui.QBrush(base_bg))
                item.setForeground(QtGui.QBrush(base_fg))

    def _sequence_length(
        self,
        seq_codes: Optional[Sequence[object]],
        res_names: Optional[Sequence[object]],
    ) -> int:
        if seq_codes is not None:
            try:
                return len(seq_codes)
            except Exception:
                pass
        if res_names is not None:
            try:
                return len(res_names)
            except Exception:
                pass
        return 0

    def _update_system_info(self, object_id: Optional[str] = None) -> None:
        active_id = object_id or self.viewer.get_active_object_id()
        entry = self._object_store.get(active_id) if active_id else None

        if entry is None:
            lines = ["(no system loaded)"]
        else:
            structure = entry.get("structure")
            if structure is not None:
                try:
                    n_atoms = structure.n_atoms
                except Exception:
                    n_atoms = "?"
                try:
                    n_res = structure.n_residues
                except Exception:
                    n_res = "?"
                try:
                    r_g_val = structure.radius_gyration
                    r_g = f"{r_g_val:.1f}"
                except Exception:
                    r_g = "?"
                system_label = "System: protein"
            else:
                n_atoms = entry.get("n_atoms", "?")
                seq_codes, res_names = self.viewer.get_sequence_arrays(active_id)
                if seq_codes is not None:
                    n_res = len(seq_codes)
                elif res_names is not None:
                    n_res = len(res_names)
                else:
                    n_res = "?"
                r_g = "?"
                system_label = "System: coordinates"

            lines = [
                system_label,
                "",
                f"File: {entry.get('path', '?')}",
                f"Atoms: {n_atoms}",
                f"Residues: {n_res}",
                f"Radius of gyration: {r_g}",
            ]

            # RMF specifics
            try:
                state = self.viewer._get_active_state()
                if state and getattr(state, "restraints", None):
                    lines.append(f"Restraints: {len(state.restraints)}")
                if state and getattr(state, "rmf_provenance", None):
                    lines.append("")
                    lines.append("RMF Provenance:")
                    for prov in state.rmf_provenance:
                        lines.append(f"  {prov.get('name', '?')}: {prov.get('value', '?')}")
            except Exception:
                pass

            try:
                n_frames = self.viewer.get_frame_count(active_id)
            except Exception:
                n_frames = 0
            if n_frames > 1:
                try:
                    frame_idx = self.viewer.get_active_frame_index(active_id)
                except Exception:
                    frame_idx = 0
                lines.append(f"Frame: {frame_idx + 1} / {n_frames}")

            sel_idx = self._selected_residue_indices()
            if sel_idx:
                seq_codes, res_names = self.viewer.get_sequence_arrays(active_id)
                try:
                    resno_arr = self.viewer.get_residue_numbers(active_id)
                except Exception:
                    resno_arr = None
                desc: list[str] = []
                for i in sel_idx:
                    try:
                        if resno_arr is not None and 0 <= i < len(resno_arr):
                            res_no = int(resno_arr[i])
                        else:
                            res_no = i + 1
                    except Exception:
                        res_no = i + 1
                    name = (
                        str(res_names[i])
                        if res_names is not None and i < len(res_names)
                        else "?"
                    )
                    one = (
                        str(seq_codes[i])
                        if seq_codes is not None and i < len(seq_codes)
                        else "?"
                    )
                    desc.append(f"{res_no}: {name} ({one})")

                lines.append("")
                lines.append("Selected residues:")
                lines.append(", ".join(desc))

        text = "\n".join(str(x) for x in lines)
        try:
            self.viewer.set_system_info_text(text)
        except Exception:
            pass

    def on_sequence_selection_changed(self) -> None:
        """Sync 3D selection and info text when the sequence selection changes."""

        # Map QListWidget selection to residue indices
        idx = self._selected_residue_indices()
        try:
            self._apply_sequence_selection_styles(set(idx))
        except Exception:
            pass
        object_id = self.viewer.get_active_object_id()
        if object_id is None:
            return

        try:
            self.viewer.set_selected_residues(idx, object_id=object_id)
        except Exception:
            pass

        try:
            self._update_system_info(object_id)
        except Exception:
            pass

    def on_viewer_residue_selection_changed(self, object_id, indices) -> None:
        """Update sequence selection to match picks from the 3D viewer."""

        active_id = self.viewer.get_active_object_id()
        if object_id != active_id:
            return

        if not isinstance(indices, (list, tuple)):
            try:
                indices = list(indices)
            except Exception:
                indices = []

        self.seq_list.blockSignals(True)
        try:
            self.seq_list.clearSelection()
            # Map residue indices from the viewer back to visible rows using
            # the stored sequence index role, so alignment gaps are handled
            # correctly.
            try:
                target_idx = {int(i) for i in indices if int(i) >= 0}
            except Exception:
                target_idx = set()
            if target_idx:
                for row in range(self.seq_list.count()):
                    item = self.seq_list.item(row)
                    if item is None:
                        continue
                    seq_idx = item.data(_SEQ_INDEX_ROLE)
                    if isinstance(seq_idx, int) and seq_idx in target_idx:
                        item.setSelected(True)
        finally:
            self.seq_list.blockSignals(False)

        try:
            self._apply_sequence_selection_styles({int(i) for i in indices})
        except Exception:
            pass

        try:
            self._update_system_info(object_id)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Object management helpers
    # ------------------------------------------------------------------

    def _load_structure_from_path(self, path: Path, *, name: Optional[str] = None) -> str:
        """Load a structure or coordinate file and register it as a new object."""

        source_path = str(path)
        if name is not None:
            try:
                fake_path = Path(str(name))
            except Exception:
                fake_path = None
            display_name = self._make_object_name(fake_path)
        else:
            display_name = self._make_object_name(path)

        structure = None
        coords_arr: Optional[np.ndarray] = None
        primary_exc: Optional[Exception] = None

        # Special-case volumetric EM maps (MRC/CCP4/EMDB) and load them as
        # point clouds that can be rendered via the dots representation.
        name_lower = path.name.lower()
        is_map = name_lower.endswith((
            ".mrc",
            ".map",
            ".ccp4",
            ".mrc.gz",
            ".map.gz",
            ".ccp4.gz",
        ))

        if is_map:
            points, meta = load_mrc_as_points(path)
            pts_arr = np.asarray(points, dtype=float)
            if pts_arr.ndim != 2 or pts_arr.shape[1] != 3 or pts_arr.shape[0] == 0:
                raise ValueError(f"No valid voxel positions in map {path!s}")

            object_id = self.viewer.add_coordinates(
                pts_arr,
                name=display_name,
                source_path=source_path,
            )
            n_atoms: Any = int(pts_arr.shape[0])

            try:
                self.viewer.set_dots_visible(True)
            except Exception:
                pass

            entry: dict[str, Any] = {
                "name": display_name,
                "path": source_path,
                "structure": None,
                "ss_codes": None,
                "visible": True,
                "n_atoms": n_atoms,
                "mrc_meta": meta,
            }
            self._object_store[object_id] = entry
            self._add_object_list_item(object_id, entry)
            self._select_object_in_ui(object_id)
            return object_id

        if name_lower.endswith((".rmf", ".rmf3")):
            try:
                data = load_rmf_full(path)
                hierarchy = data["hierarchy"]
                frames = data["frames"]
                radii = data["radii"]
                restraints = data.get("restraints")
                rmf_provenance = data.get("rmf_provenance")
                bond_pairs = data.get("bond_pairs")
                
                object_id = self.viewer._create_object(name=display_name, source_path=source_path).object_id
                self.viewer.set_rmf_data(
                    hierarchy=hierarchy,
                    frames=frames,
                    radii=radii,
                    restraints=restraints,
                    rmf_provenance=rmf_provenance,
                    bond_pairs=bond_pairs,
                    object_id=object_id
                )
                
                n_atoms = int(frames.shape[1])
                entry: dict[str, Any] = {
                    "name": display_name,
                    "path": source_path,
                    "visible": True,
                    "n_atoms": n_atoms,
                    "rmf_hierarchy": hierarchy,
                }
                self._object_store[object_id] = entry
                self._add_object_list_item(object_id, entry)
                self._select_object_in_ui(object_id)
                self.hierarchy.set_hierarchy(hierarchy)
                return object_id
            except RmfNotAvailableError as e:
                QtWidgets.QMessageBox.warning(self, "RMF Not Available", str(e))
                raise
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "RMF Load Error", f"Failed to load RMF: {e}")
                raise

        # First try the standard IMP/Structure-based loader for static files.
        try:
            structure, coords = load_structure_payload(
                path,
                structure_factory=_ChiSurfStructure,
            )
        except Exception as e:
            primary_exc = e
            structure = None
            coords = None
        else:
            if coords is not None:
                coords_arr = np.asarray(coords, dtype=float)

        object_id: str
        n_atoms: Any

        if structure is not None:
            object_id = self.viewer.add_structure(
                structure,
                name=display_name,
                source_path=source_path,
            )
            n_atoms = getattr(structure, "n_atoms", "?")
        elif coords_arr is not None:
            object_id = self.viewer.add_coordinates(
                coords_arr,
                name=display_name,
                source_path=source_path,
            )
            n_atoms = 0 if coords_arr is None else coords_arr.shape[0]
        else:
            # No usable static structure/coords; try specialised trajectory
            # loaders (RMF via IMP/PMI, then MDTraj for e.g. GRO/HDF5).
            suffix = path.suffix.lower()

            if suffix in {".rmf", ".rmf3"}:
                try:
                    frames = load_rmf_frames(path)
                except RmfNotAvailableError:
                    # Surface a clear message to GUI/cmd callers.
                    raise
                except Exception:
                    if primary_exc is not None:
                        raise primary_exc
                    raise

                arr = np.asarray(frames, dtype=float)
                if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] == 0:
                    raise ValueError(
                        f"Invalid RMF trajectory array from {path!s}: shape={arr.shape!r}"
                    )

                first = arr[0]
                object_id = self.viewer.add_coordinates(
                    np.asarray(first, dtype=float),
                    name=display_name,
                    source_path=source_path,
                )

                try:
                    self.viewer.set_frames(arr, object_id=object_id)
                except Exception:
                    # If anything goes wrong, we still keep the first frame as a
                    # static coordinate set.
                    pass

                try:
                    n_beads = int(first.shape[0])
                except Exception:
                    n_beads = 0
                if n_beads > 0:
                    try:
                        cov = np.eye(3, dtype=float)[np.newaxis, :, :]
                        cov = np.tile(cov, (n_beads, 1, 1))
                        self.viewer.set_atom_features(
                            {"gaussian_covariances": cov},
                            object_id=object_id,
                        )
                        self.viewer.set_atom_gaussians_visible(True)
                    except Exception:
                        pass

                n_atoms = int(first.shape[0])
            else:
                # Fallback: MDTraj-based trajectory loading for formats IMP does
                # not support (e.g. GRO/HDF5).
                try:
                    frames = load_trajectory_frames(path)
                except MdtrajNotAvailableError:
                    # Surface a clear message to GUI/cmd callers.
                    raise
                except Exception:
                    if primary_exc is not None:
                        raise primary_exc
                    raise

                arr = np.asarray(frames, dtype=float)
                if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] == 0:
                    raise ValueError(
                        f"Invalid trajectory array from {path!s}: shape={arr.shape!r}"
                    )

                first = arr[0]
                object_id = self.viewer.add_coordinates(
                    np.asarray(first, dtype=float),
                    name=display_name,
                    source_path=source_path,
                )

                try:
                    self.viewer.set_frames(arr, object_id=object_id)
                except Exception:
                    # If anything goes wrong, we still keep the first frame as a
                    # static coordinate set.
                    pass

                n_atoms = int(first.shape[0])

        entry: dict[str, Any] = {
            "name": display_name,
            "path": source_path,
            "structure": structure,
            "ss_codes": None,
            "visible": True,
            "n_atoms": n_atoms,
        }
        self._object_store[object_id] = entry
        self._add_object_list_item(object_id, entry)
        self._select_object_in_ui(object_id)
        return object_id

    def _refresh_objects_from_viewer(self) -> None:
        """Rebuild the object list/UI from the viewer's current objects (e.g. after split_chains)."""

        if self.viewer is None:
            return

        objects = self.viewer.list_objects()

        # Reset store and UI list
        self._object_store.clear()
        try:
            self.object_list.blockSignals(True)
            self.object_list.clear()
        finally:
            self.object_list.blockSignals(False)

        for obj in objects:
            oid = str(obj.get("id"))
            name = obj.get("name") or oid
            entry = {
                "name": name,
                "path": obj.get("source_path"),
                "structure": None,
                "ss_codes": None,
                "visible": bool(obj.get("visible", True)),
                "n_atoms": obj.get("n_atoms", obj.get("has_geometry")),
            }
            self._object_store[oid] = entry
            self._add_object_list_item(oid, entry)

        # Reselect the active object if possible
        active_id = self.viewer.get_active_object_id()
        if active_id is None and objects:
            active_id = objects[0].get("id")
        if active_id is not None:
            self._select_object_in_ui(active_id)
        self._update_sequence_view()
        self._update_system_info()

    def _set_object_visible(self, object_id: str, visible: bool) -> None:
        try:
            self.viewer.set_object_visible(object_id, visible)
        except Exception:
            pass

        entry = self._object_store.get(object_id)
        if entry is not None:
            entry["visible"] = bool(visible)

    def _make_object_name(self, path: Optional[Path]) -> str:
        if path is None:
            base = f"Molecule {self._default_object_name_counter + 1}"
        else:
            base = path.stem or "Molecule"

        existing = {entry.get("name") for entry in self._object_store.values()}
        if base not in existing:
            return base

        suffix = 2
        while f"{base} ({suffix})" in existing:
            suffix += 1
        return f"{base} ({suffix})"

    def _add_object_list_item(self, object_id: str, entry: dict[str, Any]) -> None:
        item = self.objects.create_item(object_id, entry)

        self._block_object_list_signals = True
        try:
            item.setCheckState(QtCore.Qt.Checked)
        finally:
            self._block_object_list_signals = False

        self.object_list.addItem(item)
        entry["item"] = item

    def _select_object_in_ui(self, object_id: Optional[str]) -> None:
        if object_id is None:
            self.object_list.clearSelection()
            self._handle_active_object_change(None)
            return

        self.objects.set_current_object(object_id)

    def _handle_active_object_change(self, object_id: Optional[str]) -> None:
        if object_id is None:
            self._active_object_id = None
            self._update_sequence_view(None)
            self._update_system_info(None)
            if hasattr(self, "hierarchy"):
                self.hierarchy.set_hierarchy(None)
            return

        self._active_object_id = object_id
        try:
            self.viewer.set_active_object(object_id)
        except Exception:
            pass

        self._update_sequence_view(object_id)
        self._update_system_info(object_id)

        # Update hierarchy dock
        if hasattr(self, "hierarchy"):
            try:
                state = self.viewer._get_active_state()
                self.hierarchy.set_hierarchy(state.rmf_hierarchy)
            except Exception:
                self.hierarchy.set_hierarchy(None)

    def on_object_selection_changed(self) -> None:
        if self._block_object_list_signals:
            return

        item = self.object_list.currentItem()
        if item is None:
            self._handle_active_object_change(None)
            return

        object_id = item.data(_OBJECT_ID_ROLE)
        if not object_id:
            self._handle_active_object_change(None)
            return

        self._handle_active_object_change(str(object_id))

    def on_object_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._block_object_list_signals or item is None:
            return

        object_id = item.data(_OBJECT_ID_ROLE)
        if not object_id:
            return

        visible = item.checkState() == QtCore.Qt.Checked
        self._set_object_visible(object_id, visible)

        row_info = self._sequence_rows.get(str(object_id)) if hasattr(self, "_sequence_rows") else None
        if isinstance(row_info, dict):
            btn = row_info.get("button")
            lst = row_info.get("list")
            if isinstance(btn, QtWidgets.QToolButton):
                try:
                    btn.blockSignals(True)
                    btn.setChecked(bool(visible))
                    btn.blockSignals(False)
                except Exception:
                    pass
            if isinstance(lst, QtWidgets.QListWidget):
                lst.setEnabled(bool(visible))
            try:
                self._update_sequence_row_colors(str(object_id), bool(visible))
            except Exception:
                pass

        if object_id == self.viewer.get_active_object_id():
            self._sequence_visible = bool(visible)
            try:
                self.seq_label.blockSignals(True)
                self.seq_label.setChecked(bool(visible))
                self.seq_label.blockSignals(False)
            except Exception:
                pass
            try:
                self._apply_sequence_selection_styles(set(self._selected_residue_indices()))
            except Exception:
                pass

    def _on_object_list_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self.object_list)
        action_select_all = menu.addAction("Select All")
        action_clear_selection = menu.addAction("Clear Selection")
        menu.addSeparator()
        action_delete = menu.addAction("Delete Selected")

        chosen = menu.exec_(self.object_list.mapToGlobal(pos))
        if chosen == action_select_all:
            self.object_list.selectAll()
        elif chosen == action_clear_selection:
            self.object_list.clearSelection()
        elif chosen == action_delete:
            self._delete_selected_objects()

    def _delete_selected_objects(self) -> None:
        items = self.object_list.selectedItems()
        if not items:
            return

        object_ids: list[str] = []
        for item in items:
            oid = item.data(_OBJECT_ID_ROLE)
            if oid:
                object_ids.append(str(oid))

        if not object_ids:
            return

        for oid in object_ids:
            try:
                self.viewer.remove_object(oid)
            except Exception:
                pass
            self._object_store.pop(oid, None)

        self._refresh_objects_from_viewer()

    def on_seq_label_toggled(self, checked: bool) -> None:
        active_id = self.viewer.get_active_object_id()
        if active_id is None:
            return

        self._block_object_list_signals = True
        try:
            self._set_object_visible(active_id, checked)
            self.objects.set_item_checked(active_id, checked)
        finally:
            self._block_object_list_signals = False

        self._sequence_visible = bool(checked)
        try:
            self._apply_sequence_selection_styles(set(self._selected_residue_indices()))
        except Exception:
            pass

    def on_seq_row_toggled(self, object_id: str, checked: bool) -> None:
        if not object_id:
            return

        active_id = self.viewer.get_active_object_id()

        self._block_object_list_signals = True
        try:
            self._set_object_visible(object_id, checked)
            self.objects.set_item_checked(object_id, checked)
        finally:
            self._block_object_list_signals = False

        row_info = self._sequence_rows.get(str(object_id)) if hasattr(self, "_sequence_rows") else None
        if isinstance(row_info, dict):
            lst = row_info.get("list")
            if isinstance(lst, QtWidgets.QListWidget):
                lst.setEnabled(bool(checked))
            try:
                self._update_sequence_row_colors(str(object_id), bool(checked))
            except Exception:
                pass

        if active_id == object_id:
            self._sequence_visible = bool(checked)
            try:
                self.seq_label.blockSignals(True)
                self.seq_label.setChecked(bool(checked))
                self.seq_label.blockSignals(False)
            except Exception:
                pass
            try:
                self._apply_sequence_selection_styles(set(self._selected_residue_indices()))
            except Exception:
                pass

    def _selected_residue_indices(self) -> list[int]:
        if self.seq_list.count() == 0:
            return []
        indices: list[int] = []
        for model_idx in self.seq_list.selectedIndexes():
            item = self.seq_list.item(model_idx.row())
            if item is None:
                continue
            seq_idx = item.data(_SEQ_INDEX_ROLE)
            if isinstance(seq_idx, int) and seq_idx >= 0:
                indices.append(seq_idx)
        return sorted(set(indices))

    def _populate_sequence_numbers(
        self,
        max_len: int,
        step: int,
        *,
        active_length: int = 0,
        residue_numbers: Optional[np.ndarray] = None,
    ) -> None:
        self.seq_numbers_list.clear()
        if max_len <= 0:
            return
        digits = len(str(max(1, max_len)))
        seq_cfg = self.sequence.sequence_config()
        fg_rgba = seq_cfg.get("number_color", [0.9, 0.9, 0.9, 1.0])
        bg_rgba = seq_cfg.get("number_bg_color", [0.12, 0.12, 0.12, 1.0])
        fallback_fg = QtGui.QBrush(self.sequence.color_from_rgba(fg_rgba, (0.9, 0.9, 0.9, 1.0)))
        fallback_bg = QtGui.QBrush(self.sequence.color_from_rgba(bg_rgba, (0.12, 0.12, 0.12, 1.0)))

        resno_arr: Optional[np.ndarray]
        try:
            if residue_numbers is not None:
                arr_res = np.asarray(residue_numbers)
                if arr_res.ndim == 1:
                    resno_arr = arr_res
                else:
                    resno_arr = None
            else:
                resno_arr = None
        except Exception:
            resno_arr = None

        color_arr: Optional[np.ndarray]
        try:
            if residue_colors is not None:
                c_arr = np.asarray(residue_colors, dtype=float)
                if c_arr.ndim == 2 and c_arr.shape[1] >= 3:
                    color_arr = c_arr
                else:
                    color_arr = None
            else:
                color_arr = None
        except Exception:
            color_arr = None

        # Build per-position labels so that residue indices are laid out as a
        # single monospaced string like "1   5    10   15   20" where each
        # character aligns with one residue cell in the sequence row below.
        step_val = step if step > 0 else 1
        labels: list[str] = ["·"] * max_len
        if max_len > 0:
            for idx in range(max_len):
                r = -1
                if resno_arr is not None and idx < resno_arr.shape[0]:
                    try:
                        r = int(resno_arr[idx])
                    except Exception:
                        pass
                else:
                    r = idx + 1

                if r <= 0:
                    continue

                if r == 1 or r % step_val == 0:
                    text = str(r)
                    start = idx + 1 - len(text)
                    if start < 0:
                        text = text[-(idx + 1):]
                        start = 0
                    for j, ch in enumerate(text):
                        idx_char = start + j
                        if 0 <= idx_char < max_len:
                            labels[idx_char] = ch

        size_hint = None
        if self.seq_list.count() > 0:
            try:
                idx0 = self.seq_list.model().index(0, 0)
                size_hint = self.seq_list.sizeHintForIndex(idx0)
            except Exception:
                size_hint = None
        normal_font = getattr(self, "_sequence_number_font", None)
        bold_font = getattr(self, "_sequence_number_bold_font", None)

        for idx in range(max_len):
            item = QtWidgets.QListWidgetItem()
            ch = labels[idx] if idx < len(labels) else " "
            text = ch if ch else " "
            item.setText(text)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            item.setFlags(QtCore.Qt.ItemIsEnabled)

            # Use neutral number coloring; never mirror 3D per-residue colors.
            bg_brush = fallback_bg
            fg_brush = fallback_fg

            item.setBackground(bg_brush)
            item.setForeground(fg_brush)

            # Keep size in sync with sequence row.
            ref_item = self.seq_list.item(idx)
            if ref_item is not None:
                hint = ref_item.sizeHint()
                if hint.isValid():
                    item.setSizeHint(hint)
            elif size_hint is not None:
                item.setSizeHint(size_hint)

            if text.strip() and bold_font is not None:
                item.setFont(bold_font)
            elif normal_font is not None:
                item.setFont(normal_font)
            item.setData(_SEQ_INDEX_ROLE, idx)
            self.seq_numbers_list.addItem(item)

    def _build_sequence_items(
        self,
        *,
        seq_codes: Optional[Sequence[object]],
        res_names: Optional[Sequence[object]],
        ss_codes: Optional[Sequence[object]],
        max_len: int,
        enable_selection: bool,
        empty_text: str,
        res_numbers: Optional[np.ndarray] = None,
        residue_colors: Optional[np.ndarray] = None,
        index_map: Optional[np.ndarray] = None,
    ) -> list[QtWidgets.QListWidgetItem]:
        items: list[QtWidgets.QListWidgetItem] = []
        if max_len <= 0:
            placeholder = QtWidgets.QListWidgetItem(empty_text or "(no sequence)")
            placeholder.setTextAlignment(QtCore.Qt.AlignCenter)
            placeholder.setFlags(QtCore.Qt.ItemIsEnabled)
            placeholder.setData(_SEQ_INDEX_ROLE, -1)
            items.append(placeholder)
            return items

        length = self._sequence_length(seq_codes, res_names)
        seq_codes = tuple(seq_codes) if seq_codes is not None else None
        res_names = tuple(res_names) if res_names is not None else None
        ss_codes = tuple(ss_codes) if ss_codes is not None else None

        # Optional PDB residue numbers aligned with the CA trace; when present
        # they will be shown in tooltips and system-info instead of simple
        # 1-based indices.
        resno_arr: Optional[np.ndarray]
        try:
            if res_numbers is not None:
                arr_res = np.asarray(res_numbers)
                if arr_res.ndim == 1:
                    resno_arr = arr_res
                else:
                    resno_arr = None
            else:
                resno_arr = None
        except Exception:
            resno_arr = None

        tooltip_hint = empty_text or ""

        # Optional mapping from alignment-column index to true sequence index
        # (0-based along the CA trace). When provided, this exposes explicit
        # gaps where the value is negative.
        index_map_arr: Optional[np.ndarray]
        try:
            if index_map is not None:
                arr_map = np.asarray(index_map, dtype=int)
                if arr_map.ndim == 1:
                    index_map_arr = arr_map
                else:
                    index_map_arr = None
            else:
                index_map_arr = None
        except Exception:
            index_map_arr = None

        # Optional per-residue RGBA colors. Matches CA trace colors.
        color_arr: Optional[np.ndarray]
        try:
            if residue_colors is not None:
                arr_col = np.asarray(residue_colors, dtype=float)
                if arr_col.ndim == 2 and arr_col.shape[0] == length:
                    color_arr = arr_col
                else:
                    color_arr = None
            else:
                color_arr = None
        except Exception:
            color_arr = None

        for idx in range(max_len):
            item = QtWidgets.QListWidgetItem()
            # Determine the underlying sequence index for this alignment
            # column. When an alignment map is present, negative values mark
            # gaps for this object at the corresponding residue number.
            if index_map_arr is not None and idx < index_map_arr.shape[0]:
                seq_index = int(index_map_arr[idx])
            else:
                seq_index = idx if idx < length else -1

            orig_seq_index = seq_index

            if 0 <= seq_index < length:
                aa = seq_codes[seq_index] if seq_codes is not None else "?"
                aa_str = str(aa) if aa is not None else "?"
                ss = (
                    ss_codes[seq_index]
                    if ss_codes is not None and seq_index < len(ss_codes)
                    else "C"
                )
                ss_str = str(ss).upper() if ss is not None else "C"
                if (
                    color_arr is not None
                    and seq_index < color_arr.shape[0]
                    and color_arr.shape[1] >= 3
                ):
                    try:
                        r, g, b = color_arr[seq_index, :3]
                        a = color_arr[seq_index, 3] if color_arr.shape[1] >= 4 else 1.0
                        bg = QtGui.QColor.fromRgbF(float(r), float(g), float(b), float(a))
                        lum = 0.299 * float(r) + 0.587 * float(g) + 0.114 * float(b)
                        fg = QtGui.QColor(255, 255, 255) if lum < 0.5 else QtGui.QColor(0, 0, 0)
                    except Exception:
                        bg, fg = SequenceDock.default_sequence_palette(ss_str)
                else:
                    bg, fg = SequenceDock.default_sequence_palette(ss_str)
                res_name = (
                    str(res_names[seq_index])
                    if res_names is not None and seq_index < len(res_names)
                    else "?"
                )
                if resno_arr is not None and seq_index < resno_arr.shape[0]:
                    try:
                        res_no = int(resno_arr[seq_index])
                    except Exception:
                        res_no = seq_index + 1
                else:
                    res_no = seq_index + 1
                tooltip = f"{res_no}: {res_name} ({aa_str}), SS={ss_str}"
                text = aa_str
            else:
                # Explicit gap on the alignment axis for this object.
                bg, fg = SequenceDock.default_sequence_palette("C")
                tooltip = tooltip_hint
                seq_index = -1
                text = "." if orig_seq_index == -2 else "-"

            item.setText(text or " ")
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            item.setData(_SEQ_COLOR_ROLE, (bg, fg))
            item.setData(_SEQ_INDEX_ROLE, seq_index)
            if tooltip:
                item.setToolTip(tooltip)
            flags = QtCore.Qt.ItemIsEnabled
            if enable_selection and seq_index >= 0:
                flags |= QtCore.Qt.ItemIsSelectable
            item.setFlags(flags)
            item.setBackground(QtGui.QBrush(bg))
            item.setForeground(QtGui.QBrush(fg))
            items.append(item)

        return items

    def _reset_scroll_targets(self) -> None:
        for bar in getattr(self, "_scroll_targets", []):
            try:
                bar.valueChanged.disconnect(self._on_target_scroll_changed)
            except Exception:
                pass
        if getattr(self, "_scroll_master", None) is not None:
            try:
                self._scroll_master.valueChanged.disconnect(self._on_master_scroll_changed)
            except Exception:
                pass
        if getattr(self, "_content_scrollbar", None) is not None:
            try:
                self._content_scrollbar.rangeChanged.disconnect(self._on_content_scroll_range_changed)
            except Exception:
                pass
            try:
                self._content_scrollbar.valueChanged.disconnect(self._on_target_scroll_changed)
            except Exception:
                pass
        self._content_scrollbar = None

        self._scroll_targets = []
        self._scroll_master = None

        master = getattr(self, "seq_scrollbar", None)
        content_bar = self.seq_list.horizontalScrollBar() if hasattr(self, "seq_list") else None
        # Read sequence display configuration to determine whether scrolling
        # should be synchronized across all rows or independent per row.
        seq_cfg = self.sequence.sequence_config()
        raw_independent = seq_cfg.get("independent_scroll", False)
        # Only treat a real boolean True as enabling independent scrolling;
        # avoid truthiness of strings like "False".
        independent = bool(raw_independent) if isinstance(raw_independent, bool) else False

        if master is None or content_bar is None:
            if master is not None:
                master.blockSignals(True)
                master.setRange(0, 0)
                master.setPageStep(0)
                master.setValue(0)
                master.setEnabled(False)
                master.blockSignals(False)
            return

        # When independent scrolling is enabled, hide/disable the shared
        # master scrollbar and allow each row's own horizontal scrollbar to
        # operate normally.
        if independent:
            try:
                master.blockSignals(True)
                master.setRange(0, 0)
                master.setPageStep(0)
                master.setValue(0)
                master.setEnabled(False)
                master.blockSignals(False)
            except Exception:
                pass

            # Enable individual horizontal scrollbars for all sequence lists.
            try:
                if hasattr(self, "seq_list") and self.seq_list is not None:
                    self.seq_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            except Exception:
                pass
            try:
                if hasattr(self, "seq_numbers_list") and self.seq_numbers_list is not None:
                    self.seq_numbers_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            except Exception:
                pass
            try:
                for row in getattr(self, "_sequence_rows", {}).values():
                    lst = row.get("list")
                    if isinstance(lst, QtWidgets.QListWidget):
                        lst.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            except Exception:
                pass

            # No shared-scroll wiring in this mode.
            return

        # Synchronized scrolling mode (default): ensure per-row scrollbars are
        # hidden and driven by the shared master scrollbar.
        try:
            if hasattr(self, "seq_list") and self.seq_list is not None:
                self.seq_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        except Exception:
            pass
        try:
            if hasattr(self, "seq_numbers_list") and self.seq_numbers_list is not None:
                self.seq_numbers_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        except Exception:
            pass
        try:
            for row in getattr(self, "_sequence_rows", {}).values():
                lst = row.get("list")
                if isinstance(lst, QtWidgets.QListWidget):
                    lst.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        except Exception:
            pass
        self._scroll_master = master
        master.valueChanged.connect(self._on_master_scroll_changed)
        self._content_scrollbar = content_bar
        content_bar.rangeChanged.connect(self._on_content_scroll_range_changed)

        targets: list[QtWidgets.QScrollBar] = []
        targets.append(content_bar)
        if hasattr(self, "seq_numbers_list") and self.seq_numbers_list is not None:
            bar = self.seq_numbers_list.horizontalScrollBar()
            if bar is not None and bar is not master:
                targets.append(bar)

        for row in self._sequence_rows.values():
            lst = row.get("list")
            if isinstance(lst, QtWidgets.QListWidget):
                bar = lst.horizontalScrollBar()
                if bar is not None and bar is not master:
                    targets.append(bar)

        self._scroll_targets = targets
        value = content_bar.value()
        for bar in self._scroll_targets:
            bar.setValue(value)
            bar.valueChanged.connect(self._on_target_scroll_changed)

        self._update_shared_scrollbar_range()

    def _on_master_scroll_changed(self, value: int) -> None:
        if self._scroll_updating:
            return
        self._scroll_updating = True
        try:
            for bar in self._scroll_targets:
                if bar.value() != value:
                    bar.setValue(value)
        finally:
            self._scroll_updating = False

    def _on_target_scroll_changed(self, value: int) -> None:
        if self._scroll_updating:
            return
        if self._scroll_master is None:
            return

        # When any target scrollbar moves (including the main sequence row or
        # additional rows), drive the shared master scrollbar and explicitly
        # synchronize all other targets. We do this here instead of relying on
        # _on_master_scroll_changed (which is skipped while _scroll_updating
        # is True) so that scrolling inside any row keeps all rows aligned.
        self._scroll_updating = True
        try:
            if self._scroll_master.value() != value:
                self._scroll_master.setValue(value)
            for bar in self._scroll_targets:
                try:
                    if bar.value() != value:
                        bar.setValue(value)
                except Exception:
                    pass
        finally:
            self._scroll_updating = False

    def _on_content_scroll_range_changed(self, minimum: int, maximum: int) -> None:
        self._update_shared_scrollbar_range()

    def _update_shared_scrollbar_range(self) -> None:
        master = getattr(self, "seq_scrollbar", None)
        content = getattr(self, "_content_scrollbar", None)
        if master is None:
            return
        if content is None:
            master.blockSignals(True)
            master.setRange(0, 0)
            master.setPageStep(0)
            master.setValue(0)
            master.setEnabled(False)
            master.blockSignals(False)
            return
        master.blockSignals(True)
        master.setRange(content.minimum(), content.maximum())
        master.setPageStep(content.pageStep())
        master.setSingleStep(max(1, content.singleStep()))
        master.setEnabled(content.maximum() > content.minimum())
        master.setValue(content.value())
        master.blockSignals(False)

    def _apply_sequence_selection_styles(self, selected_rows: set[int]) -> None:
        if self.seq_list.count() == 0:
            return
        seq_cfg = self.sequence.sequence_config()
        sel_bg = self.sequence.color_from_rgba(
            seq_cfg.get("selection_color", [1.0, 0.95, 0.4, 1.0]),
            [1.0, 0.95, 0.4, 1.0],
        )
        sel_fg = self.sequence.color_from_rgba(
            seq_cfg.get("selection_text_color", [0.1, 0.1, 0.1, 1.0]),
            [0.1, 0.1, 0.1, 1.0],
        )
        for row in range(self.seq_list.count()):
            item = self.seq_list.item(row)
            if item is None:
                continue
            palette = item.data(_SEQ_COLOR_ROLE)
            if isinstance(palette, tuple) and len(palette) == 2:
                base_bg, base_fg = palette
            else:
                base_bg = QtGui.QColor(220, 220, 200)
                base_fg = QtGui.QColor(0, 0, 0)
            seq_idx = item.data(_SEQ_INDEX_ROLE)
            if not getattr(self, "_sequence_visible", True):
                gray_bg = QtGui.QColor(200, 200, 200)
                gray_fg = QtGui.QColor(140, 140, 140)
                item.setBackground(QtGui.QBrush(gray_bg))
                item.setForeground(QtGui.QBrush(gray_fg))
            elif isinstance(seq_idx, int) and seq_idx >= 0 and seq_idx in selected_rows:
                item.setBackground(QtGui.QBrush(sel_bg))
                item.setForeground(QtGui.QBrush(sel_fg))
            else:
                item.setBackground(QtGui.QBrush(base_bg))
                item.setForeground(QtGui.QBrush(base_fg))

    def _apply_representation_to_selection(self, cartoon=None, ball=None) -> None:
        idx = self._selected_residue_indices()
        if not idx:
            return
        try:
            self.viewer.set_residue_representation(idx, cartoon=cartoon, ball=ball)
        except Exception as e:
            try:
                cs.logging.warning(
                    "MolViewPluginWindow._apply_representation_to_selection failed: %s",
                    e,
                )
            except Exception:
                pass

    def _get_secondary_structure_codes(self, object_id: Optional[str], n_res: int) -> list[str] | None:
        """Return a list of secondary-structure codes (H/E/C) for residues.

        Delegates to :func:`assign_ss_c3_from_file` in :mod:`ss`, which
        implements a simplified DSSP-style assignment inspired by PyDSSP.
        If anything fails, returns ``None`` and the sequence view falls back
        to plain coloring.
        """

        if n_res <= 0 or object_id is None:
            return None
        entry = self._object_store.get(object_id)
        if entry is None:
            return None
        if entry.get("ss_codes") is not None:
            return entry["ss_codes"]

        structure = entry.get("structure")
        atoms = getattr(structure, "atoms", None) if structure is not None else None
        if atoms is None:
            return None

        codes = assign_ss_c3_from_atoms(atoms, n_res)
        if not codes:
            return None
        entry["ss_codes"] = codes
        return codes

if __name__ == "plugin":
    # When launched via the ChiSurf plugin system, __name__ is set to
    # "plugin" and a QApplication is already running.
    window = MolViewPluginWindow()
    window.resize(1000, 700)
    window.show()

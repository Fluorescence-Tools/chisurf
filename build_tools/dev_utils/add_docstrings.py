#!/usr/bin/env python3
"""Add NumPy-style docstrings to functions without them in chisurf/gui/.

For each function without a docstring:
- If the purpose is CLEAR from its name → adds a short NumPy-style docstring.
- If the purpose is NOT clear → adds '# TODO: needs docstring' above the def.

Usage:
    python _add_docstrings.py        # preview (dry-run)
    python _add_docstrings.py apply  # apply changes
"""
import ast
import os
import sys
import re
import tokenize
import io

# Resolve repo root: script at build_tools/dev_utils/add_docstrings.py, go up 2 levels
GUI_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chisurf", "gui")
SKIP_DIRS = {"__pycache__", ".git", "resources"}

# Map of function name patterns -> docstring templates
CLEAR_PURPOSE = {
    "__init__": "Initialize the {class_name} instance.",
    "__enter__": "Enter the runtime context.",
    "__exit__": "Exit the runtime context.",
    "__str__": "Return a string representation of this {class_name}.",
    "__call__": "Call this {class_name} instance.",
    "__post_init__": "Post-init processing for the dataclass.",
    "__del__": "Clean up resources on deletion.",
    "__getattr__": "Get an attribute value.",
    "__setattr__": "Set an attribute value.",

    # Qt event handlers
    "closeEvent": "Handle the close event for this widget.",
    "showEvent": "Handle the show event for this widget.",
    "resizeEvent": "Handle the resize event for this widget.",
    "paintEvent": "Paint the widget content.",
    "mousePressEvent": "Handle mouse press events.",
    "mouseReleaseEvent": "Handle mouse release events.",
    "mouseMoveEvent": "Handle mouse move events.",
    "mouseDoubleClickEvent": "Handle mouse double-click events.",
    "wheelEvent": "Handle mouse wheel events.",
    "keyPressEvent": "Handle key press events.",
    "focusOutEvent": "Handle focus-out events.",
    "enterEvent": "Handle mouse enter events.",
    "leaveEvent": "Handle mouse leave events.",
    "hideEvent": "Handle widget hide events.",
    "contextMenuEvent": "Handle context menu events.",
    "eventFilter": "Filter events for installed objects.",
    "sizeHint": "Return the recommended size for the widget.",
    "dragEnterEvent": "Accept drag-enter events carrying file URLs.",
    "dragMoveEvent": "Handle drag-move events.",
    "dropEvent": "Handle drop events with file URLs.",
    "startDrag": "Start a drag operation.",
    "setDragEnabled": "Enable or disable drag-and-drop.",

    # View helpers
    "update": "Update the widget or its contents.",
    "updateUI": "Update the UI elements from the current state.",
    "clear": "Clear all contents.",

    # History / state
    "undo": "Undo the last operation.",
    "redo": "Redo the last undone operation.",
    "can_undo": "Check whether undo is available.",
    "can_redo": "Check whether redo is available.",
    "undo_step": "Perform a single undo step.",
    "redo_step": "Perform a single redo step.",
    "move_cursor": "Move the cursor to a specific index.",
    "get_ui_state": "Get the current UI state.",
    "set_ui_state": "Set the UI state.",

    # validate/fixup
    "validate": "Validate the input and return a corrected value.",
    "fixup": "Fix up the input string to a valid form.",

    # conversion
    "to_dict": "Convert to a dictionary.",

    # standard
    "get_data": "Get the data object.",
    "get_data_sets": "Get the data sets.",
    "get_settings": "Get the current settings.",
    "get_edited_data": "Get the edited data.",
    "get_microtime_ranges": "Get the microtime ranges.",
    "reload": "Reload the data.",
    "stop": "Stop the operation.",
    "start": "Start the operation.",
    "cancel": "Cancel the operation.",
    "qInitResources": "Initialize Qt resources.",
    "qCleanupResources": "Clean up Qt resources.",
    "setup_connections": "Set up signal/slot connections.",
    "setup_ui": "Set up the UI elements.",
    "check_cancel": "Check if the operation was cancelled.",
    "fin": "Finalize or finish the operation.",

    # plot related
    "update_plots": "Update all plots.",
    "update_plot": "Update the plot.",
    "update_path": "Update the path.",
    "update_parameter": "Update the parameter.",
    "update_output_path": "Update the output path.",
    "update_geometry": "Update the geometry.",

    # main
    "updateUI": "Update the UI elements from the current state.",
    "init_setups": "Initialize experiment setups from configuration.",
    "init_console": "Initialize the IPython console.",
    "define_actions": "Define and connect all GUI actions.",
    "arrange_widgets": "Arrange dock widgets and window layout.",
    "load_tools": "Load tools and toolbar plugins.",
    "warmup_imports": "Preload heavy modules to improve first-use responsiveness.",
    "upgrade": "Upgrade or update the component.",

    # Regex patterns for various prefixes (checked AFTER exact matches)
}

# These are checked in order. All keys containing regex chars are treated as regex.
PATTERNS = [
    (r"^current_", "Get the current {short_name}."),
    (r"^_current_", "Get the current {short_name}."),
    (r"^set[A-Z]", "Set the {short_name}."),
    (r"^get[A-Z]", "Get the {short_name}."),
    (r"^on[A-Z]", "Handle the {short_name} event."),
    (r"^_on[A-Z]", "Handle the {short_name} event internally."),
    (r"^load[A-Z]", "Load {short_name}."),
    (r"^save[A-Z]", "Save {short_name}."),
    (r"^read[A-Z]", "Read {short_name}."),
    (r"^write[A-Z]", "Write {short_name}."),
    (r"^to_", "Convert to {short_name}."),
    (r"^from_", "Create from {short_name}."),
    (r"^is_", "Check whether {short_name}."),
    (r"^has_", "Check whether {short_name}."),
    (r"^_load_", "Load {short_name}."),
    (r"^_run_", "Run {short_name}."),
    (r"^_set_", "Set {short_name}."),
    (r"^_get_", "Get {short_name}."),
    (r"^_start_", "Start {short_name}."),
    (r"^_stop_", "Stop {short_name}."),
    (r"^_schedule_", "Schedule {short_name}."),
    (r"^_ensure_", "Ensure {short_name}."),
    (r"^_populate_", "Populate {short_name}."),
    (r"^_convert_", "Convert {short_name}."),
    (r"^_merge_", "Merge {short_name}."),
    (r"^_install_", "Install {short_name}."),
    (r"^_split_", "Split {short_name}."),
    (r"^_append_", "Append {short_name}."),
    (r"^_insert_", "Insert {short_name}."),
    (r"^_move_", "Move {short_name}."),
    (r"^_duplicate_", "Duplicate {short_name}."),
    (r"^_extract_", "Extract {short_name}."),
    (r"^_register_", "Register {short_name}."),
    (r"^_trace_", "Trace {short_name}."),
    (r"^_record_", "Record {short_name}."),
    (r"^_edit_", "Edit {short_name}."),
    (r"^_restore_", "Restore {short_name}."),
    (r"^_link_", "Handle link {short_name}."),
    (r"^_unlink_", "Unlink {short_name}."),
    (r"^_highlight_", "Highlight {short_name}."),
    (r"^_layout_", "Layout {short_name}."),
    (r"^_exec_", "Execute {short_name}."),
    (r"^_redraw_", "Redraw {short_name}."),
    (r"^_reload_", "Reload {short_name}."),
    (r"^_reinit_", "Reinitialize {short_name}."),
    (r"^_finali", "Finalize {short_name}."),
    (r"^_refresh_", "Refresh {short_name}."),
    (r"^_update_", "Update {short_name}."),
    (r"^_parse_", "Parse {short_name}."),
    (r"^_format_", "Format {short_name}."),
    (r"^_apply_", "Apply {short_name}."),
    (r"^_read_", "Read {short_name}."),
    (r"^_write_", "Write {short_name}."),
    (r"^_sync_", "Synchronize {short_name}."),
    (r"^_show_", "Show {short_name}."),
    (r"^_init_", "Initialize {short_name}."),
    (r"^_build_", "Build {short_name}."),
    (r"^_create_", "Create {short_name}."),
    (r"^_add_", "Add {short_name}."),
    (r"^_remove_", "Remove {short_name}."),
    (r"^_clear_", "Clear {short_name}."),
    (r"^_open_", "Open {short_name}."),
    (r"^_close_", "Close {short_name}."),
    (r"^_compute_", "Compute {short_name}."),
    (r"^_calc_", "Calculate {short_name}."),
    (r"^_find_", "Find {short_name}."),
    (r"^_resolve_", "Resolve {short_name}."),
    (r"^_copy_", "Copy {short_name}."),
    (r"^_watch_", "Watch {short_name}."),

    # Methods that don't start with underscore but are common
    (r"^load[A-Z]", "Load {short_name}."),
    (r"^save[A-Z]", "Save {short_name}."),
    (r"^read[A-Z]", "Read {short_name}."),
    (r"^write[A-Z]", "Write {short_name}."),
    (r"^on[A-Z]_", "Handle the {short_name} event."),

    # Property-like names that are single camelCase words
    (r"^[a-z]+[A-Z]", "Get or set the {short_name}."),

    # Closures with recognizable purposes
    (r"^_key$", "Generate a normalized key for deduplication."),
    (r"^_fn$", "Execute the deferred function."),
    (r"^_prog$", "Progress reporter callback."),
    (r"^wrapper$", "Wrapper function."),
    (r"^slot$", "Slot function connected to a Qt signal."),
    (r"^resolver$", "Resolve an identifier to its object."),
    (r"^format_key$", "Format a parameter key as a string."),
    (r"^resolve_param$", "Resolve a parameter by its key."),
    (r"^read_module_docstring$", "Read the module docstring from an __init__.py."),
    (r"^pick$", "Select or pick an item."),
    (r"^parse_mtr$", "Parse microtime range values."),

    # Remaining common names
    (r"^get_fortune$", "Get a random fortune message."),
    (r"^get_win$", "Get the main application window."),
    (r"^get_app$", "Get the QApplication instance."),
    (r"^set_app_style$", "Set the application style."),
    (r"^create_plots$", "Create the plot widgets."),
    (r"^place_plots$", "Place and arrange the plot widgets."),
    (r"^install_file_drop$", "Install file-drop handling on a widget."),
    (r"^install_filter_mode_visibility$", "Set up filter mode visibility toggling."),
    (r"^update_parameter_visibility$", "Update parameter visibility based on mode."),
    (r"^get_mdi_components$", "Get the MDI subwindow components."),
    (r"^on_calc_g_factor$", "Handle the calculate G-factor action."),
    (r"^custom_close_event$", "Custom close event handler."),
    (r"^qtpy_loadUi$", "Load a UI file using qtpy."),
    (r"^apply_from_plugin$", "Apply settings from a plugin."),
    (r"^apply_alignment_fix$", "Apply alignment fix to the ribbon."),
    (r"^noop_mousePressEvent$", "No-op mouse press event handler."),
    (r"^noop_mouseMoveEvent$", "No-op mouse move event handler."),
    (r"^noop_mouseDoubleClickEvent$", "No-op mouse double-click event handler."),
    (r"^getEnergy$", "Get the energy value."),
    (r"^to_dict$", "Convert to a dictionary."),
    (r"^from_dict$", "Create from a dictionary."),
    (r"^main$", "Main entry point."),
    (r"^onRegionUpdate", "Handle the region update event."),
    (r"^onUpdatePhasor$", "Update the phasor plot."),
    (r"^onParametersChanged$", "Handle parameter changes."),
    (r"^onSampleChanged$", "Handle sample changes."),
    (r"^onItemChanged$", "Handle item changes."),
    (r"^onCurveChanged$", "Handle curve changes."),
    (r"^onAddFit$", "Handle add fit action."),
    (r"^change_event$", "Handle selection change events."),
    (r"^handleSelectionChange$", "Handle selection changes in the widget."),
    (r"^onChangeCurveName$", "Handle curve name changes."),
    (r"^onRemoveDataset$", "Handle dataset removal."),
    (r"^onSaveDataset$", "Handle dataset saving."),
    (r"^onLoadDataset$", "Handle dataset loading."),
    (r"^onGroupDatasets$", "Handle dataset grouping."),
    (r"^onUnGroupDatasets$", "Handle dataset ungrouping."),
    (r"^onRemoveFit$", "Handle fit removal."),
    (r"^onLoadFit$", "Handle fit loading."),
    (r"^onSaveFit$", "Handle fit saving."),
    (r"^show_selector$", "Show the fit selector."),
    (r"^onDatasetChanged$", "Handle dataset changes."),
    (r"^onErrorEstimate$", "Handle error estimation."),
    (r"^onRunFit$", "Handle the run fit action."),
    (r"^onFitRangeChanged$", "Handle fit range changes."),
    (r"^onAutoFitRange$", "Handle auto fit range."),
    (r"^on_change_plot$", "Handle plot changes."),
    (r"^updateStatusBar$", "Update the status bar."),
    (r"^on_reader_help_clicked$", "Handle the reader help button click."),
    (r"^event$", "Handle Qt events."),
    (r"^center$", "Center the widget on the screen."),
    (r"^make_group$", "Create a group of widgets."),
    (r"^_open_in_editor$", "Open a file in the code editor."),
    (r"^_find_main_window$", "Find the main application window."),
    (r"^_show_no_target_message$", "Show a message when no target is found."),
    (r"^_show_error_message$", "Show an error message."),
    (r"^_is_incomplete_start_event$", "Check if event is an incomplete start event."),
    (r"^_is_state_action$", "Check if event is a state action."),
    (r"^_log_info$", "Log an informational message."),
    (r"^_log_code$", "Log executed code."),
    (r"^_log_from_signal$", "Log from a signal."),
    (r"^log_on_gui_thread$", "Log a message on the GUI thread."),
    (r"^_apply_filter$", "Apply a filter to the widget."),
    (r"^_show_selected_details$", "Show details of the selected item."),
    (r"^_on_event_recorded$", "Handle the event recorded signal."),
    (r"^_on_item_clicked$", "Handle item click."),
    (r"^_on_clear_clicked$", "Handle clear button click."),
    (r"^_on_save$", "Handle save action."),
    (r"^_on_editing_finished$", "Handle editing finished."),
    (r"^_on_editing_started$", "Handle editing started."),
    (r"^_on_fix_toggled$", "Handle fix toggle."),
    (r"^_on_value_changed$", "Handle value changes."),
    (r"^_on_text_changed$", "Handle text changes."),
    (r"^_on_combo_edit_start$", "Handle combo edit start."),
    (r"^_on_combo_edit_finish$", "Handle combo edit finish."),
    (r"^_on_slider_edit_start$", "Handle slider edit start."),
    (r"^_on_slider_edit_finish$", "Handle slider edit finish."),
    (r"^sync_from_slider$", "Sync value from slider."),
    (r"^sync_from_spin$", "Sync value from spinbox."),
    (r"^_on_changed$", "Handle changed signal."),
    (r"^_on_committed$", "Handle committed signal."),
    (r"^_on_calc_g_factor$", "Handle calculate G-factor."),
    (r"^_compute_required_width$", "Compute the required width for display."),
    (r"^_on_stack_index_changed$", "Handle stack index changes."),
    (r"^_on_stack_clean_changed$", "Handle stack clean state changes."),
    (r"^_jump_to_index$", "Jump to a specific index."),
    (r"^name_taken$", "Check if a name is already taken."),
    (r"^_default_title_for_type$", "Get the default title for a node type."),
    (r"^_build_example_graph$", "Build an example graph."),
    (r"^_on_undo_shortcut$", "Handle undo shortcut."),
    (r"^_on_redo_shortcut$", "Handle redo shortcut."),
    (r"^_on_load_json_clicked$", "Handle load JSON click."),
    (r"^_on_save_json_clicked$", "Handle save JSON click."),
    (r"^_safe_version$", "Safely get version string from a module."),
    (r"^_build_folded_text$", "Build the folded text for display."),
    (r"^_overlaps_any$", "Check if an item overlaps any existing item."),
    (r"^_on_compute_pch_clicked$", "Handle compute PCH button click."),
    (r"^_on_add_pch_clicked$", "Handle add PCH button click."),
    (r"^_on_clear_preview_clicked$", "Handle clear preview button click."),
    (r"^_load_preview_from_file$", "Load preview from a file."),
    (r"^_load_preview_from_paths$", "Load preview from file paths."),
    (r"^_refresh_preview_plot$", "Refresh the preview plot."),
    (r"^onOpenFile$", "Handle opening a file."),
    (r"^onLoadSample$", "Handle loading a sample."),
    (r"^onLoadStructure$", "Handle loading a structure."),
    (r"^onLoadAvJSON$", "Handle loading an AV JSON file."),
    (r"^onOpenPotentialFile$", "Handle opening a potential file."),
    (r"^onChainChanged$", "Handle chain selection changes."),
    (r"^onResidueChanged$", "Handle residue selection changes."),
    (r"^update_chain$", "Update the chain selection."),
    (r"^calc_rmsd$", "Calculate the RMSD."),
    (r"^setParameterSphere$", "Set the parameter sphere."),
    (r"^setParameterProbe$", "Set the parameter probe."),
    (r"^onParametersChanged$", "Handle parameter changes."),
    (r"^onSetupChanged$", "Handle setup changes."),
    (r"^onExperimentChanged$", "Handle experiment changes."),
    (r"^_on_simulate_clicked$", "Handle simulate button click."),
    (r"^_refresh_simulation_plot$", "Refresh the simulation plot."),
    (r"^_on_preview_mode_changed$", "Handle preview mode changes."),
    (r"^onChangeCsvParameter$", "Handle CSV parameter changes."),
    (r"^_format_dataset_label$", "Format a dataset label."),
    (r"^_update_combo_tooltip$", "Update the combo box tooltip."),
    (r"^_resolve_fit_idx$", "Resolve a fit index."),
    (r"^_run_fit_impl$", "Implementation of the fit run."),
    (r"^_collect_parameter_snapshot$", "Collect a snapshot of parameters."),
    (r"^_collect_fit_range_snapshot$", "Collect a snapshot of fit ranges."),
    (r"^_record_history$", "Record history entry."),
    (r"^_result_changed$", "Handle result changes."),
    (r"^_trigger_model_update$", "Trigger a model update."),
    (r"^_update_role_visuals$", "Update role visual indicators."),
    (r"^_locate_parameter$", "Locate a parameter by name."),
    (r"^_locate_parameter_uids$", "Locate parameters by UID."),
    (r"^_parameter_context$", "Get the parameter context."),
    (r"^_trace_operation$", "Trace an operation for logging."),
    (r"^_build_details_tooltip_text$", "Build tooltip text for parameter details."),
    (r"^_open_details_popup$", "Open a details popup for a parameter."),
    (r"^_on_error_estimate_clicked$", "Handle error estimate button click."),
    (r"^_on_label_mouse_press$", "Handle mouse press on a label."),
    (r"^_on_main_value_changed$", "Handle main value changes."),
    (r"^_on_main_bounds_on_toggled$", "Handle bounds toggle."),
    (r"^_on_main_fixed_toggled$", "Handle fixed toggle."),
    (r"^_on_main_bounds_changed$", "Handle bounds changes."),
    (r"^make_linkcall$", "Create a link callback."),
    (r"^build_link_menu$", "Build a link selection menu."),
    (r"^onLinkFitGroup$", "Handle linking a fit group."),
    (r"^_set_obj$", "Set the wrapped object."),
    (r"^_safe_float$", "Safely convert a value to float."),
    (r"^_sync_l2_from_l1$", "Synchronize L2 channel from L1."),
    (r"^_update_l2_enabled_state$", "Update L2 channel enabled state."),
    (r"^_on_link_l1_l2_toggled$", "Handle L1-L2 link toggle."),
    (r"^_create_next_fit$", "Create the next fit in the queue."),
    (r"^_startup_update_check$", "Check for updates on startup."),
    (r"^_find_free_port$", "Find a free TCP port."),
    (r"^_reader$", "Reader thread for Jupyter output."),
    (r"^_warmup$", "Warm-up callback for background stages."),
    (r"^_on_bg_complete$", "Complete callback for background stages."),
    (r"^_should_open_onboarding$", "Check if onboarding should open."),
    (r"^_open_onboarding$", "Open the onboarding wizard."),
    (r"^gui_imports$", "Phase 1: import GUI submodules."),
    (r"^deferred_gui_imports$", "Phase 2: import heavy functional submodules."),
    (r"^setup_ipython$", "Set up the IPython console widget."),
    (r"^startup_interface$", "Create and return the main window."),
    (r"^setup_style$", "Set up the application style sheet."),
    (r"^_apply_stylesheet$", "Apply a style sheet to the application."),
    (r"^_resolve_style_path$", "Resolve a style file path."),
    (r"^_apply_style_by_name$", "Apply a style by its name."),
    (r"^populate_plugins$", "Populate the Plugins menu."),
    (r"^populate_notebooks$", "Populate the Notebooks menu."),
    (r"^add_notebook$", "Add a notebook file to the menu."),
    (r"^setup_logging_widgets$", "Set up logging handlers for GUI widgets."),
    (r"^setup_log_list_widget$", "Replace the log widget with a LogListWidget."),
    (r"^filter_log_content$", "Filter log entries based on filter text."),
    (r"^update_log_filter$", "Update log filtering when new entries are added."),
    (r"^_format_message$", "Format a message with optional truncation."),
    (r"^_recent_projects_file$", "Get the path to the recent projects file."),
    (r"^setMaxMessageLength$", "Set the maximum message length for truncation."),
    (r"^showMessage$", "Show a formatted message on the status bar."),
    (r"^_normalize_values$", "Normalize numeric values."),
    (r"^_format_values$", "Format numeric values for display."),
    (r"^setValue$", "Set the value."),
    (r"^setText$", "Set the text."),
    (r"^setLabelText$", "Set the label text."),
    (r"^set_title$", "Set the title."),
    (r"^set_data$", "Set the data."),
    (r"^set_collapsed", "Set the collapsed state."),
    (r"^toggle_collapsed", "Toggle the collapsed state."),
    (r"^set_connected$", "Set the connected state."),
    (r"^set_fixed$", "Set the fixed state."),
    (r"^set_end_port$", "Set the end port for an edge."),
    (r"^set_temp_end_pos$", "Set a temporary end position."),
    (r"^update_bounds$", "Update the bounding rectangle."),
    (r"^set_background_pattern_enabled$", "Enable or disable the background pattern."),
    (r"^set_background_pattern_step$", "Set the background pattern step size."),
    (r"^background_pattern_enabled$", "Check if background pattern is enabled."),
    (r"^register_edge$", "Register an edge in the scene."),
    (r"^register_overlay_item$", "Register an overlay item in the scene."),
    (r"^update_edges_for_port$", "Update edges connected to a port."),
    (r"^_create_node_item_from_model$", "Create a node item from a model."),
    (r"^_duplicate_node$", "Duplicate a node."),
    (r"^_create_node_by_type$", "Create a node by its type name."),
    (r"^_build_nx_graphs$", "Build NetworkX graphs for topological analysis."),
    (r"^is_directed_acyclic$", "Check if the graph is a directed acyclic graph."),
    (r"^update_cycle_highlighting$", "Update cycle highlighting on nodes."),
    (r"^_port_to_entry$", "Convert a port to an entry dict."),
    (r"^_parse_port$", "Parse a port definition."),
    (r"^available_node_types$", "Get available node types."),
    (r"^clear_graph$", "Clear the graph."),
    (r"^_on_item_activated$", "Handle item activation."),
    (r"^set_show_moves$", "Set whether moves are shown."),
    (r"^set_show_folds$", "Set whether folds are shown."),
    (r"^set_undo_stack$", "Set the undo stack to track."),
    (r"^_MoveNodesCommand$", "..."),
    (r"^_GraphStateCommand$", "..."),
    (r"^_parse_scalar$", "Parse a scalar value from a string."),
    (r"^_find_top_level_flags_heading$", "Find the top-level flags heading in settings."),
    (r"^_is_theme_setting$", "Check if a setting is a theme option."),
    (r"^_create_theme_editor$", "Create a theme editor widget."),
    (r"^setOpacity$", "Set the opacity level."),
    (r"^_setup_ui$", "Set up the UI elements."),
    (r"^_update_visibility$", "Update widget visibility."),
    (r"^_on_clicked$", "Handle click events."),
    (r"^_init_history_browser$", "Initialize the history browser widget."),
    (r"^_focus_widget_has_native_undo_redo$", "Check if focused widget has native undo/redo."),
    (r"^_history_undo$", "Perform an undo through the history browser."),
    (r"^_history_redo$", "Perform a redo through the history browser."),
    (r"^_sync_history_navigation_actions$", "Sync undo/redo action enabled states."),
    (r"^_refresh_parameter_widgets$", "Refresh all parameter controllers."),
    (r"^_refresh_plots$", "Refresh all fit plots."),
    (r"^_on_history_cursor_changed$", "Handle history cursor changes."),
    (r"^_select_dataset_by_identity$", "Select a dataset by its identity."),
    (r"^_select_fit_by_identity$", "Select a fit by its identity."),
    (r"^_init_system_info_watermark$", "Initialize the system info watermark."),
    (r"^_update_system_info_watermark_geometry$", "Update watermark label geometry."),
    (r"^_ensure_initial_plot$", "Ensure the initial plot is created."),
    (r"^_sync_fit_widget_range$", "Sync the fit widget range with the fit."),
    (r"^ensure_plot_created$", "Ensure the plot widget is created."),
    (r"^show_selector$", "Show the fit selector."),
    (r"^_prog$", "Progress callback function."),
    (r"^_on_progress$", "Handle progress updates."),
    (r"^_on_finished$", "Handle completion."),
    (r"^_on_cancel$", "Handle cancellation."),
    (r"^update_new$", "Update with new data."),
    (r"^linkcall$", "Create a link callback callable."),
    (r"^fmt$", "Format a value."),
    (r"^selected_fit$", "Get the currently selected fit."),
    (r"^local_first$", "Get whether local fitting runs first."),
    (r"^n_steps$", "Get the number of fitting steps."),
    (r"^n_runs$", "Get the number of fitting runs."),
    (r"^_format_dataset_label$", "Format a label for a dataset."),
    (r"^_update_combo_tooltip$", "Update the combo box tooltip."),
    (r"^change_dataset$", "Change the active dataset."),
    (r"^show_selector$", "Show the fit selector widget."),
    (r"^selected_fit_idx$", "Get the selected fit index."),
    (r"^selected_dataset_idx$", "Get the selected dataset index."),
    (r"^selected_dataset_indices$", "Get the selected dataset indices."),
    (r"^xmin$", "Get the minimum fit range."),
    (r"^xmax$", "Get the maximum fit range."),
    (r"^xmin2$", "Get the secondary minimum fit range."),
    (r"^xmax2$", "Get the secondary maximum fit range."),
    (r"^current_fit_type$", "Get the current fit type."),
    (r"^onItemChanged$", "Handle item changes in the list."),
    (r"^keyPressEvent$", "Handle key press events."),
    (r"^onCurveChanged$", "Handle curve changes."),
    (r"^onChangeCurveName$", "Handle curve name changes."),
    (r"^onRemoveFit$", "Handle fit removal."),
    (r"^onSaveFit$", "Handle fit saving."),
    (r"^on_reader_help_clicked$", "Handle reader help button click."),
    (r"^event$", "Handle Qt events."),
    (r"^make_group$", "Create a group of widgets."),
    (r"^_open_in_editor$", "Open the file in the code editor."),
    (r"^_find_main_window$", "Find the main window instance."),
    (r"^_show_no_target_message$", "Show a message when no target is found."),
    (r"^_show_error_message$", "Show an error message."),
    (r"^_on_item_activated$", "Handle item activation."),
    (r"^_on_clear_clicked$", "Handle clear button clicks."),
    (r"^_on_save$", "Handle save action."),
    (r"^_on_editing_finished$", "Handle editing finished signal."),
    (r"^_on_editing_started$", "Handle editing started signal."),
    (r"^_on_value_changed$", "Handle value changed signal."),
    (r"^_on_text_changed$", "Handle text changed signal."),
    (r"^_on_fix_toggled$", "Handle fix toggle."),
    (r"^sync_from_slider$", "Sync the value from a slider."),
    (r"^sync_from_spin$", "Sync the value from a spinbox."),
    (r"^set_linked$", "Set the linked state."),
    (r"^_resolve_fit_idx$", "Resolve the fit index."),
    (r"^_on_region_change_finished$", "Handle region change completion."),
]

DEFAULT_DOCSTRING = "# TODO: needs docstring"


def _camel_to_words(name):
    """Convert camelCase to words."""
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    parts = name.replace('_', ' ').split()
    return parts[-1].lower() if parts else name


def _get_short_name(func_name, parent_class):
    """Derive a short human-readable name from function name + class."""
    if func_name.startswith("_") and not func_name.startswith("__"):
        name = func_name[1:]
    else:
        name = func_name
    if name.startswith("on") and len(name) > 2 and name[2].isupper():
        return _camel_to_words(name[2:])
    if name.startswith("set") and len(name) > 3 and name[3].isupper():
        return _camel_to_words(name[3:])
    if name.startswith("get") and len(name) > 3 and name[4].isupper():
        return _camel_to_words(name[4:])
    if name.startswith("load") and len(name) > 4 and name[4].isupper():
        return _camel_to_words(name[4:])
    if name.startswith("save") and len(name) > 4 and name[4].isupper():
        return _camel_to_words(name[4:])
    return _camel_to_words(name)


def _make_docstring(func_name, parent_class):
    """Try to generate a docstring. If not possible, return None."""
    # Exact matches first
    if func_name in CLEAR_PURPOSE:
        template = CLEAR_PURPOSE[func_name]
        sn = _get_short_name(func_name, parent_class)
        cls = parent_class or "object"
        return template.format(class_name=cls, short_name=sn)

    # Regex patterns
    for pattern, template in PATTERNS:
        try:
            if re.match(pattern, func_name):
                sn = _get_short_name(func_name, parent_class)
                cls = parent_class or "object"
                return template.format(class_name=cls, short_name=sn)
        except re.error:
            continue

    return None


def _get_indent(lines, lineno):
    """Get the leading whitespace of the line at lineno (1-indexed)."""
    line = lines[lineno - 1]
    m = re.match(r'^(\s*)', line)
    return m.group(1) if m else ""


def _insert_docstring(lines, func_def, doc_text):
    """Insert docstring text into lines for the given function definition node."""
    lineno = func_def.lineno
    end_lineno = func_def.end_lineno or lineno

    # The function body starts after the signature
    # Find the ':' at end of signature, then advance to body
    body_lineno = None

    # If there's a body, use the first body node's lineno
    if func_def.body:
        body_lineno = func_def.body[0].lineno
    else:
        # No body - shouldn't happen for real functions
        body_lineno = end_lineno + 1

    if body_lineno is None or body_lineno > len(lines):
        return False, "body line out of range"

    base_indent = _get_indent(lines, lineno)
    body_indent = _get_indent(lines, body_lineno)

    # Insert docstring before the first body line
    idx = body_lineno - 1

    if doc_text == DEFAULT_DOCSTRING:
        lines.insert(idx, f"{base_indent}# TODO: needs docstring\n")
    else:
        lines.insert(idx, f'{body_indent}"""{doc_text}"""\n')

    return True, ""


def process_file(filepath, dry_run=True):
    """Process a single Python file."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return 0, str(e)

    # Collect all functions without docstrings
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node):
                continue

            # Find parent class
            parent_class = None
            for n2 in ast.walk(tree):
                if isinstance(n2, ast.ClassDef):
                    if any(item is node for item in n2.body):
                        parent_class = n2.name
                        break
                    # Check nested functions
                    for item in ast.walk(n2):
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if any(sub is node for sub in ast.walk(item)):
                                if parent_class is None:
                                    parent_class = n2.name
                                break

            doc = _make_docstring(node.name, parent_class)
            if doc is None:
                doc = DEFAULT_DOCSTRING

            edits.append((node, doc))

    if not edits:
        return 0, ""

    lines = source.splitlines(True)
    # Apply in reverse line order
    for func_def, doc in sorted(edits, key=lambda x: -x[0].lineno):
        ok, err = _insert_docstring(lines, func_def, doc)
        if not ok:
            return 0, err

    new_source = "".join(lines)

    if not dry_run:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_source)

    return len(edits), ""


def main():
    dry_run = len(sys.argv) < 2 or sys.argv[1] != "apply"

    total_processed = 0
    total_files = 0
    total_errors = 0
    files_with_todos = []

    for dirpath, dirnames, filenames in os.walk(GUI_DIR):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, os.path.dirname(GUI_DIR))

            count, err = process_file(fp, dry_run=dry_run)
            if err:
                print(f"  ERROR {rel}: {err}")
                total_errors += 1
            elif count:
                print(f"  {rel}: {count} docstring(s) added")
                total_processed += count
                total_files += 1
                if not dry_run:
                    with open(fp) as f:
                        if "# TODO: needs docstring" in f.read():
                            files_with_todos.append(rel)
            else:
                print(f"  {rel}: nothing to do")

    print(f"\n{'Dry run' if dry_run else 'Applied'} — {total_files} file(s), {total_processed} function(s) processed.")
    if dry_run and total_processed > 0:
        print(f"Run with: python _add_docstrings.py apply")
    if not dry_run and files_with_todos:
        print(f"\nFiles with TODOs ({len(files_with_todos)}):")
        for f in files_with_todos:
            print(f"  {f}")

if __name__ == "__main__":
    main()

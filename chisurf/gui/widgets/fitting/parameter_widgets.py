from __future__ import annotations

import os
import typing
import pathlib
import textwrap

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, uic, QtCore, QtGui
import matplotlib.colors as mcolors

import chisurf.data
import chisurf.fitting
import chisurf.decorators
import chisurf.gui.decorators
import chisurf.settings

import chisurf.gui.widgets
import chisurf.gui.widgets.experiments.widgets
from chisurf.gui.widgets import Controller
from chisurf.math.optimization.leastsqbound import OptimizationCancelled

parameter_settings = chisurf.settings.parameter

class FittingParameterDetailPopup(QtWidgets.QDialog):

    def __init__(self, controller: 'FittingParameterWidget'):
        super().__init__(controller)
        # Use Popup flag so clicks outside cause deactivation; we then hide on focus loss
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Popup)
        self.controller = controller
        self.setObjectName('FittingParameterDetailPopup')
        # Ensure we hide if the window deactivates (extra safety beyond Qt.Popup)
        self.installEventFilter(self)
        # Ensure the popup can take focus and is activated when shown
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, False)
        # Counter to temporarily suspend auto-hide on focus loss (e.g., while link menu is open)
        self._suspend_auto_hide = 0
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        self.lbl_title = QtWidgets.QLabel(f"{controller.fitting_parameter.name}")
        font = self.lbl_title.font()
        font.setBold(True)
        self.lbl_title.setFont(font)
        layout.addWidget(self.lbl_title)

        # Optional human-readable description of the parameter, taken from
        # the underlying Parameter/FittingParameter "description" attribute.
        self.lbl_description = QtWidgets.QLabel("")
        self.lbl_description.setWordWrap(True)
        self.lbl_description.setStyleSheet("color: gray; font-size: 9pt")
        layout.addWidget(self.lbl_description)

        # Link info and actions
        link_row = QtWidgets.QHBoxLayout()
        self.lbl_link = QtWidgets.QLabel("")
        self.btn_change_link = QtWidgets.QToolButton()
        self.btn_change_link.setText("Link…")
        self.btn_unlink = QtWidgets.QToolButton()
        self.btn_unlink.setText("Unlink")
        link_row.addWidget(self.lbl_link, 1)
        link_row.addWidget(self.btn_change_link)
        link_row.addWidget(self.btn_unlink)
        layout.addLayout(link_row)

        # Value editor
        val_row = QtWidgets.QHBoxLayout()
        val_row.addWidget(QtWidgets.QLabel("Value:"))
        self.sb_value = pg.SpinBox(dec=True, decimals=self.controller.widget_value.opts.get('decimals', 6), finite=False)
        val_row.addWidget(self.sb_value)
        layout.addLayout(val_row)

        # Fixed checkbox
        self.cb_fixed = QtWidgets.QCheckBox("Fixed")
        layout.addWidget(self.cb_fixed)

        # Bounds group
        bounds_group = QtWidgets.QGroupBox("Bounds")
        b_layout = QtWidgets.QGridLayout(bounds_group)
        self.cb_bounds_on = QtWidgets.QCheckBox("Enable bounds")
        b_layout.addWidget(self.cb_bounds_on, 0, 0, 1, 2)
        b_layout.addWidget(QtWidgets.QLabel("Lower:"), 1, 0)
        self.sb_lb = pg.SpinBox(dec=True, decimals=self.controller.widget_lower_bound.opts.get('decimals', 6))
        b_layout.addWidget(self.sb_lb, 1, 1)
        b_layout.addWidget(QtWidgets.QLabel("Upper:"), 2, 0)
        self.sb_ub = pg.SpinBox(dec=True, decimals=self.controller.widget_upper_bound.opts.get('decimals', 6))
        b_layout.addWidget(self.sb_ub, 2, 1)
        layout.addWidget(bounds_group)


        # Close hint
        hint = QtWidgets.QLabel("Click outside to close")
        hint.setStyleSheet("color: gray; font-size: 9pt")
        layout.addWidget(hint)

        # Connections
        self.btn_change_link.clicked.connect(self._on_change_link)
        self.btn_unlink.clicked.connect(self._on_unlink)
        self.cb_fixed.toggled.connect(self._on_fixed_toggled)
        self.cb_bounds_on.toggled.connect(self._on_bounds_on_toggled)
        self.sb_lb.editingFinished.connect(self._on_bounds_changed)
        self.sb_ub.editingFinished.connect(self._on_bounds_changed)
        self.sb_value.editingFinished.connect(self._on_value_changed)

        self.refresh_from_model()

    def _begin_suspend_auto_hide(self):
        try:
            self._suspend_auto_hide += 1
        except Exception:
            self._suspend_auto_hide = 1

    def _end_suspend_auto_hide(self):
        try:
            self._suspend_auto_hide -= 1
            if self._suspend_auto_hide < 0:
                self._suspend_auto_hide = 0
        except Exception:
            self._suspend_auto_hide = 0

    def eventFilter(self, obj, event):
        # Hide the popup when it loses focus or the window deactivates, unless suspended
        if event is not None:
            et = int(event.type())
            if et == int(QtCore.QEvent.FocusOut) or et == int(QtCore.QEvent.WindowDeactivate):
                if getattr(self, '_suspend_auto_hide', 0) > 0:
                    # Do not hide; let event pass through
                    return False
                # Use hide (not close) as requested
                self.hide()
                return True
        return super().eventFilter(obj, event)

    def focusOutEvent(self, event: QtGui.QFocusEvent):
        # Extra safety: hide on focus out unless suspended
        try:
            if getattr(self, '_suspend_auto_hide', 0) == 0:
                self.hide()
        finally:
            event.accept()

    def _on_change_link(self):
        menu = self.controller.build_link_menu()
        # Show menu under the button
        pos = self.btn_change_link.mapToGlobal(QtCore.QPoint(0, self.btn_change_link.height()))
        # While the menu is open and linking occurs, do not auto-hide the popup
        self._begin_suspend_auto_hide()
        try:
            menu.exec_(pos)
            # After possible changes, refresh UI/model
            self.controller.finalize()
            self.refresh_from_model()
        finally:
            self._end_suspend_auto_hide()
            # Keep the popup open and focused after linking
            try:
                self.raise_()
                self.activateWindow()
                self.setFocus(QtCore.Qt.PopupFocusReason)
            except Exception:
                pass

    def _on_unlink(self):
        fp = self.controller.fitting_parameter
        chisurf.run(
            f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].link = None\n"
            f"chisurf.fits[{fp.fit_idx}].update()"
        )
        self.controller.finalize()
        self.refresh_from_model()

    def _on_fixed_toggled(self):
        fp = self.controller.fitting_parameter
        chisurf.run(
            f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].fixed = {self.cb_fixed.isChecked()}\n"
            f"chisurf.fits[{fp.fit_idx}].update()"
        )
        self.controller.finalize()

    def _on_bounds_on_toggled(self):
        fp = self.controller.fitting_parameter
        checked = self.cb_bounds_on.isChecked()
        # Toggle bounds_on in the model
        chisurf.run(
            f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].bounds_on = {checked}"
        )
        # Enable/disable editors immediately for better UX
        self.sb_lb.setEnabled(checked)
        self.sb_ub.setEnabled(checked)
        # If turning ON and current bounds are invalid/missing, initialize them from the UI spin boxes
        if checked:
            bounds_valid = False
            try:
                b = getattr(fp, 'bounds', None)
                bounds_valid = isinstance(b, (tuple, list)) and len(b) == 2
            except Exception:
                bounds_valid = False
            if not bounds_valid:
                chisurf.run(
                    f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].bounds = ({self.sb_lb.value()}, {self.sb_ub.value()})"
                )
        # Refresh UI/model without risking unpack errors
        self.controller.finalize()
        self.refresh_from_model()

    def _on_bounds_changed(self):
        fp = self.controller.fitting_parameter
        chisurf.run(
            f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].bounds = ({self.sb_lb.value()}, {self.sb_ub.value()})"
        )
        self.controller.finalize()
        self.refresh_from_model()

    def _on_value_changed(self):
        fp = self.controller.fitting_parameter
        chisurf.run(
            f"parameter = chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}']\n"
            f"fixed = parameter.fixed \n"
            f"parameter.fixed = False\n"
            f"parameter.value = {self.sb_value.value()} \n"
            f"parameter.fixed = fixed\n"
            f"chisurf.fits[{fp.fit_idx}].finalize()"
        )
        self.controller.finalize()

    def refresh_from_model(self):
        fp = self.controller.fitting_parameter
        # Description text (may be empty)
        try:
            desc = getattr(fp, 'description', "")
        except Exception:
            desc = ""
        self.lbl_description.setVisible(bool(desc))
        if desc:
            self.lbl_description.setText(str(desc))
        # Update link label
        if getattr(fp, 'link', None) is not None:
            self.lbl_link.setText(f"Linked to: {fp.link.name}")
            self.btn_unlink.setEnabled(True)
        else:
            self.lbl_link.setText("Not linked")
            self.btn_unlink.setEnabled(False)
        # Value
        try:
            v = float(fp.value)
        except Exception:
            v = self.controller.widget_value.value()
        self.sb_value.setValue(v)
        # Fixed
        self.cb_fixed.blockSignals(True)
        self.cb_fixed.setChecked(bool(fp.fixed))
        self.cb_fixed.blockSignals(False)
        # Bounds
        self.cb_bounds_on.blockSignals(True)
        self.sb_lb.blockSignals(True)
        self.sb_ub.blockSignals(True)
        self.cb_bounds_on.setChecked(bool(fp.bounds_on))
        # Enable/disable editors based on bounds_on
        self.sb_lb.setEnabled(bool(fp.bounds_on))
        self.sb_ub.setEnabled(bool(fp.bounds_on))
        try:
            b = getattr(fp, 'bounds', None)
            if isinstance(b, (tuple, list)) and len(b) == 2:
                lb, ub = b
                self.sb_lb.setValue(float(lb))
                self.sb_ub.setValue(float(ub))
        except Exception:
            pass
        self.cb_bounds_on.blockSignals(False)
        self.sb_lb.blockSignals(False)
        self.sb_ub.blockSignals(False)




class FittingParameterWidget(Controller):

    def _build_details_tooltip_text(self) -> str:
        fp = self.fitting_parameter
        lines = [str(getattr(fp, 'name', ''))+":"]

        try:
            desc = getattr(fp, 'description', "")
        except Exception:
            desc = ""
        if desc:
            desc_lines = []
            for para in str(desc).splitlines():
                wrapped = textwrap.wrap(para, width=30)
                if wrapped:
                    desc_lines.extend(wrapped)
                else:
                    desc_lines.append("")
            lines.append("\n".join(desc_lines).strip())

        # Link info
        link_param = getattr(fp, 'link', None)
        if bool(getattr(fp, 'is_linked', False)) and link_param is not None:
            target_param_name = getattr(link_param, 'name', "?")
            try:
                target_fit_idx = getattr(link_param, 'fit_idx', "?")
            except Exception:
                target_fit_idx = "?"
            lines.append("")
            lines.append(textwrap.fill(
                f"Linked to fit '{target_fit_idx}', parameter '{target_param_name}'",
                width=60
            ))
        else:
            lines.append("")
            lines.append("Not linked")

        # Value / fixed
        try:
            v = float(getattr(fp, 'value', float('nan')))
        except Exception:
            v = float('nan')
        lines.append(f"Value: {v}")
        lines.append(f"Fixed: {bool(getattr(fp, 'fixed', False))}")

        # Bounds
        if bool(getattr(fp, 'bounds_on', False)):
            b = getattr(fp, 'bounds', None)
            if isinstance(b, (tuple, list)) and len(b) == 2:
                lines.append(f"Bounds: ({b[0]}, {b[1]})")
            else:
                lines.append("Bounds: on (unset)")
        else:
            lines.append("Bounds: off")

        return "\n".join([l for l in lines if l is not None])

    def make_linkcall(self, fit_idx: int, parameter_name: str):
        def linkcall():
            try:
                self.blockSignals(True)

                # Fetch current and target parameters
                param_self = chisurf.fits[self.fitting_parameter.fit_idx].model.parameters_all_dict[self.fitting_parameter.name]
                param_other = chisurf.fits[fit_idx].model.parameters_all_dict[parameter_name]

                # Check for recursion using the Parameter class method
                if param_self.check_recursive_link(param_other, param_self):
                    QtWidgets.QMessageBox.warning(
                        self,  # Parent widget
                        "Linking Error",
                        "Recursion detected: Cannot link a parameter to itself or create a cyclic dependency.",
                        QtWidgets.QMessageBox.Ok
                    )
                else:
                    tooltip = " linked to " + parameter_name
                    s = (
                        f"chisurf.fits[{self.fitting_parameter.fit_idx}].model.parameters_all_dict['{self.fitting_parameter.name}'].link = "
                        f"chisurf.fits[{fit_idx}].model.parameters_all_dict['{parameter_name}'] \n"
                        f"chisurf.fits[{self.fitting_parameter.fit_idx}].update()"
                    )
                    # Execute the link assignment in the global chisurf
                    # context; the Parameter.link setter will update the
                    # follower's controller state via set_linked(True).
                    chisurf.run(s)

                    # Refresh this widget from the underlying parameter so it
                    # reflects the follower/linked role. The target parameter
                    # (master) remains visually unchanged (no check mark), so
                    # the user can always use this row's checkbox to unlink.
                    self.widget_link.setToolTip(tooltip)
                    try:
                        self.finalize()
                    except Exception:
                        pass

            finally:
                self.blockSignals(False)

        return linkcall

    def build_link_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        menu.setTitle(
            "Link " + self.fitting_parameter.name + " to:"
        )

        for fit_idx, f in enumerate(chisurf.fits):
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)

                # Sorted by "Aggregation"
                for a in fs.model.aggregated_parameters:
                    action_submenu = QtWidgets.QMenu(submenu)
                    action_submenu.setTitle(a.name)
                    ut = a.parameters_all
                    ut.sort(key=lambda x: x.name, reverse=False)
                    for p in ut:
                        if p is not self.fitting_parameter:
                            Action = action_submenu.addAction(p.name)
                            Action.triggered.connect(
                                self.make_linkcall(fit_idx, p.name)
                            )
                    submenu.addMenu(action_submenu)
                action_submenu = QtWidgets.QMenu(submenu)

                # Simply all parameters
                action_submenu.setTitle("All parameters")
                keys = list(fs.model.parameters_all_dict.keys())
                sorted_keys = sorted(keys)
                for key in sorted_keys:
                    p = fs.model.parameters_all_dict[key]
                    if p is not self:
                        Action = action_submenu.addAction(p.name)
                        Action.triggered.connect(self.make_linkcall(fit_idx, p.name))
                submenu.addMenu(action_submenu)

                menu.addMenu(submenu)
        return menu

    def contextMenuEvent(self, event: QtGui.QCloseEvent):

        menu = self.build_link_menu()
        menu.exec_(event.globalPos())

    def __str__(self):
        return ""

    @chisurf.gui.decorators.init_with_ui("variable_widget.ui")
    def __init__(
            self,
            fitting_parameter: chisurf.fitting.parameter.FittingParameter,
            layout: QtWidgets.QLayout = None,
            decimals: int = None,
            hide_label: bool = None,
            hide_error: bool = None,
            fixable: bool = None,
            hide_bounds: bool = None,
            name: str = None,
            label_text: str = None,
            hide_link: bool = None,
            suffix: str = "",
            callback: typing.Callable = None
    ):
        if hide_link is None:
            hide_link = parameter_settings['hide_link']
        if hide_bounds is None:
            hide_bounds = parameter_settings['hide_bounds']
        if name is None:
            name = self.__class__.__name__
        if label_text is None:
            label_text = name
        if fixable is None:
            fixable = parameter_settings['fixable']
        hide_fix_checkbox = fixable
        if hide_error is None:
            hide_error = parameter_settings['hide_error']
        if hide_label is None:
            hide_label = parameter_settings['hide_label']
        if decimals is None:
            decimals = parameter_settings['decimals']

        self.callback = callback
        self.name = fitting_parameter.name
        self.fitting_parameter = fitting_parameter
        self._details_popup = None  # created lazily on first label click
        self._is_output_param = bool(getattr(fitting_parameter, "is_output", False))

        # Allow HTML/RichText labels (e.g. "cpm<sub>all</sub>") so that
        # parameter names can be decorated with subscripts/superscripts
        # while keeping the underlying parameter name unchanged.
        try:
            self.label.setTextFormat(QtCore.Qt.RichText)
        except Exception:
            pass

        self.widget_value = pg.SpinBox(
            dec=True,
            decimals=decimals,
            suffix=suffix,
            finite=False
        )
        self.widget_value.opts['compactHeight'] = False
        self.horizontalLayout.addWidget(self.widget_value)

        self.widget_lower_bound = pg.SpinBox(
            dec=True,
            decimals=decimals
        )
        self.horizontalLayout_2.addWidget(self.widget_lower_bound)

        self.widget_upper_bound = pg.SpinBox(
            dec=True,
            decimals=decimals
        )
        self.horizontalLayout_2.addWidget(self.widget_upper_bound)

        # Hide and disable widgets
        self.label.setVisible(not hide_label)
        self.lineEdit.setVisible(not hide_error)
        self.widget_bounds_on.setDisabled(hide_bounds)
        self.widget_fix.setVisible(fixable or not hide_fix_checkbox)
        self.widget.setHidden(hide_bounds)
        self.widget_link.setDisabled(hide_link)

        if self._is_output_param:
            # Output parameters are displayed as read-only result cells.
            # Keep the row layout identical (checkboxes stay visible) but
            # prevent any user interaction and remove spin buttons so the
            # value looks like a plain, non-editable field.
            try:
                # Try to hide spin buttons directly on the SpinBox.
                self.widget_value.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            except Exception:
                # Fallback for pyqtgraph.SpinBox implementations that expose
                # an inner "spin" widget.
                try:
                    spin = getattr(self.widget_value, "spin", None)
                    if spin is not None and hasattr(spin, "setButtonSymbols"):
                        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
                except Exception:
                    pass
            try:
                self.widget_value.setReadOnly(True)
            except Exception:
                pass
            try:
                # Avoid focus so wheel / keyboard cannot change the value.
                self.widget_value.setFocusPolicy(QtCore.Qt.NoFocus)
            except Exception:
                pass
            try:
                self.widget_fix.setEnabled(False)
            except Exception:
                pass
            try:
                self.widget_bounds_on.setEnabled(False)
            except Exception:
                pass
            try:
                self.widget_link.setEnabled(False)
            except Exception:
                pass
            # Hide the lower/upper bound spin boxes for outputs; only keep
            # the (disabled) bounds checkbox for alignment.
            self.widget.setHidden(True)

        # Make label interactive: clicking opens a details popup
        try:
            self.label.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            try:
                self.label.setToolTip(self._build_details_tooltip_text() + "\n\nClick to view and edit details")
            except Exception:
                try:
                    self.label.setToolTip(f"{getattr(self.fitting_parameter, 'name', '')}\n\nClick to view and edit details")
                except Exception:
                    self.label.setToolTip("Click to view and edit details")
            # install a mousePress handler
            self.label.mousePressEvent = self._on_label_mouse_press  # type: ignore
        except Exception:
            pass

        # Display of values
        try:
            _init_v = float(fitting_parameter.value)
        except Exception:
            _init_v = self.widget_value.value() if hasattr(self, 'widget_value') else 0.0
        self.widget_value.setValue(_init_v)
        # Do not left-pad HTML labels with spaces; this breaks rich text.
        # For plain-text labels we keep the original padding.
        try:
            if "<" in label_text or ">" in label_text:
                self.label.setText(label_text)
            else:
                self.label.setText(label_text.ljust(5))
        except Exception:
            self.label.setText(label_text)

        # variable bounds
        if not fitting_parameter.bounds_on:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Checked)

        # variable fixed
        if fitting_parameter.fixed:
            self.widget_fix.setCheckState(QtCore.Qt.Checked)
        else:
            self.widget_fix.setCheckState(QtCore.Qt.Unchecked)
        self.widget.hide()

        # The variable value
        self.widget_value.editingFinished.connect(self._on_main_value_changed)
        if callback:
            self.widget_value.editingFinished.connect(self.callback)

        self.widget_fix.toggled.connect(
            lambda: chisurf.run(
                f"chisurf.fits[{self.fitting_parameter.fit_idx}].model.parameters_all_dict['{fitting_parameter.name}'].fixed = "
                f"{self.widget_fix.isChecked()} \n"
                f"chisurf.fits[{self.fitting_parameter.fit_idx}].update()")
        )

        # Variable is bounded
        self.widget_bounds_on.toggled.connect(self._on_main_bounds_on_toggled)

        self.widget_lower_bound.editingFinished.connect(
            lambda: chisurf.run(
                f"chisurf.fits[{self.fitting_parameter.fit_idx}].model.parameters_all_dict['%s'].bounds = (%s, %s)" %
                (
                    fitting_parameter.name,
                    self.widget_lower_bound.value(),
                    self.widget_upper_bound.value()
                )
            )
        )

        self.widget_upper_bound.editingFinished.connect(
            lambda: chisurf.run(
                f"chisurf.fits[{self.fitting_parameter.fit_idx}].model.parameters_all_dict['%s'].bounds = (%s, %s)" %
                (
                    fitting_parameter.name,
                    self.widget_lower_bound.value(),
                    self.widget_upper_bound.value()
                )
            )
        )

        self.widget_link.clicked.connect(self.onLinkFitGroup)

        if isinstance(layout, QtWidgets.QLayout):
            layout.addWidget(self)
        try:
            self._update_role_visuals()
        except Exception:
            pass


    def _on_label_mouse_press(self, event: QtGui.QMouseEvent):
        try:
            if event.button() == QtCore.Qt.LeftButton:
                self._open_details_popup()
            else:
                # fall back to default behavior
                super().mousePressEvent(event)
        except Exception:
            pass

    def _open_details_popup(self):
        if getattr(self, "_is_output_param", False):
            return
        # Lazy-create popup
        if self._details_popup is None or not isinstance(self._details_popup, FittingParameterDetailPopup):
            self._details_popup = FittingParameterDetailPopup(self)
        # Position popup under the label
        try:
            global_pos = self.label.mapToGlobal(self.label.rect().bottomLeft())
        except Exception:
            global_pos = QtGui.QCursor.pos()
        self._details_popup.move(global_pos)
        self._details_popup.refresh_from_model()
        self._details_popup.show()
        # Ensure the popup gains focus and is on top
        try:
            self._details_popup.raise_()
            self._details_popup.activateWindow()
            self._details_popup.setFocus(QtCore.Qt.PopupFocusReason)
            # Some platforms need delayed activation
            QtCore.QTimer.singleShot(0, self._details_popup.activateWindow)
        except Exception:
            pass

    def _on_label_mouse_press(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self._open_details_popup()
        else:
            # For other buttons, fall back to default behavior (e.g., open context menu on right click)
            super().mousePressEvent(event)

    def _update_role_visuals(self):
        try:
            if getattr(self, "_is_output_param", False):
                role = "output"
            elif getattr(self.fitting_parameter, "is_linked", False):
                role = "linked"
            else:
                role = "input"
            bg_color = None
            if role == "output":
                bg_color = parameter_settings.get("role_color_output")
            elif role == "linked":
                bg_color = parameter_settings.get("role_color_linked")
            else:
                bg_color = parameter_settings.get("role_color_input")
            parts = []
            if bg_color:
                parts.append(f"background-color: {bg_color};")
            if role == "linked":
                parts.append("text-decoration: underline;")
            style = " ".join(parts)
            self.widget_value.setStyleSheet(style)
        except Exception:
            pass

    def set_linked(self, is_linked: bool):
        if getattr(self, "_is_output_param", False):
            self.widget_link.setCheckState(QtCore.Qt.Unchecked)
            return
        # Interpret linking state in terms of three visual roles:
        #   - Unchecked: not linked at all.
        #   - PartiallyChecked: this parameter follows another one (slave).
        #   - Checked: this parameter is the master within a fit group.
        is_master = bool(getattr(self.fitting_parameter, "is_link_master", False))

        if is_linked:
            # Follower: value is controlled by the master; disable editing
            # and show a partially-checked box.
            self.widget_link.setCheckState(QtCore.Qt.PartiallyChecked)
            self.widget_value.setEnabled(False)
        else:
            if is_master:
                # Master within the fit group: keep value editable but mark
                # the checkbox as fully checked so the user sees it as the
                # source of the group link.
                self.widget_link.setCheckState(QtCore.Qt.Checked)
                self.widget_value.setEnabled(True)
            else:
                # Not linked at all.
                self.widget_link.setCheckState(QtCore.Qt.Unchecked)
                self.widget_value.setEnabled(True)

        try:
            self._update_role_visuals()
        except Exception:
            pass

    def onLinkFitGroup(self):
        # Clicking the link checkbox should have intuitive semantics:
        #
        # - If this parameter is currently a *follower* (linked to some
        #   master), a click unlinks **only this parameter**.
        # - Otherwise (unlinked or acting as fit-group master), we delegate
        #   to the group-level macro so the user can establish or remove a
        #   fit-group link.
        fp = self.fitting_parameter
        if getattr(self, "_is_output_param", False):
            return

        is_linked = bool(getattr(fp, "is_linked", False))
        is_master = bool(getattr(fp, "is_link_master", False))

        self.blockSignals(True)
        try:
            if is_linked and not is_master:
                # Per-parameter unlink: this row was following another
                # parameter via ``fp.link``. Clear the link so only this
                # parameter becomes free again.
                try:
                    fp.link = None
                except Exception:
                    try:
                        chisurf.logging.warning(
                            f"FittingParameterWidget: failed to unlink parameter '{getattr(fp, 'name', '?')}'."
                        )
                    except Exception:
                        pass
            else:
                # Group-level behaviour: interpret the current checkbox
                # state as a request to link/unlink the whole fit group for
                # this parameter name.
                state = int(self.widget_link.checkState())
                chisurf.run(
                    f"chisurf.macros.link_fit_group('{fp.name}', {state})"
                )

            try:
                self.finalize()
            except Exception:
                pass
        finally:
            self.blockSignals(False)

    def setValue(self, v):
        self.widget_value.setValue(v)

    def _on_main_value_changed(self):
        if getattr(self, "_is_output_param", False):
            return
        fp = self.fitting_parameter
        try:
            fit_idx = fp.fit_idx
        except Exception:
            fit_idx = -1
        # Guard against invalid indices so we never accidentally target chisurf.fits[-1]
        try:
            n_fits = len(chisurf.fits)
        except Exception:
            n_fits = 0
        if not isinstance(fit_idx, int) or fit_idx < 0 or fit_idx >= n_fits:
            try:
                chisurf.logging.warning(
                    f"FittingParameterWidget: invalid fit_idx {fit_idx} for parameter '{getattr(fp, 'name', '?')}', "
                    f"skipping value change."
                )
            except Exception:
                pass
            return

        value = self.widget_value.value()
        chisurf.run(
            f"parameter = chisurf.fits[{fit_idx}].model.parameters_all_dict['{fp.name}']\n"
            f"fixed = parameter.fixed \n"
            f"parameter.fixed = False\n"
            f"parameter.value = {value} \n"
            f"parameter.fixed = fixed\n"
            f"chisurf.fits[{fit_idx}].finalize()"
        )

    def _on_main_bounds_on_toggled(self):
        if getattr(self, "_is_output_param", False):
            return
        fp = self.fitting_parameter
        checked = self.widget_bounds_on.isChecked()
        # Toggle bounds_on in the model
        chisurf.run(
            f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].bounds_on = {checked}"
        )
        # If turning ON and current bounds are invalid/missing, initialize them from the UI spin boxes
        if checked:
            bounds_valid = False
            try:
                b = getattr(fp, 'bounds', None)
                bounds_valid = isinstance(b, (tuple, list)) and len(b) == 2
            except Exception:
                bounds_valid = False
            if not bounds_valid:
                chisurf.run(
                    f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].bounds = ({self.widget_lower_bound.value()}, {self.widget_upper_bound.value()})"
                )
        # Refresh UI/model without risking unpack errors
        self.finalize()

    def finalize(self, *args):
        # Ensure execution on the widget's thread (GUI thread). If called from another thread,
        # reschedule finalize to run on the correct thread and return immediately.
        if QtCore.QThread.currentThread() is not self.thread():
            try:
                # Queue the call to this object's thread (GUI thread)
                QtCore.QMetaObject.invokeMethod(self, "finalize", QtCore.Qt.QueuedConnection)
            except Exception:
                # Fallback: schedule via QApplication event loop
                app = QtWidgets.QApplication.instance()
                if app is not None:
                    QtCore.QTimer.singleShot(0, lambda: self.finalize())
            return
        #super().update(*args)
        self.blockSignals(True)

        # Sync link UI state first
        try:
            self.set_linked(self.fitting_parameter.is_linked)
        except Exception:
            pass

        # Update value of widget (guard against None)
        try:
            _v = float(self.fitting_parameter.value)
        except Exception:
            _v = self.widget_value.value()
        self.widget_value.setValue(_v)
        self.widget_fix.setCheckState(QtCore.Qt.Checked if self.fitting_parameter.fixed else QtCore.Qt.Unchecked)

        # Sync bounds UI safely (no unpack unless valid)
        try:
            self.widget_bounds_on.blockSignals(True)
            self.widget_lower_bound.blockSignals(True)
            self.widget_upper_bound.blockSignals(True)
            bounds_on = bool(getattr(self.fitting_parameter, 'bounds_on', False))
            self.widget_bounds_on.setCheckState(QtCore.Qt.Checked if bounds_on else QtCore.Qt.Unchecked)

            # Default to current UI values; replace with model values only if valid
            lb_val = self.widget_lower_bound.value()
            ub_val = self.widget_upper_bound.value()
            b = getattr(self.fitting_parameter, 'bounds', None)
            if isinstance(b, (tuple, list)) and len(b) == 2:
                try:
                    lb_val = float(b[0])
                    ub_val = float(b[1])
                except Exception:
                    pass
            self.widget_lower_bound.setValue(lb_val)
            self.widget_upper_bound.setValue(ub_val)
        except Exception:
            pass
        finally:
            try:
                self.widget_bounds_on.blockSignals(False)
                self.widget_lower_bound.blockSignals(False)
                self.widget_upper_bound.blockSignals(False)
            except Exception:
                pass

        # Tooltip (guard against invalid bounds)
        if getattr(self.fitting_parameter, 'bounds_on', False):
            b = getattr(self.fitting_parameter, 'bounds', None)
            if isinstance(b, (tuple, list)) and len(b) == 2:
                lower, upper = b
                tooltip_text = f"bound: ({lower}, {upper})\n"
            else:
                tooltip_text = "bounds: on (unset)\n"
        else:
            tooltip_text = "bounds: off\n"

        link_param = getattr(self.fitting_parameter, 'link', None)
        if self.fitting_parameter.is_linked and link_param is not None:
            target_param_name = getattr(link_param, 'name', "?")
            target_fit_label = "?"
            try:
                target_fit_idx = getattr(link_param, 'fit_idx', -1)
            except Exception:
                target_fit_idx = -1
            try:
                if isinstance(target_fit_idx, int) and target_fit_idx >= 0:
                    fits = getattr(chisurf, 'fits', None)
                    if fits is not None and 0 <= target_fit_idx < len(fits):
                        target_fit = fits[target_fit_idx]
                        target_fit_label = getattr(target_fit, 'name', str(target_fit_idx))
                    else:
                        target_fit_label = str(target_fit_idx)
            except Exception:
                pass
            tooltip_text += f"linked to fit '{target_fit_label}', \n parameter '{target_param_name}'"
        self.widget_value.setToolTip(tooltip_text)

        try:
            try:
                details = self._build_details_tooltip_text()
            except Exception:
                details = ""
            if details:
                self.label.setToolTip(details + "\n\nClick to view and edit details")
            else:
                try:
                    self.label.setToolTip(f"{getattr(self.fitting_parameter, 'name', '')}\n\nClick to view and edit details")
                except Exception:
                    self.label.setToolTip("Click to view and edit details")
        except Exception:
            pass

        # Error-estimate
        value = float(self.fitting_parameter.value)
        if not np.isfinite(value):
            rel_error = "NA"
        else:
            error_estimate = self.fitting_parameter.error_estimate
            rel_error = abs(error_estimate / (value + 1e-12) * 100.0)

        if self.fitting_parameter.fixed or not isinstance(error_estimate, float):
            self.lineEdit.setText("NA")
            # Reset background color to default
            self.lineEdit.setStyleSheet("")
        else:
            self.lineEdit.setText("NA" if np.isnan(rel_error) else f"{rel_error:.0f}%")

            # Set background color based on relative error
            if not np.isnan(rel_error):
                # Create a colormap from error_color_small to error_color_large
                # Use default values if settings are not found
                error_color_small = parameter_settings.get('error_color_small', 'green')
                error_color_large = parameter_settings.get('error_color_large', 'magenta')
                error_threshold_small = parameter_settings.get('error_threshold_small', 20)
                error_threshold_large = parameter_settings.get('error_threshold_large', 100)

                cmap = mcolors.LinearSegmentedColormap.from_list(
                    'error_color_gradient',
                    [(0, error_color_small), (1, error_color_large)]
                )

                # Normalize error value: error_threshold_small -> error_color_small, error_threshold_large -> error_color_large
                error_range = error_threshold_large - error_threshold_small
                norm_error = min(1.0, max(0.0, (rel_error - error_threshold_small) / error_range))

                # Get RGB color from colormap
                rgb_color = cmap(norm_error)

                # Convert RGB to hex for stylesheet
                hex_color = mcolors.rgb2hex(rgb_color)

                # Set background color and ensure text is readable
                # Use white text for darker backgrounds, black for lighter ones
                r, g, b = rgb_color[:3]
                brightness = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "white" if brightness < 0.5 else "black"

                # Set background color and text color
                self.lineEdit.setStyleSheet(f"background-color: {hex_color}; color: {text_color};")

        # Link
        if link_param is not None:
            target_param_name = getattr(link_param, 'name', "?")
            target_fit_label = "?"
            try:
                target_fit_idx = getattr(link_param, 'fit_idx', -1)
            except Exception:
                target_fit_idx = -1
            try:
                if isinstance(target_fit_idx, int) and target_fit_idx >= 0:
                    fits = getattr(chisurf, 'fits', None)
                    if fits is not None and 0 <= target_fit_idx < len(fits):
                        target_fit = fits[target_fit_idx]
                        target_fit_label = getattr(target_fit, 'name', str(target_fit_idx))
                    else:
                        target_fit_label = str(target_fit_idx)
            except Exception:
                pass
            tooltip = f"linked to fit '{target_fit_label}', parameter '{target_param_name}'"
            self.widget_link.setToolTip(tooltip)
            self.widget_value.setEnabled(False)

        # If the details popup is open, refresh its contents to reflect latest model state
        try:
            if getattr(self, '_details_popup', None) is not None and self._details_popup.isVisible():
                self._details_popup.refresh_from_model()
        except Exception:
            pass

        # If the details popup is open, refresh its contents to reflect latest model state
        try:
            if getattr(self, '_details_popup', None) is not None and self._details_popup.isVisible():
                self._details_popup.refresh_from_model()
        except Exception:
            pass

        try:
            self._update_role_visuals()
        except Exception:
            pass

        self.blockSignals(False)


class FittingParameterGroupWidget(QtWidgets.QGroupBox):

    def __init__(
            self,
            parameter_group: chisurf.fitting.parameter.FittingParameterGroup,
            n_col: int = None,
            layout: QtWidgets.QVBoxLayout = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        if n_col is None:
            n_col = chisurf.settings.gui['fit_models']['n_columns']

        self.parameter_group = parameter_group
        self.n_col = n_col
        self.n_row = 0

        self.setTitle(parameter_group.name)
        if layout is None:
            layout = QtWidgets.QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(layout)
        for i, p in enumerate(parameter_group.parameters_all):
            label_text = p.__dict__.get('label_text', p.name)
            pw = make_fitting_parameter_widget(
                fitting_parameter=p,
                label_text=label_text
            )
            col = i % self.n_col
            row = i // self.n_col
            layout.addWidget(pw, row, col)


def make_fitting_parameter_widget(
        fitting_parameter: chisurf.fitting.parameter.FittingParameter,
        label_text: str = None,
        layout: QtWidgets.QLayout = None,
        decimals: int = None,
        hide_label: bool = None,
        hide_error: bool = None,
        fixable: bool = None,
        hide_bounds: bool = None,
        name: str = None,
        hide_link: bool = None,
        suffix: str = "",
        callback: typing.Callable = None
) -> FittingParameterWidget:
    if label_text is None:
        # Safely get label_text from parameter's __dict__ or use name as fallback
        label_text = fitting_parameter.__dict__.get('label_text', fitting_parameter.name)
    # If no explicit suffix was provided, infer simple unit suffixes from the
    # parameter name (e.g. *_nm -> " nm", *_um -> " µm"). This keeps
    # backwards compatibility while improving readability for standard unit
    # conventions used throughout ChiSurf.
    auto_suffix = suffix
    if not auto_suffix:
        n = str(fitting_parameter.name)
        if n.endswith("_nm"):
            auto_suffix = " nm"
        elif n.endswith("_um"):
            auto_suffix = " µm"
        elif n.endswith("_ms"):
            auto_suffix = " ms"
        elif n.endswith("_us"):
            auto_suffix = " µs"
        elif n.endswith("_ns"):
            auto_suffix = " ns"
        elif n.endswith("_K"):
            auto_suffix = " K"

    widget = FittingParameterWidget(
        fitting_parameter,
        hide_label=hide_label,
        layout=layout,
        decimals=decimals,
        hide_error=hide_error,
        fixable=fixable,
        hide_bounds=hide_bounds,
        name=name,
        hide_link=hide_link,
        label_text=label_text,
        suffix=auto_suffix,
        callback=callback
    )
    fitting_parameter.controller = widget
    return widget


def make_fitting_parameter_group_widget(
        fitting_parameter_group: chisurf.fitting.parameter.FittingParameterGroup,
        *args,
        **kwargs
):
    return FittingParameterGroupWidget(
        fitting_parameter_group,
        *args,
        **kwargs
    )

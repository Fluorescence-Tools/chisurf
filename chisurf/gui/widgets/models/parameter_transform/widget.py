from __future__ import annotations

import pathlib
import pickle
import yaml

import typing

import chisurf.core.fio as io

from qtpy import QtCore, QtWidgets
import chisurf.core.fitting
import chisurf.gui.widgets
import chisurf.gui.decorators
from qtpy import uic

from chisurf.gui import plots
from chisurf.core.models.parameter_transform.model import ParameterTransformModel
from chisurf.gui.widgets.models import model_widget as model


class ParameterTransformWidget(model.ModelWidget, ParameterTransformModel):
    """Widget for the parameter transform model with code editor and parameter table."""

    plot_classes = [
                    (plots.FitInfo, {}),
                    # (plots.ResidualPlot, {})
    ]

    def create_parameter_widgets(self):
        """Rebuild the parameter editor widgets from the current model parameters."""
        layout = self.w.gridLayout
        chisurf.gui.widgets.clear_layout(layout)

        try:
            n_columns = chisurf.core.settings.gui['fit_models']['n_columns']
        except Exception:
            n_columns = 2
        row = 1

        p_dict = self.parameters_all_dict
        p_keys = list(p_dict.keys())
        p_keys.sort()

        self.set_default_parameter_values()

        for i, pk in enumerate(p_keys):
            p = p_dict[pk]
            # Optimize for space in 2-column layout by hiding the error estimates
            pw = chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
                p,
                callback=self.finalize,
                hide_error=True
            )
            try:
                pw.label.setMinimumWidth(35)
            except Exception:
                pass
            column = i % n_columns
            if column == 0:
                row += 1
            layout.addWidget(pw, row, column)

    def set_default_parameter_values(self):
        """Set parameter values and bounds from the YAML model definition."""
        d = self.codes[self.code_name]['initial']
        param_keys = list(self.parameters_all_dict.keys())

        for k in param_keys:
            initial = d.get(k, None)
            if initial is not None:
                self.parameters_all_dict[k].value = initial['value']
                self.parameters_all_dict[k].bounds = initial['bounds']
                self.parameters_all_dict[k].bounds_on = True

    @property
    def code_name(self):
        """Name of the currently selected code definition."""
        current_index = self.w.comboBox.currentIndex()

        if not self.codes:
            return ""

        code_keys = list(self.codes.keys())
        if not code_keys:
            return ""

        if current_index < 0 or current_index >= len(code_keys):
            return code_keys[0] if code_keys else ""

        return code_keys[current_index]

    @code_name.setter
    def code_name(self, v: str):
        """Set the currently selected code definition by name."""
        idx = self.w.comboBox.findText(v)
        if idx == -1:
            return
        self.w.comboBox.setCurrentIndex(idx)

    @property
    def codes(self) -> typing.Dict:
        """Dictionary of loaded code definitions keyed by name."""
        return self._codes

    @codes.setter
    def codes(self, v: typing.Dict):
        """Set the codes dictionary and populate the combo box."""
        self._codes = v
        self.w.comboBox.clear()

        if not v:
            return

        code_keys = list(v.keys())
        self.w.comboBox.addItems(code_keys)

    def format_description(self, desc: str) -> str:
        """Format the description string, converting Markdown and simple math formulas to HTML.

        Parameters
        ----------
        desc : str
            The raw description string.

        Returns
        -------
        str
            The formatted HTML description.
        """
        if not desc:
            return ""

        # If the description already contains HTML tags, return it as is
        if "<p>" in desc or "<b>" in desc or "<i>" in desc or "<br/>" in desc:
            return desc

        # Otherwise, perform simple markdown and math formatting:
        # Convert newlines to breaks
        html = desc.replace("\n", "<br/>")

        # Format inline variables (e.g. sD, sA, R0, RDA, dRDAE, dRmp) as italicized with subscripts
        import re

        # Replace -> with &rarr;
        html = html.replace("->", "&rarr;")

        # Common variables formatting
        replacements = {
            r"\bsD\b": "<i>s</i><sub>D</sub>",
            r"\bsA\b": "<i>s</i><sub>A</sub>",
            r"\bR0\b": "<i>R</i><sub>0</sub>",
            r"\bRDA\b": "<i>R</i><sub>DA</sub>",
            r"\bdRDAE\b": "<i>d</i><sub>RDAE</sub>",
            r"\bdRmp\b": "<i>d</i><sub>Rmp</sub>",
            r"\bsDA\b": "<i>s</i><sub>DA</sub>",
            r"\bE\b": "<i>E</i>",
            r"\bk\b": "<i>k</i>",
        }
        for pattern, repl in replacements.items():
            html = re.sub(pattern, repl, html)

        return html

    def onCodeChanged(self):
        """Handle selection of a different code definition from the combo box."""
        try:
            code = self.codes[self.code_name]['code']
            self.w.textEdit.setPlainText(code)

            # Retrieve, format, and display the model description
            desc = self.codes[self.code_name].get('description', '')
            formatted_desc = self.format_description(desc)
            self.w.descriptionBrowser.setHtml(formatted_desc)

            self.onFunctionUpdate()
        except Exception as e:
            # Log the error and show a message to the user
            import logging
            logging.error(f"Error changing code: {str(e)}")
            # Show error message to the user
            from qtpy import QtWidgets
            QtWidgets.QMessageBox.warning(
                self, 
                "Code Error", 
                f"Error loading code: {str(e)}\n\nPlease check the code definition in the YAML file."
            )

    def onFunctionUpdate(self):
        """Update the model function from the code editor contents."""
        t = self.w.textEdit.toPlainText()
        try:
            self.function = str(t)
            self.create_parameter_widgets()
        except Exception as e:
            # Log the error and show a message to the user
            import logging
            logging.error(f"Error updating function: {str(e)}")
            # Show error message to the user
            from qtpy import QtWidgets
            QtWidgets.QMessageBox.warning(
                self, 
                "Function Error", 
                f"Error in function definition: {str(e)}\n\nPlease correct the function and try again."
            )

    def load_model_file(self, filename: pathlib.Path):
        """Load a YAML code definition file.

        Parameters
        ----------
        filename : pathlib.Path
            Path to the YAML file.
        """
        with io.open_maybe_zipped(filename, 'r') as fp:
            self._code_file = filename
            yaml_content = yaml.safe_load(fp)
            self.codes = yaml_content
            
            # Shorten the displayed path and add full path as tooltip
            self.w.lineEdit.setToolTip(str(filename.as_posix()))
            parts = filename.parts
            if len(parts) > 3:
                display_path = ".../" + "/".join(parts[-3:])
            else:
                display_path = str(filename.as_posix())
            self.w.lineEdit.setText(display_path)
            self.w.lineEdit.setReadOnly(True)

    def __init__(
            self,
            fit: chisurf.core.fitting.fit.Fit,
            *args,
            code_file: pathlib.Path = None,
            **kwargs
    ):
        """Initialize the parameter transform widget.

        Parameters
        ----------
        fit : chisurf.core.fitting.fit.Fit
            The fit object this widget belongs to.
        code_file : pathlib.Path, optional
            Path to a YAML file with code definitions.
        """
        super().__init__(fit, *args, **kwargs)

        path = pathlib.Path(__file__).parent.absolute()
        w = uic.loadUi(path / "parameter_transform.ui")

        l = QtWidgets.QVBoxLayout()
        self.setLayout(l)
        l.addWidget(w)
        self.w = w

        self._codes = {}
        if code_file is None:
            import chisurf
            code_file = pathlib.Path(chisurf.__file__).parent / 'core' / 'models' / 'parameter_transform' / 'models.yaml'
        self._code_file = code_file.absolute().as_posix()

        self.load_model_file(code_file)

        self.w.actionFunctionUpdate.triggered.connect(self.onFunctionUpdate)
        self.w.actionCodeChanges.triggered.connect(self.onCodeChanged)

        self.w.checkBox.setChecked(False)
        self.w.comboBox.setCurrentIndex(1)

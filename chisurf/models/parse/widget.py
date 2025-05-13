from __future__ import annotations

import pathlib
import re
import os
import tempfile
import io as python_io
from chisurf import typing

import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qtpy import QtWidgets, QtCore, QtGui
import sympy


import chisurf
import chisurf.decorators
import chisurf.fio as io
import chisurf.fitting
import chisurf.gui.decorators
import chisurf.models
import chisurf.settings
import chisurf.gui.widgets
import chisurf.gui.tools


class EquationDialog(QtWidgets.QDialog):
    """A dialog that displays a formatted equation."""

    def __init__(self, parent=None, title="Equation"):
        super().__init__(parent)
        self.setWindowTitle(title)
        # Remove the fixed minimum size to allow the dialog to resize based on content
        # self.setMinimumSize(400, 200)

        # Create layout
        layout = QtWidgets.QVBoxLayout(self)

        # Create QTextBrowser for the equation
        self.equationBrowser = QtWidgets.QTextBrowser()
        # Disable scrollbars to ensure the dialog resizes to fit the content
        self.equationBrowser.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.equationBrowser.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.equationBrowser)

        # Create close button
        self.closeButton = QtWidgets.QPushButton("Close")
        self.closeButton.clicked.connect(self.close)
        layout.addWidget(self.closeButton, alignment=QtCore.Qt.AlignRight)

        self.setLayout(layout)

    def setEquation(self, html):
        """Set the HTML content of the equation browser."""
        self.equationBrowser.setHtml(html)

        # Adjust the size of the dialog to fit the content
        # This ensures that the dialog is properly sized when it's first shown
        self.adjustSize()

from chisurf.models.model import ModelWidget
from chisurf.models.parse import ParseModel


class ParseFormulaWidget(QtWidgets.QWidget):

    @chisurf.gui.decorators.init_with_ui("parseWidget.ui")
    def __init__(
            self,
            n_columns: int = None,
            model_file: pathlib.Path = None,
            model_name: str = None,
            model: chisurf.models.Model = None
    ):
        self.model: chisurf.models.parse.ParseModel = model
        if n_columns is None:
            n_columns = chisurf.settings.gui['fit_models']['n_columns']
        self.n_columns = n_columns

        # Initialize temp files list
        self._temp_files = []

        self._models = {}
        if model_file is None:
            model_file = pathlib.Path(__file__).parent / 'models.yaml'
        self._model_file = model_file.absolute().as_posix()
        self.load_model_file(model_file)

        if model_name is None:
            model_name = list(self._models)[0]
        self.model_name = model_name

        function_str = self.models[self.model_name]['equation']
        self.model.func = f"{function_str}"
        self.set_default_parameter_values(model_name)
        self.plainTextEdit.setPlainText(function_str)
        self.create_parameter_widgets()

        # Initialize equation dialog to None
        self.equationDialog = None

        self.toolButton_4.clicked.connect(self.onShowEquation)

        self.editor = chisurf.gui.tools.code_editor.CodeEditor(None, language='yaml', can_load=False)
        self.editor.hide()

        # Initialize textEdit with description and equation
        func = self.models[self.model_name]['equation']
        formatted_equation = self.format_equation(func)
        description = self.models[self.model_name]['description']
        combined_html = f"{description}<hr/><h3>Equation:</h3>{formatted_equation}"
        self.textEdit.setHtml(combined_html)
        self.textEdit.setVisible(True)  # Make textEdit visible by default

        # Enable link clicking in the textEdit widget
        self.textEdit.setOpenLinks(False)
        self.textEdit.anchorClicked.connect(self.onEquationClicked)

        self.actionFormulaChanged.triggered.connect(self.onEquationChanged)
        self.actionModelChanged.triggered.connect(self.onModelChanged)
        self.actionLoadModelFile.triggered.connect(self.onLoadModelFile)
        self.actionEdit_model_file.triggered.connect(self.onEdit_model_file)

        # Connect to the destroyed signal to clean up temp files
        self.destroyed.connect(self.cleanup_temp_files)

    def load_model_file(self, filename: pathlib.Path):
        with io.open_maybe_zipped(filename, 'r') as fp:
            self._model_file = filename
            self.models = yaml.safe_load(fp)
            self.lineEdit.setText(str(filename.as_posix()))

    def onLoadModelFile(self, filename: pathlib.Path = None):
        if filename is None:
            filename = chisurf.gui.widgets.get_filename("Model-YAML file", file_type='*.yaml')
        self.load_model_file(filename)

    def onEdit_model_file(self):
        self.editor.load_file(self._model_file)
        self.editor.show()

    @property
    def models(self) -> typing.Dict:
        return self._models

    @models.setter
    def models(self, v: typing.Dict):
        self._models = v
        self.comboBox.clear()
        self.comboBox.addItems(list(v.keys()))

    @property
    def model_name(self) -> typing.List[str]:
        return list(self.models.keys())[self.comboBox.currentIndex()]

    @model_name.setter
    def model_name(self, v: str):
        idx = self.comboBox.findText(v)
        self.comboBox.setCurrentIndex(idx)

    @property
    def model_file(self) -> str:
        return self._model_file

    @model_file.setter
    def model_file(self, v: str):
        self._model_file = v
        self.load_model_file(v)

    def set_default_parameter_values(self, model_name: str = None):
        if model_name is None:
            model_name = self.model_name
        ivs = self.models[model_name]['initial']
        for key in ivs.keys():
            self.model.parameter_dict[key].value = float(ivs[key])

    def onUpdateFunc(self):
        fit_idx = chisurf.fitting.find_fit_idx_of_model(model=self.model)
        function_str = str(self.plainTextEdit.toPlainText()).strip()
        chisurf.run(f"chisurf.macros.model_parse.change_model('{function_str}', {fit_idx})")

    def onModelChanged(self):
        func = self.models[self.model_name]['equation']
        self.plainTextEdit.setPlainText(func)

        # Format the equation
        formatted_equation = self.format_equation(func)

        # Combine description and formatted equation in textEdit
        description = self.models[self.model_name]['description']
        combined_html = f"{description}<hr/><h3>Equation:</h3>{formatted_equation}"
        self.textEdit.setHtml(combined_html)
        self.textEdit.setVisible(True)  # Make sure textEdit is visible

        # Update the equation in the dialog if it's visible (keeping for backward compatibility)
        if self.equationDialog is not None and self.equationDialog.isVisible():
            self.equationDialog.setEquation(formatted_equation)
            # Update the dialog title with the new model name
            self.equationDialog.setWindowTitle(f"Equation: {self.model_name}")

        self.onEquationChanged()

    def create_parameter_widgets(self):
        layout = self.gridLayout_1
        chisurf.gui.widgets.clear_layout(layout)
        n_columns = self.n_columns
        row = 1
        p_eq = self.model._parameters_equation
        for i, p in enumerate(p_eq):
            pw = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(p)
            column = i % n_columns
            if column == 0:
                row += 1
            layout.addWidget(pw, row, column)

    def render_math_equation(self, equation: str, large: bool = False) -> QtGui.QPixmap:
        """
        Render a math equation using matplotlib's built-in mathtext and return a QPixmap.

        This method takes a TeX-formatted equation (already processed by sympy or other means)
        and renders it using matplotlib's built-in mathtext.

        Args:
            equation (str): The TeX-formatted equation to render
            large (bool): Whether to render a larger version of the equation

        Returns:
            QtGui.QPixmap: The rendered equation as a QPixmap
        """
        # No need for regex transformations here as the equation should already be in TeX format
        # from sympy conversion

        # Create a formatted equation
        math_equation = f"${equation}$"

        # Set up matplotlib to use built-in mathtext instead of full LaTeX
        rcParams['text.usetex'] = False
        rcParams['mathtext.default'] = 'regular'

        # Create a figure with transparent background
        if large:
            # Larger figure for the dialog
            fig = plt.figure(figsize=(5, 2), dpi=100)  # Larger size for the dialog
        else:
            # Normal figure for the inline display
            fig = plt.figure(figsize=(1.2, 0.5), dpi=100)  # Max width 120 pixels (1.2 inches at 100 dpi)

        fig.patch.set_alpha(1.0)  # Transparent

        # Add the equation as text with black color
        plt.text(0.5, 0.5, math_equation, 
                 fontsize=24 if large else 12,  # Larger font size for the dialog
                 horizontalalignment='center', verticalalignment='center',
                 color='black')  # Explicitly set text color to black

        # Remove axes
        plt.axis('off')

        # Save to a BytesIO object
        buf = python_io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=False)
        plt.close(fig)

        # Convert to QPixmap
        buf.seek(0)
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(buf.getvalue())

        return pixmap

    def convert_python_to_tex(self, equation: str) -> str:
        """
        Convert a Python equation to a TeX-like equation using sympy.

        This method uses sympy to parse a Python equation and convert it to a TeX-like
        equation. It handles Python's power operator ** automatically, as sympy understands
        this operator. It also handles common mathematical functions like sin, cos, etc.

        Args:
            equation (str): The Python equation to convert

        Returns:
            str: The TeX-like equation, or the original equation if conversion fails
        """
        try:
            # Find all variable names in the equation
            # This regex finds all alphanumeric identifiers that are not part of function calls
            # It will match variable names like x, y, alpha, beta1, etc.
            variables = set(re.findall(r'(?<![a-zA-Z0-9_])([a-zA-Z][a-zA-Z0-9_]*)(?!\()', equation))
            # Remove known function names and constants
            function_names = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'pi', 'np'}
            variables = variables - function_names

            # Create symbols for all variables
            symbols = {var: sympy.symbols(var) for var in variables}

            # Create a comprehensive dictionary of mathematical functions and constants
            math_dict = {
                # Basic sympy functions
                'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
                'exp': sympy.exp, 'log': sympy.log, 'sqrt': sympy.sqrt,
                'pi': sympy.pi, 'E': sympy.E, 'I': sympy.I,

                # Additional sympy functions
                'asin': sympy.asin, 'acos': sympy.acos, 'atan': sympy.atan,
                'sinh': sympy.sinh, 'cosh': sympy.cosh, 'tanh': sympy.tanh,
                'asinh': sympy.asinh, 'acosh': sympy.acosh, 'atanh': sympy.atanh,
                'factorial': sympy.factorial, 'gamma': sympy.gamma,
                'abs': sympy.Abs, 'erf': sympy.erf,

                # Constants
                'inf': sympy.oo, 'nan': sympy.nan,

                # Numpy module (for np.* functions)
                'np': np
            }

            # Combine symbols and math functions
            locals_dict = {**symbols, **math_dict}

            # Replace numpy functions with sympy functions
            equation = equation.replace('np.', '')

            # Parse the equation using sympy's sympify function if possible
            try:
                expr = sympy.sympify(equation)
                tex_equation = sympy.latex(expr)
                return tex_equation
            except Exception:
                # If sympify fails, fall back to eval
                try:
                    expr = eval(equation, {"__builtins__": {}}, locals_dict)
                    tex_equation = sympy.latex(expr)
                    return tex_equation
                except Exception as e:
                    print(f"Error converting equation to sympy expression: {e}")
                    return equation

        except Exception as e:
            print(f"Error in convert_python_to_tex: {e}")
            return equation

    def cleanup_temp_files(self):
        """Clean up temporary files created for equation rendering."""
        # Clean up temporary files
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error cleaning up temporary file {temp_file}: {e}")
        self._temp_files = []

        # Close the equation dialog if it's open (for backward compatibility)
        if self.equationDialog is not None:
            self.equationDialog.close()
            self.equationDialog = None

    def format_equation(self, equation: str) -> str:
        """
        Format the equation nicely using matplotlib's built-in mathtext and sympy.

        This method converts the equation to TeX using sympy, which handles Python's
        power operator ** automatically. It then renders the equation using matplotlib's
        built-in mathtext and returns an HTML string that displays the rendered equation.

        If sympy conversion or mathtext rendering fails, it falls back to simple HTML formatting.

        Args:
            equation (str): The equation to format

        Returns:
            str: HTML string that displays the formatted equation
        """
        try:
            # Convert the equation to TeX using sympy
            tex_equation = self.convert_python_to_tex(equation)
            # Use the TeX equation for rendering
            equation = tex_equation

            # Render with matplotlib's mathtext
            pixmap = self.render_math_equation(equation)

            # Create a temporary file to save the image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                pixmap.save(tmp.name, 'PNG')
                tmp_path = tmp.name

            # Create HTML that displays the image with transparent background and make it clickable
            # Include the original equation in the URL so we can render it again in the dialog
            formatted_equation = f'<div style="text-align: center; display: inline-block; margin: 0 auto;"><a href="equation://{tmp_path}?eq={equation}"><img src="{tmp_path}" style="max-width: 120px;" /></a></div>'

            # Store the path to delete it later
            self._temp_files.append(tmp_path)

            return formatted_equation
        except Exception as e:
            # Fallback to simple HTML formatting if mathtext rendering fails
            print(f"Mathtext rendering failed: {e}")

            # Replace ** with ^ (Python power operator)
            # Handle various forms of power expressions
            equation = re.sub(r'(\w+)\*\*(\d+)', r'\1<sup>\2</sup>', equation)
            equation = re.sub(r'(\w+)\*\*([a-zA-Z]+)', r'\1<sup>\2</sup>', equation)
            equation = re.sub(r'(\([^)]+\))\*\*(\d+)', r'\1<sup>\2</sup>', equation)
            equation = re.sub(r'(\w+)\*\*(\([^)]+\))', r'\1<sup>\2</sup>', equation)
            # Handle any remaining ** operators
            equation = re.sub(r'\*\*(\w+)', r'<sup>\1</sup>', equation)
            equation = re.sub(r'\*\*(\([^)]+\))', r'<sup>\1</sup>', equation)

            # Replace ^ with superscript
            equation = re.sub(r'\^(\d+)', r'<sup>\1</sup>', equation)
            equation = re.sub(r'\^([a-zA-Z]+)', r'<sup>\1</sup>', equation)

            # Replace _ with subscript
            equation = re.sub(r'_(\d+)', r'<sub>\1</sub>', equation)
            equation = re.sub(r'_([a-zA-Z]+)', r'<sub>\1</sub>', equation)

            # Replace * with × (multiplication symbol)
            equation = equation.replace('*', ' × ')

            # Replace / with ÷ or a fraction
            equation = equation.replace('/', ' ÷ ')

            # Add spacing around operators
            equation = re.sub(r'([+\-])', r' \1 ', equation)

            # Wrap the equation in a div with styling (transparent background, black text) and make it clickable
            # Use a special URL scheme to indicate that this is a fallback equation
            formatted_equation = f'<div style="text-align: center; display: inline-block; margin: 0 auto;"><a href="fallback://{equation}"><div style="font-size: 11pt; font-family: Times; text-align: center; color: black; display: inline-block; margin: 0 auto; max-width: 120px;">{equation}</div></a></div>'
            return formatted_equation

    def onShowEquation(self, checked: bool = None):
        """Toggle the visibility of the equation and description in the textEdit field."""
        # Toggle the visibility of the textEdit field
        self.textEdit.setVisible(self.toolButton_4.isChecked())

        # If making visible, ensure the equation is up to date
        if self.textEdit.isVisible():
            # Get the equation from the text edit
            equation = self.plainTextEdit.toPlainText()
            formatted_equation = self.format_equation(equation)

            # Update the textEdit with the equation and description
            description = self.models[self.model_name]['description']
            combined_html = f"{description}<hr/><h3>Equation:</h3>{formatted_equation}"
            self.textEdit.setHtml(combined_html)

        # For backward compatibility, also handle the dialog if it exists
        if self.equationDialog is not None and self.equationDialog.isVisible():
            self.equationDialog.close()

    def _on_dialog_closed(self):
        """Called when the equation dialog is closed."""
        # Set the dialog to None so it can be garbage collected
        self.equationDialog = None

    def onEquationClicked(self, url):
        """Handle click on the equation image."""
        if url.scheme() == "equation":
            # Extract the file path from the URL
            file_path = url.path()
            # On Windows, the path starts with a '/', so we need to remove it
            if file_path.startswith('/'):
                file_path = file_path[1:]

            # Extract the equation from the query string if present
            equation = None
            query = url.query()
            if query:
                # Parse the query string
                query_items = query.split('&')
                for item in query_items:
                    if item.startswith('eq='):
                        equation = item[3:]  # Remove 'eq=' prefix
                        break

            self.showLargeEquation(file_path, equation)
        elif url.scheme() == "fallback":
            # Extract the equation from the URL
            equation = url.path()
            # On Windows, the path starts with a '/', so we need to remove it
            if equation.startswith('/'):
                equation = equation[1:]
            self.showLargeFallbackEquation(equation)

    def showLargeEquation(self, image_path, equation=None):
        """Show a larger version of the equation in a dialog."""
        # Create a new dialog if it doesn't exist
        if self.equationDialog is None:
            self.equationDialog = EquationDialog(self, f"Equation: {self.model_name}")
            self.equationDialog.finished.connect(self._on_dialog_closed)

        # If equation is not provided, get it from the plainTextEdit
        if equation is None:
            equation = self.plainTextEdit.toPlainText()

        # Convert the equation to TeX using sympy
        tex_equation = self.convert_python_to_tex(equation)

        # Render a larger version of the equation
        pixmap = self.render_math_equation(tex_equation, large=True)

        # Save the larger image to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            pixmap.save(tmp.name, 'PNG')
            tmp_path = tmp.name

        # Store the path to delete it later
        self._temp_files.append(tmp_path)

        # Create HTML that displays the larger image
        html = f'<div style="text-align: center;"><img src="{tmp_path}" /></div>'

        # Set the HTML content of the dialog
        self.equationDialog.setEquation(html)

        # Resize the dialog to fit the image plus some padding
        # Get the size of the pixmap
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Add padding for the dialog frame, close button, and margins
        dialog_width = pixmap_width + 40  # 20px padding on each side
        dialog_height = pixmap_height + 80  # 40px for close button and margins

        # Set the size of the dialog
        self.equationDialog.resize(dialog_width, dialog_height)

        # Show the dialog
        self.equationDialog.show()

    def showLargeFallbackEquation(self, equation):
        """Show a larger version of the fallback equation in a dialog."""
        # Create a new dialog if it doesn't exist
        if self.equationDialog is None:
            self.equationDialog = EquationDialog(self, f"Equation: {self.model_name}")
            self.equationDialog.finished.connect(self._on_dialog_closed)

        # Create a larger version of the equation with a larger font size
        # Apply the same formatting as in the format_equation method

        # Replace ** with ^ (Python power operator)
        # Handle various forms of power expressions
        equation = re.sub(r'(\w+)\*\*(\d+)', r'\1<sup>\2</sup>', equation)
        equation = re.sub(r'(\w+)\*\*([a-zA-Z]+)', r'\1<sup>\2</sup>', equation)
        equation = re.sub(r'(\([^)]+\))\*\*(\d+)', r'\1<sup>\2</sup>', equation)
        equation = re.sub(r'(\w+)\*\*(\([^)]+\))', r'\1<sup>\2</sup>', equation)
        # Handle any remaining ** operators
        equation = re.sub(r'\*\*(\w+)', r'<sup>\1</sup>', equation)
        equation = re.sub(r'\*\*(\([^)]+\))', r'<sup>\1</sup>', equation)

        # Replace ^ with superscript
        equation = re.sub(r'\^(\d+)', r'<sup>\1</sup>', equation)
        equation = re.sub(r'\^([a-zA-Z]+)', r'<sup>\1</sup>', equation)

        # Replace _ with subscript
        equation = re.sub(r'_(\d+)', r'<sub>\1</sub>', equation)
        equation = re.sub(r'_([a-zA-Z]+)', r'<sub>\1</sub>', equation)

        # Replace * with × (multiplication symbol)
        equation = equation.replace('*', ' × ')

        # Replace / with ÷ or a fraction
        equation = equation.replace('/', ' ÷ ')

        # Add spacing around operators
        equation = re.sub(r'([+\-])', r' \1 ', equation)

        # Create HTML that displays the larger equation with a larger font size
        html = f'<div style="font-size: 24pt; font-family: Times; text-align: center; color: black; padding: 20px;">{equation}</div>'

        # Set the HTML content of the dialog
        self.equationDialog.setEquation(html)

        # Estimate the size of the equation based on the length of the text
        # This is a rough estimate and may need adjustment
        char_width = 15  # Average width of a character in pixels at 24pt
        char_height = 30  # Height of a line in pixels at 24pt

        # Calculate the width and height based on the equation length
        # Add some extra width for formatting (superscripts, subscripts, etc.)
        estimated_width = min(max(len(equation) * char_width, 300), 800)  # Min 300px, max 800px
        estimated_height = char_height + 40  # Add padding

        # Add padding for the dialog frame, close button, and margins
        dialog_width = estimated_width + 40  # 20px padding on each side
        dialog_height = estimated_height + 80  # 40px for close button and margins

        # Set the size of the dialog
        self.equationDialog.resize(dialog_width, dialog_height)

        # Show the dialog
        self.equationDialog.show()

    def onEquationChanged(self):
        self.onUpdateFunc()
        self.set_default_parameter_values()
        self.create_parameter_widgets()
        self.model.update_model()

        # Get the new equation and format it
        equation = self.plainTextEdit.toPlainText()
        formatted_equation = self.format_equation(equation)

        # Update the textEdit with the new equation while preserving the description
        description = self.models[self.model_name]['description']
        combined_html = f"{description}<hr/><h3>Equation:</h3>{formatted_equation}"
        self.textEdit.setHtml(combined_html)
        self.textEdit.setVisible(True)  # Make sure textEdit is visible

        # Update the equation in the dialog if it's visible (keeping for backward compatibility)
        if self.equationDialog is not None and self.equationDialog.isVisible():
            self.equationDialog.setEquation(formatted_equation)



class ParseModelWidget(ParseModel, ModelWidget):

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            *args,
            **kwargs
    ):
        super().__init__(fit, *args, **kwargs)
        parse = ParseFormulaWidget(
            model=self,
            model_file=kwargs.get('model_file', None)
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(parse)
        self.setLayout(layout)
        self.parse = parse

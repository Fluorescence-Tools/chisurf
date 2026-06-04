import sys
import os
import time
import pathlib
import csv
import tempfile
import shutil

from qtpy import QtWidgets, QtCore
from qtpy.QtWidgets import (
    QApplication, QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QProgressBar, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QDialog, QTableWidget, QTableWidgetItem, QLineEdit, QAbstractItemView,
)
from qtpy.QtCore import Qt
import chisurf  # your chisurf module with fits, macros, etc.


# Custom QListWidget that supports drag-and-drop.
class FileListWidget(QListWidget):
    def __init__(self, parent=None):
        """Initialize the list widget as a file drop target.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        # allow external drops
        self.setAcceptDrops(True)
        # permit drops (but not internal drags)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        # make it clear we’re copying files in
        self.setDefaultDropAction(Qt.CopyAction)
        self.setDropIndicatorShown(True)

    def dragEnterEvent(self, event):
        """Accept the drag if it carries file URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Accept the drag-move if it carries file URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Add dropped file paths as new (checked) list items."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path):
                    item = QListWidgetItem(file_path)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked)
                    self.addItem(item)
            event.acceptProposedAction()
        else:
            event.ignore()


# Dialog to show progress during file processing
class ProgressWindow(QDialog):
    def __init__(self, title="Processing Files", message="Loading files...", max_value=100, parent=None):
        """Create a modal progress dialog with a label and a progress bar.

        Parameters
        ----------
        title : str, optional
            Window title.
        message : str, optional
            Initial label text.
        max_value : int, optional
            Upper bound of the progress bar.
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.WindowModal)
        layout = QVBoxLayout()
        self.label = QLabel(message)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max_value)
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def set_value(self, value: int):
        """Update the progress bar and pump the event loop.

        Parameters
        ----------
        value : int
            New value (0..max_value) for the progress bar.
        """
        self.progress_bar.setValue(value)
        QApplication.processEvents()


# The final results page that displays the CSV in a table
class ResultsPage(QWizardPage):
    def __init__(self, parent=None):
        """Initialize the results page with an empty table.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setTitle("Step 3: Fit Results")
        self.setSubTitle("The results of the fits are displayed below.")
        layout = QVBoxLayout()
        self.results_table = QTableWidget()
        layout.addWidget(self.results_table)
        self.setLayout(layout)

    def initializePage(self):
        """Load the CSV produced by the analysis page into the results table."""
        csv_filename = self.wizard().fit_results_file
        if not csv_filename or not os.path.exists(csv_filename):
            QMessageBox.warning(self, "No Results", "No CSV file was found with the fit results.")
            return

        with open(csv_filename, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        # Now there are six columns: Run, Filename, Parameter, Fixed, Value, Chi2r
        self.results_table.clear()
        self.results_table.setColumnCount(6)
        self.results_table.setRowCount(len(rows))
        self.results_table.setHorizontalHeaderLabels(["Run", "Filename", "Parameter", "Fixed", "Value", "Chi2r"])

        for row_idx, row in enumerate(rows):
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(row["Run"]))
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(row["Filename"]))
            self.results_table.setItem(row_idx, 2, QTableWidgetItem(row["Parameter"]))
            self.results_table.setItem(row_idx, 3, QTableWidgetItem(row["Fixed"]))
            self.results_table.setItem(row_idx, 4, QTableWidgetItem(str(row["Value"])))
            self.results_table.setItem(row_idx, 5, QTableWidgetItem(str(row["Chi2r"])))


# The main wizard which includes all pages
class BatchProcessingWizard(QWizard):
    def __init__(self):
        """Initialize the wizard and register its pages."""
        super().__init__()
        self.setWindowTitle("Batch Processing Wizard")
        # Smaller window size: width 600, height 400
        self.setGeometry(100, 100, 700, 500)
        self.setWizardStyle(QWizard.ModernStyle)

        # Capture page IDs for navigation control
        self.pid_welcome = self.addPage(WelcomePage(self))
        # New optional page to select already loaded datasets
        self.loaded_data_selection_page = LoadedDataSelectionPage(self)
        self.pid_loaded = self.addPage(self.loaded_data_selection_page)
        # Page to add files and select fit
        self.file_and_fit_selection_page = FileAndFitSelectionPage(self)
        self.pid_file = self.addPage(self.file_and_fit_selection_page)
        # Analysis and results pages
        self.analysis_page = AnalysisPage(self)
        self.pid_analysis = self.addPage(self.analysis_page)
        self.results_page = ResultsPage(self)
        self.pid_results = self.addPage(self.results_page)

        # Storage for results and selections
        self.fit_results_file = ""
        self.selected_loaded_datasets = []

        # Hook page change to enforce type check when navigating
        self._block_page_change = False
        self.currentIdChanged.connect(self._on_current_id_changed)

    def _datasets_have_mixed_types(self, datasets):
        """Return True when the selected datasets have more than one experiment class.

        Parameters
        ----------
        datasets : list
            List of ChiSurf dataset objects.

        Returns
        -------
        bool
            ``True`` if there is more than one experiment type.
        """
        try:
            if len(datasets) <= 1:
                return False
            classes = set()
            for ds in datasets:
                try:
                    exp = getattr(ds, 'experiment', None)
                    classes.add(type(exp))
                except Exception:
                    classes.add(type(None))
            return len(classes) > 1
        except Exception:
            # On any unexpected issue, do not block navigation
            return False

    def _on_current_id_changed(self, new_id: int):
        """Block navigation if the user picked datasets of different experiment types.

        Parameters
        ----------
        new_id : int
            The page ID that the wizard is about to switch to.
        """
        # Avoid re-entrancy when we programmatically change pages
        if self._block_page_change:
            return
        try:
            # Only enforce when navigating to pages after the loaded selection page
            if new_id in (self.pid_file, self.pid_analysis, self.pid_results):
                sel = list(getattr(self, 'selected_loaded_datasets', []) or [])
                if self._datasets_have_mixed_types(sel):
                    # Warn and send user back to the selection page
                    self._block_page_change = True
                    try:
                        type_names = sorted({type(getattr(ds, 'experiment', None)).__name__ for ds in sel})
                        QMessageBox.warning(
                            self,
                            "Mixed Experiment Types",
                            "Please select datasets of the same experiment type.\nFound types: " + ", ".join(type_names)
                        )
                    finally:
                        # Return to selection page
                        self.setCurrentId(self.pid_loaded)
                        self._block_page_change = False
        except Exception:
            # Fail-safe: do nothing on errors to avoid locking the wizard
            self._block_page_change = False


class WelcomePage(QWizardPage):
    def __init__(self, parent=None):
        """Initialize the welcome page with introductory instructions.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setTitle("Welcome to the Batch Processing Wizard")
        self.setSubTitle("Introduction")
        layout = QVBoxLayout()
        info_label = QLabel(
            "This wizard helps you process files in batch.\n\n"
            "Follow these steps:\n"
            "1. Load the data that you want to process in ChiSurf.\n"
            "2. Create a fit in ChiSurf for one of the loaded data sets.\n"
            "3. Open / reopen the batch processing wizard.\n"
            "IMPORTANT:\n"
            "The initial parameter values are taken from the template fit. "
            "Before batch processing, you should manually optimize the parameters of "
            "this template fit using data similar to the files you plan to process. "
            "This ensures that the batch results are reliable and meaningful.\n\n"
            "Click 'Next' to proceed."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        self.setLayout(layout)


class LoadedDataSelectionPage(QWizardPage):
    def __init__(self, parent=None):
        """Initialize the page that lets the user pick already loaded datasets.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setTitle("Step 1: Select Already Loaded Data (optional)")
        self.setSubTitle("Select already loaded datasets to process. Leave empty to add files in the next step.")
        layout = QVBoxLayout()
        info = QLabel("If you have already loaded data in ChiSurf, you can select them here and skip reloading from files.")
        info.setWordWrap(True)
        layout.addWidget(info)
        self.loaded_list = QListWidget()
        layout.addWidget(self.loaded_list)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.populate_loaded_list)
        layout.addWidget(self.refresh_button)
        self.setLayout(layout)

    def initializePage(self):
        """Populate the list of available datasets when the page becomes active."""
        self.populate_loaded_list()

    def populate_loaded_list(self):
        """Populate the list widget with the currently imported datasets (excluding the global one)."""
        self.loaded_list.clear()
        try:
            for idx, ds in enumerate(chisurf.imported_datasets):
                # Skip the global dataset from being listed/selectable
                ds_name_attr = getattr(ds, 'name', None)
                if isinstance(ds_name_attr, str) and ds_name_attr == 'Global Dataset':
                    continue
                # Prefer explicit name, fallback to filename, then a generic label
                name = ds_name_attr or getattr(ds, 'filename', None) or f"Dataset {idx+1}"
                text = f"{idx+1}. {name}"
                item = QListWidgetItem(text)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                item.setData(Qt.UserRole, ds)
                self.loaded_list.addItem(item)
        except Exception as e:
            print(f"Error populating loaded datasets: {e}")

    def get_selected_loaded_datasets(self):
        """Return the dataset objects selected by the user.

        Returns
        -------
        list
            The list of dataset objects whose items are checked.
        """
        selected = []
        for i in range(self.loaded_list.count()):
            item = self.loaded_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected

    def validatePage(self):
        """Ensure all selected datasets share the same experiment type before proceeding.

        Returns
        -------
        bool
            ``True`` if the selection is valid (or empty); ``False`` on mixed types.
        """
        try:
            selected = self.get_selected_loaded_datasets()
            # Validate that all selected datasets are of the same experiment type
            if len(selected) > 1:
                def exp_cls(ds):
                    """Return the type of the dataset's experiment, or ``type(None)`` on failure.

                    Parameters
                    ----------
                    ds : object
                        A ChiSurf dataset.

                    Returns
                    -------
                    type
                        Experiment type, or ``type(None)`` if unavailable.
                    """
                    try:
                        exp = getattr(ds, 'experiment', None)
                        return type(exp)
                    except Exception:
                        return type(None)
                classes = {exp_cls(ds) for ds in selected}
                if len(classes) > 1:
                    # Build a readable list of type names for the message
                    type_names = sorted({cls.__name__ if cls is not None else 'None' for cls in classes})
                    QMessageBox.warning(
                        self,
                        "Mixed Experiment Types",
                        "Please select datasets of the same experiment type.\nFound types: " + ", ".join(type_names)
                    )
                    return False
            # Store selection on the wizard
            self.wizard().selected_loaded_datasets = selected
        except Exception:
            # In case of unexpected issues, allow proceeding without selection
            self.wizard().selected_loaded_datasets = []
        return True

class FileAndFitSelectionPage(QWizardPage):
    def __init__(self, parent=None):
        """Initialize the file-and-fit selection page widgets.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setTitle("Step 2: Select Files and Fit")
        self.setSubTitle("Add the files you want to process and select the fitting method to use.")
        self.layout = QVBoxLayout()

        self.file_list_label = QLabel("Drag and Drop Files Here:")
        self.file_list_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.file_list_label)

        # Use the custom FileListWidget here.
        self.file_list = FileListWidget()
        # Connect double-click to remove the file.
        self.file_list.itemDoubleClicked.connect(self.remove_item)
        self.layout.addWidget(self.file_list)

        self.file_button = QPushButton("Add Files...")
        self.file_button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.file_button)

        self.fit_combo_box = QComboBox()
        self.populate_fit_combo_box()
        self.fit_combo_box.setToolTip("Select a fit method for analysis.")
        self.layout.addWidget(QLabel("Select a Fit:"))
        self.layout.addWidget(self.fit_combo_box)

        self.setLayout(self.layout)

    def remove_item(self, item):
        """Remove the double-clicked item from the file list."""
        row = self.file_list.row(item)
        self.file_list.takeItem(row)

    def open_file_dialog(self):
        """Open a file dialog and add the chosen file paths to the list as checked items."""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Files")
        for file_path in file_paths:
            item = QListWidgetItem(file_path)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.file_list.addItem(item)

    def validatePage(self):
        """Require both a fit selection and at least one file (or loaded dataset).

        Returns
        -------
        bool
            ``True`` if the page is valid and the wizard may advance.
        """
        fit_ok = bool(self.fit_combo_box.currentText())
        has_files = self.file_list.count() > 0
        loaded = []
        try:
            loaded = list(getattr(self.wizard(), 'selected_loaded_datasets', []) or [])
        except Exception:
            loaded = []
        if not fit_ok:
            QMessageBox.warning(self, "Incomplete Selection", "Please select a fit method before proceeding.")
            return False
        if not has_files and not loaded:
            QMessageBox.warning(
                self, "No Data Selected",
                "Please select already loaded datasets on the previous page or add at least one file."
            )
            return False
        return True

    def get_selected_files(self):
        """Return the file paths whose list items are checked.

        Returns
        -------
        list of str
            Selected file paths.
        """
        selected_files = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_files.append(item.text())
        return selected_files

    def populate_fit_combo_box(self):
        """Populate the fit combo box from ``chisurf.fits`` and try to restore the previous selection."""
        try:
            # Remember previously selected fit object if available
            prev_fit_obj = None
            if self.fit_combo_box.count() > 0:
                prev_fit_obj = self.fit_combo_box.currentData()
            prev_text = self.fit_combo_box.currentText() if self.fit_combo_box.count() > 0 else ""

            self.fit_combo_box.clear()
            for f in chisurf.fits:
                # store the actual fit object for robust matching
                self.fit_combo_box.addItem(f.name, f)

            # Try to restore by object identity first
            if prev_fit_obj is not None:
                for i in range(self.fit_combo_box.count()):
                    if self.fit_combo_box.itemData(i) is prev_fit_obj:
                        self.fit_combo_box.setCurrentIndex(i)
                        break
                else:
                    # Fallback to restoring by text if object not found
                    if prev_text:
                        idx = self.fit_combo_box.findText(prev_text)
                        if idx >= 0:
                            self.fit_combo_box.setCurrentIndex(idx)
            else:
                # No previous object, try by text
                if prev_text:
                    idx = self.fit_combo_box.findText(prev_text)
                    if idx >= 0:
                        self.fit_combo_box.setCurrentIndex(idx)
        except Exception as e:
            print(f"Error loading fits from chisurf.fits: {e}")
            self.fit_combo_box.addItem("No fits available")

    def initializePage(self):
        """Refresh the fit combo box when the page becomes active."""
        # Refresh available fits each time this page becomes active
        self.populate_fit_combo_box()

    def get_selected_fit(self):
        """Return the name of the currently selected fit.

        Returns
        -------
        str
            The display name of the selected fit.
        """
        return self.fit_combo_box.currentText()

    def get_selected_fit_index(self) -> int:
        """Return the index of the selected fit in chisurf.fits using object identity when possible."""
        try:
            fit_obj = self.fit_combo_box.currentData()
            if fit_obj is not None:
                for idx, f in enumerate(chisurf.fits):
                    if f is fit_obj:
                        return idx
        except Exception:
            pass
        # Fallback to combobox index if object resolution failed
        try:
            return self.fit_combo_box.currentIndex()
        except Exception:
            return -1


class AnalysisPage(QWizardPage):
    def __init__(self, parent=None):
        """Initialize the analysis page widgets and per-run state.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setTitle("Step 3: Run Fits")
        self.setSubTitle("Run fits on your selected data. Fit results will be saved to a CSV file.")
        layout = QVBoxLayout()

        self.analysis_label = QLabel("Click 'Run Fits' to start processing. Progress will be shown below.")
        self.analysis_label.setWordWrap(True)
        layout.addWidget(self.analysis_label)

        # UI for selecting where to save the results file.
        save_file_layout = QHBoxLayout()
        save_file_label = QLabel("Results File:")
        self.save_file_line_edit = QLineEdit()
        # Leave the line edit empty by default.
        self.save_file_line_edit.setText("")
        self.save_file_button = QPushButton("Browse...")
        self.save_file_button.clicked.connect(self.browse_save_file)
        save_file_layout.addWidget(save_file_label)
        save_file_layout.addWidget(self.save_file_line_edit)
        save_file_layout.addWidget(self.save_file_button)
        layout.addLayout(save_file_layout)

        self.run_fits_button = QPushButton("Run Fits")
        self.run_fits_button.clicked.connect(self.run_fits)
        layout.addWidget(self.run_fits_button)

        # A list to show processed items as they complete.
        self.results_list = QListWidget()
        layout.addWidget(QLabel("Processed Data:"))
        layout.addWidget(self.results_list)

        self.setLayout(layout)
        # This will store all results as a list of dictionaries.
        self.results = []
        # Temp dir for screenshots and map from filename to saved path
        self._temp_dir = None
        self._screenshot_map = {}
        # Temp dir for per-fit exports (numeric results)
        self._fit_exports_dir = None

    def browse_save_file(self):
        """Open a file dialog to choose where the results CSV should be written."""
        filename, _ = QFileDialog.getSaveFileName(self, "Save Results File", "", "CSV Files (*.csv);;All Files (*)")
        if filename:
            self.save_file_line_edit.setText(filename)

    def dummy_run_fit(self, file, fit_idx):
        """
        Run the fit for a file.
        This function calls chisurf to load the file and update the chosen fit's data.
        """
        file_path = pathlib.Path(file).as_posix().replace("\\", "/")
        chisurf.actions.dispatch(
            name="dataset.add",
            payload={"filename": file_path, "experiment_reader": None},
        )
        chisurf.actions.dispatch(
            name="fit.set_dataset",
            payload={
                "fit_index": int(fit_idx),
                "dataset_index": -1,
            },
        )
        chisurf.actions.dispatch(
            name="fit.run",
            payload={"fit_index": int(fit_idx)},
        )
        print(f"Running fit on: {file}")

    def _get_or_create_temp_dir(self) -> str:
        """Return the existing screenshot temp dir, creating it on first use.

        Returns
        -------
        str
            Absolute path of the screenshot temp directory.
        """
        if self._temp_dir and os.path.isdir(self._temp_dir):
            return self._temp_dir
        self._temp_dir = tempfile.mkdtemp(prefix="chisurf_batch_")
        return self._temp_dir

    def _get_or_create_fit_exports_dir(self) -> str:
        """Return the existing per-run fit exports temp dir, creating it on first use.

        Returns
        -------
        str
            Absolute path of the per-run fit exports directory.
        """
        if self._fit_exports_dir and os.path.isdir(self._fit_exports_dir):
            return self._fit_exports_dir
        # Keep all per-run fit result exports in a dedicated temp dir
        base = tempfile.mkdtemp(prefix="chisurf_batch_fit_exports_")
        self._fit_exports_dir = base
        return base

    def _sanitize_filename(self, name: str) -> str:
        """Return a filename-safe version of ``name`` (no extension).

        Parameters
        ----------
        name : str
            Original filename.

        Returns
        -------
        str
            Sanitized base name (no path, no extension), or ``"file"`` if empty.
        """
        # Keep base name without extension, replace problematic chars
        base = pathlib.Path(name).stem
        safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in base)
        return safe or "file"

    def _norm_key(self, file_path: str) -> str:
        """Create a stable, absolute key for grouping results/screenshots."""
        try:
            return str(pathlib.Path(file_path).resolve())
        except Exception:
            return os.path.abspath(file_path)

    def _find_target_window(self):
        """Return the best window to capture for screenshots, falling back to the wizard itself.

        Returns
        -------
        QWidget
            A visible top-level widget, or ``self.wizard()`` if nothing matches.
        """
        # Try to find a likely main window to capture; fallback to wizard itself
        try:
            for w in QtWidgets.QApplication.topLevelWidgets():
                title = w.windowTitle() if hasattr(w, 'windowTitle') else ''
                if w.isVisible() and ("Chi" in title or "Fit" in title or "PCH" in title or "FIDA" in title):
                    return w
            aw = QtWidgets.QApplication.activeWindow()
            if aw and aw.isVisible():
                return aw
        except Exception:
            pass
        return self.wizard()

    def _capture_screenshot_for_file(self, file: str, run_index: int) -> str:
        """Capture a PNG screenshot of the current fit window for a file/run.

        Parameters
        ----------
        file : str
            Original filename used to label the screenshot.
        run_index : int
            1-based run index used in the screenshot filename.

        Returns
        -------
        str
            Path to the saved PNG, or an empty string on failure.
        """
        # Ensure UI updates before capture and mimic save_fit target (MDI current subwindow)
        QApplication.processEvents()
        time.sleep(0.05)
        try:
            cs = chisurf.cs
            fit_window = getattr(cs.mdiarea, 'currentSubWindow', lambda: None)()
        except Exception:
            fit_window = None
        widget = fit_window if fit_window is not None else self._find_target_window()
        try:
            pixmap = widget.grab()
            temp_dir = self._get_or_create_temp_dir()
            safe = self._sanitize_filename(file)
            png_path = os.path.join(temp_dir, f"{run_index:03d}_{safe}.png")
            pixmap.save(png_path, 'PNG')
            return png_path
        except Exception as e:
            print(f"Failed to capture screenshot for {file}: {e}")
            return ""

    def _create_docx_report(self, docx_path: str, file_order: list) -> bool:
        """Create a DOCX report with per-file screenshots and a consolidated results table.

        Parameters
        ----------
        docx_path : str
            Path of the DOCX file to write.
        file_order : list
            Ordered list of filenames used to group and order the results.

        Returns
        -------
        bool
            ``True`` if the DOCX was saved successfully.
        """
        try:
            from docx import Document
            from docx.shared import Inches
        except Exception as e:
            QMessageBox.information(self, "DOCX not created", f"python-docx not available: {e}")
            return False
        # Group results by normalized key
        grouped = {}
        for row in self.results:
            key = row.get("GroupKey", self._norm_key(row.get("Filename", "")))
            grouped.setdefault(key, []).append(row)
        doc = Document()
        doc.add_heading('Batch Fit Results', level=0)
        doc.add_paragraph(f"CSV: {os.path.basename(self.wizard().fit_results_file)}")
        
        # First, add per-file headings and screenshots (no tables here)
        for idx, filename in enumerate(file_order, start=1):
            key = self._norm_key(filename)
            doc.add_heading(f"{idx}. {os.path.basename(filename)}", level=1)
            img = self._screenshot_map.get(key, "")
            if img and os.path.exists(img):
                try:
                    doc.add_picture(img, width=Inches(6))
                except Exception as e:
                    doc.add_paragraph(f"[Could not add image: {e}]")
        
        # Build a single consolidated results table for all files
        # Columns: Filename, Parameter, Fixed, Value, Chi2r, Run
        table = None
        # Flatten rows in the order of file_order
        consolidated_rows = []
        for filename in file_order:
            key = self._norm_key(filename)
            rows = grouped.get(key, [])
            if not rows:
                continue
            # Preserve the order as collected during fitting
            for r in rows:
                consolidated_rows.append(r)
        
        if consolidated_rows:
            table = doc.add_table(rows=1, cols=6)
            hdr = table.rows[0].cells
            hdr[0].text = 'Filename'
            hdr[1].text = 'Parameter'
            hdr[2].text = 'Fixed'
            hdr[3].text = 'Value'
            hdr[4].text = 'Chi2r'
            hdr[5].text = 'Run'
            for r in consolidated_rows:
                cells = table.add_row().cells
                cells[0].text = str(r.get('Filename', ''))
                cells[1].text = str(r.get('Parameter', ''))
                cells[2].text = str(r.get('Fixed', ''))
                cells[3].text = str(r.get('Value', ''))
                cells[4].text = str(r.get('Chi2r', ''))
                cells[5].text = str(r.get('Run', ''))
        else:
            doc.add_paragraph('No parameters found.')
        
        try:
            doc.save(docx_path)
            return True
        except Exception as e:
            QMessageBox.warning(self, "DOCX Save Error", f"Could not save DOCX: {e}")
            return False

    def run_fits(self):
        """Run the selected fit on every chosen file/dataset and write the consolidated results."""
        wizard = self.wizard()
        file_selection_page = wizard.file_and_fit_selection_page

        selected_files = file_selection_page.get_selected_files()
        selected_fit_name = file_selection_page.get_selected_fit()
        loaded_datasets = list(getattr(wizard, 'selected_loaded_datasets', []) or [])

        if not selected_fit_name:
            QMessageBox.warning(self, "No Fit Selected", "Please select a fit method before proceeding.")
            return

        if not selected_files and not loaded_datasets:
            QMessageBox.warning(self, "No Data Selected", "Please select already loaded datasets or add files to process.")
            return

        # If the results file field is empty, prompt the user to choose a save location.
        csv_filename = self.save_file_line_edit.text().strip()
        if not csv_filename:
            csv_filename, _ = QFileDialog.getSaveFileName(self, "Save Results File", "",
                                                          "CSV Files (*.csv);;All Files (*)")
            if csv_filename:
                self.save_file_line_edit.setText(csv_filename)
            else:
                QMessageBox.warning(self, "Save File", "Please specify a file to save the results.")
                return

        fit_idx = file_selection_page.get_selected_fit_index()

        # Clear previous results if any.
        self.results_list.clear()
        self.results = []

        # Get the chosen fit and save the initial parameters.
        fit = chisurf.fits[fit_idx]
        initial_params = {param.name: (param.value, param.fixed) for param in fit.model.parameters_all}

        # Build processing queue: first loaded datasets, then files
        items = []
        for ds in loaded_datasets:
            # Derive a human-readable name
            name = getattr(ds, 'name', None) or getattr(ds, 'filename', None) or f"Dataset {len(items)+1}"
            items.append({"kind": "dataset", "value": ds, "name": str(name)})
        for fpath in selected_files:
            items.append({"kind": "file", "value": fpath, "name": fpath})

        # Create and show the progress window.
        progress_window = ProgressWindow(title="Processing Data", message="Running fits...", max_value=100,
                                         parent=self)
        progress_window.show()

        total_items = len(items)
        for i, item in enumerate(items, start=1):
            display_name = item["name"]
            key = self._norm_key(display_name)
            # Restore the initial parameter values before each run
            for param in fit.model.parameters_all:
                if param.name in initial_params:
                    param.value, param.fixed = initial_params[param.name]

            # Run fit depending on item type
            try:
                if item["kind"] == "dataset":
                    ds = item["value"]
                    ds_idx = chisurf.imported_datasets.index(ds)
                    chisurf.actions.dispatch(
                        name="fit.set_dataset",
                        payload={
                            "fit_index": int(fit_idx),
                            "dataset_index": int(ds_idx),
                        },
                    )
                    chisurf.actions.dispatch(
                        name="fit.run",
                        payload={"fit_index": int(fit_idx)},
                    )
                else:
                    # Reuse existing file-based loader
                    self.dummy_run_fit(display_name, fit_idx)
            except Exception as e:
                print(f"Fit run failed for {display_name}: {e}")

            # Save per-run fit results (numeric export)
            try:
                exports_dir = self._get_or_create_fit_exports_dir()
                safe = self._sanitize_filename(display_name)
                base = os.path.join(exports_dir, f"{i:03d}_{safe}")
                fit.save(base, 'csv', save_curves=True)
            except Exception as e:
                print(f"Per-run fit export failed for {display_name}: {e}")

            # Take a screenshot of the fit window after each fit
            try:
                img_path = self._capture_screenshot_for_file(display_name, i)
                if img_path:
                    self._screenshot_map[key] = img_path
            except Exception as e:
                print(f"Screenshot step failed for {display_name}: {e}")

            # Collect results rows
            for param in fit.model.parameters_all:
                result = {
                    "Run": str(i),
                    "Filename": display_name,
                    "GroupKey": key,
                    "Parameter": param.name,
                    "Fixed": "Yes" if param.fixed else "No",
                    "Value": param.value,
                    "Chi2r": fit.chi2r
                }
                self.results.append(result)

            # Update progress and the list view.
            progress = int((i / total_items) * 100)
            progress_window.set_value(progress)
            self.results_list.addItem(display_name)
            time.sleep(0.2)

        progress_window.close()

        # Write the results to the chosen CSV file.
        fieldnames = ["Run", "Filename", "Parameter", "Fixed", "Value", "Chi2r"]
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in self.results:
                writer.writerow(row)

        # Store the CSV filename in the wizard so the final page can access it.
        wizard.fit_results_file = csv_filename

        # Create DOCX report alongside the CSV
        processed_names = [item["name"] for item in items]
        docx_path = os.path.splitext(csv_filename)[0] + ".docx"
        docx_created = self._create_docx_report(docx_path, processed_names)
        if docx_created:
            wizard.fit_results_docx = docx_path

        temp_dir = self._get_or_create_temp_dir()

        # Zip all per-run fit exports to the target folder (next to CSV/DOCX)
        zip_base = os.path.splitext(csv_filename)[0] + "_fit_results"
        zip_out = zip_base + ".zip"
        try:
            exports_dir = self._get_or_create_fit_exports_dir()
            created = shutil.make_archive(zip_base, 'zip', root_dir=exports_dir)
            zip_out = created if created else zip_out
        except Exception as e:
            print(f"Failed to create ZIP of per-run fit results: {e}")

        msg = (
            f"All fits have been completed and results saved."
            f"\nCSV: {csv_filename}"
            f"\nScreenshots saved in: {temp_dir}"
        )
        if docx_created:
            msg += f"\nDOCX report: {docx_path}"
        if zip_out and os.path.exists(zip_out):
            msg += f"\nPer-run fit results ZIP: {zip_out}"
        else:
            msg += "\nPer-run fit results ZIP: [failed to create]"
        QMessageBox.information(self, "Analysis Complete", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    wizard = BatchProcessingWizard()
    wizard.show()
    sys.exit(app.exec())

if __name__ == "plugin":
    wizard = BatchProcessingWizard()
    wizard.show()

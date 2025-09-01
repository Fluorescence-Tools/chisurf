import sys
import os
import time
import pathlib
import csv
import tempfile
import shutil

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QApplication, QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QProgressBar, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QDialog, QTableWidget, QTableWidgetItem, QLineEdit, QAbstractItemView
)
from PyQt5.QtCore import Qt
import chisurf  # your chisurf module with fits, macros, etc.


# Custom QListWidget that supports drag-and-drop.
class FileListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # allow external drops
        self.setAcceptDrops(True)
        # permit drops (but not internal drags)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        # make it clear we’re copying files in
        self.setDefaultDropAction(Qt.CopyAction)
        self.setDropIndicatorShown(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
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
        self.progress_bar.setValue(value)
        QApplication.processEvents()


# The final results page that displays the CSV in a table
class ResultsPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Step 3: Fit Results")
        self.setSubTitle("The results of the fits are displayed below.")
        layout = QVBoxLayout()
        self.results_table = QTableWidget()
        layout.addWidget(self.results_table)
        self.setLayout(layout)

    def initializePage(self):
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
        super().__init__()
        self.setWindowTitle("Batch Processing Wizard")
        # Smaller window size: width 600, height 400
        self.setGeometry(100, 100, 700, 500)
        self.setWizardStyle(QWizard.ModernStyle)

        self.addPage(WelcomePage(self))
        self.file_and_fit_selection_page = FileAndFitSelectionPage(self)
        self.addPage(self.file_and_fit_selection_page)
        self.analysis_page = AnalysisPage(self)
        self.addPage(self.analysis_page)
        self.results_page = ResultsPage(self)
        self.addPage(self.results_page)

        # This will hold the CSV filename where the results are stored.
        self.fit_results_file = ""


class WelcomePage(QWizardPage):
    def __init__(self, parent=None):
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


class FileAndFitSelectionPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Step 1: Select Files and Fit")
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
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Files")
        for file_path in file_paths:
            item = QListWidgetItem(file_path)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.file_list.addItem(item)

    def validatePage(self):
        if self.file_list.count() == 0 or self.fit_combo_box.currentText() == "":
            QMessageBox.warning(
                self, "Incomplete Selection",
                "Please add at least one file and select a fit method before proceeding."
            )
            return False
        return True

    def get_selected_files(self):
        selected_files = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_files.append(item.text())
        return selected_files

    def populate_fit_combo_box(self):
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
        # Refresh available fits each time this page becomes active
        self.populate_fit_combo_box()

    def get_selected_fit(self):
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
        super().__init__(parent)
        self.setTitle("Step 2: Run Fits")
        self.setSubTitle("Run fits on your selected files. Fit results will be saved to a CSV file.")
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

        # A list to show processed file names as they complete.
        self.results_list = QListWidget()
        layout.addWidget(QLabel("Processed Files:"))
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
        filename, _ = QFileDialog.getSaveFileName(self, "Save Results File", "", "CSV Files (*.csv);;All Files (*)")
        if filename:
            self.save_file_line_edit.setText(filename)

    def dummy_run_fit(self, file, fit_idx):
        """
        Run the fit for a file.
        This function calls chisurf to load the file and update the chosen fit's data.
        """
        file_path = pathlib.Path(file).as_posix().replace("\\", "/")
        chisurf.run(f'chisurf.macros.add_dataset(filename=r"{file_path}")')
        chisurf.run(f'chisurf.fits[{fit_idx}].data = chisurf.imported_datasets[-1]')
        chisurf.run(f'chisurf.fits[{fit_idx}].run()')
        print(f"Running fit on: {file}")

    def _get_or_create_temp_dir(self) -> str:
        if self._temp_dir and os.path.isdir(self._temp_dir):
            return self._temp_dir
        self._temp_dir = tempfile.mkdtemp(prefix="chisurf_batch_")
        return self._temp_dir

    def _get_or_create_fit_exports_dir(self) -> str:
        if self._fit_exports_dir and os.path.isdir(self._fit_exports_dir):
            return self._fit_exports_dir
        # Keep all per-run fit result exports in a dedicated temp dir
        base = tempfile.mkdtemp(prefix="chisurf_batch_fit_exports_")
        self._fit_exports_dir = base
        return self._fit_exports_dir

    def _sanitize_filename(self, name: str) -> str:
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
        wizard = self.wizard()
        file_selection_page = wizard.file_and_fit_selection_page

        selected_files = file_selection_page.get_selected_files()
        selected_fit_name = file_selection_page.get_selected_fit()

        if not selected_files:
            QMessageBox.warning(self, "No Files Selected", "Please select at least one file for processing.")
            return

        if not selected_fit_name:
            QMessageBox.warning(self, "No Fit Selected", "Please select a fit method before proceeding.")
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

        # Create and show the progress window.
        progress_window = ProgressWindow(title="Processing Files", message="Running fits...", max_value=100,
                                         parent=self)
        progress_window.show()

        total_files = len(selected_files)
        for i, file in enumerate(selected_files, start=1):
            key = self._norm_key(file)
            # Restore the initial parameter values before each file's fit.
            for param in fit.model.parameters_all:
                if param.name in initial_params:
                    param.value, param.fixed = initial_params[param.name]

            self.dummy_run_fit(file, fit_idx)

            # Save per-run fit results (numeric export)
            try:
                exports_dir = self._get_or_create_fit_exports_dir()
                safe = self._sanitize_filename(file)
                base = os.path.join(exports_dir, f"{i:03d}_{safe}")
                # Use the same API as core_fit.save_fit uses internally
                fit.save(base, 'csv', save_curves=True)
            except Exception as e:
                print(f"Per-run fit export failed for {file}: {e}")

            # Take a screenshot of the fit window after each fit
            try:
                img_path = self._capture_screenshot_for_file(file, i)
                if img_path:
                    self._screenshot_map[key] = img_path
            except Exception as e:
                print(f"Screenshot step failed for {file}: {e}")

            # Access fit parameters via fit.model.parameters_all and chi2r via fit.chi2r.
            for param in fit.model.parameters_all:
                result = {
                    "Run": str(i),
                    "Filename": file,
                    "GroupKey": key,
                    "Parameter": param.name,
                    "Fixed": "Yes" if param.fixed else "No",
                    "Value": param.value,
                    "Chi2r": fit.chi2r
                }
                self.results.append(result)

            # Update progress and the file list view.
            progress = int((i / total_files) * 100)
            progress_window.set_value(progress)
            self.results_list.addItem(file)
            time.sleep(0.2)  # Short pause to keep UI responsive; adjust as needed

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
        docx_path = os.path.splitext(csv_filename)[0] + ".docx"
        docx_created = self._create_docx_report(docx_path, selected_files)
        if docx_created:
            wizard.fit_results_docx = docx_path

        temp_dir = self._get_or_create_temp_dir()

        # Zip all per-run fit exports to the target folder (next to CSV/DOCX)
        zip_base = os.path.splitext(csv_filename)[0] + "_fit_results"
        zip_out = zip_base + ".zip"
        try:
            exports_dir = self._get_or_create_fit_exports_dir()
            # Create the archive; make_archive returns the filename it created
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

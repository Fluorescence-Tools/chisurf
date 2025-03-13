import sys
import os
import time
import pathlib
import csv

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QApplication, QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QProgressBar, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QDialog, QTableWidget, QTableWidgetItem, QLineEdit
)
from PyQt5.QtCore import Qt
import chisurf  # your chisurf module with fits, macros, etc.


# Custom QListWidget that supports drag-and-drop.
class FileListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.NoDragDrop)
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
            "1. Select the files you want to process (drag and drop supported).\n"
            "2. Choose a fit method which will serve as a template for the batch.\n"
            "3. Specify where to save the results.\n"
            "4. Run the fits and review the detailed results.\n\n"
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
            self.fit_combo_box.clear()
            for fit in chisurf.fits:
                self.fit_combo_box.addItem(fit.name)
        except Exception as e:
            print(f"Error loading fits from chisurf.fits: {e}")
            self.fit_combo_box.addItem("No fits available")

    def get_selected_fit(self):
        return self.fit_combo_box.currentText()


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
        print(f"Running fit on: {file}")

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

        fit_idx = file_selection_page.fit_combo_box.currentIndex()

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
            # Restore the initial parameter values before each file's fit.
            for param in fit.model.parameters_all:
                if param.name in initial_params:
                    param.value, param.fixed = initial_params[param.name]

            self.dummy_run_fit(file, fit_idx)
            # Access fit parameters via fit.model.parameters_all and chi2r via fit.chi2r.
            for param in fit.model.parameters_all:
                result = {
                    "Run": str(i),
                    "Filename": file,
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
            time.sleep(1)  # Simulate processing delay; adjust or remove as needed

        progress_window.close()

        # Write the results to the chosen CSV file.
        fieldnames = ["Run", "Filename", "Parameter", "Fixed", "Value", "Chi2r"]
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.results:
                writer.writerow(row)

        # Store the CSV filename in the wizard so the final page can access it.
        wizard.fit_results_file = csv_filename

        QMessageBox.information(self, "Analysis Complete", "All fits have been completed and results saved.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    wizard = BatchProcessingWizard()
    wizard.show()
    sys.exit(app.exec())

if __name__ == "plugin":
    wizard = BatchProcessingWizard()
    wizard.show()

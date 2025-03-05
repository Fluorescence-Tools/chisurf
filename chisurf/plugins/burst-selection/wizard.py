from pathlib import Path
from qtpy import QtWidgets, QtCore, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import chisurf.gui.decorators
import chisurf.gui.widgets
import chisurf.gui.widgets.wizard

from chisurf import logging

from scipy.optimize import curve_fit


def single_gaussian(x, A, mu, sigma):
    """
    Single Gaussian function:
      A: Amplitude
      mu: Mean
      sigma: Std. deviation
    """
    return A * np.exp(-(x - mu)**2 / (2.0 * sigma**2))


def multi_gaussian(x, *params):
    """
    Returns the sum of k Gaussians.
    params should have length 3*k: [A1, mu1, sigma1, A2, mu2, sigma2, ..., Ak, muk, sigmak].
    """
    k = len(params) // 3
    result = np.zeros_like(x, dtype=float)
    for i in range(k):
        A = params[3*i]
        mu = params[3*i + 1]
        sigma = params[3*i + 2]
        result += A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return result


class BrickMicWizard(QtWidgets.QMainWindow):

    @chisurf.gui.decorators.init_with_ui("burst-selection/gui.ui", path=chisurf.settings.plugin_path)
    def __init__(self, *args, **kwargs):
        # ---------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------
        # base class init is called by decorator
        # super().__init__(*args, **kwargs)

        self.channel_definer = chisurf.gui.widgets.wizard.DetectorWizardPage(
            parent=self,
            json_file=None
        )
        self.verticalLayout_7.addWidget(self.channel_definer)

        self.burst_finder = chisurf.gui.widgets.wizard.WizardTTTRPhotonFilter(
            windows=self.channel_definer.windows,
            detectors=self.channel_definer.detectors,
            show_dT=True,
            show_burst=False,
            show_mcs=True,
            show_decay=False,
            show_filter=False
        )
        self.verticalLayout_2.addWidget(self.burst_finder)

        # Create a matplotlib Figure and Canvas for the histogram.
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        while self.verticalLayout_4.count():
            child = self.verticalLayout_4.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.verticalLayout_4.addWidget(self.canvas)

        # Store references to data and UI elements.
        self.node_objects: dict = {}
        self.node_data: dict = {}
        self.connections: list[(int, int)] = []
        self.current_df = None  # We'll keep the final, concatenated DataFrame here

        # Setup table copy behavior
        self.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.copyShortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+C"), self.tableWidget)
        self.copyShortcut.activated.connect(self.copy_entire_table)

        # Connect signals to update the histogram
        self.comboBox.currentTextChanged.connect(self.update_histogram)
        self.spinBox_3.valueChanged.connect(self.update_histogram)
        self.doubleSpinBox_3.valueChanged.connect(self.update_histogram)
        self.doubleSpinBox_4.valueChanged.connect(self.update_histogram)
        self.spinBox.valueChanged.connect(self.update_histogram)

        # Connect buttons
        self.pushButton.clicked.connect(self.process_all_files)
        self.pushButton_2.clicked.connect(self.clear_data)

        # Hide some unused UI elements in burst_finder
        self.burst_finder.groupBox_3.hide()
        self.burst_finder.toolButton_6.hide()
        self.burst_finder.toolButton_3.hide()
        self.burst_finder.toolButton_4.hide()

    # --------------------------------------------------------------------------
    # Drag & Drop Events
    # --------------------------------------------------------------------------
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept only file URLs being dragged into the main window."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """
        Extract file paths from the dropped URLs and process them as needed.
        For example, you might want to store these paths in burst_finder.settings
        and then call process_all_files, or simply parse them here.
        """
        file_paths = []
        for url in event.mimeData().urls():
            # Convert to local file path (handles local files, not necessarily remote)
            local_path = url.toLocalFile()
            if local_path:
                file_paths.append(local_path)

        event.acceptProposedAction()

        # For demonstration, print the dropped files:
        logging.info(f"Dropped files: {file_paths}")

        # Optionally, store these in your burst_finder.settings
        # and auto-process them:
        self.burst_finder.settings['tttr_filenames'] = file_paths
        self.process_all_files()

    # --------------------------------------------------------------------------
    # Processing
    # --------------------------------------------------------------------------
    def process_all_files(self) -> None:
        """
        Calls save_selection first to store burst selections,
        then loads and processes the saved burst files for display.
        """
        tttr_files = self.burst_finder.settings.get('tttr_filenames', [])
        if not tttr_files:
            logging.info("No TTTR files to process.")
            return

        # First, save the selection (ensures all bursts are processed)
        self.burst_finder.save_selection()
        logging.info("Photon selection saved. Now loading burst files.")

        accumulated_results = []
        ui_columns = [
            "First Photon",
            "Last Photon",
            "Duration (ms)",
            "Number of Photons (red)",
            "Number of Photons (green)"
        ]

        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(len(tttr_files))
        self.progressBar.setValue(0)
        self.centralwidget.setEnabled(False)

        for index, fn in enumerate(tttr_files):
            file_path = Path(fn)
            analysis_folder_name = self.burst_finder.lineEdit_2.text()
            bur_file_path = file_path.parent / analysis_folder_name / 'bi4_bur' / f"{file_path.stem}.bur"

            if not bur_file_path.exists():
                logging.info(f"Warning: Expected .bur file not found: {bur_file_path}")
                continue

            # Load the pre-saved burst file
            df = pd.read_csv(bur_file_path, sep="\t")

            # Ensure all expected columns exist
            missing_cols = [col for col in ui_columns if col not in df.columns]
            for col in missing_cols:
                df[col] = 0

            # Create a limited subset DataFrame for UI
            df_ui = df[ui_columns].copy()

            # Compute Proximity Ratio for UI
            df_ui["Proximity Ratio"] = df_ui.apply(
                lambda row: row["Number of Photons (red)"] /
                            (row["Number of Photons (red)"] + row["Number of Photons (green)"])
                if (row["Number of Photons (red)"] + row["Number of Photons (green)"]) > 0 else 0,
                axis=1
            ).round(3)

            accumulated_results.append(df_ui)
            self.progressBar.setValue(index + 1)
            QtWidgets.QApplication.processEvents()

        # Combine all loaded data and update UI
        if accumulated_results:
            final_df = pd.concat(accumulated_results, ignore_index=True)
            self.current_df = final_df
            self.populate_table(final_df)
            self.update_histogram()

        logging.info("All burst files loaded successfully.")
        self.centralwidget.setEnabled(True)

    def update_histogram(self):
        if self.current_df is None:
            return

        selected_feature = self.comboBox.currentText()
        if not selected_feature:
            return

        data = self.current_df[selected_feature]
        try:
            data = pd.to_numeric(data)
        except Exception as e:
            logging.info(f"Could not convert data in column {selected_feature} to numeric: {e}")
            return

        num_bins = int(self.spinBox_3.value())
        min_val = float(self.doubleSpinBox_3.value())
        max_val = float(self.doubleSpinBox_4.value())

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Plot histogram.
        n, bins, patches = ax.hist(
            data.dropna(),
            bins=num_bins,
            range=(min_val, max_val),
            color='blue',
            edgecolor='black',
            alpha=0.7
        )
        ax.set_title(f"Histogram of {selected_feature}")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")

        # Number of Gaussians to fit.
        k = int(self.spinBox.value())
        if k > 0:
            try:
                # Bin centers from histogram.
                bin_centers = (bins[:-1] + bins[1:]) / 2.0

                # Initial guess
                initial_guess = []
                for i in range(k):
                    A_guess = max(n) / k
                    mu_guess = min_val + (max_val - min_val) * (i + 0.5) / k
                    sigma_guess = (max_val - min_val) / (2 * k)
                    initial_guess.extend([A_guess, mu_guess, sigma_guess])

                # Fit mixture model
                popt, pcov = curve_fit(multi_gaussian, bin_centers, n, p0=initial_guess)

                # Generate x values for the fitted curve.
                x_fit = np.linspace(min_val, max_val, 200)
                # Sum of all Gaussians
                y_fit = multi_gaussian(x_fit, *popt)

                colors = plt.cm.viridis(np.linspace(0, 1, k))
                for i in range(k):
                    A = popt[3*i]
                    mu = popt[3*i + 1]
                    sigma = popt[3*i + 2]
                    y_gauss_i = single_gaussian(x_fit, A, mu, sigma)
                    ax.plot(
                        x_fit,
                        y_gauss_i,
                        '--',
                        color=colors[i],
                        label=f'Gaussian {i + 1}'
                    )

                ax.plot(x_fit, y_fit, 'r-', label='Sum of all Gaussians')
                ax.legend()

                # Build a results table
                header = ["Gaussian #", "Amplitude", "Mean", "Sigma"]
                rows = []
                col_widths = [len(h) for h in header]

                for i in range(k):
                    A = popt[3*i]
                    mu = popt[3*i + 1]
                    sigma = popt[3*i + 2]
                    try:
                        errorA = np.sqrt(pcov[3*i, 3*i])
                        errorMu = np.sqrt(pcov[3*i + 1, 3*i + 1])
                        errorSigma = np.sqrt(pcov[3*i + 2, 3*i + 2])
                    except Exception:
                        errorA, errorMu, errorSigma = None, None, None

                    if errorA is not None:
                        amp_str = f"{A:.3f} ± {errorA:.3f}"
                        mu_str = f"{mu:.3f} ± {errorMu:.3f}"
                        sigma_str = f"{sigma:.3f} ± {errorSigma:.3f}"
                    else:
                        amp_str = f"{A:.3f}"
                        mu_str = f"{mu:.3f}"
                        sigma_str = f"{sigma:.3f}"

                    row = [f"{i + 1}", amp_str, mu_str, sigma_str]
                    rows.append(row)
                    for j, item in enumerate(row):
                        col_widths[j] = max(col_widths[j], len(item))

                def build_format_string(widths):
                    return "  ".join(f"{{:<{w}}}" for w in widths)

                fmt = build_format_string(col_widths)
                table_lines = []
                table_lines.append(fmt.format(*header))
                total_width = sum(col_widths) + 2 * (len(col_widths) - 1)
                table_lines.append("-" * total_width)
                for row in rows:
                    table_lines.append(fmt.format(*row))

                table_str = "\n".join(table_lines)

                # Display table
                self.plainTextEdit.setPlainText(table_str)

            except Exception as e:
                err_msg = "Gaussian mixture fit failed: " + str(e)
                logging.info(err_msg)
                self.plainTextEdit.setPlainText(err_msg)
        else:
            self.plainTextEdit.clear()

        self.canvas.draw()

    def populate_table(self, df):
        """
        Populates self.tableWidget with data from the DataFrame and
        updates self.comboBox with the column names if the content is new.
        Only every 10th burst (row) from the DataFrame is added to the table as a preview.
        """
        df_preview = df.iloc[::10]
        final_columns = df.columns.tolist()

        self.tableWidget.clear()
        self.tableWidget.setRowCount(len(df_preview))
        self.tableWidget.setColumnCount(len(final_columns))
        self.tableWidget.setHorizontalHeaderLabels(final_columns)

        for row_index, row in enumerate(df_preview.itertuples(index=False)):
            for col_index, value in enumerate(row):
                item_text = f"{value:.3f}" if isinstance(value, float) else str(value)
                item = QtWidgets.QTableWidgetItem(item_text)
                self.tableWidget.setItem(row_index, col_index, item)

        # Update comboBox if columns differ
        current_items = [self.comboBox.itemText(i) for i in range(self.comboBox.count())]
        if current_items != final_columns:
            self.comboBox.clear()
            self.comboBox.addItems(final_columns)

        # Set default to 'Proximity Ratio' if present
        idx = self.comboBox.findText("Proximity Ratio")
        if idx != -1:
            self.comboBox.setCurrentIndex(idx)

    def copy_entire_table(self):
        """
        Copies the entire table content, including header labels,
        to the clipboard as tab-delimited text.
        """
        row_count = self.tableWidget.rowCount()
        col_count = self.tableWidget.columnCount()

        # Collect column headers
        header_items = []
        for col in range(col_count):
            header_item = self.tableWidget.horizontalHeaderItem(col)
            header_text = header_item.text() if header_item else ""
            header_items.append(header_text)
        clipboard_text = "\t".join(header_items) + "\n"

        # Collect cell data
        for row in range(row_count):
            row_data = []
            for col in range(col_count):
                item = self.tableWidget.item(row, col)
                cell_text = item.text() if item else ""
                row_data.append(cell_text)
            clipboard_text += "\t".join(row_data) + "\n"

        # Place the final text in the clipboard
        QtWidgets.QApplication.clipboard().setText(clipboard_text)

    def clear_data(self):
        """
        Clears all current data from the table, histogram, text fields,
        and resets the data frame.
        """
        self.current_df = None
        # clear data reader
        self.burst_finder.toolButton_6.click()

        # Clear the table
        self.tableWidget.clear()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(0)

        # Clear the figure and redraw an empty canvas
        self.figure.clear()
        self.canvas.draw()

        # Clear the plain text area
        self.plainTextEdit.clear()

        # Optionally reset the progress bar if desired
        self.progressBar.setValue(0)

        logging.info("Data cleared.")


if __name__ == "plugin":
    brick_mic_wiz = BrickMicWizard()
    brick_mic_wiz.show()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    brick_mic_wiz = BrickMicWizard()
    brick_mic_wiz.setWindowTitle('BRICK-Mic')
    brick_mic_wiz.show()
    sys.exit(app.exec_())

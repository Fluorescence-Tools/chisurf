from pathlib import Path
from qtpy import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pandas as pd
import numpy as np

import chisurf.gui.decorators
import chisurf.gui.widgets
import chisurf.gui.widgets.wizard

from scipy.optimize import curve_fit


# Define a function for a mixture of Gaussians.
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

    def update_bursts(self):
        """
        Generates the bursts summary DataFrame, calculates the proximity ratio,
        saves the result to a .bst file in an analysis folder, populates the table,
        and updates the histogram.
        """
        fn = self.burst_finder.filename

        df = chisurf.gui.widgets.wizard.create_bur_summary(
            start_stop=self.burst_finder.burst_start_stop,
            tttr=self.burst_finder.tttr,
            filename=fn,
            windows=self.burst_finder.windows,
            detectors=self.burst_finder.detectors
        )

        # Define the desired columns.
        desired_columns = [
            "First Photon",
            "Last Photon",
            "Duration (ms)",
            "Number of Photons (red)",
            "Number of Photons (green)"
        ]

        # Ensure any missing columns are added with a default value.
        missing_cols = [col for col in desired_columns if col not in df.columns]
        if missing_cols:
            print("Warning: The following expected columns are missing:", missing_cols)
            for col in missing_cols:
                df[col] = 0

        # Create a subset DataFrame and calculate Proximity Ratio.
        df_subset = df[desired_columns].copy()
        df_subset["Proximity Ratio"] = df_subset.apply(
            lambda row: row["Number of Photons (red)"] /
                        (row["Number of Photons (red)"] + row["Number of Photons (green)"])
                        if (row["Number of Photons (red)"] + row["Number of Photons (green)"]) > 0 else 0,
            axis=1
        )
        # Round to 3 digits
        df_subset["Proximity Ratio"] = df_subset["Proximity Ratio"].round(3)

        # Use pathlib to create an analysis folder and derive the bursts filename.
        file_path = Path(fn)
        fn = self.burst_finder.lineEdit_2.text()
        analysis_folder = file_path.parent / fn
        analysis_folder.mkdir(parents=True, exist_ok=True)
        bst_filename = analysis_folder / f"{file_path.stem}.bst"
        df_subset.to_csv(bst_filename, sep="\t", index=False)
        print("Saved bursts to", bst_filename)

        # Update the table widget.
        self.tableWidget.setUpdatesEnabled(False)
        self.tableWidget.setSortingEnabled(False)
        self.populate_table(df_subset)
        self.tableWidget.setUpdatesEnabled(True)
        self.tableWidget.setSortingEnabled(True)

        # Store the DataFrame for histogram use.
        self.current_df = df_subset

        # Update the histogram (using the feature currently selected in comboBox).
        self.update_histogram()

    def process_all_files(self) -> None:
        """
        Iterates over all TTTR files registered in the burst finder by setting the
        burst finder’s spinBox_4 to each valid index. For each file, it reads the TTTR file,
        generates its burst summary DataFrame (using chisurf.gui.widgets.wizard.create_bur_summary),
        reduces the results to a subset of desired columns, calculates the proximity ratio, and writes
        the burst data to a separate .bst file in an analysis folder. After all files are processed,
        the accumulated results are concatenated and the UI (table and histogram) is updated.
        """
        # Retrieve the list of TTTR filenames from the burst finder settings.
        tttr_files = self.burst_finder.settings.get('tttr_filenames', [])
        num_files = len(tttr_files)
        if num_files == 0:
            print("No TTTR files to process.")
            return

        # Initialize the progress bar.
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(num_files)
        self.progressBar.setValue(0)

        accumulated_results = []

        # Desired columns for the burst summary.
        desired_columns = [
            "First Photon",
            "Last Photon",
            "Duration (ms)",
            "Number of Photons (red)",
            "Number of Photons (green)"
        ]

        for index in range(num_files):
            # Set the current file index in the burst finder.
            self.burst_finder.spinBox_4.setValue(index)
            QtWidgets.QApplication.processEvents()

            # Create the burst summary DataFrame using the module function.
            fn = self.burst_finder.filename  # Original filename for the current file.
            df = chisurf.gui.widgets.wizard.create_bur_summary(
                start_stop=self.burst_finder.burst_start_stop,
                tttr=self.burst_finder.tttr,
                filename=fn,
                windows=self.burst_finder.windows,
                detectors=self.burst_finder.detectors
            )

            # Ensure the desired columns exist.
            missing_cols = [col for col in desired_columns if col not in df.columns]
            if missing_cols:
                print("Warning: The following expected columns are missing:", missing_cols)
                for col in missing_cols:
                    df[col] = 0

            # Create a subset DataFrame and calculate Proximity Ratio.
            df_subset = df[desired_columns].copy()
            df_subset["Proximity Ratio"] = df_subset.apply(
                lambda row: row["Number of Photons (red)"] /
                            (row["Number of Photons (red)"] + row["Number of Photons (green)"])
                if (row["Number of Photons (red)"] + row["Number of Photons (green)"]) > 0 else 0,
                axis=1
            )
            # Round to 3 digits
            df_subset["Proximity Ratio"] = df_subset["Proximity Ratio"].round(3)

            # Write the burst data to a separate file.
            file_path = Path(fn)
            # Retrieve the folder name from the burst finder's lineEdit_2.
            analysis_folder_name = self.burst_finder.lineEdit_2.text()
            analysis_folder = file_path.parent / analysis_folder_name
            analysis_folder.mkdir(parents=True, exist_ok=True)
            bst_filename = analysis_folder / f"{file_path.stem}.bst"
            df_subset.to_csv(bst_filename, sep="\t", index=False)
            print("Saved bursts to", bst_filename)

            # Accumulate this file's results.
            accumulated_results.append(df_subset)

            # Update progress bar.
            self.progressBar.setValue(index + 1)
            QtWidgets.QApplication.processEvents()
            print(f"Processed file {index + 1} of {num_files}")

        if accumulated_results:
            # Concatenate all individual DataFrames.
            final_df = pd.concat(accumulated_results, ignore_index=True)
            self.current_df = final_df
            # Update the table widget.
            self.tableWidget.setUpdatesEnabled(False)
            self.tableWidget.setSortingEnabled(False)
            self.populate_table(final_df)
            self.tableWidget.setUpdatesEnabled(True)
            self.tableWidget.setSortingEnabled(True)
            # Update the histogram.
            self.update_histogram()
        else:
            print("No burst data was accumulated.")

        print("All files processed.")

    def update_histogram(self):
        """
        Updates the histogram in self.verticalLayout_4 based on the feature selected
        in self.comboBox, using the number of bins and range specified by
        self.spinBox_3 (bins), self.doubleSpinBox_3 (min value), and self.doubleSpinBox_4 (max value).
        If the Gaussian fit option is enabled (self.checkBox_Gauss is checked),
        a Gaussian mixture (with the number of components given by self.spinBox) is fitted to the histogram,
        the resulting fitted curve is overlaid on the plot, and the fit results are written to self.plainTextEdit.
        """
        if self.current_df is None:
            return

        selected_feature = self.comboBox.currentText()
        if not selected_feature:
            return

        # Retrieve the data for the selected feature.
        data = self.current_df[selected_feature]

        # Try converting the data to numeric; if conversion fails, skip updating the histogram.
        try:
            data = pd.to_numeric(data)
        except Exception as e:
            print(f"Could not convert data in column {selected_feature} to numeric: {e}")
            return

        # Retrieve histogram parameters from the widgets.
        num_bins = int(self.spinBox_3.value())
        min_val = float(self.doubleSpinBox_3.value())
        max_val = float(self.doubleSpinBox_4.value())

        # Clear the current figure and add a new subplot.
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # Plot the histogram and retrieve the bin counts and edges.
        n, bins, patches = ax.hist(data.dropna(), bins=num_bins, range=(min_val, max_val),
                                   color='blue', edgecolor='black', alpha=0.7)
        ax.set_title(f"Histogram of {selected_feature}")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")

        # Get the number of Gaussians to fit.
        k = int(self.spinBox.value())
        if k > 0:
            try:
                # Compute bin centers from the histogram bins.
                bin_centers = (bins[:-1] + bins[1:]) / 2.0
                # Build an initial guess: for each Gaussian, we guess:
                #   Amplitude: max(n)/k,
                #   Mean: equally spaced between min_val and max_val,
                #   Sigma: (max_val - min_val)/(2*k)
                initial_guess = []
                for i in range(k):
                    A_guess = max(n) / k
                    mu_guess = min_val + (max_val - min_val) * (i + 0.5) / k
                    sigma_guess = (max_val - min_val) / (2 * k)
                    initial_guess.extend([A_guess, mu_guess, sigma_guess])
                # Fit the mixture model.
                popt, pcov = curve_fit(multi_gaussian, bin_centers, n, p0=initial_guess)
                # Generate x values for the fitted curve.
                x_fit = np.linspace(min_val, max_val, 100)
                y_fit = multi_gaussian(x_fit, *popt)
                ax.plot(x_fit, y_fit, 'r-', label='Gaussian Mixture Fit')
                ax.legend()

                # Write the fit results to self.plainTextEdit.
                results = []
                for i in range(k):
                    A = popt[3 * i]
                    mu = popt[3 * i + 1]
                    sigma = popt[3 * i + 2]
                    try:
                        errorA = np.sqrt(pcov[3 * i, 3 * i])
                        errorMu = np.sqrt(pcov[3 * i + 1, 3 * i + 1])
                        errorSigma = np.sqrt(pcov[3 * i + 2, 3 * i + 2])
                    except Exception as err:
                        errorA, errorMu, errorSigma = None, None, None

                    if errorA is not None:
                        results.append(
                            f"Gaussian {i+1}:\n  Amplitude = {A:.3f} ± {errorA:.3f}\n"
                            f"  Mean = {mu:.3f} ± {errorMu:.3f}\n"
                            f"  Sigma = {sigma:.3f} ± {errorSigma:.3f}")
                    else:
                        results.append(
                            f"Gaussian {i+1}:\n  Amplitude = {A:.3f}\n"
                            f"  Mean = {mu:.3f}\n"
                            f"  Sigma = {sigma:.3f}")

                result_str = "\n\n".join(results)
                self.plainTextEdit.setPlainText(result_str)
            except Exception as e:
                err_msg = "Gaussian mixture fit failed: " + str(e)
                print(err_msg)
                self.plainTextEdit.setPlainText(err_msg)
        else:
            self.plainTextEdit.clear()

        self.canvas.draw()

    def populate_table(self, df):
        """
        Populates self.tableWidget with data from the DataFrame and fills self.comboBox
        with the column names only if the content is new.
        Only every 10th burst (row) from the DataFrame is added to the table as a preview.
        Also, if 'Proximity Ratio' is one of the columns, it is set as the default selected feature.
        """
        # Use a preview: only every 10th row.
        df_preview = df.iloc[::10]
        final_columns = df.columns.tolist()

        # Update the table widget.
        self.tableWidget.clear()
        self.tableWidget.setRowCount(len(df_preview))
        self.tableWidget.setColumnCount(len(final_columns))
        self.tableWidget.setHorizontalHeaderLabels(final_columns)

        for row_index, row in enumerate(df_preview.itertuples(index=False)):
            for col_index, value in enumerate(row):
                if isinstance(value, float):
                    item_text = f"{value:.3f}"
                else:
                    item_text = str(value)
                item = QtWidgets.QTableWidgetItem(item_text)
                self.tableWidget.setItem(row_index, col_index, item)

        # Update the comboBox only if its content is different.
        current_items = [self.comboBox.itemText(i) for i in range(self.comboBox.count())]
        if current_items != final_columns:
            self.comboBox.clear()
            self.comboBox.addItems(final_columns)
        # Set default selected feature to 'Proximity Ratio' if it exists.
        index = self.comboBox.findText("Proximity Ratio")
        if index != -1:
            self.comboBox.setCurrentIndex(index)

    def copy_entire_table(self):
        """
        Copies the entire table content, including header labels,
        to the clipboard as tab-delimited text.
        """
        row_count = self.tableWidget.rowCount()
        col_count = self.tableWidget.columnCount()

        # Get header labels for all columns
        header_items = []
        for col in range(col_count):
            header_item = self.tableWidget.horizontalHeaderItem(col)
            header_text = header_item.text() if header_item is not None else ""
            header_items.append(header_text)
        # Join headers with a tab delimiter and end with a newline
        clipboard_text = "\t".join(header_items) + "\n"

        # Iterate over all rows and columns to gather cell text
        for row in range(row_count):
            row_data = []
            for col in range(col_count):
                item = self.tableWidget.item(row, col)
                cell_text = item.text() if item is not None else ""
                row_data.append(cell_text)
            clipboard_text += "\t".join(row_data) + "\n"

        # Place the final text in the clipboard
        QtWidgets.QApplication.clipboard().setText(clipboard_text)

    def clear_data(self):
        """
        Clears all current data from the table, histogram, text fields, and resets the data frame.
        """
        self.current_df = None

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

        print("Data cleared.")


    @chisurf.gui.decorators.init_with_ui("brick-mic/gui.ui", path=chisurf.settings.plugin_path)
    def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)

        self.channel_definer = chisurf.gui.widgets.wizard.DetectorWizardPage(
            parent=self,
            json_file=None #'chisurf/plugins/brick-mic/channel_settings.json'
        )
        self.verticalLayout_7.addWidget(self.channel_definer)

        self.burst_finder = chisurf.gui.widgets.wizard.WizardTTTRPhotonFilter(
            windows=self.channel_definer.windows,
            detectors=self.channel_definer.detectors,
            #callback_function=self.update_bursts,
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
        # Clear any existing items from verticalLayout_4 and add the canvas.
        while self.verticalLayout_4.count():
            child = self.verticalLayout_4.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.verticalLayout_4.addWidget(self.canvas)

        # python objects of nodes
        self.node_objects: dict = dict()

        self.node_data: dict = dict()
        self.connections: list[(int, int)] = list()

        # Set the selection mode as desired (the copy function will always copy the entire table)
        self.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)

        # Install a QShortcut for Ctrl+C on the table widget.
        self.copyShortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+C"), self.tableWidget)
        self.copyShortcut.activated.connect(self.copy_entire_table)

        # Connect signals to update the histogram.
        self.comboBox.currentTextChanged.connect(self.update_histogram)
        self.spinBox_3.valueChanged.connect(self.update_histogram)
        self.doubleSpinBox_3.valueChanged.connect(self.update_histogram)
        self.doubleSpinBox_4.valueChanged.connect(self.update_histogram)
        self.spinBox.valueChanged.connect(self.update_histogram)

        # Connect pushButton to process files
        self.pushButton.clicked.connect(self.process_all_files)

        # Connect pushButton_2 to clear the data
        self.pushButton_2.clicked.connect(self.clear_data)

        # Store the current DataFrame for histogram updates.
        self.current_df = None

        self.burst_finder.groupBox_3.hide()
        self.burst_finder.toolButton_3.hide()
        self.burst_finder.toolButton_4.hide()


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

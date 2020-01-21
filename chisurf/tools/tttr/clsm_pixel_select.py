from __future__ import annotations
import typing

import sys
import os
import yaml

import pyqtgraph as pg
import pyqtgraph.graphicsItems.GradientEditorItem
from PyQt5 import QtCore, QtWidgets, QtGui

import numpy as np
import cv2
import tttrlib

import chisurf.curve
import chisurf.fluorescence.tcspc
import chisurf.experiments
import chisurf.math.signal
import chisurf.settings
import chisurf.widgets
import chisurf.widgets.experiments

plot_settings = chisurf.settings.gui['plot']
clsm_settings = yaml.safe_load(
    open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "clsm_settings.yaml"
        )
    )
)


class CLSMPixelSelect(
    QtWidgets.QWidget
):

    name: str = "pixel-decay"
    tttr_data: tttrlib.TTTR = None
    clsm_image_object: tttrlib.CLSMImage = None
    brush_kernel: np.ndarray = None
    brush_size: int = 7
    brush_width: float = 3
    img_plot: pyqtgraph.PlotWindow = None
    current_decay: chisurf.curve.Curve = None
    current_setup: str = None
    images = dict()

    @property
    def curve_name(
            self
    ) -> str:
        s = str(self.lineEdit.text())
        if len(s) == 0:
            return "no-name"
        else:
            return s

    def onRemoveDataset(self):
        selected_index = [
            i.row() for i in self.cs.selectedIndexes()
        ]
        l = list()
        for i, c in enumerate(self._curves):
            if i not in selected_index:
                l.append(c)
        self._curves = l
        self.cs.update()
        self.plot_curves()

    def clear_curves(self):
        self._curves = list()
        plot = self.plot.getPlotItem()
        plot.clear()
        self.cs.update()

    def get_data_curves(
            self,
            *args,
            **kwargs
    ) -> typing.List[chisurf.curve.Curve]:
        return self._curves

    def plot_curves(self):
        plot = self.plot.getPlotItem()
        plot.clear()
        curve = self.current_decay
        plot.plot(
            x=curve.x, y=curve.y,
            pen=pg.mkPen(
                "#CB4154",
                width=2
            ),
            name="Current selection"
        )

        self.legend = plot.addLegend()
        plot.setLogMode(x=False, y=True)
        plot.showGrid(True, True, 1.0)

        current_curve = self.cs.selected_curve_index
        lw = plot_settings['line_width']
        for i, curve in enumerate(self._curves):
            w = lw * 0.5 if i != current_curve else 2.5 * lw
            plot.plot(
                x=curve.x, y=curve.y,
                pen=pg.mkPen(
                    chisurf.settings.colors[i % len(chisurf.settings.colors)]['hex'],
                    width=w
                ),
                name=curve.name
            )
        plot.autoRange()

    def add_curve(self):
        self._curves.append(self.current_decay)
        self.cs.update()
        self.plot_curves()

    def open_file(
            self,
            filename: str = None
    ):
        if not isinstance(filename, str):
            tentative_filename = str(self.lineEdit.text())
            if os.path.isfile(tentative_filename):
                filename = tentative_filename
            else:
                filename = chisurf.widgets.get_filename(
                    description='TTTR file'
                )
                self.lineEdit.setText(filename)
        tttr_type = str(self.comboBox_4.currentText())
        self.tttr_data = tttrlib.TTTR(
            filename,
            tttr_type
        )
        frame_marker = [int(i) for i in str(self.lineEdit_2.text()).split(",")]
        line_start_marker = int(self.spinBox_4.value())
        line_stop_marker = int(self.spinBox_5.value())
        event_type_marker = int(self.spinBox_6.value())
        pixel_per_line = int(self.spinBox_7.value())
        reading_routine = str(self.comboBox.currentText())
        self.clsm_image_object = tttrlib.CLSMImage(
            self.tttr_data,
            frame_marker,
            line_start_marker,
            line_stop_marker,
            event_type_marker,
            pixel_per_line,
            reading_routine
        )

    def add_image(self):
        channels = [int(i) for i in str(self.lineEdit_3.text()).split(",")]
        self.clsm_image_object.clear_pixels()
        self.clsm_image_object.fill_pixels(
            tttr_data=self.tttr_data,
            channels=channels
        )

        image_type = str(self.comboBox_3.currentText())
        if image_type == "Mean micro time":
            n_ph_min = int(self.spinBox.value())
            mean_micro_time = self.clsm_image_object.get_mean_tac_image(
                self.tttr_data,
                n_ph_min
            )
            data = mean_micro_time.astype(np.float64)
        else: # image_type == "Intensity":
            intensity_image = self.clsm_image_object.get_intensity_image()
            data = intensity_image.astype(np.float64)

        image_name = str(self.lineEdit_4.text())
        self.images[image_name] = data
        self.comboBox_7.clear()
        self.comboBox_7.addItems(
            list(self.images.keys())
        )

        self.image_changed()

    def image_changed(self):
        # update image
        if self.checkBox_5.isChecked():
            image_name = self.comboBox_7.currentText()
            if image_name in self.images.keys():
                image = self.images[image_name]
                self.spinBox_8.setMinimum(0)
                self.spinBox_8.setMaximum(image.shape[0])

                self.hist.gradient.loadPreset(
                    self.comboBox_2.currentText()
                )
                if image is not None:
                    if self.checkBox_2.isChecked():
                        data = image.sum(axis=0)
                    else:
                        frame_idx = self.spinBox_8.value()
                        data = image[frame_idx]
                    # transpose image (row, column) -> (column, row)
                    # pyqtgraph is column major by default
                    self.img.setImage(data.T)
                    # self.img_drawn.setImage(np.zeros_like(data))
                    self.hist.setLevels(data.min(), data.max())
                    self.brush_kernel *= max(data.flatten()) / max(self.brush_kernel.flatten())
                    # zoom to fit image
                    self.img_plot.autoRange()
                    if self.img_drawn.image is None:
                        self.img_drawn.setImage(np.zeros_like(data))
                    elif self.img_drawn.image.shape != self.img.image.shape:
                        self.img_drawn.setImage(np.zeros_like(data))

        else:
            self.img.setImage(np.zeros((512, 512)))

        # update mask
        if not self.checkBox_6.isChecked():
            w = self.img_drawn.getPixmap()
            w.hide()

    def save_current_image(self):
        image_name = self.comboBox_7.currentText()
        if image_name in self.images.keys():
            filename = chisurf.widgets.save_file(
                description='Image file',
                file_type='All files (*.png)'
            )
            image = self.images[image_name]
            if image is not None:
                if self.checkBox_2.isChecked():
                    data = image.sum(axis=0)
                else:
                    frame_idx = self.spinBox_8.value()
                    data = image[frame_idx]
                cv2.imwrite(
                    filename,
                    data.T
                )

    def update_plot(self):
        # Transpose sel (pyqtgraph is column major)
        # (column, row) -> (row , column)
        sel = np.copy(self.img_drawn.image).T
        sel[sel > 0] = 1
        sel[sel < 0] = 0
        sel = sel.astype(dtype=np.short)

        tac_coarsening = int(self.comboBox_6.currentText())
        n_lines = self.clsm_image_object.n_lines
        n_pixel = self.clsm_image_object.n_pixel
        n_frames = self.clsm_image_object.n_frames
        selection = np.broadcast_to(sel, (n_frames, n_lines, n_pixel))
        stack_frames = self.checkBox_2.isChecked()
        decay = self.clsm_image_object.get_decays(
            tttr_data=self.tttr_data,
            selection=selection,
            tac_coarsening=tac_coarsening,
            stack_frames=stack_frames
        )
        header = self.tttr_data.get_header()
        x = np.arange(decay.shape[1])
        t = x * header.micro_time_resolution / tac_coarsening
        if stack_frames:
            y = decay.sum(axis=0)
        else:
            y = decay[self.spinBox_8.value()]

        y_pos = np.where(y > 0)[0]
        if len(y_pos) > 0:
            i_y_max = y_pos[-1]
            y = y[:i_y_max]
            t = t[:i_y_max]

        ey = chisurf.fluorescence.tcspc.weights(y)
        self.current_decay = chisurf.experiments.data.DataCurve(
            x=t, y=y, ey=ey
        )
        self.plot_curves()

    def setup_image_plot(
            self,
            decay_plot: pyqtgraph.PlotWindow
    ):
        win = pg.GraphicsLayoutWidget()
        # A plot area (ViewBox + axes) for displaying the image
        self.img_plot = win.addPlot(title="")
        p1 = self.img_plot

        # Item for displaying image data
        p1.addItem(self.img)
        p1.addItem(self.img_drawn)
        self.img_drawn.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)

        # Contrast/color control
        hist = self.hist
        hist.setImageItem(self.img)
        hist.gradient.loadPreset('inferno')
        win.addItem(hist)

        win.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )
        )
        win.show()

        # Monkey-patch the image to use our custom hover function.
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this.
        def imagehoverEvent(event):
            """Show the position, pixel, and value under the mouse cursor.
            """
            if event.isExit():
                p1.setTitle("")
                return
            pos = event.pos()
            i, j = pos.y(), pos.x()
            i = int(np.clip(i, 0, self.img.image.shape[0] - 1))
            j = int(np.clip(j, 0, self.img.image.shape[1] - 1))
            val = self.img.image[i, j]
            ppos = self.img.mapToParent(pos)
            x, y = ppos.x(), ppos.y()
            p1.setTitle("pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %g" % (x, y, i, j, val))

        self.img_drawn.hoverEvent = imagehoverEvent

        # Monkey-patch the image to use our custom hover function.
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this.
        def imageMouseDragEvent(event):
            """Show the position, pixel, and value under the mouse cursor.
            """
            if event.button() != QtCore.Qt.LeftButton:
                return
            elif self.img_drawn.drawKernel is not None:
                # draw on the image
                event.accept()
                self.img_drawn.drawAt(event.pos(), event)
            if self.checkBox_4.isChecked():
                self.update_plot()
        self.img_drawn.mouseDragEvent = imageMouseDragEvent
        return win

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        pass

    def save_pixel_mask(self):
        filename = chisurf.widgets.save_file(
            description='Image file',
            file_type='All files (*.png)'
        )
        image = self.img_drawn.image
        image[image > 0] = 255
        cv2.imwrite(
            filename,
            image
        )

    def clear_pixel_mask(self):
        self.img_drawn.setImage(
            np.zeros_like(
                self.img_drawn.image
            )
        )

    def load_pixel_mask(self):
        filename = chisurf.widgets.get_filename(
            description='Image file',
            file_type='All files (*.png)'
        )
        image = cv2.imread(
            filename
        )
        self.img_drawn.setImage(
            image=cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            ).astype(np.float64)
        )

    def update_brush(self):
        self.brush_size = int(self.spinBox_2.value())
        self.brush_width = float(self.doubleSpinBox.value())
        self.brush_kernel = chisurf.math.signal.gaussian_kernel(
            self.brush_size,
            self.brush_width
        )

        self.img_drawn.setDrawKernel(
            self.brush_kernel,
            mask=self.brush_kernel,
            center=(1, 1), mode='add'
        )
        if isinstance(self.img.image, np.ndarray):
            data = self.img.image
            self.brush_kernel *= 255.0 / max(self.brush_kernel.flatten())

        # The brush is set to selection mode
        select = self.radioButton.isChecked()
        if not select:
            self.brush_kernel *= -1

    def setup_changed(self):
        current_setup = self.comboBox_5.currentText()

        tttr_type = clsm_settings[current_setup]['tttr_type']
        tttr_type_idx = self.comboBox_4.findText(tttr_type)
        self.comboBox_4.setCurrentIndex(tttr_type_idx)

        clsm_routine = clsm_settings[current_setup]['routine']
        clsm_routine_idx = self.comboBox.findText(clsm_routine)
        self.comboBox.setCurrentIndex(clsm_routine_idx)

        # frame marker
        self.lineEdit_2.setText(
            str(clsm_settings[current_setup]['frame_marker'])
        )

        # line start
        self.spinBox_4.setValue(
            clsm_settings[current_setup]['line_start_marker']
        )

        # line stop
        self.spinBox_5.setValue(
            clsm_settings[current_setup]['line_stop_marker']
        )

        # event marker
        self.spinBox_6.setValue(
            clsm_settings[current_setup]['event_type_marker']
        )

    @chisurf.decorators.init_with_ui(
        ui_filename="clsm_pixel_select.ui"
    )
    def __init__(self):

        self.current_setup = list(clsm_settings.keys())[0]
        self.comboBox_5.addItems(
            list(clsm_settings.keys())
        )

        routines = list()
        for k in clsm_settings:
            routines.append(clsm_settings[k]['routine'])
        self.comboBox.addItems(
            list(set(routines))
        )

        self.img = pg.ImageItem()
        self.img_drawn = pg.ImageItem()
        self.hist = pg.HistogramLUTItem()
        self.comboBox_2.addItems(
            list(pyqtgraph.graphicsItems.GradientEditorItem.Gradients.keys())
        )

        self._curves = list()
        self.update_brush()

        # Add curve experiment curve selector
        self.cs = chisurf.widgets.experiments.ExperimentalDataSelector(
            get_data_sets=self.get_data_curves,
            click_close=False
        )
        self.verticalLayout_9.addWidget(self.cs)

        # Add plot of image
        self.plot = pg.PlotWidget()
        plot = self.plot.getPlotItem()
        self.verticalLayout.addWidget(self.plot)
        self.legend = plot.addLegend()
        plot_item = self.plot.getPlotItem()
        plot_item.setLogMode(
            x=False,
            y=True
        )
        self.cs.onRemoveDataset = self.onRemoveDataset

        # Add image view
        image_widget = self.setup_image_plot(
            decay_plot=self.plot
        )
        self.verticalLayout_4.addWidget(image_widget)

        # Signal slots
        self.actionLoad_file.triggered.connect(self.open_file)
        self.actionchange_brush_size.triggered.connect(self.update_brush)
        self.actionSave_pixel_mask.triggered.connect(self.save_pixel_mask)
        self.actionClear_pixel_mask.triggered.connect(self.clear_pixel_mask)
        self.actionLoad_pixel_mask.triggered.connect(self.load_pixel_mask)
        self.actionAdd_decay.triggered.connect(self.add_curve)
        self.actionClear_decay_curves.triggered.connect(self.clear_curves)
        self.actionSetup_changed.triggered.connect(self.setup_changed)
        self.actionAdd_image.triggered.connect(self.add_image)
        self.actionImage_changed.triggered.connect(self.image_changed)
        self.actionUpdate_plot.triggered.connect(self.update_plot)
        self.actionSave_image.triggered.connect(self.save_current_image)

        # update the UI
        self.setup_changed()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = CLSMPixelSelect()
    win.show()
    sys.exit(app.exec_())





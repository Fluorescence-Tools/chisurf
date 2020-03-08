from __future__ import annotations

import os
import sys
import unittest

from qtpy.QtWidgets import QApplication
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt
from qtpy import QtWidgets

import chisurf
import chisurf.widgets
import chisurf.macros
import chisurf.__main__
import chisurf.widgets.experiments


app = QApplication(sys.argv)
cs_app = chisurf.__main__.qt_app()


def add_fit(
        data_set_name: str,
        dataset_selector: chisurf.widgets.experiments.ExperimentalDataSelector,
        push_button: QtWidgets.QPushButton,
        model_selector: QtWidgets.QComboBox,
        model_name: str
):
    # select data_set
    for i in chisurf.widgets.get_all_items(dataset_selector):
        if i.text(0) == data_set_name:
            rect = dataset_selector.visualItemRect(i)
            QTest.mouseClick(
                dataset_selector.viewport(),
                Qt.LeftButton,
                Qt.NoModifier,
                rect.center()
            )
            break
    # select model
    model_idx = model_selector.findText(model_name)
    model_selector.setCurrentIndex(model_idx)

    # click on add fit
    QTest.mouseClick(push_button, Qt.LeftButton)


def setup_reader(
        experiment_name: str,
        experiment_selector_combobox: QtWidgets.QComboBox,
        setup_name: str,
        setup_selector_combobox: QtWidgets.QComboBox
):
    experiment_idx = experiment_selector_combobox.findText(experiment_name)
    experiment_selector_combobox.setCurrentIndex(experiment_idx)

    setup_idx = setup_selector_combobox.findText(setup_name)
    setup_selector_combobox.setCurrentIndex(setup_idx)


class Tests(unittest.TestCase):
    """
    Test the kappa2 distribution GUI
    """

    def test_tcspc(self):
        """
        Open a TCSPC dataset, and create a lifetime fit
        """
        cs = chisurf.cs
        setup_reader(
            experiment_name="TCSPC",
            experiment_selector_combobox=cs.comboBox_experimentSelect,
            setup_name="TCSPCReader",
            setup_selector_combobox=cs.comboBox_setupSelect
        )
        filename_decay = "./test/data/tcspc/ibh_sample/Decay_577D.txt"
        filename_irf = "./test/data/tcspc/ibh_sample/Prompt.txt"

        cs.current_setup.skiprows = 11
        cs.current_setup.reading_routine = 'csv'
        cs.current_setup.is_jordi = False
        cs.current_setup.use_header = True
        cs.current_setup.matrix_columns = []
        cs.current_setup.polarization = 'vm'
        cs.current_setup.rep_rate = 10.0
        cs.current_setup.dt = 0.0141

        chisurf.macros.add_dataset(
            filename=filename_decay
        )
        chisurf.macros.add_dataset(
            filename=filename_irf
        )
        model_name = 'Lifetime fit'
        data_set_name = "Decay_577D.txt"
        add_fit(
            data_set_name=data_set_name,
            dataset_selector=cs.dataset_selector,
            push_button=cs.pushButton_2,
            model_selector=cs.comboBox_Model,
            model_name=model_name
        )
        model_name = 'FRET: FD (Gaussian)'
        data_set_name = "Decay_577D.txt"
        add_fit(
            data_set_name=data_set_name,
            dataset_selector=cs.dataset_selector,
            push_button=cs.pushButton_2,
            model_selector=cs.comboBox_Model,
            model_name=model_name
        )

    def test_fcs(self):
        """
        Open a FCS dataset, and create a lifetime fit
        """
        cs = chisurf.cs
        setup_reader(
            experiment_name="FCS",
            experiment_selector_combobox=cs.comboBox_experimentSelect,
            setup_name="Seidel Kristine",
            setup_selector_combobox=cs.comboBox_setupSelect
        )
        filename_fcs = "./test/data/fcs/Kristine/Kristine_with_error.cor"
        chisurf.macros.add_dataset(
            filename=filename_fcs
        )
        model_name = 'Parse-Model'
        data_set_name = 'Kristine_with_error'
        add_fit(
            data_set_name=data_set_name,
            dataset_selector=cs.dataset_selector,
            push_button=cs.pushButton_2,
            model_selector=cs.comboBox_Model,
            model_name=model_name
        )

    def test_global_fit(self):
        """
        Create a global fit
        """
        cs = chisurf.cs
        model_name = 'Global fit'
        data_set_name = 'Global-fit'
        add_fit(
            data_set_name=data_set_name,
            dataset_selector=cs.dataset_selector,
            push_button=cs.pushButton_2,
            model_selector=cs.comboBox_Model,
            model_name=model_name
        )


if __name__ == "__main__":
    unittest.main()

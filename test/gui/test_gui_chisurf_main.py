from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import utils

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
utils.set_search_paths(TOPDIR)

from qtpy.QtTest import QTest
from qtpy.QtCore import Qt
from qtpy import QtWidgets

import chisurf as cs
import chisurf.gui
import chisurf.gui.widgets
import chisurf.macros
import chisurf.gui.widgets.experiments

cs_app = cs.gui.get_app()


def add_fit(
        data_set_name: str,
        dataset_selector: cs.gui.widgets.experiments.ExperimentalDataSelector,
        push_button: QtWidgets.QPushButton,
        model_selector: QtWidgets.QComboBox,
        model_name: str
):
    # select data_set
    for i in cs.gui.widgets.get_all_items(dataset_selector):
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
        gui = cs.cs
        setup_reader(
            experiment_name="TCSPC",
            experiment_selector_combobox=gui.comboBox_experimentSelect,
            setup_name="TCSPCReader",
            setup_selector_combobox=gui.comboBox_setupSelect
        )
        filename_decay = "./test/data/tcspc/ibh_sample/Decay_577D.txt"
        filename_irf = "./test/data/tcspc/ibh_sample/Prompt.txt"

        gui.current_setup.skiprows = 11
        gui.current_setup.reading_routine = 'csv'
        gui.current_setup.is_jordi = False
        gui.current_setup.use_header = True
        gui.current_setup.matrix_columns = []
        gui.current_setup.polarization = 'vm'
        gui.current_setup.rep_rate = 10.0
        gui.current_setup.dt = 0.0141

        cs.macros.add_dataset(
            filename=filename_decay
        )
        cs.macros.add_dataset(
            filename=filename_irf
        )
        model_name = 'Lifetime fit'
        data_set_name = "Decay_577D.txt"
        add_fit(
            data_set_name=data_set_name,
            dataset_selector=gui.dataset_selector,
            push_button=gui.pushButton_2,
            model_selector=gui.comboBox_Model,
            model_name=model_name
        )
        model_name = 'FRET: FD (Gaussian)'
        data_set_name = "Decay_577D.txt"
        add_fit(
            data_set_name=data_set_name,
            dataset_selector=gui.dataset_selector,
            push_button=gui.pushButton_2,
            model_selector=gui.comboBox_Model,
            model_name=model_name
        )

    def test_fcs(self):
        """
        Open a FCS dataset, and create a lifetime fit
        """
        gui = cs.cs
        setup_reader(
            experiment_name="FCS",
            experiment_selector_combobox=gui.comboBox_experimentSelect,
            setup_name="Seidel Kristine",
            setup_selector_combobox=gui.comboBox_setupSelect
        )
        filename_fcs = "./test/data/fcs/kristine/Kristine_with_error.cor"
        cs.macros.add_dataset(
            filename=filename_fcs
        )
        model_name = 'Parse-Model'
        data_set_name = 'Kristine_with_error'
        add_fit(
            data_set_name=data_set_name,
            dataset_selector=gui.dataset_selector,
            push_button=gui.pushButton_2,
            model_selector=gui.comboBox_Model,
            model_name=model_name
        )

    def test_global_fit(self):
        """
        Create a global fit
        """
        gui = cs.cs
        model_name = 'Global fit'
        data_set_name = 'Global-fit'
        add_fit(
            data_set_name=data_set_name,
            dataset_selector=gui.dataset_selector,
            push_button=gui.pushButton_2,
            model_selector=gui.comboBox_Model,
            model_name=model_name
        )


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import pathlib

import chisurf.core.structure
import chisurf.core.models.global_model
import chisurf.core.fitting
import chisurf.core.fitting.fit
import chisurf.core.models.tcspc.fret_structure as fret_structure

class TestFRETStructure(unittest.TestCase):

    def test_fret_structure_core(self):
        # 1. Setup minimal FitGroup
        # We need a data curve to pass to FitGroup
        x_data = np.linspace(0, 10, 100)
        y_data = np.zeros_like(x_data)
        ey_data = np.ones_like(x_data)
        data = chisurf.core.data.DataCurve(x=x_data, y=y_data, ey=ey_data)
        fit_group = chisurf.core.fitting.fit.FitGroup(
            data=chisurf.core.data.DataGroup([data])
        )

        # 2. Instantiate FRETStructure model
        model = fret_structure.FRETStructure(
            fit=fit_group,
            res_1=18,
            res_2=577,
            atom_name_1='CB',
            atom_name_2='CB'
        )

        # 3. Load structure
        pdb_path = './test/data/atomic_coordinates/pdb_files/hGBP1_closed.pdb'
        s = chisurf.core.structure.Structure(pdb_path)

        # 4. Test appending structure
        model.append(s, amplitude=0.6)
        self.assertEqual(len(model.names), 1)
        self.assertEqual(model.names[0], s.name)
        self.assertEqual(len(model._amplitudes), 1)
        self.assertAlmostEqual(model.amplitudes[0], 1.0)  # normalized amplitude is 1.0 for single structure

        # 5. Check distance distribution calculation
        dist_dist = model.distance_distribution
        self.assertEqual(dist_dist.ndim, 3)
        self.assertEqual(dist_dist.shape[0], 1)
        self.assertEqual(dist_dist.shape[1], 2)
        # Probability array should sum to 1.0 (excluding zeros/thresholded parts)
        probs = dist_dist[0, 0]
        self.assertTrue(np.any(probs > 0.0))

        # 6. Test pop
        model.pop()
        self.assertEqual(len(model.names), 0)
        self.assertEqual(len(model._amplitudes), 0)

    def test_fret_structure_widget(self):
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        # 1. Setup FitGroup
        x_data = np.linspace(0, 10, 100)
        y_data = np.zeros_like(x_data)
        ey_data = np.ones_like(x_data)
        data = chisurf.core.data.DataCurve(x=x_data, y=y_data, ey=ey_data)
        fit_group = chisurf.core.fitting.fit.FitGroup(
            data=chisurf.core.data.DataGroup([data])
        )

        # 2. Import and instantiate widget
        from chisurf.gui.widgets.models.tcspc.fret_structure import FRETStructureWidget
        widget = FRETStructureWidget(fit=fit_group)
        self.assertIsNotNone(widget)
        self.assertEqual(widget.res_1, 0)
        self.assertEqual(widget.res_2, 0)

        # Test setting and getting values through GUI spinboxes
        widget.res_1 = 18
        widget.res_2 = 577
        self.assertEqual(widget.res_1, 18)
        self.assertEqual(widget.res_2, 577)
        self.assertEqual(widget.donor_res.value(), 18)
        self.assertEqual(widget.acceptor_res.value(), 577)

    def test_fret_structure_serialization(self):
        # 1. Setup FitGroup
        x_data = np.linspace(0, 10, 100)
        y_data = np.zeros_like(x_data)
        ey_data = np.ones_like(x_data)
        data = chisurf.core.data.DataCurve(x=x_data, y=y_data, ey=ey_data)
        fit_group = chisurf.core.fitting.fit.FitGroup(
            data=chisurf.core.data.DataGroup([data])
        )

        # 2. Create model & append structures
        model = fret_structure.FRETStructure(
            fit=fit_group,
            res_1=18,
            res_2=577,
            atom_name_1='CB',
            atom_name_2='CB'
        )
        pdb_path = './test/data/atomic_coordinates/pdb_files/hGBP1_closed.pdb'
        s = chisurf.core.structure.Structure(pdb_path)
        model.append(s, amplitude=0.6)

        # 3. Serialize state
        state = model.get_state()
        self.assertIn("extra", state)
        self.assertEqual(state["extra"]["res_1"], 18)
        self.assertEqual(state["extra"]["res_2"], 577)
        self.assertEqual(len(state["extra"]["structures"]), 1)
        self.assertEqual(state["extra"]["structures"][0]["filename"], pdb_path)

        # 4. Restore state on a new model instance
        model2 = fret_structure.FRETStructure(
            fit=fit_group,
            res_1=0,
            res_2=0
        )
        model2.set_state(state)

        # Assert structure was reloaded and parameters/amplitudes restored
        self.assertEqual(model2.res_1, 18)
        self.assertEqual(model2.res_2, 577)
        self.assertEqual(len(model2.names), 1)
        self.assertEqual(model2.filenames[0], pdb_path)
        self.assertAlmostEqual(model2.amplitudes[0], 1.0)

        # 5. Restore state on a widget instance
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        from chisurf.gui.widgets.models.tcspc.fret_structure import FRETStructureWidget
        widget = FRETStructureWidget(fit=fit_group)
        widget.set_state(state)

        # Assert widget synchronized all UI components and reloaded structure
        self.assertEqual(widget.res_1, 18)
        self.assertEqual(widget.donor_res.value(), 18)
        self.assertEqual(widget.acceptor_res.value(), 577)
        self.assertEqual(widget.pdb_line_widget.toPlainText().strip(), pdb_path)

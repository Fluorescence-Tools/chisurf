import unittest
from unittest.mock import MagicMock
import chisurf
import chisurf.fitting.fit
import chisurf.fitting.parameter
from chisurf.project import fit_state

class MockFit(chisurf.base.Base):
    def __init__(self, name, uid):
        super().__init__()
        self.name = name
        self.meta_data['unique_identifier'] = uid
        self.model = MagicMock()
        # Mocking find_parameters to do nothing
        self.model.find_parameters = MagicMock()

class TestCrossFitLinking(unittest.TestCase):
    def setUp(self):
        # Clear global fits
        chisurf.fits = []
        
        # Create two fits with UIDs
        self.fit_a = MockFit("FitA", "UID-A")
        self.fit_b = MockFit("FitB", "UID-B")
        
        # Add to global registry
        chisurf.fits.append(self.fit_a)
        chisurf.fits.append(self.fit_b)
        
        # Setup parameters
        self.pa = chisurf.fitting.parameter.FittingParameter(name="amp_a", value=1.0)
        self.pb = chisurf.fitting.parameter.FittingParameter(name="amp_b", value=2.0)
        
        self.fit_a.model.parameters_all_dict = {"amp_a": self.pa}
        self.fit_b.model.parameters_all_dict = {"amp_b": self.pb}

    def test_cross_fit_serialization(self):
        # Link pa to pb
        self.pa.link = self.pb
        
        # Serialize fit_a
        state = fit_state._model_to_state(self.fit_a.model)
        
        # Verify link target info
        p_state = state["parameters"]["amp_a"]
        self.assertEqual(p_state["link_target"], "amp_b")
        self.assertEqual(p_state["link_target_fit_uid"], "UID-B")
        
    def test_cross_fit_restoration(self):
        # Link pa to pb
        self.pa.link = self.pb
        state = fit_state._model_to_state(self.fit_a.model)
        
        # Create fresh pa (new model for fit_a)
        new_pa = chisurf.fitting.parameter.FittingParameter(name="amp_a", value=0.0)
        new_model_a = MagicMock()
        new_model_a.parameters_all_dict = {"amp_a": new_pa}
        
        # Apply state
        fit_state._apply_state_to_model(new_model_a, state)
        
        # Verify link
        self.assertIs(new_pa.link, self.pb, "Parameter should be linked to the live instance in fit_b")
        print("Cross-fit link restoration verified.")

if __name__ == "__main__":
    unittest.main()

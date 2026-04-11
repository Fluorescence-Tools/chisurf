import unittest
from unittest.mock import MagicMock
import numpy as np
import chisurf
import chisurf.fitting.parameter
from chisurf.models.global_model.globalfit import GlobalFitModel
from chisurf.project import fit_state

class TestGlobalSerialization(unittest.TestCase):
    def test_global_parameter_discovery(self):
        # Create a GlobalFitModel
        mock_fit = MagicMock()
        model = GlobalFitModel(fit=mock_fit)
        
        # Add a global parameter
        p_global = chisurf.fitting.parameter.FittingParameter(name="shared_tau", value=5.0)
        model.append_global_parameter(p_global)
        
        # Trigger discovery
        print(f"Model __dict__ keys: {list(model.__dict__.keys())}")
        if "_global_parameters" in model.__dict__:
            print(f"_global_parameters contents: {list(model._global_parameters.keys())}")
        
        model.find_parameters()
        
        # Check if it was discovered
        all_params_list = model.parameters_all
        print(f"Original p_global ID: {id(p_global)}")
        print(f"Parameters in parameters_all list IDs: {[id(p) for p in all_params_list]}")
        
        all_params = model.parameters_all_dict
        print(f"Discovered parameters: {list(all_params.keys())}")
        
        self.assertIn("shared_tau", all_params, "Global parameter should be discovered by find_parameters")
        self.assertIs(all_params["shared_tau"], p_global)
        
        # Check serialization
        state = fit_state._model_to_state(model)
        self.assertIn("shared_tau", state["parameters"], "Global parameter should be serialized")
        self.assertEqual(state["parameters"]["shared_tau"]["value"], 5.0)

if __name__ == "__main__":
    unittest.main()

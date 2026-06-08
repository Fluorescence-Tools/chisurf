import os
import json
import numpy as np
import pytest
import chisurf as cs
from chisurf.core.data import DataCurve
from chisurf.core.fitting.fit import Fit
from chisurf.core.models.model import ModelCurve
from chisurf.core.fitting.parameter import FittingParameter
from chisurf.core.project import Project, save_project, load_project as project_load_json
from chisurf.macros.core_fit import add_fit, save_project as macro_save_project, load_project as macro_load_project

# Mock cs global state for tests
class MockCS:
    def __init__(self):
        self.current_experiment = None
        self.dataset_selector = type('DS', (), {'selected_curve_index': 0})()

class E2ELinearModel(ModelCurve):
    """Small concrete model used for E2E tests."""
    name = "E2ELinearModel"

    def __init__(self, fit: Fit, **kwargs):
        super().__init__(fit, **kwargs)
        self.p0 = FittingParameter(name="p0", value=0.0)
        self.find_parameters()

    def update_model(self, **kwargs):
        x = self.fit.data.x
        if x is None:
            x = np.arange(self.fit.data.y.size, dtype=float)
        self.x = x
        self.y = float(self.p0.value) + x

def test_e2e_with_real_file_headless(tmp_path):
    # Setup
    val_p0 = 7.5
    project_dir = tmp_path / "e2e_headless"
    csv_file = tmp_path / "data.csv"
    with open(csv_file, "w") as f:
        f.write("x,y\n0,1\n1,2\n2,3\n")
        
    orig_imported_datasets = getattr(cs, "imported_datasets", [])
    orig_fits = getattr(cs, "fits", [])
    orig_cs = getattr(cs, "cs", None)

    try:
        print("\n--- Phase 1: Clear state ---")
        cs.imported_datasets = []
        cs.fits = []
        # TRUE HEADLESS
        cs.cs = None
        
        print("--- Phase 2: Create dataset ---")
        # Use a real DataCurve that points to the file
        ds = DataCurve(x=np.arange(3, dtype=float), y=np.arange(3, dtype=float)+1.0, name="real_ds")
        ds.path = str(csv_file)
        cs.imported_datasets.append(ds)
        
        # We need to ensure the experiment and model are resolvable
        class MockExperiment:
            name = "MockExp"
            model_names = ["E2ELinearModel"]
            model_classes = [E2ELinearModel]
        mock_exp = MockExperiment()
        ds.experiment = mock_exp
        
        print("--- Phase 3: Add fit ---")
        # Add fit and modify param
        add_fit(dataset_indices=[0], model_name="E2ELinearModel")
        assert len(cs.fits) == 1
        fit = cs.fits[0]
        fit.model.p0.value = val_p0
        
        print("--- Phase 4: Save project ---")
        # Save
        # macro_save_project(target_path, project_name)
        # created dir: target_path/project_name
        macro_save_project(str(tmp_path), "e2e_headless")
        assert (project_dir / "project.json").exists()
        
        print("--- Phase 5: Reload ---")
        # Restart simulation
        cs.imported_datasets = []
        cs.fits = []
        cs.cs = None
        
        # Reload
        macro_load_project(str(project_dir))
        
        print("--- Phase 6: Verify ---")
        # Verify
        assert len(cs.fits) == 1
        loaded_fit = cs.fits[0]
        # Check that the parameter value was restored
        print(f"Loaded p0 value: {loaded_fit.model.p0.value}")
        assert np.isclose(loaded_fit.model.p0.value, val_p0)
        print("--- Success! ---")
        
    except Exception as e:
        print(f"\n--- FAILED with error: {e} ---")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        cs.imported_datasets = orig_imported_datasets
        cs.fits = orig_fits
        cs.cs = orig_cs

if __name__ == "__main__":
    import pathlib
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_e2e_with_real_file_headless(pathlib.Path(tmp_dir))

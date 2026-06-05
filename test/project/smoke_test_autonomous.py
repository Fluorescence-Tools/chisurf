import os
import json
import numpy as np
import chisurf
import pathlib
import tempfile
import sys
import traceback

# Core imports
from chisurf.core.data import DataCurve
from chisurf.macros.core_fit import save_project, load_project, add_fit
from chinet.node import Node
from chisurf.core.models.model import ModelCurve
from chisurf.core.fitting.parameter import FittingParameter

# Define a model that will be found by the global resolution fallback
class SmokeLinearModel(ModelCurve):
    name = "SmokeLinearModel"
    def __init__(self, fit, **kw):
        super().__init__(fit, **kw)
        self.slope = FittingParameter(name="slope", value=1.0)
        self.intercept = FittingParameter(name="intercept", value=0.0)
        self.find_parameters()

    def update_model(self, **kw):
        x = getattr(self.fit.data, "x", None)
        if x is None:
            x = np.linspace(0, 10, 100)
        self.y = self.slope.value * x + self.intercept.value

def run_smoke_tests():
    print("=== ChiSurf Autonomous Smoke Test Suite ===")
    
    # 1. Chinet Stress
    print("\n[1/7] Chinet Stress Test...")
    try:
        nodes = [Node() for _ in range(500)]
        print(f"  - Created {len(nodes)} nodes successfully.")
    except Exception as e:
        print(f"  - FAILED: {e}")
        return False

    # 2. Dataset Loading
    print("\n[2/7] Dataset Loading (CSV simulation)...")
    try:
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.01, 100)
        ds = DataCurve(x=x, y=y, name="smoke_ds")
        # We must give it a path for project save/load to recognize it as 'saveable'
        ds.path = "smoke_data.csv" 
        chisurf.imported_datasets = [ds]
        chisurf.fits = []
        chisurf.cs = None
        print(f"  - Dataset '{ds.name}' registered.")
    except Exception as e:
        print(f"  - FAILED: {e}")
        return False

    # 3. Add Model Fit
    print("\n[3/7] Add Model Fit...")
    try:
        # Use add_fit directly with the model name. 
        # Since SmokeLinearModel is not in any experiment, it will use the global resolution fallback.
        add_fit(dataset_indices=[0], model_name="SmokeLinearModel")
        if len(chisurf.fits) == 0:
             print("  - FAILED: No fit was added (resolution failed).")
             return False
        assert chisurf.fits[0].model.name == "SmokeLinearModel"
        print(f"  - Fit added successfully via global resolution.")
    except Exception as e:
        print(f"  - FAILED: {e}")
        traceback.print_exc()
        return False

    # 4. Run Fit (Simulate adjustment)
    print("\n[4/7] Run Fit (Simulated)...")
    try:
        fit_group = chisurf.fits[0]
        fit_group.model.slope.value = 2.1
        fit_group.model.intercept.value = 0.95
        print(f"  - Fit parameters updated (simulated).")
    except Exception as e:
        print(f"  - FAILED: {e}")
        return False

    # 5. Save & Load Project
    print("\n[5/7] Save & Load Project...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = pathlib.Path(tmp_dir)
        project_dir = target / "smoke_proj"
        try:
            save_project(str(target), "smoke_proj")
            print("  - Project saved.")
            
            # Clear state
            chisurf.fits = []
            chisurf.imported_datasets = []
            
            # Load
            load_project(str(project_dir))
            
            if len(chisurf.fits) == 0:
                print("  - FAILED: No fits loaded.")
                return False
                
            val = float(np.atleast_1d(chisurf.fits[0].model.slope.value)[0])
            print(f"  - Loaded slope: {val}")
            if not np.isclose(val, 2.1):
                print(f"  - FAILED: slope {val} != 2.1")
                return False
            print("  - Project loaded and verified.")
        except Exception as e:
            print(f"  - FAILED: {e}")
            traceback.print_exc()
            return False

    # 6. Cross-Fit Link
    print("\n[6/7] Cross-Fit Link logic check...")
    try:
        p1 = chisurf.fits[0].model.slope
        p2 = FittingParameter(name="p2", value=1.0)
        p2.link = p1
        p1.value = 3.5
        if not np.isclose(float(np.atleast_1d(p2.value)[0]), 3.5):
             print(f"  - FAILED: Linked p2.value {p2.value} != 3.5")
             return False
        print("  - Parameter linking logic verified.")
    except Exception as e:
        print(f"  - FAILED: {e}")
        return False

    # 7. Launch (Headless check)
    print("\n[7/7] Launch (Headless check)...")
    try:
        from chisurf.gui.main import Main
        print("  - GUI classes importable (headless).")
    except Exception as e:
        print(f"  - FAILED: {e}")
        return False
    
    print("\n=== All Autonomous Smoke Tests Passed! ===")
    return True

if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)

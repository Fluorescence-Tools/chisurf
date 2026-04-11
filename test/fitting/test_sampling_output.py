
import os
import shutil
import json
import time
import pytest
import numpy as np
import chisurf.fitting.fit as fit_module
from chisurf.fitting.parameter import FittingParameter

def test_sampling_directory_structure(tmp_path):
    """
    Test that sample_fit creates the expected timestamped directory structure
    and metadata files.
    """
    # 1. Setup a minimal Fit object
    class MockModel:
        def __init__(self):
            self.parameters = [
                FittingParameter(name="p1", value=1.0),
                FittingParameter(name="p2", value=2.0)
            ]
            self.parameter_values = [1.0, 2.0]
            self.parameter_names = ["p1", "p2"]
            self.n_points = 100
            self.n_free = 2
            self.__class__.__name__ = "MockModel"
            self.meta_data = {}
        def update(self):
            pass
            
    class MockFit:
        def __init__(self):
            self.model = MockModel()
            self.name = "TestFit"
            self.n_free = 2
            self.meta_data = {}
            
    fit = MockFit()
    
    # 2. Mock chisurf.macros.core_fit.save_project to avoid actual saving
    import chisurf.macros.core_fit
    original_save = chisurf.macros.core_fit.save_project
    chisurf.macros.core_fit.save_project = lambda target_path: os.makedirs(target_path, exist_ok=True)
    
    # 3. Mock the sampling backends to do nothing but return dummy results
    import chisurf.fitting.sample
    def mock_sample_emcee(fit, **kwargs):
        return {
            'chi2r': np.array([1.0, 1.1]),
            'parameter_values': np.array([[1.0, 2.0], [1.1, 2.1]]),
            'parameter_names': ["p1", "p2"]
        }
    
    original_emcee = chisurf.fitting.sample.sample_emcee
    chisurf.fitting.sample.sample_emcee = mock_sample_emcee
    
    try:
        output_base = str(tmp_path / "test_sample")
        # Run sample_fit with n_runs=1
        fit_module.sample_fit(fit, output_base, method='emcee', n_runs=1, steps=10)
        
        # 4. Verify directory structure
        # Find the timestamped directory
        dirs = [d for d in os.listdir(tmp_path) if os.path.isdir(tmp_path / d)]
        assert len(dirs) == 1, f"Expected 1 timestamped directory, found {dirs}"
        
        sampling_dir = tmp_path / dirs[0]
        assert (sampling_dir / "project").is_dir()
        assert (sampling_dir / "parameters.json").is_file()
        assert (sampling_dir / "chains").is_dir()
        
        # Check parameters.json content
        with open(sampling_dir / "parameters.json", "r") as f:
            meta = json.load(f)
            assert meta["fit_name"] == "TestFit"
            assert len(meta["parameters"]) == 2
            assert meta["parameters"][0]["name"] == "p1"
            
        # Check chains folder
        chain_files = os.listdir(sampling_dir / "chains")
        assert any(f.endswith(".er4") for f in chain_files)
        
    finally:
        # Restore mocks
        chisurf.macros.core_fit.save_project = original_save
        chisurf.fitting.sample.sample_emcee = original_emcee

if __name__ == "__main__":
    pytest.main([__file__])

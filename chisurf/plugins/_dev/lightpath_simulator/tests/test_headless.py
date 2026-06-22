"""Tests for headless optical simulation and mmCIF export."""
import pytest
import numpy as np
import sys
import json
from unittest.mock import MagicMock

# Ensure we don't have a QApplication running during these tests to prove headless-ness
def test_import_no_qt():
    """Verify we can import the simulator without Qt or a display."""
    # We should be able to import these without triggering any Qt errors
    from chisurf.plugins._dev.lightpath_simulator.simulator import OpticalPathSimulator
    from chisurf.plugins._dev.lightpath_simulator.mmcif_export import InstrumentSetting
    assert True

def test_propagate_headless():
    """Test full propagation on a minimal graph dict."""
    from chisurf.plugins._dev.lightpath_simulator.simulator import OpticalPathSimulator
    from chisurf.plugins._dev.lightpath_simulator.crosstalk import WAVELENGTHS
    
    # Simple Mock DB
    mock_db = MagicMock()
    # Mock return for get_standardized_optical_properties (Sample Dye)
    mock_db.get_standardized_optical_properties.return_value = {"qy": 0.8, "ext_coeff": 92000}
    # Mock return for get_probe_by_id
    mock_db.get_probe_by_id.return_value = {"chromophore_name": "Test Dye"}
    # Mock return for get_spectrum
    # We return a simple unit spectrum (flat)
    mock_db.get_spectrum.return_value = (WAVELENGTHS, np.ones_like(WAVELENGTHS))

    # Minimal Graph: Laser(488) -> Sample(Test Dye) -> Detector
    graph_dict = {
        "nodes": [
            {
                "id": "node_laser", 
                "type": "light_source", 
                "title": "Laser", 
                "inputs": [], 
                "outputs": [{"name": "Light", "is_output": True}],
                "config": {"source_mode": "manual", "manual_lines": "488:1.0"}
            },
            {
                "id": "node_sample", 
                "type": "sample", 
                "title": "Sample", 
                "inputs": [{"name": "In", "is_output": False}], 
                "outputs": [{"name": "Out", "is_output": True}],
                "config": {"spectrum_ids": [1]}
            },
            {
                "id": "node_det", 
                "type": "detector", 
                "title": "Detector", 
                "inputs": [{"name": "In", "is_output": False}], 
                "outputs": [],
                "config": {"detector_name": "Main Channel", "spectrum_id": 999}
            }
        ],
        "edges": [
            {"source": "node_laser", "source_port": 0, "target": "node_sample", "target_port": 0},
            {"source": "node_sample", "source_port": 1, "target": "node_det", "target_port": 0}
        ]
    }

    sim = OpticalPathSimulator(mock_db)
    sim.load_from_dict(graph_dict)
    states = sim.propagate()
    
    assert "node_det" in states
    signals = sim.get_detector_signals()
    assert len(signals) > 0
    assert signals[0]["detector"] == "Main Channel"
    assert signals[0]["intensity"] > 0

def test_export_instrument_setting():
    """Test mapping simulator state to mmCIF dataclasses."""
    from chisurf.plugins._dev.lightpath_simulator.simulator import OpticalPathSimulator
    from chisurf.plugins._dev.lightpath_simulator.mmcif_export import InstrumentSetting
    
    mock_db = MagicMock()
    mock_db.get_standardized_optical_properties.return_value = {"qy": 0.5, "ext_coeff": 100000}
    mock_db.get_probe_by_id.return_value = {"chromophore_name": "ATTO 488"}
    
    # State with 1 laser and 1 sample
    sim = OpticalPathSimulator(mock_db)
    sim.load_from_dict({
        "nodes": [
            {"id": "l1", "type": "light_source", "title": "L", "inputs": [], "outputs": ["X"], "config": {"manual_lines": "488:1.0"}},
            {"id": "s1", "type": "sample", "title": "S", "inputs": ["I"], "outputs": ["O", "D"], "config": {"spectrum_ids": [42]}}
        ],
        "edges": []
    })
    
    setting = sim.to_instrument_setting()
    assert isinstance(setting, InstrumentSetting)
    assert len(setting.lasers) == 1
    assert setting.lasers[0].wavelength_nm == 488.0
    assert len(setting.fluorophores) == 1
    assert setting.fluorophores[0].name == "ATTO 488"
    assert setting.fluorophores[0].quantum_yield == 0.5

def test_export_to_json():
    """Verify JSON serialization contains expected keys."""
    from chisurf.plugins._dev.lightpath_simulator.mmcif_export import InstrumentSetting, LaserLine
    
    setting = InstrumentSetting(id="test_inst", instrument="TestInst", lasers=[LaserLine(488.0)])
    js = setting.to_json()
    data = json.loads(js)
    
    assert data["id"] == "test_inst"
    assert "lasers" in data
    assert data["lasers"][0]["wavelength_nm"] == 488.0

if __name__ == "__main__":
    import json
    try:
        print("Running test_import_no_qt...")
        test_import_no_qt()
        print("Running test_propagate_headless...")
        test_propagate_headless()
        print("Running test_export_instrument_setting...")
        test_export_instrument_setting()
        print("Running test_export_to_json...")
        test_export_to_json()
        print("All tests passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

import pytest
import chisurf.data
import chisurf.macros.core_data

def test_dataset_grouping():
    # Create two mock FCS data groups which inherit from DataCurveGroup
    mock_fcs1 = chisurf.data.ExperimentDataCurveGroup([chisurf.data.DataCurve()])
    mock_fcs1.name = "FCS_File_1"
    mock_fcs2 = chisurf.data.ExperimentDataCurveGroup([chisurf.data.DataCurve()])
    mock_fcs2.name = "FCS_File_2"
    
    # Backup original datasets
    original_datasets = chisurf.imported_datasets
    try:
        # Import them
        chisurf.imported_datasets = [mock_fcs1, mock_fcs2]
        
        # Group them together
        chisurf.macros.core_data.group_datasets([0, 1], _from_controller=True)
        
        # Expect the resulting group to be an ExperimentDataCurveGroup
        grouped = chisurf.imported_datasets[-1]
        assert isinstance(grouped, chisurf.data.ExperimentDataCurveGroup)
        assert hasattr(grouped, 'x')
        assert grouped.x is not None
    finally:
        # Restore original datasets
        chisurf.imported_datasets = original_datasets

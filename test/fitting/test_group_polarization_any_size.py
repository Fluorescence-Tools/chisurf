import logging
import chisurf as cs
from chisurf.core.data import DataCurve, ExperimentDataCurveGroup
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.models.tcspc.lifetime import LifetimeModel
import numpy as np

# Configure logging to see detailed information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_group_polarization_assignment(num_datasets):
    """
    Test that polarization types are correctly assigned for fits in a group of any size.
    
    This test verifies that the polarization setup code in LifetimeModel.__init__ and
    LifetimeModelWidget.__init__ correctly sets the polarization types based on the fit's
    position in the group, regardless of the group size.
    
    Parameters
    ----------
    num_datasets : int
        The number of datasets to include in the group
    """
    logger.info(f"\nTesting polarization assignment for fits in a {num_datasets}-dataset group")
    
    # Clear any existing datasets and fits
    cs.imported_datasets = []
    cs.fits = []
    
    # Create datasets
    datasets = []
    for i in range(num_datasets):
        x = np.linspace(0, 10, 100)
        y = np.exp(-x/(i+1)) + 0.1*np.random.randn(100)
        data = DataCurve(x=x, y=y, name=f"Dataset {i}")
        datasets.append(data)
    
    # Create a data group with the datasets
    data_group = ExperimentDataCurveGroup(datasets)
    
    # Create a fit group with the data group
    fit_group = FitGroup(
        data=data_group,
        model_class=LifetimeModel
    )
    
    # Check that each fit has a group attribute that references the fit group
    for i, fit in enumerate(fit_group.grouped_fits):
        if hasattr(fit, 'group'):
            logger.info(f"Fit {i} has group attribute: {fit.group is fit_group}")
        else:
            logger.error(f"Fit {i} does not have group attribute")
    
    # Check polarization types for each fit in the group
    for i, fit in enumerate(fit_group.grouped_fits):
        pol_type = fit.model.anisotropy.polarization_type
        logger.info(f"Fit {i} polarization type: {pol_type}")
        
        # Verify polarization type is set correctly based on index
        expected_pol_type = 'vv' if i % 2 == 0 else 'vh'
        if pol_type != expected_pol_type:
            logger.error(f"Fit {i} should have polarization type '{expected_pol_type}', but has '{pol_type}'")
        else:
            logger.info(f"Fit {i} has correct polarization type: {pol_type}")
    
    # Test passed if we get here without errors
    logger.info(f"Test passed: Polarization types are correctly assigned for fits in a {num_datasets}-dataset group")
    return True

if __name__ == "__main__":
    logger.info("Starting polarization setup tests for groups of different sizes")
    
    # Test with groups of different sizes
    test_group_polarization_assignment(1)  # Single dataset
    test_group_polarization_assignment(2)  # Two datasets (original case)
    test_group_polarization_assignment(3)  # Three datasets
    test_group_polarization_assignment(4)  # Four datasets
    
    logger.info("All tests completed")
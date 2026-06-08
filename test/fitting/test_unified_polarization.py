import logging
import chisurf as cs
from chisurf.core.data import DataCurve, ExperimentDataCurveGroup
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.models.tcspc.lifetime import LifetimeModel
import numpy as np

# Configure logging to see detailed information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_unified_polarization_assignment():
    """
    Test that polarization types are correctly assigned using the unified approach.
    
    This test verifies that the unified polarization assignment logic in 
    Anisotropy.set_polarization_by_group_position works correctly for both
    LifetimeModel and LifetimeModelWidget.
    """
    logger.info("Testing unified polarization assignment")
    
    # Clear any existing datasets and fits
    cs.imported_datasets = []
    cs.fits = []
    
    # Create datasets of different sizes to test various scenarios
    test_group_sizes = [1, 2, 3, 4]
    
    for size in test_group_sizes:
        logger.info(f"\nTesting with group size: {size}")
        
        # Create datasets
        datasets = []
        for i in range(size):
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
        
        # Check polarization types for each fit in the group
        for i, fit in enumerate(fit_group.grouped_fits):
            pol_type = fit.model.anisotropy.polarization_type
            logger.info(f"Fit {i} polarization type: {pol_type}")
            
            # Verify polarization type is set correctly based on index
            if size == 1:
                expected_pol_type = 'vm'  # Single fit gets magic angle
            else:
                expected_pol_type = 'vv' if i % 2 == 0 else 'vh'  # Even indices get 'vv', odd indices get 'vh'
            
            if pol_type != expected_pol_type:
                logger.error(f"Fit {i} should have polarization type '{expected_pol_type}', but has '{pol_type}'")
            else:
                logger.info(f"Fit {i} has correct polarization type: {pol_type}")
    
    logger.info("\nAll tests completed")

if __name__ == "__main__":
    logger.info("Starting unified polarization assignment tests")
    test_unified_polarization_assignment()
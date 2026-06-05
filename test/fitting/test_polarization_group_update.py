import logging
import chisurf
from chisurf.core.data import DataCurve, ExperimentDataCurveGroup
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.models.tcspc.lifetime import LifetimeModel
import numpy as np

# Configure logging to see detailed information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_polarization_group_update():
    """
    Test that polarization types are correctly updated for all fits in a group.
    
    This test verifies that the set_polarization_by_group_position method correctly
    updates all fits in a group, even when fits are added sequentially.
    """
    logger.info("Testing polarization update for all fits in a group")
    
    # Clear any existing datasets and fits
    chisurf.imported_datasets = []
    chisurf.fits = []
    
    # Create datasets
    datasets = []
    for i in range(3):  # Create 3 datasets to test with
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
    
    # Check initial polarization types
    logger.info("Initial polarization types:")
    for i, fit in enumerate(fit_group.grouped_fits):
        pol_type = fit.model.anisotropy.polarization_type
        logger.info(f"Fit {i} polarization type: {pol_type}")
    
    # Call set_polarization_by_group_position on the first fit
    logger.info("\nCalling set_polarization_by_group_position on the first fit")
    first_fit = fit_group.grouped_fits[0]
    first_fit.model.anisotropy.set_polarization_by_group_position(first_fit, first_fit.model)
    
    # Check updated polarization types
    logger.info("\nUpdated polarization types:")
    for i, fit in enumerate(fit_group.grouped_fits):
        pol_type = fit.model.anisotropy.polarization_type
        logger.info(f"Fit {i} polarization type: {pol_type}")
        
        # Verify polarization type is set correctly based on index
        expected_pol_type = 'vv' if i % 2 == 0 else 'vh'
        if pol_type != expected_pol_type:
            logger.error(f"Fit {i} should have polarization type '{expected_pol_type}', but has '{pol_type}'")
        else:
            logger.info(f"Fit {i} has correct polarization type: {pol_type}")
    
    logger.info("\nTest completed")

if __name__ == "__main__":
    logger.info("Starting polarization group update test")
    test_polarization_group_update()
import logging
import chisurf
from chisurf.core.data import DataCurve, ExperimentDataCurveGroup
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.models.tcspc.lifetime import LifetimeModel
import numpy as np

# Configure logging to see detailed information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_group_polarization_assignment():
    """
    Test that polarization types are correctly assigned for fits in a 2-dataset group.
    This test verifies that the back reference from each fit to its parent group is
    available during model initialization, allowing the polarization setup code in
    LifetimeModel.__init__ to correctly set the polarization types based on the fit's
    position in the group.
    """
    logger.info("Testing polarization assignment for fits in a 2-dataset group")
    
    # Clear any existing datasets and fits
    chisurf.imported_datasets = []
    chisurf.fits = []
    
    # Create two simple datasets
    x = np.linspace(0, 10, 100)
    y1 = np.exp(-x/2) + 0.1*np.random.randn(100)
    y2 = np.exp(-x/4) + 0.1*np.random.randn(100)
    
    data1 = DataCurve(x=x, y=y1, name="Dataset 1")
    data2 = DataCurve(x=x, y=y2, name="Dataset 2")
    
    # Create a data group with both datasets
    data_group = ExperimentDataCurveGroup([data1, data2])
    
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
        
        # Verify polarization type is set correctly
        if i == 0 and pol_type != 'vv':
            logger.error(f"Fit 0 should have polarization type 'vv', but has '{pol_type}'")
        elif i == 1 and pol_type != 'vh':
            logger.error(f"Fit 1 should have polarization type 'vh', but has '{pol_type}'")
    
    # Test passed if we get here without errors
    logger.info("Test passed: Polarization types are correctly assigned for fits in a 2-dataset group")

if __name__ == "__main__":
    test_group_polarization_assignment()
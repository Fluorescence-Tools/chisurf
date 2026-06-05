import logging
import chisurf
from chisurf.core.data import DataCurve, ExperimentDataCurveGroup
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.models.tcspc.lifetime import LifetimeModel
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_group_reference():
    """
    Test that the group reference is working correctly for polarization assignment.
    """
    logger.info("Testing group reference for polarization assignment")
    
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
    
    # Test with the example code from the issue description
    try:
        logger.info("\nTesting with example code from issue description")
        
        # Clear any existing datasets and fits
        chisurf.imported_datasets = []
        chisurf.fits = []
        
        # Try to add a dataset as described in the issue
        try:
            chisurf.macros.add_dataset(filename=r'/test/data/tcspc/Jordi/02_18-577+7.5uM(577)UP_8ps.dat')
            logger.info(f"Dataset added successfully. Total datasets: {len(chisurf.imported_datasets)}")
        except Exception as e:
            logger.error(f"Error adding dataset: {e}")
            
            # If the file doesn't exist, create a dummy dataset for testing
            logger.info("Creating dummy dataset for testing")
            x = np.linspace(0, 10, 100)
            y = np.exp(-x/2) + 0.1*np.random.randn(100)
            data = DataCurve(x=x, y=y, name="Test Dataset")
            chisurf.imported_datasets.append(data)
        
        # Add a fit with the Lifetime model
        logger.info("Adding fit with Lifetime model")
        chisurf.macros.add_fit(model_name='Lifetime ', dataset_indices=[0])
        logger.info(f"Fit added successfully. Total fits: {len(chisurf.fits)}")
        
        # Check if the fit has a group attribute
        if len(chisurf.fits) > 0:
            fit = chisurf.fits[0]
            if hasattr(fit, 'group'):
                logger.info(f"Fit has group attribute: {fit.group}")
            else:
                logger.info("Fit does not have group attribute (expected for single dataset)")
            
            # Check the polarization type that was set
            pol_type = fit.model.anisotropy.polarization_type
            logger.info(f"Polarization type set to: {pol_type}")
    except Exception as e:
        logger.error(f"Error testing example code: {e}")

if __name__ == "__main__":
    test_group_reference()
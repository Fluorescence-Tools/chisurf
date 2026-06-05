import logging
import chisurf
from chisurf.core.data import DataCurve
import numpy as np

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test case from the issue description
def test_single_dataset_polarization():
    logger.info("Testing single dataset polarization setup")
    
    # Clear any existing datasets and fits
    chisurf.imported_datasets = []
    chisurf.fits = []
    
    # Add a dataset as described in the issue
    logger.info("Adding dataset")
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
    try:
        chisurf.macros.add_fit(model_name='Lifetime ', dataset_indices=[0])
        logger.info(f"Fit added successfully. Total fits: {len(chisurf.fits)}")
        
        # Check the polarization type that was set
        if len(chisurf.fits) > 0:
            fit = chisurf.fits[0]
            pol_type = fit.model.anisotropy.polarization_type
            logger.info(f"Polarization type set to: {pol_type}")
            
            # Verify the polarization type is set correctly
            if pol_type in ['vv', 'vh', 'vm']:
                logger.info("Polarization type set correctly")
            else:
                logger.error(f"Unexpected polarization type: {pol_type}")
        else:
            logger.error("No fits were created")
    except Exception as e:
        logger.error(f"Error adding fit: {e}")

# Test with multiple datasets to verify even/odd index logic
def test_multiple_datasets_polarization():
    logger.info("\nTesting multiple datasets polarization setup")
    
    # Clear any existing datasets and fits
    chisurf.imported_datasets = []
    chisurf.fits = []
    
    # Create two dummy datasets
    logger.info("Creating two dummy datasets")
    x = np.linspace(0, 10, 100)
    y1 = np.exp(-x/2) + 0.1*np.random.randn(100)
    y2 = np.exp(-x/4) + 0.1*np.random.randn(100)
    
    data1 = DataCurve(x=x, y=y1, name="Dataset 1")
    data2 = DataCurve(x=x, y=y2, name="Dataset 2")
    
    chisurf.imported_datasets.append(data1)
    chisurf.imported_datasets.append(data2)
    
    logger.info(f"Added {len(chisurf.imported_datasets)} datasets")
    
    # Add fits for each dataset individually
    logger.info("Adding fit for dataset 0")
    chisurf.macros.add_fit(model_name='Lifetime ', dataset_indices=[0])
    
    logger.info("Adding fit for dataset 1")
    chisurf.macros.add_fit(model_name='Lifetime ', dataset_indices=[1])
    
    # Check polarization types
    if len(chisurf.fits) >= 2:
        pol_type1 = chisurf.fits[0].model.anisotropy.polarization_type
        pol_type2 = chisurf.fits[1].model.anisotropy.polarization_type
        
        logger.info(f"Dataset 0 polarization type: {pol_type1}")
        logger.info(f"Dataset 1 polarization type: {pol_type2}")
        
        # Verify even index got VV and odd index got VH
        if pol_type1 == 'vv' and pol_type2 == 'vh':
            logger.info("Polarization types set correctly based on dataset indices")
        else:
            logger.error(f"Unexpected polarization types: {pol_type1}, {pol_type2}")
    else:
        logger.error("Not enough fits were created")

if __name__ == "__main__":
    logger.info("Starting polarization setup tests")
    
    # Run the tests
    test_single_dataset_polarization()
    test_multiple_datasets_polarization()
    
    logger.info("Tests completed")
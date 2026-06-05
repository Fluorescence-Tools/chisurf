import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

try:
    # Import the necessary modules
    from chisurf.core.models.tcspc.fret import FRETModel
    from chisurf.core.fitting.fit import FitGroup
    import numpy as np
    
    # Create a simple test data
    x = np.linspace(0, 10, 100)
    y = np.exp(-x)
    
    # Create a mock fit object with data
    class MockData:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    class MockFit:
        def __init__(self, x, y):
            self.data = MockData(x, y)
            self.xmin = 0
            self.xmax = len(x)
    
    # Create a mock FitGroup
    fit = MockFit(x, y)
    
    # Try to create a FRETModel and access the reference property
    model = FRETModel(fit)
    
    # This should now work without raising an AttributeError
    reference = model.reference
    
    print("Test successful! The reference property can be accessed without errors.")
    print(f"Reference shape: {reference.shape}")
    
except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
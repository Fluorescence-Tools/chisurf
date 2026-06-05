import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

try:
    # Import necessary modules
    from chisurf.gui.plots.lineplot.lineplot import LinePlot
    from chisurf.core.models.tcspc.lifetime import LifetimeModel
    import numpy as np
    
    # Create a simple test data
    x = np.linspace(0, 10, 100)
    y = np.exp(-x)
    
    # Create a mock data class
    class MockData:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    # Create a mock fit class
    class MockFit:
        def __init__(self, x, y):
            self.data = MockData(x, y)
            self.xmin = 0
            self.xmax = len(x)
            
        def get_curves(self):
            return {'data': MockData(self.data.x, self.data.y)}
    
    # Create a mock model without reference attribute
    class MockModelWithoutReference:
        def __init__(self, fit):
            self.fit = fit
            
    # Test with a model that doesn't have reference attribute
    print("Testing with model without reference attribute...")
    fit = MockFit(x, y)
    fit.model = MockModelWithoutReference(fit)
    
    # Create a LinePlot with reference_curve=True
    plot = LinePlot(fit, reference_curve=True)
    
    # Update the plot - this should not raise an error
    plot.update()
    
    # Check if the reference checkbox is disabled
    print(f"Reference checkbox enabled: {plot.plot_controller.checkBox_5.isEnabled()}")
    print(f"Reference checkbox checked: {plot.plot_controller.use_reference}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
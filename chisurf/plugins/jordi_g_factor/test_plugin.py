"""
Simplified test script for the Jordi G-Factor Calculator plugin.

This script creates a synthetic Jordi data file that can be used to test the plugin
within the ChiSurf application.
"""

import os
import numpy as np

# Create synthetic Jordi data for testing
def create_synthetic_jordi_data(filename, n_points=1000):
    """Create a synthetic Jordi file for testing."""
    # Time axis
    time = np.linspace(0, 10, n_points)
    
    # Create exponential decays
    decay = np.exp(-time / 2.0)
    
    # Add some noise
    np.random.seed(42)  # For reproducibility
    noise_level = 0.05
    parallel = decay + noise_level * np.random.randn(n_points)
    perpendicular = 0.6 * decay + noise_level * np.random.randn(n_points)  # g-factor of ~1.67
    
    # Ensure positive values
    parallel = np.maximum(parallel, 0.001)
    perpendicular = np.maximum(perpendicular, 0.001)
    
    # Combine into Jordi format (parallel followed by perpendicular)
    jordi_data = np.concatenate([parallel, perpendicular])
    
    # Save to file
    np.savetxt(filename, jordi_data, fmt='%.6f')
    
    print(f"Created synthetic Jordi data file: {filename}")
    print(f"Expected g-factor: ~1.67")
    
    return filename

if __name__ == "__main__":
    # Create a test file in the current directory
    test_file = os.path.join(os.path.dirname(__file__), 'test_jordi_data.dat')
    create_synthetic_jordi_data(test_file)
    
    print("\nTest Instructions:")
    print("1. Launch ChiSurf")
    print("2. Go to Plugins > Analysis > Jordi G-Factor Calculator")
    print("3. Click 'Load Jordi File' and select the test file at:", test_file)
    print("4. Adjust the region selector (blue shaded area) to the tail of the decay")
    print("5. Observe the g-factor and standard deviation values update")
    print("6. The expected g-factor is approximately 1.67")
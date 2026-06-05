import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

# Configure logging to show info messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the chisurf directory to the path
sys.path.append('/')

import chisurf
from chisurf.core.curve import Curve
from chisurf.core.data import DataCurve
from chisurf.core.fitting.fit import Fit
from chisurf.core.models.tcspc.lifetime import LifetimeModel

# Create a simple test data
x = np.linspace(0, 10, 100)
y = np.exp(-x) + 0.1 * np.random.randn(len(x))
data = DataCurve(x=x, y=y)

# Create a simple IRF
irf_x = np.linspace(0, 10, 100)
irf_y = np.exp(-((irf_x - 2) ** 2) / 0.5)
irf = Curve(x=irf_x, y=irf_y)

# Create a fit object
fit = Fit(model_class=LifetimeModel, data=data)

# Set the IRF
fit.model.convolve._irf = irf

# Print the initial IRF
print("Initial IRF shape:", fit.model.convolve.irf.y.shape)
print("Initial IRF sum:", np.sum(fit.model.convolve.irf.y))

# Set the IRF truncation parameters
fit.model.convolve.irf_start = 20
fit.model.convolve.irf_stop = 80

# Get the truncated IRF
truncated_irf = fit.model.convolve.irf

# Print the truncated IRF
print("Truncated IRF shape:", truncated_irf.y.shape)
print("Truncated IRF sum:", np.sum(truncated_irf.y))
print("Non-zero values in truncated IRF:", np.count_nonzero(truncated_irf.y))

# Plot the original and truncated IRF
plt.figure(figsize=(10, 6))
plt.plot(irf_x, irf_y, 'b-', label='Original IRF')
plt.plot(truncated_irf.x, truncated_irf.y, 'r-', label='Truncated IRF')
plt.axvline(x=irf_x[20], color='g', linestyle='--', label='irf_start')
plt.axvline(x=irf_x[80], color='m', linestyle='--', label='irf_stop')
plt.legend()
plt.title('Original vs Truncated IRF')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.savefig('irf_truncation_test.png')

# Get the curves from the fit object
curves = fit.get_curves()
print("Curves keys:", curves.keys())

# Check if the IRF in the curves is truncated
if 'IRF' in curves:
    irf_curve = curves['IRF']
    print("IRF curve from get_curves() shape:", irf_curve.y.shape)
    print("IRF curve from get_curves() sum:", np.sum(irf_curve.y))
    print("Non-zero values in IRF curve from get_curves():", np.count_nonzero(irf_curve.y))
    
    # Plot the IRF from get_curves()
    plt.figure(figsize=(10, 6))
    plt.plot(irf_x, irf_y, 'b-', label='Original IRF')
    plt.plot(irf_curve.x, irf_curve.y, 'g-', label='IRF from get_curves()')
    plt.axvline(x=irf_x[20], color='g', linestyle='--', label='irf_start')
    plt.axvline(x=irf_x[80], color='m', linestyle='--', label='irf_stop')
    plt.legend()
    plt.title('Original vs IRF from get_curves()')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.savefig('irf_from_get_curves_test.png')
else:
    print("No IRF curve in get_curves() result")

print("Test completed successfully!")
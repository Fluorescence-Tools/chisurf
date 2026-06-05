import logging
import chisurf
from chisurf.core.data import DataCurve, ExperimentDataCurveGroup
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.models.tcspc.lifetime import LifetimeModel
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

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

# Check polarization types for each fit in the group
for i, fit in enumerate(fit_group.grouped_fits):
    pol_type = fit.model.anisotropy.polarization_type
    logging.info(f"Fit {i} polarization type: {pol_type}")

# Expected output:
# Fit 0 polarization type: vv
# Fit 1 polarization type: vh
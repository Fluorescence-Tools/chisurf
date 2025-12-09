"""
Test script for the AnisotropyToKappaModel class.

This script creates an instance of the AnisotropyToKappaModel class,
sets some anisotropy values, and checks that the kappa^2 value is
calculated correctly.
"""

import numpy as np
from chisurf.fitting.fit import Fit
from chisurf.models.anisotropy_to_kappa import AnisotropyToKappaModel
from chisurf.plugins.kappa2_dist.k2dfun import kappasq, s2delta

# Create a fit object (required by the model)
fit = Fit()

# Create an instance of the AnisotropyToKappaModel
model = AnisotropyToKappaModel(fit=fit, r0=0.38)

# Set some anisotropy values
model.r_donor.value = 0.1
model.r_acceptor.value = 0.15
model.r_sensitized.value = 0.02

# Update the model to calculate kappa^2
model.update_model()

# Print the model information
print(model)

# Calculate the expected kappa^2 value manually for verification
s2_donor = -np.sqrt(model.r_donor.value / model.r0)
s2_acceptor = np.sqrt(model.r_acceptor.value / model.r0)
s2delta_val, delta = s2delta(
    s2_donor=s2_donor,
    s2_acceptor=s2_acceptor,
    r_inf_AD=model.r_sensitized.value,
    r_0=model.r0
)
expected_kappa_sq = kappasq(
    delta=delta,
    sD2=s2_donor,
    sA2=s2_acceptor,
    beta1=0.0,
    beta2=0.0
)

# Compare the calculated and expected values
print(f"Calculated kappa^2: {model.kappa_squared.value:.6f}")
print(f"Expected kappa^2:  {expected_kappa_sq:.6f}")
print(f"Difference:        {abs(model.kappa_squared.value - expected_kappa_sq):.6e}")

# Try different anisotropy values
print("\nTesting with different anisotropy values:")
test_values = [
    (0.05, 0.05, 0.01),
    (0.15, 0.20, 0.03),
    (0.25, 0.10, 0.05),
    (0.30, 0.30, 0.10)
]

for r_d, r_a, r_s in test_values:
    model.r_donor.value = r_d
    model.r_acceptor.value = r_a
    model.r_sensitized.value = r_s
    model.update_model()
    
    # Calculate expected value
    s2_donor = -np.sqrt(r_d / model.r0)
    s2_acceptor = np.sqrt(r_a / model.r0)
    s2delta_val, delta = s2delta(
        s2_donor=s2_donor,
        s2_acceptor=s2_acceptor,
        r_inf_AD=r_s,
        r_0=model.r0
    )
    expected_kappa_sq = kappasq(
        delta=delta,
        sD2=s2_donor,
        sA2=s2_acceptor,
        beta1=0.0,
        beta2=0.0
    )
    
    print(f"r_donor={r_d:.2f}, r_acceptor={r_a:.2f}, r_sensitized={r_s:.2f}")
    print(f"  Calculated kappa^2: {model.kappa_squared.value:.6f}")
    print(f"  Expected kappa^2:  {expected_kappa_sq:.6f}")
    print(f"  Difference:        {abs(model.kappa_squared.value - expected_kappa_sq):.6e}")
# FRET Lines Documentation

## Overview

FRET lines are a powerful tool for analyzing FRET (Förster Resonance Energy Transfer) experiments in ChiSurf. They visualize the relationship between fluorescence-averaged lifetime (τ_F) and species-averaged lifetime (τ_X).

## Quick Start

### Static FRET Line (Single State)

```python
import numpy as np
import chisurf.core.models.tcspc.fret
import chisurf.core.fluorescence.fret.fret_line

# Setup
chisurf.core.models.tcspc.fret.rda_axis = np.logspace(start=np.log(1), stop=np.log(500))

# Create with R0=52, sigma=6, tau0=4
static_fl = chisurf.core.fluorescence.fret.fret_line.StaticFRETLine(
    n_points=100,
    parameter_range=(10, 100)  # Distance range in Angstrom
)
static_fl.sigma = 6.0
static_fl.update()

# Get data
tau_f, tau_x = static_fl.conversion_function
```

### Dynamic FRET Line (Two States)

```python
# Two-state system: R1=40A, R2=80A, both sigma=6A
dynamic_fl = chisurf.core.fluorescence.fret.fret_line.DynamicFRETLine(
    distance_1=40.0,
    distance_2=80.0,
    sigma_1=6.0,
    sigma_2=6.0,
    n_points=100,
    parameter_range=(0, 1)  # Vary x(G,2) from 0 to 1
)
dynamic_fl.update()
```

## Visualization

```python
import matplotlib.pyplot as plt

# Static line
fig, ax = plt.subplots()
tau_f, tau_x = static_fl.conversion_function
ax.plot(tau_f, tau_x, 'b-', label='Static FRET Line')
ax.plot(tau_f, tau_f, 'r--', label='1:1 line')
ax.set_xlabel('$\\tau_F$ [ns]')
ax.set_ylabel('$\\tau_X$ [ns]')
ax.legend()
ax.grid(True)
ax.set_aspect('equal')
plt.show()

# Dynamic line with color gradient
fig, ax = plt.subplots()
tau_f, tau_x = dynamic_fl.conversion_function
x_values = dynamic_fl.parameter_values
sc = ax.scatter(tau_f, tau_x, c=x_values, cmap='viridis', s=50)
ax.plot(tau_f, tau_f, 'r--', label='1:1 line')
fig.colorbar(sc, label='x(G,2)')
plt.show()
```

## Theory

### Definitions

- **τ_X (Species-averaged lifetime)**: ⟨τ⟩
- **τ_F (Fluorescence-averaged lifetime)**: ⟨τ²⟩/⟨τ⟩
- **R₀ (Förster radius)**: Distance at which FRET efficiency is 50% (default: 52 Å)
- **τ₀ (Donor lifetime)**: Donor lifetime without FRET (default: 4 ns)
- **σ (Sigma)**: Width of Gaussian distance distribution

### FRET Efficiency

For a distance r:
```
E(r) = R₀⁶ / (R₀⁶ + r⁶)
```

For a distribution P(r):
```
⟨E⟩ = ∫ E(r) P(r) dr
```

### Static FRET Line

- Single Gaussian: P(r) = (1/σ√(2π)) exp(-(r-μ)²/2σ²)
- τ_X = τ₀ · (1 - ⟨E⟩)
- τ_F = τ_X² / τ₀

### Dynamic FRET Line

For two states with fractions x₁ and x₂ = 1 - x₁:
```
τ_X = x₁·τ_X₁ + x₂·τ_X₂
τ_F = (x₁·τ_X₁² + x₂·τ_X₂²) / (x₁·τ_X₁ + x₂·τ_X₂)
```

## API Reference

### StaticFRETLine

**Constructor:**
```python
StaticFRETLine(
    n_points=100,
    parameter_range=(0.1, 100.0),
    **kwargs
)
```

**Properties:**
- `sigma`: Get/set Gaussian width
- `model`: The underlying GaussianModel
- All properties from FRETLineGenerator

**Methods:**
- `update(parameter_name='R(G,1)', parameter_range=None, verbose=None, n_points=None)`

### DynamicFRETLine

**Constructor:**
```python
DynamicFRETLine(
    distance_1=40.0,
    distance_2=80.0,
    sigma_1=6.0,
    sigma_2=6.0,
    n_points=100,
    parameter_range=(0, 1),
    **kwargs
)
```

**Properties:**
- `mean_distance_1`, `mean_distance_2`: State mean distances
- `sigma_1`, `sigma_2`: State distribution widths
- `model`: The underlying GaussianModel
- All properties from FRETLineGenerator

**Methods:**
- `update(parameter_name=None, parameter_range=None, verbose=None, n_points=None)`

### FRETLineGenerator (Base Class)

**Properties:**
- `conversion_function`: (τ_F array, τ_X array)
- `parameter_values`: Array of varied parameter values
- `fret_efficiencies`: Array of FRET efficiencies
- `fluorescence_averaged_lifetimes`: τ_F array
- `species_averaged_lifetimes`: τ_X array
- `polynom_coefficients`: Polynomial coefficients
- `conversion_function_string`: String for plotting
- `transfer_efficency_string`: Transfer efficiency formula
- `fdfa_string`: FD/FA ratio formula

## Files Created

1. **Examples**: `examples/fret_lines_example.py` - Complete runnable example with plots
2. **Tests**: `test/models/test_fret_line.py` - Enhanced test suite
3. **Bug Fixes**: `chisurf/core/fluorescence/fret/fret_line.py` - Fixed import and numpy.float issues
4. **Documentation**: `docs/manual/fret_lines.rst` - Comprehensive manual entry

## Complete Example

Run the example script:
```bash
cd /path/to/chisurf
python examples/fret_lines_example.py
```

This generates:
- Console output with parameter values and calculations
- Plot file: `fret_lines_example.png`

## References

1. Peulen, T. O., Opanasyuk, O., & Seidel, C. A. M. (2017). Combining Graphical and Analytical Methods with Molecular Simulations To Analyze Time-Resolved FRET Measurements of Labeled Macromolecules Accurately. J. Phys. Chem. B, 121(35), 8211-8241.
2. Kalinin, S., et al. (2010). Detection of structural dynamics by FRET. J. Phys. Chem. B, 114(23), 7983-7995.
3. Hellenkamp, B., et al. (2018). Precision and accuracy of single-molecule FRET measurements. Nat. Methods, 15(9), 669-676.

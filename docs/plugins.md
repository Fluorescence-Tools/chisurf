# Plugin Reference

Auto-generated documentation for all ChiSurf plugins.

## Table of Contents

- Burst
  - variance analysis
- Main
  - Tools:ndXplorer
- Spectroscopy
  - Fluorescence decay:Jordi Anisotropy Decay
  - Fluorescence decay:Jordi G-Factor Calculator
  - Single-Molecule:PCH
- Structure
  - Chimol
  - Computation:QuEst
  - FRET:Kappa2 Distribution
- Tools
  - AI Settings
- Uncategorized
  - burst
  - calculator
  - chisurf
  - fcs
  - fluorescence_decay
  - icon_utils
  - microscopy
  - misc
  - modelling
  - traj
  - tttr

## Burst

## Main

### Tools:ndXplorer

*Module: `ndxplorer`*

ndXplorer

This plugin provides a powerful interface for analyzing and visualizing multidimensional 
fluorescence data within ChiSurf.

Features:
- Burst analysis for single-molecule fluorescence experiments
- Multiparameter fluorescence detection (MFD) analysis
- Interactive selection and filtering of burst events
- Visualization of multidimensional data through histograms and plots
- Support for FRET efficiency calculations and proximity ratio analysis
- Application to both solution-based measurements and image spectroscopy data

The ndXplorer tool is particularly useful for analyzing complex fluorescence datasets 
where multiple parameters need to be correlated, such as fluorescence intensity, 
lifetime, anisotropy, and spectral information. It provides an intuitive interface 
for exploring relationships between different fluorescence parameters.

For single-molecule experiments, ndXplorer enables detailed burst analysis with 
capabilities to select, filter, and categorize individual molecule detection events 
based on multiple criteria. The tool also supports advanced FRET analysis with 
various correction factors and calculation methods.

When working with image spectroscopy data, ndXplorer allows pixel-by-pixel analysis 
of multiparameter fluorescence information, enabling spatial correlation of 
spectroscopic properties.

## Spectroscopy

### Fluorescence decay:Jordi Anisotropy Decay

*Module: `jordi_anisotropy`*

Jordi Anisotropy Calculator

This plugin provides interactive computation and visualization of fluorescence anisotropy r(t)
for Jordi files.

Features
- Load Jordi ASCII files and split into VV (parallel) and VH (perpendicular) decays
- Compute r(t) = (VV − g·VH) / (VV + 2·g·VH)
- Apply user-specified g-factor, optional constant backgrounds (BG VV, BG VH), and a fractional
  channel shift between VV and VH (VH relative to VV)
- Plot background-corrected decays on a semilogarithmic axis and r(t) using pyqtgraph
- Select a region on r(t) to estimate r∞; r∞ is subtracted from r(t) and saved alongside the data
- Use channel indices (0..N−1) for the x-axis
- Fix the anisotropy y-range to [0, 0.45] for visual consistency
- Save outputs via “Save…”:
  • Shifted decays as a Jordi file (<base>_shifted.dat)
  • Anisotropy decay as text with columns: channel, r(t), r(t)−r∞
  • r∞ metadata CSV including source filename, region bounds, BG VV, BG VH, and g-factor
- Batch processing with drag-and-drop file list (Batch…) and CSV export of per-file r∞

This widget can run as a ChiSurf plugin (see chisurf.plugins.jordi_anisotropy.__plugin__) or standalone.

---

### Fluorescence decay:Jordi G-Factor Calculator

*Module: `jordi_g_factor`*

Jordi G-Factor Calculator

This standalone plugin calculates the g-factor based on tail matching for Jordi files.
It allows users to:
- Load and display Jordi files
- Select a tail matching region using interactive controls
- Calculate the g-factor based on the selected region
- Display the g-factor and its standard deviation in copyable text fields
- Visualize the tail-matched decays in a separate plot
- Display decay curves in semilog plots (logarithmic y-axis) for better visualization of exponential decays
- Update calculations when the region selection changes
- Apply background correction with a separate background region selector
- Display both corrected and uncorrected g-factors

The g-factor is an important correction factor in fluorescence anisotropy measurements,
accounting for the different detection efficiencies of the parallel and perpendicular
emission components.

Background correction is useful for removing constant offsets in the data, which can
improve the accuracy of the g-factor calculation, especially for data with significant
background signal.

This plugin can run independently of ChiSurf or as a ChiSurf plugin.

---

### Single-Molecule:PCH

*Module: `pch`*

Photon Counting Histogram (PCH) Analysis

This plugin provides tools for analyzing the distribution of photon counts in
fluorescence time traces. PCH analysis can reveal information about:
- Molecular brightness (ε)
- Number of molecules in the detection volume (⟨N⟩)
- Presence of multiple species with different brightness values

The plugin supports loading TTTR files, calculating PCH histograms, and fitting
them with theoretical models for single or multiple species.

## Structure

### Chimol

*Module: `chimol`*

Thin Chimol ChiSurf plugin shim.

This module is intentionally very small. It exposes the plugin menu name and
creates a :class:`MolViewPluginWindow` from the core viewer package when
loaded as a plugin or when run as a standalone application.

---

### Computation:QuEst

*Module: `quenching_estimator`*

Quenching Estimator Plugin (QuEst)

This plugin is part of the ChiSurf application and provides tools for simulating 
fluorescence quenching processes in macromolecules.

The plugin implements a diffusion simulation approach to analyze time-resolved FRET 
measurements of labeled macromolecules. It simulates the diffusion of fluorescent dyes 
around a macromolecule and calculates quenching effects based on the proximity to 
quencher residues (such as Tryptophan, Tyrosine, and Histidine).

Key features:
- Simulation of dye diffusion using accessible volume (AV) calculations
- Calculation of fluorescence quenching based on proximity to quencher residues
- Generation of fluorescence decay histograms
- Visualization of diffusion trajectories and protein structures

The diffusion simulation methodology is based on the approach described in:
Peulen, T. O., Opanasyuk, O., & Seidel, C. A. M. (2017). 
"Combining Graphical and Analytical Methods with Molecular Simulations To Analyze 
Time-Resolved FRET Measurements of Labeled Macromolecules Accurately." 
The Journal of Physical Chemistry B, 121(35), 8211-8241.
https://pubs.acs.org/doi/10.1021/acs.jpcb.7b03441

The plugin provides a graphical interface for setting up and running these simulations,
as well as for analyzing and visualizing the results.

---

### FRET:Kappa2 Distribution

*Module: `kappa2_dist`*

Kappa2 Distribution

This plugin provides tools for calculating and visualizing the distribution of the orientation factor κ² 
for Förster Resonance Energy Transfer (FRET) experiments.

Features:
- Calculate κ² distributions using different models (Wobbling-in-Cone, Diffusion-during-Lifetime)
- Visualize the distribution of κ² values
- Calculate the effect of κ² uncertainty on apparent FRET distances
- Support for known and unknown donor-acceptor orientations
- Incorporate steady-state anisotropy measurements to estimate fluorophore mobility

The orientation factor κ² is a critical parameter in FRET that describes the relative orientation 
of the donor emission dipole and the acceptor absorption dipole. It affects the calculation of 
the Förster radius (R₀) and consequently the distance measurements derived from FRET experiments.

In most FRET applications, κ² is assumed to be 2/3 (≈0.667), which is valid only when both 
fluorophores undergo isotropic rotational diffusion that is much faster than the fluorescence 
lifetime. However, in many biological systems, this assumption may not hold due to restricted 
rotational mobility of the fluorophores.

This plugin allows researchers to model more realistic κ² distributions based on experimental 
anisotropy data, providing more accurate distance measurements in FRET experiments where the 
standard assumptions about fluorophore mobility may not apply.

## Tools

### AI Settings

*Module: `ai_settings`*

AI Settings plugin for configuring API providers and backends.

This plugin provides a GUI for managing centralized AI API settings,
including provider selection, base URL, model selection, and API key.

## Uncategorized

### burst

*Module: `burst`*

No description available.

---

### calculator

*Module: `calculator`*

No description available.

---

### chisurf

*Module: `chisurf`*

No description available.

---

### fcs

*Module: `fcs`*

No description available.

---



### fluorescence_decay

*Module: `fluorescence_decay`*

No description available.

---

### icon_utils

*Module: `icon_utils`*

No description available.

---

### microscopy

*Module: `microscopy`*

No description available.

---

### misc

*Module: `misc`*

No description available.

---

### modelling

*Module: `modelling`*

No description available.

---

### traj

*Module: `traj`*

No description available.

---

### tttr

*Module: `tttr`*

No description available.

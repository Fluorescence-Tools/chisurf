# TTTR:Compute Microtime LUT

This plugin provides an interactive tool for Time-Tagged Time-Resolved (TTTR) TAC linearization using the Felekyan et al. algorithm (RSI 2005).

## Features

- **Interactive GUI**: Visual selection of linear regions in TAC histograms
- **Parameter Controls**: Adjustable NTAC requirements, Noffset, RNG seed, and preview settings
- **Real-time Preview**: Live visualization of linearized histograms
- **File Support**: Load multiple TTTR file formats (.spc, .ht3, .ptu, .t3r, .t2r)
- **Data Cleanup**: Tools for zeroing out problematic bins and applying thresholds
- **Export Options**: Save LUTs in various formats (.txt, .csv, .npy, .npz)
- **Drag & Drop**: Easy file loading interface

## Algorithm

Implements the paper-faithful Felekyan et al. TAC linearization algorithm:

1. Build TAC histogram from microtime data
2. Identify linear plateau region
3. Compute relative bin widths and cumulative scaling factors
4. Apply stochastic rebinning to correct non-linearities

## Dependencies

- `tttrlib`: Required for loading TTTR files
- `pyqtgraph`: Interactive plotting
- `PyQt5`: GUI framework
- `numpy`: Numerical computations
- `scipy`: Scientific computing
- `click`: Command-line interface (for CLI mode)

## Usage

1. Load TTTR files using the file dialog or drag & drop
2. Adjust the orange region selector to choose the linear plateau
3. Fine-tune basic parameters (NTAC requirements, Noffset, preview photons, normalization)
4. Click "Advanced Parameters..." to open a separate dialog with additional controls:
   - RNG seed for reproducible results
   - Spike mitigation options (wrap spike prevention)
   - Low-count threshold for data cleanup
   - Zero-out tools for bin cleanup
5. Use "Clear list" button to remove all loaded files and reset the interface
6. Preview the linearized histogram in real-time
7. Save the LUT or export corrected microtimes

## Default Settings

- **Normalize raw TAC by region mean**: Disabled (unchecked)
- **Mitigate wrap spike**: Disabled (unchecked)
- **Advanced Parameters**: Accessible via separate dialog

The streamlined interface focuses on the most essential parameters while keeping advanced options accessible in a dedicated dialog window.

## Installation

Ensure all dependencies are installed:

```bash
conda install -c conda-forge tttrlib pyqtgraph pyqt numpy scipy click
```

## References

Felekyan et al. "Analysis of diffusion-induced broadening of the Förster resonance energy transfer efficiency histogram: nanoparticle sizing and screening of interparticle interaction." Review of Scientific Instruments 76.8 (2005): 083104.

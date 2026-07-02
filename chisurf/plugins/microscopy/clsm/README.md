# Confocal Laser Scanning Microscopy (CLSM) Image Analysis Plugin

This plugin provides a graphical interface for analyzing and processing CLSM image data from Time-Tagged Time-Resolved (TTTR) measurements.

## Features

- Loading and visualization of CLSM-TTTR data files
- Creation of various image representations (intensity, mean microtime)
- Interactive pixel selection with adjustable brush tools
- Region of interest (ROI) creation and management
- Generation of fluorescence decay histograms from selected pixels
- Fourier Ring Correlation (FRC) analysis for image resolution estimation
- Export of decay histograms for further analysis in ChiSurf
- Support for multiple frames and channels

## Overview

The CLSM plugin is particularly useful for fluorescence lifetime imaging microscopy (FLIM) data analysis, allowing researchers to extract time-resolved fluorescence information from specific regions within microscopy images. The pixel selection tools enable precise isolation of structures of interest, while the integrated decay histogram generation provides immediate feedback on the fluorescence decay characteristics of the selected area.

The plugin supports various TTTR file formats and microscope setups, with configurable parameters for frame markers, line markers, and pixel settings to accommodate different CLSM acquisition systems.

## Architecture

The plugin follows the standard ChiSurf core/api/rpc/cli split, with an
AutoForm-driven GUI:

| Path | Role |
|------|------|
| `core/` | Qt-free algorithms — FRC, image representations, decay extraction, setup presets. Safe to import headlessly. |
| `api/` | `models.py` dataclasses, `contract.py` RPC contract, and `clsm.py` orchestration returning JSON-safe dicts. |
| `backend/services.py` | `register_services(dispatcher)` wiring the `clsm.*` RPC methods. |
| `client.py` | `ClsmClient`, transport-agnostic (in-process dispatcher or ZMQ). |
| `cli/` | `csc clsm …` Click commands (`setups`, `info`, `representation`, `decay`, `frc`, `contract`). |
| `gui/` | `view_model.py` (Qt-free state + logic), `clsm.view.json` (AutoForm layout), `sections.py` (custom imaging widgets), `tool.py` (the tool). |
| `manifest.json` | Plugin id, entrypoints (gui/cli/services) and `rpc_methods`. |

The static settings render as AutoForm fields; the dynamic control bar, the
image-brush canvas and the ROI list are registered AutoForm `custom` sections;
the decay and FRC plots are declarative `plot` sections.

### Headless usage

```bash
csc clsm setups
csc clsm info data.ptu --setup "Leica SP5" --channels 0,1
csc clsm decay data.ptu --setup "Leica SP5" -c 0,1 --threshold 0.5 -o decay.txt
csc clsm frc data.ptu --setup "Leica SP5" -c 0,1 -o frc.txt
```

## Requirements

- Python packages:
  - PyQt5, pyqtgraph
  - numpy, scipy, numba
  - scikit-image (ROI mask import/export)
  - tttrlib (for TTTR/CLSM file handling)

## Usage

1. Launch the plugin from the ChiSurf menu: Imaging > CLSM-Draw
2. Load a TTTR file containing CLSM data
3. Configure image parameters:
   - Frame and line markers
   - Pixel settings
   - Channel selection
4. Visualize the image with different representations:
   - Intensity image
   - Mean microtime image
   - Other derived parameters
5. Select regions of interest using the brush tools
6. Generate and analyze fluorescence decay histograms from selected pixels
7. Export data for further analysis in ChiSurf

## Applications

- Fluorescence Lifetime Imaging Microscopy (FLIM)
- Time-resolved fluorescence microscopy
- Subcellular localization studies
- Protein-protein interaction analysis via FRET-FLIM
- Microenvironment sensing using lifetime-sensitive fluorophores
- Multi-parameter image analysis
- Correlative microscopy studies

## Benefits

- Integrates spatial and temporal fluorescence information
- Enables precise selection of regions of interest
- Provides immediate feedback on fluorescence decay characteristics
- Supports advanced analysis techniques like FRC for resolution estimation
- Streamlines the workflow from image acquisition to decay analysis
- Facilitates extraction of quantitative data from microscopy images

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.
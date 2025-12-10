# RICS TTTR/ICS files {#rics-tttr-ics}

This section describes how to load data for **Raster Image Correlation Spectroscopy (RICS)**
from TTTR and image stack files into ChiSurf using the **RICS** experiment.

## Supported input formats

The RICS reader supports two main input types:

- **TTTR files** handled by `tttrlib.TTTR`, typically PicoQuant PTU or similar
  time-tagged formats.
- **Image stacks** (TIFF): single-plane or multi-frame TIFF files that contain
  pre-binned intensity images.

## Reader setup: RICS (TTTR/ICS)

In the main GUI, select:

- **Experiment**: `RICS`
- **Setup**: `RICS (TTTR/ICS)`

Then load either:

- A TTTR file (e.g. `*.ptu`), or
- A TIFF image stack (e.g. `*.tif`, `*.tiff`).

The reader will:

1. For TIFF stacks
   - Read the image data via `imageio`.
   - Normalize the stack to shape `(n_frames, ny, nx)`.
   - Optionally select a single color channel for multi-channel images.
2. For TTTR files
   - Load the TTTR container using `tttrlib.TTTR`.
   - Pass the TTTR data (and selected routing channels) to
     `tttrlib.CLSMImage` to reconstruct a confocal image stack.
3. Compute the image correlation spectroscopy (ICS) stack using
   `tttrlib.CLSMImage.compute_ics`, with options for:
   - ROI selection (`x_range`, `y_range`),
   - Average subtraction mode (`subtract_average` = frame/stack),
   - Frame shifting (`frame_shift`) for temporal RICS variants,
   - Optional FFT centering (`fftshift`).
4. Build a ChiSurf `DataCurve` where the x-axis indexes pixels and the y-axis
   contains the flattened mean ICS, with uncertainties from the per-pixel
   standard error over frames.

## Typical workflow

1. Configure basic RICS settings in the GUI (channel, ROI, average subtraction,
   frame shift, etc.).
2. Load a TTTR file or TIFF stack recorded with raster scanning.
3. Inspect the intensity image and ICS maps in the RICS controller.
4. Choose a suitable RICS model (e.g. simple or triplet model) from the model
   dropdown.
5. Fit the model and inspect residuals, diffusion coefficients, and other
   parameters.

For details on RICS models and extensions, see the main RICS documentation and
related ChiSurf manuals.

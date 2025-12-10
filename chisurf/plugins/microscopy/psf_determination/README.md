# PSF Determination Plugin

This plugin provides **3D Point Spread Function (PSF) characterization** for fluorescence microscopy based on 3D Gaussian PSF fits to bead scans.

## Features

### Current Implementation

- **3D Stack Visualization**
  - Load TIFF image stacks (single-channel or multi-channel).
  - Browse z-slices using `pyqtgraph.ImageView`.
  - Histogram-based intensity adjustment.

- **Manual Bead Selection**
  - Click on bead positions in the image viewer.
  - Red circle marker shows selected position.
  - Selection works across all z-slices.

- **3D Gaussian PSF Fitting**
  - Extracts ROI around selected bead (configurable xy and z size).
  - Fits full 3D Gaussian model: `A * exp(-0.5 * ((x-x0)/σx)² + (y-y0)/σy)² + (z-z0)/σz)²) + offset`.
  - Uses scipy's robust `least_squares` optimizer with bounds.

- **Fit Results Display**
  - **Pixel units**: fitted center (x, y, z), sigma (σ_x, σ_y, σ_z), amplitude, offset.
  - **Physical units** (nm): FWHM_x, FWHM_y, FWHM_z, average FWHM_xy.
  - **PSF shape metrics**: axial ratio (σ_z / σ_xy).
  - **Fit quality**: residual norm, optimization success flag.

- **Configurable Parameters**
  - Pixel size (nm): Camera pixel size.
  - Z step (nm): Distance between z-slices.
  - ROI xy (pixels): Lateral ROI size for fitting.
  - ROI z (slices): Axial ROI size for fitting.

## Usage

1. **Launch the plugin**
   - From ChiSurf: `Plugins → Imaging:PSF Determination`
   - Or run directly: `python -m chisurf.plugins.psf_determination`

2. **Load a bead scan**
   - Click `Load Stack`.
   - Select a 3D TIFF file (e.g., bead scan with 100 nm TetraSpec beads).

3. **Configure parameters**
   - Set pixel size and z step to match your microscope settings.
   - Adjust ROI size (default: 15×15×15 is usually good for isolated beads).

4. **Select and fit beads**
   - Browse z-slices to find a well-isolated bead.
   - Click on the bead center in the image.
   - Click `Fit Selected Bead`.
   - Results appear in the right panel.

5. **Interpret results**
   - **FWHM_xy**: Lateral PSF resolution (typically 200–400 nm for confocal).
   - **FWHM_z**: Axial PSF resolution (typically 500–1000 nm for confocal).
   - **Axial ratio**: Should be ~2–3 for confocal microscopy (z worse than xy).
   - **Success: True**: Fit converged successfully.

## Requirements

- `numpy`
- `scipy` (for `least_squares` fitting)
- `pyqtgraph` (for 3D visualization)
- `imageio` (for TIFF loading)
- `qtpy` (Qt bindings)

## Planned Features

### Phase 1: Automated Bead Detection
- Quantile-based threshold detection per z-slice.
- Distance matrix clustering to find isolated beads.
- Automatic rejection of beads near borders or other beads.

### Phase 2: Advanced PSF Models
- 1D Gaussian fits along x, y, z cuts.
- Gaussian beam width profiles: w_x(z), w_y(z).
- Fit Gaussian beam model to extract w0, z_R (Rayleigh range).

### Phase 3: Batch Processing
- Process all beads in a stack automatically.
- Filter beads by axial ratio, fit quality, etc.
- Export ROIs as individual TIFF stacks.
- Average filtered beads into a single super-resolved PSF.

### Phase 4: Multi-Channel Support
- Detect beads in channel 1, apply to all channels.
- Compare PSF characteristics across channels.

## Algorithm Details

### 3D Gaussian Model

The fitted function is:

```
I(x, y, z) = A * exp(-0.5 * [(x-x₀)²/σₓ² + (y-y₀)²/σᵧ² + (z-z₀)²/σᵧ²]) + offset
```

Where:
- `A`: amplitude (peak intensity above background)
- `x₀, y₀, z₀`: PSF center coordinates
- `σₓ, σᵧ, σᵧ`: Gaussian standard deviations in each dimension
- `offset`: background intensity

### FWHM Calculation

Full Width at Half Maximum (FWHM) is related to sigma by:

```
FWHM = 2√(2 ln 2) * σ ≈ 2.355 * σ
```

This is the standard measure of resolution in microscopy.

### Axial Ratio

```
Axial ratio = σ_z / σ_xy
```

where `σ_xy = (σₓ + σᵧ) / 2`.

For confocal microscopy, typical values are 2–3 (axial resolution is worse than lateral).

## References

1. **PSF Characterization**
   - Pawley, J. (2006). *Handbook of Biological Confocal Microscopy*. Springer.
   - Richards, B., & Wolf, E. (1959). *Electromagnetic diffraction in optical systems. II. Structure of the image field in an aplanatic system*. Proc. R. Soc. Lond. A, 253, 358-379.

2. **Gaussian Beam Optics**
   - Siegman, A. E. (1986). *Lasers*. University Science Books.

## License

Same as ChiSurf (check project root).

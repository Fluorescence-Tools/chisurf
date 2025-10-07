# IRF Estimator Plugin

## Overview

The **IRF Estimator** plugin provides blind instrument response function (IRF) estimation from fluorescence decay data without requiring separate IRF measurements. This plugin implements the algorithm described in Gómez-Sánchez et al., Biophysical Reports, 2024.

## Features

- **Load Decay Data**: Load fluorescence decay data in Jordi format
- **Automatic IRF Estimation**: Estimate IRF using blind deconvolution
- **Interactive Parameters**: Adjust estimation parameters in real-time
- **Side-by-Side Visualization**: View decay and estimated IRF simultaneously
- **Save IRF**: Export estimated IRF in Jordi format
- **ChiSurf Integration**: Load estimated IRF directly into ChiSurf for analysis

## Usage

### 1. Load Decay Data

2. Select a Jordi format file containing fluorescence decay data
3. The decay will be displayed in the left plot (logarithmic scale)

### 2. Adjust Parameters (Optional)

- **Time/Channel (ns)**: Time per channel in nanoseconds (default: 1.0 ns) - used only for lifetime calculation
- **SG Window Length**: Savitzky-Golay filter window for boundary detection (range: 5-101, default: 11, must be odd)
- **SG Poly Order**: Polynomial order for SG filter (range: 1-10, default: 3)
- **RL Iterations**: Number of Richardson-Lucy deconvolution iterations (range: 5-2000, default: 500)
- **Regularization**: Median filter size for noise reduction (range: 1-51, default: 3, set to 1 to disable)
- **Manual Background**: Background offset to subtract from data (auto-estimated from last 10% of data)
- **Use Range Selection**: Enable interactive range selection in the decay plot (default: off)
- **Auto-Update IRF**: Automatically re-estimate IRF when parameters change using 50 RL iterations (default: off)

**Note**: The plugin works in channel units. The Time/Channel parameter is only used to convert the estimated lifetime to nanoseconds for display.

#### Manual Background Correction

The background value is auto-estimated from the last 10% of the decay data when a file is loaded. You can:
1. **Accept auto-estimate**: Use the suggested value (usually accurate)
2. **Adjust manually**: Fine-tune the background level based on visual inspection
3. **Set to 0**: Disable background correction entirely

When background > 0:
- A cyan dashed line shows the background-corrected decay
- IRF estimation uses the corrected data
- This improves IRF quality by removing constant offset

#### Auto-Update IRF

When "Auto-Update IRF" is checked:
1. IRF is automatically re-estimated when you change parameters
2. Uses **50 RL iterations** for fast updates (instead of the full 500)
3. Provides real-time feedback as you adjust parameters
4. Click "Estimate IRF" button for final high-quality estimation with full iterations

**Use Cases:**
- Quickly explore different parameter combinations
- Find optimal SG window length and polynomial order
- Adjust regularization and see immediate effects
- Fine-tune background correction interactively

#### Range Selection

When "Use Range Selection" is checked:
1. A green shaded region appears on the decay plot
2. Drag the edges to select the region of interest
3. Data outside this range will be zeroed out for IRF estimation (not visible in plot)
4. This helps focus the estimation on the relevant decay region
5. Useful for excluding artifacts, background, or unwanted signal regions

### 3. Estimate IRF

1. Click **"Estimate IRF"** button
2. Wait for the estimation process to complete (progress dialog shown)
3. View results:
   - **Estimated Lifetime (τ)**: Fluorescence lifetime in nanoseconds
   - **Decay Rate (k)**: Decay rate constant
   - **Amplitude (A)**: Fitted amplitude
   - **Offset (C)**: Background offset
4. View plot showing all results (all in one plot for easy comparison):
   - **Blue solid**: Measured decay (original data)
   - **Cyan dashed**: Background-corrected decay (if background > 0)
   - **Red dashed**: Fitted exponential (over full channel range)
   - **Green solid**: Estimated IRF (scaled to decay height, thresholded at 1 count)
   - **Orange dashed**: Forward model (IRF ⊗ Exponential) - should match measured decay
   
   **X-axis**: Channel number (not time)
   **Y-axis**: Intensity in counts/channel (logarithmic scale)
   **Note**: IRF is thresholded at 1 count to remove noise floor and show only significant signal

### 4. Save or Use IRF

- **Save IRF (Jordi)**: Save the estimated IRF to a Jordi format file
- **Load IRF to ChiSurf**: Prepare IRF for use in ChiSurf analysis

## Algorithm

The plugin uses a multi-step blind IRF estimation algorithm:

1. **Decay Boundary Detection**: Savitzky-Golay filtering identifies decay start (t0) and end (t1)
2. **Exponential Fitting**: Fits truncated exponential model to decay region
3. **Kernel Generation**: Creates normalized deconvolution kernel
4. **Richardson-Lucy Deconvolution**: Iteratively deconvolves IRF from measured data
5. **Regularization**: Applies median filtering to reduce noise amplification

## Tips for Best Results

### Data Quality
- Use high signal-to-noise ratio decay data
- Ensure sufficient photon counts (>10,000 peak counts recommended)
- Avoid saturated or clipped data

### Parameter Tuning

**For Noisy Data:**
- Increase regularization (e.g., 5, 7, or 9)
- Reduce RL iterations (e.g., 100-200)
- Increase SG window length (e.g., 15, 21, or 31)

**For Clean Data:**
- Use default or lower regularization (3)
- Use default or increase RL iterations for better resolution (500-1000)
- Use smaller SG window length (e.g., 9 or 11)

**For High Resolution IRF:**
- Increase RL iterations significantly (1000-2000)
- Use minimal regularization (1 or 3)
- Ensure high SNR in decay data

**Time/Channel:**
- Set accurately based on your TCSPC system for correct lifetime calculation
- Common values: 0.0244 ns/ch (4096 channels, 100 ns range), 0.0488 ns/ch (2048 channels)
- This parameter only affects the lifetime display in nanoseconds
- The IRF estimation itself works in channel units

**Range Selection:**
- Use to exclude early artifacts (e.g., laser pulse, scatter)
- Use to exclude late noise or background regions
- Focus estimation on the clean exponential decay region
- The selected range should include the main decay but exclude problematic regions

### Validation

After estimation, check the single plot for:
1. **Fitted Exponential** (red dashed): Should match measured decay (blue) well across the entire channel range
2. **IRF Shape** (green, scaled & thresholded): Should be smooth and bell-shaped, narrower than the decay, positioned near the decay start
   - Thresholded at 1 count to show only significant signal
   - Should be much narrower than the measured decay
3. **Lifetime**: Should be reasonable for your sample (typically 0.5-10 ns) - displayed in both ns and channels
4. **Forward Model** (orange dashed): The convolution of IRF ⊗ Exponential should closely match the measured decay (blue)
   - **This is the most important validation check**
   - If orange dashed line overlaps with blue solid line, the IRF estimation is accurate
   - Discrepancies indicate issues with the exponential model or estimation parameters
   - Good agreement confirms the IRF can reproduce the measured data
   
**All curves are displayed in the same plot for easy visual comparison on a logarithmic scale.**

## File Formats

### Jordi Format
- Text file with two concatenated channels
- First half: VV (parallel) channel
- Second half: VH (perpendicular) channel
- For IRF files, both channels contain the same IRF data

## Limitations

1. **Single Exponential Assumption**: Works best with mono-exponential or dominant single-exponential decays
2. **Shared Decay Rate**: All channels share the same decay rate
3. **CPU Only**: No GPU acceleration
4. **Memory**: Entire dataset must fit in RAM (not an issue for typical TCSPC data)

## Reference

**Gómez-Sánchez, A., Fersini, F., Zappone, S., Slenders, E., Donato, M., Pelicci, S., Tortarolo, G., Bega, G., Bouzin, M., Cardarelli, F., Lanzanò, L., Koho, S. V., & Vicidomini, G. (2024).** "Blind instrument response function identification from fluorescence decays." _Biophysical Reports_, 4(2), 100155.

DOI: [10.1016/j.bpr.2024.100155](https://doi.org/10.1016/j.bpr.2024.100155)

## Troubleshooting

### "No IRF" Warning
- Make sure you've loaded a decay file first
- Click "Estimate IRF" before trying to save

### Estimation Fails
- Check that your decay data has sufficient signal
- Try adjusting parameters (increase regularization, reduce iterations)
- Ensure time step is set correctly

### Poor IRF Quality
- Increase regularization to reduce noise
- Check that decay data quality is sufficient
- Try different SG filter parameters

### IRF Too Smooth
- Reduce regularization (try 1 or 3)
- Increase RL iterations
- Use smaller SG window length

## Integration with ChiSurf

The estimated IRF can be used in ChiSurf for:
- Fluorescence lifetime analysis
- Time-resolved anisotropy measurements
- FRET analysis
- Any analysis requiring IRF deconvolution

## Standalone Usage

The plugin can also run standalone without ChiSurf:

```bash
python -m chisurf.plugins.irf_estimator
```

## Support

For issues or questions:
- ChiSurf GitHub: https://github.com/fluorescence-tools/chisurf
- Plugin location: `chisurf/plugins/irf_estimator/`

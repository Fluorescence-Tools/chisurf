# SM Acquisition Simulation CLI

## Overview

The `run_simulation.py` script provides a **headless command-line interface** for running single-molecule photon simulations without launching the full GUI. It generates comprehensive output including:

- **SPC files**: Multiple .spc files containing simulated photon data
- **Decay histograms**: TCSPC decay curves for each detection channel
- **FCS correlations**: Auto- and cross-correlation curves
- **Count rate traces**: Time-resolved photon count rates
- **Summary statistics**: JSON summary with key metrics

## Quick Start

### Basic Usage

```bash
# Run simulation with default parameters (1M photons, 10 files)
python run_simulation.py

# Quick test with fewer photons
python run_simulation.py --n-photons 10000 --output-dir ./test_sim

# Production run with 10M photons
python run_simulation.py --n-photons 10000000 --photons-per-file 500000
```

### Custom Parameters

```bash
# Customize molecular properties
python run_simulation.py \
    --n-photons 5000000 \
    --diffusion 5.0 \
    --brightness 100 \
    --n-molecules 30 \
    --output-dir ./sim_d5_q100

# Fast simulation without plots
python run_simulation.py --n-photons 1000000 --no-plots
```

## Command-Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--n-photons` | `-n` | 1000000 | Total number of photons to generate |
| `--photons-per-file` | `-p` | 100000 | Photons per SPC output file |
| `--output-dir` | `-o` | Auto-generated | Output directory path |
| `--diffusion` | `-D` | 3.0 | Diffusion coefficient (μm²/s) |
| `--brightness` | `-q` | 50.0 | Molecular brightness (photons/molecule/μs) |
| `--n-molecules` | `-M` | 50.0 | Initial number of molecules |
| `--debug` | `-d` | False | Enable debug logging |
| `--no-plots` | | False | Skip plot generation (faster) |

## Output Files

Each simulation creates a timestamped directory containing:

```
simulation_output_YYYYMMDD_HHMMSS/
├── m000.spc                    # SPC file 0 (first 100k photons)
├── m001.spc                    # SPC file 1 (next 100k photons)
├── ...
├── m009.spc                    # SPC file 9 (last batch)
├── decay_histograms.png        # TCSPC decay curves (4 channels)
├── correlations.png            # FCS correlation curves (G00, G11, G01)
├── count_rates.png             # Count rate traces (4 channels)
├── summary.json                # Summary statistics (JSON format)
└── simulation.log              # Detailed execution log
```

## Understanding the Output

### 1. SPC Files

Binary files in BH SPC-132 format containing:
- Macrotime (photon arrival time)
- Microtime (TCSPC time within laser period)
- Channel routing (0-3)

Compatible with:
- ChiSurf TCSPC analysis tools
- FCS correlation analysis
- Burst analysis pipelines

### 2. Decay Histograms (`decay_histograms.png`)

Shows TCSPC decay curves for each detection channel:
- **Channel 0**: Parallel polarization
- **Channel 1**: Perpendicular polarization  
- **Channels 2-3**: Additional routing channels

Key information:
- Photon count per channel
- Decay pattern (should be ~flat for CW excitation)
- Background levels

### 3. Correlation Curves (`correlations.png`)

FCS correlation functions:
- **G00**: Auto-correlation channel 0 (parallel)
- **G11**: Auto-correlation channel 1 (perpendicular)
- **G01**: Cross-correlation (parallel vs perpendicular)

Expected features:
- Diffusion time τ_D ~ w₀²/(4D)
- Amplitude G(0) ~ 1/N
- Decay shape depends on focus geometry

### 4. Count Rate Traces (`count_rates.png`)

Time-resolved photon counting rates:
- Shows rate stability over measurement
- Mean count rate per channel
- Fluctuations indicate molecular dynamics

### 5. Summary JSON (`summary.json`)

Machine-readable statistics:
```json
{
  "timestamp": "2025-11-16T14:30:00",
  "parameters": { ... },
  "statistics": {
    "total_photons": 1000000,
    "channels": {
      "0": {
        "photon_count": 507234,
        "fraction": 0.507,
        "mean_count_rate_kHz": 12.5
      },
      ...
    }
  },
  "correlations": { ... }
}
```

## Examples

### Example 1: Test Simulation (Quick)

```bash
python run_simulation.py --n-photons 10000 --output-dir ./test
```

**Output**: 1 file (m000.spc), ~40 KB, completes in <1 second

### Example 2: Standard Measurement

```bash
python run_simulation.py \
    --n-photons 1000000 \
    --photons-per-file 100000 \
    --diffusion 3.0 \
    --brightness 50.0
```

**Output**: 10 files, ~4 MB total, ~10-30 seconds

### Example 3: Bright, Slowly Diffusing Molecules

```bash
python run_simulation.py \
    --n-photons 5000000 \
    --diffusion 1.0 \
    --brightness 200.0 \
    --n-molecules 20 \
    --output-dir ./bright_slow
```

**Output**: 50 files, ~20 MB, longer correlation times

### Example 4: Production Run

```bash
python run_simulation.py \
    --n-photons 50000000 \
    --photons-per-file 1000000 \
    --output-dir ./production \
    --debug
```

**Output**: 50 files, ~200 MB, detailed logging

## Simulation Parameters

### Physical Parameters

- **N_species**: Number of molecular species (currently: 1)
- **M**: Initial molecule count (default: 50)
- **D**: Diffusion coefficient in μm²/s (default: 3.0)
  - Typical: GFP ~80, small dyes ~300-500, large proteins ~10-50
- **q**: Molecular brightness in photons/molecule/μs (default: 50.0 per channel)
  - Depends on laser power, quantum yield, cross-section
- **box_xy, box_z**: Simulation volume size (2 × 4 μm)
- **focus_param**: [w₀, z₀] = [0.3, 2.0] μm (confocal focus)

### Technical Parameters

- **N_tac_channels**: TAC resolution (4096 bins)
- **tac_dt**: TAC bin width (4.069 ps)
- **laser_period**: Laser repetition (13.596 ns for 73.6 MHz)
- **pulsed_exc**: 0=CW, 1=pulsed excitation

## Troubleshooting

### Issue: No photons generated

**Symptom**: `summary.json` shows 0 photons  
**Solution**: Increase `--n-molecules` or `--brightness`

### Issue: Simulation too slow

**Symptom**: Takes >1 minute for 1M photons  
**Solutions**:
- Use `--no-plots` to skip visualization
- Reduce `--n-photons`
- Check system resources (CPU/memory)

### Issue: Missing correlation plots

**Symptom**: `correlations.png` not created  
**Cause**: tttrlib not available or too few photons  
**Solution**: Ensure >10k photons per channel

### Issue: Plots look wrong

**Symptom**: Decay histogram is empty or correlation is flat  
**Solutions**:
- Check that photons were generated (inspect .spc files)
- Verify `--brightness` is not too low
- Increase `--n-photons`

## Integration with ChiSurf GUI

The generated SPC files can be opened in ChiSurf for:

1. **TCSPC Fitting**: Load decay histograms
2. **FCS Analysis**: Import for correlation analysis
3. **Burst Analysis**: Process photon streams
4. **Anisotropy**: Use parallel/perpendicular channels

## Performance

Typical performance on modern hardware:

| Photons | Files | Time | Throughput |
|---------|-------|------|------------|
| 10k | 1 | <1s | ~50k ph/s |
| 100k | 1 | ~2s | ~50k ph/s |
| 1M | 10 | ~20s | ~50k ph/s |
| 10M | 100 | ~3min | ~55k ph/s |

*Note: Includes simulation + file I/O + plotting*

## Technical Details

### Data Flow

1. **Simulation**: Burbulator DLL generates photon trajectories
2. **Conversion**: Photons → BH SPC-132 format
3. **File Writing**: Batch into multiple .spc files
4. **Processing**: Decode photon records
5. **Analysis**: Compute histograms, correlations, count rates
6. **Visualization**: Generate plots with matplotlib
7. **Summary**: Export statistics to JSON

### File Formats

**SPC-132 Format** (8 bytes per photon):
```
Byte 0-3: Macrotime (lower 32 bits)
Byte 4-5: Microtime (12 bits) + routing (4 bits)
Byte 6-7: Macrotime overflow + markers
```

## Advanced Usage

### Scripting Multiple Simulations

```bash
#!/bin/bash
# Scan diffusion coefficients

for D in 1.0 2.0 3.0 5.0 10.0; do
    python run_simulation.py \
        --n-photons 1000000 \
        --diffusion $D \
        --output-dir ./scan_D_${D} \
        --no-plots
done
```

### Parameter Files

Save frequently-used parameters in a shell script:

```bash
# params_standard.sh
N_PHOTONS=1000000
PHOTONS_PER_FILE=100000
DIFFUSION=3.0
BRIGHTNESS=50.0
N_MOLECULES=50

python run_simulation.py \
    --n-photons $N_PHOTONS \
    --photons-per-file $PHOTONS_PER_FILE \
    --diffusion $DIFFUSION \
    --brightness $BRIGHTNESS \
    --n-molecules $N_MOLECULES
```

## See Also

- `cli_sm_acquisition.py` - Launch GUI with custom parameters
- `debug_simulation_params.py` - Inspect/validate parameters
- Main GUI: Full acquisition interface with live plots

## Support

For issues or questions:
1. Check the `simulation.log` file in output directory
2. Run with `--debug` for verbose logging
3. Verify parameters with `debug_simulation_params.py`

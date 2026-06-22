# SM Acquisition CLI and Debug Tools

This directory contains command-line tools for launching and debugging the SM Acquisition GUI.

## Quick Launcher (Recommended)

Use the convenient launcher script from the chisurf root directory:

```bash
# Show available commands
python launch_sm.py

# Launch GUI with debug
python launch_sm.py gui --debug

# Debug parameters
python launch_sm.py debug

# Show detailed help
python launch_sm.py help
```

## Files

- `cli_sm_acquisition.py` - Main CLI for launching SM Acquisition GUI
- `debug_simulation_params.py` - Debug tool for inspecting simulation parameters
- `SM_ACQUISITION_CLI_README.md` - This documentation file

## Direct Usage

You can also run the tools directly:

## CLI Options

### cli_sm_acquisition.py

- `--debug, -d`: Enable debug logging
- `--standalone, -s`: Run in standalone mode without chisurf
- `--n-photons, -n`: Number of photons to generate (N_ph_max)
- `--photons-per-file, -p`: Photons per output file (N_ph_per_file)
- `--output-dir, -o`: Output directory for SPC files
- `--species, -S`: Number of molecular species (default: 1)

### debug_simulation_params.py

- `--save FILENAME`: Save default parameters to JSON file
- `--load FILENAME`: Load and validate parameters from JSON file

## Usage Workflow

1. **Debug Parameters**: Use `debug_simulation_params.py` to inspect and validate simulation settings
2. **Launch GUI**: Use `cli_sm_acquisition.py` with appropriate parameters
3. **Configure**: In the GUI:
   - Select "Simulation" from device type dropdown
   - Click "Init Device"
   - Click "Setup" to configure detailed parameters
   - Click "Start" to run simulation
4. **Monitor**: Watch debug output showing parameter values and file generation progress

## Debug Output

The CLI provides detailed logging including:
- Parameter validation and current values
- Simulation progress and photon generation
- File writing progress
- Error messages and stack traces

## Troubleshooting

### Common Issues

1. **"name 'logging' is not defined"**: Import issue - check Python path
2. **"chisurf not available"**: Falls back to standalone mode automatically
3. **No simulation data**: Make sure to select "Simulation" device type and click "Init Device"
4. **Wrong photon counts**: Check the debug output for parameter values

### Debug Tips

- Use `--debug` flag for verbose logging
- Check `sm_acquisition_debug.log` for complete log output
- Use `debug_simulation_params.py` to validate parameter files
- Look for the "DEBUG: simulation_params = ..." output when starting simulation

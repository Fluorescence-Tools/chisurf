# BH SPC Acquisition Plugin

This plugin provides tools for acquiring single molecule data, e.g., using a Becker & Hickl SPC 830 TCSPC board.
It displays fluorescence decays of user-defined channels (up to 4) and correlation curves.
Data is acquired into RAM and saved at the end of data acquisition.

## Features

- Acquisition of time-tagged time-resolved (TTTR) data from BH SPC 830
- Display of fluorescence decays for up to 4 user-defined channels
- Display of correlation curves
- Configurable acquisition time
- RAM usage monitoring
- Data saving at the end of acquisition
- Manufacturer-agnostic wrapper for future extension to other hardware (e.g., Picoquant)
- **Built-in help system** with integrated documentation viewer
- **Command-line interface** for advanced users and automation

## Requirements

- Becker & Hickl SPC-830 TCSPC board (optional, simulation mode available)
- Becker & Hickl SPCM-DLL (part of the TCSPC Package or SPCM Data Acquisition Software)
- Python packages:
  - PyQt5
  - pyqtgraph
  - numpy
  - psutil
  - tttrlib

## Usage

### GUI Mode

1. Launch the plugin from the ChiSurf menu: Single-Molecule > BH SPC Acquisition
2. Click the blue **"Help"** button in the acquisition dock for integrated documentation
3. Initialize the device (in simulation mode or hardware mode)
4. Configure the acquisition parameters:
   - Set the acquisition time
   - Select the channels to display (up to 4)
5. Start the acquisition
6. Monitor the fluorescence decays and correlation curves in real-time
7. Save the data when the acquisition is complete

### Command Line Mode

The `sm_acquisition` plugin can be started as a **single entry point**
that supports both GUI and headless simulation CLI.

#### Unified module entry point

From the repository root:

```bash
# Start SM Acquisition (GUI or plugin mode, PyQt required)
python -m chisurf.plugins.sm_acquisition

# Force standalone GUI window (without starting full chisurf)
python -m chisurf.plugins.sm_acquisition --standalone
```

#### Headless simulation CLI (no PyQt required)

The same entry point exposes the simulation CLI when you pass `--cli`.
This uses the Burbulator DLL and the same parameter structure as the
enhanced simulation setup dialog.

```bash
# 1) Generate a simple JSON config
python -m chisurf.plugins.sm_acquisition --cli config \
  --output my_simulation.json \
  --n-species 1 \
  --n-ph-max 100000 \
  --spc-output-dir simulation_output

# 2) Run the simulation and write SPC-132 files
python -m chisurf.plugins.sm_acquisition --cli run \
  my_simulation.json \
  --output-dir simulation_output \
  --n-ph-max 100000 \
  --batch-size 50000
```

For FRET/FCS examples, the CLI also provides a preset for an
**N-state FRET system** with a brightness gradient between low- and
high-FRET states plus dynamic nearest-neighbor exchange. For example,
with 2 FRET states (low/high):

```bash
python -m chisurf.plugins.sm_acquisition --cli config \
  --output dynamic_fret_exchange_config.json \
  --fret-states 2 \
  --M 0.01 --M 0.01 \
  --D 3.0 --D 3.0 \
  --n-ph-max 200000 \
  --spc-output-dir simulation_output_dynamic_fret \
  --exchange-rate-ms 1.0
```

Then run the simulation as above with the `run` subcommand.

👉 See
`tcspc_devices/simulation/CLI_README.md` for detailed documentation of
all CLI options, JSON structure, and additional FRET/FCS presets.

#### Typical CLI workflow (including JSON editing)

1. **Generate a base JSON configuration**

   Use the `config` subcommand to create a starting JSON file:

   ```bash
   python -m chisurf.plugins.sm_acquisition --cli config \
     --output my_simulation.json \
     --n-species 1 \
     --n-ph-max 100000 \
     --spc-output-dir simulation_output
   ```

   or, for FRET/FCS:

   ```bash
   python -m chisurf.plugins.sm_acquisition --cli config \
     --output dynamic_fret.json \
     --fret-states 2 \
     --M 0.01 --M 0.01 \
     --D 3.0 --D 3.0 \
     --n-ph-max 200000 \
     --spc-output-dir simulation_output_fret
   ```

2. **(Optional) Edit the JSON**

   Open the generated JSON (e.g. `my_simulation.json` or
   `dynamic_fret.json`) in a text editor if you want to fine-tune
   parameters beyond the CLI flags:

   - Adjust the number of molecules or diffusion coefficients in
     `M` and `D`.
   - For FRET systems, directly modify the per-state brightness in
     `q` or the kinetic exchange matrix `k_nrad`.
   - Change output paths, TAC/IRF parameters, or RNG seeds.

   Advanced users can also load the JSON into the GUI **simulation
   setup dialog**, adjust parameters interactively, and save it back
   out for use with the CLI `run` step.

3. **Run the simulation headless**

   Once the JSON reflects the desired settings, call the `run`
   subcommand to generate SPC files:

   ```bash
   python -m chisurf.plugins.sm_acquisition --cli run \
     my_simulation.json \
     --output-dir simulation_output \
     --n-ph-max 100000 \
     --batch-size 50000
   ```

4. **Inspect and analyze the output**

   The CLI writes Becker & Hickl SPC-132 files (`m000.spc`, `m001.spc`,
   …) into the chosen output folder. These can be opened in downstream
   tools (e.g. FCS, burst analysis, or the GUI) for further analysis.

## Data Format

The plugin saves data in the following formats:

1. Raw data: Binary file (.bin) containing the 32-bit records from the TCSPC board
2. Decay data: NPZ file (.decay.npz) containing the fluorescence decay histograms for each channel
3. Decay data: CSV files (.ch{channel}.csv) containing the time and counts data for each channel
4. FCS curves: Kristine files (.cor) containing the correlation times, amplitudes, mean countrate, and the actual acquisition time (measured during data collection)

## Extending to Other Hardware

The plugin uses a manufacturer-agnostic wrapper (`TCSPCDevice` class) from the `chisurf.plugins.bh_spc_wrapper` module 
that can be extended to support other TCSPC hardware, such as Picoquant. To add support for a new hardware type:

1. Add a new device type to the `TCSPCDevice` class in the `chisurf.plugins.bh_spc_wrapper.wrapper` module
2. Implement the device-specific methods for initialization, measurement control, and data acquisition
3. Update the UI to include the new device type in the device selection dropdown

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.

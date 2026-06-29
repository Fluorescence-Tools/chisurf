# TCSPC Simulation CLI (headless tool)

This directory contains a command-line tool that interacts with the
Burbulator-based TCSPC simulation used by the `acq` plugin.

It is intentionally **GUI-free** and can be used on headless machines.
The JSON format it uses is compatible with the enhanced GUI
`EnhancedSimulationSetupDialog` in `setup_dialog.py`.

---

## Unified CLI: `simulation_cli.py`

The unified CLI provides two separate steps as Click subcommands:

- `config` – generate a JSON configuration for the simulation device.
- `run` – read such a JSON file and generate SPC-132 files via the
  Burbulator DLL.

### 1. `config` subcommand (generate JSON)

**File:** `simulation_cli.py` (subcommand `config`)

**Purpose:**

- Generate a JSON configuration with the same parameter structure as
  `EnhancedSimulationSetupDialog.get_parameters()`.
- Cover the **full kinetic scheme** and all relevant simulation fields:
  species arrays (`M`, `D`, `q`, `q_bg`), kinetic matrices (`k_rad`,
  `k_nrad`), dark-state rates (`k_bd`, `k_bb`, `k_db`), TAC/IRF,
  anisotropy, RNG seeds, output folder, etc.
- The resulting JSON can be:
  - loaded into the GUI via **Load JSON**,
  - edited by hand, and/or
  - used directly with the `run` subcommand.

**Usage (from repository root):**

```bash
python chisurf/plugins/core/acq/tcspc_devices/simulation/simulation_cli.py config \
  --output my_simulation.json \
  --n-species 1 \
  --n-ph-max 1000 \
  --spc-output-dir simulation_output
```

Important options:

- `--output`: JSON file to write (default `simulation_config.json`).
- `--n-species`: number of molecular species (resizes `M`, `D`, kinetics,
  decay parameters, etc.).
- `--M`: per-species initial molecule numbers (one or more values).
- `--D`: per-species diffusion coefficients.
- `--q-first-species`: six values for brightness of the first species
  `[G_P, G_S, R_P, R_S, Y_P, Y_S]`.
- `--excitation-mode`: `CW` or `Pulsed` (sets `excitation_mode` and
  `pulsed_exc` in the JSON).
- `--n-ph-max`: override `N_ph_max`.
- `--spc-output-dir`: default SPC output directory stored in the JSON
  (`spc_output_path`).

After generation you can:

- open the JSON in the GUI setup dialog, adjust any UI element (including
  kinetics), and save it back, or
- use it as-is with the `run` subcommand.

---

### 2. `run` subcommand (simulate and write SPC files)

**File:** `simulation_cli.py` (subcommand `run`)

**Purpose:**

- Read a JSON configuration and generate **Becker & Hickl SPC-132** files
  using the same Burbulator DLL as the simulation TCSPC device.
- The JSON can come either from the GUI dialog (Save JSON) or from the
  `config` subcommand.

**DLL dependency:**

- Uses `BurbulatorDLL` from `burbulator_dll_wrapper.py` in this directory.
- Requires a working Burbulator DLL as described in
  `BUILD_BURBULATOR.md`.

**Usage (from repository root):**

```bash
# Step 1: generate a small test config
python chisurf/plugins/core/acq/tcspc_devices/simulation/simulation_cli.py config \
  --output cli_test_config.json \
  --n-species 1 \
  --n-ph-max 1000 \
  --spc-output-dir simulation_output_cli_test

# Step 2: run simulation and write SPC files
python chisurf/plugins/core/acq/tcspc_devices/simulation/simulation_cli.py run \
  cli_test_config.json \
  --output-dir simulation_output_cli_test \
  --n-ph-max 1000 \
  --batch-size 500
```

This will:

- call `BurbulatorDLL.simulate_ov3` with the parameters from the JSON,
- generate up to `N_ph_max` photons,
- convert them to SPC-132 records via `convert_to_spc132`, and
- write multiple files `m000.spc`, `m001.spc`, … in the chosen output
  directory.

Key options:

- positional `config_file`: path to JSON configuration.
- `--output-dir`: override the SPC output directory
  (otherwise uses `spc_output_path` or `simulation_output/` next to the
  JSON file).
- `--n-ph-max`: override `N_ph_max` from the JSON.
- `--batch-size`: photons per SPC file (`N_ph_per_file`).

---

## Relationship to the GUI `setup_dialog`

- The JSON keys are compatible with `EnhancedSimulationSetupDialog`:
  `N_species`, `M`, `D`, `q`, `q_bg`, `k_rad`, `k_nrad`, `k_bd`, `k_bb`,
  `k_db`, `box_xy`, `box_z`, `focus_type`, `focus_param`, `dt`,
  `N_ph_max`, `N_channels`, `N_tac_channels`, `tac_dt`, `laser_period`,
  `use_gaussian_irf`, `gaussian_irf_fwhm`, `gaussian_irf_mean`,
  `gaussian_irf_sigma`, `irf_file`, `r0`, `g_factor`, `l1`, `l2`,
  `green_enabled`, `red_enabled`, `yellow_enabled`, RNG fields, output
  settings, and decay-related parameters.
- You can therefore move configurations back and forth:
  - CLI → GUI (Load JSON) → tweak → GUI → Save JSON → CLI.

---

## Current limitations

- The `run` subcommand currently supports **CW excitation only**:
  - If `pulsed_exc` is non-zero in the JSON, the CLI raises an error
    because `convert_to_spc132` would require IRF-related `F` and
    `lookup` tables. Those are not yet computed in this headless tool.
  - For pulsed use-cases, use the GUI or extend the CLI to construct the
    necessary `F`/`lookup` arrays from the decay parameters.

Other than this limitation, the CLI reuses the same DLL, parameter
structure, and BH SPC conversion path as the GUI-based simulation
acquisition device and does not modify any existing behavior.

---

## Worked example: N-state dynamic FRET/FCS (CW, nearest-neighbor exchange)

This example shows how to configure **N FRET states** suitable for FCS
and FRET-style analysis using a single preset. As a concrete case we
use `N = 2` (low- and high-FRET) but the same pattern works for
`--fret-states 3` (low/mid/high) or more.

For `--fret-states 2` we get:

- **Two species** with the **same diffusion coefficient** (same FCS
  diffusion time).
- Each species has **concentration 0.01** (i.e. `M = [0.01, 0.01]`).
- Both species diffuse identically (`D = [D, D]`).
- They differ in **FRET level** (low-FRET vs high-FRET) via different
  brightness in green vs red detection channels.
- The two species are in **dynamic exchange** with
  `k12 = k21 = 1/ms` (symmetric nearest-neighbor exchange for N=2).

The underlying simulation is CW (`excitation_mode = 'CW'`).

Internally, the 6 detection channels are ordered as::

    [G_P, G_S, R_P, R_S, Y_P, Y_S]

We treat:

- Channels 0–1 (G_P, G_S) as **donor (green)**.
- Channels 2–3 (R_P, R_S) as **acceptor (red)**.

### Step 1 – generate a base JSON for N FRET states

From the repository root (`e:\dev\chisurf`):

```bash
python -m chisurf.plugins.core.acq --cli config \
  --output dynamic_fret_exchange_config.json \
  --fret-states 2 \
  --M 0.01 --M 0.01 \
  --D 3.0 --D 3.0 \
  --n-ph-max 200000 \
  --spc-output-dir simulation_output_dynamic_fret \
  --exchange-rate-ms 1.0
```

This creates `dynamic_fret_exchange_config.json` with:

- `N_species = fret_states = 2`.
- `M = [0.01, 0.01]`.
- `D = [3.0, 3.0]`.
- CW excitation (`pulsed_exc = 0`).
- `q` set to a low/high-FRET pattern by linearly interpolating between
  the endpoints defined by `--fret-low-*` and `--fret-high-*` (for
  `fret-states=2`, this reduces to the low and high endpoints).
- `k_nrad` set to a symmetric nearest-neighbor exchange matrix with
  `k12 = k21 = exchange_rate_ms / 1000` in 1/µs.

For more states, e.g. three FRET levels (low/mid/high), use
`--fret-states 3` and provide three concentrations `--M ...` and
diffusion coefficients `--D ...`. The brightness of intermediate states
is interpolated between the low and high endpoints.

### Step 2 – run the simulation and generate SPC files

From the repository root:

```bash
python -m chisurf.plugins.core.acq --cli run \
  dynamic_fret_exchange_config.json \
  --output-dir simulation_output_dynamic_fret \
  --n-ph-max 200000 \
  --batch-size 50000
```

This will:

- Call the Burbulator DLL with an **N-state**, same-diffusion,
  dynamically interconverting FRET system (for this example, N=2).
- Generate up to 200k photons.
- Write SPC-132 files (`m000.spc`, `m001.spc`, …) into
  `simulation_output_dynamic_fret`.

The resulting data can be used for FCS analysis (correlation of total
or channel-resolved intensities) and for FRET-state analysis (e.g.
time-resolved or burst-wise donor/acceptor ratios) that includes
dynamic exchange between low-FRET and high-FRET states.

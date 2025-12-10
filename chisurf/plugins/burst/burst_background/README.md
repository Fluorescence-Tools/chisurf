# Burst Background Estimation Plugin

This plugin estimates background count rates from TTTR single-molecule data.

## Features

- Uses the same detector / PIE-window definitions as other TTTR tools via `DetectorWizardPage`.
- Loads one or more TTTR files and computes background rates per detector.
- Displays results in a table that can be copied or exported.

## Usage

- Open the plugin from the ChiSurf **Plugins → Single-Molecule → Burst Background Estimation** menu (exact label may vary).
- Configure detectors and PIE windows in the *Channel Definition* tab.
- Load TTTR files in the *Files & Results* tab.
- Click **Estimate Background** to run the analysis and populate the results table.

Implementation details are in `chisurf/plugins/burst_background/__init__.py` and use `chisurf.fluorescence.burst.estimate_background_from_bursts` for the core computation.

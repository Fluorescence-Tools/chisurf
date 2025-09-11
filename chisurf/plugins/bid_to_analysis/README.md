# BID → Analysis Converter Plugin

This plugin converts Seidel-style BID (Burst ID) files into a burstwise analysis
folder that Chisurf and related tools can consume.

Features:
- Detector setup tab to define/select detector channels and PIE windows (same as used elsewhere, via DetectorWizardPage)
- Reads BID files containing start/stop photon indices
- Finds the corresponding TTTR file by filename stem (same name, common TTTR extensions)
- Computes per-burst summary statistics using the same core as WizardTTTRPhotonFilter
- Writes BUR files to `analysis/bi4_bur/<tttr_stem>.bur`
- Writes Info files (`Info/photon_selection_parameters.json`, `Info/datetime.txt`) similar to WizardTTTRPhotonFilter
- Updates/creates an MTI summary in `analysis/Info/*.mti`
- Optional: write combined compact HDF5 of all bursts (`analysis/hdf5/burst_data_*.h5`)
- Optional: write SL5 selection files (`analysis/sl5/<tttr_stem>.json.gz`) using the BID selection mask

Usage:
1. Open from Plugins menu as "Burst: BID→Analysis"
2. Select one or more BID files (*.bid, *.bst, *.txt)
3. The plugin will write the analysis folder next to the TTTR file

Programmatic usage:
```python
from chisurf.plugins.bid_to_analysis import convert_bid_file, convert_many

# Convert single BID file
bur_path = convert_bid_file(r"C:\data\dataset.bst")

# Convert multiple BID files
bur_paths = convert_many([r"C:\data\a.bst", r"C:\data\b.bst"]) 
```

Notes:
- The plugin creates default detector definitions based on routing channels and a
  single micro-time window covering the full range.
- This is intended to provide a minimal, general solution. If your experiment
  requires specific detector/channel definitions or micro-time windows, consider
  adapting the conversion logic accordingly.

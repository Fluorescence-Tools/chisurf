# ndXplorer headless CLI recipes (PRD-31)

ndXplorer's two core workflows — **burst filtering** and **imaging** — run without
a window. Each prints JSON to stdout so pipelines can chain on it. There are two
layers:

- **`ndxplorer filter|image`** — the pure primitives (in `modules/ndxplorer`,
  chisurf-free): operate on a burst folder / image file on disk.
- **`csc ndxplorer filter|image`** — the chisurf MFDB wrappers: resolve an MFDB
  artifact to a local path, run the primitive, and optionally register the result
  back under a sample.

All commands run headless; set `QT_QPA_PLATFORM=offscreen` if a `QApplication`
needs to construct.

## 1. ndXplorer as a burst filter

### Local (primitive)

```bash
# Filter a burst folder by parameter gates (AND across --select; range per param).
ndxplorer filter --folder bursts/ \
    --select "proximity_ratio:0.30-0.70" \
    --select "n_photons:50-" \
    --out bursts_filtered/
# => {"input": …, "n_in": N, "n_out": M, "out": "bursts_filtered/"}

# Or an arbitrary pandas expression:
ndxplorer filter --folder bursts/ --query "n_photons > 50 and proximity_ratio < 0.7" \
    --out bursts_filtered/
```

The filtered folder writes `<tttr>.bst` files that reference photons **by index in
the original TTTR file** — the selection stays next to the same raw data, it is
never copied.

### MFDB round trip

```bash
# 1) raw + sample -> Burst Selection (PRD-28)
csc burst-selection analyze m000.spc m001.spc \
    --filetype SPC-130 --detectors-json det.json \
    --mfdb --db mfdb.sqlite --sample-name "DNA burst sample"
#   => prints the output-folder artifact id.

# 2) Burst Selection -> ndXplorer filter (PRD-31)
csc ndxplorer filter --from-mfdb <output_folder_artifact_id> --db mfdb.sqlite \
    --select "proximity_ratio:0.30-0.70" --select "n_photons:50-" \
    --to-mfdb --sample-id <sample_id>
#   => writes a FILTERED burst folder beside the same TTTR and registers it as a
#      new burst selection (operation_type="burst_filter", parent=source).
```

`--from-mfdb`/`--db` resolve against the configured database by default; pass the
**same** DB to both CLIs for a real round trip. With `--to-mfdb` omitted, the
command only writes the filtered folder and prints its path.

## 2. ndXplorer headless imaging

### Local (primitive)

```bash
# Render a per-pixel parameter map. "intensity" = photon count per pixel;
# any other name = mean of that column per pixel (e.g. lifetime).
ndxplorer image --file clsm.h5 --map lifetime --out lifetime_map.tiff
# => {"input": …, "map": "lifetime", "shape": [h, w], "n_selected_px": K, "out": …}
```

- `.tiff`/`.tif` out preserves the quantitative float32 map; `.png`/`.jpg` writes
  a normalized 8-bit preview.
- Restrict the render with parameter gates and/or a pixel ROI; export the kept
  photons/bursts as a sub-selection beside the source TTTR:

```bash
ndxplorer image --file clsm.h5 --map intensity \
    --select "lifetime:1.0-3.0" --roi roi_mask.tiff \
    --out roi_intensity.tiff --out-selection roi_bursts/
```

The ROI is a TIFF class mask (nonzero = selected); it is binarized and resized to
the pixel grid if needed.

### MFDB round trip

```bash
csc ndxplorer image --from-mfdb <tttr_or_image_artifact_id> --db mfdb.sqlite \
    --map lifetime --roi roi_mask.tiff \
    --out lifetime_map.tiff \
    --to-mfdb --sample-id <sample_id>
#   => registers the exported map (and any masked sub-selection) under the sample,
#      parented to the source artifact.
```

## Canonical `--select` parameters

`--select "<param>:<min>-<max>"` (open-ended `min-` or `-max` allowed). `param`
matches ndXplorer's column ids; the common burst columns are: `proximity_ratio`,
`stoichiometry`, `lifetime`, `duration`, `count rate`, `n_photons`. Use `--query`
for expressions over any loaded column.

## Round-trip parity

Outputs re-open identically in the GUI and resolve through `mfdb.datasets.open`,
so `raw+sample → BS → ndXplorer filter` and `raw(+sample) → ndXplorer image` work
the same in scripts and by hand.

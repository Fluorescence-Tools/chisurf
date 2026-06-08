# FRET Modeling CLI Reference

The FRET modeling CLI is built using the `click` library and provides command-line interfaces for all modeling tasks.

## Subcommands

### `info-backends`
Prints available and active Accessible Volume (AV) simulation backends.
```bash
python -m chisurf.plugins.modelling.fret info-backends
```

### `info`
Prints a summary of positions and distances configured in an `fps.json` file.
```bash
python -m chisurf.plugins.modelling.fret info --fps labeling.fps.json
```

### `dock`
Runs rigid-body docking using FRET-restraints.
```bash
python -m chisurf.plugins.modelling.fret dock \
    --fps labeling.fps.json \
    --pdb structure.pdb \
    --output dock_out/ \
    --n-trials 3 \
    --max-iterations 50000
```

### `refine`
Runs iterative docking and AV recalculation cycles.
```bash
python -m chisurf.plugins.modelling.fret refine \
    --fps labeling.fps.json \
    --pdb structure.pdb \
    --output refined_out/ \
    --n-cycles 3
```

### `bootstrap`
Runs parametric bootstrap error estimation.
```bash
python -m chisurf.plugins.modelling.fret bootstrap \
    --fps labeling.fps.json \
    --pdb structure.pdb \
    --output boot_out/ \
    --n-bootstrap 100
```

### `sample`
Generates Metropolis Monte Carlo coordinates trajectory.
```bash
python -m chisurf.plugins.modelling.fret sample \
    --fps labeling.fps.json \
    --pdb structure.pdb \
    --output sample_out/ \
    --n-samples 1000
```

### `screen`
Screens a directory of structure files.
```bash
python -m chisurf.plugins.modelling.fret screen \
    --fps labeling.fps.json \
    --pdb-dir structures/ \
    --output screening.csv \
    --n-threads 4
```

### `evaluate`
Performs OLGA-style structure evaluations.
```bash
python -m chisurf.plugins.modelling.fret evaluate \
    --fps labeling.fps.json \
    --pdb structures_dir/ \
    --input-type "PDB Directory" \
    --output evaluation.csv
```

### `select-pairs`
Selects optimal informative label pairs from ensemble.
```bash
python -m chisurf.plugins.modelling.fret select-pairs \
    --fps labeling.fps.json \
    --pdb-dir ensemble/ \
    --output select_report.txt \
    --max-pairs 5
```

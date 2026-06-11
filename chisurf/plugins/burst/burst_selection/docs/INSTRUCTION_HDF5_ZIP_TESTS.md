# Instruction: HDF5/ZIP API Writer Tests

## Goal

Add tests for the newly implemented `write_hdf5` and `zip_output_folder`
functions in `api/io.py`. These functions were written to achieve parity with
the legacy `WizardTTTRPhotonFilter.save_selection` output path.

## Background

`INSTRUCTION_HDF5_ZIP_TESTS.md` was delayed because the functions were
implemented only after STATUS.md was drafted. Now they exist and need tests.

The legacy format is documented in detail in the audit done 2026-06-10; the key
constraints are:

- HDF5: `pd.HDFStore`, fixed format, key `'results'`, no index, compression
  level 9, category_map JSON attribute for object column decoding.
- ZIP: `zipfile.ZIP_DEFLATED`, relative paths, directory tree preserved.

## Files to read

```bash
# The implementation under test:
cat chisurf/plugins/burst/burst_selection/api/io.py

# An example legacy output for reference:
cat modules/imp-tricks/src/IMP/bff/cgdye/scripts/mfd_burst_py.py  # if it exists

# Existing test patterns for reference:
cat chisurf/plugins/burst/burst_selection/tests/test_api.py
```

## Tasks

### 1. Add test file

Create `chisurf/plugins/burst/burst_selection/tests/test_io.py` with the
following test functions:

#### `test_get_unique_folder_path`

- Create a temporary directory
- Verify that `get_unique_folder_path(base)` returns `base` when neither folder
  nor `.zip` exists
- Create a `.zip` file at `{base}.zip`, verify the function returns `base_0`
- Create a folder at `base`, verify the function returns `base_1` (or higher)
- Clean up

#### `test_write_hdf5_roundtrip`

- Use real data: load `BH_SPC_FILE`, run `analyze_file` to get a DataFrame
- Call `write_hdf5([df], tmp_path / "test.h5")`
- Read the HDF5 back using the legacy decode pattern:
  ```python
  import json
  with pd.HDFStore(path, mode='r') as store:
      df_read = store['results']
      cat_map = json.loads(store.get_storer('results').attrs.category_map)
  for col, categories in cat_map.items():
      df_read[col] = pd.Categorical.from_codes(
          df_read[col].where(df_read[col] >= 0, -1), categories
      )
  ```
- Verify that `df_read` columns match the original DataFrame columns
- Verify that the category_map round-trips correctly (no data loss)

#### `test_write_hdf5_empty_input`

- Call `write_hdf5([], tmp_path / "empty.h5")`
- Read it back and verify it's an empty DataFrame

#### `test_zip_output_folder`

- Create a temporary directory with some nested files
- Call `zip_output_folder(tmp_dir, tmp_path / "out.zip")`
- Verify the `.zip` exists
- Extract and verify file contents match originals

#### `test_zip_output_folder_nonexistent`

- Verify `FileNotFoundError` is raised for a nonexistent folder

### 2. Run tests

```bash
/Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest \
    chisurf/plugins/burst/burst_selection/tests/test_io.py -q --no-cov \
    --tb=short
```

### 3. Run full suite

```bash
/Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest \
    chisurf/plugins/burst/burst_selection/tests -q --no-cov
```

All 29 + new tests should pass. No skips, no xfail.

### 4. Lint

```bash
/Users/tpeulen/mambaforge/bin/ruff check \
    chisurf/plugins/burst/burst_selection/api/io.py \
    chisurf/plugins/burst/burst_selection/tests/test_io.py
```

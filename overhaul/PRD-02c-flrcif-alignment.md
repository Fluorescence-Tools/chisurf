# PRD-02c: Alignment of ChiSurf MFDB Export to flrCIF

## 1. Goal

Ensure that parameters exported from ChiSurf to `mfdb` strictly adhere to the `flrCIF` standard. The `flrCIF` dictionaries (and the local `mfdb_flr_ext.dic` extension) are the **canonical source of truth** for parameter definitions. ChiSurf's internal short abbreviations must be mapped to these standard definitions so that when data is stored in `mfdb`, it perfectly matches `flrCIF`.

## 2. Background

ChiSurf internally uses short abbreviations for parameters (e.g., `E_FRET`, `bg`), which are currently defined in `chisurf/core/settings/constants/fitting_parameters.json` alongside their descriptions. Since these parameters are not exclusively "fitting" parameters, this JSON registry should be renamed to a more generic filename `parameter_registry.json`.

However, `flrCIF` is the public standard for archiving fluorescence data, and `mfdb`'s schema is built to mirror this standard. When storing data to `mfdb`, ChiSurf must output parameters that match strict `flrCIF` standards. 

To maintain alignment and avoid duplication, the `.dic` files must remain the canonical source of truth. The renamed JSON registry is for internal ChiSurf use only and does not represent a public database. We need to extend this JSON file to map the internal short names to the canonical `.dic` parameter IDs and rely on the `.dic` files for parameter definitions.

## 3. Requirements

1. **`.dic` as Canonical Source**:
   - The standard `flrCIF` dictionary (and `mfdb_flr_ext.dic` for extensions) is the absolute source of truth for parameter definitions, names, and descriptions.
   - The JSON registry should not dictate `.dic` contents. Instead, the renamed registry should map internal short names to the `flrCIF` parameter IDs.

2. **Add Missing Parameters to `.dic`**:
   - Identify parameters used by ChiSurf that are missing from the standard `flrCIF` dictionary.
   - Add these missing parameters to `chisurf/core/mfdb/data/mfdb_flr_ext.dic`, adhering strictly to `flrCIF` formatting and naming conventions.
   - You can use the existing short names and descriptions in the JSON registry to help draft the initial `.dic` entries, but once added, the `.dic` becomes the canonical source.

3. **Map Internal Short Names to flrCIF**:
   - Establish a mapping between ChiSurf's internal parameter short abbreviations and the canonical `flrCIF` parameter items.
   - Add a field `"flrcif_item_id"` to the JSON registry to explicitly define this mapping (e.g., `"flrcif_item_id": "_flr_chisurf_parameter.E_FRET"`).

4. **Update Export Logic**:
   - Update the `mfdb` export logic to utilize this mapping. When a parameter with a short name is exported from ChiSurf to `mfdb`, it must be exported using its canonical `flrCIF` identifier and category.

## 4. Implementation Steps (Detailed for Coding Agent)

**Step 1. Rename and Audit Parameters Registry**
- **File renaming**: Rename `chisurf/core/settings/constants/fitting_parameters.json` to `chisurf/core/settings/constants/parameter_registry.json`.
- **Codebase updates**: Update all references to `fitting_parameters.json`. This includes:
  - `chisurf/core/settings/__init__.py`: Update the loaded filename and the python variable name from `fitting_parameters` to `parameter_registry`.
  - `chisurf/core/parameter.py` (Line ~372): Change `getattr(chisurf.core.settings, "fitting_parameters", {})` to `getattr(chisurf.core.settings, "parameter_registry", {})`.
  - Scripts in `build_tools/dev_utils/` (like `export_fitting_parameters.py`, `export_fcs_parameters.py`, `export_tcspc_parameters.py`, `fill_fcs_descriptions.py`, `fill_tcspc_descriptions.py`): Update any hardcoded strings pointing to `fitting_parameters.json`.
  - Documentation in `docs/parameter_registry_tools.rst`.

**Step 2. Extend `mfdb_flr_ext.dic` with Missing Parameters**
- Write a Python script (`build_tools/dev_utils/align_flrcif_parameters.py`) that performs the following automated alignment.
- Read `parameter_registry.json` and extract the parameter short names and descriptions.
- Check if they exist in standard `flrCIF`. For those that don't, auto-generate `.dic` entries and append them to `chisurf/core/mfdb/data/mfdb_flr_ext.dic`.
- **Formatting Example for `.dic`**:
  ```text
  save__flr_chisurf_parameter.E_FRET
     _item.name                "_flr_chisurf_parameter.E_FRET"
     _item.category_id         flr_chisurf_parameter
     _item_type.code           float
     _chisurf_schema.table_name  flr_chisurf_parameter
     _chisurf_schema.column_name e_fret
     _item_description.description
  ;     Apparent FRET efficiency parameter E_FRET (0e00..1).
  ;
  ```
  *(Note: ensure a matching category definition `save_flr_chisurf_parameter` exists in the `.dic` file).*

**Step 3. Create the Mapping in `parameter_registry.json`**
- As part of the same `align_flrcif_parameters.py` script, modify `parameter_registry.json` in-place.
- For each parameter object in the JSON file, add a new key `"flrcif_item_id"`.
- Example: For `"E_FRET"`, inject `"flrcif_item_id": "_flr_chisurf_parameter.E_FRET"`.
- To completely avoid duplication, the implementation should explore pulling ChiSurf's internal parameter descriptions dynamically from the loaded `.dic` schema to avoid duplicating them in the JSON registry. If doing so, the "description" field in the JSON can be eventually removed or treated as a fallback.

**Step 4. Update Export Logic**
- Find the export mechanisms that save fitting parameters and model results to MFDB (typically localized in `chisurf/core/mfdb/pdbx_metadata.py`, `chisurf/core/mfdb/dictionary_schema_map.py`, or the MFDB SQLAlchemy models).
- During export from ChiSurf to MFDB, any model parameter must be written using its canonical identifier located in `"flrcif_item_id"`. This ensures the exported database perfectly mirrors the `flrCIF` standard.

**Step 5. Tests**
- Add tests in `test/fio/` verifying that:
  - All mapped parameters are successfully translated to their `flrcif_item_id`.
  - The extended dictionary `mfdb_flr_ext.dic` parses correctly using the CIF parser.
  - Exported data to `mfdb` validates against the combined `flrCIF` dictionaries.

## 5. Acceptance Criteria

- [ ] `fitting_parameters.json` is successfully renamed to `parameter_registry.json` and all codebase references are updated.
- [ ] `mfdb_flr_ext.dic` contains standard-compliant definitions for all ChiSurf parameters missing from the core `flrCIF` dictionary.
- [ ] `parameter_registry.json` contains a new `"flrcif_item_id"` field linking the internal short name to the canonical `.dic` item.
- [ ] Duplication of parameter descriptions is minimized or eliminated by treating `.dic` files as the canonical source.
- [ ] The ChiSurf to `mfdb` export functions correctly translate internal short names into standard `flrCIF` identifiers.

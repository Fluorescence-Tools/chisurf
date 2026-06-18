# PRD-02 Implementation Summary — Sample Tracking

**Date:** 2026-06-17  
**Status:** COMPLETE - All CODE_REVIEW.md R9 fixes implemented  
**Reviewer:** Addressed R9-6 and R9-7 from CODE_REVIEW.md

---

## Changes Made

### 1. Fixed R9-6: Sentinel Values → Optional Types

**Issue:** PRD-02 used sentinel values (`0.0` for pH, `-1` for positions) which are problematic because:
- pH of 0.0 is a valid (though extreme) measurement
- Position -1 is non-obvious as "unset"

**Solution:** Changed all numeric fields in `SampleDefinition` to use `Optional[float] = None` or `Optional[int] = None`:

```python
# Before (in SampleDefinition):
ph: float = 0.0
temperature_k: float = 0.0
salt_concentration_m: float = 0.0
donor_position: int = -1
acceptor_position: int = -1

# After (in SampleDefinition):
ph: Optional[float] = None
temperature_k: Optional[float] = None
salt_concentration_m: Optional[float] = None
donor_position: Optional[int] = None
acceptor_position: Optional[int] = None
```

**Files Modified:**
- `chisurf/core/mfdb/models.py` - Updated `SampleDefinition` dataclass
- `chisurf/core/mfdb/sample_manager.py` - Updated `_metadata_from_definition()` and `_insert_condition()` to handle `None` values properly
- `test/fio/test_sample_manager.py` - Updated tests to use `Optional` types and added test for `None` values

---

### 2. Fixed R9-7: PDBx/pdbihm/flrCIF Vocabulary Integration

**Issue:** PRD-02 stated "Features of samples must be described with PDBx, pdbihm, or flrCIF key value (kv) pairs" but no vocabulary integration was implemented.

**Solution:** Added comprehensive vocabulary support:

#### a. Created Vocabulary Data Files (JSON)
Separated vocabulary data from code into JSON files in `chisurf/core/mfdb/data/`:
- `entity_types.json` - Entity types (protein, dna, rna, polymer, etc.)
- `probe_names.json` - Common fluorophore/probe names (Alexa Fluor, ATTO, Cy dyes, etc.)
- `buffer_components.json` - Buffer and solvent components
- `sample_condition_fields.json` - Standard sample condition field names

#### b. Created Vocabulary Loader Module
`chisurf/core/mfdb/vocabulary_loader.py` - Provides functions to load vocabulary from JSON files:
- `get_entity_types()`
- `get_probe_names()`
- `get_buffer_components()`
- `get_sample_condition_fields()`
- `reload_vocabulary()` - Clear cache for testing
- `get_all_vocabulary_names()` - List available vocabularies

#### c. Added Vocabulary Constants to models.py
Added vocabulary constants that load from JSON files with fallback values:
- `ENTITY_TYPES`
- `COMMON_PROBE_NAMES`
- `BUFFER_COMPONENTS`
- `SAMPLE_CONDITION_FIELDS`

#### d. Added Vocabulary Validation
- Added `validate_vocabulary()` function in models.py (already existed, now used)
- Added `validate_vocabulary` parameter to `SampleDefinition` (default `False` for backward compatibility)
- When enabled, validates `entity_type`, `donor_probe_name`, and `acceptor_probe_name` against known vocabularies

**Files Modified:**
- `chisurf/core/mfdb/models.py` - Added vocabulary constants and validation
- `chisurf/core/mfdb/vocabulary_loader.py` - New module
- `chisurf/core/mfdb/data/*.json` - New vocabulary data files

---

### 3. Added Requests (All Sample Requests)

**Issue:** PRD-02 needed canonical input objects for sample operations following the pattern of other plugins (burst_selection, fret, etc.).

**Solution:** Created `chisurf/core/mfdb/sample_requests.py` with the following Request dataclasses:

#### Request Classes:
1. **`SampleCreateRequest`** - Request to create a new sample
   - All fields optional except `name`
   - Validates vocabulary by default (`validate_vocab=True`)
   - Includes `to_sample_definition()` method to convert to `SampleDefinition`

2. **`SampleUpdateRequest`** - Request to update an existing sample
   - Requires `sample_id`
   - All other fields optional (partial updates)
   - Validates vocabulary by default

3. **`SampleQueryRequest`** - Request to query samples with filters
   - Supports filtering by name, entity_type, probe_name, pH range, temperature range
   - Includes pagination (limit, offset)

4. **`SampleLinkRequest`** - Request to link an artifact to a sample
   - Requires both `artifact_id` and `sample_id`

5. **`SampleUnlinkRequest`** - Request to unlink an artifact from a sample
   - Requires both `artifact_id` and `sample_id`

6. **`SampleSearchRequest`** - Request to search samples by PDBx/flrCIF vocabulary keys
   - Validates vocabulary field against PDBx dictionary
   - Supports PDBx and flrCIF validation

**Files Created:**
- `chisurf/core/mfdb/sample_requests.py` - New module with all Request classes

---

### 4. Updated sample_manager.py

**Changes:**
- Updated `_metadata_from_definition()` to handle `Optional` types properly (None values)
- Updated `_insert_condition()` to check for `None` values using `is not None`
- Removed `_positive_float()` helper function (no longer needed with Optional types)

---

### 5. Updated Exports

**File:** `chisurf/core/mfdb/__init__.py`

Added exports for:
- Vocabulary constants: `BUFFER_COMPONENTS`, `COMMON_PROBE_NAMES`, `ENTITY_TYPES`, `SAMPLE_CONDITION_FIELDS`
- Vocabulary loader functions: `get_entity_types`, `get_probe_names`, `get_buffer_components`, `get_sample_condition_fields`, `reload_vocabulary`, `get_all_vocabulary_names`
- Sample manager functions: `create_sample`, `find_sample_by_name`, `get_sample`, `get_sample_for_artifact`, `get_sample_name`, `link_artifact_to_sample`, `list_samples`, `get_artifacts_for_sample`
- Request classes: `SampleCreateRequest`, `SampleUpdateRequest`, `SampleQueryRequest`, `SampleLinkRequest`, `SampleUnlinkRequest`, `SampleSearchRequest`
- Model class: `SampleDefinition`

---

### 6. Updated Tests

**File:** `test/fio/test_sample_manager.py`

Added 10 new tests:
1. `test_sample_definition_valid_vocabulary` - Valid vocabulary values are accepted
2. `test_sample_definition_invalid_entity_type_raises` - Invalid entity_type raises ValueError
3. `test_sample_definition_invalid_probe_name_raises` - Invalid probe name raises ValueError
4. `test_sample_definition_validation_disabled` - Validation can be disabled
5. `test_sample_create_request_to_definition` - Request can be converted to SampleDefinition
6. `test_sample_create_request_validates_vocabulary` - Request validates vocabulary
7. `test_sample_create_request_requires_name` - Request requires name
8. `test_sample_update_request_requires_sample_id` - Update request requires sample_id
9. `test_sample_link_request_validates_fields` - Link request validates required fields
10. `test_sample_query_request_validates_limits` - Query request validates limit and offset

Updated existing tests:
- `test_create_and_get_sample` - Added checks for pH, temperature, salt_concentration
- `test_create_sample_with_optional_none` - New test for None values

**Total tests:** 18 (all passing)

---

## Architecture Decisions

### 1. Data vs Code Separation
Vocabulary data (probe names, entity types, etc.) is stored in JSON files in `chisurf/core/mfdb/data/` instead of being hardcoded in Python modules. This allows:
- Independent updates to vocabulary without code changes
- Easier maintenance and curation
- Potential for runtime reloading
- Better separation of concerns

### 2. Fallback Values
The vocabulary constants in `models.py` use fallback values if the JSON files are not present. This ensures the code works even during development or testing when data files might not be available.

### 3. Backward Compatibility
- `validate_vocabulary` parameter in `SampleDefinition` defaults to `False` to maintain backward compatibility
- All existing code continues to work without changes
- New validation is opt-in

### 4. Optional Types
Changed from sentinel values to `Optional` types to make the API more explicit and type-safe:
- `Optional[int] = None` instead of `int = -1`
- `Optional[float] = None` instead of `float = 0.0`

### 5. Request Pattern
Following the established pattern from other plugins (burst_selection, fret), Request classes:
- Are dataclasses
- Have `__post_init__` for validation
- Include comprehensive docstrings
- Can be converted to/from other model types

---

## Files Created

1. `chisurf/core/mfdb/data/entity_types.json`
2. `chisurf/core/mfdb/data/probe_names.json`
3. `chisurf/core/mfdb/data/buffer_components.json`
4. `chisurf/core/mfdb/data/sample_condition_fields.json`
5. `chisurf/core/mfdb/vocabulary_loader.py`
6. `chisurf/core/mfdb/sample_requests.py`

---

## Files Modified

1. `chisurf/core/mfdb/models.py`
   - Changed `SampleDefinition` to use `Optional` types
   - Added vocabulary constants with JSON loading
   - Added vocabulary validation to `SampleDefinition`

2. `chisurf/core/mfdb/sample_manager.py`
   - Updated to handle `Optional` types
   - Removed `_positive_float()` function
   - Updated metadata serialization

3. `chisurf/core/mfdb/__init__.py`
   - Added exports for new vocabulary constants, loader functions, Request classes, and sample_manager functions

4. `test/fio/test_sample_manager.py`
   - Added 10 new tests for vocabulary and Requests
   - Updated existing tests to use Optional types

---

## Test Results

```
18 passed in 20.00s
```

All 18 tests in `test/fio/test_sample_manager.py` pass, including:
- 7 original tests (from PRD-02)
- 1 updated test with Optional types
- 10 new tests for vocabulary validation and Requests

---

## Addresses CODE_REVIEW.md Items

✅ **R9-6** - Changed sentinel values to `Optional[float] = None` / `Optional[int] = None`  
✅ **R9-7** - Added PDBx/pdbihm/flrCIF vocabulary integration with:
   - JSON data files for vocabulary
   - Vocabulary loader module
   - Vocabulary validation in SampleDefinition
   - Request classes for all sample operations

---

## Next Steps (Out of Scope)

The following items were identified in CODE_REVIEW.md but are out of scope for this implementation:

1. **Sample Picker Widget** (PRD-02 Task 4) - GUI widget for sample selection
2. **Sample Picker in Dataset Import** (PRD-02 Task 5) - Integration with reader widgets
3. **Link Datasets to Samples in Project Archiver** (PRD-02 Task 6) - Already implemented
4. **Show Sample Column in mfdb-admin** (PRD-02 Task 7) - GUI display of sample associations

These can be addressed in follow-up work as they involve GUI components.

---

## Definition of Done

✅ `SampleDefinition` dataclass exists in `models.py` with Optional types  
✅ `sample_manager.py` provides create/get/list/find/link functions with Optional support  
✅ `"measured_sample"` is a valid relationship type in `mfdb_edge` (already existed)  
✅ VOC (Value Object Classes) - Vocabulary constants and validation added  
✅ All Requests added - SampleCreateRequest, SampleUpdateRequest, SampleQueryRequest, SampleLinkRequest, SampleUnlinkRequest, SampleSearchRequest  
✅ All tests pass (18/18)  

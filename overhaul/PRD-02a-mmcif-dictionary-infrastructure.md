# PRD-02a: mmCIF Dictionary Infrastructure

**Depends on:** [PRD-020: SQLAlchemy MFDB Mapping](PRD-020-sqlalchemy-mfdb-mapping.md)
**Prerequisite for:** PRD-02 (Sample Tracking — vocabulary validation)
**Priority:** High — blocks proper sample description validation

## Execution order

Do **not** start this PRD until PRD-020 has established the bounded MFDB
SQLAlchemy mapping and fixed sample-scoped FRET pair persistence. PRD-02a can
remain mostly file/cache based, but its validation APIs will be consumed by the
sample/probe ORM boundary. That boundary must exist first so dictionary
validation is wired into one canonical persistence path instead of being
duplicated across raw SQL call sites.

## Goal

Parse all bundled mmCIF dictionary files (.dic) to extract categories, items,
descriptions, data types, and **enumerated allowed values**. Make this data
available as a fast, cached Python API for vocabulary validation, GUI
autocomplete, and flrCIF export compliance checking.

The parser must expose complete field metadata, not only enumerated fields.
Non-enumerated required fields such as `_flr_sample.id` are just as important
as enum-bearing fields because export validation depends on them.

## Current state

### What exists

| Component | Location | Status |
|-----------|----------|--------|
| Dictionary files | `chisurf/core/mfdb/data/*.dic` | 7 files downloaded (20 MB total) |
| Update script | `chisurf/core/mfdb/data/update_dictionaries.sh` | Bash script, downloads all 7 dicts |
| Parser | `chisurf/core/mfdb/pdbx_metadata.py` | **Partially implemented but incomplete** — see below |
| Cache | `chisurf/core/mfdb/data/_dictionary_cache.json` | Exists, but may reflect parser omissions |
| python-ihm | `ihm.flr` module (pip installed) | 35+ FLR classes, reads/writes flrCIF |
| MFDB ORM boundary | `chisurf/core/mfdb/orm/` | Defined by PRD-020; must be in place before this PRD is wired into sample creation |

### What's broken in `pdbx_metadata.py`

The legacy parser had these problems:

1. **Only parses one dictionary** — hardcoded to `mmcif_pdbx_v50.dic`. All FLR,
   IHM, ModelCIF categories are invisible.
2. **Descriptions always empty** — `get_pdbx_metadata_descriptions()` returns
   `{}` for all 6740 keys. The parser's `_parse_item_description` function looks
   for `_item.description` on a single line, but descriptions in .dic files
   are multi-line (`;` delimited blocks).
3. **No enumeration parsing** — the parser doesn't extract `_item_enumeration`
   values. These are the allowed values for fields like `fluorophore_type`
   (donor/acceptor/unspecified), `entity.type` (polymer/non-polymer/...).
4. **No data type parsing** — doesn't extract `_item_type.code` (int, float,
   text, code, etc.).
5. **No category metadata** — doesn't extract `_category.description` or
   `_category.mandatory_code`.
6. **No parent-child relationships** — doesn't extract `_item_linked.parent_name`
   / `_item_linked.child_name` (foreign key relationships between categories).

The current rewritten parser has fixed some of this, but R14 review found these
remaining blockers:

1. **Drops item blocks without `loop_` data** — `DictItem` objects are only
   registered through loop processing. Valid required fields such as
   `_flr_sample.id` and `_flr_sample.num_of_probes` are missing from lookups.
2. **Incomplete flrCIF coverage** — current evidence showed only 11 `flr_*`
   categories available, while the flrCIF extension has roughly 36 categories.
3. **Descriptions still incomplete** — semicolon-delimited descriptions that
   start on the next line are still fragile.
4. **Quoted loop values are still fragile** — `str.split()` breaks CIF tokens
   containing spaces or quoted values.
5. **CLI stats path crashes** — `python -m chisurf.core.mfdb.pdbx_metadata --stats`
   still references missing `dic.items`.
6. **Public request validation depends on dictionary completeness** —
   `SampleSearchRequest(vocabulary_field="flr_sample.id", ...)` should accept
   valid dictionary fields once this parser is complete.

### Dictionary file structure

Each `.dic` file uses DDL2 format. Key structures:

```
save_<category_name>                    # Category definition
   _category.id              <name>
   _category.description     <text>
   _category.mandatory_code  yes|no
   _category_key.name        "<key_item>"

save_<_category.item_name>              # Item (field) definition
   _item.name                "<_category.item>"
   _item.category_id         <category>
   _item.mandatory_code      yes|no
   _item_type.code           <type>     # int, float, text, code, ucode, ...
   _item_description.description <text> # often multi-line ;...; block
   loop_
   _item_enumeration.value
   _item_enumeration.detail
     value1  "description1"
     value2  "description2"
```

### Bundled dictionaries

| File | Categories | Description |
|------|-----------|-------------|
| `mmcif_pdbx_v50.dic` | ~700 | Core PDBx/mmCIF (coordinates, entities, sequences) |
| `mmcif_ihm_ext.dic` | ~60 | Integrative/hybrid modeling (IHM) |
| `mmcif_ihm_flr_ext.dic` | 36 | **flrCIF** — fluorescence/FRET (our primary extension) |
| `mmcif_ma.dic` | ~40 | ModelCIF (computed structure models) |
| `mmcif_std.dic` | ~180 | Original mmCIF standard |
| `mmcif_ddl.dic` | ~30 | Dictionary definition language |
| `mmcif_pdbx_v5_next.dic` | ~750 | Development/next version |

### flrCIF categories (36 categories)

These map directly to ChiSurf's `flr_*` schema tables:

| flrCIF category | ChiSurf table | Notes |
|----------------|---------------|-------|
| `flr_sample` | `flr_sample` | Sample identity |
| `flr_sample_condition` | `flr_sample_condition` | pH, temperature, buffer |
| `flr_sample_probe_details` | `flr_sample_probe` | Probe-sample mapping |
| `flr_poly_probe_position` | `flr_poly_probe_position` | Probe attachment site |
| `flr_probe_list` | `probes` | Probe registry |
| `flr_fret_forster_radius` | `flr_fret_forster_radius` | R₀, κ², n |
| `flr_fret_analysis` | `flr_fret_analysis` | Analysis records |
| `flr_fret_distance_restraint` | `flr_fret_distance_restraint` | Distance measurements |
| `flr_fret_calibration_parameters` | `flr_fret_calibration_parameters` | α, β, γ, δ |
| `flr_instrument` | `flr_instrument` | Instrument metadata |
| `flr_inst_setting` | `flr_inst_setting` | Instrument settings |
| `flr_experiment` | `flr_experiment` | Experiment records |
| `flr_entity_assembly` | `flr_entity_assembly` | Entity assemblies |
| `flr_exp_condition` | `flr_exp_condition` | Experimental conditions |
| `flr_probe_descriptor` | — | Not yet in schema |
| `flr_poly_probe_conjugate` | — | Not yet in schema |
| `flr_FPS_*` (6 categories) | — | FPS modeling (future) |
| `flr_reference_measurement*` | — | Reference measurements (future) |
| `flr_lifetime_fit_model` | — | Lifetime fitting (future) |
| `flr_peak_assignment` | — | Peak assignment (future) |
| `flr_kinetic_rate_analysis` | — | Kinetics (future) |
| `flr_relaxation_time_analysis` | — | Relaxation (future) |

### python-ihm `ihm.flr` classes (already installed)

The `ihm.flr` module provides Python classes that map 1:1 to flrCIF categories:

```python
import ihm.flr

# Key classes for sample description:
ihm.flr.Sample              # entity_assembly, num_of_probes, condition
ihm.flr.SampleCondition     # details
ihm.flr.SampleProbeDetails  # sample, probe, fluorophore_type, poly_probe_position
ihm.flr.Probe               # probe_list_entry, probe_descriptor
ihm.flr.ProbeList           # chromophore_name, reactive_probe_flag, probe_origin
ihm.flr.PolyProbePosition   # resatom, mutation_flag, modification_flag
ihm.flr.EntityAssembly      # entity, num_copies
ihm.flr.FRETForsterRadius   # donor_probe, acceptor_probe, forster_radius
ihm.flr.FRETAnalysis        # experiment, probes, forster_radius, calibration
ihm.flr.FRETDistanceRestraint  # probes, distance, error, type
ihm.flr.FLRData             # top-level container
```

## Tasks

### Task 1: Repair and complete dictionary parser

**File**: `chisurf/core/mfdb/pdbx_metadata.py`

Repair the current parser so it correctly handles DDL2 format and retains all
items, including item blocks that do not contain loop data.

**Data model to extract per item:**

```python
@dataclass
class DictItem:
    """A single item (field) from an mmCIF dictionary."""
    name: str              # e.g. "_flr_sample.id"
    category: str          # e.g. "flr_sample"
    attribute: str         # e.g. "id"
    description: str       # Multi-line description
    type_code: str         # int, float, text, code, ucode, ...
    mandatory: bool        # From _item.mandatory_code
    enumerations: list[str]  # Allowed values from _item_enumeration
    enum_details: dict[str, str]  # value → detail mapping
    parent: str | None     # Parent item name (FK relationship)

@dataclass
class DictCategory:
    """A category (table) from an mmCIF dictionary."""
    name: str              # e.g. "flr_sample"
    description: str
    mandatory: bool
    key_item: str          # Primary key item name
    items: dict[str, DictItem]  # attribute_name → DictItem
```

**API to expose:**

```python
class MmcifDictionary:
    """Parsed mmCIF dictionary with fast lookup."""

    def __init__(self, *dic_paths: Path):
        """Parse one or more .dic files."""

    @classmethod
    def load_bundled(cls) -> "MmcifDictionary":
        """Load all bundled dictionary files from chisurf/core/mfdb/data/."""

    def get_category(self, name: str) -> DictCategory | None:
        """Look up a category by name (e.g. 'flr_sample')."""

    def get_item(self, full_name: str) -> DictItem | None:
        """Look up an item by full name (e.g. '_flr_sample.id')."""

    def get_enumerations(self, full_name: str) -> list[str]:
        """Return allowed values for an item, empty if unrestricted."""

    def get_description(self, full_name: str) -> str:
        """Return the description for an item or category."""

    def search_items(self, query: str) -> list[DictItem]:
        """Substring search across all item names and descriptions."""

    def categories(self) -> list[str]:
        """Return all category names sorted."""

    def flr_categories(self) -> list[str]:
        """Return flrCIF categories only (prefix 'flr_')."""

    def validate_value(self, full_name: str, value: str) -> str | None:
        """Check if value is valid for item. Returns error message or None."""
```

**Implementation notes:**

- Multi-line descriptions use `;` delimiters:
  ```
  _item_description.description
  ;     The identifier for the sample.
        This should be unique within the data block.
  ;
  ```
  Parse by detecting `;` at start of line, collecting until next `;`.

- Enumerations use `loop_` blocks:
  ```
  loop_
  _item_enumeration.value
  _item_enumeration.detail
    donor        .
    acceptor     .
    unspecified  .
  ```

- Categories are `save_` blocks WITHOUT a leading `_` in the name.
  Items are `save_` blocks WITH a leading `_` in the name.

- Register the current `DictItem` whenever an item `save_` block ends, whether
  or not the item had a `loop_` block. This is mandatory for fields such as
  `_flr_sample.id`.

- Parse CIF loop rows with a tokeniser that respects single quotes, double
  quotes, semicolon-delimited text blocks, and `.` / `?` missing-value markers.
  Do not use plain `str.split()` for loop data.

- Use lazy loading with `functools.lru_cache` on `load_bundled()`.

- The parser must handle ALL .dic files in `data/`, not just one. Parse in
  order: ddl → std → pdbx_v50 → ihm_ext → ihm_flr_ext → ma (extensions
  overlay base).

### Task 2: Cache parsed dictionary

**File**: `chisurf/core/mfdb/pdbx_metadata.py`

Parsing 20 MB of .dic files takes time. Cache the parsed result:

1. After parsing all .dic files, serialize the result to
   `chisurf/core/mfdb/data/_dictionary_cache.json`.
2. On subsequent loads, check if the cache exists and is newer than all
   .dic files. If so, load from cache (fast path).
3. If any .dic file is newer than the cache, re-parse and regenerate.

Default cache format is JSON with categories and items as nested dicts.
Enumerations are lists. This avoids pickle versioning issues.

PRD-020 may add SQLAlchemy mappings for vocabulary/dictionary metadata. If so,
keep JSON as the import-speed cache and use any database-backed dictionary
tables only as query/index tables. Do not make SQLAlchemy models the only source
of dictionary truth; the `.dic` files remain canonical.

### Task 3: Vocabulary validation functions

**File**: `chisurf/core/mfdb/pdbx_metadata.py`

Add functions that use the parsed dictionary for runtime validation:

```python
def validate_flr_sample(fields: dict[str, str]) -> list[str]:
    """Validate a dict of flr_sample fields against the dictionary.

    Returns a list of validation errors (empty = valid).
    Checks:
    - Required fields are present (_item.mandatory_code = yes)
    - Enumerated fields have valid values
    - Types are compatible (int fields get ints, etc.)
    """

def validate_flr_value(category: str, attribute: str, value: str) -> str | None:
    """Validate a single value against the dictionary.

    Returns error message or None if valid.
    """

def suggest_values(category: str, attribute: str, prefix: str = "") -> list[str]:
    """Return allowed values for a field, filtered by prefix.

    For enumerated fields, returns the enum values.
    For probe names, returns COMMON_PROBE_NAMES.
    For entity types, returns ENTITY_TYPES.
    """
```

### Task 4: Wire validation into the sample API boundary

**Files**:
- `chisurf/core/mfdb/orm/sample_repository.py` (from PRD-020)
- `chisurf/core/mfdb/sample_manager.py`
- `chisurf/core/mfdb/sample_requests.py`

When creating a sample, validate fields that have dictionary enumerations at
the ORM-backed sample graph boundary, then keep `sample_manager.create_sample()`
as a public wrapper.

```python
from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary

def validate_sample_definition(definition):
    dic = MmcifDictionary.load_bundled()

    # Validate entity_type against _entity.type enumerations
    if definition.entity_type:
        err = dic.validate_value("_entity.type", definition.entity_type)
        if err:
            raise ValueError(f"Invalid entity_type: {err}")

    # Validate fluorophore_type
    # Validate solvent_phase against _flr_sample.solvent_phase
    # etc.
```

Use **warn, don't reject** for most fields because labs use non-standard names.
Only reject for truly closed vocabularies such as `entity_type` and
`fluorophore_type`.

Request-layer probe-name validation must match `SampleDefinition`: unknown dye
names warn, not reject. Custom dyes are valid and must not require callers to
disable validation globally.

### Task 4b: Wire dictionary field search into request objects

**File**: `chisurf/core/mfdb/sample_requests.py`

`SampleSearchRequest` must validate `vocabulary_field` against
`MmcifDictionary.load_bundled()`, not the older `chisurf.core.fio.mmcif`
metadata path. It must accept valid fields with or without a leading underscore:

- `flr_sample.id`
- `_flr_sample.id`
- `entity.type`
- `_entity.type`

If dictionary loading fails, skip field validation with a warning rather than
rejecting all searches.

### Task 5: Add dictionary introspection CLI

**File**: `chisurf/core/mfdb/pdbx_metadata.py` (or a new script)

Add a `__main__` block or management command for exploring the dictionary:

```bash
# List all flrCIF categories
python -m chisurf.core.mfdb.pdbx_metadata --list-categories flr

# Show all items in a category
python -m chisurf.core.mfdb.pdbx_metadata --category flr_sample

# Show enumerations for a field
python -m chisurf.core.mfdb.pdbx_metadata --enums _flr_sample_probe_details.fluorophore_type

# Search items by keyword
python -m chisurf.core.mfdb.pdbx_metadata --search "forster radius"

# Validate a field value
python -m chisurf.core.mfdb.pdbx_metadata --validate _entity.type protein
```

### Task 6: Write tests

**File**: `test/fio/test_pdbx_metadata.py`

```python
import pytest
from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary, DictItem

@pytest.fixture(scope="module")
def dic():
    return MmcifDictionary.load_bundled()

def test_load_bundled_finds_categories(dic):
    """Bundled dictionaries contain PDBx + IHM + FLR categories."""
    cats = dic.categories()
    assert len(cats) > 500  # PDBx alone has ~700
    assert "entity" in cats
    assert "entity_poly_seq" in cats

def test_flr_categories_present(dic):
    """flrCIF extension categories are parsed."""
    flr = dic.flr_categories()
    assert "flr_sample" in flr
    assert "flr_sample_condition" in flr
    assert "flr_sample_probe_details" in flr
    assert "flr_fret_forster_radius" in flr
    assert "flr_poly_probe_position" in flr
    assert len(flr) >= 30

def test_item_lookup(dic):
    item = dic.get_item("_flr_sample.id")
    assert item is not None
    assert item.category == "flr_sample"
    assert item.attribute == "id"

def test_fluorophore_type_enumerations(dic):
    """_flr_sample_probe_details.fluorophore_type has donor/acceptor/unspecified."""
    enums = dic.get_enumerations("_flr_sample_probe_details.fluorophore_type")
    assert "donor" in enums
    assert "acceptor" in enums
    assert "unspecified" in enums

def test_entity_type_enumerations(dic):
    """_entity.type has polymer, non-polymer, etc."""
    enums = dic.get_enumerations("_entity.type")
    assert "polymer" in enums
    assert "non-polymer" in enums
    assert "water" in enums

def test_descriptions_parsed(dic):
    """Item descriptions are non-empty."""
    desc = dic.get_description("_flr_sample.id")
    assert len(desc) > 0

def test_non_loop_items_are_retained(dic):
    """Required fields without enumeration loops are still parsed."""
    assert dic.get_item("_flr_sample.id") is not None
    assert dic.get_item("_flr_sample.num_of_probes") is not None

def test_sample_search_accepts_valid_dictionary_field():
    from chisurf.core.mfdb.sample_requests import SampleSearchRequest

    request = SampleSearchRequest(
        vocabulary_field="flr_sample.id",
        vocabulary_value="sample_1",
    )
    assert request.vocabulary_field == "flr_sample.id"

def test_validate_valid_value(dic):
    assert dic.validate_value("_flr_sample_probe_details.fluorophore_type", "donor") is None

def test_validate_invalid_value(dic):
    err = dic.validate_value("_flr_sample_probe_details.fluorophore_type", "emitter")
    assert err is not None
    assert "donor" in err or "allowed" in err.lower()

def test_search_items(dic):
    results = dic.search_items("forster")
    assert len(results) > 0
    names = [r.name for r in results]
    assert any("forster" in n for n in names)

def test_cache_roundtrip(tmp_path):
    """Parsed dictionary survives JSON cache serialize/deserialize."""
    dic1 = MmcifDictionary.load_bundled()
    cache_path = tmp_path / "cache.json"
    dic1.save_cache(cache_path)
    dic2 = MmcifDictionary.load_cache(cache_path)
    assert set(dic1.categories()) == set(dic2.categories())
    assert dic1.get_enumerations("_flr_sample_probe_details.fluorophore_type") == \
           dic2.get_enumerations("_flr_sample_probe_details.fluorophore_type")

def test_stats_cli_runs():
    """Dictionary CLI stats command does not crash."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "chisurf.core.mfdb.pdbx_metadata", "--stats"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Items:" in result.stdout
```

## Updating dictionaries

Run:
```bash
bash chisurf/core/mfdb/data/update_dictionaries.sh
```

This downloads the latest versions from wwPDB. The JSON cache
(`_dictionary_cache.json`) auto-regenerates on next import if any `.dic` file
is newer than the cache.

## API cheat sheet

| Thing | Correct | Wrong |
|-------|---------|-------|
| Dictionary data dir | `chisurf/core/mfdb/data/` | Hardcoded single file path |
| Multi-line descriptions | Parse `;`-delimited blocks | Single-line `_item.description` |
| Enumerations | `loop_` `_item_enumeration.value` | Not parsed at all |
| Category vs item | `save_flr_sample` (no `_`) = category | `save__flr_sample.id` (leading `_`) = item |
| Load | `MmcifDictionary.load_bundled()` | `parse_pdbx_keys()` (single file) |
| FLR category prefix | `flr_` (lowercase) | `FLR_` (dict files use mixed case in save blocks) |

## Definition of Done

- [ ] PRD-020 is complete enough that sample/probe/FRET persistence has one ORM-backed boundary
- [ ] `MmcifDictionary` class parses all 7 bundled `.dic` files
- [ ] Item blocks without `loop_` data are retained (`_flr_sample.id` exists)
- [ ] Descriptions correctly parsed from multi-line `;`-delimited blocks
- [ ] Enumerations extracted for all fields that have `_item_enumeration`
- [ ] Quoted loop values are parsed without `str.split()` corruption
- [ ] Data types extracted (`_item_type.code`)
- [ ] JSON cache auto-regenerates when `.dic` files change
- [ ] `validate_value()` checks enumerations and types
- [ ] `suggest_values()` returns completions for GUI autocomplete
- [ ] `search_items()` enables keyword search across all items
- [ ] CLI introspection works (`python -m chisurf.core.mfdb.pdbx_metadata`)
- [ ] flrCIF categories available (≥30 `flr_*` categories)
- [ ] PDBx categories available (≥500 categories)
- [ ] `SampleSearchRequest(vocabulary_field="flr_sample.id", ...)` accepts valid fields
- [ ] Request-layer custom probe names warn rather than raise
- [ ] `python -m chisurf.core.mfdb.pdbx_metadata --stats` exits successfully
- [ ] All tests pass
- [ ] `update_dictionaries.sh` downloads all 7 files from wwPDB

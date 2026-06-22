# PRD-030: Result Payload Formats & Codecs

> 🔢 **Infrastructure PRD.** Like PRD-020 (SQLAlchemy mapping), this is a
> foundational layer named with the `0X0` convention. It **precedes PRD-03**
> (Result Registry) and is a hard prerequisite for it. See
> `overhaul/CODE_REPORT.md` for why.

## Goal

Define, **formally and with strong typing**, the on-disk payload format for every
scientific result kind MFDB stores (FCS correlations, spectra, TTTR references/open TTTR
streams, TCSPC decays, burst tables, burst selections, anisotropy curves, PDA histograms,
…). Replace the current ad-hoc, lossy JSON serialization with a **msgpack-based,
schema-validated codec layer** that round-trips losslessly and is self-describing.

The deliverable is a small, well-tested module — `chisurf/core/mfdb/payload_codec.py` —
that exposes:

```python
blob, data_format = encode_payload(kind, obj)     # typed object -> bytes
obj               = decode_payload(blob)           # bytes -> typed object (self-describing)
schema            = get_payload_schema(kind)       # the formal schema for a kind
```

`register_result()` (PRD-03) and any `read_result()` call **only** these functions. No
plugin or registry code serializes payloads by hand.

## Why This Is Needed (the weakness)

Read `overhaul/CODE_REPORT.md`. Today:

- `data_format` on `mfdb_artifact` is a **free-text label** (`json`, `csv`, `hdf5`, …)
  that drives **no** deserialization. `MFDatabase.get_object()` returns raw `bytes`; the
  caller must guess how to parse them.
- `register_result()` serializes DataFrames via `to_json(orient="records")` and dicts via
  `json.dumps()` — **lossy**: dtype, column order, array shape, units, NaN/inf, and the
  distinction between an FCS curve and a spectrum are all lost.
- There are **three inconsistent storage idioms** already: object-store bytes, the typed
  `analysis_data` table (x/y as float64 blobs + units), and `add_spectrum` (float64 blobs
  + units). None is a general, self-describing format.
- There is **no formal schema** describing what a payload of a given `kind` must contain.

Consequences: write-only data, uneven provenance, no validation, no safe round-trip.

## Design Decisions (locked)

1. **Serialization: msgpack, not JSON.** Compact, binary, typed, fast. JSON is too big and
   weakly typed; HDF5 is too heavyweight for per-artifact payloads. `msgpack==1.1.2` is
   already installed. `msgpack_numpy` must be installed in the `arm64` conda environment
   via `mamba` and declared as a runtime dependency. The codec may still encode arrays
   explicitly (see Array Encoding) when that keeps the envelope stable.
2. **Self-describing envelope.** Every payload blob is a msgpack map with a fixed envelope
   so `decode_payload()` needs no external hint:
   ```
   {
     "v":      1,                # envelope schema version (int)
     "kind":   "fcs_correlation",# result kind (drives the body schema)
     "schema": 1,                # body schema version for this kind
     "units":  {...},            # unit annotations (see Units)
     "meta":   {...},            # free-form, small, JSON-safe metadata
     "body":   {...}             # the typed payload (validated against the kind schema)
   }
   ```
3. **`data_format` becomes meaningful and (mostly) closed.** Add `"msgpack"` to
   `DATA_FORMATS`. Payloads encoded by this layer set `data_format="msgpack"`. Raw vendor
   measurement files (ptu/spc/bh/photon_hdf5) keep their existing format labels and are
   **not** re-encoded by default — they are referenced, not rewritten (see TTTR below).
4. **Strong typing at the boundary.** Each kind has a dataclass (the in-memory type) and a
   formal schema (field names, dtypes, shapes, required/optional, units). Encoding
   validates the object against the schema; decoding validates the blob and returns the
   dataclass. Validation failures raise `PayloadSchemaError` — they are bugs, not silent.
5. **Reading routines define the schemes.** Per the requirement that "the reading routines
   should help to define general schemes": the canonical schema for each kind is derived
   from a single source of truth (the dataclass + a `FIELDS` spec), and the decoder is
   generated from it. One definition produces both the validator and the (de)serializer —
   no drift between writer and reader.
6. **Align definitions with flrCIF where the bundled dictionary has a real match.** Payload
   field names remain Pythonic and stable, but `get_payload_schema()` exposes
   `flrcif_item_id` metadata for matched fields. These IDs must come from the existing
   ChiSurf parameter registry / chinet translation layer and must resolve in the bundled
   mmCIF/flrCIF dictionaries. Do not invent flrCIF item IDs for raw arrays when the
   dictionary has no corresponding item.

## Array Encoding

Numpy arrays are encoded as a small typed sub-map so they round-trip exactly and keep the
envelope stable even though `msgpack_numpy` is installed:

```python
def _encode_array(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"__ndarray__": True, "dtype": str(a.dtype), "shape": list(a.shape), "data": a.tobytes()}

def _decode_array(d: dict) -> np.ndarray:
    return np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"])
```

- `data` is raw little-endian bytes via `tobytes()`; `dtype` is preserved exactly
  (`float64`, `int64`, `uint16`, …). Big arrays stay compact (no base64, no text bloat).
- Use msgpack with `use_bin_type=True` (default in 1.x) so `bytes` survive as `bin`.
- For NaN/inf: float dtypes preserve them natively in `tobytes()`; no special handling.

## The Payload Kinds (formal schemas)

Each kind below gets: a dataclass in `payload_models.py`, a `FIELDS` spec, and a schema
version. Required fields marked **R**, optional **O**. Units live in the envelope `units`
map keyed by field name.

### `fcs_correlation` — FCS / FCCS correlation curves
| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `lag` | float64[n] | R | correlation lag times (s) |
| `correlation` | float64[n] or float64[m, n] | R | G(τ); 2D for multiple curves (ACF/CCF) |
| `error` | float64[…] | O | same shape as `correlation` |
| `curve_names` | list[str] | O | names per curve when 2D (e.g. `["ACF_g","ACF_r","CCF"]`) |
| `weights` | float64[…] | O | fitting weights |
units: `{"lag": "s", "correlation": "1"}`

### `spectra` — absorption / emission spectra
| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `wavelength` | float64[n] | R | (nm) |
| `intensity` | float64[n] | R | |
| `spectrum_type` | str | R | `"absorption"` / `"emission"` / `"excitation"` |
| `normalized` | bool | O | |
units: `{"wavelength": "nm", "intensity": "1"}`

### `tcspc_decay` — TCSPC fluorescence decays
| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `time` | float64[n] | R | nanotime axis (ns) |
| `counts` | int64[n] or float64[n] | R | |
| `irf` | float64[n] | O | instrument response, same axis |
| `channel` | str | O | detection channel id |
| `adc_resolution_ns` | float | O | flrCIF: `_flr_chisurf_parameter.dtTAC[ns]` |
| `micro_time_resolution_ns` | float | O | microtime calibration / resolution; flrCIF: `_flr_chisurf_parameter.dtMT[ns]` |
units: `{"time": "ns", "counts": "1"}`

### `anisotropy_curve` — time-resolved anisotropy / polarized pair
| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `time` | float64[n] | R | (ns) |
| `vv` | float64[n] | R | parallel |
| `vh` | float64[n] | R | perpendicular |
| `l1` | float64[n] | O | anisotropy mixing component 1; flrCIF: `_flr_chisurf_parameter.l1` |
| `l2` | float64[n] | O | anisotropy mixing component 2; flrCIF: `_flr_chisurf_parameter.l2` |
| `g_factor` | float | O | flrCIF: `_flr_chisurf_parameter.g` |
units: `{"time": "ns"}`

### `pda_histogram` — photon-distribution-analysis histograms
| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `edges` | float64[k+1] or list[float64[…]] | R | bin edges (1D or per-axis for 2D) |
| `counts` | float64[…] | R | histogram counts; ndim matches axes |
| `axis_names` | list[str] | O | e.g. `["E","S"]` for E–S 2D |
units: per-axis in `units`

### `burst_table` — per-burst tabular results
| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `columns` | list[str] | R | column names, ordered |
| `dtypes` | list[str] | R | numpy dtype per column |
| `data` | dict[str, ndarray] | R | column → 1D array (columnar, preserves dtype) |
units: per-column in `units` |
> Columnar (not row records) so dtypes and column order survive. `to_dataframe()` /
> `from_dataframe()` helpers convert to/from `pandas.DataFrame` losslessly.

### `burst_selection` — selected burst populations
| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `source_artifact_id` | str | O | source burst table or TTTR artifact |
| `burst_ids` | int64[n] | O | selected burst identifiers |
| `start_indices` | uint64[n] | O | inclusive TTTR/event start index per burst |
| `stop_indices` | uint64[n] | O | exclusive TTTR/event stop index per burst |
| `mask` | bool[n] | O | selection mask over a source burst table |
| `criteria` | dict | O | JSON-safe thresholds, channels, microtime gates, model labels, etc. |
| `labels` | list[str] | O | label per selected burst |
At least one of `burst_ids`, `start_indices`/`stop_indices`, or `mask` is required.

### `tttr_reference` — raw photon streams (preferred transport, NOT re-encoded)
TTTR data is large and vendor files carry reader-critical metadata. Preserve the original
PTU/SPC/BH/Photon-HDF5 file whenever possible. This kind is a **reference** to the raw file
artifact:
| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `source_artifact_id` | str | R | artifact holding the raw file (ptu/spc/bh/photon_hdf5) |
| `vendor_format` | str | R | `"ptu"`/`"spc"`/`"bh"`/`"photon_hdf5"` |
| `n_records` | int | O | |
| `routing_channels` | list[int] | O | |
| `macro_time_resolution_s` | float | O | |
| `micro_time_resolution_s` | float | O | |
| `header_json` | dict/str | O | raw `tttrlib` header JSON when available |
| `tags` | dict | O | vendor/header tags exposed by the reader |
The raw file is registered as a normal `raw_measurement` artifact (its own format label);
`tttr_reference` carries the structured header so the stream is queryable without parsing.

### `tttr_photon_stream` — optional open msgpack TTTR derivative
Use this only when a reader explicitly normalizes a vendor file into an open ChiSurf
msgpack stream. It is a derivative of the raw file, not a replacement for preserving the
original file.
| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `macro_times` | uint64[n] | R | macro time ticks |
| `micro_times` | uint32[n] | R | micro time channels |
| `routing_channels` | uint16[n] or uint32[n] | R | detector/routing channel per record |
| `event_types` | uint8[n] / uint16[n] / uint32[n] | O | marker/overflow/photon event type |
| `macro_time_resolution_s` | float | R | seconds per macro tick |
| `micro_time_resolution_s` | float | R | seconds per micro channel |
| `source_artifact_id` | str | O | raw vendor artifact this was derived from |
| `vendor_format` | str | O | original format, e.g. `"ptu"` |
| `record_type` | str | O | tttrlib/PicoQuant record type when known |
| `container_type` | str | O | tttrlib container type when known |
| `header_json` | dict/str | O | full reader header JSON |
| `tags` | dict | O | vendor tags, e.g. PTU tags as emitted by reader |
All event arrays must have the same length. Header/tag maps must be JSON-safe so they can
carry PTU metadata without hidden Python objects.

### `generic_table` / `generic_curve` — fallbacks
For payloads that don't fit a specific kind. `generic_curve` = `{x, y, ex?, ey?, mask?}`
(mirrors `chisurf.core.data.DataCurve`); `generic_table` = same shape as `burst_table`.
These give plugins a typed path before a bespoke kind exists, instead of raw JSON.

## Tasks

### Task 1: Add the `msgpack` data format + dependencies

- Add `"msgpack"` to `DATA_FORMATS` in `chisurf/core/mfdb/models.py`.
- Confirm `msgpack` and `msgpack-numpy` are declared runtime dependencies.
- Install both in the `arm64` conda environment with:
  ```bash
  /Users/tpeulen/mambaforge/bin/mamba install -n arm64 -c conda-forge msgpack-python msgpack-numpy
  ```

### Task 2: Payload models (single source of truth)

**File**: `chisurf/core/mfdb/payload_models.py`

Define one `@dataclass` per kind above (`FcsCorrelation`, `Spectrum`, `TcspcDecay`,
`AnisotropyCurve`, `PdaHistogram`, `BurstTable`, `BurstSelection`, `TttrReference`,
`TttrPhotonStream`, `GenericCurve`, `GenericTable`). Each declares:

```python
@dataclass
class FcsCorrelation:
    SCHEMA_VERSION = 1
    KIND = "fcs_correlation"
    # field -> (dtype-or-type, required, unit)
    FIELDS = {
        "lag":         ("f8[]", True,  "s"),
        "correlation": ("f8[]", True,  "1"),
        "error":       ("f8[]", False, "1"),
        "curve_names": ("str[]", False, None),
        "weights":     ("f8[]", False, "1"),
    }
    lag: np.ndarray
    correlation: np.ndarray
    error: Optional[np.ndarray] = None
    curve_names: Optional[list[str]] = None
    weights: Optional[np.ndarray] = None
```

The `FIELDS` spec is the schema. A tiny type mini-language (`"f8[]"` = float64 array,
`"i8[]"`, `"str[]"`, `"f8"` scalar, `"str"`, `"bool"`, `"int"`) keeps it declarative. The
codec and validator are **generated** from `FIELDS`, so reader and writer can never drift.
Models may also declare `FLRCIF_ITEMS = {"field": "_category.item"}` for fields that have
an exact bundled flrCIF / ChiSurf-extension dictionary mapping. Tests must verify those
IDs through the existing registry translation and `MmcifDictionary.load_bundled()`.

### Task 3: The codec

**File**: `chisurf/core/mfdb/payload_codec.py`

```python
def encode_payload(kind: str, obj: Any, *, meta: dict | None = None) -> tuple[bytes, str]:
    """Validate `obj` against `kind`'s schema and return (msgpack_bytes, "msgpack")."""

def decode_payload(blob: bytes) -> Any:
    """Read the envelope, dispatch on `kind`+`schema`, validate, return the dataclass."""

def get_payload_schema(kind: str) -> dict:
    """Return the formal schema (FIELDS + units + versions) for a kind."""

class PayloadSchemaError(ValueError): ...
```

Requirements:
- Build the envelope (`v/kind/schema/units/meta/body`) in `encode_payload`.
- Encode arrays via `_encode_array`; pack with `msgpack.packb(..., use_bin_type=True)`.
- `decode_payload` unpacks with `msgpack.unpackb(..., raw=False)`, checks `v`, looks up the
  model by `kind`, **migrates** older `schema` versions via per-kind upgraders, validates,
  and reconstructs arrays via `_decode_array`.
- Validation: every **R** field present, dtype/shape match the `FIELDS` mini-language,
  no unexpected top-level body keys. Raise `PayloadSchemaError` with a precise message.
- A central `REGISTRY: dict[str, type]` maps `kind -> dataclass`. New kinds register here.

### Task 4: pandas / DataCurve interop

In `payload_models.py`:
- `BurstTable.from_dataframe(df)` / `.to_dataframe()` — columnar, dtype-preserving.
- `GenericCurve.from_data_curve(dc)` / `.to_data_curve()` — bridge to
  `chisurf.core.data.DataCurve` (`x, y, ex, ey, mask`). This replaces `DataCurve.to_dict()`
  JSON round-tripping for archival.

### Task 5: Migration shims for the existing idioms

Provide adapters so existing typed storage maps onto codecs without a data migration:
- `analysis_data` (x/y float64 blobs + units) ↔ `GenericCurve` / `tcspc_decay`.
- `add_spectrum` (float64 blobs + units) ↔ `Spectrum`.
Document that **new** writes go through codecs; old rows remain readable via adapters.

### Task 6: Tests

**File**: `test/fio/test_payload_codec.py`

- Round-trip each kind: build dataclass → `encode_payload` → `decode_payload` →
  assert arrays equal (`np.testing.assert_array_equal`), dtypes identical, units preserved.
- NaN/inf survive a float round-trip.
- 2D `correlation` (multi-curve) and 2D `pda_histogram` round-trip with shape intact.
- `BurstTable` ↔ DataFrame preserves column order and per-column dtypes.
- Missing required field → `PayloadSchemaError`.
- Wrong dtype/shape → `PayloadSchemaError`.
- Unknown `kind` in `decode_payload` → `PayloadSchemaError`.
- Schema-version migration: a v1 blob decodes after a v2 model is introduced (stub an
  upgrader to prove the path).
- Blob size sanity: a 10k-point float64 FCS curve encodes to roughly `n*8` bytes + small
  overhead (proves no JSON bloat).
- flrCIF alignment: every emitted `flrcif_item_id` resolves in the bundled dictionary and
  matches the existing ChiSurf parameter translation for known fields (`g`, `l1`, `l2`,
  `dtTAC[ns]`, `dtMT[ns]`).

## Definition of Done

- [ ] `"msgpack"` added to `DATA_FORMATS`; `msgpack` and `msgpack-numpy` declared as
      runtime dependencies and installed in the `arm64` conda environment
- [ ] `payload_models.py` defines one dataclass per kind with a declarative `FIELDS` schema
      (single source of truth for validator + codec)
- [ ] `payload_codec.py` exposes `encode_payload`, `decode_payload`, `get_payload_schema`,
      `PayloadSchemaError`, and a `kind -> model` `REGISTRY`
- [ ] Self-describing msgpack envelope (`v/kind/schema/units/meta/body`); arrays preserve
      dtype/shape/NaN/inf
- [ ] TTTR raw files are preserved by default; `tttr_reference` stores reader metadata and
      `tttr_photon_stream` covers optional open msgpack derivatives
- [ ] pandas and `DataCurve` interop helpers exist and round-trip losslessly
- [ ] Adapters map existing `analysis_data` / `add_spectrum` rows onto codec types
- [ ] Per-kind schema versioning with a working migration path
- [ ] Matched fields expose `flrcif_item_id` schema metadata verified against the bundled
      dictionary and existing ChiSurf parameter translation
- [ ] All tests in `test/fio/test_payload_codec.py` pass
- [ ] PRD-03 updated to consume these codecs (its `_store_data`/read path use
      `encode_payload`/`decode_payload`, not `to_json`/`json.dumps`)

## Boundary / Non-Goals

- **Not** a new storage backend — payloads still go into the existing content-addressed
  object store via `MFDatabase.put_object`. This PRD defines only the **bytes** that go in.
- **Not** a mandatory TTTR re-encoding effort — raw photon streams stay in vendor formats
  unless a reader explicitly creates an open `tttr_photon_stream` derivative.
- **Not** sample/provenance modeling — that's PRD-02 / PRD-03.

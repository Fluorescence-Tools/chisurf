# Instruction: Integrate tttrlib.BurstFilter

## Goal

Replace the current ChiSurf mask-based burst detection
(`chisurf/core/fluorescence/burst/burst.py` and `count_rate.py`) with
`tttrlib.BurstFilter` behind the same API, preserving fully identical output.

## Background

The current `api/selection.py` wraps existing ChiSurf functions:

- `apply_photon_filters` → `burst_filter()` + `count_rate_filter()` in core
- `find_bursts` → core mask-based start/stop detection
- `summarize_bursts` → `generate_burst_dataframe()` in core

`tttrlib.BurstFilter` (at `/Users/tpeulen/dev/tttrlib/include/BurstFilter.h`)
is a C++ implementation designed to be the canonical burst engine. Switching to
it would give performance improvements and a single code path.

**Constraint:** The switch must be transparent. Every API function must produce
identical output for identical input. This requires a comprehensive parity test
suite first.

## Prerequisites

- `INSTRUCTION_GUI_VERIFICATION.md` is complete and the migrated GUI is stable.
- All 29+ Burst Selection tests pass.
- `tttrlib` is installed and `BurstFilter` is accessible.

## Files to study

```bash
# Current implementation:
cat chisurf/plugins/burst/burst_selection/api/selection.py
cat chisurf/plugins/burst/burst_selection/api/features.py
cat chisurf/plugins/burst/burst_selection/api/io.py

# Core implementations being replaced:
cat chisurf/core/fluorescence/burst/burst.py
cat chisurf/core/fluorescence/burst/count_rate.py
cat chisurf/core/fio/fluorescence/burst.py

# tttrlib BurstFilter header:
cat /Users/tpeulen/dev/tttrlib/include/BurstFilter.h

# tttrlib Python bindings — check what's available:
python3 -c "import tttrlib; help(tttrlib.BurstFilter)" 2>&1 | head -60
```

## Tasks

### 1. Add a parity test fixture

Create `tests/test_burstfilter_parity.py` with tests that:

- Load `m000.spc`
- Apply identical settings through both the API (current path) and
  `tttrlib.BurstFilter`
- Compare: photon mask, burst start/stop indices, burst DataFrame

Use `pytest.mark.parametrize` for multiple channel/filter configurations.

```python
@pytest.fixture
def bh_spc_data():
    """Load BH SPC132 test data once."""
    ...

def test_burstfilter_mask_parity(bh_spc_data):
    """Photon selection mask matches between API and BurstFilter."""
    ...

def test_burstfilter_start_stop_parity(bh_spc_data):
    """Burst start/stop indices match."""
    ...

def test_burstfilter_dataframe_parity(bh_spc_data):
    """Burst summary DataFrame matches."""
    ...
```

### 2. Create an optional internal backend switch

In `api/selection.py`, add:

```python
_BURST_ENGINE: str = "chisurf"  # or "tttrlib"

def set_burst_engine(engine: str) -> None:
    """Switch between 'chisurf' (default) and 'tttrlib' burst engines."""
    global _BURST_ENGINE
    assert engine in ("chisurf", "tttrlib")
    _BURST_ENGINE = engine
```

### 3. Implement the tttrlib path

In `api/selection.py`, branch on `_BURST_ENGINE`:

- `apply_photon_filters`: delegate to `tttrlib.BurstFilter.apply(...)` or
  equivalent
- `find_bursts`: use `tttrlib.BurstFilter.get_burst_starts_stops()`
- `summarize_bursts`: needs careful column-by-column comparison

**Do NOT** switch the default. Leave `_BURST_ENGINE = "chisurf"` until parity
tests pass 100%.

### 4. Run parity tests

```bash
/Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest \
    chisurf/plugins/burst/burst_selection/tests/test_burstfilter_parity.py \
    -q --no-cov --tb=short
```

Fix any discrepancies by adjusting the `tttrlib` path.

### 5. Flip the default

Once parity tests pass for all parametrized configurations:

```python
_BURST_ENGINE: str = "tttrlib"
```

Then run the full test suite:

```bash
/Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest \
    chisurf/plugins/burst/burst_selection/tests -q --no-cov
```

### 6. Clean up

- Remove or deprecate direct core function imports from `chisurf.core.*` that
  are now routed through `tttrlib.BurstFilter`.
- Keep the parity tests as regression tests (run both engines always).
- Update `STATUS.md`.

## Success criteria

- All existing tests pass with `_BURST_ENGINE = "tttrlib"`.
- No change in `.bur` output for any input.
- Parity tests prove identity for ≥3 different filter configurations.
- The legacy engine remains available as a fallback.

## Risks

- `tttrlib.BurstFilter` API might not expose all the features used by the
  current code (e.g., detector-specific windows, microtime ranges). In that
  case, implement a hybrid: use BurstFilter for basic burst detection, keep
  ChiSurf code for per-detector/per-window summary.
- Edge cases in zero-row interleaving (`.bur` vs HDF5). Compare both paths.

# PRD-06: Expand Fluorophore Database

## Goal

MFDB ships with real spectral data for common smFRET dyes. Users can look up a
donor-acceptor pair and get the Forster radius automatically computed from the spectral
overlap integral.

## Background

Read these files before starting:
- `chisurf/core/mfdb/seed_data.py` -- current seed data (7 probes, 3-point placeholder spectra)
- `chisurf/core/mfdb/schema.py` -- search for `probes`, `optical_properties`, `spectra`, `flr_fret_forster_radius`
- `chisurf/core/mfdb/repository.py` -- search for probe/spectra methods
- `chisurf/core/models/tcspc/fret.py` -- search for `R0`, `FRETParameters`

## Current State

Two parallel realities exist:

**1. The shipped seed (`chisurf/core/mfdb/seed_data.py`)** — 7 probes with placeholder
spectra (3 wavelength points each): Alexa488, Alexa594, Cy3, Cy5, ATTO647N, Trp,
2-aminopurine. The `spectra` table stores wavelength/intensity arrays as BLOBs; the
`flr_fret_forster_radius` table stores R0 for donor-acceptor pairs. Neither table is
populated with real data in the shipped DB.

**2. Two un-integrated `_dev` plugins with a large scraped dataset (the real story).**
A working fluorophore-curation engine and a scraper already exist under
`chisurf/plugins/_dev/` and are NOT wired into the app:

- **`_dev/fluorophore_db/`** — the curation engine. `mfdb_adapter.py`
  (`FluorophoreDatabase(MFDatabase)`) already speaks the **canonical MFDB schema**
  (`probes`/`optical_properties`/`spectra`/`probe_types`/`images`), with `add_probe`,
  `add_spectrum`, `add_optical_property`, `get_probe_full`, `search_probes`,
  `validate_probe`, and `get_standardized_items(include_uncurated=…)`. A curation GUI
  (`db_manager_widget.py`, `editor.py`) browses/plots spectra and toggles
  `is_curated`/`quality_flag`.
- **`_dev/spectra_downloader/`** — the scraper. `download_manager.py` runs per-source
  importer scripts (`download/`: `fpbase.py`, `atto.py`, `chroma.py`, `omega_optical.py`,
  `thorlabs.py`, `photochemcad_common_compounds.py`, `import_qfe_spectra.py`, …) as
  subprocesses against a `--db` path, writing into a `FluorophoreDatabase`.
- **`_dev/fluorophore_db/spectra.db`** — already a full canonical-schema MFDB carrying
  **~1,954 probes, ~3,226 spectra, ~15,456 optical-property rows** scraped from those
  sources.

**Why this is "not properly integrated" (the problems this PRD now owns):**

1. **Separate store.** The scraped data lives in a plugin-local `spectra.db`, not the
   user's working MFDB. The curation GUI and scraper both point only at that file.
2. **In `_dev/`.** Neither plugin is discoverable/shipped.
3. **Unverified, low quality.** Only **~35 of 1,954** probes are marked
   `is_curated=1`; `quality_flag` is uniformly the default `1` (never graded);
   `category` is mostly `other`/NULL. The data is a useful **initial reference set**
   but must be treated as **unverified / provisional until reviewed and approved**.
4. **No source provenance.** The `probes` table has no `source` column, so an entry
   can't say it came from FPbase vs ATTO vs user entry, nor carry a source URL /
   retrieval date.
5. **Downstream not wired.** `forster.py` (Task 1, done), the seed, and
   `lookup_forster_radius` don't consume this curated data.

This PRD now treats those two tools — and the verification/approval workflow that
makes their scraped data trustworthy — as first-class deliverables (Tasks 7–9).

## Tasks

### Task 1: Create Forster Radius Calculator

**File to create**: `chisurf/core/fluorescence/fret/forster.py`

```python
"""Forster radius computation from spectral overlap.

R0 = 0.02108 * (kappa2 * QD * n^-4 * J)^(1/6)  [in Angstrom]

where:
    kappa2 = orientation factor (typically 2/3 for freely rotating dyes)
    QD = donor quantum yield
    n = refractive index of medium (typically 1.33 for water)
    J = spectral overlap integral [M^-1 cm^-1 nm^4]
"""
import numpy as np
from typing import Tuple


def overlap_integral(
    donor_emission_wavelengths: np.ndarray,
    donor_emission_intensity: np.ndarray,
    acceptor_absorption_wavelengths: np.ndarray,
    acceptor_extinction_coefficient: np.ndarray,
) -> float:
    """Compute the spectral overlap integral J(lambda).

    J = integral[ F_D(lambda) * epsilon_A(lambda) * lambda^4 ] d(lambda)
        / integral[ F_D(lambda) ] d(lambda)

    where:
        F_D(lambda) = normalized donor emission spectrum
        epsilon_A(lambda) = acceptor molar extinction coefficient [M^-1 cm^-1]
        lambda = wavelength [nm]

    Args:
        donor_emission_wavelengths: Wavelengths in nm for donor emission.
        donor_emission_intensity: Donor emission intensity (arbitrary units).
        acceptor_absorption_wavelengths: Wavelengths in nm for acceptor absorption.
        acceptor_extinction_coefficient: Acceptor extinction coefficient in M^-1 cm^-1.

    Returns:
        J in units of M^-1 cm^-1 nm^4.
    """
    # Interpolate both spectra onto a common wavelength grid (1 nm spacing)
    wl_min = max(donor_emission_wavelengths[0], acceptor_absorption_wavelengths[0])
    wl_max = min(donor_emission_wavelengths[-1], acceptor_absorption_wavelengths[-1])

    if wl_min >= wl_max:
        return 0.0

    common_wl = np.arange(wl_min, wl_max + 1, 1.0)

    fd = np.interp(common_wl, donor_emission_wavelengths, donor_emission_intensity)
    ea = np.interp(common_wl, acceptor_absorption_wavelengths, acceptor_extinction_coefficient)

    # Normalize donor emission
    fd_area = np.trapz(fd, common_wl)
    if fd_area <= 0:
        return 0.0
    fd_norm = fd / fd_area

    # Compute overlap integral
    integrand = fd_norm * ea * common_wl**4
    J = np.trapz(integrand, common_wl)

    return J


def forster_radius(
    donor_quantum_yield: float,
    overlap_integral_J: float,
    kappa2: float = 2.0 / 3.0,
    refractive_index: float = 1.33,
) -> float:
    """Compute Forster radius R0 in Angstrom.

    R0 = 0.02108 * (kappa2 * QD * n^-4 * J)^(1/6)  [Angstrom]

    The constant 0.02108 gives R0 in Angstrom when J is in M^-1 cm^-1 nm^4.

    Args:
        donor_quantum_yield: Donor quantum yield (0 to 1).
        overlap_integral_J: Spectral overlap integral in M^-1 cm^-1 nm^4.
        kappa2: Orientation factor (default 2/3).
        refractive_index: Refractive index of medium (default 1.33 for water).

    Returns:
        R0 in Angstrom.
    """
    if overlap_integral_J <= 0 or donor_quantum_yield <= 0:
        return 0.0

    R0_6 = 0.02108**6 * kappa2 * donor_quantum_yield * refractive_index**(-4) * overlap_integral_J
    # Actually the standard formula:
    # R0 = 0.02108 * (kappa2 * QD * n^-4 * J)^(1/6)
    # where the constant absorbs Avogadro's number and the 9*ln(10)/(128*pi^5) factor
    # More precisely:
    # R0^6 = (9 * Q_D * ln(10) * kappa^2 * J) / (128 * pi^5 * n^4 * N_A)
    # where N_A = 6.022e23
    #
    # Using SI-ish units with J in M^-1 cm^-1 nm^4 and R0 in nm:
    # R0 = (8.79e-25 * kappa2 * QD * J / n^4)^(1/6) * 1e7  [nm -> Angstrom multiply by 10]

    # Standard constant: 8.79e-25 mol (when J in M^-1 cm^-1 nm^4, R0 in cm)
    R0_6_cm = 8.79e-25 * kappa2 * donor_quantum_yield * overlap_integral_J / (refractive_index**4)
    R0_cm = R0_6_cm ** (1.0 / 6.0)
    R0_angstrom = R0_cm * 1e8  # cm to Angstrom

    return R0_angstrom


def compute_forster_radius_from_spectra(
    donor_emission_wl: np.ndarray,
    donor_emission_intensity: np.ndarray,
    acceptor_absorption_wl: np.ndarray,
    acceptor_extinction_coeff: np.ndarray,
    donor_quantum_yield: float,
    kappa2: float = 2.0 / 3.0,
    refractive_index: float = 1.33,
) -> Tuple[float, float]:
    """Convenience function: compute R0 from raw spectral data.

    Returns:
        (R0_angstrom, J_overlap): Forster radius in Angstrom and overlap integral.
    """
    J = overlap_integral(
        donor_emission_wl, donor_emission_intensity,
        acceptor_absorption_wl, acceptor_extinction_coeff,
    )
    R0 = forster_radius(donor_quantum_yield, J, kappa2, refractive_index)
    return R0, J
```

**Important**: Verify the constant. The standard Forster radius formula is:

R0^6 = (9 * Q_D * ln(10) * kappa^2) / (128 * pi^5 * N_A * n^4) * J

With J in M^-1 cm^-1 nm^4 and R0 in cm, the prefactor is 8.79e-25 mol.
Multiply R0 in cm by 1e8 to get Angstrom.

Write a test that computes R0 for a well-known pair (e.g., Cy3-Cy5, expected ~54 A)
to validate the constant.

### Task 2: Add Real Spectral Data

**File to create**: `chisurf/core/mfdb/spectral_data/` (directory)

Create a Python module with embedded spectral data for common dyes. Each dye gets
absorption and emission spectra as numpy arrays.

**File to create**: `chisurf/core/mfdb/spectral_data/__init__.py`

```python
"""Curated spectral data for common fluorescent probes.

Data sources: published literature values and manufacturer specifications.
All spectra are normalized (peak = 1.0) with 1 nm wavelength spacing.
Extinction coefficients are in M^-1 cm^-1.
"""

# Each entry: {
#   "name": str,
#   "type": "organic_dye" | "amino_acid" | "nucleic_acid",
#   "abs_max_nm": float,
#   "em_max_nm": float,
#   "quantum_yield": float,
#   "ext_coeff": float,  # M^-1 cm^-1 at abs_max
#   "absorption": {"wavelengths": [...], "intensity": [...]},
#   "emission": {"wavelengths": [...], "intensity": [...]},
# }

PROBES = {}
```

Then create one file per dye family:

**File to create**: `chisurf/core/mfdb/spectral_data/alexa.py`

```python
"""Alexa Fluor dye spectral data."""

ALEXA_488 = {
    "name": "Alexa Fluor 488",
    "type": "organic_dye",
    "abs_max_nm": 495,
    "em_max_nm": 519,
    "quantum_yield": 0.92,
    "ext_coeff": 73000,
    "absorption": {
        "wavelengths": [350, 360, 370, ...],  # 1nm spacing, 350-650nm
        "intensity": [0.01, 0.012, ...],       # normalized, peak = 1.0
    },
    "emission": {
        "wavelengths": [480, 481, 482, ...],
        "intensity": [0.05, 0.06, ...],
    },
}

# ... more Alexa dyes ...
```

**Where to get the data**: The actual spectral data arrays need to be sourced from:
1. Published literature
2. FPbase (fluorophore database)
3. Manufacturer data sheets

For now, generate approximate Gaussian spectra based on known abs_max, em_max, and
spectral widths. This is better than 3-point placeholders:

```python
import numpy as np

def gaussian_spectrum(center_nm, fwhm_nm, wl_range=(300, 800)):
    """Generate a Gaussian approximation of a spectrum."""
    wl = np.arange(wl_range[0], wl_range[1] + 1, 1.0)
    sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    intensity = np.exp(-0.5 * ((wl - center_nm) / sigma) ** 2)
    return wl.tolist(), intensity.tolist()
```

Typical FWHM values:
- Alexa dyes: absorption FWHM ~25-35 nm, emission FWHM ~30-45 nm
- Cy dyes: absorption FWHM ~20-30 nm, emission FWHM ~25-40 nm
- ATTO dyes: absorption FWHM ~20-30 nm, emission FWHM ~25-35 nm

### Task 3: Update Seed Data

**File**: `chisurf/core/mfdb/seed_data.py`

Replace the 3-point placeholder spectra with the real/Gaussian spectra from Task 2.
Add more probes. Target: at least 20 common dyes.

Minimum set for smFRET:

| Dye | abs_max | em_max | QY | ext_coeff | FWHM_abs | FWHM_em |
|-----|---------|--------|-----|-----------|----------|---------|
| Alexa 488 | 495 | 519 | 0.92 | 73000 | 28 | 37 |
| Alexa 546 | 556 | 573 | 0.79 | 104000 | 25 | 33 |
| Alexa 555 | 555 | 565 | 0.10 | 155000 | 27 | 38 |
| Alexa 568 | 578 | 603 | 0.69 | 91300 | 28 | 40 |
| Alexa 594 | 590 | 617 | 0.66 | 87000 | 28 | 40 |
| Alexa 647 | 650 | 665 | 0.33 | 270000 | 28 | 35 |
| Cy3 | 550 | 570 | 0.15 | 150000 | 25 | 35 |
| Cy3B | 558 | 572 | 0.67 | 130000 | 25 | 33 |
| Cy5 | 649 | 670 | 0.27 | 250000 | 25 | 35 |
| Cy5.5 | 673 | 707 | 0.23 | 209000 | 25 | 40 |
| ATTO 488 | 501 | 523 | 0.80 | 90000 | 22 | 30 |
| ATTO 532 | 532 | 553 | 0.90 | 115000 | 23 | 32 |
| ATTO 550 | 554 | 576 | 0.80 | 120000 | 23 | 32 |
| ATTO 565 | 563 | 592 | 0.90 | 120000 | 23 | 35 |
| ATTO 590 | 594 | 624 | 0.80 | 120000 | 24 | 36 |
| ATTO 594 | 601 | 627 | 0.85 | 120000 | 25 | 35 |
| ATTO 647N | 644 | 669 | 0.65 | 150000 | 22 | 30 |
| ATTO 655 | 663 | 684 | 0.30 | 125000 | 22 | 28 |
| ATTO 680 | 680 | 700 | 0.30 | 125000 | 23 | 30 |
| Rhodamine 110 | 496 | 520 | 0.97 | 80000 | 26 | 35 |

### Task 4: Auto-Compute Forster Radii for Common Pairs

**File**: `chisurf/core/mfdb/seed_data.py`

After seeding probes with spectra, compute R0 for common donor-acceptor pairs and store
in `flr_fret_forster_radius` table.

Common smFRET pairs to precompute:

| Donor | Acceptor | Expected R0 (A) |
|-------|----------|-----------------|
| Alexa 488 | Alexa 594 | ~54 |
| Alexa 488 | Alexa 647 | ~52 |
| Cy3 | Cy5 | ~54 |
| Cy3B | ATTO 647N | ~62 |
| ATTO 488 | ATTO 647N | ~58 |
| Alexa 555 | Alexa 647 | ~51 |
| ATTO 532 | ATTO 647N | ~59 |

```python
from chisurf.core.fluorescence.fret.forster import compute_forster_radius_from_spectra

# After seeding probes and spectra:
pairs = [
    ("Alexa Fluor 488", "Alexa Fluor 594"),
    ("Alexa Fluor 488", "Alexa Fluor 647"),
    ("Cy3", "Cy5"),
    # ... etc
]

for donor_name, acceptor_name in pairs:
    donor = get_probe_with_spectra(db, donor_name)
    acceptor = get_probe_with_spectra(db, acceptor_name)

    R0, J = compute_forster_radius_from_spectra(
        donor_emission_wl=np.array(donor["emission"]["wavelengths"]),
        donor_emission_intensity=np.array(donor["emission"]["intensity"]),
        acceptor_absorption_wl=np.array(acceptor["absorption"]["wavelengths"]),
        acceptor_extinction_coeff=np.array(acceptor["absorption"]["intensity"]) * acceptor["ext_coeff"],
        donor_quantum_yield=donor["quantum_yield"],
    )

    db.con.execute(
        """INSERT OR REPLACE INTO flr_fret_forster_radius
           (donor_probe_id, acceptor_probe_id, forster_radius, kappa_squared,
            index_of_refraction, overlap_integral)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (donor["probe_id"], acceptor["probe_id"], R0, 2.0/3.0, 1.33, J)
    )
```

### Task 5: R0 Lookup in FRET Model

When a user selects a donor-acceptor pair (via the sample), the FRET model should
auto-populate R0 from the database. **However, R0 may be "god given"** -- the user
enters it directly from literature without spectral data. The lookup is a convenience,
not a requirement.

**File**: `chisurf/core/models/tcspc/fret.py`

Add a class method or utility function:

```python
def lookup_forster_radius(donor_name: str, acceptor_name: str, db=None) -> float:
    """Look up R0 for a donor-acceptor pair from MFDB.

    Returns R0 in Angstrom, or 0.0 if not found.

    This is a convenience lookup. Users may also enter R0 directly without
    spectral data (literature values, prior experiments). In that case, R0
    is stored as a calibration artifact with method="user_provided" via
    register_calibration() -- see PRD-05 Task 2b.
    """
    if db is None:
        from chisurf.core.mfdb.result_registry import _get_global_db
        db = _get_global_db()
    if db is None:
        return 0.0

    row = db.con.execute(
        """SELECT fr.forster_radius
           FROM flr_fret_forster_radius fr
           JOIN probes d ON d.probe_id = fr.donor_probe_id
           JOIN probes a ON a.probe_id = fr.acceptor_probe_id
           WHERE d.name = ? AND a.name = ?""",
        (donor_name, acceptor_name)
    ).fetchone()

    return row[0] if row else 0.0
```

**GUI integration**: When the FRET model widget loads, try `lookup_forster_radius()`.
If a value is found, pre-fill the R0 field. If not found, leave R0 empty for the user
to enter manually. Either way, the user can override the value. When the fit is archived
(PRD-05), the actual R0 used is stored as a calibration -- either with
`method="spectral_overlap"` (if from lookup) or `method="user_provided"` (if entered
manually).

### Task 6: Write Tests

**File to create**: `test/fluorescence/test_forster_radius.py`

```python
"""Tests for Forster radius computation."""
import numpy as np
import pytest
from chisurf.core.fluorescence.fret.forster import (
    overlap_integral,
    forster_radius,
    compute_forster_radius_from_spectra,
)


def _gaussian(center, fwhm, wl):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * ((wl - center) / sigma) ** 2)


def test_overlap_integral_nonoverlapping_is_zero():
    """Non-overlapping spectra should give J = 0."""
    wl_d = np.arange(400, 500)
    em_d = _gaussian(450, 30, wl_d)
    wl_a = np.arange(700, 800)
    abs_a = _gaussian(750, 30, wl_a) * 100000
    J = overlap_integral(wl_d, em_d, wl_a, abs_a)
    assert J == 0.0


def test_overlap_integral_positive_for_overlapping():
    """Overlapping spectra should give J > 0."""
    wl = np.arange(400, 700)
    em_d = _gaussian(520, 35, wl)
    abs_a = _gaussian(590, 30, wl) * 87000  # ext_coeff scaled
    J = overlap_integral(wl, em_d, wl, abs_a)
    assert J > 0


def test_forster_radius_known_pair():
    """Approximate check: Alexa488-Alexa594 should give R0 ~ 50-60 A."""
    wl = np.arange(400, 750)
    # Alexa 488 emission
    em_d = _gaussian(519, 37, wl)
    # Alexa 594 absorption (extinction coefficient)
    abs_a = _gaussian(590, 28, wl) * 87000

    R0, J = compute_forster_radius_from_spectra(
        donor_emission_wl=wl,
        donor_emission_intensity=em_d,
        acceptor_absorption_wl=wl,
        acceptor_extinction_coeff=abs_a,
        donor_quantum_yield=0.92,
    )

    # R0 should be roughly 50-60 A for this pair
    assert 40 < R0 < 70, f"R0 = {R0:.1f} A, expected ~54 A"


def test_forster_radius_zero_quantum_yield():
    """Zero QY should give R0 = 0."""
    R0 = forster_radius(0.0, 1e15)
    assert R0 == 0.0
```

## Task 7: Integrate the Fluorophore-DB + Spectra-Downloader tools with MFDB

The two `_dev` tools (see Current State) are the real-data engine. They supersede the
embedded-Gaussian approach of Task 2 — the data is sourced from authoritative
databases, not hand-rolled. The work is **integration + curation**, not building from
scratch. Subtasks:

**7.1 Promote both plugins out of `_dev/`.** Move `fluorophore_db` and
`spectra_downloader` to shipped plugins (manifest-discovered). Mark them
**experimental** initially (see the experimental-tool mechanism) since the data is
unverified.

**7.2 Make MFDB the single store (no parallel `spectra.db`).** The plugin already
subclasses `MFDatabase` on the canonical schema, so the gap is *which database* it
opens. Point the curation GUI and the scraper at the active MFDB resolved through
`ChiSurfAPI`/`resolve_database_path` (the configured DB, or an explicit `--db`),
instead of the hardcoded plugin-local `spectra.db`. Keep `spectra.db` only as an
*import source* — a one-time `import_reference_set` that copies its
probes/spectra/optical-properties into the target MFDB (idempotent, dedup by
name+source), stamping every imported row as unverified (Task 8). No second
source-of-truth.

**7.3 Add `source` provenance.** Add a `source` column to `probes` (e.g. `fpbase`,
`atto`, `chroma`, `photochemcad`, `qfe`, `thorlabs`, `omega`, `user`, `literature`)
plus `source_ref` (URL/accession) and `retrieved_at`. Follow the **`.dic`-driven
schema rule** (PRD-19): if `probes` becomes a `.dic`-declared table, add these via the
dictionary + `reconcile_schema`; if it stays hand-DDL in `schema.py`, add the columns
there and in the adapter's INSERT/UPDATE. Each scraper records its own `source` +
`source_ref` when it writes a row.

**7.4 Wire downstream consumers to *approved* data only.** `lookup_forster_radius`
(Task 5), the seed (Task 3/4 precompute), the Light Path Simulator crosstalk/R₀
(PRD-08 Task 8), and the PRD-05 computed-from-spectra calibration must read
**only verification_status=approved** probes by default (Task 8), so unverified
scraped rows never silently feed a published R₀.

**7.5 Headless path.** Importer + reconcile + approval transitions must be runnable
without the GUI (`csc fluorophore import-reference-set`, `… approve <probe>`,
`… list --status unverified`), per the project's headless-feature rule. The
curation GUI is thin wiring over these.

Keep the importers/curation pure-Python (network + spectra parsing); MFDB read/write
goes through the plugin backend services mirroring the repository methods.

## Task 8: Verification / approval / quality workflow (provenance)

The scraped 1,954 entries are an **initial reference set of unverified, low-quality
data** — usable as a starting point, but nothing downstream should trust them until a
human approves. Model this explicitly rather than via the current binary
`is_curated` flag.

**8.1 Verification + quality vocabulary.** Introduce:
- `verification_status` ∈ {`unverified`, `under_review`, `approved`, `rejected`,
  `superseded`} (sourced from the `.dic` enumeration per PRD-19 — never a hand Python
  list). `approved` ⟺ the existing `is_curated=1`; keep `is_curated` as a generated
  mirror for back-compat or migrate callers.
- `quality` grade ∈ {`unknown`, `low`, `medium`, `high`} (replaces the meaningless
  uniform `quality_flag`).
- `verified_by` / `verified_at` (who approved, when) — provenance, reusing the PRD-17
  identity resolver.

**8.2 On import, everything enters `unverified` / `quality=unknown`.** The reference
set from `spectra.db` and every fresh scrape land provisional. This is the concrete
"unverified, low quality till updated and approved" requirement.

**8.3 Approval transitions.** `approve` / `reject` / `mark_under_review` /
`set_quality` API + CLI, validated and recorded. Consider modeling these as the
PRD-12 lifecycle state machine (states + transitions already `.dic`-declared) and the
approval action as a PRD-21 provenance operation, so curation history is queryable —
but a small set of status columns + an audit row is an acceptable minimum if the
lifecycle wiring is heavier than warranted.

**8.4 Downstream filter.** Default every consumer (Task 7.4) to approved-only; allow
an explicit `include_unverified=True` escape hatch for power users.

## Task 9: AI-assisted triage and curation (human-in-the-loop)

1,954 mostly-unverified entries are too many to hand-curate cold. Add an **AI triage
pass that proposes** category, quality grade, canonical name, duplicate clusters, and
an approve/flag recommendation **with rationale** — the human (or a downstream rule)
makes the final `approve`/`reject` call (Task 8). **AI proposes, human disposes** —
the AI must never auto-set `verification_status=approved`; it writes proposals to a
review queue.

**What the AI does per probe (or duplicate cluster):**
- **Classify** `category` (organic_dye / protein / nucleic_acid / quantum_dot /
  nanoparticle / other) and `fluorophore_type`.
- **Canonicalize names** (e.g. `ATTO-647N` / `Atto647N` / `ATTO 647N` → one canonical
  name) and **cluster duplicates** across sources.
- **Sanity-grade quality**: flag implausible values — `QY` outside (0,1], `em_max <
  abs_max` (negative Stokes shift), abs/em maxima outside ~[200,1000] nm, extinction
  coefficient out of range, spectra whose peak disagrees with the stated max, missing
  abs/em — and propose `quality` ∈ {low,medium,high}.
- **Recommend** `approve` / `needs_review` / `reject` with a short rationale string.

**Implementation — use the EXISTING AI settings; provider-neutral, local-first
(`chisurf/core/fluorescence/curation/ai_triage.py`):**
- **Reuse `chisurf/core/settings/ai_settings.py`** — do NOT add a new provider config
  or any vendor SDK. It already supports **OpenAI, Mistral, Local (Ollama / LMStudio),
  and Custom (OpenAI-compatible)** via `get_api_settings(provider)` →
  `{base_url, api_key, text_model/model, temperature, top_p, max_tokens}`. The default
  provider is whatever the user configured (a **local LLM or Mistral** is the intended
  target — no Anthropic/Claude dependency).
- **Call the OpenAI-compatible `/chat/completions` endpoint** with the same thin
  `requests.post(base_url + "/chat/completions", headers={"Authorization": f"Bearer
  {api_key}"}, json=payload)` pattern already used in
  `plugins/core/code_editor/agent_panel.py:_llm_call` (model/temperature/max_tokens
  from `ai_settings`). No `anthropic`/`openai` package — keep it to `requests` like the
  rest of the codebase. Factor the shared call into a small helper if it's worth it.
- **Structured output without vendor features:** request `response_format={"type":
  "json_object"}` when the endpoint supports it (OpenAI/Mistral do; many local servers
  do too), but **don't rely on it** — local models vary. Put the exact JSON schema in
  the prompt, parse the reply with `json.loads`, and on parse failure retry once with a
  "return valid JSON only" reminder, then fall back to `needs_review` with the raw text
  kept. Validate the parsed object against the expected fields in Python.
- **Deterministic checks run first, in plain Python** (8.3 ranges, peak-vs-max, Stokes
  sign). The LLM only does the fuzzy parts — name canonicalization, duplicate
  clustering across sources, category from a messy name/description, an overall
  judgement. Don't ask a local model to do arithmetic the code does exactly; this also
  keeps the prompt small for smaller local models.
- **Batch by iterating** — there is no provider-agnostic batch endpoint. Process rows
  in a loop (optionally a bounded thread pool for remote providers; serial for a local
  server). Keep per-call prompts small so an 8B-class local model can handle them.
- **Offline/degrade-safe and opt-in:** if no provider is configured (`base_url`/`model`
  empty), the deterministic checks still run and populate the review queue; the LLM
  step is skipped with a logged note. Network failures never block import or approval.
  Gate behind a setting; the existing **AI Settings** GUI/plugin already lets the user
  pick the provider and model.
- **Provenance:** AI proposals are recorded as a distinct `source`/operation
  (`ai_triage`) with the resolved provider + model id + timestamp, so an approved
  value's lineage shows it was AI-proposed (by which model) then human-approved.
  Headless: `csc fluorophore ai-triage [--status unverified] [--provider <key>]`
  writes proposals; `… review-queue` lists them.

**Acceptance:** on a sample of the scraped set, the deterministic checks catch the
obvious bad rows, the AI proposals are written to the queue with rationales, and no
row is auto-approved. Tests **mock the `/chat/completions` HTTP call** (no live
network, no real provider), and the no-provider-configured path is covered. Works
end-to-end against a **local Ollama/LMStudio** model with no cloud key.

## Status — Task 1 (Förster calculator) landed

`chisurf/core/fluorescence/fret/forster.py` ships the canonical, grid-agnostic primitive:
`overlap_integral(...) -> J`, `forster_radius(J, *, donor_quantum_yield, kappa2=2/3,
refractive_index=1.33) -> R0 [Å]` (`R0 = 0.02108·(κ²·Q_D·n⁻⁴·J)^(1/6)·10`), and
`forster_radius_from_spectra(...) -> (R0, J)`. Donor emission area-normalized internally;
fail-loud on shape mismatch / non-positive donor area / negative inputs. Tests:
`test/fluorescence/test_forster.py` (9). The Light Path Simulator's grid-specific
`calculate_r0` can later delegate here.

**Reworked 2026-06-27 (Tasks 7–9):** the real data is the ~1,954-entry scraped set in
the two `_dev` plugins (`fluorophore_db` curation engine + `spectra_downloader`
scraper), not hand-rolled Gaussians. Remaining work is: integrate both tools with the
live MFDB (Task 7), add a verification/approval/quality + `source` provenance workflow
that treats all scraped data as unverified until human-approved (Task 8), and an
AI-assisted triage pass that proposes category/quality/dedup/approval with rationale
for human sign-off (Task 9, **provider-neutral via the existing
`chisurf/core/settings/ai_settings.py` — local LLM / Mistral / OpenAI-compatible, no
vendor SDK**). Tasks 2–3 (hand-authored Gaussian data) become a fallback only.

## Definition of Done

- [x] `forster.py` computes overlap integral and R0 correctly
- [ ] At least 20 common dyes have **approved** spectral data in MFDB (the curated
      real entries; Gaussian approximations only as a fallback)
- [ ] Both `_dev` plugins (`fluorophore_db`, `spectra_downloader`) are promoted out of
      `_dev/` and operate on the **live MFDB** (no parallel `spectra.db` source of
      truth; `spectra.db` used only as a one-time reference import)
- [ ] `probes` carries `source` / `source_ref` / `retrieved_at` provenance, and each
      scraper records its source
- [ ] Verification/approval workflow exists: `verification_status`
      {unverified…approved/rejected}, a `quality` grade, `verified_by`/`verified_at`;
      imported/scraped rows enter **unverified**; approve/reject API **and** CLI
- [ ] Downstream consumers (`lookup_forster_radius`, seed precompute, PRD-08 R₀,
      PRD-05 calibration) read **approved-only** by default
- [ ] AI triage pass (`ai_triage.py`) proposes category/quality/canonical-name/
      duplicate-cluster/approval-recommendation with rationale to a review queue;
      **never auto-approves**; uses the existing `ai_settings` (local LLM / Mistral /
      OpenAI-compatible via `/chat/completions`, no vendor SDK); deterministic numeric
      checks run with no provider configured; tests mock the HTTP call (no live network)
- [ ] R0 is precomputed for at least 5 common FRET pairs from approved spectra
- [ ] `lookup_forster_radius()` retrieves R0 by dye names (approved-only)
- [ ] R0 lookup is optional -- user can enter R0 directly (stored as `method="user_provided"`)
- [ ] Known-pair test gives reasonable R0 value (within 20% of literature)
- [ ] All tests pass (headless paths for import / approve / ai-triage included)

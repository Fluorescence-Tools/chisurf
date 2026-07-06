---
type: PRD
prd: "06"
title: "PRD-06: Expand the Fluorophore Database"
description: Populate MFDB with real, provenance-tracked spectral data for common dyes and compute Förster radii from spectral overlap.
status: in-progress
phase: "4"
resource: chisurf/core/mfdb
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
MFDB should ship real spectral data (absorption/emission, quantum yield, extinction) for common single-molecule FRET dyes, so a donor–acceptor pair yields a Förster radius computed from the spectral-overlap integral. The core Förster calculator (`forster.py`: overlap integral, R0, R0-from-spectra) has landed. The remaining work integrates two previously un-shipped internal tools — a fluorophore-curation engine and a spectral scraper carrying ~1,954 scraped probes — into the live MFDB as a single store, adds `source`/verification/quality provenance, and routes downstream consumers (R0 lookup, calibration feeds) to approved-only data. An AI-assisted triage pass (provider-neutral, via the existing local/OpenAI-compatible AI settings) proposes categories, quality grades, name canonicalization, and deduplication for human approval, never auto-approving.

# Status
In progress. Task 1 (Förster calculator + tests) is done; integration of the curation/scraper tools, the verification/approval workflow, and AI triage remain.

# Goal
MFDB ships with real spectral data for common smFRET dyes. Users can look up a
donor-acceptor pair and get the Förster radius automatically computed from the
spectral overlap integral.

# Background
Relevant code:
- `chisurf/core/mfdb/seed_data.py` — current seed data (7 probes, 3-point placeholder spectra)
- `chisurf/core/mfdb/schema.py` — `probes`, `optical_properties`, `spectra`, `flr_fret_forster_radius`
- `chisurf/core/mfdb/repository.py` — probe/spectra methods
- `chisurf/core/models/tcspc/fret.py` — `R0`, `FRETParameters`

# Current state
Two parallel realities exist.

**1. The shipped seed (`chisurf/core/mfdb/seed_data.py`)** — 7 probes with
placeholder spectra (3 wavelength points each): Alexa488, Alexa594, Cy3, Cy5,
ATTO647N, Trp, 2-aminopurine. The `spectra` table stores wavelength/intensity
arrays as BLOBs; the `flr_fret_forster_radius` table stores R0 for donor-acceptor
pairs. Neither table is populated with real data in the shipped DB.

**2. Two un-integrated `_dev` tools with a large scraped dataset (the real
story).** A working fluorophore-curation engine and a spectra scraper already
exist under `chisurf/plugins/_dev/` and are NOT wired into the app:

- **`_dev/fluorophore_db/`** — the curation engine. `mfdb_adapter.py`
  (`FluorophoreDatabase(MFDatabase)`) already speaks the **canonical MFDB schema**
  (`probes`/`optical_properties`/`spectra`/`probe_types`/`images`), with
  `add_probe`, `add_spectrum`, `add_optical_property`, `get_probe_full`,
  `search_probes`, `validate_probe`, and
  `get_standardized_items(include_uncurated=…)`. A curation GUI
  (`db_manager_widget.py`, `editor.py`) browses/plots spectra and toggles
  `is_curated`/`quality_flag`.
- **`_dev/spectra_downloader/`** — the scraper. `download_manager.py` runs
  per-source importer scripts (one per external fluorophore/spectra database or
  optics-vendor datasheet) as subprocesses against a `--db` path, writing into a
  `FluorophoreDatabase`.
- **`_dev/fluorophore_db/spectra.db`** — already a full canonical-schema MFDB
  carrying **~1,954 probes, ~3,226 spectra, ~15,456 optical-property rows**
  scraped from those sources.

**Why this is "not properly integrated" (the problems this PRD owns):**
1. **Separate store.** The scraped data lives in a plugin-local `spectra.db`, not
   the user's working MFDB. Both the curation GUI and the scraper point only at
   that file.
2. **In `_dev/`.** Neither tool is discoverable/shipped.
3. **Unverified, low quality.** Only **~35 of 1,954** probes are marked
   `is_curated=1`; `quality_flag` is uniformly the default `1` (never graded);
   `category` is mostly `other`/NULL. The data is a useful **initial reference
   set** but must be treated as **unverified / provisional until reviewed and
   approved**.
4. **No source provenance.** The `probes` table has no `source` column, so an
   entry can't say which external database or vendor it came from, nor carry a
   source URL / retrieval date.
5. **Downstream not wired.** `forster.py` (Task 1, done), the seed, and
   `lookup_forster_radius` don't consume this curated data.

This PRD treats those two tools — and the verification/approval workflow that
makes their scraped data trustworthy — as first-class deliverables (Tasks 7–9).

# Design and tasks

## Task 1: Förster radius calculator (done)
`chisurf/core/fluorescence/fret/forster.py`. The spectral overlap integral:

`J = ∫ F_D(λ)·ε_A(λ)·λ⁴ dλ / ∫ F_D(λ) dλ` — with `F_D` the normalized donor
emission, `ε_A` the acceptor molar extinction coefficient (M⁻¹ cm⁻¹), λ in nm; J
in M⁻¹ cm⁻¹ nm⁴. The Förster radius:

`R0 = 0.02108 · (κ²·Q_D·n⁻⁴·J)^(1/6)` (Å), κ² default 2/3, n default 1.33
(water). Verify the constant against the standard form
`R0⁶ = 9·Q_D·ln(10)·κ²·J / (128·π⁵·N_A·n⁴)` (prefactor 8.79e-25 mol with J in
M⁻¹ cm⁻¹ nm⁴, R0 in cm; ×1e8 for Å). Test against a well-known pair (Cy3-Cy5,
expected ~54 Å).

## Task 2: Add real spectral data (fallback)
`chisurf/core/mfdb/spectral_data/` — a Python module with embedded absorption/
emission spectra per dye (normalized, 1 nm spacing, extinction in M⁻¹ cm⁻¹). Where
real arrays are unavailable, generate Gaussian approximations from known abs_max,
em_max, and FWHM — better than 3-point placeholders. Typical FWHM: Alexa abs
~25–35 nm / em ~30–45 nm; Cy abs ~20–30 / em ~25–40; ATTO abs ~20–30 / em ~25–35.
Superseded by Tasks 7–9 (real scraped data); kept as a fallback only.

## Task 3: Update seed data
`chisurf/core/mfdb/seed_data.py` — replace 3-point placeholders with real/Gaussian
spectra; target ≥20 common dyes. Minimum smFRET set (abs_max / em_max / QY /
ext_coeff / FWHM_abs / FWHM_em): Alexa 488, 546, 555, 568, 594, 647; Cy3, Cy3B,
Cy5, Cy5.5; ATTO 488, 532, 550, 565, 590, 594, 647N, 655, 680; Rhodamine 110.

## Task 4: Auto-compute Förster radii for common pairs
After seeding probes with spectra, compute R0 for common donor-acceptor pairs and
store in `flr_fret_forster_radius`. Pairs (expected R0 Å): Alexa 488/594 (~54),
Alexa 488/647 (~52), Cy3/Cy5 (~54), Cy3B/ATTO 647N (~62), ATTO 488/647N (~58),
Alexa 555/647 (~51), ATTO 532/647N (~59).

## Task 5: R0 lookup in the FRET model
`chisurf/core/models/tcspc/fret.py` — `lookup_forster_radius(donor_name,
acceptor_name, db=None) -> float` retrieves R0 by dye names from
`flr_fret_forster_radius`. This is a **convenience** lookup: R0 may be "god given"
(entered directly from literature without spectra). GUI pre-fills the R0 field if
found; either way the user can override. When the fit is archived
([PRD-05](prd-05.md)), the R0 actually used is stored as a calibration — either
`method="spectral_overlap"` (from lookup) or `method="user_provided"` (entered
manually).

## Task 6: Tests
`test/fluorescence/test_forster_radius.py` — non-overlapping spectra give J = 0;
overlapping give J > 0; a known pair gives R0 within ~20% of literature; zero QY
gives R0 = 0.

## Task 7: Integrate the curation + scraper tools with MFDB
The two `_dev` tools are the real-data engine and supersede the embedded-Gaussian
approach of Task 2 — the data is sourced from authoritative databases, not
hand-rolled. The work is **integration + curation**, not building from scratch.

- **7.1 Promote both tools out of `_dev/`** to shipped, manifest-discovered
  plugins. Mark them **experimental** initially (the data is unverified).
- **7.2 Make MFDB the single store (no parallel `spectra.db`).** The engine
  already subclasses `MFDatabase` on the canonical schema, so the gap is *which
  database* it opens. Point the curation GUI and the scraper at the active MFDB
  resolved through `ChiSurfAPI`/`resolve_database_path` (the configured DB, or an
  explicit `--db`), not the hardcoded plugin-local `spectra.db`. Keep `spectra.db`
  only as an *import source* — a one-time `import_reference_set` that copies its
  probes/spectra/optical-properties into the target MFDB (idempotent, dedup by
  name+source), stamping every imported row as unverified (Task 8).
- **7.3 Add `source` provenance.** Add a `source` column to `probes` identifying
  the origin (a specific public fluorophore database, dye manufacturer,
  filter/optics vendor, `user`, or `literature`) plus `source_ref` (URL/accession)
  and `retrieved_at`. Follow the `.dic`-driven schema rule (PRD-19): if `probes`
  becomes a `.dic`-declared table, add these via the dictionary +
  `reconcile_schema`; if it stays hand-DDL in `schema.py`, add the columns there
  and in the adapter's INSERT/UPDATE. Each scraper records its own `source` +
  `source_ref` when it writes a row.
- **7.4 Wire downstream consumers to *approved* data only.**
  `lookup_forster_radius` (Task 5), the seed precompute (Tasks 3/4), the Light
  Path Simulator crosstalk/R₀ ([PRD-08](prd-08.md) Task 8), and the
  [PRD-05](prd-05.md) computed-from-spectra calibration must read **only
  `verification_status=approved`** probes by default (Task 8), so unverified
  scraped rows never silently feed a published R₀.
- **7.5 Headless path.** Importer + reconcile + approval transitions runnable
  without the GUI (`csc fluorophore import-reference-set`, `… approve <probe>`,
  `… list --status unverified`). The curation GUI is thin wiring over these.

Keep the importers/curation pure-Python (network + spectra parsing); MFDB
read/write goes through the plugin backend services mirroring the repository
methods.

## Task 8: Verification / approval / quality workflow (provenance)
The scraped 1,954 entries are an **initial reference set of unverified,
low-quality data** — usable as a starting point, but nothing downstream should
trust them until a human approves.

- **8.1 Vocabulary.** `verification_status` ∈ {`unverified`, `under_review`,
  `approved`, `rejected`, `superseded`} (sourced from the `.dic` enumeration per
  PRD-19 — never a hand Python list); `approved` ⟺ the existing `is_curated=1`
  (keep as a generated mirror or migrate callers). `quality` grade ∈ {`unknown`,
  `low`, `medium`, `high`} (replaces the uniform `quality_flag`).
  `verified_by`/`verified_at` (who approved, when — reusing the PRD-17 identity
  resolver).
- **8.2 On import, everything enters `unverified` / `quality=unknown`.** The
  reference set and every fresh scrape land provisional.
- **8.3 Approval transitions.** `approve` / `reject` / `mark_under_review` /
  `set_quality` API + CLI, validated and recorded. Consider modeling these as the
  [PRD-12](prd-12.md) lifecycle state machine and the approval as a PRD-21
  provenance operation, so curation history is queryable — but status columns + an
  audit row is an acceptable minimum.
- **8.4 Downstream filter.** Default every consumer (7.4) to approved-only; allow
  an explicit `include_unverified=True` escape hatch for power users.

## Task 9: AI-assisted triage and curation (human-in-the-loop)
1,954 mostly-unverified entries are too many to hand-curate cold. Add an **AI
triage pass that proposes** category, quality grade, canonical name, duplicate
clusters, and an approve/flag recommendation **with rationale** — the human (or a
downstream rule) makes the final `approve`/`reject` call (Task 8). **AI proposes,
human disposes** — the AI never auto-sets `verification_status=approved`; it writes
proposals to a review queue.

Per probe (or duplicate cluster) the AI: **classifies** `category` (organic_dye /
protein / nucleic_acid / quantum_dot / nanoparticle / other) and
`fluorophore_type`; **canonicalizes names** (e.g. `ATTO-647N` / `Atto647N` / `ATTO
647N` → one canonical name) and **clusters duplicates** across sources;
**sanity-grades quality** (flag implausible values — QY outside (0,1], `em_max <
abs_max`, maxima outside ~[200,1000] nm, extinction out of range, peak/max
disagreement, missing abs/em) and proposes a grade; **recommends** approve /
needs_review / reject with a short rationale.

Implementation — use the **existing AI settings**; provider-neutral, local-first
(`chisurf/core/fluorescence/curation/ai_triage.py`):
- **Reuse `chisurf/core/settings/ai_settings.py`** — do NOT add a new provider
  config or any vendor SDK. It already supports local and OpenAI-compatible
  providers via `get_api_settings(provider)` → `{base_url, api_key,
  text_model/model, temperature, top_p, max_tokens}`. A local model is the
  intended default (no dependency on any specific cloud provider).
- **Call the OpenAI-compatible `/chat/completions` endpoint** with the same thin
  `requests.post(...)` pattern already used in
  `plugins/core/code_editor/agent_panel.py:_llm_call`. No vendor package — keep it
  to `requests`.
- **Structured output without vendor features:** request `response_format={"type":
  "json_object"}` when supported, but don't rely on it — put the JSON schema in the
  prompt, parse with `json.loads`, retry once on failure, then fall back to
  `needs_review` keeping the raw text. Validate parsed fields in Python.
- **Deterministic checks run first, in plain Python** (ranges, peak-vs-max, Stokes
  sign). The LLM only does the fuzzy parts (name canonicalization, dedup, category
  from a messy name, an overall judgement) — never arithmetic the code does
  exactly.
- **Batch by iterating** (optional bounded thread pool for remote providers;
  serial for a local server). Keep per-call prompts small for smaller local models.
- **Offline/degrade-safe and opt-in:** with no provider configured, the
  deterministic checks still run and populate the review queue; the LLM step is
  skipped with a logged note. Network failures never block import or approval.
- **Provenance:** AI proposals are recorded as a distinct `source`/operation
  (`ai_triage`) with the resolved provider + model id + timestamp, so an approved
  value's lineage shows it was AI-proposed (by which model) then human-approved.
  Headless: `csc fluorophore ai-triage [...]` writes proposals; `… review-queue`
  lists them.

**Acceptance:** on a sample of the scraped set, deterministic checks catch obvious
bad rows, AI proposals are written to the queue with rationales, and no row is
auto-approved. Tests **mock the `/chat/completions` HTTP call** (no live network,
no real provider), and the no-provider-configured path is covered.

# Implementation status — Task 1 (Förster calculator) landed
`chisurf/core/fluorescence/fret/forster.py` ships the canonical, grid-agnostic
primitive: `overlap_integral(...) -> J`, `forster_radius(J, *,
donor_quantum_yield, kappa2=2/3, refractive_index=1.33) -> R0 [Å]`
(`R0 = 0.02108·(κ²·Q_D·n⁻⁴·J)^(1/6)·10`), and `forster_radius_from_spectra(...) ->
(R0, J)`. Donor emission area-normalized internally; fail-loud on shape mismatch /
non-positive donor area / negative inputs. Tests:
`test/fluorescence/test_forster.py` (9). The Light Path Simulator's grid-specific
`calculate_r0` can later delegate here.

**Reworked (Tasks 7–9):** the real data is the ~1,954-entry scraped set in the two
`_dev` tools (curation engine + scraper), not hand-rolled Gaussians. Remaining
work: integrate both tools with the live MFDB (Task 7), add a
verification/approval/quality + `source` provenance workflow treating all scraped
data as unverified until human-approved (Task 8), and an AI-assisted triage pass
that proposes category/quality/dedup/approval with rationale for human sign-off
(Task 9, provider-neutral via the existing `ai_settings` — local/OpenAI-compatible,
no vendor SDK). Tasks 2–3 (hand-authored Gaussian data) become a fallback only.

# Definition of Done
- [x] `forster.py` computes overlap integral and R0 correctly
- [ ] At least 20 common dyes have **approved** spectral data in MFDB (curated
      real entries; Gaussian approximations only as a fallback)
- [ ] Both `_dev` tools (`fluorophore_db`, `spectra_downloader`) are promoted out
      of `_dev/` and operate on the **live MFDB** (no parallel `spectra.db` source
      of truth; `spectra.db` used only as a one-time reference import)
- [ ] `probes` carries `source` / `source_ref` / `retrieved_at` provenance, and
      each scraper records its source
- [ ] Verification/approval workflow exists: `verification_status`
      {unverified…approved/rejected}, a `quality` grade, `verified_by`/
      `verified_at`; imported/scraped rows enter **unverified**; approve/reject API
      **and** CLI
- [ ] Downstream consumers (`lookup_forster_radius`, seed precompute, PRD-08 R₀,
      PRD-05 calibration) read **approved-only** by default
- [ ] AI triage pass (`ai_triage.py`) proposes category/quality/canonical-name/
      duplicate-cluster/approval-recommendation with rationale to a review queue;
      **never auto-approves**; uses the existing `ai_settings` (local/
      OpenAI-compatible via `/chat/completions`, no vendor SDK); deterministic
      numeric checks run with no provider configured; tests mock the HTTP call
- [ ] R0 is precomputed for at least 5 common FRET pairs from approved spectra
- [ ] `lookup_forster_radius()` retrieves R0 by dye names (approved-only)
- [ ] R0 lookup is optional — user can enter R0 directly (stored as `method="user_provided"`)
- [ ] Known-pair test gives reasonable R0 value (within 20% of literature)
- [ ] All tests pass (headless paths for import / approve / ai-triage included)

# Relationships
- Feeds R0 / crosstalk into [PRD-08](prd-08.md) (optical configuration) and the calibration provenance work.
- Reagent inventory [PRD-15](prd-15.md) may link fluorophore lots to probe records.
- Builds on the dictionary-driven schema of [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).

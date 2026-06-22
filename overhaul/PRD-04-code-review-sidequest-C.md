# PRD-04 Sidequest C (G-Factor Calibration Provenance) — Code Review

Review of the G-factor archival implementation. Source: `PRD-04-burst-pipeline.md`
"Sidequest C: G-Factor Calibration Provenance".

Reviewed (2026-06-20):

- `chisurf/plugins/jordi_g_factor/backend/services.py` (`archive_g_factor_handler`)
- `chisurf/plugins/jordi_g_factor/gui/client.py`, `gui/tool.py`
- `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_channel_definition_tttr_io.py`
- `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_detector_setups.py`
- `chisurf/core/mfdb/{schema.py, data/mfdb_flr_ext.dic, repository.py}`
- `chisurf/core/fluorescence/fcs/channel_setups.py` (FCS regression)
- `chisurf/plugins/jordi_g_factor/test/test_calib_provenance.py`

Tests: 24 passed across jordi_g_factor, fcs_channel_preset, setup prerequisites,
detector setups, and user-migration suites.

## Verdict

**Solid — the provenance chain is built end to end and all tests pass.** The
reference decay is archived, the G-factor is registered as a parented calibration
via PRD-03, the detector channel carries a dictionary-declared link, and the FCS
read-only-construction regression is fixed. Findings are minor (one test gap, one
robustness guard, two optional hardenings).

## Task status

### ✅ C1 — reference decay registered
`archive_g_factor_handler` registers the VV/VH Jordi file via
`register_raw_measurement(file_path, metadata=...)` with channel-role metadata
(`parallel`/VV, `perpendicular`/VH), filename, micro-time resolution, and user
(`services.py:213`). Test asserts `artifact_kind == "raw_measurement"` and
metadata.

### ✅ C2 — G-factor archived as a parented calibration
`register_calibration(data=payload, calibration_type="g_factor",
parent_artifact_id=ref_decay_id, parameters=calib_params, method="jordi_g_factor",
notes=...)` (`services.py:243`). Scalars recorded: `g_factor`, `g_factor_stddev`,
`g_factor_uncorrected`/`corrected`, `r_inf` (auto-computed when absent),
`region_min/max`, `decay_shift`, `flip`, `use_bg`, `bg_vv/vh`, `l1`, `l2`. Reuses
PRD-03, no hand-rolled writes. Best-effort: returns `{ok: False}` and never raises
into the UI; a missing-DB graceful-failure test passes.

### ✅ C3 — detector channel ↔ calibration link
`g_factor_calibration_id` added to `mfdb_setup_detector_channel`: dictionary-declared
(`mfdb_flr_ext.dic:2571`), generated/migrated (v34 `_ensure_column`,
`schema.py:3738`), covered by the total-coverage gate (`_SETUP_CATEGORIES`).
Persisted by `save_setup` (`repository.py:4027`), round-tripped by
`_setup_row_data` (`tttr_detector_setups.py:92`), and **actually written** when the
wizard applies a computed G-factor (`tttr_channel_definition_tttr_io.py:236,243`).

### ✅ C4 — broken integration fixed
The Jordi GUI→client→service→MFDB path is wired and tested. The FCS regression is
resolved: `load_fcs_channel_setups` regained `skip_migration` / `db_path` /
`user_id` (`channel_setups.py:129-164`), and the dialog constructs read-only
(`_load_state` passes `skip_migration=True` for both the FCS and detector reads,
`fcs_channel_preset/__init__.py:198-207`).

## 🟡 Minor / follow-ups

1. **The parent edge is not asserted in the test.** `test_archive_g_factor_provenance`
   ("registers the reference decay and *parented* calibration") checks both
   artifacts and the payload, but never verifies the provenance edge linking the
   calibration back to the reference decay — the exact traceability this sidequest
   exists for. The code passes `parent_artifact_id=ref_decay_id`, but nothing
   tests it landed. Add an assertion on the edge (e.g. query `mfdb_edge` /
   `derived_from` from `calib_id` to `ref_decay_id`).
2. **`notes` f-string crashes when `g_factor` is None.**
   `notes=f"Calculated G-factor: {g_val:.4f} ..."` (`services.py:249`) runs
   unconditionally; if `parameters["g_factor"]` is absent, `None:.4f` raises
   `TypeError`, caught into `{ok: False}` — the calibration is silently dropped.
   Guard `g_val` (early return with a clear warning, or format conditionally)
   so a missing g-factor is an explicit error, not a swallowed format crash.
   This path is untested (the test always passes `g_factor=1.5`).
3. **No FK on `g_factor_calibration_id`.** The `.dic` item is plain `text` with no
   `_chisurf_schema.foreign_key` to `mfdb_artifact(artifact_id)`. A loose
   reference is acceptable and consistent with other id-in-column patterns, but
   since the table is generated, adding the FK is cheap and would enforce
   integrity. Optional.
4. **C3 round-trip not directly tested.** The plumbing
   (`save_setup` → child row → `_setup_row_data`) exists, but no test asserts that
   saving a channel with `g_factor_calibration_id` and reloading preserves it. A
   small focused test would lock the link.

## Note

I could not independently reproduce the original "Jordi G-factor calc does not
work" GUI failure (the `core/calculations.py` unit tests already passed before
this work, and the integration tests pass now). The flow is correct and archives
as specified, but the review can't confirm root-cause vs. symptom — worth a
one-line note from the implementer on what the actual bug was.

## Required before "complete"

- [ ] Minor 1: assert the calibration→reference-decay provenance edge in the test.
- [ ] Minor 2: guard the `notes` f-string / missing `g_factor` so it fails loudly,
      not silently.
- [ ] (Optional) Minor 3/4: FK on `g_factor_calibration_id`; C3 round-trip test.

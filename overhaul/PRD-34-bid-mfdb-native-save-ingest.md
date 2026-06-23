# PRD-34: BID (burst selection) saves to MFDB, downstream plugins ingest MFDB BIDs directly

## Status

Proposed — separate PRD requested during manual testing of the burst → ndXplorer
round trip.

## Goal

When chisurf is **connected to MFDB**, saving a **BID** (a burst-identification /
burst-selection result — the `.bur`/`bi4_bur` "BID" folder produced by Burst
Selection and consumed by the FRET/burst tools) should write the selection to
**MFDB**, not (only) to a loose file on disk. Downstream burst plugins (ndXplorer,
BVA, Burst MLE, burst-wise FCS, Trace/Burst Browser, BID→Analysis, …) should then
**ingest BIDs directly from MFDB** — pick a registered BID via the dataset picker
instead of browsing for a folder.

```
Burst Selection ──(MFDB connected)──▶ BID registered in MFDB (sample-linked)
                                            │
        ┌───────────────────────────────────┼───────────────────────────────┐
        ▼                                   ▼                                 ▼
   ndXplorer (open BID)           BVA / Burst MLE / burst-FCS         BID→Analysis
   from the dataset picker        ingest the same MFDB BID            ingest the same MFDB BID
```

## Why

Today a BID is saved as a folder on disk and every downstream tool re-browses the
filesystem for it. That:
- scatters provenance (no link between the BID, its raw measurement, the sample,
  and the operation/parameters that produced it),
- forces manual path juggling between tools and invites "wrong folder" errors,
- duplicates the burst output and loses the single-group/named conventions MFDB
  already provides (PRD-28),
- and is inconsistent with the rest of the overhaul, where results are registered
  artifacts (PRD-03/11/16) discoverable through the dataset picker (PRD-10).

The plumbing already exists: Burst Selection can register a run into MFDB
(`--mfdb`, PRD-28: raw inputs sample-linked, burst tables, and a single
`external_reference` output-folder group), and `mfdb.datasets.open` resolves that
group back to the on-disk BID folder co-located with the TTTR. This PRD makes that
the **default save path when connected**, and makes the downstream tools open BIDs
from MFDB.

## Key constraint (carry over from PRD-28)

**`.bur`/BID files reference photons by index in the original TTTR file.** A BID is
only meaningful next to its co-located TTTR. So "save to MFDB" means **register a
reference** (a directory `external_reference` with its on-disk path in metadata,
co-located with the TTTR), **not** copy the `.bur` bytes into the object store —
copying would break the photon-index linkage. `mfdb.datasets.open` materializes the
reference to that path; downstream tools open the path as today.

## Scope

### A. Save behavior (Burst Selection + any BID producer)
- When an MFDB connection/sample context is active, the default "save BID" action
  registers the BID in MFDB (sample-linked, `operation_type="burst_selection"`,
  the single output-folder group + burst tables) instead of only writing a folder.
- Keep a **file-only** escape hatch (no MFDB / explicit "export to folder") for
  disconnected use; when disconnected, behavior is unchanged (write to disk).
- Idempotent/no double-registration; reuse the PRD-28 single-group + real-name
  conventions.

### B. Ingest behavior (downstream burst plugins)
- Each downstream tool gains an **"Open BID from MFDB…"** entry that uses the
  existing `MfdbDatasetPickerDialog` filtered to BID artifacts
  (`kind="external_reference"`, `format="directory"`, burst operation), resolves
  the artifact to a local BID path via `mfdb.datasets.open`, and loads it exactly
  as a folder open would.
- File-browse open stays available for disconnected use.
- A small shared helper (mirroring `chisurf/plugins/ndxplorer/mfdb_launcher.py`'s
  `resolve_dataset_path`) so each tool wires "pick → resolve → open" without
  reimplementing MFDB access.

### C. Provenance
- A registered BID links to its raw measurement(s), sample, setup, and the burst
  operation parameters, so downstream results (BVA/MLE/FCS) can be parented to the
  BID artifact — a clean lineage chain raw → BID → analysis.

## Existing pieces to reuse (do not reinvent)
- **Registration:** `burst_selection/api/mfdb.py` + the `--mfdb` CLI path (PRD-28):
  raw+sample registration, burst tables, single output-folder group, real names.
- **Resolve/open:** `mfdb.datasets.open` / `MFDatabase.open_dataset` →
  co-located BID folder; `resolve_dataset_path` in the ndXplorer launcher.
- **Picker:** `chisurf/gui/widgets/mfdb/dataset_browser.MfdbDatasetPickerDialog`
  (`kinds`/`formats`/`scope`), already used by ndXplorer (PRD-28) and the Microtime
  Shifter.
- **Identity/scope:** PRD-17 resolver so "Mine/All" scope works in the picker.

## Tasks
1. Define the BID artifact contract (kinds/formats/operation_type/metadata) so a
   "BID" is unambiguously discoverable in the picker across producers/consumers.
2. Burst Selection GUI: make "save BID" register to MFDB when connected (default),
   with a file-only escape hatch; reuse the PRD-28 registration path.
3. Shared `open_bid_from_mfdb(...)` helper (pick → `resolve_dataset_path` → return
   local BID path) for downstream tools.
4. Wire "Open BID from MFDB…" into the downstream burst plugins (ndXplorer already
   has it via PRD-28; add to BVA, Burst MLE, burst-FCS, Trace/Burst Browser,
   BID→Analysis).
5. Parent downstream results to the BID artifact for lineage.
6. Tests: register a BID when connected → it appears in the picker (one named
   group) → each downstream tool resolves and opens it; disconnected path still
   writes/opens a folder. Hermetic-harness round trips.

## Definition of Done
- [ ] When connected, saving a BID registers it in MFDB (reference, sample-linked,
      single named group); disconnected still writes a folder.
- [ ] Downstream burst plugins can pick a registered BID from MFDB and open it
      without browsing the filesystem.
- [ ] Photon-index linkage preserved (BID stays a reference next to the TTTR; no
      object-store copy of `.bur`).
- [ ] Downstream results are parented to the BID artifact (lineage).
- [ ] Tests + a manual round trip pass.

## Definition of Clean
Reuse the PRD-28 registration + `mfdb.datasets.open` + the PRD-10 picker (no new
browser, no `.bur` copying); one shared resolve/open helper; identity/scope via
PRD-17; file paths only as the disconnected escape hatch.

## Relationship
Generalizes **PRD-28** (which did this for ndXplorer specifically) to *all* BID
producers/consumers and makes MFDB the default save target when connected. Rides
the **PRD-03/11/16** registry/operation/transformer spine and **PRD-10** picker.
Complements **PRD-33** (acquisition→MFDB registration) — same "register, don't just
write a file when connected" principle, applied to burst selections.

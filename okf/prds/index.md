# PRDs — Design Notes

Product-requirement / design notes driving current work. Each PRD is one **self-contained** concept (`prd-<n>.md`) carrying its full design text and keeping its **persistent number** as a stable id; references to external software are described by role, not named. These were formerly the top-level `overhaul/` folder, now retired into this group. The authoritative implementation ordering is [PRD Implementation Order](master-order.md).

Status: ✅ done · 🚧 in-progress · ✏️ draft · 🌱 stub · 📋 planned · ⛔ superseded

# Foundation — sample tracking, dictionary infrastructure, ORM/payload

* ✅ [PRD-02: Sample Tracking — Deep Sample Description](prd-02.md) — Link every dataset and result to a full atomistic, flrCIF-aligned sample description.
* ✅ [PRD-02a: mmCIF Dictionary Infrastructure](prd-02a.md) — Parse bundled mmCIF dictionaries into a cached API for vocabulary validation and autocomplete.
* ✅ [PRD-02b: MFDB Admin Overhaul — Manual Inspection & Editing](prd-02b.md) — Make the mfdb-admin plugin inspect, add, and edit every record the sample data model produces.
* ✅ [PRD-02c: Aligning ChiSurf MFDB Export to flrCIF](prd-02c.md) — Map ChiSurf's internal parameter short names to canonical flrCIF dictionary items on export.
* ✅ [PRD-020: SQLAlchemy MFDB Mapping](prd-020.md) — A bounded SQLAlchemy relationship layer for the MFDB sample, probe, and FRET-pair tables.
* ✅ [PRD-030: Result Payload Formats & Codecs](prd-030.md) — A typed, msgpack-based, schema-validated codec layer for MFDB scientific result payloads.

# Phase 0 — Current line (result registry + burst pipeline)

* 📋 [PRD-01: Fix MFDB Project Round-Trip](prd-01.md) — Make archiving a project to MFDB and restoring it produce an identical project.
* 🚧 [PRD-03: Result Registry](prd-03.md) — A single register_result() API so any plugin can archive output to MFDB with full provenance.
* 🚧 [PRD-04: Stable Burst Pipeline MFDB Integration](prd-04.md) — Register burst-selection results in MFDB with stable, queryable provenance across all callers.
* ✅ [PRD-09: Microtime Shifter — Workflow Plugin + MFDB Provenance](prd-09.md) — Convert the microtime-shifter tool into a layered api/backend/cli/gui workflow plugin with RPC and full MFDB provenance.
* 🚧 [PRD-10: MFDB Dataset Browser Widget](prd-10.md) — A reusable Qt widget to pick a registered MFDB dataset; file-group membership remains open.

# Phase 1 — Architecture foundations

* 🚧 [PRD-17: Canonical Identity / Session Context](prd-17.md) — Fixes ownership/visibility around the canonical user resolver; explicit SessionContext threading remains open.
* 🚧 [PRD-18: Dependency Injection + Hermetic Test Harness](prd-18.md) — Makes the suite hermetic and contracts the real MFDB client, while handler/session injection remains incomplete.
* 🚧 [PRD-19: Single Canonical Dictionary-Driven Schema](prd-19.md) — Collapses legacy duplicate table families and seeds vocabulary from the dictionary; hand-written core DDL remains.
* 🚧 [PRD-25: Consistency Hardening + Correctness Primitives](prd-25.md) — A set of cross-cutting correctness changes — uniform fail-loud errors, one RPC envelope, a single sample read path, caching, N+1 removal, dead-code removal — plus typed IDs, first-class units, and boundary validation.
* 🚧 [PRD-27: Event-Sourced, Append-Only Provenance Core](prd-27.md) — Locks the append-only-lite decision and implements branch/event-log pieces, but full reconstructable append-only provenance remains incomplete.

# Phase 2 — Operation/transformer spine + model-driven layer

* ✅ [PRD-11: Transformers as Abstract Data-Operation Nodes](prd-11.md) — Model every data-manipulation step as a uniform MFDB operation node with typed, dictionary-declared parameters and input/output ports.
* ✅ [PRD-16: General Transformer Contract](prd-16.md) — Defines one uniform contract every data-transformer plugin must obey — typed ports, dictionary-declared parameters, a pure transform, and uniform provenance registration.
* 🚧 [PRD-26: Model-Driven Data Layer](prd-26.md) — Adds dictionary-derived DAO, validation, docs, and admin registry pieces; hand-maintained CRUD/SQL remains.
* 🚧 [PRD-28: Companion-Tool ↔ MFDB Burst-Selection Round Trip](prd-28.md) — Implements open-from-MFDB and CLI handoff; the direct "send current Burst Selection result" GUI path remains open.
* 📋 [PRD-29: Visual Burst Programming — Node-Graph Editor for Burst Analysis](prd-29.md) — Turns the existing node editor into a visual programming canvas for composing, running, previewing, and provenance-recording burst-analysis pipelines.
* 📋 [PRD-31: Headless CLI for the Companion Photon-Data Exploration Tool](prd-31.md) — Adds a windowless CLI to the companion exploration tool for parameter-based burst filtering and imaging, integrated with MFDB.

# Phase 3 — Provenance + LIMS

* ✅ [PRD-12: Lifecycle State Machines + Transition History](prd-12.md) — Turn flat entity status flags into tracked lifecycles with a recorded, validated transition log (who, when, why).
* ✅ [PRD-13: Study / Project Entity with Configurable Fields](prd-13.md) — Promote the loose project-id string into a real study entity that groups samples and datasets with ownership, membership, and custom fields.
* ✅ [PRD-14: Protocol Entity — Named, Versioned Procedures](prd-14.md) — Add named, versioned measurement/processing protocols with declared parameter schemas that operations reference for reproducibility.
* ✅ [PRD-21: Provenance/Lineage Query API + Event Model](prd-21.md) — Makes the provenance graph queryable through a first-class lineage API, stores a replayable compute spec on each derived artifact, and adds an in-process event model for reactive behaviour.

# Phase 4 — Composition + remaining features

* 🚧 [PRD-05: Calibration Provenance](prd-05.md) — Track calibration parameters in MFDB with links to the reference measurements they derive from.
* 🚧 [PRD-06: Expand the Fluorophore Database](prd-06.md) — Populate MFDB with real, provenance-tracked spectral data for common dyes and compute Förster radii from spectral overlap.
* 📋 [PRD-08: Optical Configuration Schema](prd-08.md) — Replace opaque setup JSON blobs with structured, queryable tables describing the full optical path from source to detector.
* ✅ [PRD-15: Lightweight Reagent / Consumable Inventory](prd-15.md) — Track consumables (dye lots, buffers, filters, kits) with lot/expiry and link them to operations, setups, and samples for reproducibility.
* ✅ [PRD-22: Workflow / Pipeline Engine on the Transformer Contract](prd-22.md) — Lets users compose conformant transformers into a type-checked, node-based dataflow pipeline that executes headlessly and is recorded in MFDB as a reproducible chain of operations.
* 📋 [PRD-39: Sequence Provenance & External References](prd-39.md) — Records each entity's canonical sequence/structure cross-references and its engineered mutations as structured, exportable flrCIF/PDBx data using the standard struct_ref category family.

# Phase 5 — Capstone

* 📋 [PRD-24: Extract MFDB into a Standalone Package](prd-24.md) — Moves MFDB (schema, dictionary, generator, repository/API, server, admin backend) out of chisurf into an independent module with a stable public API and no chisurf imports.

# Cross-cutting (interleave throughout)

* 🚧 [PRD-23: Thin Widgets / View–API Separation](prd-23.md) — Makes GUI widgets pure view — no data processing, no database or acquisition-library calls, no side effects on construction — with mandatory construction smoke tests and a shared dockable-tool base.
* 📋 [PRD-30: CLI Pipeline Tools with Unix Pipe Support](prd-30.md) — Gives burst/TTTR CLI tools stdin/stdout streaming via a self-describing msgpack frame format so they compose as Unix pipes.
* 📋 [PRD-32: Acquisition Standard Output Folder](prd-32.md) — Adds a single user-configurable standard output folder to acquisition so new measurements have a predictable save location.
* 📋 [PRD-33: Acquisition-to-MFDB Registration](prd-33.md) — Adds a save mode that writes a newly acquired measurement directly into MFDB with sample linkage and provenance.
* 📋 [PRD-34: Burst-ID Native MFDB Save and Downstream Ingest](prd-34.md) — Makes MFDB the default save target for burst-identification selections when connected, and lets downstream burst tools ingest them directly from the dataset picker.
* 🚧 [PRD-36: Dockable-Tool Base Migration Tracker](prd-36.md) — Tracks the per-tool rollout of the shared dockable-tool base across remaining QMainWindow plugin tools so drag-drop, dock, geometry, and MFDB-connectivity boilerplate is implemented once.
* 🚧 [PRD-43: Align GUI Operation History with MFDB Provenance](prd-43.md) — Makes the in-memory operation history a projection over a durable MFDB event log so undo/redo and the exploration trail survive database save/restore, and incrementally aligns the GUI event stream with the backend provenance model.
* ✏️ [PRD-47: Relocate Spectroscopy Physics into an External Biophysical Modeling Framework](prd-47.md) — Consolidate duplicated fluorescence-spectroscopy physics so an external biophysical modeling framework becomes the single home, reducing ChiSurf to fitting-model glue, GUI, and data-IO that calls into it.

# Feature / roadmap tracks (independent)

* ⛔ [PRD-07: Plugin MFDB Integration](prd-07.md) — Have high-priority plugins register their results in MFDB via a single registration call at each output point.
* 📋 [PRD-35: Light-Path Optical Presets Stored in MFDB](prd-35.md) — Migrates optical-path presets from JSON files on disk into MFDB as queryable, versioned, provenance-tracked entities.
* ✏️ [PRD-37: Network Deployment Security — transport auth + RPC authn/authz](prd-37.md) — Adds encrypted, endpoint-authenticated transport plus per-call authentication, authorization, and scoped event broadcast so the server may bind beyond loopback.
* 🚧 [PRD-38: Model/UI Split — view-spec JSON drives auto-generated model editors](prd-38.md) — Splits a fitting model's compute definition from its editor by describing the editor in a co-located JSON view spec that a generic GUI renderer turns into the control panel.
* 📋 [PRD-40: A ChiSurf-native declarative dataset-to-editor framework](prd-40.md) — Provides one reusable way to declare a typed dataset once and auto-generate its Qt editor across models, settings, and tool panels, replacing the several ad-hoc type-to-widget mappers.
* ✏️ [PRD-41: FDB4ChemBio Access-Layer & Interoperability Strategy](prd-41.md) — A design note fixing the architectural boundary for MFDB as a prototype public resource — the dictionary is the product, deposition and dissemination differ, and a future read-only GraphQL endpoint is generated from the dictionary.
* 📋 [PRD-42: Confine pyqtgraph to plot-only widgets](prd-42.md) — Restricts pyqtgraph to plot-canvas code and replaces its non-plot uses (numeric spin boxes, parameter trees) with dependency-free Qt-native equivalents.
* ✅ [PRD-44: Vendor-Neutral Dictionary Schema Namespace](prd-44.md) — Renames the MFDB dictionary's local extension tags from an application-branded namespace to a store-keyed vendor-neutral one, behind a backward-compatible parser, so MFDB is usable by software beyond ChiSurf.
* ✏️ [PRD-45: Chemical registry-number as a first-class chemical identity in MFDB](prd-45.md) — Promotes the chemical registry number from an ad-hoc free-text property to a dictionary-defined, validated, indexed, cross-entity chemical identity surfaced across GUI, CLI, and RPC.
* 📋 [PRD-46: Scripts as first-class citizens in the test pipeline](prd-46.md) — Brings shipped example scripts into the automated test suite by running headless/process scripts under a shebang-driven runner with numeric assertions, plus unit tests for the public model API they exercise.
* ✏️ [PRD-48: Provider-Agnostic ELN Integration](prd-48.md) — A provider-agnostic ELN integration layer for MFDB with a single gateway abstraction and two concrete electronic-lab-notebook backends, supporting bidirectional deposit, import, and reconciliation.
* ✏️ [PRD-49: Multiparameter-Fluorescence Feature Parity](prd-49.md) — Roadmap PRD to reach and surpass feature parity with an established multiparameter-fluorescence analysis suite across every analysis modality, delegating implementation to per-module sub-PRDs.
* ✏️ [PRD-50: Photon Distribution Analysis (PDA) Family](prd-50.md) — Wrap the existing PDA histogram engine in ChiSurf models and AutoForm view specs, covering static distance-distribution PDA, dynamic/N-state kinetic PDA, error surfaces, three-color PDA, and a kinetic consistency check.
* 🌱 [PRD-51: Imaging Correlation — N&B, tICS/STICS, iMSD, Spectral RICS](prd-51.md) — Extend the existing RICS/CLSM core with Number & Brightness, temporal/spatiotemporal image correlation, iMSD, and crosstalk-free spectral RICS.
* 🌱 [PRD-52: Phasor-FLIM Imaging & Particle Tracking](prd-52.md) — Add per-pixel phasor-FLIM imaging, universal-circle ROI segmentation, per-PIE/spectral-channel phasor, and phasor-based particle detection and tracking.
* 🌱 [PRD-53: Simulation Workflow (Diffusion + FRET + Photon + Camera)](prd-53.md) — Surface the existing Monte-Carlo simulation engines as a headless + AutoForm workflow producing synthetic ground truth for downstream analysis validation.
* 🌱 [PRD-54: Spectral Unmixing, pCF, nsFCS & FCCS Models](prd-54.md) — Close the remaining spectroscopy gaps — spectral unmixing/spectral phasor, pair-correlation analysis, nanosecond-FCS/antibunching, a dedicated FCCS fit model, and a 2-photon FCS model.
* ✏️ [PRD-55: Phasor Analysis Toolkit (open-library parity)](prd-55.md) — Turn ChiSurf's phasor viewer into a phasor analysis toolkit by natively implementing apparent-lifetime readout, g,s filtering, component fraction/unmixing, and cursor masks, with no new dependencies.
* 🚧 [PRD-56: Companion-Tool ↔ ChiSurf RPC + Phasor Overlays](prd-56.md) — Make the companion photon-data exploration tool a live phasor front-end over a first-class RPC client, with ChiSurf serving phasor math and shared overlay-line geometry.
* 📋 [PRD (detector-setup): Centralized Detector Setup Selection](prd-detector-setup.md) — Replace the full detector/PIE-window wizard page embedded across 15+ plugin UIs with a lightweight setup-selector widget that opens the full editor on demand.
* 🚧 [PRD-57: ChiMOL Command Parity and Renderer Migration](prd-57.md) — Grow the ChiMOL viewer's `cmd` surface toward reference-language parity and migrate its renderer to an immediate-mode GUI backend behind a Qt-free controller/scene contract (Tiers 0–3 landed).
* ✏️ [PRD-58: FRET Plugin as a Strict FPS + OLGA Superset](prd-58.md) — Make the `fret` plugin a strict superset of the legacy FPS and OLGA tools — embedded fps.json editing with live AV preview, refine/bootstrap/sample/evaluate workflows, and informative pair selection (evaluator subpackage landed).
* 🚧 [PRD-59: Pluggable MFDB Authentication (local / LDAP)](prd-59.md) — Replace MFDB's hardcoded local auth with a pluggable provider layer (local + LDAP/Active-Directory) behind one `login()` orchestrator, resolving to the existing Principal/session with JIT provisioning and directory-group mapping, behind an extensible provider registry (local + LDAP + headless CLI + hardening landed).
* ✏️ [PRD-60: Amortised (Surrogate) Neural Estimator for H2MM](prd-60.md) — An optional simulation-based-inference fast path that predicts H2MM parameters in one neural-network forward pass instead of iterating Baum-Welch EM, for large single-molecule FRET datasets (draft module + POC landed; approximate, opt-in, no pretrained model shipped).

# Master implementation order (all PRDs)

Single authoritative sequence for the MFDB/chisurf overhaul: the original feature
PRDs (03–10), the operation/transformer spine + LIMS layers (11–16), and the
architecture track (17–27). Ordered by dependency → de-risking → value. Phases
ship independently; cross-cutting PRDs interleave.

> **Two foundational decisions are pulled to the front** because everything built
> after them must be built *on* them, or it gets rebuilt:
> - **PRD-19** — collapse to one canonical dictionary-driven schema (folds ideas
>   I + K); flrCIF stays authoritative and the `.dic` extends it; declarative
>   reconcile replaces the version chain. Decide before PRD-11 adds many new
>   definitions.
> - **PRD-27** — make the provenance/state core append-only (idea M). Decide before
>   PRD-12 (lifecycle) and PRD-21 (events), which then become *projections* over it
>   rather than mutable status + a parallel event store.
> Deciding both early is the single biggest lever for "minimize reimplementation."

## Phase 0 — Current (finish first)

- **PRD-03** result registry, **PRD-04** burst pipeline — *we are here (3/4)*.
- PRD-09/10 (shifter, dataset browser) were added on this line and are largely
  built.

## Phase 1 — Architecture foundation (de-risk before building more)

These make everything after cheaper and safer; do them right after 03/04.

1. **PRD-18** Dependency injection + **hermetic test harness** — land the
   autouse temp-DB fixture *immediately* (tests currently touch the real DB), then
   the DI/`MFDBClient.call` contract.
2. **PRD-17** Canonical identity/session context — kills the owner-mismatch class.
3. **PRD-19** Single canonical dictionary-driven schema (folds **I**+**K**) —
   flrCIF authoritative (extended via the `.dic`); delete legacy `fdb_*` and the
   `mfdb_*` duplicates of flrCIF concepts; vocab from the `.dic`; versionless
   declarative `reconcile_schema`. Removes
   the vocab-drift class and the brittle 39-version chain *before* PRD-11 adds many
   new definitions. (Subsumes PRD-25 H3; fulfils PRD-02c by making flrCIF the single
   authoritative model.)
4. **PRD-27** Append-only provenance/state core (idea **M**) — go/no-go +
   append-only-lite now (recorded facts never mutated in place; transitions and
   supersessions are new rows; deletes are tombstones). Decide here so PRD-12/21 are
   built once as projections.
5. **PRD-25 H1/H2 + N1/N3** (uniform fail-loud; one RPC envelope; typed IDs;
   validate at the boundary) — reinforce PRD-18; do these bits here. Remaining
   H/N items interleave later.

## Phase 2 — Operation/transformer spine + model-driven layer

6. **PRD-11** operation-node abstraction + **PRD-16** transformer contract
   (ship together; **supersedes PRD-07**). Refactor Burst Selection (PRD-04) and
   Microtime Shifter (PRD-09) as the two reference conformant transformers, applying
   **PRD-23** (thin widgets) to those tools as you touch them.
   - **PRD-28** ndXplorer ↔ MFDB burst-selection round trip — a manual-test enabler
     that rides this spine (send a burst selection to ndXplorer; open one from MFDB via
     the dataset picker). Do once Burst Selection registers conformantly. The CLI
     handoff (`BS analyze --mfdb` → resolvable group artifact) is implemented and
     verified.
   - **PRD-31** ndXplorer headless CLI (parameter-based burst filtering + headless
     imaging) — completes the CLI leg of PRD-28; specified for separate
     implementation.
7. **PRD-26** model-driven data layer (idea **J**) — the `.dic` generates the
   DAO/repository, admin entity registry, RPC validation, and docs. Builds on
   PRD-19's generator + PRD-11's parameter schemas; subsumes most hand-maintained
   admin/validation work. Do alongside or just after PRD-11.

## Phase 3 — Provenance + LIMS process layer (projections over PRD-27)

8. **PRD-21** lineage API + event model — events feed the lifecycle; lineage feeds
   admin, browser, and PRD-05's impact query. *Projects/publishes over PRD-27's log.*
9. **PRD-12** lifecycle state machine (LIMS P1) — consumes PRD-21 events; the
   transition log *is* PRD-27's state projection.
10. **PRD-14** protocols (LIMS P3) — needs PRD-11 parameter schemas.
11. **PRD-13** study/project (LIMS P2) — independent; may run parallel with 9/10.

## Phase 4 — Composition + remaining features

12. **PRD-22** pipeline/workflow engine — needs the spine (11/16) + lineage (21);
    "what-if" reprocessing rides PRD-27 branches.
13. **PRD-15** reagent inventory (LIMS P4), **PRD-05 remainder** (gamma/crosstalk/
    R0 — g-factor done via PRD-04 C/D), **PRD-06** fluorophore DB, **PRD-08**
    optical configuration — independent; slot as needed.

## Phase 5 — Capstone

14. **PRD-24** extract MFDB into `modules/mfdb` — last, once PRD-17/18/19/26 have
    stabilized the interfaces (per `TODO.md`).

## Cross-cutting (interleave throughout)

- **PRD-23** thin widgets / view–api separation — apply per tool, starting with the
  transformers in Phase 2.
- **PRD-25** consistency hardening (+ idea **N**) — H1/H2 + N1/N3 in Phase 1; H3 is
  subsumed by PRD-19 (one canonical schema); N2 (units) lands with PRD-11 params;
  H4 (caching), H5 (N+1), H6 (dead-code, mostly subsumed by PRD-19) as convenient.
- **PRD-32** acquisition standard output folder — setup-defined default save path
  for new measurements; independent and shippable now.
- **PRD-33** acquisition-to-MFDB registration — sample-linked or new-sample
  measurement registration; depends on PRD-02/03 and stays separate from the
  file-output path.
- **PRD-34** BID saves to MFDB + downstream plugins ingest MFDB BIDs directly —
  when connected, a burst selection (BID) registers to MFDB (reference, not a loose
  file) and the burst tools open BIDs from the dataset picker. Generalizes PRD-28
  to all BID producers/consumers; rides PRD-03/11/16 + PRD-10.

## Dependency graph

```
PRD-03/04 (current)
   │
   ▼
Phase 1:  PRD-18 ─┬─ PRD-17 ─┬─ PRD-19 (I+K) ─┬─ PRD-27 (M)   (+ PRD-25 H1/H2 N1/N3)
                  │          │ canonical schema│ append-only core
                  ▼          ▼                 ▼
Phase 2:        PRD-11 ──► PRD-16 ──► PRD-26 (J)   (supersedes PRD-07; refactor 04+09; apply 23)
                  │        contract   model-driven layer (on 19+11)
   ┌──────────────┼───────────────┐
   ▼              ▼               ▼
PRD-21 ──► PRD-12        PRD-14        PRD-13        (Phase 3 — projections over PRD-27)
   │          │            (needs 11)   (independent)
   └────► PRD-22 (Phase 4; + spine; "what-if" on 27 branches)
                  PRD-15 / PRD-05* / PRD-06 / PRD-08  (independent features)
                  │
                  ▼
              PRD-24 (Phase 5, capstone)
```

## One-line rationale per architecture PRD

- **17** identity context → one current-user resolver (owner-mismatch class).
- **18** DI + hermetic tests → no test touches the real DB; injected db/session.
- **19** one canonical `.dic`-driven schema (I+K) → three-table-family + vocab-drift
  + brittle-migration classes gone; flrCIF authoritative + `.dic`-extended;
  versionless reconcile.
- **21** lineage API + events → queryable provenance, reactive workflows (projects
  over 27).
- **22** pipeline engine → composable transformer graph (chinet at data level).
- **23** thin widgets → no logic/side-effects in GUI; smoke-tested tools.
- **24** package extraction → clean `modules/mfdb` boundary (capstone).
- **25** hardening (+N) → fail-loud, one RPC envelope, typed IDs, units, boundary
  validation, caching, N+1, dead code.
- **26** model-driven layer (J) → `.dic` generates DAO/admin/validation/docs; the
  hand-maintained drift surface disappears.
- **27** append-only provenance/state core (M) → audit, reproducibility, "what-if"
  branches; PRD-12/21 become projections, not parallel stores.

## Recommended first action

Finish 03/04, then **Phase 1** in order: PRD-18 (test harness) → PRD-17 (identity) →
**PRD-19** (canonical schema) → **PRD-27** (append-only core). Those two structural
decisions (19, 27) are what make the PRD-11/16 spine and PRD-12/21/26 build once
instead of twice — the cheapest, highest-leverage de-risking before the spine.

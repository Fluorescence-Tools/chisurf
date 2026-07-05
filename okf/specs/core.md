---
type: Specification
title: Core Domain Layer — Target
description: The clean-architecture target for the scientific domain — objects, parameters, models, fitting, actions.
resource: chisurf/core/
tags: [target, core, domain, fitting, models]
timestamp: '2026-07-05T00:00:00Z'
---

> The clean-architecture target for the scientific domain. Current shape: [core subsystem](/subsystems/core.md). Current-state gaps: [assessment](assessment.md).

## Purpose

The core is the scientific model of ChiSurf: the data being analyzed, the
parameters that describe it, the models that predict it, and the fitting that
reconciles the two. It is the part that would still make sense if every UI,
server, and storage backend were replaced. It owns *what the science is*, not
*how it is shown, moved, or stored* — those belong to [Plugins](plugins.md),
[RPC & API](rpc.md), and [MFDB](mfdb.md) respectively.

## Design principles

- **Headless and pure.** The core computes and holds domain objects. It never
  imports the GUI or a Qt toolkit, and it does not reach out to a server,
  a database, or the filesystem to do its job. Given inputs, it produces outputs.
- **One vocabulary of objects.** Data, parameters, curves, models, and fits are a
  small, coherent set of abstractions with clear relationships — not a sprawl of
  overlapping types that do almost-the-same thing.
- **One way to identify and find things.** Every domain object has a single kind
  of stable identity, and there is one mechanism for looking objects up. Identity
  is by that identifier, never by mutable class names or object address.
- **State changes go through actions.** Mutations to domain state are expressed as
  named, replayable actions, so history, undo, scripting, and remoting all share
  one path rather than each poking objects directly.
- **Models are data-described.** A model declares its structure and its editable
  surface as data, so UIs and the metadata store can be generated from that
  declaration rather than hand-written per model.

## Target architecture

The core is a layered set of abstractions, each depending only on the ones below it:

| Layer | Responsibility |
|-------|----------------|
| **Objects** | A common base giving every domain object stable identity, structured children, and serialization. |
| **Parameters** | Named, bounded, optionally linked values with a dependency graph — the fitting degrees of freedom. |
| **Data & curves** | Measured/derived numeric datasets and their groupings — the things a model is fit against. |
| **Models** | Predictors that turn parameters into a curve, and declare their editable structure as data. |
| **Fitting** | The engine that adjusts parameters to fit a model to data: residuals, optimization, sampling, error analysis. |
| **Actions** | The named, replayable mutations that are the *only* sanctioned way to change domain state. |
| **Transforms / pipeline** | Composable operations that turn data into data, described declaratively. |
| **Project / persistence** | Save and restore a whole analysis as a self-contained project. |

Two flows define the layer:

- **Fitting flow.** A fit binds a model to a dataset over a range. The engine
  evaluates the model, forms weighted residuals against the data, and drives an
  optimizer over the free parameters; error analysis and sampling reuse that same
  evaluation. Parameter links let many fits share degrees of freedom — the
  "global" in global analysis.
- **Description flow.** A model describes its own parameters and editable
  structure as data. UIs render that description; the metadata store records it.
  Neither hand-codes knowledge of specific models. In the current tree this is
  realized by the [AutoForm framework](/subsystems/gui-autoform.md).

State ownership: the core defines the objects, but *authority* over which
datasets and fits currently exist belongs to the session layer reached through
the facade — not to process-global lists (see [runtime globals](/architecture/runtime-globals.md)).
The core is where the objects live; it is not where "the current session" lives.

## Rules

1. Core modules MUST NOT import the GUI or a Qt toolkit.
2. Core computation MUST NOT depend on a running server, database, or network.
3. Every domain object has exactly one form of stable identity; lookups use it.
   Identity MUST NOT depend on mutable class names or object address.
4. Domain state changes SHOULD be expressed as [actions](/architecture/action-layer.md),
   so one path serves history, scripting, and remoting.
5. Abstract base types MUST be non-instantiable — a missing override fails at
   construction, not deep inside a fit.
6. A model's editable structure is declared as data; UIs and metadata derive from
   that declaration rather than embedding per-model knowledge.
7. There is one owner of "current session" state, reached through the facade —
   not duplicated across process globals.

## Steering notes

Today the core carries a historic mess: several overlapping ways to track and
find instances, class-renaming that forces name-based type checks, abstract
methods that don't actually prevent instantiation, and process-global lists that
compete with the facade for ownership of session state. The target is one object
vocabulary, one identity mechanism, actions as the sole mutation path, and the
facade as the sole owner of session state. The concrete backlog is the `BUG-*`,
`INC-01`, `INC-02`, and `SV-03` findings in [assessment](assessment.md).

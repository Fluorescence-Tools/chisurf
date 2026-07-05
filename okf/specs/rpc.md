---
type: Specification
title: RPC & API Facade — Target
description: The clean-architecture target for the single boundary between UI and domain.
resource: chisurf/core/api/
tags: [target, api, facade, rpc, server]
timestamp: '2026-07-05T00:00:00Z'
---

> The clean-architecture target for the domain boundary. Current shape: [API facade](/architecture/api-facade.md) and [server](/architecture/server.md). Current-state gaps: [assessment](assessment.md).

## Purpose

This subsystem is the single boundary between everything that presents ChiSurf
(GUI, plugins, macros, CLI, console) and the domain that computes it. It has two
faces: a **facade** that presentation code calls, and a **headless service
layer** that owns the state and does the work. The facade may satisfy a call
in-process or route it to the service layer running in a separate process — and
the caller cannot tell which. It owns *how state is reached and moved*, not *what
the state means* ([Core](core.md)) or *how it persists* ([MFDB](mfdb.md)).

## Design principles

- **One door.** Presentation reaches the domain only through the facade. There is
  no informal second path around it.
- **Location transparency.** The same facade call behaves identically whether the
  service layer is in-process or remote. Callers never branch on "am I local".
- **Headless service layer.** The service side and everything it imports run
  without a GUI or Qt, so ChiSurf works as a terminal tool, a subprocess, or a
  server with no code change.
- **Data across the wire, not objects.** The boundary carries plain,
  serializable data with stable identifiers. Identity, type, and deep mutation do
  not survive the crossing, so the contract never assumes they do.
- **One transport, one protocol, one error shape.** Remote calls share a single
  request/response/error contract. Failures are reported one way, and callers
  distinguish "it failed" from "it succeeded" without guesswork.

## Target architecture

```text
presentation ─▶ ChiSurfAPI / PluginContext        the facade: the only door
                      │
                      ├─ in-process ─▶ service layer (direct call)
                      └─ remote ─────▶ transport ─▶ service layer
                                                       │ owns
                                              session + domain state
```

- **Facade** (`ChiSurfAPI`, `PluginContext`). A stable, verb-oriented surface —
  list/get/create/run/update datasets, fits, parameters, projects, sessions. It
  is what presentation and plugins hold. It selects local or remote execution
  internally; that choice is an implementation detail, not part of the contract.
- **Service layer.** Handlers grouped by domain noun (dataset, fit, parameter,
  project, session, model, graph). Each receives the session, performs the
  operation, and returns plain result data. The service layer is the **only**
  writer of domain state.
- **Session.** The authoritative container of what currently exists — the
  datasets, fits, and parameters of the live analysis. One owner, reached only
  through the service layer.
- **Transport.** A single request/response protocol plus an event channel for
  the service layer to announce state changes. Presentation subscribes; it does
  not poll.
- **Contract types.** The shapes crossing the boundary are defined once and are
  the single source of truth for both sides. Every entity carries a stable
  identifier; the far side refers to entities by that id, never by object
  identity.

## Rules

1. The service layer and its imports MUST NOT import the GUI or a Qt toolkit.
2. Presentation MUST reach domain state only through the facade — never by
   importing service internals or domain globals.
3. A facade call MUST behave the same in-process and remote (location transparency).
4. Everything crossing the boundary MUST be plain serializable data carrying a
   stable identifier; live domain objects MUST NOT cross.
5. Boundary shapes are defined once and used by both sides; local and remote
   results for the same call MUST have the same shape.
6. There is one transport and one protocol for remote calls, and one error shape.
   A caller MUST be able to tell success from failure by the contract, not by
   sniffing fields.
7. State-change events are declared alongside the methods that emit them; a
   method emits only declared events, and every declared event has an emitter.
8. Long-running work is uniform: one mechanism to start, observe, and cancel it.

## Steering notes

Today the boundary is uneven: the GUI leaks into the supposedly headless service
layer, the shared contract types exist but are unused so local and remote replies
have drifted apart, application errors hide inside "successful" responses, event
names disagree between emitter and declaration, and legacy flat method names sit
beside the namespaced ones. The target is a clean facade with location
transparency, one authoritative set of contract shapes, one error convention, and
a declared event catalogue. The backlog is the `SV-01`, `SV-02`, `SV-04`,
`SV-05`, `BUG-02`, `BUG-03`, `INC-03`, and `INC-08` findings in
[assessment](assessment.md).

---
type: Specification
title: ChiSurf — Target Architecture
description: The north-star whole-system shape, dependency direction, and shared architectural principles.
tags: [target, architecture, north-star, principles]
timestamp: '2026-07-05T00:00:00Z'
---

> The north star these specs steer toward. Where today's code diverges: [assessment](assessment.md).

ChiSurf is an interactive global-analysis platform for time-resolved and
single-molecule fluorescence data (TCSPC, FCS, smFRET). These specs describe the
**clean architecture the software should converge on** — a target, not a
description of the current tree. The current code is a historic mess; the gap
between it and this target is tracked in [assessment](assessment.md). For the
current-state layout, see the [architecture](/architecture/index.md) and
[subsystems](/subsystems/index.md) groups and `docs/architecture.md`.

## The shape

Four subsystems, each with a single clear responsibility and a spec:

| Spec | Subsystem | Owns |
|------|-----------|------|
| [Core](core.md) | Domain layer | The scientific objects: data, parameters, curves, fits, models, and the math that operates on them |
| [RPC & API](rpc.md) | Server + facade | The one boundary between UI and domain: a headless service layer and the `ChiSurfAPI` facade over it |
| [MFDB](mfdb.md) | Metadata store | Provenance and metadata — what was measured, how it was analyzed, and where results came from |
| [Plugins](plugins.md) | Extension system | How features are added: self-describing plugins that reach the domain only through the facade |

## How they fit together

One direction of dependency, one boundary to cross:

```text
UI · plugins · macros · CLI          (presentation — knows nothing about storage or transport)
        │  speaks only to ▼
   ChiSurfAPI  /  PluginContext       (the facade — the single door to the domain)
        │  which routes to ▼
   service layer (headless, Qt-free)  (owns session + domain state; the only writer)
        │  reads/writes ▼
   domain objects  ·  MFDB            (core scientific model + provenance store)
```

Presentation never touches domain state directly and never talks to storage or
transport. It goes through the facade. The facade routes to a headless service
layer that owns the state and is the only component allowed to mutate it. That
service layer can run in-process or in a separate process behind an RPC
transport — presentation cannot tell the difference, because it only ever holds
the facade.

## Architectural principles

These bind every subsystem. Each spec restates the ones it must uphold.

1. **One door to the domain.** All state access goes through the facade
   (`ChiSurfAPI` / `PluginContext`). There is no second, informal path.
2. **State has one owner.** The service/session layer owns domain state and is
   its only mutator. Everyone else holds references or copies, never authority.
3. **The core is headless.** The domain and service layers never import the GUI
   or a Qt toolkit, so the whole system runs in a terminal, a subprocess, or a
   server unchanged.
4. **The boundary carries data, not objects.** What crosses between presentation
   and domain is plain, serializable data with stable identifiers — never live
   Python objects whose identity or type the far side would depend on.
5. **One transport, one protocol.** Remote operation uses a single, documented
   request/response contract. No side channels.
6. **Metadata is authored once, upstream.** MFDB's schema derives from its
   dictionaries; provenance is recorded as an operation graph, not reconstructed
   after the fact.
7. **Features are plugins.** Anything beyond the core is a self-describing
   plugin that declares its contract and reaches the domain only through the
   facade.
8. **Convergence is additive.** The system moves toward this target in
   backward-compatible steps; nothing in flight breaks to get there.

## Using these specs

- Changing a subsystem? Read its spec's **Design principles** and **Rules** — they
  are the bar the change must clear.
- Adding something? The relevant spec says where the seams are.
- Cleaning up? [assessment](assessment.md) is the backlog of where the current
  code falls short of these targets, most severe first.

Each spec follows the short shared shape in [template](template.md): purpose,
design principles, target architecture, rules, steering notes.

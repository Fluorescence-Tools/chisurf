# ChiSurf Target Specifications

These concepts describe the **clean architecture ChiSurf should steer toward** —
a target, a north star. They are deliberately general and timeless: they say
where the software is going, not what the current tree looks like. The current
code is a historic mess; the gap between it and these targets is tracked
separately in the [assessment backlog](assessment.md).

The current-state companion is the rest of this bundle — the
[architecture](/architecture/index.md) and [subsystems](/subsystems/index.md)
groups describe how the code is organized *today*; the specs below describe what
good looks like. Read [overview](overview.md) first for the whole-system shape.

# Specifications

* [Target Architecture (Overview)](overview.md) - The whole-system shape, dependency direction, and the architectural principles every subsystem shares.
* [Core Domain Layer](core.md) - The scientific objects and the math that operates on them.
* [RPC & API Facade](rpc.md) - The single boundary between UI and domain.
* [MFDB — Metadata & Provenance](mfdb.md) - Provenance and metadata as the target sees it.
* [Plugin System](plugins.md) - How features are packaged and integrated.

# Backlog

* [Assessment — Cleanup Backlog](assessment.md) - Concrete, verified findings where today's code diverges from the targets, most severe first.

# Authoring

* [Spec Template](template.md) - The shared shape every target spec follows.

# Two documents, two jobs

- **The specs** say what good looks like. They stay stable as the code is
  cleaned up — a target does not move every time you take a step toward it.
- **[The assessment](assessment.md)** is the backlog: concrete, verified findings
  where the current code falls short of the specs, most severe first. It shrinks
  as the code converges. When it is empty for a subsystem, that subsystem has
  reached its spec.

Each spec's **Steering notes** point at the assessment findings relevant to it,
so you can go from "what should this be" to "what's wrong today" in one hop.

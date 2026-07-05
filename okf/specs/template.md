---
type: Template
title: Spec Template
description: The shared shape every ChiSurf target specification follows.
tags: [template, authoring, specs]
timestamp: '2026-07-05T00:00:00Z'
---

Every target spec in this group describes the **target architecture**: where the
software should steer, a clean design to converge on. It is **not** a description
of the current code — that lives in the [architecture](/architecture/index.md)
and [subsystems](/subsystems/index.md) groups, and the current-state gaps live in
the [assessment](assessment.md).

## Rules of thumb

- Write the north star, not the inventory. Describe the desired abstractions and
  boundaries in general terms. Avoid `file:line` citations and current-code
  tours — that is what the [assessment](assessment.md) is for.
- Keep it short. One screenful of principles beats ten screens of contract detail.
- Timeless voice: "X owns Y", "Z flows one way". A reader should not be able to
  tell how messy today's code is from reading the spec.
- The gap between this target and today's code lives in the
  [assessment](assessment.md) — link there, don't inline it.

## The shape

Every `specs/*.md` target spec follows this section order:

- **> blockquote** — one line: the clean-architecture target for `<subsystem>`,
  with links to the current-state concept and the [assessment](assessment.md).
- **Purpose** — one paragraph: what this subsystem is for, and where its
  responsibility ends (what belongs to a sibling).
- **Design principles** — the 3–6 north stars for this area; the ideas every
  future change should reinforce.
- **Target architecture** — the desired shape: the key abstractions, who owns
  what state, and how data/control flows across the boundary. General enough to
  survive refactors. Small diagrams or a short table are fine.
- **Rules** — a short list of invariants the design upholds. Plain sentences,
  testable, no ceremony.
- **Steering notes** — the direction of travel from today toward this target, in
  a few sentences. Point at the relevant [assessment](assessment.md) findings
  rather than restating them.

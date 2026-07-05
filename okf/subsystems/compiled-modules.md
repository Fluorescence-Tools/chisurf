---
type: Subsystem
title: Compiled Modules
description: C++ extensions in modules/ that must be built before tests.
resource: modules/
tags: [cpp, extensions, build, swig]
timestamp: '2026-07-05T00:00:00Z'
---

# Extensions

The compiled extensions in `modules/` — `chinet`, `ndxplorer`, `clsmview`,
`quest` — plus the burbulator C++ library must be built before the test
suites run. `ndxplorer` and `quest` are git submodules (see `.gitmodules`).

# Building

Run `pixi run build-extensions` if imports of those modules fail. The `test*`
[pixi tasks](/workflows/build-and-env.md) already `depends-on`
`build-extensions`, so a plain `pixi run test` builds them first.

The related `tttrlib` package (used for photon/TTTR data) is an editable
install that occasionally needs a manual rebuild after C++/`.i` changes.

# Citations

[1] [ChiSurf architecture doc](/references/architecture-doc.md)

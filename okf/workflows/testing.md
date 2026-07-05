---
type: Playbook
title: Testing
description: The non-GUI, GUI, smoke, and doctest suites and how to run a single test.
resource: pixi.toml
tags: [testing, pytest, ci]
timestamp: '2026-07-05T00:00:00Z'
---

# Test tasks

```bash
pixi run test           # non-GUI test suite (build-extensions first)
pixi run test-gui       # GUI/widget tests (-k 'widget or gui')
pixi run test-smoke     # fast smoke test (test/test_basic.py)
pixi run test-doctest   # doctests
```

All `test*` tasks `depends-on` `build-extensions`, so the
[compiled modules](/subsystems/compiled-modules.md) are built first.

# Single test

Run one test directly with pytest, e.g.

```bash
pytest test/test_basic.py::test_name -q
```

The `slow` marker is excluded from default runs. Tests live in `test/`, in
per-plugin `**/test/` directories, and in
`chisurf/gui/widgets/node_editor/tests`.

# Feature testing

Every feature should have a headless test path (API/CLI), not GUI-only.
Model/UI changes have a dedicated headless check via the `test-model-editor`
skill.

# Citations

[1] [Project instructions (CLAUDE.md)](/references/claude-md.md)

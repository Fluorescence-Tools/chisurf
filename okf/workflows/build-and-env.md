---
type: Playbook
title: Environment & Build
description: Pixi is the canonical environment/build manager; build compiled extensions before tests.
resource: pixi.toml
tags: [pixi, build, environment]
timestamp: '2026-07-05T00:00:00Z'
---

# Environment

Pixi is the canonical environment/build manager (CI uses it). Run everything
through `pixi run <task>`. Python must run inside the project environment —
never the conda `base` env — because the Qt stack and the
[compiled extensions](/subsystems/compiled-modules.md) are not present there.
Python is pinned to 3.12.

# Common commands

```bash
pixi run chisurf            # launch the GUI (== python -m chisurf)
pixi run build-extensions   # build modules/ C++ extensions
pixi run lint               # ruff check + ruff format --check
pixi run fmt                # ruff format + ruff check --fix
pixi run typecheck          # mypy chisurf/
```

# Style

ruff (line length 100, py310 target, NumPy-style docstrings) and mypy.

# Citations

[1] [Project instructions (CLAUDE.md)](/references/claude-md.md)

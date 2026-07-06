---
type: PRD
prd: "46"
title: "PRD-46: Scripts as first-class citizens in the test pipeline"
description: Brings shipped example scripts into the automated test suite by running headless/process scripts under a shebang-driven runner with numeric assertions, plus unit tests for the public model API they exercise.
status: planned
phase: "unassigned"
resource: test/scripts/
tags: [prd, core]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-46 brings ChiSurf's runnable example scripts, which exercise the core model API end-to-end but sit outside CI, into the automated test suite so regressions in the public model API are caught automatically. A parametrized pytest runner discovers `scripts/*.py` and runs each according to its shebang: process scripts as subprocesses with stdout and numeric-output assertions, and console/ipython scripts headlessly by exec against a thin `cs` namespace stub that provides the real core but no GUI. It also adds unit tests for the public model API surface (`chain_length`, `persistence_length`, mixture `fractions` setter) and wires a `test-scripts` task into the default test suite. Scripts stay runnable interactively without modification — the harness is a thin wrapper.

# Status
Planned (unassigned phase, STATUS TABLE authoritative). Runner design, headless stub, numeric assertions, unit tests, and CI integration with acceptance criteria are specified.

# Problem
ChiSurf ships runnable example scripts (e.g. `examples/scripts/protein_unfolding_fret_line.py`, `examples/scripts/protein_unfolding_gui.py`) that exercise the core model API end-to-end. They are currently **outside the automated test suite**: no CI job runs them, no assertion checks their output, and regressions in the public model API go undetected until a user manually runs a script and notices something is wrong.

# Goals
1. **Every script in `scripts/` is tested automatically** on each CI run.
2. **Headless (process) scripts** are run as subprocesses; their stdout is captured and key numeric outputs are asserted.
3. **GUI/IPython scripts** (shebang `# !chisurf: console` or `# !chisurf: ipython`) are exercised headlessly by importing and calling their logic through the public model API — no display required.
4. **Public model API surface** exposed by PRD-46 work (`chain_length`, `persistence_length`, `fractions` setter, etc.) is covered by unit tests.
5. Scripts remain runnable interactively without modification — the test harness is a thin wrapper, not an invasive change to the scripts themselves.

# Non-goals
- Running Qt event loops in CI (handled separately by `pixi run test-gui`).
- Full integration tests of the Code Editor toolbar (those belong in `test-gui`).

# Design

## 1. Script test runner — `test/scripts/test_scripts.py`
A pytest file that discovers all `scripts/*.py` files and runs each one according to its shebang:

```
# !chisurf: process   → subprocess test (assert exit 0, assert stdout patterns)
# !chisurf: console   → import-and-exec test (headless, cs namespace stub)
# !chisurf: ipython   → headless exec test (same as console)
# (no shebang)        → subprocess test (default)
```

Parametrized via `pytest.mark.parametrize` so each script is a separate test node in the output.

## 2. Headless execution of `console`/`ipython` scripts
A thin `cs` namespace stub is injected before exec so GUI scripts can be run without a display:

```python
# test/scripts/conftest.py
import chisurf as cs_real

class HeadlessStub:
    """Minimal cs stub: real core, no GUI."""
    core       = cs_real.core
    experiment = {}           # empty — add_fit will skip GUI creation
    imported_datasets = []
    fits = []
    macros = cs_real.macros
    gui = None

    def __getattr__(self, name):
        return getattr(cs_real, name)
```

Scripts that call `cs.macros.core_fit.add_fit(...)` will proceed through the core path (`FitGroup` creation, parameter setup) but skip all Qt widget construction (the macro already guards with `if gui is not None`).

## 3. Numeric assertions for `process` scripts
`protein_unfolding_fret_line.py` writes four text files. The test reads them back and asserts:

- `unfolding_fret_line.txt`: E at f=0 > 0.85 (folded, high FRET); E at f=1 < 0.70 (WLC unfolded, lower FRET); E is monotonically decreasing.
- `wlc_sweep_fret_lines.txt`: larger Lc → lower E at f=1 (longer chain = lower FRET).
- Files are not empty; header lines are present.

## 4. Unit tests for the new public model API
`test/models/test_wlc_public_api.py`:

```python
def test_chain_length_property():
    fit = Fit(model_class=WormLikeChainModel, data=dummy())
    fit.model.chain_length = 80.0
    assert fit.model.chain_length == 80.0

def test_persistence_length_property():
    fit = Fit(model_class=WormLikeChainModel, data=dummy())
    fit.model.persistence_length = 60.0
    assert fit.model.persistence_length == 60.0

def test_mixture_fractions_setter():
    ...
    mm.fractions = [0.3, 0.7]
    np.testing.assert_allclose(mm.fractions, [0.3, 0.7])
```

## 5. CI integration
Add to `pixi.toml`:

```toml
[tasks.test-scripts]
cmd = "pytest test/scripts/ -p no:cov -o addopts='' -q"
depends-on = ["build-extensions"]
```

Add `test-scripts` as a dependency of the top-level `test` task so it runs in the default suite.

# File layout
```
test/
  scripts/
    conftest.py          # HeadlessStub, tmp output dir fixture
    test_scripts.py      # parametrized runner for scripts/*.py
  models/
    test_wlc_public_api.py   # unit tests for new public properties
```

# Acceptance criteria
- `pixi run test-scripts` passes with zero failures on a clean checkout.
- Adding a new script to `scripts/` with a shebang automatically adds a test node — no manual registration needed.
- `WormLikeChainModel.chain_length`, `.persistence_length` and `LifetimeMixtureModel.fractions` setter are covered by at least one passing unit test each.
- CI (`pixi run test`) is green after PRD-46 work lands.

# Relationships
- Adds a headless test path for scripts, complementing the separate GUI test task.
- Covers the public model API surface exposed through the model work related to [PRD-38](prd-38.md)/[PRD-40](prd-40.md).
- Reinforces the [Core target](/specs/core.md) by asserting the stable model API.

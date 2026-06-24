# Burst Selection — Agent Instruction Files

This directory contains granular instruction files for an autonomous agent to
complete the remaining Burst Selection work. Each file is self-contained and
can be executed independently.

## Prerequisite reading

1. `STATUS.md` — current codebase state
2. `PRD.md` — product requirements and target architecture
3. `NEW_GUI_MIGRATION.md` — stepwise migration plan
4. `REFERENCE_IMPLEMENTATION.md` — reference workflow/MFDB plugin pattern

## Instruction files (execute in order)

| Priority | File | Description | Est. time |
|----------|------|-------------|-----------|
| **P0** | `INSTRUCTION_HDF5_ZIP_TESTS.md` | Tests for the newly implemented `write_hdf5` and `zip_output_folder` API functions | 30 min |
| **P0** | `INSTRUCTION_GUI_VERIFICATION.md` | Desktop validation of migrated GUI vs legacy | 1-2 hr |
| **P1** | `INSTRUCTION_TTTRLIB_BURSTFILTER.md` | Integrate `tttrlib.BurstFilter` behind the API | 4-8 hr |
| **P2** | `INSTRUCTION_WEBUI.md` | Electron/WebUI planning & prototype | weeks |

## How to use

```bash
# Read the current state first
cat chisurf/plugins/burst/burst_selection/docs/STATUS.md
cat chisurf/plugins/burst/burst_selection/docs/PRD.md

# Pick an instruction file and follow it:
cat chisurf/plugins/burst/burst_selection/docs/INSTRUCTION_HDF5_ZIP_TESTS.md

# Run tests after each step:
cd /Users/tpeulen/dev/chisurf
/Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest chisurf/plugins/burst/burst_selection/tests -q --no-cov
```

## Environment

- **Conda env:** `arm64` (`/Users/tpeulen/mambaforge/envs/arm64/bin/python3`)
- **Working dir:** `/Users/tpeulen/dev/chisurf`
- **Ruff:** `/Users/tpeulen/mambaforge/bin/ruff check`
- **Test target:** `chisurf/plugins/burst/burst_selection/tests`

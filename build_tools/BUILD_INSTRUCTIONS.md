# Creating installation files

## Prerequisites

- [Pixi](https://pixi.sh/) - install via:

  **Linux/macOS:**
  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```

  **Windows (PowerShell):**
  ```powershell
  Invoke-WebRequest -Uri "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.zip" -OutFile "pixi.zip"
  Expand-Archive -Path "pixi.zip" -DestinationPath "$env:LOCALAPPDATA\pixi" -Force
  Remove-Item "pixi.zip"
  # Add to PATH: $env:PATH = "$env:LOCALAPPDATA\pixi;$env:PATH"
  ```

  Or via pip (limited functionality):
  ```bash
  pip install pixi
  ```

## Quick Start

```bash
pixi run build-pkg         # 1. Build the chisurf conda package (rattler-build)
pixi run build-installer   # 2. Build the native installer for THIS OS
```

`build-installer` auto-detects the platform and produces:

| OS      | Artifact (in `dist/`)          | Wrapper        |
|---------|--------------------------------|----------------|
| Linux   | `ChiSurf-<ver>-x86_64.AppImage`| linuxdeploy    |
| macOS   | `ChiSurf-<ver>.dmg`            | hdiutil (.app) |
| Windows | `ChiSurf-Windows-Setup-<ver>.exe` | Inno Setup  |

## How it works

A single orchestrator, `build_tools/build_installer.py`, runs one shared
pipeline on every platform; only the final wrap differs:

1. Build (or, with `--no-build`, locate) the `chisurf-*.conda` in `conda-bld/`.
2. `make_runtime()` — `micromamba create` a self-contained env *from that
   package*, then pip-install the extras the recipe does not bundle:
   `labellib`, `latexify-py`, `imp-tricks` (from `modules/imp-tricks` if present,
   else cloned from GitLab), the local `modules/*` (chinet/clsmview/ndxplorer/
   quest), and `tttrlib` (conda on mac/linux, pip on Windows).
3. `strip_bloat()` — slim the env: drop unused Qt5 modules, remove build tools,
   dependency test suites, headers, `pip`/`wheel`, `__pycache__`; trim ChiSurf's
   own test data, bundled examples and structural-potential source artifacts; and
   gzip the bundled mmCIF dictionaries (`.dic` → `.dic.gz`, read transparently at
   runtime). Runs in an isolated child process, so a failure only warns.
4. Wrap into the platform installer.

Useful flags: `--no-build` (reuse an existing conda package), `--audit` (print
the largest dirs in the runtime env), `--platform {linux,macos,windows}`.

### Requirements
- **All:** `micromamba` and `rattler-build` on PATH (provided by the `build`
  pixi feature / the CI `build` env).
- **Windows:** the Inno Setup compiler `ISCC.exe` on PATH (`choco install innosetup`).
- **Linux:** `linuxdeploy` is downloaded automatically on first run.

### Versioning
- Override with `CHISURF_VERSION` (PEP 440). If unset, the version is derived
  from git tags by `rattler-recipe/generate_version.py` (falls back to `YY.devN`).
- The build freezes the resolved version into `chisurf/core/_version.py` (via
  `setup.py`), so the installed app reads a static string instead of spawning
  git on every `import chisurf`. Editable/`develop` installs skip this and stay
  git-derived (see `chisurf/core/info.py`).

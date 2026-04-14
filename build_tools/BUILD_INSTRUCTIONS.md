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
pixi run build-pkg        # Build conda package
pixi run build-setup      # Build Windows installer
pixi run build-osx-app   # Build macOS app
pixi run build-appimage  # Build Linux AppImage
```

## osx

### Creating a ChiSurf.app

A distributable dmg file (including the .app) can be built using Pixi:

```bash
pixi run build-osx-app
```

Or directly via the shell script:

```bash
./build_tools/osx/build-osx-app.sh -i=../chisurf/gui/resources/icons/cs_logo.png -n=ChiSurf -m=chisurf -p=.. -o=../dist
```

This creates a new environment and installs necessary dependencies. The
environment is placed in a ChiSurf.app together with the `chisurf` folder
from the project directory. The `chisurf` module is installed using
`--use-local`. The compiled binary is used as an entry point for the
ChiSurf.app. Unnecessary folders and files listed in `remove_list.txt` are
stripped from the ChiSurf.app folder. Finally, the ChiSurf.app is bundled
in a .dmg image that is placed in the `dist` folder.

## Windows

The Windows installation of ChiSurf is effectively a runtime environment with
an installed ChiSurf package. The ChiSurf package is built with
`rattler-build`.

Windows versions are bundled in setup.exe files created using Inno Setup. The
setup files will install a runtime environment that is used to run the chisurf
module. A setup file is created by calling

```cmd
pixi run build-setup
```

Or directly:

```cmd
build_tools\win\build-setup.bat
```

The script will create a new environment in `dist/win` using a conda-style
package manager (`micromamba`, `mamba`, or `conda`) if one is available on
`PATH`.
Next, a package of `chisurf` is built using the `rattler-recipe`
located in the folder `rattler-recipe` of the project root. The `chisurf` package
is installed to the environment in `dist/win`. Next, using `jinja2`, the
file `setup_template.jinja2` is written to the file `installer_config.iss` using
`create_installer_script.py`. The script `create_installer_script.py` will read details from
`pyproject.toml` and `chisurf/info.py` (version number, entry points, etc.).
Finally, Inno Setup reads `installer_config.iss` and writes an installation file
`chisurf_windows_setup_version.exe` to `dist/`.

> **Note:** The helper automatically bootstraps the Inno Setup compiler into your
> `%LOCALAPPDATA%\Programs\Inno Setup 6` folder when it is not already
> installed. No global admin installation or Chocolatey dependency is required.

### Versioning

- The recommended build-time override is `CHISURF_VERSION` (PEP 440 compatible).
- If unset, the recipe falls back to a dev-style version `YY.dev0`.

## Linux

### AppImage

Uses linuxdeploy to build AppImage:

```bash
pixi run build-appimage
```

Or directly via the script:

```bash
./build_tools/linuxdeploy/build.sh
```

Modify the `linuxdeploy-plugin-conda.sh` script if necessary (adjust Python/Conda version).

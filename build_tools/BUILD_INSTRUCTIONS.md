# Creating installation files 

## osx

### Creating a ChiSurf.app

A distributable dmg file (including the .app) can be built using 

```bash
build-dmg
```

in the build_tools/osx folder. The command will create a new
environment and install the necessary dependencies. The environment is placed in a ChiSurf.app together with the
`chisurf` folder located in the project directory. The `chisurf` module is installed using `--use-local`. The compiled binary is used as an entry point
for the ChiSurf.app. Unnecessary folders and files listed in `remove_list.txt` 
are stripped from the ChiSurf.app folder. Finally, the ChiSurf.app is bundled 
in a .dmg image that is placed in the ``dist`` folder. 

```bash
./osx/build-osx-app.sh --python=3.10 -i=../chisurf/gui/resources/icons/cs_logo.png -n=ChiSurf -m=chisurf -p=.. -o=../dist
```

## Windows

The Windows installation of ChiSurf is effectively a runtime environment with
an installed ChiSurf package. The ChiSurf package is built with
`rattler-build` via Pixi.

Windows versions are bundled in setup.exe files created using Inno Setup. The
setup files will install a runtime environment that is used to run the chisurf
module. A setup file is created by calling

```cmd
pixi run -e build build-setup
```

The script will create a new environment in `dist/win` using Pixi. 
Next, a package of `chisurf` is built using the `rattler-recipe`
located in the folder `rattler-recipe` of the project root. The `chisurf` package
is installed to the environment in `dist/win`. Next, using `jinja2`, the 
file `setup_template.jinja2` is written to the file `installer_config.iss` using
`create_installer_script.py`. The script `create_installer_script.py` will read details from 
`pyproject.toml` and `chisurf/info.py` (version number, entry points, etc.).
Finally, Inno Setup reads `installer_config.iss` and writes an installation file
`chisurf_windows_setup_version.exe` to `dist/`.

### Versioning

- The recommended build-time override is `CHISURF_VERSION` (PEP 440 compatible).
- If unset, the recipe falls back to a dev-style version `YY.dev0`.

## Linux

### Flatpak

Build with recipe with
```bash
flatpak-builder --force-clean --install-deps-from=flathub --repo=repo --user --install builddir xyz.peulen.ChiSurf.yml
```

Build bundle with
```bash
flatpak-builder --force-clean --install-deps-from=flathub --repo=repo --user --install builddir xyz.peulen.ChiSurf.yml
```

TODOs: Check how to deploy on Flatpak Hub.

### AppImage

Uses linuxdeploy to build AppImage. Execute `build.sh` in the linuxdeploy folder. 
Modify the `linuxdeploy-plugin-conda.sh` script if necessary (adjust Python/Conda version).
Currently (24.10.08) the build does not work on the latest Miniconda and default/unmodified  
`linuxdeploy-plugin-conda.sh` script.


# Creating installation files 

## osx

### Creating a ChiSurf.app

A distributable dmg file (including the .app) can be built using 

```bash
build-dmg
```

in the build_tools/osx folder. The command will create a new conda
environment and install the necessary dependencies. The conda environment is placed in a ChiSurf.app together with the
`chisurf` folder located in the project directory. The `chisurf` module is installed using `--use-local`. The compiled binary is used as an entry point
for the ChiSurf.app. Unnecessary folders and files listed in `remove_list.txt` 
are stripped from the ChiSurf.app folder. Finally, the ChiSurf.app is bundled 
in a .dmg image that is placed in the ``dist`` folder. 

```bash
./osx/build-osx-app.sh --python=3.10 -i=../chisurf/gui/resources/icons/cs_logo.png -n=ChiSurf -m=chisurf -p=.. -o=../dist
```

## Windows

The Windows installation of ChiSurf is effectively a conda environment with
an installed ChiSurf conda package. The ChiSurf conda package is built with
``conda build``.

Windows versions are bundled in setup.exe files created using Inno Setup. The
setup files will install a conda environment that is used to run the chisurf
module. A setup file is created by calling

```cmd
build-setup.bat
```

The script will create a new conda environment in `dist/win` for a compatible Python version. 
Next, a conda package of `chisurf` is built using the `conda-recipe`
located in the folder `conda-recipe` of the project root. The `chisurf` package
is installed to the conda environment in `dist/win`. Next, using `jinja2`, the 
file `setup_template.iss` is written to the file `setup.iss` using
`make_inno_setup.py`. The script `make_inno_setup.py` will read details from 
the setup file `setup.py`, i.e., the version number, the entry points, etc. 
Finally, Inno Setup reads `setup.iss` and writes an installation file
`setup_version_number.exe` to `dist/`.

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


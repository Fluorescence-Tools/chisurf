# Creating installation files 

## osx

### Creating a ChiSurf.app

A distributable dmg file (including the .app) can be build using 

```bash
build-dmg
```

in the build_tools/osx folder. The command will create a new conda
environment using `enviroment.yml`  in the project's root directory as a
template. The conda environment is placed in a ChiSurf.app together with the
`chisurf` folder located in the project directory. In the chisurf module is
compiled to a binary using nuitka. The compiled binary is used as entry point
for the ChiSurf.app. Unnecessary folders and file listed in `remove_list.txt` 
are stripped from the ChiSurf.app folder. Finally, the ChiSurf.app is bundeled 
in a .dmg image that is placed in the ``dist`` folder. 

```bash
./osx/build-osx-app.sh -f=../env_osx.yml -i=../chisurf/gui/resources/icons/cs_logo.png -n=ChiSurf -m=chisurf -p=.. -o=../dist
```

## Windows

The Windows installation of ChiSurf is effectively a conda environment with
an installed ChiSurf conda package. The ChiSurf conda package is build with
``conda build``.

Windows versions are bundled in setup.exe files created using innosetup. The
setup files will install a conda environment that is used to run the chisurf
module. A setup file is created by calling

```cmd
build-setup.bat
```

The script will create a new conda environment in the `dist/win` for 
python=3.7.3. Next, a conda package of chisurf is build using the conde-recipe
located in the folder `conda-recipe` of the project root. The `chisurf` package
is installed to the conda environemnt in `dist/win`. Next, using jinja2 the 
file `setup_tample.iss` is written to the file `setup.iss` using
`make_inno_setup.py`. The script `make_inno_setup.py` will read details from 
the setup file `setup.py`, i.e., the version number, the entry points, etc. 
Finnaly, innosetup reads `setup.iss` and writes a installation file
`setup_version_number.exe` to `dist/`.

## Linux

### Flatpak

Build with recipe with
```bash
flatpak-builder --force-clean --install-deps-from=flathub --repo=repo --user --install builddir xyz.peulen.ChiSurf.yml
```

build bundle with
```bash
flatpak-builder --force-clean --install-deps-from=flathub --repo=repo --user --install builddir xyz.peulen.ChiSurf.yml
```

TODOs: Check how to deploy on flatpak hub.

### Appimage

Uses linuxdeploy to build app image execute `build.sh` in linuxdeploy folder. 
Maybe modify the `linuxdeploy-plugin-conda.sh` script (adjust python/conda version).
Currently (24.10.08) build does not work on latest miniconda and default/unmodified  
`linuxdeploy-plugin-conda.sh` script.

TODOs:
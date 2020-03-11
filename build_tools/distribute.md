# Creating installation files 

## osx

### Creating a ChiSurf.app

An distributable dmg file (including the .app) can be build using 

```
create-dist-dmg
```

in the build_tools/osx folder. The command will create a new conda
environment using `enviroment.yml`  in the project's root directory as a
template. The conda environment is placed in a ChiSurf.app together with the
`chisurf` folder located in the project directory. In the chisurf module is
compiled to a binary using nuitka. The compiled binary is used as entry point
for the ChiSurf.app. Unnecessary folders and file listed in `remove_list.txt` 
are stripped from the ChiSurf.app folder. Finally, the ChiSurf.app is bundeled 
in a .dmg image that is placed in the ``dist`` folder. 

## Windows

Windows versions are bundled in setup.exe files created using innosetup. The
setup files will install a conda environment that is used to run the chisurf
module.
 
see:

http://www.entropyreduction.al/python/distutils/2017/09/21/bundle-python-app-w-inno-setup.html

## Linux

### snap store

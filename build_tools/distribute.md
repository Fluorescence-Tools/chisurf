# Creating installation files 

## osx

### Creating a ChiSurf.app

To distribute chisurf on osx a new conda environment is created in a template
for a macOS app. The template for the macOS app is located in `build_tools`
under the name ``ChiSurf.app``. 

In macOS apps are folders with the file suffix `.app`. In the ``ChiSurf.app
`` folder there is a file `ChiSurf.app/Contents/MacOS/ChiSurf` that is  
a bash script that sets up the environment and is executed when the use
 double clicks on the ``ChiSurf.app``:

```bash
#!/usr/bin/env bash
script_dir=$(dirname "$(dirname "$0")")
cd ./$script_dir/Resources/bin
export PYTHONPATH="$PWD"
../bin/python chisurf/gui.py $@
```

Additionally, there is a `Info.plist` file containing some additional
 information
 
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleDisplayName</key>
	<string>ChiSurf</string>
	<key>CFBundleExecutable</key>
	<string>ChiSurf</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>181208</string>
	<key>CFBundleName</key>
	<string>ChiSurf</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>181208</string>
	<key>CFBundleSignature</key>
	<string>????</string>
	<key>CFBundleVersion</key>
	<string>181208</string>
	<key>LSMinimumSystemVersion</key>
	<string>10.11.0</string>
	<key>LSUIElement</key>
	<false/>
	<key>NSAppTransportSecurity</key>
	<dict>
		<key>NSAllowsArbitraryLoads</key>
		<true/>
	</dict>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>NSHumanReadableCopyright</key>
	<string>© 2018 Thomas-Otavio Peulen</string>
	<key>NSMainNibFile</key>
	<string>MainMenu</string>
	<key>NSPrincipalClass</key>
	<string>NSApplication</string>
</dict>
</plist>
```

To create a new ``ChiSurf.app`` run the script `build_app.sh` in the
 `build_tools` folder
 
```bash
#!/usr/bin/env bash

script_dir=$(dirname "$(dirname "$0")")

# The .app will be placed in the dist folder. The template
# is copied and a new conda environemnt is created using
# the .app folder as a target
mkdir ../dist
cp -R ChiSurf.app ../dist
conda env create -f ../environment.yml --prefix ../dist/ChiSurf.app/Contents/Resources/ --force

# We will simpliy copy over the directory containing the code
# Hence make sure that all necessary extensions are up to data
cd ..
python setup.py build_ext --inplace --force
cd $script_dir
cp -R ../chisurf ../dist/ChiSurf.app/Contents/Resources/lib/python3.7/site-packages/chisurf

# update the Info.plist file
conda activate ../dist/ChiSurf.app/Contents/Resources/
./create_app_plist.py --module chisurf --template ./ChiSurf.app/Contents/Info.plist --output ../dist/ChiSurf.app/Contents/Info.plist -e ChiSurf
```

The script copies the ``ChiSurf.app`` template to the dist folder and updates
the `Info.plist` file with the information in the `chisurf` module.


### Using create_app_plist

macOS apps require a `Info.plist` file. Using the script `create_app_plist` such
a plist file can be create via the command line

```bash
./create_app_plist.py --module chisurf --template ./ChiSurf.app/Contents/Info.plist --output ../dist/ChiSurf.app/Contents/Info.plist -e ChiSurf
```

The script will take a python module specified by the parameter module and 
create a 

The script automates the process of making a new ChiSurf osx application. When
started from build_tools folder. 
```bash
./build-osx-app.sh -f=../environment.yml -i=../chisurf/gui/resources/icons/cs_logo.png -n=ChiSurf -m=chisurf -p=..
```

currently there is an issue (2020.3.10) with the generated app. The app can only
be started from a terminal windows

```bash
open -a ChiSurf.app
```

An distributable dmg file (including the .app) can be build using 

```
create-dist-dmg
```
in the build_tools/osx folder


## Windows

## Linux

### snap store

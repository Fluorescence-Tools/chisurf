wget -c "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
chmod a+x linuxdeploy-x86_64.AppImage

cat > chisurf.desktop <<\EOF
[Desktop Entry]
Version=1.0
Name=ChiSurf
Name[de]=ChiSurf
Comment=Time-resolved fluorescence analysis
Comment[de]=Zeitaufgeloeste Fluoreszenzanalyse
Exec=chisurf %F
Terminal=false
Type=Application
Icon=chisurf-logo
Categories=Science;Engineering;
StartupNotify=true
EOF

export CONDA_CHANNELS=tpeulen CONDA_PACKAGES=chisurf ARCH=x86_64
./linuxdeploy-x86_64.AppImage --appdir AppDir -d chisurf.desktop --plugin conda -i chisurf-logo.svg  --output appimage 

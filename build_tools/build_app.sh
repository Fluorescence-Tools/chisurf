#!/usr/bin/env bash

mkdir dist
cp -R ./build_tools/ChiSurf.app ./dist/ChiSurf.app
cp -R ~/miniconda3/envs/dist/* ./dist/ChiSurf.app/Contents/Resources/
cp -R chisurf ./dist/ChiSurf.app/Contents/Resources/chisurf

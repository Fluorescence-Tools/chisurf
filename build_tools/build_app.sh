#!/usr/bin/env bash

cp -R /Volumes/VMware\ Shared\ Folders/thoma/PycharmProjects/ChiSurf/build_tools/ChiSurf.app ~/ChiSurf.app
cp -R ~/miniconda2/envs/chisurf/* ~/ChiSurf.app/Contents/Resources/
cp -R /Volumes/VMware\ Shared\ Folders/thoma/PycharmProjects/ChiSurf/chisurf ~/ChiSurf.app/Contents/Resources/chisurf

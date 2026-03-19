@echo off
setlocal enabledelayedexpansion
REM Build script that sets CHISURF_VERSION before building
cd /d "%~dp0..\..\"
for /f "delims=" %%v in ('python rattler-recipe/generate_version.py --print') do (
    set "CHISURF_VERSION=%%v"
)
echo Building with CHISURF_VERSION=!CHISURF_VERSION!
rattler-build build --recipe rattler-recipe/recipe.yaml --output-dir conda-bld --no-test

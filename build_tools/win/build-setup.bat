set "CONDA_ENVIRONMENT_YAML=..\..\environment.yml"
set "DIST_PATH=..\..\dist"
set "SCRIPT_PATH=%~dp0"
set "APP_PATH=..\..\dist\win"
set "SOURCE_PATH=..\..\"
set "CONDA_RECIPE_FOLDER=..\..\conda-recipe"
call conda install -y conda-build jinja2 pysftp
rem call conda build %CONDA_RECIPE_FOLDER%

md %DIST_PATH%
md %APP_PATH%
call conda create -y --prefix %APP_PATH% chisurf python=3.7.3 jinja2 -c local --force
call conda activate %APP_PATH%

REM write setup.iss
call conda activate %APP_PATH%
python make_inno_setup.py
call conda deactivate
call conda activate base

REM Create an Installer with Inno Setup
"C:\Program Files (x86)\Inno Setup 5\Compil32.exe" /cc setup.iss

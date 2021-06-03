@ECHO OFF
setlocal
REM Disable user site packages
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH=%cd%;%PYTHONPATH%"
set "PATH=%cd%;%cd%\bin;%cd%\Library;%cd%\Scripts;%cd%\Library\bin;%cd%\Library\lib"
set "QT_PLUGIN_PATH=%cd%\Library\plugins"
@ECHO OFF
START /b %1
exit
endlocal
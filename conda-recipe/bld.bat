rem "%BUILD_PREFIX%"\pyrcc5.bat chisurf/gui/resources/resource.qrc -o chisurf/gui/resources/resource.py
python setup.py build_ext --force --inplace
python setup.py install --single-version-externally-managed --record=record.txt

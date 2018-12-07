cd ..
C:\Miniconda2\Library\bin\pyrcc5 ./chisurf/mfm/ui/rescource.qrc -o ./chisurf/mfm/ui/rescource.py
c:\Miniconda2\python.exe setup.py build_ext --inplace --force
conda create --name chisurf -y python=2.7

activate chisurf

REM https://github.com/conda-forge/numpy-feedstock/issues/84
REM numpy in conda is built against MKL (600 MB) in size
REM numpy in conda-forge is built against OpenBLAS (<10MB)
conda install -y pyqt pyyaml python-slugify emcee mdtraj ipython pyqtgraph qtconsole qscintilla2 pyopengl numpy numba scipy sympy numexpr pytables matplotlib

REM Create an Installer with Inno Setup
"C:\Program Files (x86)\Inno Setup 5\Compil32.exe" /cc setup.iss

deactivate

conda remove --name chisurf --all
cd build_tools

import os
import sys
import pathlib
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
from setuptools.command.build_py import build_py as _build_py

class CustomBuildPy(_build_py):
    """Custom build_py to ensure extensions are built."""
    def run(self):
        self.run_command('build_ext')
        return super().run()

# Define extension modules
extensions = [
    Extension(
        "chisurf.core.structure.av.fps_",
        sources=[
            "chisurf/core/structure/av/fps_.pyx",
            "chisurf/core/structure/av/mt19937cok.cpp",
        ],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        "chisurf.core.structure.potential.cPotentials_",
        sources=["chisurf/core/structure/potential/cPotentials_.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
]

setup(
    name="chisurf",
    version=os.environ.get('CHISURF_VERSION', '26.dev0'),
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level=3),
    include_package_data=True,
    zip_safe=False,
    cmdclass={'build_py': CustomBuildPy},
)

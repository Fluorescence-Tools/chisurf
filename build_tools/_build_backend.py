"""Custom build backend for ChiSurf with Cython extensions.

This module wraps setuptools.build_meta to provide proper Cython extension building
with NumPy includes.
"""
from setuptools import build_meta as _orig
from setuptools import Extension
import setuptools
import os
import sys
import pathlib
import numpy as np
from Cython.Build import cythonize

# Define extension modules
def get_extensions():
    """Generate Cython extension modules with NumPy includes."""
    extensions = [
        Extension(
            "chisurf.structure.av.fps_",
            sources=[
                "chisurf/structure/av/fps_.pyx",
                "chisurf/structure/av/mt19937cok.cpp",
            ],
            include_dirs=[np.get_include()],
            language="c++",
        ),
        Extension(
            "chisurf.structure.potential.cPotentials_",
            sources=["chisurf/structure/potential/cPotentials_.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
        ),
    ]
    return cythonize(extensions, language_level=3)

class CustomBuildPy(setuptools.command.build_py.build_py):
    """Custom build_py to ensure extensions are built."""
    def run(self):
        self.run_command('build_ext')
        return super().run()

def custom_setup(*args, **kwargs):
    if 'ext_modules' not in kwargs:
        kwargs['ext_modules'] = get_extensions()
    cmdclass = dict(kwargs.get('cmdclass', {}))
    cmdclass.setdefault('build_py', CustomBuildPy)
    kwargs['cmdclass'] = cmdclass
    return _orig.setup(*args, **kwargs)

def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    import setuptools
    original_setup = setuptools.setup
    setuptools.setup = custom_setup
    try:
        return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
    finally:
        setuptools.setup = original_setup

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    import setuptools
    original_setup = setuptools.setup
    setuptools.setup = custom_setup
    try:
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)
    finally:
        setuptools.setup = original_setup

def build_sdist(sdist_directory, config_settings=None):
    import setuptools
    original_setup = setuptools.setup
    setuptools.setup = custom_setup
    try:
        return _orig.build_sdist(sdist_directory, config_settings)
    finally:
        setuptools.setup = original_setup

def get_requires_for_build_wheel(config_settings=None):
    base_requires = _orig.get_requires_for_build_wheel(config_settings) or []
    return base_requires + ['numpy<2.0', 'Cython>=0.29,<3.1']

def get_requires_for_build_sdist(config_settings=None):
    base_requires = _orig.get_requires_for_build_sdist(config_settings) or []
    return base_requires + ['numpy<2.0', 'Cython>=0.29,<3.1']

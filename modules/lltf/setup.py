#!/usr/bin/env python

try:
    import numpy as np
except ImportError:
    np = None
from setuptools import setup, find_packages

NAME = "lltf"
VERSION = "0.1.0"
AUTHOR = "ChiSurf Team"
LICENSE = "MIT"
DESCRIPTION = "Lazy Lifetime Fitter (lltf) module"
LONG_DESCRIPTION = """lltf: Lazy Lifetime Fitter module for fluorescence decay analysis."""
URL = ""
EMAIL = ""


def dict_from_txt(fn):
    d = {}
    with open(fn) as f:
        for line in f:
            (key, val) = line.split()
            d[str(key)] = val
    return d


console_scripts = {"lltf": "lltf.cli:main"}
gui_scripts = {}


metadata = dict(
    name=NAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
    ],
    keywords='fluorescence spectroscopy lifetime',
    packages=find_packages(
        include=(NAME + "*",)
    ),
    package_dir={
        NAME: NAME
    },
    include_package_data=True,
    package_data={
        '': [
            '*.yaml', '*.yml'
        ]
    },
    install_requires=[
        'numpy',
        'scipy',
        'PyYAML',
        'matplotlib',
        'click',
        'click-didyoumean',
        'numba',
        'typing-extensions'
    ],
    setup_requires=[
        'numpy',
        'PyYAML',
        'setuptools'
    ],
    entry_points={
        "console_scripts": [
            "%s=%s" % (key, console_scripts[key]) for key in console_scripts
        ],
        "gui_scripts": [
            "%s=%s" % (key, gui_scripts[key]) for key in gui_scripts
        ]
    }
)

setup(**metadata)
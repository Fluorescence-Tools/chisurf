#!/usr/bin/python
import sys
import numpy
import platform
from setuptools import setup, find_packages, Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    build_ext = None
    print("WARNING: could not import cython. Will not build cython extenstions.")


__name__ = 'chisurf'
__version__ = '20.2.22'
__license__ = 'GPL2.1'


def make_extension(ext):
    """generate an Extension object from its dotted name
    """
    name = (ext[0])[2:-4]
    name = name.replace("/", ".")
    name = name.replace("\\", ".")
    sources = ext[0:]

    if platform.system() == "Darwin":
        extra_compile_args = ["-O3", "-stdlib=libc++"]
        extra_link_args = ["-stdlib=libc++"]
    else:
        extra_compile_args = []
        extra_link_args = []

    return Extension(
        name,
        sources=sources,
        include_dirs=[numpy.get_include(), "."],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=list(),
        library_dirs=["."],
        language="c++"
    )


# and build up the set of Extension objects
eList = [
    [
        './chisurf/fluorescence/simulation/_simulation.pyx',
        './chisurf/fluorescence/simulation/mt19937cok.cpp'
    ],
    [
        './chisurf/fluorescence/av/fps.pyx',
        './chisurf/fluorescence/av/mt19937cok.cpp'
    ],
    [
        './chisurf/structure/potential/cPotentials.pyx'
    ],
    [
        './chisurf/math/reaction/_reaction.pyx'
    ]
]

extensions = [make_extension(extension) for extension in eList]


setup(
    name=__name__,
    version=__version__,
    license=__license__,
    description="Fluorescence-Fitting",
    author="Thomas-Otavio Peulen",
    author_email='thomas.otavio.peulen@gmail.com',
    url='https://fluorescence-tools.github.io/chisurf/',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Win64 (MS Windows)',
        'Environment :: X11 Applications :: Qt',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
    ],
    keywords='fluorescence single-molecule spectroscopy',
    packages=find_packages(
        include=(__name__ + "*",)
    ),
    package_dir={
        __name__: __name__
    },
    include_package_data=True,
    package_data={
        '': [
            '*.json',
            '*.yaml',
            '*.ui',
            '*.png',
            '*.svg',
            '*.css', '*.qss'
            '*.csv', '*.npy', '*.dat'
            '*.dll', '*.so', '*.pyd'
        ]
    },
    install_requires=[
        'PyQt5',
        'qtpy',
        'sip',
        'pyqtgraph',
        'slugify',
        'numba',
        'numpy',
        'numexpr',
        'scipy',
        'emcee',
        'PyYAML',
        'tables',
        'matplotlib',
        'python-docx',
        'deprecation',
        'pyopencl',
        'opencv',
        'mrcfile',
        'qtconsole',
        'ipython',
        'docx',
        'mdtraj'
    ],
    setup_requires=[
        "cython",
        'numpy',
        'PyYAML',
        'setuptools'
    ],
    ext_modules=extensions,
    cmdclass={
        'build_ext': build_ext
    },
    entry_points={
        "gui_scripts": [
            "chisurf=chisurf.gui:gui"
        ]
    }
)

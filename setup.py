#!/usr/bin/python
import platform
import pathlib
from setuptools import setup, find_packages, Extension
try:
    from Cython.Distutils import build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext

from chisurf import info


NAME = info.__name__
VERSION = info.__version__
AUTHOR = info.__author__
LICENSE = info.__license__
DESCRIPTION = info.__description__
LONG_DESCRIPTION = info.LONG_DESCRIPTION
URL = info.__url__
EMAIL = info.__email__

def dict_from_txt(fn):
    d = {}
    with open(fn) as f:
        for line in f:
            (key, val) = line.split()
            d[str(key)] = val
    return d


script_directory = pathlib.Path(__file__).parent.absolute()
gui_scripts = dict_from_txt(script_directory / "chisurf/entry_points/gui.txt")
console_scripts = dict_from_txt(script_directory / "chisurf/entry_points/cmd.txt")


def get_extensions():

    def make_extension(ext):
        """generate an Extension object from its dotted name
        """
        import numpy as np

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
            include_dirs=[np.get_include(), "."],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=list(),
            library_dirs=["."],
            language="c++"
        )

    # and build up the set of Extension objects
    eList = [
        [
            './chisurf/fluorescence/simulation/simulation_.pyx',
            './chisurf/fluorescence/simulation/mt19937cok.cpp'
        ],
        [
            './chisurf/structure/av/fps_.pyx',
            './chisurf/structure/av/mt19937cok.cpp'
        ],
        [
            './chisurf/structure/potential/cPotentials_.pyx'
        ],
        [
            './chisurf/math/reaction/reaction_.pyx'
        ]
    ]
    try:
        return [make_extension(extension) for extension in eList]
    except ImportError:
        return list()


metadata = dict(
    name=NAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
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
        include=(NAME + "*",)
    ),
    package_dir={
        NAME: NAME
    },
    include_package_data=True,
    package_data={
        '': [
            '*.json', '*.yaml',
            '*.ui', '*.css', '*.qss',
            '*.png', '*.svg',
            '*.csv', '*.npy', '*.dat',
            '*.dll', '*.so', '*.pyd', '*.ipynb',
            '*.db'
        ]
    },
    install_requires=[
        'PyQt5', 'qtpy', 'sip', 'pyqtgraph', 'qtconsole',
        'numba', 'numpy', 'numexpr', 'scipy',
        'cython', 'deprecation',
        'emcee',
        'PyYAML', 'typing-extensions',
        'tables',
        'matplotlib', 'python-docx',
        'mdtraj',
        'ipython'
    ],
    extras_require={
        'ml': ['scikit-learn>=1.0.0'],  # Optional machine learning dependencies
    },
    setup_requires=[
        "cython",
        'numpy',
        'PyYAML',
        'setuptools'
    ],
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': build_ext
    },
    entry_points={
        "console_scripts": [
            f"{key}={console_scripts[key]}" for key in console_scripts
        ],
        "gui_scripts": [
            f"{key}={gui_scripts[key]}" for key in gui_scripts
        ]
    }
)

if __name__ == "__main__":
    setup(**metadata)

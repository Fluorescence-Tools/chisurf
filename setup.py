#!/usr/bin/python
import platform
import pathlib
import os
import datetime
import re
from setuptools import setup, find_packages, Extension
try:
    from Cython.Distutils import build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

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


class CustomBuildPy(build_py):
    """Custom build command that replaces the dynamic version with a hardcoded version.

    This command is used during pip builds to ensure that the version number in the
    installed package is a fixed string representing the date at build time, rather
    than a dynamic value that changes each time the module is imported.

    The command:
    1. Replaces the dynamic version in info.py with a hardcoded version (current date)
    2. Runs the standard build_py command to build the package
    3. Restores the original dynamic version in the source code after the build

    This approach ensures that:
    - The installed package has a fixed version number (the date at build time)
    - The source code remains unchanged after the build process
    - The behavior is consistent with the conda build process
    """

    def run(self):
        # Get the path to the info.py file
        info_file = os.path.join(os.path.dirname(__file__), 'chisurf', 'info.py')

        # Get the current date and format it as yy.mm.dd
        today = datetime.datetime.now()
        version = today.strftime('%y.%m.%d')

        # Read the current content of info.py
        with open(info_file, 'r') as f:
            content = f.read()

        # Replace the dynamic version with the hardcoded version
        # This ensures that the installed package has a fixed version number
        pattern = r'__version__ = str\(today\.strftime\("%y\.%m\.%d"\)\)'
        replacement = f'__version__ = "{version}"'
        content = re.sub(pattern, replacement, content)

        # Write the modified content back to info.py
        with open(info_file, 'w') as f:
            f.write(content)

        # Call the original build_py run method to perform the actual build
        build_py.run(self)

        # After the build is complete, restore the original dynamic version
        # This ensures that the source code remains unchanged
        with open(info_file, 'r') as f:
            content = f.read()

        # Replace the hardcoded version with the dynamic version
        pattern = f'__version__ = "{version}"'
        replacement = '__version__ = str(today.strftime("%y.%m.%d"))'
        content = re.sub(pattern, replacement, content)

        # Write the restored content back to info.py
        with open(info_file, 'w') as f:
            f.write(content)


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
    # Register custom build commands
    # - build_ext: The standard Cython build command for building extensions
    # - build_py: Our custom command that handles version replacement during pip builds
    cmdclass={
        'build_ext': build_ext,
        'build_py': CustomBuildPy
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

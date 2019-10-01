#!/usr/bin/python
import sys
import numpy
import yaml
import os
import platform

from Cython.Distutils import build_ext
from setuptools import setup, find_packages
from setuptools.extension import Extension


settings = {
    'version': 'NA'
}
settings_file = os.path.join(
    './mfm/settings/',
    'chisurf.yaml'
)
with open(settings_file) as fp:
    settings.update(yaml.safe_load(fp))


args = sys.argv[1:]
# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext")+1, "--inplace")


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
        './mfm/fluorescence/simulation/_simulation.pyx',
        './mfm/math/rand/mt19937cok.cpp'
    ],
    [
        './mfm/fluorescence/fps/_fps.pyx',
        './mfm/fluorescence/fps/mt19937cok.cpp'
    ],
    [
        './mfm/structure/potential/cPotentials.pyx'
    ],
    [
        './mfm/math/reaction/_reaction.pyx'
    ]
]

extensions = [make_extension(extension) for extension in eList]


setup(
    version=settings['version'],
    description="Fluorescence-Fitting",
    author="Thomas-Otavio Peulen",
    author_email='thomas.otavio.peulen@gmail.com',
    url='https://fluorescence-tools.github.io/chisurf/',
    name="ChiSurf",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Win64 (MS Windows)',
        'Environment :: X11 Applications :: Qt',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
    ],
    keywords='fluorescence single-molecule spectroscopy',
    packages=find_packages(),
    package_data={
        # If any package contains the listed file types and include them:
        '': ['*.json', '*.yaml', '*.ui', '*.png', '*.svg', '*.css'],
    },
    install_requires=[
        'numpy',
        'slugify',
        'sip',
        'PyQt5',
        'emcee',
        'numba',
        'scipy',
        'pyqtgraph',
        'sympy',
        'PyYAML',
        'tables',
        'numexpr',
        'matplotlib',
        'python-docx',
        'deprecation',
        'pyopencl',
        'qdarkstyle',
        'qtpy',
        'mrcfile'
    ],
    ext_modules=extensions,
    cmdclass={
        'build_ext': build_ext
    }
)


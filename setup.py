#!/usr/bin/python

import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import numpy


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
    return Extension(
        name,
        sources=sources,
        include_dirs=[numpy.get_include(), "."],
        extra_compile_args=[],
        extra_link_args=[],
        libraries=[],
        library_dirs=["."],
        language="c++")

# and build up the set of Extension objects
eList = [
    #[
    #    './chisurf//mfm/fluorescence/simulation/_simulation.pyx',
    #    './chisurf//mfm/math/rand/mt19937cok.cpp'
    #],
    #[
    #    './chisurf//mfm/math/reaction/_reaction.pyx'
    #],
    #[
    #    './chisurf/mfm/fluorescence/pda/_sgsr.pyx',
    #    './chisurf/mfm/fluorescence/pda/sgsr_misc.cpp'
    #]
]

extensions = [make_extension(extension) for extension in eList]

long_description = "ChiSurf"
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    version="18.9.1",
    description="Fluorescence-Fitting",
    long_description=long_description,
    author="Thomas-Otavio Peulen",
    author_email='thomas.otavio.peulen@gmail.com',
    url='www.fret.at',
    name="ChiSurf",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Win64 (MS Windows)',
        'Environment :: X11 Applications :: Qt',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
    ],
    keywords='fluorescence single-molecule',
    packages=find_packages(),
    package_data={
        # If any package contains the listed file types and include them:
        '': ['*.json', '*.yaml', '*.ui', '*.png', '*.svg', '*.css'],
    },
    install_requires=['numpy'],
    ext_modules=extensions,
    cmdclass={
        'build_ext': build_ext
    }
)


#!/usr/bin/python

import sys
import numpy
import platform
from setuptools import setup, find_packages, Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    build_ext = None
    print(
        "WARNING: could not import cython. "
        "Will not build cython extenstions.",
        file=sys.stderr
    )


import chisurf
__name__ = chisurf.__name__
__version__ = chisurf.__version__
__author__ = chisurf.__author__
__license__ = chisurf.__license__
__description__ = chisurf.__description__


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

metadata = dict(
    name=__name__,
    version=__version__,
    license=__license__,
    description=__description__,
    author=__author__,
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
        'opencv-python',
        'mrcfile',
        'qtconsole',
        'ipython',
        'python-docx',
        'mdtraj',
        'typing-extensions'
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
        "console_scripts": [
        ],
        "gui_scripts": [
            "chisurf=chisurf.__main__:main",
            "kappa2distribution=chisurf.gui.tools.kappa2_distribution.__main__:main",
            "fps.json_edit=chisurf.gui.tools.structure.create_av_json.__main__:main",
            "fret_calculator=chisurf.gui.tools.fret.calculator.__main__:main",
            "f_test=chisurf.gui.tools.f_test.__main__:main",
            "tttr_clsm_pixel_select=chisurf.gui.tools.tttr.clsm_pixel_select.__main__:main",
            "tttr_convert=chisurf.gui.tools.tttr.convert.__main__:main",
            "tttr_correlator=chisurf.gui.tools.tttr.correlate.__main__:main",
            "tttr_decay_histogram=chisurf.gui.tools.tttr.decay.__main__:main",
            "traj_align=chisurf.gui.tools.structure.align_trajectory.__main__:main",
            "traj_convert=chisurf.gui.tools.structure.convert_trajectory.__main__:main",
            "traj_fret_analysis=chisurf.gui.tools.structure.fret_trajectory.__main__:main",
            "traj_join=chisurf.gui.tools.structure.join_trajectories.__main__:main",
            "traj_energy_calculator=chisurf.gui.tools.structure.potential_energy.__main__:main",
            "traj_remove_clashed_frames=chisurf.gui.tools.structure.remove_clashed_frames.__main__:main",
            "traj_rotate_translate=chisurf.gui.tools.structure.rotate_translate_trajectory.__main__:main",
            "traj_save_topology=chisurf.gui.tools.structure.save_topology.__main__:main",
        ]
    }
)


setup(**metadata)

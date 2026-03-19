import os
import sys
import platform
import inspect
import setuptools
import pathlib

# Only import cmake_build_extension when it's actually needed
# This allows conda-build to load the version without requiring cmake_build_extension
try:
    import cmake_build_extension
    HAS_CMAKE_BUILD_EXTENSION = True
except ImportError:
    HAS_CMAKE_BUILD_EXTENSION = False

# Read version from include/info.h
def read_version(header_file):
    version = "0.0.0"
    with open(header_file, "r") as fp:
        for line in fp.readlines():
            if "#define" in line and "CHINET_VERSION" in line:
                version = line.split()[-1]
    return version.replace('"', '')

VERSION = read_version(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include', 'info.h'))

# Importing the bindings inside the build_extension_env context manager is necessary only
# in Windows with Python>=3.8.
# See https://github.com/diegoferigo/cmake-build-extension/issues/8.
# Note that if this manager is used in the init file, cmake-build-extension becomes an
# install_requires that must be added to the setup.cfg. Otherwise, cmake-build-extension
# could only be listed as build-system requires in pyproject.toml since it would only
# be necessary for packaging and not during runtime.
init_py = inspect.cleandoc(
    """
    import cmake_build_extension
    with cmake_build_extension.build_extension_env():
        from . import bindings
    """
)

# Define setup arguments
setup_args = {
    'version': VERSION,
    'classifiers': [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ]
}

# Only add cmake_build_extension related arguments if it's available
if HAS_CMAKE_BUILD_EXTENSION:
    setup_args.update({
        'ext_modules': [
            cmake_build_extension.CMakeExtension(
                name='chinet',
                install_prefix="chinet",
                cmake_configure_options=[
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DPYTHON_VERSION:str=%s" % platform.python_version(),
                    "-DCMAKE_CXX_FLAGS='-w'",
                    "-DBUILD_PYTHON_INTERFACE=ON",
                    "-DPython_ROOT_DIR='%s'" % pathlib.Path(sys.executable).parent
                ]
            )
        ],
        'cmdclass': dict(
            # Enable the CMakeExtension entries defined above
            build_ext=cmake_build_extension.BuildExtension,
        )
    })

setuptools.setup(**setup_args)

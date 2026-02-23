"""Custom build backend for ChiSurf with Cython extensions.

This module wraps setuptools.build_meta to provide proper Cython extension building
with NumPy includes.
"""
from setuptools import build_meta as _orig
from setuptools import Extension
from setuptools.command.build_py import build_py
import sys
import os
import datetime
import pathlib
import re
import subprocess


# Re-export the standard backend functions
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
build_sdist = _orig.build_sdist


def get_requires_for_build_wheel(config_settings=None):
    """Add numpy and Cython to build requirements."""
    base_requires = _orig.get_requires_for_build_wheel(config_settings) or []
    return base_requires + ['numpy<2.0', 'Cython>=0.29,<3.1']


def get_requires_for_build_sdist(config_settings=None):
    """Add numpy and Cython to build requirements."""
    base_requires = _orig.get_requires_for_build_sdist(config_settings) or []
    return base_requires + ['numpy<2.0', 'Cython>=0.29,<3.1'] + ['numpy>=1.20', 'Cython>=0.29']


def get_extensions():
    """Generate Cython extension modules with NumPy includes.
    
    Returns:
        list: List of Extension objects for Cython modules
    """
    import numpy as np
    from Cython.Build import cythonize
    import platform
    
    # Platform-specific compiler flags
    if platform.system() == "Darwin":
        extra_compile_args = ["-O3", "-stdlib=libc++"]
        extra_link_args = ["-stdlib=libc++"]
    else:
        extra_compile_args = []
        extra_link_args = []
    
    # Define extension modules (only include sources that still exist)
    extensions = [
        Extension(
            "chisurf.structure.av.fps_",
            sources=[
                "chisurf/structure/av/fps_.pyx",
                "chisurf/structure/av/mt19937cok.cpp",
            ],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++",
        ),
        Extension(
            "chisurf.structure.potential.cPotentials_",
            sources=["chisurf/structure/potential/cPotentials_.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++",
        ),
    ]
    
    # Cythonize extensions
    return cythonize(extensions, compiler_directives={'language_level': '3'})


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
        # Get the path to the info.py file (project root / chisurf / info.py)
        project_root = pathlib.Path(__file__).resolve().parent.parent
        info_file = project_root / 'chisurf' / 'info.py'

        # Determine the version to bake into the built package.
        # Prefer explicit override; otherwise derive from git tags.
        version = os.environ.get('CHISURF_VERSION')
        if version:
            version = version.strip()
        else:
            try:
                desc = subprocess.check_output(
                    [
                        'git', 'describe', '--tags', '--long',
                        '--match', 'v[0-9]*',
                    ],
                    cwd=str(project_root),
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except Exception:
                desc = ''

        # Expected: vX.Y.Z-N-g<sha>
        m = re.match(r'^(v[0-9.]+)-(\d+)-g([0-9a-f]+)$', desc)
        if m:
            tag = m.group(1).lstrip('v')
            # PEP 440: remove leading zeros from dot-separated numeric segments
            parts = tag.split('.')
            norm = []
            for p in parts:
                if not p.isdigit():
                    norm = []
                    break
                norm.append(str(int(p)))
            base = '.'.join(norm) if norm else None
            if base is not None:
                distance = int(m.group(2))
                if distance == 0:
                    version = base
                else:
                    # Extract year from base tag for dev version
                    base_parts = base.split(".")
                    if len(base_parts) >= 1:
                        year = base_parts[0]
                        version = f'{year}.dev{distance}'
                    else:
                        # Fallback to current year if tag format is unexpected
                        today = datetime.datetime.now()
                        version = f"{today.strftime('%y')}.dev{distance}"

        if not version:
            # Final fallback: deterministic dev-style version at build time.
            today = datetime.datetime.now()
            version = today.strftime('%y.dev0')
        
        # Read the current content of info.py
        with open(info_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Store original content for restoration
        original_content = content
        
        # Replace the dynamic version computation with the hardcoded version.
        # This ensures that the installed package has a fixed version string.
        replacement = f'__version__ = "{version}"'
        content, n = re.subn(
            r'^__version__\s*=\s*_compute_version\(\)\s*$',
            replacement,
            content,
            flags=re.MULTILINE,
        )
        if n == 0:
            # Fallback for older layouts.
            pattern = r'__version__\s*=\s*str\(today\.strftime\("%y\.%m\.%d"\)\)'
            content = re.sub(pattern, replacement, content)
        
        # Write the modified content back to info.py
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        try:
            # Call the original build_py run method to perform the actual build
            build_py.run(self)
        finally:
            # After the build is complete, restore the original dynamic version
            # This ensures that the source code remains unchanged
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(original_content)


def get_requires_for_build_editable(config_settings=None):
    """Add numpy and Cython to build requirements for editable installs."""
    return get_requires_for_build_wheel(config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build an editable wheel with extensions."""
    return build_wheel(wheel_directory, config_settings, metadata_directory)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel with Cython extensions.
    
    This wraps setuptools.build_meta and injects our Cython extensions.
    """
    # Temporarily inject our extensions into setuptools
    import setuptools
    
    # Get the original setup function
    original_setup = setuptools.setup
    
    def custom_setup(*args, **kwargs):
        # Inject our extensions
        if 'ext_modules' not in kwargs:
            kwargs['ext_modules'] = get_extensions()
        return original_setup(*args, **kwargs)
    
    # Replace setup temporarily
    setuptools.setup = custom_setup
    
    try:
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)
    finally:
        # Restore original setup
        setuptools.setup = original_setup

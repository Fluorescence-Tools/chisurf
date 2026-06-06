# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import importlib.util
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'chisurf'
copyright = '2025, chisurf developers'
author = 'chisurf developers'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

autosummary_generate = False
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# Tolerate 'Modes' section used in ChiSurfAPI docstring (NumPy-style allows it)
numpydoc_validate = False

suppress_warnings = [
    'autodoc.cannot_be_local_function',
]

for optional_extension in [
    'numpydoc',
    'sphinx_autodoc_typehints',
    'myst_parser',
]:
    if importlib.util.find_spec(optional_extension) is not None:
        extensions.append(optional_extension)

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme' if importlib.util.find_spec('sphinx_rtd_theme') else 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builder finishes
# and put into the build directory. The builtin copy mechanism is used.
# html_static_path = ['_static']

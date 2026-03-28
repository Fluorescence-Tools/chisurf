.. _installation::

Installation
============

``chinet`` can either be installed from prebuilt binaries or from the source code. For most users it is recommended
to install ``chinet`` using the prebuilt binaries. Below the installation using prebuilt binaries and the
prerequisites to compile ``chinet`` are briefly outlined.


Prebuilt binaries
-----------------
It is recommended to install ``chinet`` for Python environments via ``conda`` by. ::

    conda install -c tpeulen chinet


Alternatively, ``chinet`` can be installed via ``pip``.::


    pip install chinet


Compliation
-----------

``chinet`` can be compiled and installed using the source code provided in the git repository. ::

    git clone --recursive https://github.com/tpeulen/chinet.git
    cd chinet
    sudo python setup.py install

To compile ``chinet`` a set of prerequisites need to be fulfilled:

    1. An installed compiler.
    2. The `HDF5 <https://www.hdfgroup.org/>`_ library with C/C++ include files.
    3. A recent 64bit `Python <https://www.python.org/>`_ installation with include files.
    4. `cmake <https://cmake.org/>`_
    5. `SWIG <http://www.swig.org/>`_


Windows
^^^^^^^
On windows ``chinet`` is best compiled with the `Visual Studio 2017 <https://visualstudio.microsoft.com/>`_. For
compilation the Visual Studion Community edition is sufficient. In addition to Visual Studio the libraries and the
include files as listed above need to be installed. The prebuilt binaries are compiled on Windows 10 with using
64bit anaconda Python environments `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ using the conda
build recipe that is provided with the source code in the ``conda-recipe`` folder.

MacOS
^^^^^
For MacOS the prebuilt binaries are compiled on MacOS 10.13 with the Apple clang compiler using a anaconda
distribution and the provided ``conda-recipe``.

Linux
^^^^^
The Linux prebuilt binaries are compiled on Ubuntu 18.04 in an anaconda distribution and the provided
``conda-recipe``.

Conda
^^^^^

A conda recipe is provided in the folder 'conda-recipe' to build the chinet library with the
`conda build <https://docs.conda.io/projects/conda-build/en/latest/>`_ environment.

To build the library download the recipe, install the conda build package and use the provided
recipe to build the library.

.. code-block:: bash

    conda install conda-build
    conda build conda-recipe


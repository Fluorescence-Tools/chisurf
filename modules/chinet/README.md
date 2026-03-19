# chinet
[![Anaconda-Server Version](https://anaconda.org/tpeulen/chinet/badges/version.svg)](https://anaconda.org/tpeulen/chinet)
[![Anaconda-Server Badge](https://anaconda.org/tpeulen/chinet/badges/platforms.svg)](https://anaconda.org/tpeulen/chinet)
![conda build](https://github.com/tpeulen/chinet/actions/workflows/conda-build.yml/badge.svg)


## General description
chinet is a C++ library to create optimize, sample, and archive global 
models. A global model is a model, that unites multiple data-sets and
seeks for a joint description of the united dataset. chinet is the backend of 
[ChiSurf](https://github.com/fluorescence-tools/chisurf).

Global models can unite datasets of the same kind or datasets of 
different types. A typical examples of a global model in fluorescence 
experiments is the joint description of multiple fluorescence 
correlation curves in a titration and the joint description of multiple
fluorescence decay curves reporting on FRET in a biomolecular structure
by a single structural model.

Computing a global model with a large diverse set of different data 
can be computationally expensive. To reduce the computational costs
and to decrease the evaluation time of a global model defined by chinet,
the mutual dependencies of the model parameters are modeled by a graph
structure that connects "computing nodes". When a a set of parameters is
changed only computing nodes that are affected by these changes are 
evaluated. Independent nodes are evaluated in parallel. 

The state of the evaluation graph can be written to a database for 
documentation purposes and reconstructed using unique identifies 
provided by the database. 

## Goals

*   reactive dataflow model framework  
*   fast inter computation node communication
*   define and store models jointly with associated data identifies in data base.
*   Platform independent C/C++ library with interfaces for scripting libraries 

## Building and Installation

### C++ shared library

The C++ shared library can be installed from source with [cmake](https://cmake.org/):

```console
git clone --recursive https://github.com/tpeulen/chinet.git
mkdir chinet/build; cd chinet/build
cmake ..
sudo make install
```

On Linux you can build and install a package instead (prefered):

### Python bindings
The Python bindings can be either be installed by downloading and 
compiling the source code or by using a precompiled distribution for 
Python anaconda environment.

The following commands can be used to download and compile the source 
code:

```console
git clone --recursive https://github.com/tpeulen/chinet.git
cd chinet
sudo python setup.py install
```

In an [anaconda](https://www.anaconda.com/) environment the library can 
be installed by the following command: 
```console
conda install -c tpeulen chinet
```

For most users the later approach is recommended. Currently, 
pre-compiled packages for the anaconda distribution system are 
available for:

*   Windows: Python 3.7
*   Linux: Python 3.7
*   macOS: Python 3.7

Legacy 32-bit platforms and the deprecated Python 2.7 are not supported.

## Examples

```python
import chinet as cn

v1 = 23.0
v2 = 29.0
p1 = cn.Port()
p1.value = v1
p2 = cn.Port(v1)
p3 = cn.Port(
    value=v1,
    fixed=True
)
p4 = cn.Port(
    value=v1,
    fixed=False
)
p5 = cn.Port(v2)
p5.link = p4
print(p4.value == p1.value)
```

## License
chinet is released under the open source MIT license.

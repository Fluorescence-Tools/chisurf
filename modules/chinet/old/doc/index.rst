chinet
======


chinet is a C++ library to create optimize, sample, and archive global
models. A global model is a model, that unites multiple data-sets and
seeks for a joint description of the united dataset.

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

chinet is NOT intended as ready-to-use software for specific application
purposes.


Contents
--------

.. toctree::
   :maxdepth: 1

   installation
   curves
   cpp_api
   glossary
   references


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`


License
-------

chinet is released under the open source `MIT license <https://opensource.org/licenses/MIT>`_.

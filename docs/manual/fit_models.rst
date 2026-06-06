Fit models
----------

In ChiSurf a model combines (variable) parameters and approaches to compute theoretical data (forward model). In ChiSurf instances of models are usually tight to the data in a ":strong:`Fit`". The currently active fit, the corresponding data, and the model can be accessed in the shell.

.. code-block:: python

  cs.current_fit
  cs.current_fit.data
  cs.current_fit.model

Usually, ":strong:`Fit`" instances, and model instances are created jointly in the graphical user interface.

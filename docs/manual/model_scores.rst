Model scores
------------

The fitting minimizes the difference between the model and the data. Sampling samples over variable (free) model parameters to estimate probability distributions over parameters. Both, sampling, and optimization require a score. The score depends on the model, the data, and the data noise. The value of a score can of the current fit can be accessed as follows:

.. code-block:: python

  fit = cs.current_fit
  fit.get_score(score_type='chi2')

Here, the score_type defines the type of the score. Default score types that must be implemented for all fits/model combinations is the sum of squared data-noise weighted deviations between the model and the data, 'chi2', and 'chi2r', the sum of squared data-noise weighted deviations reduced by the degrees of freedom (the number of observations - number of model parameters). Values of 'chi2' and 'chi2r' can also be accessed as follows:

.. code-block:: python

  fit = cs.current_fit
  fit.chi2
  fit.chi2r

For a fit object the code

.. code-block:: python

  fit.get_wres()

returns the deviation between the data and the model weighted by the data noise.

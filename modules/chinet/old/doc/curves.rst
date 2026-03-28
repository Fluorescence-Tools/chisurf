Curves
======

The library ``chinet`` uses the class ``Curve`` to save and operate on the data of a model. One dimensional
models derive from the ``Curve`` class. A ``Curve`` object has properties for the x-values, the y-values, and
the errors of the x- and y-values.

.. code-block:: python

    import chinet as flm
    import numpy as np

    curve = flm.Curve()
    x = np.linspace(0, 6, 100)
    y = np.sin(x)
    curve.set_x(x)
    curve.set_y(y)
    (y == curve.get_y())
    (x == curve.get_x())

``Curve`` objects have the methods ``add``, ``mul``, ``div``, and ``sub`` to add, multiply, divide and subtract
and that take either ``Curve`` objects or floating point numbers as arguments. These operations act element-wise on
the y-axis values of the ``Curve`` objects. In addition to these elementary operations, the content of a ``Curve``
object can be freely shifted with respect to the x-axis. For non-integer shifts the y-values are linearly interpolated.
By default, if no error is provided for the y-values the error is initialized with ones.

.. plot:: plots/curve_1.py
   :include-source:

Above, shifting, addition, and multiplication are illustrated for a curve. Note, the multiplication operation can
also be used with two curves.


``Curve`` objects can be saved and loaded using the methods ``save`` and ``from_json``


.. code-block:: python

    import chinet as flm
    import numpy as np

    curve = flm.Curve()
    x = np.linspace(0, 6, 100)
    y = np.sin(x)
    curve.set_x(x)
    curve.set_y(y)

    curve.save("test.json")
    c = flm.Curve()
    c.from_json("test.json")

A

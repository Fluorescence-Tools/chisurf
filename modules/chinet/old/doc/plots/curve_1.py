import chinet as flm
import numpy as np
import pylab as p

curve = flm.Curve()
x = np.linspace(0, 6, 100)
y = np.sin(x)
curve.set_x(x)
curve.set_y(y)

p.plot(curve.get_x(), curve.get_y())

curve.shift(1.5)
p.plot(curve.get_x(), curve.get_y())

curve.add(curve)
p.plot(curve.get_x(), curve.get_y())

curve.mul(3.1)
p.plot(curve.get_x(), curve.get_y())

p.show()

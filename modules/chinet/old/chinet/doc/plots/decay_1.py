import chinet as flm
import numpy as np
import pylab as p

dt = 0.0141
nx = 4096


x, y = np.loadtxt("./examples/tcspc/ibh/Prompt.txt", skiprows=9).T
irf = flm.Curve()
irf.set_x(x * dt)
irf.set_y(y)

d = flm.Decay(dt, nx)
d.set_instrument_response_function(irf)
d.append(0.1, 4)
d.append(2, 1)
d.update()

p.semilogy(
    d.get_x(),
    d.get_y()
)
p.semilogy(
    irf.get_x(),
    irf.get_y()
)
p.ylim(1, 150000)

p.show()


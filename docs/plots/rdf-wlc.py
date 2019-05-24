import lib
import numpy as np
import matplotlib.pyplot as p
import lib.math.functions.rdf as rdf

r = np.linspace(0, 0.99, 200)

p.plot(r, rdf.worm_like_chain(r, 1.0))
p.plot(r, rdf.worm_like_chain(r, 0.8))
p.plot(r, rdf.worm_like_chain(r, 0.6))
p.plot(r, rdf.worm_like_chain(r, 0.4))
p.plot(r, rdf.worm_like_chain(r, 0.2))

p.show()

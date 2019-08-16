import lib.fluorescence.general as ge
import numpy as np
import pylab as p

rda_mean = [50., 50.0]
rda_sigma = [8, 2.0]
amplitudes = [0.5, 0.5]
rates = ge.gaussian2rates(rda_mean, rda_sigma, amplitudes, interleaved=False)
a = rates[:,0]
kFRET = rates[:,1]
ts = np.logspace(-3, 3, 512)
et = np.array([np.dot(a, np.exp(-kFRET * t)) for t in ts])
t_matrix, r_DA = ge.calc_transfer_matrix(ts, 0.1, 120, 512, space='lin')
p.imshow(t_matrix)
p.show()

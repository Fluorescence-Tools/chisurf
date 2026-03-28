import math


def sg_sr(Sg, Sr, **kwargs):
    """Green-red intensity ratio

    :param Sg:
    :param Sr:
    :param kwargs:
    :return:
    """
    if Sr != 0:
        return float(Sg) / float(Sr)
    else:
        return float('nan')


def fg_fr(Sg, Sr, **kwargs):
    BG = kwargs.get('BG', 0.0)
    BR = kwargs.get('BR', 0.0)
    return sg_sr(Sg - BG, Sr - BR, **kwargs)


def fd_fa(Sg, Sr, **kwargs):
    g = kwargs.get('det_eff_ratio', 1.0)
    return fg_fr(Sg, Sr, **kwargs) / g


def transfer_efficency(Sg, Sr, **kwargs):
    qyd = kwargs.get('qy_d', 1.0)
    qya = kwargs.get('qy_a', 1.0)
    return 1.0 / (1.0 + fg_fr(Sg, Sr, **kwargs) / qyd * qya)


def sgsr_histogram(sgsr, eq, **kwargs):
    """Generates a histogram given a SgSr-array using the function 'eq'

    :param SgSr:
    :param eq:
    :param kwargs:
    :return:

    Examples
    --------

    >>>
    """
    n_bins = kwargs.get('n_bins', 101)
    n_min = kwargs.get('n_min', 1)
    x_max = kwargs.get('x_max', 100)
    x_min = kwargs.get('x_min', 0.1)
    exp_param = kwargs.get('exp_param', dict())
    log_x = kwargs.get('log_x', False)

    histogram_x = np.zeros(n_bins, dtype=np.float64)
    histogram_y = np.zeros(n_bins, dtype=np.float64)

    if log_x:
        xminlog = math.log(x_min)

        bin_width = (math.log(x_max) - xminlog)/(float(n_bins) - 1.)
        i_bin_width = 1./bin_width
        xmincorr = math.log(x_min) - 0.5 * bin_width

        # histogram x
        for i_bin in range(0, n_bins):
            histogram_x[i_bin] = math.exp(xminlog + bin_width*float(i_bin))
        # histogram y
        for green in range(1, n_max):
            firstred = 1 if green > n_min else n_min - green
            for red in range(firstred, n_max - green):
                x1 = eq(green, red, **exp_param)
                if np.isnan(x1):
                    continue
                if x1 > 0.0:
                    x = math.log(x1)
                    i_bin = int(math.floor((x-xmincorr)*i_bin_width))
                    if 0 < i_bin < n_bins:
                        histogram_y[i_bin] += sgsr[green*(n_max+1) + red]
    else:
        bin_width = (x_max - x_min)/(float(n_bins) - 1.)
        i_bin_width = 1./bin_width
        xmincorr = x_min - 0.5*bin_width

        # histogram x
        for i_bin in range(0, n_bins):
            histogram_x[i_bin] = x_min + bin_width * float(i_bin)
        # histogram y
        for green in range(0, n_bins):
            firstred = 0 if green > n_min else n_min - green
            for red in range(firstred, n_max - green):
                x = eq(green, red, **exp_param)
                if np.isnan(x):
                    continue
                i_bin = int(math.floor((x-xmincorr)*i_bin_width))
                if 0 < i_bin < n_bins:
                    histogram_y[i_bin] += sgsr[green*(n_max+1) + red]
    return histogram_x, histogram_y


import chinet as cn
import pylab as p
import numpy as np


n_max = 90  # The maximum number of considered photons
bg = 10.1  # Green background
br = 0.5  # Red background
transfer_efficencies = np.array([0.2, 0.8], np.float64)
a = np.array([0.5, 0.5], dtype=np.float64)

pg = 1 - transfer_efficencies
pf = np.ones(n_max + 1, dtype=np.float64)

# Calculate the 2D-probability distirbution to find a certain combination of green and red photons
pda = cn.Pda()
pda.setBr(br)
pda.setBg(bg)
pda.setPF(pf)
pda.setNmax(n_max)
#pda.append(0.5, 0.8)
#pda.append(0.5, 0.2)
[pda.append(am, pl) for am, pl in zip(a, pg)]
pda.evaluate()
s = pda.getSgSr()

p.imshow(s.reshape(n_max, n_max))
p.show()

x, y = sgsr_histogram(s, fg_fr, x_min=0.01, x_max=100., log_x=True,
                      n_min=10,
                      n_max=70,
                      exp_param={
                          "BG": bg,
                          "BR": br,
                          "qy_d": 1.0,
                          "qy_a": 1.0,
                          "det_eff_ratio": 1.0
                      })
p.semilogx(x, y)
p.show()


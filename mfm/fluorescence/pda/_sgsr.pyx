import numpy as np
cimport numpy as np


cdef extern from "sgsr.h":
    void sgsr_pF(double* SgSr, double* pF, const unsigned int Nmax, double Bg, double Br, double pg)
    void sgsr_pN(double* SgSr, double* pN, const unsigned int Nmax, double Bg, double Br, double pg)
    void sgsr_pN_manypg(double* SgSr, double* pN, const unsigned int Nmax, double Bg, double Br,
                        unsigned int N_pg, double* pg, double* a)
    void sgsr_pF_manypg(double* SgSr, double* pF, const unsigned int Nmax, double Bg, double Br,
                        unsigned int N_pg, double* pg, double* a)
    void sgsr_manypF(double* SgSr, double* pF, const unsigned int Nmax, double Bg, double Br,
                     unsigned int N_pg, double* pg, double* a)
    void polynom2_conv (double* return_p, double* p, unsigned int n, double p2);
    void conv_pF(double*, double*, unsigned int, double, double);
    void poisson_0toN(double*, double, unsigned int);


def poisson_0toN_py(np.ndarray[np.double_t, ndim=1] p, double lam):
    n = p.shape[0]
    poisson_0toN(<double*> p.data,
                 <double> lam,
                 <unsigned int> n)

def convolute_pF_background(np.ndarray[np.double_t, ndim=1] sgsr,
                            np.ndarray[np.double_t, ndim=1] fgfr,
                            nmax,
                            Bg, Br):
    conv_pF(
        <double*> sgsr.data,
        <double*> fgfr.data,
        <unsigned int> nmax,
        <double> Bg,
        <double> Br
    )

def polynom2_convolution(np.ndarray[np.double_t, ndim=1] return_p, np.ndarray[np.double_t, ndim=1] p,
                  unsigned int n, double p2):
    polynom2_conv(
        <double*> return_p.data,
        <double*> p.data,
        <unsigned int> n,
        <double> p2
    )



def sgsr_many_pf(unsigned int n_max, np.ndarray[np.double_t, ndim=2] pf, double bg, double br,
                   np.ndarray[np.double_t, ndim=1] pg, np.ndarray[np.double_t, ndim=1] a):
    """

    :param n_max: maximum number of photons (integer)
    :param pf: each species has its own brightnesses
    :param bg: green background
    :param br: red background
    :param pg: 1-transfer efficiencies of species (vector)
    :param a: amplitudes of species
    :return:


    Examples
    --------

    >>> import _sgsr
    >>> import pylab as p
    >>> import numpy as np
    >>> n_max = 100
    >>> bg = 1.2
    >>> br = 20.7
    >>> pg = np.array([0.9, 0.1], dtype=np.float64)
    >>> a = np.array([0.5, 0.5], dtype=np.float64)
    >>> pf = np.ones((2, n_max + 1), dtype=np.float64)
    >>> pf[0,:] *= 0.4
    >>> s = _sgsr.sgsr_many_pf(n_max, pf, bg, br, pg, a)
    >>> p.plot(s)
    >>> p.show()
    >>> s2 = s.reshape((n_max+1, n_max+1))
    >>> p.imshow(s2)
    >>> p.show()

    """

    cdef np.ndarray[dtype=np.float64_t, ndim=1] sgsr = np.zeros((n_max + 1) * (n_max + 1), dtype=np.float64, order='C')
    cdef np.ndarray[dtype=np.float64_t, ndim=1] pf1 = pf.flatten()
    n_pg = pg.shape[0]

    sgsr_manypF(<double*> sgsr.data,
          <double*> pf1.data,
          <unsigned int> n_max,
          <double> bg,
          <double> br,
          <unsigned int> n_pg,
          <double*> pg.data,
          <double*> a.data,
    )
    return sgsr



def sgsr_pn_manypg(unsigned int n_max, np.ndarray[np.double_t, ndim=1] pf, double bg, double br,
                   np.ndarray[np.double_t, ndim=1] pg, np.ndarray[np.double_t, ndim=1] a):
    """

    :param n_max:
    :param pf:
    :param bg:
    :param br:
    :param pg:
    :param a:
    :return:



    Examples
    --------

    >>> import _sgsr
    >>> import pylab as p
    >>> import numpy as np
    >>> n_max = 101
    >>> pf = np.ones(n_max + 1, dtype=np.float64)
    >>> bg = 1.2
    >>> br = 1.7
    >>> pg = np.array([0.3, 0.6], dtype=np.float64)
    >>> a = np.array([0.5, 0.5], dtype=np.float64)
    >>> s = _sgsr.sgsr_pn_manypg(n_max, pf, bg, br, pg, a)
    >>> p.plot(s)
    >>> p.show()

    """
    # n_max is the number of photons. Therefore allocate an array of size n_max + 1 to include zero-photons.
    cdef np.ndarray[dtype=np.float64_t, ndim=1] sgsr = np.zeros((n_max + 1) * (n_max + 1), dtype=np.float64, order='C')
    n_pg = pg.shape[0]

    sgsr_pN_manypg(<double*> sgsr.data,
          <double*> pf.data,
          <unsigned int> n_max,
          <double> bg,
          <double> br,
          <unsigned int> n_pg,
          <double*> pg.data,
          <double*> a.data,
    )
    return sgsr


def sgsr_pf_manypg(unsigned int n_max, np.ndarray[np.double_t, ndim=1] pf, double bg, double br,
                   np.ndarray[np.double_t, ndim=1] pg, np.ndarray[np.double_t, ndim=1] a):
    """

    :param n_max:
    :param pf:
    :param bg:
    :param br:
    :param pg:
    :param a:
    :return:



    Examples
    --------

    >>> import _sgsr
    >>> import pylab as p
    >>> import numpy as np
    >>> n_max = 101
    >>> pf = np.ones(n_max + 1, dtype=np.float64)
    >>> bg = 1.2
    >>> br = 1.7
    >>> pg = np.array([0.3, 0.6], dtype=np.float64)
    >>> a = np.array([0.5, 0.5], dtype=np.float64)
    >>> s = _sgsr.sgsr_pf_manypg(n_max, pf, bg, br, pg, a)
    >>> p.plot(s)
    >>> p.show()

    """

    cdef np.ndarray[dtype=np.float64_t, ndim=1] sgsr = np.zeros((n_max + 1) * (n_max + 1), dtype=np.float64, order='C')
    n_pg = pg.shape[0]

    sgsr_pF_manypg(<double*> sgsr.data,
          <double*> pf.data,
          <unsigned int> n_max,
          <double> bg,
          <double> br,
          <unsigned int> n_pg,
          <double*> pg.data,
          <double*> a.data,
    )
    return sgsr


def sgsr_pf(unsigned int n_max, np.ndarray[np.double_t, ndim=1] pf, double bg, double br, double pg):
    """

    :param n_max:
    :param pf:
    :param bg:
    :param br:
    :param pg:
    :return:

    Examples
    --------

    >>> import _sgsr
    >>> import pylab as p
    >>> import numpy as np
    >>> n_max = 101
    >>> pf = np.ones(n_max + 1, dtype=np.float64)
    >>> bg = 1.2
    >>> br = 1.7
    >>> pg = 0.5
    >>> s = _sgsr.sgsr_pf(n_max, pf, bg, br, pg)

    """

    cdef np.ndarray[dtype=np.float64_t, ndim=1] sgsr = np.zeros((n_max + 1) * (n_max + 1), dtype=np.float64, order='C')
    sgsr_pF(<double*> sgsr.data,
          <double*> pf.data,
          <unsigned int> n_max,
          <double> bg,
          <double> br,
          <double> pg
    )
    return sgsr


def sgsr_pn(unsigned int n_max, np.ndarray[np.double_t, ndim=1] pf, double bg, double br, double pg):
    """

    :param n_max:
    :param pf:
    :param bg:
    :param br:
    :param pg:
    :return:

    Examples
    --------

    >>> import _sgsr
    >>> import pylab as p
    >>> import numpy as np
    >>> n_max = 101
    >>> pf = np.ones(n_max + 1, dtype=np.float64)
    >>> bg = 1.2
    >>> br = 1.7
    >>> pg = 0.5
    >>> s = _sgsr.sgsr_pn(n_max, pf, bg, br, pg)

    """

    cdef np.ndarray[dtype=np.float64_t, ndim=1] sgsr = np.zeros((n_max + 1) * (n_max + 1), dtype=np.float64, order='C')
    sgsr_pN(<double*> sgsr.data,
          <double*> pf.data,
          <unsigned int> n_max,
          <double> bg,
          <double> br,
          <double> pg
    )
    return sgsr
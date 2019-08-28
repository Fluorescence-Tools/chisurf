import numpy as np
import pylab as p
from scipy.integrate import odeint

import mfm


def stoichometry_matrix(n_species, educts, products, educts_stoichometry, products_stoichometry):
    """
    http://en.wikipedia.org/wiki/Rate_equation

    :param n_species: int
        number of species
    :param educts: list of list
        educts
    :param products: list of list
        procucts
    :param educts_stoichometry: list of list

    :param products_stoichometry: list of list
    :return:

    Examples
    --------

    >>> rs = ReactionSystem()
    >>> rs.add_reaction(educts=[0], products=[1], educt_stoichiometry=[1], product_stoichometry=[1], rate=2)
    Adding new reaction
    -------------------
    1 * [0]  -> 1 * [1]     rate: 2.00000
    >>> rs.n_species
    2
    >>> rs.educts
    [[0]]
    >>> rs.products
    [[1]]
    >>> rs.educts_stoichometry
    [[1]]
    >>> rs.products_stoichometry
    [[1]]
    >>> stoichometry_matrix(rs.n_species, rs.educts, rs.products, rs.educts_stoichometry, rs.products_stoichometry)
    array([[-1.],
    [ 1.]])
    >>> rs.add_reaction(educts=[0, 1], products=[3], educt_stoichiometry=[1, 1], product_stoichometry=[1], rate=0.005)
    Adding new reaction
    -------------------
    1 * [0] + 1 * [1]  -> 1 * [3]   rate: 0.00500

    In the first and second reaction one product is generated. In the first reaction a
    product of the kind [1] is generated in the second reaction a product of the kind [3]

    >>> rs.products_stoichometry
    [[1], [1]]
    >>> rs.products
    [[1], [3]]

    In the first reaction one educt of type [0] is consumed. In the second reaction one educt of type [0] and one
    educt of type [1] are consumed

    >>> rs.educts_stoichometry
    [[1], [1, 1]]
    >>> rs.educts
    [[0], [0, 1]]

    This is also reflected in the stoichometry_matrix

    >>> stoichometry_matrix(rs.n_species, rs.educts, rs.products, rs.educts_stoichometry, rs.products_stoichometry)
    array([[-1., -1.],
       [ 1., -1.],
       [ 0.,  0.],
       [ 0.,  1.]])
    """
    n_reactions = len(educts_stoichometry)
    m = np.zeros((n_species, n_reactions))
    for i, (e, s) in enumerate(zip(educts, educts_stoichometry)):
        for ei, si in zip(e, s):
            m[ei, i] = -si
    for i, (e, s) in enumerate(zip(products, products_stoichometry)):
        for ei, si in zip(e, s):
            m[ei, i] = si
    return m


class ChemicalSpecies(object):

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    def __init__(self, name, description=""):
        self._name = name
        self._description = description


class ReactionSystem(object):

    def __init__(self, **kwargs):
        """

        :param kwargs:
        :return:

        Examples
        --------

        Generate a new reaction system

        >>> from mfm.math.reaction.continuous import ReactionSystem
        >>> rs = ReactionSystem()

        Addition of new uni-molecular reactions
        A -> B and B -> A

        >>> rs.add_reaction(educts=[0], products=[1], educt_stoichiometry=[1], product_stoichometry=[1], rate=5.3333)
        >>> rs.add_reaction(educts=[1], products=[0], educt_stoichiometry=[1], product_stoichometry=[1], rate=8.0)

        Addition of dimerization/Bimolecular reactions
        2 * A -> AA, A + B -> AB, B + B -> BB

        >>> rs.add_reaction(educts=[0], products=[2], educt_stoichiometry=[2], product_stoichometry=[1], rate=0.005)
        >>> rs.add_reaction(educts=[0, 1], products=[3], educt_stoichiometry=[1, 1], product_stoichometry=[1], rate=0.005)
        >>> rs.add_reaction(educts=[1], products=[4], educt_stoichiometry=[2], product_stoichometry=[1], rate=0.005)

        Additional uni-molecular reactions
        AA -> AB, AB -> AA, AB -> BB, BB -> AB

        >>> rs.add_reaction(educts=[2], products=[3], educt_stoichiometry=[1], product_stoichometry=[1], rate=0.025)
        >>> rs.add_reaction(educts=[3], products=[2], educt_stoichiometry=[1], product_stoichometry=[1], rate=0.00125)
        >>> rs.add_reaction(educts=[3], products=[4], educt_stoichiometry=[1], product_stoichometry=[1], rate=0.025)
        >>> rs.add_reaction(educts=[4], products=[3], educt_stoichiometry=[1], product_stoichometry=[1], rate=0.00125)

        Helix-assoziierung
        BB -> HH, HH -> BB

        >>> rs.add_reaction(educts=[4], products=[5], educt_stoichiometry=[1], product_stoichometry=[1], rate=0.005)
        >>> rs.add_reaction(educts=[5], products=[4], educt_stoichiometry=[1], product_stoichometry=[1], rate=0.0005)

        # A, B, AA, AB, BB, HH
        >>> rs.species_brightness = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 0.65])
        # initial value
        #>>> y0 = np.array([1., 0.0, 0.0, 0.0, 0.0, 0.0])
        #>>> t_output = np.linspace(0, 1000, 500)
        #>>> rs.calc()
        #>>> rs.calc(y0, t_output)
        #>>> rs.plot()
        >>> y0 = np.array([100., 0.0, 0.0, 0.0, 0.0, 0.0])
        >>> t_output = np.linspace(0, 1000, 500)
        >>> rs.initial_concentrations = y0
        >>> rs.times = t_output
        >>> rs.calc()
        >>> rs.plot()
        """
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self._concentrations = kwargs.get('concentrations', np.array([[1.0], [1.]], dtype=np.float64))
        self._species_brightness = kwargs.get('species_brightness', np.array([[1.0], [1.]], dtype=np.float64))
        self._times = kwargs.get('times', np.array([0.0, 1.0], dtype=np.float64))

        self.educts = list()
        self.products = list()
        self.educts_stoichometry = list()
        self.products_stoichometry = list()
        self.rates = []
        self._initial_concentrations = []
        self._xmin = 0
        self._xmax = None

    def clear(self):
        self.educts = list()
        self.products = list()
        self.educts_stoichometry = list()
        self.products_stoichometry = list()
        self.rates = []
        self._concentrations = []
        self._initial_concentrations = []
        self._times = None
        self._species_brightness = []

    @property
    def n_reactions(self):
        return len(self.rates)

    @property
    def n_species(self):
        try:
            flat = reduce(lambda x, y: x+y, self.educts + self.products)
            return max(flat) + 1
        except TypeError:
            return 0

    def pop(self, i=-1, verbose=False):
        """
        Removes either the last (if no argument is provided) of the ith reaction from
        the reaction system if an argument is provided

        :param i: int
        :param verbose: bool
        :return:

        Example
        =======
        >>> from mfm.fitting.model.stopped_flow import ReactionSystem
        >>> rs = ReactionSystem()
        >>> rs.add_reaction(educts=[0], products=[1], educt_stoichiometry=[1], product_stoichometry=[1], rate=2)
        >>> rs.add_reaction(educts=[1], products=[0], educt_stoichiometry=[1], product_stoichometry=[1], rate=1)
        >>> rs.n_reactions
        2
        >>> rs.pop()
        >>> rs.n_reactions
        1
        """
        verbose = self.verbose or verbose
        if verbose:
            print("\nRemoved reaction: %i" % i)
            print(self.reaction_string(i))
            print("-----------------------\n")
        self.educts.pop(i)
        self.products.pop(i)
        self.educts_stoichometry.pop(i)
        self.products_stoichometry.pop(i)
        self.rates.pop(i)

    def reaction_string(self, i=-1, include_rate=True):
        """
        Returns a string representing the reaction
        :param int:
        :return:

        Example
        =======
        >>> from mfm.fitting.model.stopped_flow import ReactionSystem
        >>> rs = ReactionSystem()
        >>> rs.add_reaction(educts=[0], products=[1], educt_stoichiometry=[1], product_stoichometry=[1], rate=2)
        >>> rs.add_reaction(educts=[1], products=[0], educt_stoichiometry=[1], product_stoichometry=[1], rate=1)
        >>> rs.reaction_string()
        '1.0 * [1]  -> 1.0 * [0] \trate: 1.00000'
        >>> rs.reaction_string(0)
        '1.0 * [0]  -> 1.0 * [1] \trate: 2.00000'
        """
        educts = self.educts[i]
        products = self.products[i]
        educt_stoichiometry = self.educts_stoichometry[i]
        product_stoichometry = self.products_stoichometry[i]
        rate = self.rates[i]

        s = ""
        for i, a in enumerate(zip(educts, educt_stoichiometry)):
            s += "%s * [%s]" % (a[1], a[0])
            s += " + " if i + 1 < len(educts) else " "
        s += " -> "
        for i, a in enumerate(zip(products, product_stoichometry)):
            s += "%s * [%s]" % (a[1], a[0])
            s += " + " if i + 1 < len(products) else " "
        if include_rate:
            s += "\trate: %.5f" % rate
        return s

    def add_reaction(self, educts, products, educt_stoichiometry, product_stoichometry, rate, fixed=True, verbose=False):
        verbose = self.verbose or verbose
        educt_stoichiometry = np.array(educt_stoichiometry, dtype=np.float64)
        product_stoichometry = np.array(product_stoichometry, dtype=np.float64)

        self.educts.append(educts)
        self.products.append(products)
        self.educts_stoichometry.append(educt_stoichiometry)
        self.products_stoichometry.append(np.array(product_stoichometry, dtype=np.float64))
        r = mfm.parameter.FittingParameter(value=rate)
        self.rates.append(r)
        if verbose:
            print("Adding new reaction")
            print("-------------------")
            print(self.reaction_string() + "\n")

    @property
    def reactions(self):
        educts = self.educts
        products = self.products
        educts_stoichometry = self.educts_stoichometry
        products_stoichometry = self.products_stoichometry
        rates = [r.value for r in self.rates]
        return zip(educts, products, educts_stoichometry, products_stoichometry, rates)

    def rate_equation(self, y, t, reactions):
        """

        :param y: array
            concentrations
        :param t: float
            time - needed because of odeint
        :param reactions: list
            list of tuples as obtained by the attribute :py:attr:`.reactions`
        :return:

        Example
        =======
        >>> from mfm.fitting.model.stopped_flow import ReactionSystem
        >>> rs = ReactionSystem()
        >>> rs.add_reaction(educts=[0], products=[1], educt_stoichiometry=[1], product_stoichometry=[1], rate=2)
        >>> rs.add_reaction(educts=[1], products=[0], educt_stoichiometry=[1], product_stoichometry=[1], rate=1)
        >>> reactions = rs.reactions
        >>> c0 = rs.initial_concentrations
        >>> rs.rate_equation(c0, 0.0, reactions)
        """
        re = np.zeros_like(y)
        for e, p, es, ps, r in reactions:
            fr = (y[e]**es).prod() * r
            re[e] -= fr * es
            re[p] += fr * ps
        return re

    @property
    def species_fractions(self):
        return (self.concentrations.T / self.concentrations.sum(axis=1)).T

    @property
    def initial_concentrations(self):
        return np.array([c.value for c in self._initial_concentrations], dtype=np.float64)

    @initial_concentrations.setter
    def initial_concentrations(self, v):
        self._initial_concentrations = [mfm.parameter.FittingParameter(value=vi) for vi in v]

    @property
    def concentrations(self):
        return self._concentrations

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, v):
        self._times = v

    def calc(self):
        """
        Integrates the differential equations.
        :return:
        """
        rs = self
        res = odeint(
            rs.rate_equation,
            rs.initial_concentrations.flatten(),
            rs.times,
            args=(rs.reactions, ),
            full_output=True,
            mxstep=15000000
        )
        self._concentrations = res[0]

    def plot(self, t_divisor=1.0, normalize=False, show=True):
        """
        Generates a plot of the currently calculated time dependent concentrations
        :param t_divisor: float
            The time-axis is divided byt this number
        :param normalize: bool
            If True the signal intensity is divided by the maximum intensity.
        :param show: bool
            If True the generated Matplotlib plot is shown
        :return:
        """
        t = self.times
        y = self.signal_intensity
        if normalize:
            y = y / max(y)
        p.subplot(2, 2, 1)
        p.plot(t / t_divisor, y)

        p.subplot(2, 2, 2)
        xs = self.species_fractions
        for i, y in enumerate(xs.T):
            p.plot(t / t_divisor, y, label='%i' % i)
        p.legend()

        p.subplot(2, 2, 3)
        xs = self.concentrations
        for i, y in enumerate(xs.T):
            p.plot(t / t_divisor, y, label='%i' % i)
        p.legend()
        if show:
            p.show()

    @property
    def species_brightness(self):
        if isinstance(self._species_brightness, np.ndarray):
            return self._species_brightness
        else:
            return [v.value for v in self._species_brightness]

    @species_brightness.setter
    def species_brightness(self, v):
        self._species_brightness = v

    @property
    def signal_intensity(self):
        sf = self.concentrations
        q = self.species_brightness
        return np.dot(sf, q)

    def __str__(self):
        s = "Species\n"
        s += "--------\n"
        s += "Id\tbrightness"
        for i, b in enumerate(self.species_brightness):
            s += "%i\t%.2f\n" % (i, b)
        s += "\n\n"
        s += "Reactions\n"
        s += "---------\n"
        s += "Educts\tEducts-S\tProducts\tProducts-S\tRates\n"
        for i in range(self.n_reactions):
            s += self.reaction_string(i)
            s += "\n"
        return s
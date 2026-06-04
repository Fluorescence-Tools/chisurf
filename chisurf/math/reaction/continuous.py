from __future__ import annotations
from chisurf import typing

import numpy as np
import pylab as p
from scipy.integrate import odeint

import chisurf.parameter


def stoichometry_matrix(
        n_species: int,
        educts,
        products,
        educts_stoichometry,
        products_stoichometry
):
    """
    Computes the stoichiometry matrix for a given chemical reaction system.

    This function constructs a stoichiometry matrix for a set of chemical reactions
    based on the number of species involved, their educts, products, and their respective
    stoichiometry coefficients. The resulting matrix provides information about
    how each reaction transforms the number of molecules of each species.

    Parameters:
        n_species (int): The total number of chemical species in the system.
        educts (List[List[int]]): The indices of reactant species for each reaction.
        products (List[List[int]]): The indices of product species for each reaction.
        educts_stoichometry (List[List[float]]): The stoichiometric coefficients
            for reactants of each reaction.
        products_stoichometry (List[List[float]]): The stoichiometric coefficients
            for products of each reaction.

    Returns:
        np.ndarray: A 2D array of shape (n_species, n_reactions) representing the
            stoichiometry matrix, where rows correspond to species and columns
            correspond to reactions.
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
    """
    Represents a chemical species with a name and an optional description.

    This class is used to define chemical species by providing a name and an optional
    description, which describe the species' identity and details. It allows retrieval
    of these properties using the relevant accessors.
    """
    @property
    def name(self) -> str:
        """
        Represents the getter method for the 'name' property of an object, which
        returns the stored private attribute '_name'.

        @property
            Retrieves the name attribute stored in the '_name' instance attribute.

        Returns
        -------
        str
            The value of the '_name' attribute representing the name.
        """
        return self._name

    @property
    def description(self) -> str:
        """
        Property to get the description attribute of the object.

        The `description` property provides access to the private `_description`
        attribute. It does not allow modification of the attribute and is used
        to retrieve the stored value representing the object's descriptive information.

        @return: The value of the `_description` attribute.
        @rtype: str
        """
        return self._description

    def __init__(
            self,
            name: str,
            description: str = ""
    ):
        """
            Initializes an instance of the class to handle specific data and
            assign initial values to attributes.

            Parameters
            ----------
            name : str
                A name to uniquely identify the instance.
            description : str, optional
                A brief description of the instance, defaults to an empty string.
        """
        self._name = name
        self._description = description


class ReactionSystem(object):
    """
    ReactionSystem is a computational framework to model, analyze, and simulate
    chemical reactions in continuous time. It supports uni-molecular and
    bi-molecular reactions with defined stoichiometries and rates. The class
    allows storage, modification, and simulation of reactions, providing tools
    to analyze their dynamics.

    Parameters
    ----------
    verbose : bool
        Controls if verbose operation mode is active. Default value is `chisurf.settings.cs_settings['verbose']`.
    **kwargs :
        Additional keyword arguments to pass initialization data.

    Attributes
    ----------
    verbose : bool
        Indicates if verbose mode is enabled.
    educts : list
        List of educt species in the system for each reaction.
    products : list
        List of product species in the system for each reaction.
    educts_stoichometry : list
        Stoichiometric coefficients of educts for each reaction.
    products_stoichometry : list
        Stoichiometric coefficients of products for each reaction.
    rates : list
        Reaction rate constants for each reaction in the system.
    _concentrations : numpy.ndarray
        Array holding concentration values of species in the system.
    _species_brightness : numpy.ndarray
        Array defining individual species' brightness.
    _times : numpy.ndarray
        Array of time points for analyzing concentration dynamics.
    _initial_concentrations : list
        Initial concentrations of species in the system.
    _xmin : float
        Minimum x-axis value for plotting or simulation.
    _xmax : float
        Maximum x-axis value for plotting or simulation.

    Examples
    --------
    Usage to model complex chemical reaction systems with various reaction
    types, define associated properties, and simulate temporal dynamics.
    """
    def __init__(
            self,
            verbose: bool = None,
            **kwargs
    ):
        """
            Initializes the class with optional configurations for verbosity and additional
            attributes used within the model. The class includes attributes for managing
            concentrations, brightness of species, and temporal data. Additionally, lists for
            reactants, products, stoichiometry, and rates are initialized for chemical
            reaction modeling.

            Parameters
            ----------
            verbose : bool, optional
                Indicates whether verbose mode is enabled. Default is the value of chisurf.settings.cs_settings['verbose'].
            **kwargs : dict, optional
                Arbitrary keyword arguments that may include:
                - 'concentrations' : np.ndarray, optional
                    A 2D numpy array specifying the concentrations of species.
                - 'species_brightness' : np.ndarray, optional
                    A 2D numpy array specifying the brightness values of species.
                - 'times' : np.ndarray, optional
                    A 1D numpy array specifying time points related to the reactions.

            Attributes
            ----------
            verbose : bool
                Stores the verbosity state of the object.
            _concentrations : np.ndarray
                A matrix representing concentrations of species over time.
            _species_brightness : np.ndarray
                A matrix representing the brightness of species.
            _times : np.ndarray
                A list of time values for the reaction system.
            educts : list
                List of educt reactants involved in reactions.
            products : list
                List of product species resulting from reactions.
            educts_stoichometry : list
                Stochastic coefficients for each educt in reactions.
            products_stoichometry : list
                Stochastic coefficients for each product in reactions.
            rates : list
                Rate constants associated with the reactions.
            _initial_concentrations : list
                Initial concentrations of species in the system.
            _xmin : int
                Minimum time or x-axis limit for simulations/representations.
            _xmax : int, optional
                Maximum time or x-axis limit for simulations/representations.
        """
        if verbose is None:
            verbose = chisurf.settings.cs_settings['verbose']
        self.verbose = verbose
        self._concentrations = kwargs.get('concentrations', np.array([[1.0], [1.]], dtype=np.float64))
        self._species_brightness = kwargs.get(
            'species_brightness', np.array([[1.0], [1.]], dtype=np.float64)
        )
        self._times = kwargs.get(
            'times', np.array([0.0, 1.0], dtype=np.float64)
        )

        self.educts = list()
        self.products = list()
        self.educts_stoichometry = list()
        self.products_stoichometry = list()
        self.rates = []
        self._initial_concentrations = []
        self._xmin = 0
        self._xmax = None

    def clear(self):
        """
        Clears all the internal attributes of the object, resetting them to their initial empty
        or default states. This method is typically used to reset the object to a clean slate
        for reuse or reinitialization.

        Attributes reset include:
        - educts, products: Lists of chemical entities involved in a reaction.
        - educts_stoichometry, products_stoichometry: Stoichiometric coefficients of the
          respective chemical entities.
        - rates: Reaction rates associated with the chemical process.
        - _concentrations, _initial_concentrations: Internal data structures storing
          concentration values over time.
        - _times: Internal attribute for storing time data.
        - _species_brightness: Internal attribute for storing species' brightness data.

        Raises:
            No exceptions are raised by this method. It only reinitializes attributes.
        """
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
        """
        Provides the count of reactions based on the number of rates.

        Summary:
        This property method calculates the total number of reactions by
        determining the length of the rates attribute. It assumes that each
        individual rate corresponds to exactly one reaction.

        Returns:
            int: The total number of reactions.
        """
        return len(self.rates)

    @property
    def n_species(self):
        """
            Retrieves the number of unique species involved in the reaction.

            This property calculates the total number of species (educts and products)
            by identifying the highest species index and adding 1. If there are no
            valid educts or products, it returns 0.

            @return: The total number of unique species as an integer.
        """
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
        >>> from chisurf.models.stopped_flow import ReactionSystem
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
        >>> from chisurf.models.stopped_flow import ReactionSystem
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

    def add_reaction(
            self,
            educts,
            products,
            educt_stoichiometry,
            product_stoichometry,
            rate,
            fixed=True,
            verbose=False
    ):
        """Add a reaction to the reaction system.

        Parameters
        ----------
        educts : List[int]
            Indices of reactant species.
        products : List[int]
            Indices of product species.
        educt_stoichiometry : List[float]
            Stoichiometric coefficients of the reactants.
        product_stoichometry : List[float]
            Stoichiometric coefficients of the products.
        rate : float
            Rate constant of the reaction.
        fixed : bool, optional
            Whether the rate parameter is fixed (default is True).
        verbose : bool, optional
            If True, print information about the added reaction.
        """
        verbose = self.verbose or verbose
        educt_stoichiometry = np.array(
            educt_stoichiometry,
            dtype=np.float64
        )
        product_stoichometry = np.array(
            product_stoichometry,
            dtype=np.float64
        )

        self.educts.append(educts)
        self.products.append(products)
        self.educts_stoichometry.append(educt_stoichiometry)
        self.products_stoichometry.append(
            np.array(
                product_stoichometry,
                dtype=np.float64
            )
        )
        r = chisurf.parameter.Parameter(
            value=rate
        )
        self.rates.append(r)
        if verbose:
            print("Adding new reaction")
            print("-------------------")
            print(self.reaction_string() + "\n")

    @property
    def reactions(self):
        """Get an iterator over all reactions.

        Yields
        ------
        tuple
            A tuple ``(educts, products, educt_stoichiometry,
            product_stoichiometry, rate_value)`` for each reaction.
        """
        educts = self.educts
        products = self.products
        educts_stoichometry = self.educts_stoichometry
        products_stoichometry = self.products_stoichometry
        rates = [r.value for r in self.rates]
        return zip(educts, products, educts_stoichometry, products_stoichometry, rates)

    def rate_equation(
            self,
            y,
            t,
            reactions
    ):
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
        >>> from chisurf.models.stopped_flow import ReactionSystem
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
        """Get the fractional concentration of each species over time.

        Returns
        -------
        np.ndarray
            Array of shape (n_times, n_species) where each row sums to one.
        """
        return (self.concentrations.T / self.concentrations.sum(axis=1)).T

    @property
    def initial_concentrations(self):
        """Get the initial concentrations of all species.

        Returns
        -------
        np.ndarray
            1-D array of initial concentration values.
        """
        return np.array(
            [
                c.value for c in self._initial_concentrations
            ], dtype=np.float64
        )

    @initial_concentrations.setter
    def initial_concentrations(self, v):
        """Set the initial concentrations of all species.

        Parameters
        ----------
        v : list of float
            Initial concentration values.
        """
        self._initial_concentrations = [
            chisurf.fitting.parameter.FittingParameter(value=vi) for vi in v
        ]

    @property
    def concentrations(self):
        """Get the time-dependent concentrations.

        Returns
        -------
        np.ndarray
            Array of shape (n_times, n_species) with concentration values.
        """
        return self._concentrations

    @property
    def times(self):
        """Get the time points.

        Returns
        -------
        np.ndarray
            1-D array of time values.
        """
        return self._times

    @times.setter
    def times(
            self,
            v
    ):
        """Set the time points.

        Parameters
        ----------
        v : np.ndarray
            1-D array of time values.
        """
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

    def plot(
            self,
            t_divisor: float = 1.0,
            normalize: bool = False,
            show: bool = True
    ) -> None:
        """
        Generates a plot of the currently calculated time dependent
        concentrations
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
    def species_brightness(self) -> typing.List[float]:
        """Get the brightness values for each species.

        Returns
        -------
        list of float or np.ndarray
            Brightness coefficients for each species.
        """
        if isinstance(self._species_brightness, np.ndarray):
            return self._species_brightness
        else:
            return [v.value for v in self._species_brightness]

    @species_brightness.setter
    def species_brightness(
            self,
            v
    ):
        """Set the brightness values for each species.

        Parameters
        ----------
        v : list or np.ndarray
            Brightness coefficients.
        """
        self._species_brightness = v

    @property
    def signal_intensity(self) -> np.ndarray:
        """Get the total signal intensity over time.

        The signal is the dot product of concentrations and species brightness.

        Returns
        -------
        np.ndarray
            1-D array of total intensity at each time point.
        """
        sf = self.concentrations
        q = self.species_brightness
        return np.dot(sf, q)

    def __str__(self):
        """Return a string summary of the reaction system.

        Returns
        -------
        str
            Formatted table of species brightnesses and reactions.
        """
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

from chisurf import typing

import numpy as np

import chisurf.core.experiments
import chisurf.core.fitting
import chisurf.core.models
import chisurf.core.models.tcspc.fret


class FRETLineGenerator(object):

    """Calculates FRET-lines for arbitrary model

    :param kwargs:

    Attributes
    ----------
    parameter_name : str
        The parameter name of the variable which is used to generate the FRET-line.
    fret_species_averaged_lifetime : float
        The species averages lifetime of the model.
    fret_fluorescence_averaged_lifetime : float
        The fluorescence averaged lifetime of the model.
    donor_species_averaged_lifetime : float
        The species averaged lifetime of the donor of the model.
    transfer_efficiency : float
        The transfer efficiency of the model.
    parameter_values : array
        The parameter values used for the calculation of the FRET-line
    parameter_range : array
        Start and stop value of the parameter range used for calculation of the FRET-line
    n_points : int
        The number of points used to calculate the FRET-line
    model : LifetimeModel
        The TCSPC-model used to calculate the FRET-line

    Methods
    -------
    calc()
        Recaclulates the FRET-line

    Examples
    --------

    >>> from chisurf.core.fluorescence.fret.fret_line import FRETLineGenerator  # doctest: +SKIP
    >>> import chisurf.core.models.tcspc as m  # doctest: +SKIP
    >>> R1 = 80  # doctest: +SKIP
    >>> R2 = 35  # doctest: +SKIP
    >>> fl = FRETLineGenerator()  # doctest: +SKIP
    >>> fl.model = m.fret.GaussianModel  # doctest: +SKIP

    Adding donor lifetimes
    >>> fl.model.donors.append(1.0, 4)  # doctest: +SKIP
    >>> fl.model.donors.append(1.0, 2)  # doctest: +SKIP

    Add a new Gaussian distance
    >>> fl.model.append(55.0, 10, 1.0)  # doctest: +SKIP

    The fluorescence/species averaged lifetime of the model is obtained by
    >>> fl.model.find_parameters()  # doctest: +SKIP
    >>> fl.model.parameter_dict['xDOnly'].value = 0.0  # doctest: +SKIP

    >>> fl.fret_fluorescence_averaged_lifetime  # doctest: +SKIP
    2.5600809847436152
    >>> fl.fret_species_averaged_lifetime  # doctest: +SKIP
    2.2459102590812412

    The model parameters can be changed using their names by the *parameter_dict*

    >>> fl.model.parameter_dict['R(G,1)'].value = 40.0  # doctest: +SKIP
    >>> fl.model.parameter_dict['s(G,1)'].value = 8.0  # doctest: +SKIP

    Set the name of the parameter changes to calculate the distributions used for the
    FRET-lines. Here a more common parameter as the donor-acceptor separation distance
    is used to generate a static FRET-line.

    >>> fl.parameter_name = 'R(G,1)'  # doctest: +SKIP

    Set the range in which this parameter is modified to generate the line and calculate the FRET-line

    >>> fl.parameter_range = 0.1, 1000  # doctest: +SKIP
    >>> fl.calc()  # doctest: +SKIP

    By adding a second Gaussian and changing the species fraction you can also generate a dynamic-FRET line

    .. plot:: plots/fret-lines.py


    """

    def __init__(
            self,
            model: typing.Type[chisurf.core.models.tcspc.lifetime.LifetimeModel] = None,
            polynomial_degree: int = 4,
            quantum_yield_donor: float = 0.8,
            quantum_yield_acceptor: float = 0.32,
            parameter_name: str = None,
            n_points: int = 100,
            parameter_range: typing.Tuple[float, float] = (0.1, 100.0),
            fluorescence_decay_range: typing.Tuple[float, float] = (0, 500.0),
            verbose: bool = None,
            **kwargs
    ):
        """

        :param model: a fluorescence lifetime model that is used to compute a FRET line
        :param polynomial_degree: degree of the polynomial to approximate the conversion function
        :param quantum_yield_donor: fluorescence quantum yield of the donor in the absence of FRET
        :param quantum_yield_acceptor: fluorescence quantum yield of the acceptor
        :param parameter_name: the parameter that is varied to compute the FRET line
        :param n_points: number of points of the FRET line
        :param parameter_range: range of the parameter
        :param fluorescence_decay_range:
        :param verbose:
        :param kwargs:
        """
        if verbose is None:
            verbose = chisurf.core.settings.cs_settings['verbose']
        self.polynomial_degree = polynomial_degree
        self.quantum_yield_donor = quantum_yield_donor
        self.quantum_yield_acceptor = quantum_yield_acceptor
        self.parameter_name = parameter_name
        self.n_points = n_points
        self.parameter_range = parameter_range
        self.verbose = verbose

        self.fit = chisurf.core.fitting.fit.Fit()
        x = np.linspace(
                start=fluorescence_decay_range[0],
                stop=fluorescence_decay_range[1],
                num=n_points
            )
        y = np.zeros_like(x)
        self.fit.data = chisurf.core.data.DataCurve(
            x=x, y=y
        )
        if model is not None:
            self.model = model

        self.fret_efficiencies = np.zeros(
            self.n_points,
            dtype=np.float
        )
        self.fluorescence_averaged_lifetimes = np.zeros_like(
            self.fret_efficiencies
        )
        self.species_averaged_lifetimes = np.zeros_like(
            self.fluorescence_averaged_lifetimes
        )

    @property
    def conversion_function(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        The fluorescence averaged lifetime and species averaged lifetime
        """
        return self.fluorescence_averaged_lifetimes, self.species_averaged_lifetimes

    @property
    def fret_species_averaged_lifetime(self) -> float:
        """
        The current species averages lifetime of the FRET sample xi * taui
        """
        return self.model.species_averaged_lifetime

    @property
    def fret_fluorescence_averaged_lifetime(self) -> float:
        """
        The current fluorescence averaged lifetime of the
        FRET-sample = xi*taui**2 / species_averaged_lifetime
        """
        return self.model.fluorescence_averaged_lifetime

    @property
    def donor_species_averaged_lifetime(self) -> float:
        """
        The current species averaged lifetime of the donor sample xi*taui
        """
        return self.model.lifetimes.species_averaged_lifetime

    @property
    def transfer_efficiency(self) -> float:
        """
        The current transfer efficency
        """
        return 1.0 - self.fret_species_averaged_lifetime / self.donor_species_averaged_lifetime

    @property
    def polynom_coefficients(self):
        """
        A numpy array with polynomial coefficients approximating the tauX(tauF) conversion function
        """
        x, y = self.fluorescence_averaged_lifetimes, self.species_averaged_lifetimes
        return np.polyfit(x, y, self.polynomial_degree)

    @property
    def conversion_function_string(self) -> str:
        """
        A string used for plotting of the conversion function tauX(tauF)
        """
        c = self.polynom_coefficients
        s = ""
        for i, c in enumerate(c[::-1]):
            s += "%.6f*x^%i+" % (c, i)
        return s[:-1]

    @property
    def transfer_efficency_string(self) -> str:
        """
        Used for instance for plotting in origin
        """
        return "1.0-(%s)/(%.6f)" % (self.conversion_function_string, self.donor_species_averaged_lifetime)

    @property
    def fdfa_string(self) -> str:
        """
        Used for instance for plotting in origin
        """
        qd = self.quantum_yield_donor
        qa = self.quantum_yield_acceptor
        return "%s/%s / ((%s)/(%s) - 1)" % (qd, qa, self.donor_species_averaged_lifetime, self.conversion_function_string)

    @property
    def parameter_values(self):
        """
        The values the parameter as defined by :py:attr:`~chisurf.core.fluorescence.sm_FRETlines.FRETLineGenerator.parameter_name`
        """
        start, stop = self.parameter_range
        n_points = self.n_points
        return np.linspace(
            start=start,
            stop=stop,
            num=n_points
        )

    @property
    def model(self) -> chisurf.core.models.tcspc.lifetime.LifetimeModel:
        """The fluorescence lifetime model used for FRET-line calculations."""
        return self._model

    @model.setter
    def model(
            self,
            v: typing.Type[chisurf.core.models.tcspc.lifetime.LifetimeModel]
    ):
        """Set the fluorescence lifetime model.

        Parameters
        ----------
        v : type
            A LifetimeModel subclass that will be instantiated with the
            current Fit object.
        """
        self._model = v(
            fit=self.fit
        )

    def update(
            self,
            parameter_name: str = None,
            parameter_range: typing.Tuple[float, float] = None,
            verbose: bool = None,
            n_points: int = None
    ):
        """Recompute the FRET-line by varying the chosen parameter.

        Parameters
        ----------
        parameter_name : str, optional
            Name of the parameter to vary.
        parameter_range : tuple of float, optional
            (min, max) range for the parameter.
        verbose : bool, optional
            If True, print progress information.
        n_points : int, optional
            Number of points along the FRET-line.
        """
        if verbose is None:
            verbose = self.verbose
        if isinstance(n_points, int):
            self.n_points = n_points
        else:
            n_points = self.n_points

        if isinstance(parameter_name, str):
            self.parameter_name = parameter_name
        else:
            parameter_name = self.parameter_name

        if isinstance(parameter_range, list):
            self.parameter_range = parameter_range

        if verbose:
            print("Calculating FRET-Line")
            print("Using parameter: %s" % parameter_name)
            print("In a range: %.1f .. %.1f" % (self.parameter_range[0], self.parameter_range[1]))

        transfer_efficiencies = np.zeros(n_points)
        fluorescence_averaged_lifetimes = np.zeros(n_points)
        species_averaged_lifetimes = np.zeros(n_points)

        for i, parameter_value in enumerate(self.parameter_values):
            self.model.parameter_dict[parameter_name].value = parameter_value
            fluorescence_averaged_lifetimes[i] = self.fret_fluorescence_averaged_lifetime
            transfer_efficiencies[i] = self.transfer_efficiency
            species_averaged_lifetimes[i] = self.fret_species_averaged_lifetime

        self.fret_efficiencies = transfer_efficiencies
        self.species_averaged_lifetimes = species_averaged_lifetimes
        self.fluorescence_averaged_lifetimes = fluorescence_averaged_lifetimes


class StaticFRETLine(
    FRETLineGenerator
):
    """
    This class is used to calculate static-FRET lines of Gaussian distributed states.


    Examples
    --------

    >>> import chisurf.tools.sm_FRETlines as sm_FRETlines  # doctest: +SKIP
    >>> s = sm_FRETlines.StaticFRETLine()  # doctest: +SKIP
    >>> s.calc()  # doctest: +SKIP

    Now lets look at the conversion function in comparison to a 1:1 relation

    >>> import pylab as p  # doctest: +SKIP
    >>> x, y = s.conversion_function  # doctest: +SKIP
    >>> p.plot(x, y)  # doctest: +SKIP
    >>> p.plot(x, x)  # doctest: +SKIP
    >>> p.show()  # doctest: +SKIP

    This class has basically only one relevant attribute that is the width of the DA-distance distirbution
    within a state.

    A conversion polynomial for plotting purposes can be obtained

    >>> s.polynomial_string  # doctest: +SKIP

    """

    @property
    def sigma(self):
        """
        Width of the DA-distance distribution within the state
        """
        return self.model.parameter_dict['s(G,1)'].value

    @sigma.setter
    def sigma(self, v):
        """Set the width of the DA-distance distribution."""
        self.model.parameter_dict['s(G,1)'].value = v

    def __init__(self, **kwargs):
        """Initialize a StaticFRETLine with a default Gaussian model.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to FRETLineGenerator.
        """
        super().__init__(**kwargs)
        self.model = chisurf.core.models.tcspc.fret.GaussianModel
        self.model.gaussians.append(55.0, 10, 1.0)
        self.model.find_parameters()
        self.model.parameter_dict['xDOnly'].value = 0.0

    def update(
            self,
            parameter_name: str = 'R(G,1)',
            parameter_range: typing.Tuple[float, float] = None,
            verbose: bool = None,
            n_points: int = None
    ):
        """Recompute the static FRET-line by varying the distance parameter.

        Parameters
        ----------
        parameter_name : str, optional
            Name of the parameter to vary (default 'R(G,1)').
        parameter_range : tuple of float, optional
            (min, max) range for the parameter.
        verbose : bool, optional
            If True, print progress information.
        n_points : int, optional
            Number of points along the FRET-line.
        """
        self.model.parameter_dict['x(G,1)'].value = 1.0
        super().update(
            parameter_name=parameter_name,
            parameter_range=parameter_range,
            verbose=verbose,
            n_points=n_points
        )


class DynamicFRETLine(FRETLineGenerator):
    """
    This is a `convenience` class for a simple two-state Gaussian distance distribution
    dynamic FRET-model. Two Gaussian distance distributions are considered as limiting
    states and the dynamic-FRET line is calculated


    Examples
    --------

    >>> import chisurf.tools.sm_FRETlines as sm_FRETlines  # doctest: +SKIP
    >>> d = sm_FRETlines.DynamicFRETLine()  # doctest: +SKIP
    >>> print d.model  # doctest: +SKIP
    Model: Gaussian-Donor
    Parameter       Value   Bounds  Fixed   Linke
    bg      0.0000  (None, None)    False   False
    sc      0.0000  (None, None)    False   False
    t(L,1)  4.0000  (None, None)    False   False
    x(L,1)  1.0000  (None, None)    False   False
    t0      4.1000  (None, None)    True    False
    R0      52.0000 (None, None)    True    False
    k2      0.6670  (None, None)    True    False
    DOnly   0.0000  (0.0, 1.0)      False   False
    x(G,1)  1.0000  (None, None)    False   False
    x(G,2)  1.0000  (None, None)    False   False
    s(G,1)  10.0000 (None, None)    False   False
    s(G,2)  10.0000 (None, None)    False   False
    R(G,1)  40.0000 (None, None)    False   False
    R(G,2)  80.0000 (None, None)    False   False
    dt      0.1000  (None, None)    True    False
    start   0.0000  (None, None)    True    False
    stop    20.0000 (None, None)    True    False
    rep     10.0000 (None, None)    True    False
    lb      0.0000  (None, None)    False   False
    ts      0.0000  (None, None)    False   False
    p0      1.0000  (None, None)    True    False
    r0      0.3800  (None, None)    False   False
    l1      0.0308  (None, None)    False   False
    l2      0.0368  (None, None)    False   False
    g       1.0000  (None, None)    False   False

    Set a new sigma

    >>> d.sigma = 3.0  # doctest: +SKIP
    >>> print d.model  # doctest: +SKIP
    Model: Gaussian-Donor
    Parameter       Value   Bounds  Fixed   Linke
    bg      0.0000  (None, None)    False   False
    sc      0.0000  (None, None)    False   False
    t(L,1)  4.0000  (None, None)    False   False
    x(L,1)  1.0000  (None, None)    False   False
    t0      4.1000  (None, None)    True    False
    R0      52.0000 (None, None)    True    False
    k2      0.6670  (None, None)    True    False
    DOnly   0.0000  (0.0, 1.0)      False   False
    x(G,1)  1.0000  (None, None)    False   False
    x(G,2)  1.0000  (None, None)    False   False
    s(G,1)  3.0000  (None, None)    False   False
    s(G,2)  3.0000  (None, None)    False   False
    R(G,1)  40.0000 (None, None)    False   False
    R(G,2)  80.0000 (None, None)    False   False
    dt      0.1000  (None, None)    True    False
    start   0.0000  (None, None)    True    False
    stop    20.0000 (None, None)    True    False
    rep     10.0000 (None, None)    True    False
    lb      0.0000  (None, None)    False   False
    ts      0.0000  (None, None)    False   False
    p0      1.0000  (None, None)    True    False
    r0      0.3800  (None, None)    False   False
    l1      0.0308  (None, None)    False   False
    l2      0.0368  (None, None)    False   False
    g       1.0000  (None, None)    False   False

    Calculate the FRET-line

    >>> d.calc()  # doctest: +SKIP
    Calculating FRET-Line
    Using parameter: x(G,2)
    In a range: 0.1 .. 100.0

    Plot the conversion function

    >>> import pylab as p  # doctest: +SKIP
    >>> tauf, taux = d.conversion_function  # doctest: +SKIP
    >>> p.plot(tauf, taux)  # doctest: +SKIP
    >>> p.show()  # doctest: +SKIP
    """

    def __init__(
            self,
            distance_1: float = 40.0,
            distance_2: float = 80.0,
            sigma_1: float = 6.0,
            sigma_2: float = 6.0,
            *args,
            **kwargs
    ):
        """
            model: typing.Type[chisurf.core.models.tcspc.lifetime.LifetimeModel] = None,
            polynomial_degree: int = 4,
            quantum_yield_donor: float = 0.8,
            quantum_yield_acceptor: float = 0.32,
            parameter_name: str = None,
            n_points: int = 100,
            parameter_range: typing.Tuple[float, float] = (0.1, 100.0),
            fluorescence_decay_range: typing.Tuple[float, float] = (0, 500.0),
            verbose: bool = None,
            **kwargs

        :param args:
        :param kwargs:
        """
        super().__init__(
            *args,
            **kwargs
        )
        self.model = chisurf.core.models.tcspc.fret.GaussianModel
        self.model.gaussians.append(distance_1, sigma_1, 1.0)
        self.model.gaussians.append(distance_2, sigma_2, 1.0)
        self.model.find_parameters()
        self.model.parameter_dict['xDOnly'].value = 0.0

    @property
    def mean_distance_1(self):
        """
        Mean distance of first limiting state 1
       """
        return self.model.parameter_dict['R(G,1)'].value

    @mean_distance_1.setter
    def mean_distance_1(self, v):
        """Set the mean distance of first limiting state."""
        self.model.parameter_dict['R(G,1)'].value = v

    @property
    def mean_distance_2(self):
        """
        Mean distance of first limiting state 2
       """
        return self.model.parameter_dict['R(G,2)'].value

    @mean_distance_2.setter
    def mean_distance_2(self, v):
        """Set the mean distance of second limiting state."""
        self.model.parameter_dict['R(G,2)'].value = v

    @property
    def sigma_1(self):
        """
        Width of first limiting state
       """
        return self.model.parameter_dict['s(G,1)'].value

    @sigma_1.setter
    def sigma_1(self, v):
        """Set the width of first limiting state."""
        self.model.parameter_dict['s(G,1)'].value = v

    @property
    def sigma_2(self):
        """
        Width of second limiting state
       """
        return self.model.parameter_dict['s(G,2)'].value

    @sigma_2.setter
    def sigma_2(self, v):
        """Set the width of second limiting state."""
        self.model.parameter_dict['s(G,2)'].value = v

    @property
    def sigma(self):
        """
        The width of both sigmas
       """
        return self.model.parameter_dict['s(G,1)'].value, self.model.parameter_dict['s(G,2)'].value

    @sigma.setter
    def sigma(self, v):
        """Set the width of both limiting states.

        Parameters
        ----------
        v : float or tuple of float
            If float, both states get the same width. If tuple, (sigma_1, sigma_2).
        """
        try:
            self.model.parameter_dict['s(G,1)'].value = v[0]
            self.model.parameter_dict['s(G,2)'].value = v[1]
        except TypeError:
            self.model.parameter_dict['s(G,1)'].value = v
            self.model.parameter_dict['s(G,2)'].value = v

    def update(
            self,
            parameter_name: str = None,
            parameter_range: typing.Tuple[float, float] = None,
            verbose: bool = None,
            n_points: int = None
    ):
        """Recompute the dynamic FRET-line by varying the species fraction.

        Parameters
        ----------
        parameter_name : str, optional
            Name of the parameter to vary.
        parameter_range : tuple of float, optional
            (min, max) range for the parameter.
        verbose : bool, optional
            If True, print progress information.
        n_points : int, optional
            Number of points along the FRET-line.
        """
        self.model.parameter_dict['x(G,1)'].value = 1.0
        self.model.parameter_dict['x(G,2)'].value = 0.0
        FRETLineGenerator.update(
            self,
            parameter_name='x(G,2)',
            parameter_range=(0, 10),
            n_points=n_points,
            verbose=verbose
        )

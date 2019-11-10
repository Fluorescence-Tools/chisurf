import typing

import numpy as np

import chisurf.experiments
import chisurf.fitting
import chisurf.models


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

    >>> from chisurf.fluorescence.fret.fret_line import FRETLineGenerator    >>> import chisurf.models.tcspc as m
    >>> R1 = 80
    >>> R2 = 35
    >>> fl = FRETLineGenerator()
    >>> fl.model = m.fret.GaussianModel

    Adding donor lifetimes
    >>> fl.model.donors.append(1.0, 4)
    >>> fl.model.donors.append(1.0, 2)

    Add a new Gaussian distance
    >>> fl.model.append(55.0, 10, 1.0)

    The fluorescence/species averaged lifetime of the model is obtained by
    >>> fl.model.find_parameters()
    >>> fl.model.parameter_dict['xDOnly'].value = 0.0

    >>> fl.fret_fluorescence_averaged_lifetime
    2.5600809847436152
    >>> fl.fret_species_averaged_lifetime
    2.2459102590812412

    The model parameters can be changed using their names by the *parameter_dict*

    >>> fl.model.parameter_dict['R(G,1)'].value = 40.0
    >>> fl.model.parameter_dict['s(G,1)'].value = 8.0

    Set the name of the parameter changes to calculate the distributions used for the
    FRET-lines. Here a more common parameter as the donor-acceptor separation distance
    is used to generate a static FRET-line.

    >>> fl.parameter_name = 'R(G,1)'

    Set the range in which this parameter is modified to generate the line and calculate the FRET-line

    >>> fl.parameter_range = 0.1, 1000
    >>> fl.calc()

    By adding a second Gaussian and changing the species fraction you can also generate a dynamic-FRET line

    .. plot:: plots/fret-lines.py


    """

    def __init__(
            self,
            model: typing.Type[chisurf.models.tcspc.lifetime.LifetimeModel] = None,
            polynomial_degree: int = 4,
            quantum_yield_donor: float = 0.8,
            quantum_yield_acceptor: float = 0.32,
            parameter_name: str = None,
            n_points: int = 100,
            parameter_range: typing.Tuple[float, float] = (0.1, 100.0),
            fluorescence_decay_range: typing.Tuple[float, float] = (0, 500.0),
            verbose: bool = chisurf.verbose,
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
        self.polynomial_degree = polynomial_degree
        self.quantum_yield_donor = quantum_yield_donor
        self.quantum_yield_acceptor = quantum_yield_acceptor
        self.parameter_name = parameter_name
        self.n_points = n_points
        self.parameter_range = parameter_range
        self.verbose = verbose

        self.fit = chisurf.fitting.fit.Fit()
        x = np.linspace(
                start=fluorescence_decay_range[0],
                stop=fluorescence_decay_range[1],
                num=n_points
            )
        y = np.zeros_like(x)
        self.fit.data = chisurf.experiments.data.DataCurve(
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
    def polynom_coefficients(
            self
    ):
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
            s += "%s*x^%i+" % (c, i)
        return s[:-1]

    @property
    def transfer_efficency_string(self) -> str:
        """
        Used for instance for plotting in origin
        """
        return "1.0-(%s)/(%s)" % (self.conversion_function_string, self.donor_species_averaged_lifetime)

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
        The values the parameter as defined by :py:attr:`~chisurf.fluorescence.fret_lines.FRETLineGenerator.parameter_name`
        """
        start, stop = self.parameter_range
        n_points = self.n_points
        return np.linspace(
            start=start,
            stop=stop,
            num=n_points
        )

    @property
    def model(self) -> chisurf.models.tcspc.lifetime.LifetimeModel:
        return self._model

    @model.setter
    def model(
            self,
            v: typing.Type[chisurf.models.tcspc.lifetime.LifetimeModel]
    ):
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
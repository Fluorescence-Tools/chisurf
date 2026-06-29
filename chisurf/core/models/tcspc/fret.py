from __future__ import annotations

import numpy as np

import chisurf as cs
import chisurf.core.fluorescence.tcspc
import chisurf.core.fluorescence.anisotropy.kappa2 as kapp2

import chisurf.core.math
import chisurf.core.math.datatools
from chisurf.core.models.tcspc.lifetime import Lifetime, LifetimeModel
from chisurf.core.fluorescence.general import distribution2rates, rates2lifetimes
from chisurf.core.fitting.parameter import FittingParameter, FittingParameterGroup
from chisurf.core.settings.settings_utils import build_fret_rda_axis

try:
    # Prefer the central global R_DA axis defined in cs.core.fluorescence.
    rda_axis = cs.core.fluorescence.rda_axis
except Exception:
    # Fallback: reconstruct from settings if the central axis is not available.
    _fret_cfg = getattr(cs.core.settings, "fret", {}) or {}
    _rda_min = _fret_cfg.get("rda_min", 1.0)
    _rda_max = _fret_cfg.get("rda_max", 130.0)
    _rda_res = _fret_cfg.get("rda_resolution", 96)
    _rda_scale = _fret_cfg.get("rda_scale", "log")
    rda_axis = build_fret_rda_axis(_rda_min, _rda_max, _rda_res, _rda_scale)


def set_forster_radius_from_probes(
    fret_params: FRETParameters,
    donor_name: str,
    acceptor_name: str,
    db=None,
) -> bool:
    """Look up and set *R*\ :sub:`0` from the MFDB fluorophore database.

    Calls :func:`lookup_forster_radius` and, when a value is found, updates
    the ``forster_radius`` :class:`FittingParameter` in ``fret_params``.

    Parameters
    ----------
    fret_params : FRETParameters
        The parameter object to update.
    donor_name : str
        Donor chromophore name.
    acceptor_name : str
        Acceptor chromophore name.
    db : MFDatabase, optional
        Database handle. Resolved globally when ``None``.

    Returns
    -------
    bool
        ``True`` when a value was found and set.
    """
    from chisurf.core.fluorescence.fret.forster import lookup_forster_radius

    r0 = lookup_forster_radius(donor_name, acceptor_name, db=db)
    if r0 is not None:
        fret_params.forster_radius = float(r0)
        return True
    return False


class FRETParameters(FittingParameterGroup):

    name = "FRET-parameters"

    @property
    def forster_radius(self) -> float:
        """Foerster radius in Angstrom."""
        return self._forster_radius.value

    @forster_radius.setter
    def forster_radius(self, v: float):
        """Foerster radius in Angstrom."""
        self._forster_radius.value = v

    @property
    def tauD0(self) -> float:
        """Donor lifetime in the absence of FRET (ns)."""
        return self._tauD0.value

    @tauD0.setter
    def tauD0(self, v: float):
        """Donor lifetime in the absence of FRET (ns)."""
        self._tauD0.value = v

    @property
    def kappa2(self) -> float:
        """Orientation factor kappa^2."""
        return self._kappa2.value

    @kappa2.setter
    def kappa2(self, v: float):
        """Orientation factor kappa^2."""
        self._kappa2.value = v

    @property
    def xDOnly(self) -> float:
        """Donor-only fraction (molecules without acceptor)."""
        return np.sqrt(self._xDonly.value ** 2)

    @xDOnly.setter
    def xDOnly(self, v: float):
        """Donor-only fraction (molecules without acceptor)."""
        self._xDonly.value = v

    # TODO: needs docstring
    def __init__(
            self,
            forster_radius: float = cs.core.settings.fret['forster_radius'],
            tau0: float = cs.core.settings.fret['tau0'],
            xDOnly: float = 0.0,
            kappa2: float = 0.66667,
            enable_fret_efficiency: bool = True,
            **kwargs
    ):
        """Initialize the instance."""
        model = kwargs.get('models', None)

        self._tauD0 = FittingParameter(
            name='t0',
            label_text='&tau;<sub>0</sub>',
            value=tau0,
            fixed=True,
            model=model
        )
        self._forster_radius = FittingParameter(
            name='R0',
            label_text='R<sub>0</sub>',
            value=forster_radius,
            fixed=True,
            model=model
        )
        self._kappa2 = FittingParameter(
            name='k2', label_text='&kappa;<sup>2</sup>',
            value=kappa2, fixed=True,
            lb=0.0, ub=4.0,
            bounds_on=False,
            models=model
        )
        self._xDonly = FittingParameter(
            name='xDOnly',
            label_text='x<sup>(D,0)</sup>',
            value=xDOnly,
            fixed=False,
            lb=0.0,
            ub=1.0,
            bounds_on=True,
            model=model
        )

        # Base parameter list is always tau0, R0, kappa2 and xDOnly.
        parameters = [self._tauD0, self._forster_radius, self._kappa2, self._xDonly]

        # Optional E_FRET parameter (used in TCSPC FRET models).
        if enable_fret_efficiency:
            func_calc_fret = kwargs.get('func_calc_fret', None)
            if callable(func_calc_fret):
                value = func_calc_fret
            else:
                value = 0.0
                if func_calc_fret is not None:
                    try:
                        value = float(func_calc_fret)
                    except Exception:
                        cs.logging.warning(
                            f"FRETParameters: func_calc_fret={func_calc_fret!r} is not callable or numeric; defaulting E_FRET to 0.0"
                        )

            self._fret_efficiency = FittingParameter(
                name='E_FRET',
                label_text='E<sub>FRET</sub>',
                value=value,
                fixed=False,
                lb=0.0,
                ub=1.0,
                bounds_on=True,
                model=model
            )
            parameters.append(self._fret_efficiency)

        super().__init__(parameters=parameters, **kwargs)


class OrientationParameter(FittingParameterGroup):

    @property
    def orientation_spectrum(self):
        """Current orientation factor spectrum."""
        mode = str(self.mode).strip().lower()
        if mode == 'fast':
            return self._k2_fast_iso
        elif mode == 'slow':
            return self._k2_slow_iso
        return self._k2_fast_iso

    @orientation_spectrum.setter
    def orientation_spectrum(self, v):
        """Set the orientation-factor spectrum used for static averaging.

        Expects an interleaved (amplitude, k2, amplitude, k2, ...) 1D array or
        sequence. This is primarily used in "slow" mode, where
        ``FRETModel.fret_rate_spectrum`` consumes the full distribution via
        ``distribution2rates``.
        """
        arr = np.asarray(v, dtype=float).ravel()
        if arr.size < 2 or arr.size % 2 != 0:
            raise ValueError("orientation_spectrum must be an interleaved (amp, k2, ...) array")
        # Store into the slow spectrum; fast uses scalar kappa2
        self._k2_slow_iso = arr
        # Invalidate cached transformation
        self._cached_k2_transform = None

    @property
    def mode(self):
        """Orientation averaging mode (fast or slow)."""
        return self._mode

    @mode.setter
    def mode(self, v):
        """Orientation averaging mode (fast or slow)."""
        if v is None:
            self._mode = 'fast'
            return
        s = str(v).strip().lower()
        s = {
            'slow_isotropic': 'slow',
            'slow_iso': 'slow',
            'static': 'slow',
            'static_isotropic': 'slow',
            'fast_isotropic': 'fast',
            'fast_iso': 'fast',
            'dynamic': 'fast',
            'dynamic_isotropic': 'fast',
        }.get(s, s)
        if s in {'fast', 'slow'}:
            self._mode = s
            return
        raise ValueError(f"Invalid orientation mode {v!r}. Expected 'fast' or 'slow'.")

    # TODO: needs docstring
    def __init__(self, *args, **kwargs):
        """Initialize the instance."""
        # Route through the property setter so aliases are normalized.
        self.mode = kwargs.get('orientation_mode', 'fast')

        # fast
        self._k2_fast_iso = [1., 0.666]

        # slow
        k2s = np.linspace(0.01, 4, 128)
        pk2 = kapp2.p_isotropic_orientation_factor(
            k2s
        )
        self._k2_slow_iso = cs.core.math.datatools.two_column_to_interleaved(
            pk2, k2s
        )
        
        # Cache for R_app/R_DA transformation
        self._cached_k2_transform = None

        FittingParameterGroup.__init__(self, *args, **kwargs)
    
    def get_k2_distance_ratio_transform(self, n_bins: int = 256):
        """Get cached R_app/R_DA transformation of κ² distribution.
        
        This caches the transformation to avoid recomputing it on every update.
        The cache is invalidated when the orientation_spectrum is changed.
        
        Parameters
        ----------
        n_bins : int
            Number of bins for the output distribution
            
        Returns
        -------
        tuple or None
            (r_ratio, weights, k2_mean) if in slow mode, None otherwise
        """
        if self.mode != 'slow':
            return None
            
        # Check if cache is valid
        if self._cached_k2_transform is not None:
            cached_bins, cached_data = self._cached_k2_transform
            if cached_bins == n_bins:
                return cached_data
        
        # Compute transformation
        from chisurf.core.fluorescence.general import kappa2_to_distance_ratio
        
        k2_array = np.asarray(self._k2_slow_iso, dtype=float).ravel()
        if k2_array.size < 2 or k2_array.size % 2 != 0:
            return None
            
        k2 = k2_array.reshape((-1, 2))
        k2_amp = k2[:, 0]
        k2_val = k2[:, 1]
        
        try:
            r_ratio, weights, k2_mean = kappa2_to_distance_ratio(k2_amp, k2_val, n_bins=n_bins)
            result = (r_ratio, weights, k2_mean)
            
            # Cache the result
            self._cached_k2_transform = (n_bins, result)
            return result
        except Exception:
            return None


class Gaussians(FittingParameterGroup):

    name = "gaussians"

    @property
    def distribution(self) -> np.array:
        """Probability distribution of the distance or FRET parameter."""
        d = list()
        weights = self.amplitude
        if not self.is_distance_between_gaussians:
            args = zip(self.mean, self.sigma, self.shape)
            pdf = cs.core.math.functions.distributions.generalized_normal_distribution
        else:
            args = zip(self.mean, self.sigma)
            pdf = cs.core.math.functions.rdf.distance_between_gaussian
        p = cs.core.math.functions.distributions.combine_distributions(
            x_axis=rda_axis,
            dist_function=pdf,
            dist_args=args,
            weights=weights,
            normalize=True
        )

        d.append([p, rda_axis])
        d = np.array(d)
        return d

    @property
    def mean(self):
        """Mean values of the Gaussian components."""
        try:
            a = np.sqrt(np.array([g.value for g in self._gaussianMeans]) ** 2)
            return a
        except AttributeError:
            return np.array([])

    @property
    def shape(self):
        """Shape parameters of the Gaussian components."""
        try:
            a = np.array([g.value for g in self._gaussianShape])
            return a
        except AttributeError:
            return np.array([])

    @property
    def sigma(self):
        """Standard deviations of the Gaussian components."""
        try:
            return np.array([g.value for g in self._gaussianSigma])
        except AttributeError:
            return np.array([])

    @property
    def amplitude(self):
        """Amplitude array, normalized to sum to one."""
        try:
            a = np.sqrt(
                np.array([g.value for g in self._gaussianAmplitudes]) ** 2
            )
            a /= a.sum()
            return a
        except AttributeError:
            return np.array([])

    # TODO: needs docstring
    def finalize(self):
        """Finalize the component state."""
        a = self.amplitude
        for i, g in enumerate(self._gaussianAmplitudes):
            g.value = a[i]

    def pop(self):
        """
        Removes the last appended Gaussian/normal-distribution
        """
        self._gaussianMeans.pop()
        self._gaussianSigma.pop()
        self._gaussianAmplitudes.pop()
        self._gaussianShape.pop()

    def append(
            self,
            mean: float,
            sigma: float,
            x: float,
            shape: float = 0.0
    ):
        """
        Adds/appends a new Gaussian/normal-distribution

        :param mean: float
            Mean of the new normal distribution
        :param sigma: float
            Sigma/width of the normal distribution
        :param x: float
            Amplitude of the normal distribution
        :param shape: float
            Shape of the Gaussian (generalized Gaussian, log(x)...)

        """
        n = len(self)
        m = FittingParameter(
            name='R(%s,%i)' % (self.short, n + 1),
            value=mean
        )
        x = FittingParameter(
            name='x(%s,%i)' % (self.short, n + 1),
            value=x
        )
        s = FittingParameter(
            name='s(%s,%i)' % (self.short, n + 1),
            value=sigma,
            fixed=True
        )
        shape = FittingParameter(
            name='k(%s,%i)' % (self.short, n + 1),
            value=shape,
            fixed=True
        )
        self._gaussianMeans.append(m)
        self._gaussianSigma.append(s)
        self._gaussianAmplitudes.append(x)
        self._gaussianShape.append(shape)

    def __len__(self):
        """Return the number of components."""
        return len(self._gaussianAmplitudes)

    def __init__(
            self,
            short: str = 'G',
            is_distance_between_gaussians: bool = False,
            name: str = 'gaussians',
            **kwargs
    ):
        """
        This class keeps the necessary parameters to perform a fit with
        Gaussian/Normal-disitributed distances. New distance distributions
        are added using the methods append.

        :param donors: Lifetime
            The donor-only spectrum in form of a `Lifetime` object.
        :param forster_radius: float
            The Forster-radius of the FRET-pair in Angstrom. By default 52.0
            Angstrom (FRET-pair Alexa488/Alexa647)
        :param kappa2: float
            Orientation factor. By default 2./3.
        :param t0: float
            Lifetime of the donor-fluorophore in absence of FRET.
        :param donor_only: float
            Donor-only fraction. The fraction of molecules without acceptor.
        :param no_donly: bool
            If this is True the donor-only fraction is not displayed/present.
        """
        super().__init__(
            name=name,
            **kwargs
        )
        self.donor = Lifetime(
            name='donor',
            **kwargs
        )
        self.short = short
        self._gaussianMeans = list()
        self._gaussianSigma = list()
        self._gaussianShape = list()
        self._gaussianAmplitudes = list()
        self.is_distance_between_gaussians = is_distance_between_gaussians


class DiscreteDistance(FittingParameterGroup):

    name = "discrete_distance"

    @property
    def distribution(self) -> np.array:
        """Probability distribution of the distance or FRET parameter."""
        distance = self.distance
        amplitude = self.amplitude
        count, bins = np.histogram(distance, bins=rda_axis, weights=amplitude)
        count.resize(count.shape[0] + 1)
        s = np.vstack([count, rda_axis])
        return np.array([s], dtype=np.float64)

    @property
    def distance(self) -> np.array:
        """Distance values of the discrete distance components."""
        try:
            a = np.sqrt(np.array([g.value for g in self._distances]) ** 2)
            return a
        except AttributeError:
            return np.array([])

    @property
    def amplitude(self) -> np.array:
        """Amplitude array, normalized to sum to one."""
        try:
            a = np.sqrt(np.array([g.value for g in self._amplitudes]) ** 2)
            a /= a.sum()
            return a
        except AttributeError:
            return np.array([])

    # TODO: needs docstring
    def finalize(self):
        """Finalize the component state."""
        a = self.amplitude
        for i, g in enumerate(self._amplitudes):
            g.value = a[i]

    # TODO: needs docstring
    def append(self, mean: float, x: float):
        """Add a new component."""
        n = len(self)
        self._distances.append(
            FittingParameter(
                name='R(%s,%i)' % (self.short, n + 1),
                value=mean
            )
        )
        self._amplitudes.append(
            FittingParameter(
                name='x(%s,%i)' % (self.short, n + 1),
                value=x
            )
        )

    def pop(self):
        """
        Removes the last appended Gaussian/normal-distribution
        """
        self._distances.pop()
        self._amplitudes.pop()

    def __len__(self):
        """Return the number of components."""
        return len(self._amplitudes)

    # TODO: needs docstring
    def __init__(
            self,
            name: str = 'fret_rate',
            short: str = 'd',
            **kwargs
    ):
        """Initialize the instance."""
        super().__init__(**kwargs)
        self.name = name
        self.short = short
        self.donor = Lifetime(**kwargs, name='donor')
        self._distances = list()
        self._amplitudes = list()


class FRETModel(LifetimeModel):

    @property
    def distance_distribution(self) -> np.array:
        """
        The distribution of distances. The distribution should be 3D numpy array of the form

            gets distribution in form: (1,2,3)
            0: number of distribution
            1: amplitude
            2: distance

        """
        return np.array([[[1.0], [52.0]]], dtype=np.float64)

    @property
    def fret_rate_spectrum(self) -> np.array:
        """
        The FRET-rate spectrum. This takes the distance distribution of the models and calculated the resulting
        FRET-rate spectrum (excluding the donor-offset).
        """
        tauD0 = self.fret_parameters.tauD0
        kappa2_scalar = self.fret_parameters.kappa2
        forster_radius = self.fret_parameters.forster_radius
        orientation_mode = getattr(self.orientation_parameter, "mode", "fast")
        if orientation_mode == "slow":
            kappa2s = self.orientation_parameter.orientation_spectrum
        else:
            kappa2s = kappa2_scalar
        
        # Check if Fast optimization is enabled via UI checkbox
        use_fast = True
        fast_checkbox = getattr(self, "_kappa2_fft_checkbox", None)
        if fast_checkbox is not None:
            use_fast = fast_checkbox.isChecked()
        
        # Get cached κ² transformation if in slow mode and using fast convolution
        k2_transform_cache = None
        if use_fast and orientation_mode == "slow":
            k2_transform_cache = self.orientation_parameter.get_k2_distance_ratio_transform(n_bins=256)
        
        rs = distribution2rates(
            self.distance_distribution,
            tauD0,
            kappa2s,
            forster_radius,
            use_fast=use_fast,
            k2_transform_cache=k2_transform_cache
        )
        r = np.hstack(rs).reshape(-1, order='F')
        return r

    @property
    def lifetime_spectrum(self) -> np.array:
        """Interleaved (amplitude, lifetime, ...) array."""
        xDOnly = self.fret_parameters.xDOnly
        lt = rates2lifetimes(
            self.fret_rate_spectrum,
            self.donor.rate_spectrum,
            xDOnly
        )
        if cs.core.settings.cs_settings['fret']['bin_lifetime']:
            n_lifetimes = cs.core.settings.cs_settings['fret']['lifetime_bins']
            discriminate = cs.core.settings.cs_settings['fret']['discriminate']
            discriminate_amplitude = cs.core.settings.cs_settings['fret'][
                'discriminate_amplitude']
            return cs.core.fluorescence.tcspc.bin_lifetime_spectrum(
                lt, n_lifetimes=n_lifetimes,
                discriminate=discriminate,
                discriminator=discriminate_amplitude
            )
        else:
            return lt

    @property
    def donor_lifetime_spectrum(self) -> np.array:
        """
        The donor lifetime spectrum in form amplitude, lifetime, amplitude,
        lifetime.
        """
        return self.donor.lifetime_spectrum

    @donor_lifetime_spectrum.setter
    def donor_lifetime_spectrum(
            self,
            v: np.array
    ):
        """Donor lifetime spectrum."""
        self.model.donors.lifetime_spectrum = v

    @property
    def donor_species_averaged_lifetime(self) -> float:
        """
        The current species averaged lifetime of the donor sample xi*taui
        """
        return self.donor.species_averaged_lifetime

    @property
    def donor_fluorescence_averaged_lifetime(self) -> float:
        """
        The current species averaged lifetime of the donor sample xi*taui
        """
        return self.donor.fluorescence_averaged_lifetime

    @property
    def fret_species_averaged_lifetime(self) -> float:
        """
        The current species averages lifetime of the FRET sample xi * taui
        """
        return self.species_averaged_lifetime

    @property
    def fret_fluorescence_averaged_lifetime(self) -> float:
        """Fluorescence-averaged lifetime of the FRET sample."""
        return self.fluorescence_averaged_lifetime

    @property
    def fret_efficiency(self) -> float:
        """FRET efficiency calculated from lifetime contrast."""
        return 1.0 - self.fret_species_averaged_lifetime / self.donor_species_averaged_lifetime

    @fret_efficiency.setter
    def fret_efficiency(self, v: float):
        """FRET efficiency calculated from lifetime contrast."""
        sdecay = self.fit.data.y.sum()
        tau0x = self.donor_species_averaged_lifetime
        n0 = sdecay/(tau0x*(1.-v))
        self.convolve.n0 = n0

    @property
    def reference(self):
        """Reference decay curve for the FRET sample."""
        self._reference.update_model()
        ref = np.maximum(self._reference.y, 0)

        ref_max = np.max(ref)
        if ref_max > 0:
            scale = np.max(self.fit.data.y) / ref_max
            ref *= scale
        else:
            # Avoid divide-by-zero; return a ones reference with the same shape
            ref = np.ones_like(ref)
        return ref

    # TODO: needs docstring
    def calc_fret_efficiency(self) -> float:
        """Calculate the FRET efficiency from lifetime contrast."""
        try:
            eff = 1.0 - self.fret_species_averaged_lifetime / self.donor_species_averaged_lifetime
            return eff
        except AttributeError:
            return 0.0

    def __str__(self):
        """Return a string representation."""
        s = LifetimeModel.__str__(self)
        s += "\n"
        s += "FRET-parameter\n"
        s += "--------------\n"
        s += "FRET-efficiency: %s \n" % self.fret_efficiency
        s += "Donor tauX: %s \n" % self.donor_species_averaged_lifetime
        s += "Donor tauF: %s \n" % self.donor_fluorescence_averaged_lifetime
        return s

    # TODO: needs docstring
    def __init__(
            self,
            fit: cs.core.fitting.fit.FitGroup,
            lifetimes: Lifetime = None,
            **kwargs
    ):
        """Initialize the instance."""
        self.fret_parameters = kwargs.pop(
            'fret_parameters',
            FRETParameters(
                func_calc_fret=self.calc_fret_efficiency,
                **kwargs
            )
        )
        self.orientation_parameter = OrientationParameter(
            orientation_mode=cs.core.settings.cs_settings['fret']['orientation_mode']
        )
        if getattr(self, "donor", None) is None:
            if lifetimes is not None:
                self.donor = lifetimes
            else:
                self.donor = Lifetime(name='donor', fit=fit, **kwargs)

        super().__init__(fit, lifetimes=self.donor, **kwargs)

        # Ensure the canonical lifetime group is the donor widget
        self.lifetimes = self.donor
        self._reference = LifetimeModel(fit, lifetimes=self.donor, **kwargs)
        self._reference.convolve = self.convolve

    def get_parameter_widgets(self):
        """
        Get all parameter widgets for this model.

        Returns
        -------
        list
            List of parameter widgets.
        """
        widgets = super().get_parameter_widgets() if hasattr(super(), 'get_parameter_widgets') else []
        return widgets

    # TODO: needs docstring
    def get_state(self) -> dict:
        """Return a JSON-serializable state snapshot."""
        state = super().get_state()
        if not isinstance(state, dict):
            state = {}
        extra = state.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            state["extra"] = extra
        donor = getattr(self, "lifetimes", None) or getattr(self, "donor", None)
        try:
            if donor is not None:
                extra["donor_lifetimes_n"] = int(len(donor))
        except Exception:
            pass
        return state

    # TODO: needs docstring
    def set_state(self, state: dict) -> None:
        """Restore state from a JSON-serializable snapshot."""
        if not isinstance(state, dict):
            return
        extra = state.get("extra") or {}
        donor = getattr(self, "lifetimes", None) or getattr(self, "donor", None)
        try:
            target_n = extra.get("donor_lifetimes_n")
            if donor is not None and target_n is not None:
                target_n = int(target_n)
                while True:
                    try:
                        current_n = int(len(donor))
                    except Exception:
                        break
                    if current_n >= target_n:
                        break
                    try:
                        donor.append()
                    except TypeError:
                        try:
                            donor.append(amplitude=1.0, lifetime=4.0)
                        except Exception:
                            break
                try:
                    while len(donor) > target_n:
                        donor.pop()
                except Exception:
                    pass
        except Exception:
            pass
        super().set_state(state)


class GaussianModel(FRETModel):

    name = "FRET: FD (Gaussian)"

    @property
    def distance_distribution(self) -> np.array:
        """Distance distribution array."""
        dist = self.gaussians.distribution
        return dist

    # TODO: needs docstring
    def append(self, mean: float, sigma: float, species_fraction: float):
        """Add a new component."""
        self.gaussians.append(mean, sigma, species_fraction)

    # TODO: needs docstring
    def pop(self):
        """Remove the last component."""
        return self.gaussians.pop()

    # TODO: needs docstring
    def finalize(self):
        """Finalize the component state."""
        super().finalize()
        self.lifetimes.finalize()
        self.gaussians.finalize()

    # TODO: needs docstring
    def __init__(self, fit: cs.core.fitting.fit.FitGroup, **kwargs):
        """Initialize the instance."""
        super().__init__(fit, **kwargs)
        self.gaussians = kwargs.get('gaussians', Gaussians(**kwargs))

    # TODO: needs docstring
    def get_state(self) -> dict:
        """Return a JSON-serializable state snapshot."""
        state = super().get_state()
        if not isinstance(state, dict):
            state = {}
        extra = state.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            state["extra"] = extra
        gaussians = getattr(self, "gaussians", None)
        try:
            if gaussians is not None:
                extra["gaussians_n"] = int(len(gaussians))
        except Exception:
            pass
        return state

    # TODO: needs docstring
    def set_state(self, state: dict) -> None:
        """Restore state from a JSON-serializable snapshot."""
        if not isinstance(state, dict):
            return
        extra = state.get("extra") or {}
        gaussians = getattr(self, "gaussians", None)
        try:
            target_n = extra.get("gaussians_n")
            if gaussians is not None and target_n is not None:
                target_n = int(target_n)
                while True:
                    try:
                        current_n = int(len(gaussians))
                    except Exception:
                        break
                    if current_n >= target_n:
                        break
                    try:
                        gaussians.append(mean=50.0, sigma=6.0, x=1.0)
                    except TypeError:
                        try:
                            gaussians.append(50.0, 6.0, 1.0)
                        except Exception:
                            break
                try:
                    while len(gaussians) > target_n:
                        gaussians.pop()
                except Exception:
                    pass
        except Exception:
            pass
        super().set_state(state)


class FRETrateModel(FRETModel):

    name = "FRET: FD (Discrete)"

    @property
    def fret_rate_spectrum(self) -> np.array:
        """FRET-rate spectrum calculated from the distance distribution."""
        fret_rates = cs.core.fluorescence.general.distance_to_fret_rate_constant(
            self.fret_rates.distance,
            self.fret_parameters.forster_radius,
            self.fret_parameters.tauD0,
            self.fret_parameters.kappa2
        )
        amplitudes = self.fret_rates.amplitude
        r = np.ravel(np.column_stack((amplitudes, fret_rates)))
        return r

    @property
    def distance_distribution(self) -> np.array:
        """Distance distribution array."""
        dist = self.fret_rates.distribution
        return dist

    # TODO: needs docstring
    def append(self, mean: float, species_fraction: float):
        """Add a new component."""
        self.fret_rates.append(mean, species_fraction)

    # TODO: needs docstring
    def pop(self):
        """Remove the last component."""
        return self.fret_rates.pop()

    # TODO: needs docstring
    def finalize(self):
        """Finalize the component state."""
        super().finalize()
        self.fret_rates.finalize()

    # TODO: needs docstring
    def __init__(
            self,
            fit: cs.core.fitting.fit.FitGroup,
            fret_rates: DiscreteDistance = None,
            **kwargs
    ):
        """Initialize the instance."""
        FRETModel.__init__(self, fit, **kwargs)
        if fret_rates is None:
            fret_rates = DiscreteDistance(**kwargs)
        self.fret_rates = fret_rates


class WormLikeChainModel(FRETModel):

    name = "FRET: FD (Worm-like chain)"

    @property
    def distance_distribution(self):
        """Distance distribution array."""
        chain_length = self._chain_length.value
        kappa = self._persistence_length.value / chain_length
        if not self.use_dye_linker:
            prob = cs.core.math.functions.rdf.worm_like_chain(
                rda_axis,
                kappa,
                chain_length
            )
        else:
            sigma_linker = self._sigma_linker.value
            prob = cs.core.math.functions.rdf.worm_like_chain_linker(
                rda_axis, kappa,
                chain_length,
                sigma_linker
            )
        dist = np.array([prob, rda_axis]).reshape(
            [1, 2, cs.core.settings.fret['rda_resolution']]
        )
        return dist

    @property
    def chain_length(self) -> float:
        """Contour length of the worm-like chain (Å)."""
        return self._chain_length.value

    @chain_length.setter
    def chain_length(self, v: float) -> None:
        self._chain_length.value = v

    @property
    def persistence_length(self) -> float:
        """Persistence length of the worm-like chain (Å)."""
        return self._persistence_length.value

    @persistence_length.setter
    def persistence_length(self, v: float) -> None:
        self._persistence_length.value = v

    @property
    def use_dye_linker(self):
        """Whether the dye-linker model is used."""
        return self._use_dye_linker

    @use_dye_linker.setter
    def use_dye_linker(self, v):
        """Whether the dye-linker model is used."""
        self._use_dye_linker = v

    # TODO: needs docstring
    def __init__(
            self,
            fit: cs.core.fitting.fit.FitGroup,
            use_dye_linker: bool = False,
            **kwargs
    ):
        """Initialize the instance."""
        super().__init__(
            fit,
            **kwargs
        )
        self._chain_length = FittingParameter(
            name='l',
            value=100.0,
            model=self,
            fixed=False,
            text='l'
        )
        self._use_dye_linker = use_dye_linker
        self._sigma_linker = FittingParameter(
            name='w',
            value=6.0,
            model=self,
            fixed=False,
            text='lw'
        )
        self._persistence_length = FittingParameter(
            name='lp',
            value=30.0,
            model=self,
            fixed=False,
            text='lp'
        )


class SingleDistanceModel(FRETModel):

    name = "Fixed distance distribution"

    @property
    def xDOnly(self) -> float:
        """Donor-only fraction (molecules without acceptor)."""
        return self._xDOnly.value

    @property
    def distance_distribution(self) -> np.array:
        """Distance distribution array."""
        n_points = self.n_points_dist
        r = np.vstack(
            [self.prda, self.rda]
        ).reshape([1, 2,  n_points])
        return r

    @property
    def n_points_dist(self) -> int:
        """Number of points in the distance distribution."""
        return self.prda.shape[0]

    @property
    def rda(self) -> np.array:
        """Donor-acceptor distance array."""
        return self._rda

    @rda.setter
    def rda(self, v):
        """Donor-acceptor distance array."""
        self._rda = v

    @property
    def prda(self) -> np.array:
        """Probability distribution of donor-acceptor distances."""
        p = self._prda
        p /= sum(p)
        return p

    @prda.setter
    def prda(self, v: np.array):
        """Probability distribution of donor-acceptor distances."""
        self._prda = v

    # TODO: needs docstring
    def __init__(
            self,
            fit: cs.core.fitting.fit.FitGroup,
            **kwargs
    ):
        """Initialize the instance."""
        super().__init__(fit=fit, **kwargs)
        self._rda = kwargs.get('rda', np.array([100.0]))
        self._prda = kwargs.get('prda', np.array([100.0]))

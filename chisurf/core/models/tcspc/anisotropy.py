from __future__ import annotations

import numpy as np

import chisurf.core.fluorescence
import chisurf.core.fitting
import chisurf.core.math.datatools

from chisurf import logging
from chisurf.core.fitting.parameter import FittingParameterGroup


class Anisotropy(FittingParameterGroup):

    @staticmethod
    def _link_group_anisotropy_parameters(group) -> None:
        """Link anisotropy calibration parameters from first fit to others.

        Intended default for dual VV/VH fit groups: downstream fits inherit
        `g`, `l1`, and `l2` from the first (top) fit.
        """
        try:
            if group is None or len(group) < 2:
                return
            source_fit = group[0]
            source_model = getattr(source_fit, 'model', None)
            source_aniso = getattr(source_model, 'anisotropy', None)
            if source_aniso is None:
                return

            source_params = {
                'g': getattr(source_aniso, '_g', None),
                'l1': getattr(source_aniso, '_l1', None),
                'l2': getattr(source_aniso, '_l2', None),
            }
            if any(v is None for v in source_params.values()):
                return

            for local_fit in group[1:]:
                local_model = getattr(local_fit, 'model', None)
                local_aniso = getattr(local_model, 'anisotropy', None)
                if local_aniso is None:
                    continue
                for key, source_param in source_params.items():
                    local_param = getattr(local_aniso, f'_{key}', None)
                    if local_param is None:
                        continue
                    try:
                        local_param.link = source_param
                    except Exception:
                        pass
        except Exception:
            pass

    @property
    def r0(self) -> float:
        """Fundamental anisotropy r0."""
        return self._r0.value

    @r0.setter
    def r0(self, v: chisurf.core.fitting.parameter.FittingParameter):
        """Fundamental anisotropy r0."""
        self._r0.value = v

    @property
    def l1(self) -> float:
        """First polarizer transmission factor l1."""
        return self._l1.value

    @l1.setter
    def l1(self, v: chisurf.core.fitting.parameter.FittingParameter):
        """First polarizer transmission factor l1."""
        self._l1.value = v

    @property
    def l2(self) -> float:
        """Second polarizer transmission factor l2."""
        return self._l2.value

    @l2.setter
    def l2(self, v: chisurf.core.fitting.parameter.FittingParameter):
        """Second polarizer transmission factor l2."""
        self._l2.value = v

    @property
    def g(self) -> float:
        """G-factor for anisotropy correction."""
        return self._g.value

    @g.setter
    def g(self, v: chisurf.core.fitting.parameter.FittingParameter):
        """G-factor for anisotropy correction."""
        self._g.value = v

    @property
    def rho(self) -> np.array:
        """Rotational correlation times array."""
        r = np.array([rho.value for rho in self._rhos], dtype=np.float64)
        r = np.sqrt(r**2)
        for i, v in enumerate(r):
            self._rhos[i].value = v
        return r

    @property
    def b(self) -> np.array:
        """Rotational amplitudes array, normalized to r0."""
        a = np.sqrt(np.array([g.value for g in self._bs]) ** 2)
        a /= a.sum()
        a *= self.r0
        for i, g in enumerate(self._bs):
            g.value = a[i]
        return a

    @property
    def rotation_spectrum(self) -> np.array:
        """Interleaved (amplitude, rho, ...) rotation spectrum."""
        rot = np.empty(2 * len(self), dtype=np.float64)
        rot[0::2] = self.b
        rot[1::2] = self.rho
        return rot

    @property
    def polarization_type(self) -> str:
        """Current polarization type (vm, vv, vh, vv/vh)."""
        return self._polarization_type

    @polarization_type.setter
    def polarization_type(self, v: str):
        """Set current polarization type."""
        self._polarization_type = str(v).lower()
        self._update_rotation_parameter_fixed_state()

    @staticmethod
    def _is_vm_polarization(polarization: str) -> bool:
        """Return True when polarization is magic-angle VM."""
        return str(polarization).lower() == 'vm'

    def _rotation_parameters(self) -> list:
        """Return rotation amplitude and correlation-time parameters."""
        return list(self._bs) + list(self._rhos)

    def _rotation_parameter_rows(self) -> list:
        """Return rotation parameters interleaved as (b_i, rho_i) pairs.

        Used by the data-driven editor's dynamic group so each row pairs an
        amplitude with its correlation time.
        """
        rows = []
        for b, rho in zip(self._bs, self._rhos):
            rows.append(b)
            rows.append(rho)
        return rows

    def _update_rotation_parameter_fixed_state(self) -> None:
        """Keep rotation parameters fixed while VM polarization is selected."""
        if not hasattr(self, '_vm_auto_fixed_parameters'):
            self._vm_auto_fixed_parameters = set()
        if self._is_vm_polarization(self._polarization_type):
            for parameter in self._rotation_parameters():
                if not parameter.fixed:
                    self._vm_auto_fixed_parameters.add(id(parameter))
                parameter.fixed = True
        else:
            for parameter in self._rotation_parameters():
                parameter_id = id(parameter)
                if parameter_id in self._vm_auto_fixed_parameters:
                    parameter.fixed = False
                    self._vm_auto_fixed_parameters.remove(parameter_id)

    def set_polarization_by_group_position(self, fit, model_instance):
        """
        Set the polarization type based on the position of a fit in a group.

        This method unifies the polarization assignment logic that was previously
        duplicated in LifetimeModel.__init__ and LifetimeModelWidget.__init__.

        When called, this method checks every fit in the group and updates their
        polarization types based on their position in the group. This ensures that
        all fits have the correct polarization type, even if they were added
        sequentially to the group.

        Parameters
        ----------
        fit : chisurf.core.fitting.fit.Fit
            The fit object that may be part of a group
        model_instance : object
            The model instance (self) that is calling this method

        Returns
        -------
        bool
            True if polarization was set, False otherwise
        """
        logging.debug("Anisotropy: Setting polarization type based on group position")
        logging.debug(f"Model instance: {model_instance.__class__.__name__}")
        # Check if the fit has a group attribute
        logging.debug(f"Fit has group attribute: {hasattr(fit, 'group')}")
        if hasattr(fit, 'group'):
            group = fit.group
            logging.debug(f"Fit group: {group}")
            logging.debug(f"Fit group length: {len(group)}")
            if len(group) == 1:
                # Single fit in group gets magic angle
                logging.info("Single fit in group, setting polarization to 'vm'")
                self.polarization_type = 'vm'
                return True
            else:
                # Update polarization for all fits in the group
                logging.info("Updating polarization for all fits in the group")
                for i, f in enumerate(group):
                    logging.debug(f"Fit index: {i}")
                    if hasattr(f, 'model') and f.model is not None:
                        model = f.model
                        logging.debug("Using f.model")
                    else:
                        logging.debug("Using model_instance")
                        # current model (not yet added to fit)
                        model = model_instance

                    # Check if this is a stacked VV,VH dataset (2 curves in group)
                    if len(group) == 2 and hasattr(f, 'data') and hasattr(f.data, 'y'):
                        # For stacked VV,VH data, both curves should use 'vv/vh' mode
                        if f.data.y.shape[0] == 2:  # Stacked data
                            logging.info(f"Setting polarization to 'vv/vh' for stacked data at index {i}")
                            model.anisotropy.polarization_type = 'vv/vh'
                            continue

                    # Set polarization type based on index (even indices get 'vv', odd indices get 'vh')
                    if i % 2 == 0:
                        logging.info(f"Setting polarization to 'vv' for fit at index {i}")
                        model.anisotropy.polarization_type = 'vv'
                    else:
                        logging.info(f"Setting polarization to 'vh' for fit at index {i}")
                        model.anisotropy.polarization_type = 'vh'

                # Default-link anisotropy calibration parameters across the group
                # (equivalent to linking via middle-click in UI).
                self._link_group_anisotropy_parameters(group)

                return True

        # If we get here, polarization was not set
        return False

    # TODO: needs docstring
    def get_decay(self, lifetime_spectrum: np.ndarray):
        """Calculate the polarized decay."""
        return chisurf.core.fluorescence.anisotropy.decay.calculcate_spectrum(
            lifetime_spectrum=lifetime_spectrum,
            anisotropy_spectrum=self.rotation_spectrum,
            polarization_type=self.polarization_type,
            g_factor=self.g,
            l1=self.l1,
            l2=self.l2
        )

    def __len__(self):
        """Return the number of components."""
        return len(self._bs)

    # TODO: needs docstring
    def add_rotation(
            self,
            b: float = 0.2,
            rho: float = 1.0,
            lb: float = 0.0,
            ub: float = 10000.0,
            fixed: bool = False,
            bound_on: bool = False,
            **kwargs
    ):
        """Add a rotation component with GUI widget."""
        b_value = b
        rho_value = rho
        i = (len(self) + 1)
        b = chisurf.core.fitting.parameter.FittingParameter(
            value=b_value,
            lb=lb,
            ub=ub,
            name=f'b({i})',
            label_text=f'b<sub>{i}</sub>',
            fixed=fixed,
            bounds_on=bound_on
        )
        rho = chisurf.core.fitting.parameter.FittingParameter(
            value=rho_value,
            lb=lb,
            ub=ub,
            name='rho(%i)' % i,
            label_text=f'&rho;<sub>{i}</sub>',
            fixed=fixed,
            bounds_on=bound_on
        )
        self._rhos.append(rho)
        self._bs.append(b)
        self._update_rotation_parameter_fixed_state()

    # TODO: needs docstring
    def remove_rotation(self) -> None:
        """Remove the last rotation component and widget."""
        self._rhos.pop().close()
        self._bs.pop().close()

    # TODO: needs docstring
    # --- diagnostics helpers (relocated from AnisotropyWidget, PRD-38 4c) ---

    @staticmethod
    def _shift_trace_to_reference(t: np.ndarray, y: np.ndarray, delta_t: float) -> np.ndarray:
        """Return y shifted by delta_t on axis t using linear interpolation.

        Positive delta_t advances the trace towards earlier times so that a
        larger IRF timeshift can be aligned to a smaller reference shift.
        """
        if not np.isfinite(delta_t) or abs(delta_t) < 1e-15:
            return y
        return np.interp(t + float(delta_t), t, y, left=0.0, right=0.0)

    @staticmethod
    def _fit_timeshift(local_fit) -> float:
        """Extract the IRF timeshift from a local fit's convolve.

        Parameters
        ----------
        local_fit : cs.core.fitting.fit.Fit
            The local fit object.

        Returns
        -------
        float
            The timeshift value, or 0.0 if unavailable.
        """
        try:
            model = getattr(local_fit, 'model', None)
            convolve = getattr(model, 'convolve', None)
            if convolve is not None and hasattr(convolve, 'timeshift'):
                return float(convolve.timeshift)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _fit_bg_level(local_fit, default: float = 0.0) -> float:
        """Extract the background level from a local fit's generic.

        Parameters
        ----------
        local_fit : cs.core.fitting.fit.Fit
            The local fit object.
        default : float
            Fallback value if background is unavailable.

        Returns
        -------
        float
            The background level.
        """
        bg = default
        try:
            model = getattr(local_fit, 'model', None)
            generic = getattr(model, 'generic', None)
            if generic is not None and hasattr(generic, 'background'):
                bg = float(generic.background)
        except Exception:
            pass
        return bg

    def _extract_vv_vh_raw_for_diag(self):
        """Return raw VV/VH channels and default correction controls.

        Returns
        -------
        tuple
            (t, vv_raw, vh_raw, defaults)
            defaults has keys: g, l1, l2, bg_vv, bg_vh, shift_vv, shift_vh
        """
        fit = getattr(self, 'fit', None)
        data = getattr(fit, 'data', None)
        if data is None:
            return None, None, None, None

        defaults = {
            'g': float(self.g),
            'l1': float(self.l1),
            'l2': float(self.l2),
            'bg_vv': 0.0,
            'bg_vh': 0.0,
            'shift_vv': 0.0,
            'shift_vh': 0.0,
        }
        pol = str(getattr(self, 'polarization_type', 'vm')).lower()

        if pol in ('vv/vh', 'vvvh') and hasattr(data, '__len__') and hasattr(data, '__getitem__'):
            try:
                if len(data) >= 2:
                    d_vv = data[0]
                    d_vh = data[1]
                    t_vv = np.asarray(getattr(d_vv, 'x', None), dtype=np.float64)
                    t_vh = np.asarray(getattr(d_vh, 'x', None), dtype=np.float64)
                    y_vv = np.asarray(getattr(d_vv, 'y', None), dtype=np.float64)
                    y_vh = np.asarray(getattr(d_vh, 'y', None), dtype=np.float64)
                    if y_vv.ndim > 1:
                        y_vv = y_vv[0]
                    if y_vh.ndim > 1:
                        y_vh = y_vh[1] if y_vh.shape[0] > 1 else y_vh[0]
                    n = min(t_vv.size, t_vh.size, y_vv.size, y_vh.size)
                    if n >= 2:
                        t = t_vv[:n]
                        if np.max(np.abs(t - t_vh[:n])) > 1e-12:
                            t = 0.5 * (t + t_vh[:n])
                        bg_base = self._fit_bg_level(fit, 0.0)
                        defaults['bg_vv'] = self._curve_bg_level(d_vv, bg_base)
                        defaults['bg_vh'] = self._curve_bg_level(d_vh, bg_base)
                        # Do not prefill relative VV/VH shift from fit: this is
                        # intentionally user-controlled in diagnostics.
                        defaults['shift_vv'] = 0.0
                        defaults['shift_vh'] = 0.0
                        return t, y_vv[:n], y_vh[:n], defaults
            except Exception:
                pass

        group = getattr(fit, 'group', None)
        if group is not None:
            try:
                if len(group) >= 2:
                    vv_fit = None
                    vh_fit = None
                    for local_fit in group:
                        local_model = getattr(local_fit, 'model', None)
                        local_aniso = getattr(local_model, 'anisotropy', None)
                        local_pol = str(getattr(local_aniso, 'polarization_type', '')).lower()
                        if local_pol == 'vv' and vv_fit is None:
                            vv_fit = local_fit
                        elif local_pol == 'vh' and vh_fit is None:
                            vh_fit = local_fit
                    if vv_fit is None or vh_fit is None:
                        vv_fit = group[0]
                        vh_fit = group[1]

                    d_vv = getattr(vv_fit, 'data', None)
                    d_vh = getattr(vh_fit, 'data', None)
                    if d_vv is not None and d_vh is not None:
                        t_vv = np.asarray(getattr(d_vv, 'x', None), dtype=np.float64)
                        t_vh = np.asarray(getattr(d_vh, 'x', None), dtype=np.float64)
                        y_vv = np.asarray(getattr(d_vv, 'y', None), dtype=np.float64)
                        y_vh = np.asarray(getattr(d_vh, 'y', None), dtype=np.float64)
                        if y_vv.ndim > 1:
                            y_vv = y_vv[0]
                        if y_vh.ndim > 1:
                            y_vh = y_vh[1] if y_vh.shape[0] > 1 else y_vh[0]
                        n = min(t_vv.size, t_vh.size, y_vv.size, y_vh.size)
                        if n >= 2:
                            t = t_vv[:n]
                            if np.max(np.abs(t - t_vh[:n])) > 1e-12:
                                t = 0.5 * (t + t_vh[:n])
                            defaults['bg_vv'] = self._curve_bg_level(d_vv, self._fit_bg_level(vv_fit, 0.0))
                            defaults['bg_vh'] = self._curve_bg_level(d_vh, self._fit_bg_level(vh_fit, 0.0))
                            # Do not prefill relative VV/VH shift from fit: this is
                            # intentionally user-controlled in diagnostics.
                            defaults['shift_vv'] = 0.0
                            defaults['shift_vh'] = 0.0
                            return t, y_vv[:n], y_vh[:n], defaults
            except Exception:
                pass

        return None, None, None, None

    def _extract_vv_vh_model_for_diag(self):
        """Return modeled VV/VH channels for diagnostics.

        Returns
        -------
        tuple
            (t, vv_model, vh_model) or (None, None, None)
        """
        fit = getattr(self, 'fit', None)
        data = getattr(fit, 'data', None)
        if fit is None or data is None:
            return None, None, None

        pol = str(getattr(self, 'polarization_type', 'vm')).lower()

        # Stacked dataset in one fit
        if pol in ('vv/vh', 'vvvh'):
            try:
                y_model = np.asarray(getattr(getattr(fit, 'model', None), 'y', None), dtype=np.float64)
                x = np.asarray(getattr(data, 'x', None), dtype=np.float64)
                if y_model.ndim > 1 and y_model.shape[0] >= 2:
                    n = min(x.size, y_model.shape[1])
                    if n >= 2:
                        return x[:n], y_model[0, :n], y_model[1, :n]
            except Exception:
                pass

        # Paired VV/VH local fits in a group
        group = getattr(fit, 'group', None)
        if group is not None:
            try:
                if len(group) >= 2:
                    vv_fit = None
                    vh_fit = None
                    for local_fit in group:
                        local_model = getattr(local_fit, 'model', None)
                        local_aniso = getattr(local_model, 'anisotropy', None)
                        local_pol = str(getattr(local_aniso, 'polarization_type', '')).lower()
                        if local_pol == 'vv' and vv_fit is None:
                            vv_fit = local_fit
                        elif local_pol == 'vh' and vh_fit is None:
                            vh_fit = local_fit
                    if vv_fit is None or vh_fit is None:
                        vv_fit = group[0]
                        vh_fit = group[1]

                    x_vv = np.asarray(getattr(getattr(vv_fit, 'data', None), 'x', None), dtype=np.float64)
                    x_vh = np.asarray(getattr(getattr(vh_fit, 'data', None), 'x', None), dtype=np.float64)
                    y_vv = np.asarray(getattr(getattr(vv_fit, 'model', None), 'y', None), dtype=np.float64)
                    y_vh = np.asarray(getattr(getattr(vh_fit, 'model', None), 'y', None), dtype=np.float64)
                    if y_vv.ndim > 1:
                        y_vv = y_vv[0]
                    if y_vh.ndim > 1:
                        y_vh = y_vh[1] if y_vh.shape[0] > 1 else y_vh[0]
                    n = min(x_vv.size, x_vh.size, y_vv.size, y_vh.size)
                    if n >= 2:
                        t = x_vv[:n]
                        if np.max(np.abs(t - x_vh[:n])) > 1e-12:
                            t = 0.5 * (t + x_vh[:n])
                        return t, y_vv[:n], y_vh[:n]
            except Exception:
                pass

        return None, None, None

    @staticmethod
    def _curve_bg_level(curve, default: float = 0.0) -> float:
        """Extract the background level from a curve's metadata.

        Parameters
        ----------
        curve : object
            Data curve with an optional ``meta_data`` dict.
        default : float
            Fallback value.

        Returns
        -------
        float
            The background level.
        """
        bg = default
        try:
            meta = getattr(curve, 'meta_data', None)
            if isinstance(meta, dict):
                bg = float(meta.get('bg', meta.get('background', bg)))
        except Exception:
            pass
        return bg

    def __init__(
            self,
            polarization: str = None,
            name: str = 'Anisotropy',
            r0: float = 0.38,
            g_factor: float = 1.0,
            l1: float = 0.00308,
            l2: float = 0.00368,
            **kwargs
    ):
        """Initialize the instance."""
        super(Anisotropy, self).__init__(name=name, **kwargs)

        self._rhos = list()
        self._bs = list()

        if polarization is None:
            polarization = chisurf.core.settings.cs_settings['tcspc']['polarization']
        self._polarization_type = str(polarization).lower()

        self._r0 = chisurf.core.fitting.parameter.FittingParameter(
            name='r0',
            value=r0,
            fixed=True
        )
        self._g = chisurf.core.fitting.parameter.FittingParameter(
            name='g',
            value=g_factor,
            fixed=True
        )
        self._l1 = chisurf.core.fitting.parameter.FittingParameter(
            name='l1',
            value=l1,
            fixed=True
        )
        self._l2 = chisurf.core.fitting.parameter.FittingParameter(
            name='l2',
            value=l2,
            fixed=True
        )



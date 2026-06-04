from __future__ import annotations

import chisurf.fitting
import chisurf.structure
from chisurf.models.tcspc.lifetime import LifetimeModel
from chisurf.structure.av import DynamicAV
from chisurf.fitting.parameter import FittingParameter


class AVDecayModel(LifetimeModel):

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            structure: chisurf.structure.Structure = None,
            **kwargs
    ):
        """
        Example
        -------
        >>> import chisurf.curve  # doctest: +SKIP
        >>> import chisurf.fitting  # doctest: +SKIP
        >>> data_set = chisurf.data.DataCurve(filename='./test/data/tcspc/ibh_sample/Decay_577D.txt', skiprows=9)  # doctest: +SKIP
        >>> lin = chisurf.data.DataCurve(filename='./test/data/tcspc/ibh_sample/whitelight.txt', skiprows=9)  # doctest: +SKIP
        >>> data_set.ey = chisurf.fluorescence.tcspc.counting_noise(data_set.y)  # doctest: +SKIP
        >>> irf = chisurf.data.DataCurve(filename='./test/data/tcspc/ibh_sample/Prompt.txt', skiprows=9)  # doctest: +SKIP
        >>> data_set.x *= 0.0141  # doctest: +SKIP
        >>> irf.x *= 0.0141  # doctest: +SKIP
        >>> data_set = chisurf.curve.ExperimentDataCurveGroup(data_set)  # doctest: +SKIP
        >>> structure = chisurf.structure.Structure('./test/data/atomic_coordinates/pdb_files/hGBP1_closed.pdb')  # doctest: +SKIP
        >>> from chisurf.fitting.model.tcspc.av_decay import AVDecayModel  # doctest: +SKIP
        >>> model_kw={'structure': structure, 'residue_seq_number': 577, 'atom_name': 'CB'}  # doctest: +SKIP
        >>> fit = fitting.FitGroup(data=data_set, model_class=AVDecayModel, model_kw=model_kw)  # doctest: +SKIP
        >>> fit.model.convolve._irf = irf  # doctest: +SKIP
        >>> fit.model.corrections.correct_dnl = True  # doctest: +SKIP
        >>> fit.model.corrections.lintable = lin  # doctest: +SKIP
        >>> fit.model.convolve.start = 1  # doctest: +SKIP
        >>> fit.model.convolve.stop = 4090  # doctest: +SKIP
        >>> fit.model.update_model()  # doctest: +SKIP
        >>> fit.xmin = 510  # doctest: +SKIP
        >>> fit.xmax = 2000  # doctest: +SKIP
        >>> p.imshow(fit.model._av.density[:,:,20])  # doctest: +SKIP
        >>> p.show()  # doctest: +SKIP

        >>> fit.model.find_parameters()  # doctest: +SKIP
        >>> print(fit.model.parameters)  # doctest: +SKIP
        >>> print(fit.chi2r)  # doctest: +SKIP
        >>> p.semilogy(fit.model.x, fit.model.y)  # doctest: +SKIP
        >>> p.show()  # doctest: +SKIP
        >>> p.plot(fit.weighted_residuals[0])  # doctest: +SKIP
        >>> p.show()  # doctest: +SKIP

        >>> fit.run()  # doctest: +SKIP
        >>> p.plot(fit.weighted_residuals[0])  # doctest: +SKIP
        >>> print(fit.chi2r)  # doctest: +SKIP

        chi2r = list()
        sws = np.linspace(2.5, 3.2, 10)
        for sw in sws:
            fit.model._rc_ele.value = sw
            chi2r.append(fit.chi2r)
            #qr = fit.model._av.quenching_rate_map.flatten()
            #qr[qr==0] = 100000.
            #p.hist(1. / qr, bins=np.arange(0.01, fit.model._av.fluorescence_lifetime, 0.5))
            #p.show()
        p.plot(sws, chi2r)

        chi2r = list()
        sws = np.linspace(3.8, 4.5, 10)
        for sw in sws:
            fit.model._fluorescence_lifetime.value = sw
            chi2r.append(fit.chi2r)
        p.plot(sws, chi2r)

        
        chi2r = list()
        sws = np.linspace(0.4, 0.7, 10)
        for sw in sws:
            fit.model._slow_factor.value = sw
            chi2r.append(fit.chi2r)
            #p.imshow(fit.model._av.density[:,:,20])
            #p.show()
        p.plot(sws, chi2r)
        
        chi2r = list()
        cons = np.linspace(3.0, 3.6, 10)
        for sw in cons:
            fit.model._contact_distance.value = sw
            chi2r.append(fit.chi2r)
            #p.imshow(fit.model._av.density[:,:,20])
            #p.show()
        p.plot(cons, chi2r)

        
        :param args: 
        :param kwargs: 
        """
        super().__init__(
            fit,
            **kwargs
        )
        self._structure = structure
        av = DynamicAV(
            structure,
            simulation_grid_resolution=1.5,
            **kwargs
        )
        self._av = av

        self._fluorescence_lifetime = FittingParameter(value=4.2, name='tau0')
        self._diffusion_coefficient = FittingParameter(value=8.0, name='D', bounds_on=True, bounds=(0.5, 30))
        self._contact_distance = FittingParameter(value=3.5, name='cont', bounds_on=True, bounds=(0.5, 5.0))
        self._slow_factor = FittingParameter(value=0.985, name='slowf', bounds_on=True, bounds=(0.01, 1.0))
        self._rc_ele = FittingParameter(value=1.5, name='rc_ele', bounds_on=True, bounds=(0.5, 3.5))

        def update_lifetime():
            """Update the fluorescence lifetime and quenching map."""
            av.fluorescence_lifetime = self._fluorescence_lifetime.value
            av.rC_electron_transfer = self._rc_ele.value
            av.update_quenching_map()
            self.decay_changed = True
        self._fluorescence_lifetime._called_on_value_change = update_lifetime
        self._rc_ele._called_on_value_change = update_lifetime

        def update_diffusion():
            """Update the diffusion coefficient, contact distance, and slow factor."""
            av._diffusion_coefficient = self._diffusion_coefficient.value
            av._contact_distance = self._contact_distance.value
            av._slow_factor = self._slow_factor.value
            av.update_diffusion_map()
            av.update_equilibrium()
            self.decay_changed = True
        self._diffusion_coefficient._called_on_value_change = update_diffusion
        self._contact_distance._called_on_value_change = update_diffusion
        self._slow_factor._called_on_value_change = update_diffusion

        self.decay_changed = True

    # TODO: needs docstring
    def update_model(self, **kwargs):
        """Recompute the model decay."""
        decay_changed = kwargs.get('decay_changed', self.decay_changed)
        i = kwargs.get('inter', 1)
        t_step = (self.times[1] - self.times[0]) / i
        n_steps = (len(self.fit.data.y) - 1) * i
        if decay_changed:
            times, density, counts = self._av.get_donor_only_decay(
                n_it=n_steps,
                t_step=t_step,
                n_out=i
            )
            self._decay = counts
            self.decay_changed = False
        super().update_model(
            mode='full',
            **kwargs
        )

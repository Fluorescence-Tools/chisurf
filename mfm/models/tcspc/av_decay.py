import mfm
from mfm.models.tcspc.lifetime import DecayModel
from mfm.fluorescence.fps import DynamicAV
from mfm.fitting.parameter import FittingParameter


class AVDecayModel(DecayModel):

    def __init__(self, fit, **kwargs):
        """
        Example
        -------
        >>> import mfm
        >>> data_set = mfm.curve.DataCurve(filename='./sample_data/tcspc/ibh_sample/Decay_577D.txt', skiprows=9)
        >>> lin = mfm.curve.DataCurve(filename='./sample_data/tcspc/ibh_sample/whitelight.txt', skiprows=9)
        >>> data_set.weights = mfm.fluorescence.tcspc.weights(data_set.y)
        >>> irf = mfm.curve.DataCurve(filename='./sample_data/tcspc/ibh_sample/Prompt.txt', skiprows=9)
        >>> data_set.x *= 0.0141
        >>> irf.x *= 0.0141
        >>> data_set = mfm.curve.ExperimentDataCurveGroup(data_set)
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> from mfm.fitting.model.tcspc.av_decay import AVDecayModel
        >>> model_kw={'structure': structure, 'residue_seq_number': 577, 'atom_name': 'CB'}
        >>> fit = mfm.fitting.FitGroup(data=data_set, model_class=AVDecayModel, model_kw=model_kw)
        >>> fit.model.convolve._irf = irf
        >>> fit.model.corrections.correct_dnl = True
        >>> fit.model.corrections.lintable = lin
        >>> fit.model.convolve.start = 1
        >>> fit.model.convolve.stop = 4090
        >>> fit.model.update_model()
        >>> fit.xmin = 510
        >>> fit.xmax = 2000
        >>> p.imshow(fit.model._av.density[:,:,20])
        >>> p.show()

        >>> fit.model.find_parameters()
        >>> print(fit.model.parameters)
        >>> print(fit.chi2r)
        >>> p.semilogy(fit.model.x, fit.model.y)
        >>> p.show()
        >>> p.plot(fit.weighted_residuals[0])
        >>> p.show()
        
        >>> fit.run()
        >>> p.plot(fit.weighted_residuals[0])
        >>> print(fit.chi2r)

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
        DecayModel.__init__(self, fit, **kwargs)
        self._structure = kwargs.pop('structure', None)
        av = DynamicAV(self._structure,
                       simulation_grid_resolution=1.5,
                       **kwargs)
        self._av = av

        self._fluorescence_lifetime = FittingParameter(value=4.2, name='tau0')
        self._diffusion_coefficient = FittingParameter(value=8.0, name='D', bounds_on=True, bounds=(0.5, 30))
        self._contact_distance = FittingParameter(value=3.5, name='cont', bounds_on=True, bounds=(0.5, 5.0))
        self._slow_factor = FittingParameter(value=0.985, name='slowf', bounds_on=True, bounds=(0.01, 1.0))
        self._rc_ele = FittingParameter(value=1.5, name='rc_ele', bounds_on=True, bounds=(0.5, 3.5))

        def update_lifetime():
            av.fluorescence_lifetime = self._fluorescence_lifetime.value
            av.rC_electron_transfer = self._rc_ele.value
            av.update_quenching_map()
            self.decay_changed = True
        self._fluorescence_lifetime._called_on_value_change = update_lifetime
        self._rc_ele._called_on_value_change = update_lifetime

        def update_diffusion():
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

    def update_model(self, **kwargs):
        decay_changed = kwargs.get('decay_changed', self.decay_changed)
        i = kwargs.get('inter', 1)
        t_step = (self.times[1] - self.times[0]) / i
        n_steps = (len(self.fit.data.y) - 1) * i
        if decay_changed:
            times, density, counts = self._av.get_donor_only_decay(n_it=n_steps, t_step=t_step, n_out=i)
            self._decay = counts
            self.decay_changed = False
        DecayModel.update_model(self, **kwargs)

import mfm.models.tcspc.nusiance
from mfm.models import parse


class ParseDecayModel(parse.ParseModel):

    def __init__(self, fit, **kwargs):
        parse.ParseModel.__init__(self, fit, **kwargs)
        self.convolve = kwargs.get('convolve', mfm.models.tcspc.nusiance.Convolve(name='convolve', fit=fit, **kwargs))
        self.corrections = kwargs.get('corrections',
                                      mfm.models.tcspc.nusiance.Corrections(name='corrections', fit=fit, model=self, **kwargs))
        self.generic = kwargs.get('generic', mfm.models.tcspc.nusiance.Generic(name='generic', fit=fit, **kwargs))

    def update_model(self, **kwargs):
        #verbose = kwargs.get('verbose', self.verbose)
        #scatter = kwargs.get('scatter', self.generic.scatter)
        background = kwargs.get('background', self.generic.background)
        #lintable = kwargs.get('lintable', self.corrections.lintable)

        parse.ParseModel.update_model(self, **kwargs)
        decay = self._y_values
        if self.convolve.irf is not None:
            decay = self.convolve.convolve(self._y_values, mode='full')[:self._y_values.shape[0]]

        self.convolve.scale(decay, self.fit.data, bg=background, start=self.fit.xmin, stop=self.fit.xmax)
        decay += self.generic.background
        decay[decay < 0.0] = 0.0
        if self.corrections.lintable is not None:
            decay *= self.corrections.lintable
        self._y_values = decay



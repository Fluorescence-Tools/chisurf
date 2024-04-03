from __future__ import annotations

import chisurf.models.tcspc.nusiance
from chisurf.models import parse


class ParseDecayModel(parse.ParseModel):

    def __init__(self, fit, **kwargs):
        parse.ParseModel.__init__(self, fit, **kwargs)
        self.convolve = kwargs.get('convolve', chisurf.models.tcspc.nusiance.Convolve(name='convolve', fit=fit, **kwargs))
        self.corrections = kwargs.get(
            'corrections',
            chisurf.models.tcspc.nusiance.Corrections(name='corrections', fit=fit, model=self, **kwargs)
        )
        self.generic = kwargs.get(
            'generic',
            chisurf.models.tcspc.nusiance.Generic(name='generic', fit=fit, **kwargs)
        )

    def update_model(self, **kwargs):
        scatter = kwargs.get('scatter', self.generic.scatter)
        background = kwargs.get('background', self.generic.background)
        lintable = kwargs.get('lintable', self.corrections.lintable)
        super(ParseDecayModel, self).update_model(**kwargs)
        decay = self.y
        if self.convolve.irf is not None:
            decay = self.convolve.convolve(
                self.y,
                mode='full',
                scatter=scatter
            )[:self.y.shape[0]]

        self.corrections.pileup(decay)
        self.convolve.scale(
            decay,
            start=self.fit.xmin,
            stop=self.fit.xmax,
            data=self.fit.data,
            bg=background,
        )
        decay += self.generic.background
        decay[decay < 0.0] = 0.0
        if lintable is not None:
            decay *= lintable
        self.y = decay


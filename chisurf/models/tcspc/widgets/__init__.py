# This file imports all widget classes from their respective modules
# to maintain backward compatibility with existing code.

# Define plot_cls_dist_default here to avoid circular imports
import chisurf.plots
import chisurf.math.datatools

plot_cls_dist_default = [
    (
        chisurf.plots.LinePlot,
        {
            'd_scalex': 'lin',
            'd_scaley': 'log',
            'r_scalex': 'lin',
            'r_scaley': 'lin',
            'x_label': 'x',
            'y_label': 'y',
            'plot_irf': True
        }
     ),
    (chisurf.plots.FitInfo, {}),
    (chisurf.plots.ParameterScanPlot, {}),
    (chisurf.plots.ResidualPlot, {}),
    (
        chisurf.plots.DistributionPlot,
        {
            'distribution_options': {
                'Distance': {
                    'attribute': 'distance_distribution',
                    'accessor': lambda x, **kwargs: (x[0][0], x[0][1]),
                    'accessor_kwargs': {'sort': False},
                    'curve_options': {
                        'symbol': "t",
                    }
                },
                'FRET-rate constant': {
                    'attribute': 'fret_rate_spectrum',
                    'accessor': chisurf.math.datatools.interleaved_to_two_columns,
                    'accessor_kwargs': {'sort': True},
                    'curve_options': {
                        'symbol': "o"
                    }
                },
                'Lifetime': {
                    'attribute': 'lifetime_spectrum',
                    'accessor': chisurf.math.datatools.interleaved_to_two_columns,
                    'accessor_kwargs': {'sort': True},
                    'curve_options': {
                        'symbol': "o"
                    }
                }
            }
        }
    )
]

from chisurf.models.tcspc.widgets.convolve import ConvolveWidget
from chisurf.models.tcspc.widgets.corrections import CorrectionsWidget
from chisurf.models.tcspc.widgets.generic import GenericWidget
from chisurf.models.tcspc.widgets.anisotropy import AnisotropyWidget
from chisurf.models.tcspc.widgets.pddem import PDDEMWidget, PDDEMModelWidget
from chisurf.models.tcspc.widgets.lifetime import (
    LifetimeWidget, 
    LifetimeModelWidgetBase, 
    LifetimeModelWidget, 
    LifetimeMixtureModelWidget
)
from chisurf.models.tcspc.widgets.gaussian import GaussianWidget, GaussianModelWidget
from chisurf.models.tcspc.widgets.discrete_distance import DiscreteDistanceWidget
from chisurf.models.tcspc.widgets.fret_rate import FRETrateModelWidget
from chisurf.models.tcspc.widgets.worm_like_chain import WormLikeChainModelWidget
from chisurf.models.tcspc.widgets.parse_decay import ParseDecayModelWidget
from chisurf.models.tcspc.widgets.lifetime_mix import LifetimeMixModelWidget
from chisurf.models.tcspc.widgets.et import EtModelFreeWidget

# This file imports all widget classes from their respective modules
# to maintain backward compatibility with existing code.

# Define plot_cls_dist_default here to avoid circular imports
import chisurf.gui.plots
import chisurf.core.math.datatools

plot_cls_dist_default = [
    (
        chisurf.gui.plots.LinePlot,
        {
            'd_scalex': 'lin',
            'd_scaley': 'log',
            'r_scalex': 'lin',
            'r_scaley': 'lin',
            'x_label': 'time (ns)',
            'y_label': 'counts',
            'plot_irf': True
        }
     ),
    (chisurf.gui.plots.FitTablePlot, {}),
    (chisurf.gui.plots.FitInfo, {}),
    (chisurf.gui.plots.ParameterScanPlot, {}),
    (chisurf.gui.plots.ResidualPlot, {}),
    (
        chisurf.gui.plots.DistributionPlot,
        {
            'distribution_options': {
                'Distance': {
                    'attribute': 'distance_distribution',
                    'accessor': lambda x, **kwargs: (x[0][0], x[0][1]),
                    'accessor_kwargs': {'sort': False},
                    'curve_options': {
                        'symbol': "t",
                        'bar_mode': 'sticks',
                    }
                },
                'FRET-rate constant': {
                    'attribute': 'fret_rate_spectrum',
                    'accessor': chisurf.core.math.datatools.interleaved_to_two_columns,
                    'accessor_kwargs': {'sort': True},
                    'curve_options': {
                        'symbol': "o"
                    }
                },
                'Lifetime': {
                    'attribute': 'lifetime_spectrum',
                    'accessor': chisurf.core.math.datatools.interleaved_to_two_columns,
                    'accessor_kwargs': {'sort': True},
                    'curve_options': {
                        'symbol': "o"
                    }
                }
            }
        }
    )
]

from chisurf.gui.widgets.models.tcspc.convolve import ConvolveWidget
from chisurf.gui.widgets.models.tcspc.corrections import CorrectionsWidget
from chisurf.gui.widgets.models.tcspc.generic import GenericWidget
from chisurf.gui.widgets.models.tcspc.anisotropy import AnisotropyWidget
from chisurf.gui.widgets.models.tcspc.pddem import PDDEMWidget, PDDEMModelWidget
from chisurf.gui.widgets.models.tcspc.lifetime import (
    LifetimeWidget, 
    LifetimeModelWidgetBase, 
    LifetimeModelWidget, 
    LifetimeMixtureModelWidget
)
from chisurf.gui.widgets.models.tcspc.gaussian import GaussianWidget, GaussianModelWidget
from chisurf.gui.widgets.models.tcspc.discrete_distance import DiscreteDistanceWidget
from chisurf.gui.widgets.models.tcspc.fret_rate import FRETrateModelWidget
from chisurf.gui.widgets.models.tcspc.worm_like_chain import WormLikeChainModelWidget
from chisurf.gui.widgets.models.tcspc.parse_decay import ParseDecayModelWidget
from chisurf.gui.widgets.models.tcspc.lifetime_mix import LifetimeMixModelWidget
try:
    from chisurf.gui.widgets.models.tcspc.et import EtModelFreeWidget
except Exception:
    EtModelFreeWidget = None
from chisurf.gui.widgets.models.tcspc.fret_structure import FRETStructureWidget
from chisurf.gui.widgets.models.tcspc.maxent import (
    MaxEntLifetimeModelWidget,
    MaxEntFRETModelWidget,
)

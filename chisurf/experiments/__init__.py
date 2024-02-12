import chisurf.experiments.experiment
import chisurf.experiments.reader
import chisurf.experiments.fcs
import chisurf.experiments.tcspc
import chisurf.experiments.globalfit
import chisurf.experiments.modelling
from chisurf.experiments.experiment import Experiment

types = {
    'tcspc': Experiment('TCSPC'),
    'fcs': Experiment('FCS'),
    'structure': Experiment('Modelling')
}

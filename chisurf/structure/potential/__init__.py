from . import potentials
from . import widgets

potentialDict = dict()
potentialDict['H-Bond'] = widgets.HPotentialWidget
potentialDict['AV-Potential'] = widgets.AvPotentialWidget
potentialDict['Iso-UNRES'] = widgets.CEPotentialWidget
potentialDict['Miyazawa-Jernigan'] = widgets.MJPotentialWidget
potentialDict['Go-Potential'] = widgets.GoPotentialWidget
potentialDict['ASA-Calpha'] = widgets.AsaWidget
potentialDict['Radius of Gyration'] = widgets.RadiusGyrationWidget
potentialDict['Clash potential'] = widgets.ClashPotentialWidget

from . import potentials

potentialDict = dict()
potentialDict['H-Bond'] = potentials.HPotentialWidget
potentialDict['AV-Potential'] = potentials.AvPotentialWidget
potentialDict['Iso-UNRES'] = potentials.CEPotentialWidget
potentialDict['Miyazawa-Jernigan'] = potentials.MJPotentialWidget
potentialDict['Go-Potential'] = potentials.GoPotentialWidget
potentialDict['ASA-Calpha'] = potentials.AsaWidget
potentialDict['Radius of Gyration'] = potentials.RadiusGyrationWidget
potentialDict['Clash potential'] = potentials.ClashPotentialWidget

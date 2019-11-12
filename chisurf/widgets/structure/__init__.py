import chisurf.widgets.structure.potentials

potentialDict = dict()
potentialDict['H-Bond'] = chisurf.widgets.structure.potentials.HPotentialWidget
potentialDict['AV-Potential'] = chisurf.widgets.structure.potentials.AvPotentialWidget
potentialDict['Iso-UNRES'] = chisurf.widgets.structure.potentials.CEPotentialWidget
potentialDict['Miyazawa-Jernigan'] = chisurf.widgets.structure.potentials.MJPotentialWidget
potentialDict['Go-Potential'] = chisurf.widgets.structure.potentials.GoPotentialWidget
potentialDict['ASA-Calpha'] = chisurf.widgets.structure.potentials.AsaWidget
potentialDict['Radius of Gyration'] = chisurf.widgets.structure.potentials.RadiusGyrationWidget
potentialDict['Clash potential'] = chisurf.widgets.structure.potentials.ClashPotentialWidget

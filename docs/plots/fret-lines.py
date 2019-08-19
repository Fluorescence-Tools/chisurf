import matplotlib.pyplot as p
from lib.tools.fret_lines import FRETLineGenerator

import mfm.models.tcspc as m

R1 = 80
R2 = 35
fl = FRETLineGenerator()
fl.model = m.GaussianModel

fl.model.parameter_dict['DOnly'].value = 0.0
# Add Donor-Only lifetime
fl.model.donors.append(1.0, 4)
# Add distance
fl.model.append(55.0, 10, 1.0)
fl.model.parameter_dict['R(G,1)'].value = 40.0
fl.model.parameter_dict['s(G,1)'].value = 8.0


## Static FRET-line ##
# Set the parameter which is varied
fl.parameter_name = 'R(G,1)'
fl.parameter_range = 0.01, 200
fl.update_model()
p.plot(fl.fluorescence_averaged_lifetimes, fl.transfer_efficiencies)

## Dynamic FRET-line ##
fl.model.append(55.0, 10, 1.0)
fl.model.parameter_dict['R(G,1)'].value = 30
fl.model.parameter_dict['R(G,2)'].value = 80
fl.model.parameter_dict['x(G,1)'].value = 1
fl.model.parameter_dict['x(G,2)'].value = 0

fl.parameter_name = 'x(G,2)'
fl.parameter_range = 0, 10

fl.update_model()

p.plot(fl.fluorescence_averaged_lifetimes, fl.transfer_efficiencies)

p.title(r'FRET-lines')
p.ylim([0,1])
p.xlim([0,4])
p.show()
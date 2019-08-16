import matplotlib.pyplot as p

import mfm
from mfm.fluorescence.simulation.dye_diffusion import DyeDecay, FRETDecay

pdb_filename = 'y:/Documents/ChiSurf/sample_data/model/hgbp1/hGBP1_closed.pdb'
structure = mfm.Structure(pdb_filename)

f_d0 = DyeDecay(tau0=4.0)
f_d0.structure = pdb_filename
f_d0.attachment_residue = 496
f_d0.attachment_atom_name = 'CB'
f_d0.update_model()

f_a0 = DyeDecay(tau0=1.2)
f_a0.structure = pdb_filename
f_a0.attachment_residue = 540
f_a0.attachment_atom_name = 'CB'
f_a0.update_model()


f_da = FRETDecay(donor_diffusion=f_d0, acceptor_diffusion=f_a0)
r_da = f_da.r_da
p.subplot(2, 2, 1)
p.plot(r_da, 'b')
p.ylabel('RDA')

kFRET = f_da.kFRET
p.subplot(2, 2, 2)
p.plot(kFRET, 'r')
p.ylabel('kFRET')


p.subplot(2, 2, 3)
x, y = f_d0.get_histogram()
p.semilogy(x, y + 1, 'g')
x, y = f_da.get_histogram()
p.semilogy(x, y + 1, 'r')
p.ylabel('F(t)')

p.show()
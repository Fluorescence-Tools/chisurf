
FRET-efficiency of Trajectories
-------------------------------

Based on the example of hGBP1 where a transition between the FRET-major to the FRET-minor state is simulated
using the known crystal structure 1f5n as intermediate the evolution of the transfer efficiency during the course
of this transition is calculated. In hGBP1 the amino-acids 18, 254, 344, 496, 525, 540, 577 were labeled. In the
crystal structure residues are missing and the amino-acid numbers are renumbered according to:  18 -> 12,
254 -> 241, 344 -> 331, 481 -> 468, 496 -> 483, 525 -> 512, 540 -> 527, 577 -> 564

First the trajectory is loaded as :py:class:`mfm.structure.trajectory.TrajectoryFile` object.
Next a structure object is obtained from the trajectory. By default the frames of a
:py:class:`mfm.structure.TrajectoryFile` object are :py:class:`mfm.structure.Structure` objects.
This behavior is modified by changing the :py:attr:`mfm.structure.TrajectoryFile.structure` attribute.
The obtained structure object is converted into a :py:class:`mfm.structure.LabeledStructure` object and
assigned to the trajectory.

Finally, the `LabeledStructure` is labeled by a donor and acceptor. This is accomplished by assignening dictionaries
which describe the Accessible volumes. These dictionaries are internally passed to
:py:class:`mfm.fluorescence.fps.AV`.

>>> t = mfm.structure.TrajectoryFile('./sample_data/modelling/trajectory/h5-file/hgbp1_transition.h5', mode='r', stride=1)
>>> structure = mfm.structure.LabeledStructure(t[0])
>>> donor_description = {'residue_seq_number': 338, 'atom_name': 'CB'}
>>> acceptor_description = {'residue_seq_number': 564, 'atom_name': 'CB'}
>>> structure.donor_label = donor_description
>>> structure.acceptor_label = acceptor_description
>>> structure.donor_lifetime_spectrum = np.array([1., 4.])
>>> t.structure = structure

With the so generated `labelled` structure for instance the transfer efficency over the course of the trajectory
can be calculated.

>>> transfer_efficiencies = [s.transfer_efficency for s in t[:10]]
>>> import pylab as p
>>> p.plot(transfer_efficiencies)

Now these calculations are performed for a whole set of distances

.. code-block:: python

    fret_pair_names = ("18-344", "18-540", "18-577", "254-344", "254-540", "254-577", "344-481", "344-496", "344-525", "344-540", "481-525", "496-540")
    fret_pairs = ((12, 331), (12, 527), (12, 564), (241, 331), (241, 527), (241, 564), (331, 468), (331, 483), (331, 512), (331, 527), (468, 512), (483, 527))
    t = mfm.structure.TrajectoryFile('./sample_data/modelling/trajectory/h5-file/hgbp1_transition.h5', mode='r', stride=1)
    structure = mfm.structure.LabeledStructure(t[0])
    for pn, da in zip(fret_pair_names, fret_pairs):
        d, a = da
        donor_description = {'residue_seq_number': d, 'atom_name': 'CB'}
        acceptor_description = {'residue_seq_number': a, 'atom_name': 'CB'}
        structure.donor_label = donor_description
        structure.acceptor_label = acceptor_description
        structure.donor_lifetime_spectrum = np.array([1., 4.])
        t.structure = structure
        transfer_efficiencies = [s.transfer_efficency for s in t]
        fn = pn + '.txt'
        np.savetxt(fn, np.array(transfer_efficiencies))



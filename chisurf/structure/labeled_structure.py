from __future__ import annotations

import numpy as np

import chisurf.mfm as mfm
import chisurf.fluorescence
from mfm.structure.structure import Structure


def av_distance_distribution(
        structure: Structure,
        donor_av_parameter,
        acceptor_av_parameter,
        **kwargs
):
    """Get the distance distribution between two accessible volumes defined by the parameters passed as
    an argument.

    :param structure: a structure :py:class:`..Structure`
    :param donor_av_parameter: a dictionary of parameters defining the donor (see AV-class)
    :param acceptor_av_parameter: a dictionary of parameters defining the acceptor (see AV-class)
    :param kwargs:
    :return:

    Examples
    --------

import chisurf.mfm as mfm.structure    >>> structure = mfm.structure.Structure('./test/data/modelling/pdb_files/hGBP1_closed.pdb')
    >>> donor_description = {'residue_seq_number': 18, 'atom_name': 'CB'}
    >>> acceptor_description = {'residue_seq_number': 577, 'atom_name': 'CB'}
    >>> pRDA, rda = av_distance_distribution(structure, donor_av_parameter=donor_description, acceptor_av_parameter=acceptor_description)

    """
    av_donor = chisurf.fluorescence.fps.ACV(structure, **donor_av_parameter)
    av_acceptor = chisurf.fluorescence.fps.ACV(structure, **acceptor_av_parameter)
    amplitude, distance = av_donor.pRDA(av_acceptor, **kwargs)
    return amplitude, distance


def av_fret_rate_spectrum(
        structure: Structure,
        donor_av_parameter,
        acceptor_av_parameter,
        **kwargs
):
    """Get FRET-rate spectrum for a given donor and acceptor attachment point

    :param donor_av_parameter: parameters describing the donor accessible volume (see AV-class)
    :param structure: a structure
    :param acceptor_av_parameter: parameters describing the acceptor accessible volume (see AV-class)
    :return:

    Examples
    --------

import chisurf.mfm as mfm.structure    >>> structure = mfm.structure.Structure('./test/data/modelling/pdb_files/hGBP1_closed.pdb')
    >>> donor_description = {'residue_seq_number': 18, 'atom_name': 'CB'}
    >>> acceptor_description = {'residue_seq_number': 577, 'atom_name': 'CB'}
    >>> rs = av_fret_rate_spectrum(structure, donor_description, acceptor_description)

    """
    forster_radius = kwargs.get('forster_radius', mfm.settings.cs_settings['fret']['forster_radius'])
    kappa2 = kwargs.get('forster_radius', mfm.settings.cs_settings['fret']['kappa2'])
    tau0 = kwargs.get('tau0', mfm.settings.cs_settings['fret']['tau0'])
    interleave = kwargs.get('interleave', True)

    p_rda, rda = av_distance_distribution(structure, donor_av_parameter=donor_av_parameter, acceptor_av_parameter=acceptor_av_parameter, **kwargs)
    d = np.array([[p_rda, rda]])
    rs = chisurf.fluorescence.general.distribution2rates(
        d, tau0=tau0,
        kappa2=kappa2,
        forster_radius=forster_radius
    )
    if interleave:
        return np.hstack(rs).ravel([-1])
    else:
        return rs


def av_lifetime_spectrum(
        structure: Structure,
        donor_lifetime_spectrum: np.array,
        **kwargs
):
    """Get an interleaved lifetime spectrum for a given donor lifetime spectrum given two definitions of
    a donor and an acceptor accessible volume. This function uses :py:meth:`~Structure.av_fret_rate_spectrum`
    to calculate the FRET-rate spectrum and uses this rate-spectrum to calculate the lifetime spectrum.

    :param donor_lifetime_spectrum: interleaved donor lifetime spectrum
    :param donor_av_parameter: definition of the donor accessible volume
    :param acceptor_av_parameter: definition of the acceptor accessible volume
    :param kwargs:
    :return:

    Examples
    --------

import chisurf.mfm as mfm.structure    >>> structure = mfm.structure.Structure('./test/data/modelling/pdb_files/hGBP1_closed.pdb')
    >>> d_av = {'residue_seq_number': 18, 'atom_name': 'CB'} # donor attachment and description of the linker
    >>> a_av = {'residue_seq_number': 577, 'atom_name': 'CB'} # acceptor description and linker
    >>> ds = np.array([0.8, 4., 0.2, 1.5]) # donor_lifetime_spectrum
    >>> lifetime_spectrum = av_lifetime_spectrum(structure, ds, donor_av_parameter=d_av, acceptor_av_parameter=a_av)
    """
    donly = kwargs.get('donly', 0.0)
    rs = av_fret_rate_spectrum(structure, **kwargs)
    return chisurf.fluorescence.general.rates2lifetimes(rs, donor_lifetime_spectrum, x_donly=donly)


def av_filtered_fcs_weights(
        structure: Structure,
        lifetime_filters,
        time_axis: np.array,
        **kwargs
):
    """Passes all ``kwargs`` parameters to the method in :py:meth:`~Structure.av_lifetime_spectrum` and
    calculates the filters weights for a set of lifetime filters ``lifetime_filters``.

    :param lifetime_filters: numpy array
    :return:

    Examples
    --------

    Read two different structures (a open and a closed structure) calculate the fluorescence intensity decay
    for these structures. Using these structures fluorescence filters are constructed and the weight of an
    intermediate structure with respect to the two lifetime filters is calculated.

    >>> from chisurf.fluorescence.fcs.filtered import calc_lifetime_filter
    >>> from chisurf.fluorescence.general import calculate_fluorescence_decay
    >>> from mfm.structure.structure import Structure
    >>> from mfm.structure.trajectory import TrajectoryFile

    Define where the donor and the acceptor are attached to

    >>> dl = np.array([1.0, 4.]) # donor lifetime spectrum
    >>> d_av = {'residue_seq_number': 18, 'atom_name': 'CB'}
    >>> a_av = {'residue_seq_number': 577, 'atom_name': 'CB'}

    Load a set of structures

    >>> structure_closed = Structure('./test/data/modelling/pdb_files/hGBP1_closed.pdb')
    >>> structure_open = Structure('./test/data/modelling/pdb_files/hGBP1_open.pdb')
    >>> structure_middle = Structure('./test/data/modelling/pdb_files/hGBP1_middle.pdb')
    >>> ls_closed = av_lifetime_spectrum(structure_closed, donor_lifetime_spectrum=dl, donor_av_parameter=d_av, acceptor_av_parameter=a_av)
    >>> ls_open = av_lifetime_spectrum(structure_open, donor_lifetime_spectrum=dl, donor_av_parameter=d_av, acceptor_av_parameter=a_av)
    >>> ls_middle = av_lifetime_spectrum(structure_middle, donor_lifetime_spectrum=dl, donor_av_parameter=d_av, acceptor_av_parameter=a_av)

    Define an fluorescence decay which serves as a reference which is used to calculate fluorescence lifetime
    filters. The calculated fluorescence lifetime filters are later used to determine weights of the structures.
    These weights could can later be auto-correlated and cross-correlated.

    >>> times = np.linspace(0, 20, num=50)
    >>> times, d_closed = calculate_fluorescence_decay(ls_closed, time_axis=times)
    >>> times, d_open = calculate_fluorescence_decay(ls_open, time_axis=times)
    >>> times, d_middle = calculate_fluorescence_decay(ls_middle, time_axis=times)
    >>> fraction_closed = 0.33
    >>> fraction_middle = 0.33
    >>> fraction_open = 0.33
    >>> experimental_decay = fraction_closed * d_closed + fraction_middle * d_middle + d_open * fraction_open
    >>> decays = [d_closed, d_middle, d_open]
    >>> lf = calc_lifetime_filter(decays=decays, experimental_decay=experimental_decay)

    The fluorescence lifetime filter correctly identifies the open, semi-open and closed conformation.

    >>> av_filtered_fcs_weights(structure_closed, lifetime_filters=lf, time_axis=times, donor_lifetime_spectrum=dl, donor_av_parameter=d_av, acceptor_av_parameter=a_av)
    array([ 0.97265041,  0.03609784, -0.00888892])
    >>> av_filtered_fcs_weights(structure_middle, lifetime_filters=lf, time_axis=times, donor_lifetime_spectrum=dl, donor_av_parameter=d_av, acceptor_av_parameter=a_av)
    array([ 0.01791727,  0.97488255,  0.00726864])
    >>> av_filtered_fcs_weights(structure_open, lifetime_filters=lf, time_axis=times, donor_lifetime_spectrum=dl, donor_av_parameter=d_av, acceptor_av_parameter=a_av)
    array([ 0.00831755, -0.01459384,  1.00628016])


    This method can be used together with trajectories. Use the limiting states to define filters. For the protein
    hGBP1 two limiting states are known, with the fraction 0.66 (state-1) and 0.33 (state-2). Using corase-grained
    models of these limiting states fluorescence decays are calculated and the filters are determined.

import chisurf.mfm as mfm.structure    >>> structure_1 = mfm.structure.Structure('./test/data/modelling/trajectory/h5-file/steps/0_major.pdb')

    >>> structure_1 = Structure('./test/data/modelling/trajectory/h5-file/steps/3_minor.pdb')
    >>> structure_2 = Structure('./test/data/modelling/trajectory/h5-file/steps/3_minor.pdb')
    >>> ls_1 = av_lifetime_spectrum(structure_1, donor_lifetime_spectrum=dl, donor_av_parameter=d_av, acceptor_av_parameter=a_av)
    >>> ls_2 = av_lifetime_spectrum(structure_2, donor_lifetime_spectrum=dl, donor_av_parameter=d_av, acceptor_av_parameter=a_av)
    >>> times = np.linspace(0, 20, num=50)
    >>> times, d_1 = calculate_fluorescence_decay(ls_1, time_axis=times)
    >>> times, d_2 = calculate_fluorescence_decay(ls_2, time_axis=times)
    >>> fraction_1 = 0.66
    >>> fraction_2 = 0.33
    >>> experimental_decay = fraction_1 * d_1 + fraction_2 * d_2
    >>> decays = [d_1, d_2]
    >>> lf = calc_lifetime_filter(decays=decays, experimental_decay=experimental_decay)

    Now the trajectory of the transition using the crystal-structure as intermediate state was simulated. Using this
    trajectory weights are associated to each frame which correspond to the first and the second state.

    >>> traj = TrajectoryFile('./test/data/modelling/trajectory/h5-file/hgbp1_transition.h5', reading_routine='r')
    >>> d_av = {'residue_seq_number': 12, 'atom_name': 'CB'}  # the resiude numbers are slightly shifted
    >>> a_av = {'residue_seq_number': 567, 'atom_name': 'CB'}
    >>> weights = [av_filtered_fcs_weights(s, lifetime_filters=lf, time_axis=times, donor_lifetime_spectrum=dl, donor_av_parameter=d_av, acceptor_av_parameter=a_av) for s in traj]

    Using the weights which were calculated for each frame species auto-correlations and species cross-correlations
    can be calculated.

    """
    lifetime_spectrum = av_lifetime_spectrum(structure, **kwargs)
    convolve = kwargs.get('convolve', None)
    if isinstance(convolve, chisurf.chisurf.fluorescence.tcspc.convolve.Convolve):
        decay = convolve.convolve(data=lifetime_spectrum, **kwargs)
    else:
        time_axis, decay = chisurf.fluorescence.calculate_fluorescence_decay(lifetime_spectrum, time_axis=time_axis)
    weights = np.dot(lifetime_filters, decay)
    return weights


class LabeledStructure(Structure):
    """This class is handles FRET-labeled molecule (so far only a single donor, and single acceptor) and provides
    convenience Attributes which only apply to FRET-samples

    Attributes
    ----------
    donor_lifetime_spectrum : array
        Interleaved array of amplitudes and lifetimes of the donor in absence of an acceptor

    donor_label : dict
        A dictionary which describes the labeling position of the donor (see :py:class:`chisurf.fluorescence.fps.AV`)

    acceptor_label : dict
        A dictionary which describes the labeling position of the acceptor (see :py:class:`chisurf.fluorescence.fps.AV`)

    distance_distribution: list of arrays
        A histogram of the donor-acceptor distance distribution for a given pair of lables

    fret_rate_spectrum: array
        Interleaved array of amplitudes and FRET-rate constants

    lifetime_spectrum: array
        Interleaved array of amplitudes and donor-lifetimes in presence of FRET


    Examples
    --------

    >>> import chisurf.mfm as mfm
    >>> structure = mfm.structure.structure.LabeledStructure('./test/data/modelling/pdb_files/hGBP1_closed.pdb', verbose=True)
    >>> donor_description = {'residue_seq_number': 18, 'atom_name': 'CB'}
    >>> acceptor_description = {'residue_seq_number': 577, 'atom_name': 'CB'}
    >>> structure.donor_label = donor_description
    >>> structure.acceptor_label = acceptor_description

    """

    @property
    def donor_lifetime_spectrum(
            self
    ) -> np.array:
        return self._ds

    @donor_lifetime_spectrum.setter
    def donor_lifetime_spectrum(
            self,
            v: np.array
    ):
        self._ds = v

    @property
    def donor_label(self):
        return self._donor_description

    @donor_label.setter
    def donor_label(self, v):
        self._donor_description = v
        self._donor_av = chisurf.fluorescence.fps.ACV(self, **self._donor_description)

    @property
    def acceptor_label(self):
        return self._acceptor_description

    @acceptor_label.setter
    def acceptor_label(self, v):
        self._acceptor_description = v
        self._acceptor_av = chisurf.fluorescence.fps.ACV(self, **self._donor_description)

    @property
    def distance_distribution(self):
        av_donor = self.donor_av
        av_acceptor = self.acceptor_av
        amplitude, distance = av_donor.pRDA(av_acceptor)
        return amplitude, distance

    @property
    def fret_rate_spectrum(
            self
    ) -> np.array:
        forster_radius = self.forster_radius
        kappa2 = self.kappa2
        tau0 = self.tau0
        p_rda, rda = self.distance_distribution
        d = np.array([[p_rda, rda]])
        rs = chisurf.fluorescence.general.distribution2rates(
            d,
            tau0=tau0,
            kappa2=kappa2,
            forster_radius=forster_radius
        )
        return np.hstack(rs).ravel([-1])

    @property
    def lifetime_spectrum(
            self
    ) -> np.array:
        rs = self.fret_rate_spectrum
        ds = self.donor_lifetime_spectrum
        return chisurf.fluorescence.general.rates2lifetimes(rs, ds)

    @property
    def transfer_efficency(self) -> float:
        tau_x_da = chisurf.fluorescence.general.species_averaged_lifetime(self.lifetime_spectrum)
        tau_x_d0 = chisurf.fluorescence.general.species_averaged_lifetime(self.donor_lifetime_spectrum)
        return 1. - tau_x_da / tau_x_d0

    @property
    def donor_av(self):
        return self._donor_av

    @property
    def acceptor_av(self):
        return self._acceptor_av

    @property
    def pRDA(self):
        """donor-acceptor distance distribution
        """
        return self._donor_av.pRDA(self._acceptor_av)

    @property
    def dRDAE(self):
        """fluorescence weighted averaged donor-acceptor distance
        """
        return self._donor_av.dRDAE(self._acceptor_av)

    @property
    def dRDA(self):
        """average donor-acceptor distance
        """
        return self._donor_av.dRDA(self._acceptor_av)

    @property
    def dRmp(self):
        """distance between the mean positions
        """
        return self._donor_av.dRmp(self._acceptor_av)

    def update(self):
        self._acceptor_av = chisurf.fluorescence.fps.ACV(self, **self._acceptor_description)
        self._donor_av = chisurf.fluorescence.fps.ACV(self, **self._donor_description)

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(LabeledStructure, self).__init__(*args, **kwargs)
        self._donor_description = kwargs.get('donor_av_parameter', None)
        self._acceptor_description = kwargs.get('acceptor_av_parameter', None)
        self._ds = np.array([1.0, 4.0], dtype=np.float64)
        self._donor_av = None
        self._acceptor_av = None
        self.forster_radius = kwargs.get('forster_radius', mfm.settings.cs_settings['fret']['forster_radius'])
        self.tau0 = kwargs.get('tau0', mfm.settings.cs_settings['fret']['tau0'])
        self.kappa2 = kwargs.get('kappa2', mfm.settings.cs_settings['fret']['kappa2'])


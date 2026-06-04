from __future__ import annotations

import chisurf.structure.label
import chisurf.structure.label.distribution
import chisurf.structure.label.functions
from scikit_fluorescence import typing


import json
import scipy.stats
import numpy as np

import IMP.atom
import scikit_fluorescence as skf
import scikit_fluorescence.io.structure
import chisurf.structure.label.av


class AvPotential(object):
    """
    The AvPotential class provides the possibility to calculate the reduced or unreduced chi2 given a set of
    labeling positions and experimental distances. Here the labeling positions and distances are provided as
    dictionaries.

    Examples
    --------
    >>> import json
    >>> labeling_file = './test/data/model/labeling.fps.json'
    >>> labeling = json.load(open(labeling_file, 'r'))
    >>> distances = labeling['Distances']
    >>> positions = labeling['Positions']
    >>> import scikit_fluorescence as skf
    >>> import chisurf.structure.label.av
    >>> import scikit_fluorescence.io.structure
    >>> av_potential = chisurf.structure.label.av.AvPotential(distances=distances, positions=positions)
    >>> atoms = skf.io.structure.read_coordinates('./test/data/structure/1fat.cif')
    >>> av_potential.get_chi2(atoms)

    Labeling file
    >>> import scikit_fluorescence as skf
    >>> import chisurf.structure.label.av
    >>> import scikit_fluorescence.io.structure
    >>> labeling_file = './test/data/hgbp1/labeling.fps.json'
    >>> atoms = skf.io.structure.read_coordinates('./test/data/hgbp1/protein.pdb')
    >>> av_potential = chisurf.structure.label.av.AvPotential(labeling_file=labeling_file, atoms=atoms)
    >>> av_potential.get_chi2()

    """

    distances: typing.Dict = None
    positions: typing.Dict = None
    verbose: bool = False

    _atoms: np.ndarray = None
    _labeling_file: str = ''

    def __init__(
            self,
            atoms: np.ndarray = None,
            labeling_file: str = '',
            rda_axis: np.ndarray = None,
            rda_min: float = 10.0,
            rda_max: float = 200.0,
            n_rda_bins = 100,
            verbose: bool = False,
            number_of_distance_samples: int = 10000,
            minimum_number_of_points_in_av: int = 150,
            Distances: typing.Dict = None,  # Uppercase because of fps.json format
            Positions: typing.Dict = None  # Uppercase because of fps.json format
    ):
        """Initialize the labeling-distance potential evaluator."""
        self.verbose = verbose
        if rda_axis is None:
            rda_axis = np.linspace(rda_min, rda_max, n_rda_bins)
        self.rda_axis = rda_axis
        self.distances = Distances
        self.positions = Positions
        self.n_av_samples = number_of_distance_samples
        self.min_av = minimum_number_of_points_in_av
        self.avs = dict()
        self.labeling_file = labeling_file
        if isinstance(atoms, np.ndarray):
            self.atoms = atoms

    @staticmethod
    def score_structure(
            obj,
            labeling_file: str
    ) -> float:
        """Convenience method to score IMP.atom.Hierarchy and numpy atom arrays

        :param obj: either an IMP.atom.Hierarchy or a numpy atom array
        :param labeling_file: the path of a labeling file
        :return: chi2 (score) of the object

        Examples
        --------
        >>> import scikit_fluorescence as skf
        >>> import scikit_fluorescence.modeling
        >>> import IMP.atom
        >>> model = IMP.Model()
        >>> filename = './test/data/hgbp1/protein.pdb'
        >>> labeling_file = './test/data/hgbp1/labeling.fps.json'
        >>> imp_hier = IMP.atom.read_pdb(filename, model, IMP.atom.NonWaterPDBSelector())
        >>> chi2 = chisurf.structure.label.av.AvPotential.score_structure(imp_hier, labeling_file)
        >>> chi2
        1.125509321839683

        """
        avp = AvPotential(
            labeling_file=labeling_file
        )
        if isinstance(obj, IMP.atom.Hierarchy):
            atoms = scikit_fluorescence.io.structure.convert_atoms(
                IMP.atom.get_leaves(obj)
            )
        elif isinstance(obj, np.ndarray):
            atoms = obj
        else:
            raise ValueError("Only atom arrays and IMP.atom.Hierarchy are supported")
        return avp.get_chi2(atoms=atoms)

    @property
    def labeling_file(self):
        """Path to the labeling file used to load distances/positions."""
        return self._labeling_file

    @labeling_file.setter
    def labeling_file(self, v):
        """Set the labeling file path; loads distances/positions from JSON."""
        self._labeling_file = v
        with open(v, 'r') as fp:
            p = json.load(fp)
            self.distances = p["Distances"]
            self.positions = p["Positions"]

    @property
    def atoms(
            self
    ) -> np.ndarray:
        """
        The atoms used for the calculation of the accessible volumes
        """
        return self._atoms

    @atoms.setter
    def atoms(
            self,
            atoms: np.ndarray
    ):
        """Set the atoms and recompute accessible volumes."""
        self._atoms = atoms
        self.calc_avs()

    @property
    def chi2(self) -> float:
        """
        The current unreduced chi2 (recalculated at each call)
        """
        return self.get_chi2()

    def calc_avs(self):
        """
        Calculates/recalculates the accessible volumes.
        """
        if self._atoms is None:
            raise ValueError("The atoms are not set")
        if self.positions is None:
            raise ValueError("Positions not set unable to calculate AVs")
        avp = self
        arguments = [
            dict(
                {'atoms': avp.atoms, 'verbose': avp.verbose, },
                **avp.positions[position_key]
            )
            for position_key in avp.positions
        ]
        avs = map(
            lambda x: skf.modeling.label.av.LabelDistributionAV(**x),
            arguments
        )
        # in python 3 map cannot be subscribed
        av_list = list(avs)
        for i, position_key in enumerate(self.positions.keys()):
            self.avs[position_key] = av_list[i]

    def calc_distances(
            self,
            atoms: np.ndarray = None,
            verbose: bool = False
    ):
        """

        :param atoms: Structure
            If this object is provided the attributes regarding label-attachment are kept constant
            and the structure is changed prior calculation of the distances.
        :param verbose: bool
            If this is True output to stdout is generated
        """
        verbose = verbose or self.verbose
        if isinstance(atoms, np.ndarray):
            self.atoms = atoms
        for distance_key in self.distances:
            distance = self.distances[distance_key]
            av1 = self.avs[distance['position1_name']]
            av2 = self.avs[distance['position2_name']]
            distance_type = distance['distance_type']
            R0 = distance['Forster_radius']
            d12 = None
            if distance_type == 'RDAMean':
                d12 = chisurf.structure.label.functions.average_distance(av2)
            elif distance_type == 'Rmp':
                d12 = chisurf.structure.label.functions.distance_between_mean_positions(av1, av2)
            elif distance_type == 'RDAMeanE':
                d12 = av1.RDAE(av1, av2, R0)
            elif distance_type == 'get_distance_distribution':
               rda = np.array(distance['rda'])
               d12 = chisurf.structure.label.functions.histogram_rda(
                   dd1=av1,
                   dd2=av2,
                   rda_axis=rda
               )[0]
            distance['model_distance'] = d12
            if verbose:
                print("-------------")
                print("Distance: %s" % distance_key)
                print("Forster-Radius %.1f" % distance['Forster_radius'])
                print("Distance type: %s" % distance_type)
                print("Model distance: %s" % d12)
                print(
                    "Experimental distance: %.1f (-%.1f, +%.1f)" % (
                        distance['distance'],
                        distance['error_neg'], distance['error_pos']
                    )
                )

    def get_chi2(
            self,
            atoms: np.ndarray = None,
            reduced: bool = False,
            verbose: bool = False
    ):
        """

        :param atoms: Structure
            A Structure object if provided the attributes regarding label-attachment are kept constant
            and the structure is changed prior calculation of the distances.
        :param reduced: bool
            If True the reduced chi2 is calculated (by default False)
        :param verbose: bool
            Output to stdout
        :return: A float containig the chi2 (reduced or unreduced) of the current or provided structure.
        """
        verbose = self.verbose or verbose
        if isinstance(atoms, np.ndarray):
            self.atoms = atoms
        self.calc_distances(verbose=verbose)
        chi2 = 0.0
        for distance in list(self.distances.values()):
            dm = distance['model_distance']
            de = distance['distance']
            if distance['distance_type'] == 'pRDA':
                prda = np.array(distance['prda'])
                prda /= sum(prda)
                chi2 += sum((dm - prda)**2)
            else:
                error_neg = distance['error_neg']
                error_pos = distance['error_pos']
                d = dm - de
                chi2 += (d / error_neg) ** 2 if d < 0 else (d / error_pos) ** 2
        if reduced:
            return chi2 / (len(list(self.distances.keys())) - 1.0)
        else:
            return chi2

    def get_energy(
            self,
            atoms: np.ndarray,
            gauss_bond: bool = True
    ):
        """Return the total energy for the given atom array."""
        self.atoms = atoms
        if gauss_bond:
            energy = 0.0
            self.calc_distances()
            for distance in list(self.distances.values()):
                dm = distance['model_distance']
                de = distance['distance']
                error_neg = distance['error_neg']
                error_pos = distance['error_pos']
                err = error_neg if (dm - de) < 0 else error_pos
                energy -= scipy.stats.norm.pdf(de, dm, err)
            return energy
        else:
            return self.get_chi2(atoms=self.atoms)


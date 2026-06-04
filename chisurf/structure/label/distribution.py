from __future__ import annotations
from scikit_fluorescence import typing

import abc
import numpy as np

# import scikit_fluorescence as skf
# import scikit_fluorescence.io
# import scikit_fluorescence.math
# import scikit_fluorescence.decay
import chisurf.fio.structure
import chisurf.fluorescence
import chisurf.structure.label.functions


class LabelDistribution(abc.ABC):
    """
    Attributes
    ----------
    origin : numpy.array
        The reference point (zero) of origin of the label distribution.
    verbose: bool
        If set to True the methods are normally verbose
    simulation_grid_resolution: float
        The resolution of the density grid used for the simulation of
        the positional distribution, i.e., the density of the Label
    """

    _density: np.ndarray = None
    origin: np.ndarray = None
    verbose: bool = True
    simulation_grid_resolution: float
    position_name: str

    def __init__(
            self,
            simulation_type: str = None,
            origin: np.ndarray = None,
            simulation_grid_resolution: float = 0.5,
            position_name: str = "",
            verbose: bool = False,
            **kwargs
    ):
        """Initialize the abstract LabelDistribution base class."""
        self.position_name = position_name
        self.simulation_type = simulation_type
        self.origin = origin
        self.simulation_grid_resolution = simulation_grid_resolution
        self.verbose = verbose
        super().__init__()

    @property
    @abc.abstractmethod
    def points(self):
        """Sampled point cloud of the distribution (abstract)."""
        pass

    @property
    @abc.abstractmethod
    def density(self):
        """3D density array of the distribution (abstract)."""
        pass

    def save(
            self,
            filename: str,
            mode: str = 'xyz',
            density: np.ndarray = None,
            **kwargs
    ):
        """Saves the accessible volume as xyz-file or open-dx density file

        Examples
        --------
        >>> import scikit_fluorescence as skf
        >>> import scikit_fluorescence.data
        >>> import scikit_fluorescence.modeling
        >>> atoms = skf.data.protein_coordinates_hgbp1('./test/data/1fat.cif')
        >>> av = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=577, atom_name='CB')
        >>> av.save('test', reading_routine='xyz')
        >>> av.save('test', reading_routine='dx')
        >>> av.save('test', reading_routine='mrc')
        """
        if density is None:
            density = self.density
        try:
            nx, ny, nz = self.density.shape[0]
        except ValueError:
            raise ValueError("Only 3D densities are supported.")
        if nx != ny or nx != nz or ny != nz:
            assert ValueError("The density needs to be cubic.")
        if mode == 'dx':
            dg = self.simulation_grid_resolution
            ng = nx
            offset = (ng - 1) / 2 * dg
            chisurf.io.density.write_open_dx(
                filename,
                density,
                self.origin - offset,
                nx, ny, nz,
                dg, dg, dg
            )
        elif mode == 'mrc':
            dg = self.simulation_grid_resolution
            ng = nx  # assume that grid is cubic
            offset = (ng - 1) / 2 * dg
            chisurf.fio.structure.density.write_mrc(
                filename,
                density,
                self.origin - offset,
                nx, ny, nz,
                dg
            )

    def get_fret_rate_constant_spectrum(
            self,
            av: chisurf.structure.label.LabelDistribution,
            rda_axis: np.ndarray = None,
            rda_min: float = 5,
            rda_max: float = 200,
            n_rda_bins = 195,
            use_log: bool = True,
            n_samples: int = 100000,
            **kwargs
    ):
        """Compute the FRET rate constant spectrum versus donor-acceptor distance."""
        p, rda_axis = chisurf.structure.label.functions.histogram_rda(
            dd1=self,
            dd2=av,
            n_samples=n_samples,
            rda_axis=rda_axis,
            rda_min=rda_min,
            rda_max=rda_max,
            use_log=use_log,
            n_rda_bins=n_rda_bins
        )[1:]
        return chisurf.fluorescence.decay.compute_fret_rate_constants(
            p_rda=p,
            rda_axis=rda_axis,
            **kwargs
        )

    def get_donor_fret_lifetime_spectrum(
            self,
            donor_lifetime_spectrum: np.ndarray,
            **kwargs
    ):
        """
        :param donor_lifetime_spectrum:
        :param kwargs: parameters passed to get_fret_rate_constant_spectrum
        :return: interleaved spectrum of donor lifetimes in the presence of FRET
        """
        return chisurf.decay.compute_donor_lifetime_spectrum_with_fret(
            donor_lifetime_spectrum=np.array(donor_lifetime_spectrum, dtype=np.float),
            **kwargs
        )

    def get_distance_between_mean_positions(
            self,
            av: LabelDistribution,
    ):
        """
        Calculate the distance between the mean positions of the AVs

        :param av: accessible volume object
        :return:

        Examples
        --------
        >>> import scikit_fluorescence as skf
        >>> import scikit_fluorescence.io
        >>> import scikit_fluorescence.modeling
        >>> import scikit_fluorescence.data
        >>> atoms = skf.data.protein_coordinates_hgbp1()
        >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=18, atom_name='CB')
        >>> av2 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=577, atom_name='CB')
        >>> skf.modeling.label.distance_between_mean_positions(av1, av2)
        56.96909
        """
        return chisurf.structure.label.functions.distance_between_mean_positions(
            self, av
        )

    def average_distance(
            self,
            av: LabelDistribution,
            **kwargs
    ):
        """Calculate the average distance to the second accessible volume

        :param av:
        :return:

        Examples
        --------
        >>> import scikit_fluorescence as skf
        >>> import scikit_fluorescence.modeling
        >>> import scikit_fluorescence.io
        >>> import scikit_fluorescence.data
        >>> atoms = skf.data.protein_coordinates_hgbp1()
        >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=18, atom_name='CB')
        >>> av2 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=577, atom_name='CB')
        >>> av1.average_distance(av2)
        59.714254803678024
        """
        return chisurf.structure.label.functions.average_distance(self, av, **kwargs)

    def standard_deviation_of_distances(
            self,
            av: LabelDistribution,
    ):
        """Calculates the width of a DA-distance distribution

        :param av:
        :return:

        Examples
        --------
        >>> import scikit_fluorescence as skf
        >>> import scikit_fluorescence.modeling
        >>> import scikit_fluorescence.io
        >>> import scikit_fluorescence.data
        >>> atoms = skf.data.protein_coordinates_hgbp1()
        >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=18, atom_name='CB')
        >>> av2 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=577, atom_name='CB')
        >>> av1.standard_deviation_of_distances(av2)
        10.276833684832678
        """
        return chisurf.structure.label.functions.standard_deviation_of_distances(self, av)

    def mean_fret_distance(
            self,
            av: LabelDistribution,
            forster_radius: float
    ):
        """Calculate the FRET-averaged mean distance to the second accessible volume

        :param av: Accessible volume
        :return:

        Examples
        --------
        >>> import scikit_fluorescence as skf
        >>> import scikit_fluorescence.modeling
        >>> import scikit_fluorescence.io
        >>> import scikit_fluorescence.data
        >>> atoms = skf.data.protein_coordinates_hgbp1()
        >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=18, atom_name='CB')
        >>> av2 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=577, atom_name='CB')
        >>> av1.standard_deviation_of_distances(av2)
        >>> av1.mean_fret_distance(av2, forster_radius=52.0)
        57.778193205194526
        """
        return chisurf.structure.label.functions.mean_fret_distance(
            self,
            av,
            forster_radius
        )

    def get_distance_distribution(
            self,
            av: LabelDistribution,
            normalize: bool = False,
            **kwargs
    ) -> typing.Tuple[
        np.ndarray,
        np.ndarray
    ]:
        """Calculate a histogram of the distance distribution

        This method computes a histogram of the inter-label distance distribution
        between two accessible volumes. This function returns the distance / histogram
        axis and corresponding histogram.

        :param av: Accessible volume
        :param kwargs: these parameters are passed to
        :py:meth:`chisurf.structure.label.av.functions.histogram_rda`
         rda_axis (axis used to compute the histogram), rda_min (minimum distance
         value), rda_max (maximum distance value), n_rda_bins (number of histogram
         bins), use_log (if True computes a logarithmic axis), n_samples (number
         of samples taken to compute the histogram).
        :return: Tuple (histogram, histogram axis)

        Examples
        --------
        >>> import scikit_fluorescence as skf
        >>> import scikit_fluorescence.modeling
        >>> import scikit_fluorescence.io
        >>> import scikit_fluorescence.data
        >>> atoms = skf.data.protein_coordinates_hgbp1()
        >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=18, atom_name='CB')
        >>> av2 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=577, atom_name='CB')
        >>> av1.standard_deviation_of_distances(av2)
        >>> av1.mean_fret_distance(av2, forster_radius=52.0)
        >>> y, x = av1.get_distance_distribution(av2)
        """
        return chisurf.structure.label.functions.histogram_rda(
            self, av, normalize=normalize, **kwargs
        )

    @property
    @abc.abstractmethod
    def mean_position(
            self
    ) -> np.ndarray:
        """
        The mean position of the label density (average x, y, z coordinate)
        """
        pass


from __future__ import annotations
from scikit_fluorescence import typing

import math

import numba as nb
import numpy as np

import scikit_fluorescence as skf

DISTANCE_SAMPLES: int = 50000


@nb.jit
def random_distances(
        p1: np.ndarray,
        p2: np.ndarray,
        n_samples: int = DISTANCE_SAMPLES
):
    """

    :param xyzw: a 4-dim vector xyz and the weight of the coordinate
    :param nSamples:
    :return:
    """
    n_p1 = p1.shape[0]
    n_p2 = p2.shape[0]
    distances = np.empty((n_samples, 2), dtype=np.float64)
    for i in range(n_samples):
        i1 = np.random.randint(0, n_p1)
        i2 = np.random.randint(0, n_p2)
        distances[i, 0] = math.sqrt(
            (p1[i1, 0] - p2[i2, 0]) ** 2.0 +
            (p1[i1, 1] - p2[i2, 1]) ** 2.0 +
            (p1[i1, 2] - p2[i2, 2]) ** 2.0
        )
        distances[i, 1] = p1[i1, 3] * p2[i2, 3]
    return distances


@nb.jit(nopython=True)
def density2points(r, nx: int, ny: int, nz: int, dg, density, r0, threshold):
    """Convert a 3D density grid to a point cloud (numba-accelerated)."""
    gdx = np.arange(0, dg * nx, dg)
    gdy = np.arange(0, dg * ny, dg)
    gdz = np.arange(0, dg * nz, dg)
    x0, y0, z0 = r0[0], r0[1], r0[2]
    n = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if density[ix, iy, iz] > threshold:
                    r[n, 0] = gdx[ix] + x0
                    r[n, 1] = gdy[iy] + y0
                    r[n, 2] = gdz[iz] + z0
                    r[n, 3] = density[ix, iy, iz]
                    n += 1
    return n, r


def average_distance(
        dd1: ch.modeling.label.distribution.LabelDistribution,
        dd2: skf.modeling.label.distribution.LabelDistribution,
        n_samples: int = DISTANCE_SAMPLES
) -> float:
    """Calculate the mean distance between two accessible volumes

    >>> import scikit_fluorescence as skf
    >>> import scikit_fluorescence.modeling
    >>> import scikit_fluorescence.io
    >>> import scikit_fluorescence.data
    >>> atoms = skf.data.protein_coordinates_hgbp1()
    >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=72, atom_name='CB')
    >>> av2 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=134, atom_name='CB')
    >>> skf.modeling.label.distribution.average_distance(av1, av2)
    51.565737926093334
    """
    d = random_distances(
        dd1.points,
        dd2.points,
        n_samples
    )
    return np.dot(d[:, 0], d[:, 1]) / d[:, 1].sum()


def standard_deviation_of_distances(
        av1: skf.modeling.label.distribution.LabelDistribution,
        av2: skf.modeling.label.distribution.LabelDistribution,
        n_samples: int = DISTANCE_SAMPLES
):
    """Standard deviation of the distance distribution

    >>> import scikit_fluorescence as skf
    >>> import scikit_fluorescence.modeling
    >>> import scikit_fluorescence.io
    >>> import scikit_fluorescence.data
    >>> atoms = skf.data.protein_coordinates_hgbp1()
    >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=72, atom_name='CB')
    >>> av2 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=134, atom_name='CB')
    >>> skf.modeling.label.standard_deviation_of_distances(av1, av2)
    5.5771415636745045
    """
    d = random_distances(av1.points, av2.points, n_samples)
    s = np.dot(d[:, 0]**2.0, d[:, 1]) / np.sum(d[:, 1])
    mean_distance = np.dot(d[:, 0], d[:, 1]) / np.sum(d[:, 1])
    v = s - mean_distance**2.0
    return np.sqrt(v)


def mean_fret_distance(
        av1: skf.modeling.label.distribution.LabelDistribution,
        av2: skf.modeling.label.distribution.LabelDistribution,
        forster_radius: float = 52.0,
        n_samples: int = DISTANCE_SAMPLES
) -> float:
    """Calculate mean FRET efficiency expressed as distance

    averaged (PDA/Intensity) distance between two accessible volumes

    >>> import scikit_fluorescence as skf
    >>> import scikit_fluorescence.modeling
    >>> import scikit_fluorescence.io
    >>> import scikit_fluorescence.data
    >>> atoms = skf.data.protein_coordinates_hgbp1()
    >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=72, atom_name='CB')
    >>> av2 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=134, atom_name='CB')
    >>> skf.modeling.label.mean_fret_distance(av1, av2)
    51.46727361728762
    """
    d = random_distances(av1.points, av2.points, n_samples)
    r = d[:, 0]
    w = d[:, 1]
    e = (1. / (1. + (r / forster_radius) ** 6.0))
    mean_fret = np.dot(w, e) / w.sum()
    return (1./mean_fret - 1.) ** (1./6.) * forster_radius


def distance_between_mean_positions(
        av1: skf.modeling.label.distribution.LabelDistribution,
        av2: skf.modeling.label.distribution.LabelDistribution
) -> float:
    """Calculate the distance between the mean position of two accessible volumes

    >>> import scikit_fluorescence as skf
    >>> import scikit_fluorescence.modeling
    >>> import scikit_fluorescence.io
    >>> import scikit_fluorescence.data
    >>> atoms = skf.data.protein_coordinates_hgbp1()
    >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=72, atom_name='CB')
    >>> av2 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=134, atom_name='CB')
    >>> skf.modeling.label.distance_between_mean_positions(av1, av2)
    49.79157
    """
    return np.sqrt(((av1.mean_position - av2.mean_position) ** 2).sum())


def histogram_rda(
        dd1: skf.modeling.label.distribution.LabelDistribution,
        dd2: skf.modeling.label.distribution.LabelDistribution,
        rda_axis: np.ndarray = None,
        rda_min: float = 1,
        rda_max: float = 200,
        n_rda_bins = 100,
        use_log: bool = False,
        n_samples: int = DISTANCE_SAMPLES,
        normalize: bool = False
) -> typing.Tuple[
    np.ndarray,
    np.ndarray
]:
    """Calculate a histogram of the distance distribution

    This function computes a histogram of the inter-label distance distribution
    between two accessible volumes. This function returns the distance / histogram
    axis and corresponding histogram.

    :param dd1: Accessible volume
    :param dd2: Accessible volume
    :param rda_axis: if provided this axis will be used to compute a histogram
    :param rda_min: minimum value of the axis used to compute the histogram
    :param rda_max: maximum value of the axis used to compute the histogram
    :param n_rda_bins: number of axis bins
    :param use_log: if True (default is True) the axis is logarithmic
    :param n_samples: number of samples take to compute the histogram
    :param normalize: if True (default is False) the area is noramlized to one.
    :return: Tuple (histogram, histogram axis)

    Examples
    --------
    >>> import scikit_fluorescence as skf
    >>> import scikit_fluorescence.modeling
    >>> import scikit_fluorescence.io
    >>> import scikit_fluorescence.data
    >>> atoms = skf.data.protein_coordinates_hgbp1()
    >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=72, atom_name='CB')
    >>> av2 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=134, atom_name='CB')
    >>> p_rda, rda = av1.get_distance_distribution(av2)

    """
    if rda_axis is None:
        if use_log:
            rda_axis = np.logspace(
                start=np.log10(rda_min),
                stop=np.log10(rda_max),
                num=n_rda_bins,
                dtype=np.float64
            )
        else:
            rda_axis = np.linspace(
                start=rda_min,
                stop=rda_max,
                num=n_rda_bins,
                dtype=np.float64
            )
    ds = random_distances(dd1.points, dd2.points, n_samples)
    r = ds[:, 0]
    w = ds[:, 1]
    p = np.histogram(r, bins=rda_axis, weights=w)[0]
    if normalize:
        p /= np.sum(p)
    return p, rda_axis


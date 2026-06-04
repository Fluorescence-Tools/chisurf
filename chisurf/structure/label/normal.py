from __future__ import annotations

import chisurf.structure.label.functions
from scikit_fluorescence import typing

import scipy.stats
import numpy as np
from . import distribution


class DyeDistributionNormal(distribution.LabelDistribution):

    _density_shape: typing.Tuple[int, int, int]
    _covariance: np.ndarray
    _density_threshold: float
    _points: np.ndarray

    def __init__(
            self,
            density_shape: typing.Tuple[int, int, int],
            origin: np.ndarray,
            covariance: np.ndarray,
            simulation_type: str = 'Gauss',
            simulation_grid_resolution: float = 0.5,
            position_name: str = "",
            density_threshold: float = 0.95
    ):
        """

        Parameters
        ----------
        density_shape
        origin
        covariance
        simulation_type
        simulation_grid_resolution
        position_name
        density_threshold : float
            By default 95% percent of the points in the density are used. For
            more points increase the number
        """
        super(DyeDistributionNormal, self).__init__(
            simulation_type=simulation_type,
            origin=origin,
            simulation_grid_resolution=simulation_grid_resolution,
            position_name=position_name
        )
        self._density_threshold = density_threshold
        self._density = np.empty(density_shape, dtype=np.float64)
        self._density_shape = density_shape
        self._covariance = covariance
        self.__update_density()

    def __update_density(self):
        """Fill the density volume with a 3D Gaussian centered at the origin."""
        nx, ny, nz = self._density_shape
        xmin = nx // 2 * self.simulation_grid_resolution
        ymin = ny // 2 * self.simulation_grid_resolution
        zmin = nz // 2 * self.simulation_grid_resolution
        X = np.linspace(-xmin, xmin, nx)
        Y = np.linspace(-ymin, ymin, ny)
        Z = np.linspace(-zmin, zmin, nz)
        xv, yv, zv = np.meshgrid(X, Y, Z)
        pos = np.empty((nz, ny, nx, 3))
        pos[:, :, :, 0] = xv
        pos[:, :, :, 1] = yv
        pos[:, :, :, 2] = zv
        normal = scipy.stats.multivariate_normal(
            mean=np.array([0.0, 0.0, 0.0], dtype=np.float),
            cov=self._covariance
        )
        self._density = normal.pdf(pos)
        density_threshold = (1. - self._density_threshold) * np.max(self._density)
        r = np.empty((nx * ny * nz, 4), dtype=np.float)
        n, points = chisurf.structure.label.functions.density2points(
            r=r,
            nx=nx, ny=ny, nz=nz, dg=self.simulation_grid_resolution,
            density=self._density, r0=self.origin,
            threshold=density_threshold
        )
        self._points = points[:n]

    @property
    def mean_position(self):
        """Mean position of the distribution (the distribution's origin)."""
        return self.origin

    @property
    def density(self):
        """3D density array of the distribution."""
        return self._density

    @property
    def points(self):
        """Sampled point cloud representing the distribution."""
        return self._points

    def average_distance(self, v):
        """Return the average pairwise distance to another distribution *v*."""
        return chisurf.structure.label.functions.average_distance(
            dd1=self,
            dd2=v
        )

    def get_distance_between_mean_positions(
            self,
            v: distribution.LabelDistribution
    ):
        """Return the Euclidean distance between the two mean positions."""
        mp1 = self.mean_position
        mp2 = v.mean_position
        return np.linalg.norm(mp1-mp2, ord=2)

    def get_distance_distribution(
            self,
            av: distribution.LabelDistribution,
            normalize: bool = False,
            **kwargs
    ) -> typing.Tuple[
        np.ndarray,
        np.ndarray
    ]:
        """Return the donor-acceptor distance distribution ``(axis, hist)``."""
        return chisurf.structure.label.functions.histogram_rda(
            self, av, normalize=normalize, **kwargs
        )

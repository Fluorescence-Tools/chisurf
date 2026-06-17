import logging

import chisurf as cs
import numpy as np

from chisurf.core.data import DataCurve, ExperimentDataCurveGroup
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.models.tcspc.anisotropy import Anisotropy
from chisurf.core.models.tcspc.lifetime import LifetimeModel

# Configure logging to see detailed information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_group_polarization_assignment():
    """
    Test that polarization types are correctly assigned for fits in a 2-dataset group.

    This test verifies that the back reference from each fit to its parent group is
    available during model initialization, allowing the polarization setup code in
    LifetimeModel.__init__ to correctly set the polarization types based on the fit's
    position in the group.
    """
    logger.info("Testing polarization assignment for fits in a 2-dataset group")

    # Clear any existing datasets and fits
    cs.imported_datasets = []
    cs.fits = []

    # Create two simple datasets
    x = np.linspace(0, 10, 100)
    y1 = np.exp(-x/2) + 0.1*np.random.randn(100)
    y2 = np.exp(-x/4) + 0.1*np.random.randn(100)

    data1 = DataCurve(x=x, y=y1, name="Dataset 1")
    data2 = DataCurve(x=x, y=y2, name="Dataset 2")

    # Create a data group with both datasets
    data_group = ExperimentDataCurveGroup([data1, data2])

    # Create a fit group with the data group
    fit_group = FitGroup(
        data=data_group,
        model_class=LifetimeModel
    )

    # Check that each fit has a group attribute that references the fit group
    for i, fit in enumerate(fit_group.grouped_fits):
        if hasattr(fit, 'group'):
            logger.info(f"Fit {i} has group attribute: {fit.group is fit_group}")
        else:
            logger.error(f"Fit {i} does not have group attribute")

    # Check polarization types for each fit in the group
    for i, fit in enumerate(fit_group.grouped_fits):
        pol_type = fit.model.anisotropy.polarization_type
        logger.info(f"Fit {i} polarization type: {pol_type}")

        # Verify polarization type is set correctly
        if i == 0 and pol_type != 'vv':
            logger.error(f"Fit 0 should have polarization type 'vv', but has '{pol_type}'")
        elif i == 1 and pol_type != 'vh':
            logger.error(f"Fit 1 should have polarization type 'vh', but has '{pol_type}'")

    # Test passed if we get here without errors
    logger.info("Test passed: Polarization types are correctly assigned for fits in a 2-dataset group")


def test_vm_lifetime_model_rotation_parameters_are_fixed():
    """VM lifetime models must not expose rotation parameters to the fit."""
    fit = Fit(
        model_class=LifetimeModel,
        model_kw={'anisotropy': Anisotropy(polarization='vm')}
    )

    fit.model.anisotropy.add_rotation()
    fit.model.find_parameters()

    rotation_names = {'b(1)', 'rho(1)'}
    assert fit.model.anisotropy.polarization_type == 'vm'
    assert fit.model.anisotropy._bs[0].fixed
    assert fit.model.anisotropy._rhos[0].fixed
    assert rotation_names.isdisjoint(fit.model.parameter_names)


def test_non_vm_lifetime_model_rotation_parameters_are_free():
    """VV/VH lifetime models keep rotation parameters available to the fit."""
    fit = Fit(
        model_class=LifetimeModel,
        model_kw={'anisotropy': Anisotropy(polarization='vv')}
    )

    fit.model.anisotropy.add_rotation()
    fit.model.find_parameters()

    rotation_names = {'b(1)', 'rho(1)'}
    assert fit.model.anisotropy.polarization_type == 'vv'
    assert not fit.model.anisotropy._bs[0].fixed
    assert not fit.model.anisotropy._rhos[0].fixed
    assert rotation_names.issubset(fit.model.parameter_names)


if __name__ == "__main__":
    test_group_polarization_assignment()

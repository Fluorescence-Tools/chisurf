"""GMM Settings Dialog for BrickMicWizard.

Provides a dialog for adjusting Gaussian Mixture Model (GMM) settings used in the
BrickMicWizard for histogram fitting. The form is declared once as a
``dataspec`` ``ModelView`` and rendered by :class:`AutoForm`, instead of
hand-building one widget per field (PRD-40).
"""
from types import SimpleNamespace

from qtpy import QtWidgets

from chisurf.core import dataspec as ds
from chisurf.gui.autoform import AutoForm

DEFAULT_GMM_SETTINGS = {
    "covariance_type": "full",
    "random_state": 42,
    "max_iter": 300,
    "n_init": 10,
    "tol": 1e-3,
    "max_components": 10,
    "reg_covar": 1e-6,
}


def _gmm_view_spec() -> ds.ModelView:
    """Return the declarative description of the GMM settings form."""
    return ds.ModelView(sections=(
        ds.ChoiceSection(target="settings", attr="covariance_type",
                         label="Covariance Type:",
                         options=("full", "tied", "diag", "spherical")),
        ds.ValueSection(target="settings", attr="random_state", kind="int",
                        label="Random State:", minimum=0, maximum=1000),
        ds.ValueSection(target="settings", attr="max_iter", kind="int",
                        label="Max Iterations:", minimum=10, maximum=1000),
        ds.ValueSection(target="settings", attr="n_init", kind="int",
                        label="Number of Initializations:", minimum=1, maximum=20),
        ds.ValueSection(target="settings", attr="tol", kind="float",
                        label="Convergence Threshold:", minimum=1e-6, maximum=1e-1,
                        decimals=6, step=1e-4),
        ds.ValueSection(target="settings", attr="max_components", kind="int",
                        label="Max Components for Auto:", minimum=2, maximum=20),
        ds.ValueSection(target="settings", attr="reg_covar", kind="float",
                        label="Covariance Regularization:", minimum=1e-10, maximum=1e-1,
                        decimals=10, step=1e-7),
    ))


class _GMMDataset:
    """Bound object adapting the GMM settings dict to :class:`AutoForm`.

    Holds the editable settings on a ``settings`` namespace (so sections can
    bind via ``target="settings"``) and exposes the declarative form via
    :meth:`view_spec`. Edits commit to this namespace, not the caller's dict, so
    Cancel is honoured simply by ignoring :meth:`GMMSettingsDialog.get_settings`.
    """

    def __init__(self, gmm_settings):
        self.settings = SimpleNamespace(**{**DEFAULT_GMM_SETTINGS, **(gmm_settings or {})})

    def view_spec(self) -> ds.ModelView:
        return _gmm_view_spec()


class GMMSettingsDialog(QtWidgets.QDialog):
    """Dialog for adjusting GMM (Gaussian Mixture Model) settings."""

    def __init__(self, parent=None, gmm_settings=None):
        super().__init__(parent)
        self.setWindowTitle("GMM Settings")
        self.resize(400, 300)

        self._dataset = _GMMDataset(gmm_settings)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(AutoForm(self._dataset))

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_settings(self):
        """Return the current GMM settings from the dialog."""
        s = self._dataset.settings
        return {k: getattr(s, k) for k in DEFAULT_GMM_SETTINGS}

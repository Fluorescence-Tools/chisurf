from __future__ import annotations

import chisurf.core.models.parse as parse_module


class ParseFCSModel(parse_module.ParseModel):
    """Parse model for FCS correlation functions.

    Extends the generic ParseModel with FCS-specific parameter handling.
    """

    def __init__(self, fit, **kwargs):
        """Initialize the FCS parse model.

        Parameters
        ----------
        fit : chisurf.core.fitting.fit.Fit
            Fit object this model belongs to.
        """
        super().__init__(fit, **kwargs)
        # FCS-specific initial parameter values can be set here if needed

    def update_model(self, **kwargs):
        """Update the FCS correlation model."""
        super().update_model(**kwargs)
        # FCS-specific post-processing can be added here if needed
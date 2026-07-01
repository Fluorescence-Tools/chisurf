"""GUI layer for the Batch-Analysis plugin (AutoForm host + view-model)."""

from .tool import BatchAnalysisWidget, BatchProcessingWizard
from .view_model import BatchViewModel

__all__ = ["BatchAnalysisWidget", "BatchProcessingWizard", "BatchViewModel"]

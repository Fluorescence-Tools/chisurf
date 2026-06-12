"""Pure computation for {{ cookiecutter.plugin_display_name }}.

No Qt, no ZMQ, no global state. Accept and return API DTOs.
"""

from __future__ import annotations

from ..api.models import PluginSettings


def compute_example(settings: PluginSettings) -> dict:
    """Example pure computation function.

    Parameters
    ----------
    settings : PluginSettings
        Input settings.

    Returns
    -------
    dict
        JSON-serializable result.

    """
    return {"result": settings.example_number * 2}

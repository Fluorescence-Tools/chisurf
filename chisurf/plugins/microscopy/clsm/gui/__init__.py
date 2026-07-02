"""GUI layer for the CLSM plugin.

The interactive tool is assembled by :class:`AutoForm` from
``clsm.view.json``: the acquisition / brush / decay settings render as standard
AutoForm fields, while the dynamic control bar, the image-brush canvas and the
ROI list are registered ``custom`` sections (imaging widgets that AutoForm has
no built-in template for). The decay and FRC plots are declarative ``plot``
sections driven by methods on :class:`ClsmViewModel`.

Importing this package pulls in Qt; the headless ``core``/``api`` layers do not
depend on it.
"""

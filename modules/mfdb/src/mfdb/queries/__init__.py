"""Per-concern query mixins that compose the ``MFDatabase`` facade.

Each module here defines one cohesive mixin (artifacts/operations/edges, samples,
setups/calibrations, parameters, …) extracted from the former ``repository.py``
god-class. ``MFDatabase`` inherits from them, so the public method surface is
unchanged — the methods simply live in cohesive files instead of one 7k-line
class. Mixins reference ``self`` (``self.conn``, ``self.dao``, ``self.lineage``,
``self._transaction`` and sibling methods) and the shared helpers in
``mfdb._sqlutil``; they never import ``repository`` at module load, so no import
cycle forms.
"""

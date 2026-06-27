"""Stage-2 vocabulary alignment tests.

Pins three things:
  1. ``canonical()`` is a collision-free separator-normal form (pure syntax).
  2. ``ActionRegistry.resolve_name`` resolves any separator spelling — including
     names with underscores *inside* a verb, which the old two-shot ``replace``
     corrupted.
  3. The MFDB dictionary is the single source of truth for the action vocabulary:
     every registered ``@action`` name appears in the ``_mfdb_event_log.action_type``
     enumeration, and the coarse action->operation_type mapping is read from the
     dictionary's ``_item_enumeration.detail``, not from Python.

Headless: no Qt, no DB.
"""

from __future__ import annotations

import chisurf as cs
import chisurf.core.actions  # noqa: F401  (triggers @action registration)
from chisurf.core.actions import canonical, get_action_catalog
from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary


def test_canonical_collapses_separators():
    assert canonical("dataset.add") == canonical("dataset_add")
    assert canonical("fit.run.finish") == canonical("fit_run_finish")
    # underscores inside a verb must survive — only '.' is a separator
    assert canonical("dataset.restore_global_fit") == "dataset_restore_global_fit"
    assert canonical("dataset.restore_global_fit") == canonical("dataset_restore_global_fit")


def test_canonical_is_idempotent():
    for name in ["dataset.add", "dataset_add", "model.set_correction"]:
        assert canonical(canonical(name)) == canonical(name)


def test_resolve_name_handles_underscored_and_multiword():
    reg = cs.action_registry
    # underscored spelling resolves to the registered dotted key
    assert reg.resolve_name("dataset_add") == "dataset.add"
    assert reg.resolve_name("fit_run_finish") == "fit.run.finish"
    # the multi-underscore verb case the old replace() could not handle
    assert reg.resolve_name("dataset_restore_global_fit") == "dataset.restore_global_fit"
    assert reg.resolve_name("model_add_component") == "model.add_component"
    # already-canonical names pass through unchanged
    assert reg.resolve_name("dataset.add") == "dataset.add"
    # unknown names are returned as-is
    assert reg.resolve_name("nonexistent.action") == "nonexistent.action"


def _action_enum():
    dic = MmcifDictionary.load_bundled()
    return dic, dic.get_enumerations("_mfdb_event_log.action_type")


def test_dictionary_is_single_source_of_action_vocabulary():
    """Every registered @action must be declared in the dictionary enumeration."""
    _dic, enum = _action_enum()
    enum_norm = {canonical(v) for v in enum}
    registered = {c["name"] for c in get_action_catalog()}
    missing = {n for n in registered if canonical(n) not in enum_norm}
    assert not missing, f"actions missing from _mfdb_event_log.action_type enum: {sorted(missing)}"


def test_coarse_operation_mapping_comes_from_dictionary():
    """The action->operation_type map is the enumeration detail, not Python."""
    dic, _enum = _action_enum()
    item = dic.get_item("_mfdb_event_log.action_type")
    assert item is not None
    assert item.enum_details.get("dataset.add") == "measurement_import"
    assert item.enum_details.get("fit.run.finish") == "local_fit"
    assert item.enum_details.get("project.save") == "project_archive"
    # fine-grained actions carry the mmCIF null marker, i.e. no operation
    assert item.enum_details.get("parameter.value") in (".", "", "?")

from __future__ import annotations

import copy
import uuid

import pytest

from chisurf.core.base import Base, find_objects


class _ConcreteBase(Base):
    """Minimal concrete Base subclass for testing."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ── Base identity tests ─────────────────────────────────────────────────


def test_base_uid_auto_assigned():
    b = _ConcreteBase()
    assert b.unique_identifier is not None
    assert isinstance(b.unique_identifier, str)
    assert len(b.unique_identifier) > 0


def test_base_uid_preserved_on_init():
    uid = str(uuid.uuid4())
    b = _ConcreteBase(unique_identifier=uid)
    assert b.unique_identifier == uid


def test_base_eq_by_uid():
    uid = str(uuid.uuid4())
    a = _ConcreteBase(unique_identifier=uid)
    c = _ConcreteBase(unique_identifier=uid)
    assert a == c
    assert not (a != c)


def test_base_ne_by_uid():
    d = _ConcreteBase()
    e = _ConcreteBase()
    assert d != e
    assert not (d == e)


def test_base_hash_matches_eq():
    uid = str(uuid.uuid4())
    a = _ConcreteBase(unique_identifier=uid)
    c = _ConcreteBase(unique_identifier=uid)
    assert hash(a) == hash(c)
    assert len({a, c}) == 1


def test_base_eq_with_non_base():
    b = _ConcreteBase()
    assert (b == "foo") is False
    assert (b != "foo") is True


def test_base_eq_with_none():
    b = _ConcreteBase()
    assert (b is not None)
    assert (b == None) is False


def test_base_set_dedup_by_uid():
    uid = str(uuid.uuid4())
    a = _ConcreteBase(unique_identifier=uid)
    c = _ConcreteBase(unique_identifier=uid)
    d = _ConcreteBase()
    s = {a, c, d}
    assert len(s) == 2


# ── Base copy / deepcopy tests ─────────────────────────────────────────


def test_copy_preserves_uid():
    b = _ConcreteBase(name="original")
    orig_uid = b.unique_identifier
    c = copy.copy(b)
    assert c.unique_identifier == orig_uid


def test_deepcopy_preserves_uid():
    b = _ConcreteBase(name="original")
    orig_uid = b.unique_identifier
    c = copy.deepcopy(b)
    assert c.unique_identifier == orig_uid


def test_copy_independent_state():
    b = _ConcreteBase(name="original")
    c = copy.copy(b)
    c.name = "copy"
    assert b.name == "original"
    assert c.name == "copy"


def test_deepcopy_independent_state():
    b = _ConcreteBase(name="original")
    c = copy.deepcopy(b)
    c.name = "deepcopy"
    assert b.name == "original"
    assert c.name == "deepcopy"


def test_copy_hash_equals_original():
    b = _ConcreteBase()
    c = copy.copy(b)
    assert hash(b) == hash(c)
    assert b == c


def test_deepcopy_hash_equals_original():
    b = _ConcreteBase()
    c = copy.deepcopy(b)
    assert hash(b) == hash(c)
    assert b == c


# ── find_objects dedup tests ───────────────────────────────────────────


def test_find_objects_uid_dedup():
    uid = str(uuid.uuid4())
    a = _ConcreteBase(unique_identifier=uid)
    c = _ConcreteBase(unique_identifier=uid)
    result = find_objects([a, c], _ConcreteBase, remove_doublets=True)
    assert len(result) == 1


def test_find_objects_no_dedup_on_different_uids():
    a = _ConcreteBase()
    c = _ConcreteBase()
    result = find_objects([a, c], _ConcreteBase, remove_doublets=True)
    assert len(result) == 2


def test_find_objects_keep_doublets_when_disabled():
    uid = str(uuid.uuid4())
    a = _ConcreteBase(unique_identifier=uid)
    c = _ConcreteBase(unique_identifier=uid)
    result = find_objects([a, c], _ConcreteBase, remove_doublets=False)
    assert len(result) == 2


def test_find_objects_id_fallback():
    """Non-Base objects dedup by identity (default object.__hash__/__eq__)."""
    a = (1, 2)
    c = (1, 2)
    result = find_objects([a, c], tuple, remove_doublets=True)
    # a and c are different tuple instances with same content;
    # tuple __eq__ is value-based so the set dedup will collapse them
    assert len(result) == 1


# ── Project format version tests ────────────────────────────────────────


from chisurf.core.project.project import Project


def test_project_default_version_is_4():
    p = Project()
    assert p.project_format_version == 4


def test_project_to_dict_uses_version_4():
    p = Project()
    d = p.to_dict()
    assert d["project_format_version"] == 4


def test_project_load_v3_raises():
    data = {
        "project_format_version": 3,
        "meta": {},
    }
    with pytest.raises(ValueError, match="v4"):
        Project.from_dict(data)


def test_project_load_v4_ok():
    uid = str(uuid.uuid4())
    data = {
        "project_format_version": 4,
        "meta": {"name": "test"},
        "datasets": {},
        "experiments": {},
        "fits": [],
        "links": [],
        "ui": {},
    }
    p = Project.from_dict(data)
    assert p.project_format_version == 4
    assert p.name == "test"


# ── find_by_uuid tests ─────────────────────────────────────────────────


def test_find_by_uuid_returns_instance():
    b = _ConcreteBase()
    assert Base.find_by_uuid(b.unique_identifier) is b


def test_find_by_uuid_returns_none_for_unknown():
    assert Base.find_by_uuid(str(uuid.uuid4())) is None


def test_find_by_uuid_invalid_input():
    assert Base.find_by_uuid("") is None
    assert Base.find_by_uuid(None) is None


def test_find_by_uuid_module_level():
    b = _ConcreteBase()
    from chisurf.core.base import find_by_uuid
    assert find_by_uuid(b.unique_identifier) is b


def test_all_uuids_includes_live():
    b = _ConcreteBase()
    uuids = Base.all_uuids()
    assert b.unique_identifier in uuids


def test_find_by_uuid_after_copy():
    b = _ConcreteBase()
    c = copy.copy(b)
    # Both have the same UID; find_by_uuid returns the most recently
    # indexed one (the copy overwrites the original).
    found = Base.find_by_uuid(b.unique_identifier)
    assert found is not None
    assert found is c


# ── Parameter identity tests (skip if chinet unavailable) ──────────────


def test_parameter_eq_by_uid():
    chinet = pytest.importorskip("chinet")
    from chisurf.core.parameter import Parameter
    p1 = Parameter(value=1.0)
    p2 = Parameter(value=1.0)
    assert p1 != p2


def test_parameter_eq_same_uid():
    chinet = pytest.importorskip("chinet")
    from chisurf.core.parameter import Parameter
    uid = str(uuid.uuid4())
    p1 = Parameter(value=1.0, unique_identifier=uid)
    p2 = Parameter(value=1.0, unique_identifier=uid)
    assert p1 == p2
    assert hash(p1) == hash(p2)


def test_parameter_set_dedup_by_uid():
    chinet = pytest.importorskip("chinet")
    from chisurf.core.parameter import Parameter
    uid = str(uuid.uuid4())
    p1 = Parameter(value=1.0, unique_identifier=uid)
    p2 = Parameter(value=1.0, unique_identifier=uid)
    p3 = Parameter(value=2.0)
    assert len({p1, p2, p3}) == 2


def test_parameter_different_values_different_uids():
    chinet = pytest.importorskip("chinet")
    from chisurf.core.parameter import Parameter
    p1 = Parameter(value=1.0)
    p2 = Parameter(value=2.0)
    assert p1 != p2
    assert hash(p1) != hash(p2)


# ── Fit state round-trip by UID (requires full fitting stack) ──────────


def test_fit_state_roundtrip_uid_keys():
    pytest.importorskip("chinet")
    from chisurf.core.fitting.parameter import FittingParameterGroup, FittingParameter
    from chisurf.core.project.fit_state import _model_to_state
    pg = FittingParameterGroup()
    pg.tau = FittingParameter(name="tau", value=3.5)
    pg.amp = FittingParameter(name="amp", value=0.5)
    pg.find_parameters()
    state = _model_to_state(pg)
    assert "parameters" in state
    for key, ps in state["parameters"].items():
        assert ps["uid"] == key
        assert "name" in ps
        assert "value" in ps


def test_fit_state_uid_persistence():
    pytest.importorskip("chinet")
    from chisurf.core.fitting.parameter import FittingParameterGroup, FittingParameter
    from chisurf.core.project.fit_state import _model_to_state
    pg = FittingParameterGroup()
    pg.tau = FittingParameter(name="tau", value=3.5)
    pg.find_parameters()
    uid_before = str(pg.tau.unique_identifier)
    state = _model_to_state(pg)
    stored = state["parameters"][uid_before]
    assert stored["uid"] == uid_before


def test_fit_state_intra_fit_link_by_uid():
    pytest.importorskip("chinet")
    from chisurf.core.fitting.parameter import FittingParameterGroup, FittingParameter
    from chisurf.core.project.fit_state import _model_to_state
    master = FittingParameter(name="master", value=1.0)
    slave = FittingParameter(name="slave", value=0.0, link=master)
    pg = FittingParameterGroup()
    pg.__dict__['master'] = master
    pg.__dict__['slave'] = slave
    pg.find_parameters()
    state = _model_to_state(pg)
    uid_slave = str(slave.unique_identifier)
    uid_master = str(master.unique_identifier)
    assert state["parameters"][uid_slave]["link_target"] == uid_master

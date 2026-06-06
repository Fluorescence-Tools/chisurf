from __future__ import annotations

import json
import pytest

from chisurf.server.dto import (
    DatasetSummary,
    DatasetDetail,
    FitSummary,
    FitDetail,
    ParameterDTO,
    SetupDTO,
    ProjectInfoDTO,
    ActionResultDTO,
    dto_to_json,
    to_json_safe,
)


class TestDatasetSummary:
    def test_creation(self):
        d = DatasetSummary(
            index=0,
            uid="abc-123",
            name="mydata.ptu",
            type="DataCurve",
            experiment="TCSPC",
        )
        assert d.index == 0
        assert d.uid == "abc-123"
        assert d.name == "mydata.ptu"
        assert d.type == "DataCurve"
        assert d.experiment == "TCSPC"

    def test_to_dict(self):
        d = DatasetSummary(
            index=0,
            uid="abc-123",
            name="mydata.ptu",
            type="DataCurve",
            experiment="TCSPC",
        )
        serialized = d.to_dict()
        assert serialized["index"] == 0
        assert serialized["uid"] == "abc-123"
        assert serialized["name"] == "mydata.ptu"
        assert serialized["type"] == "DataCurve"
        assert serialized["experiment"] == "TCSPC"

    def test_from_dict(self):
        d = DatasetSummary.from_dict({
            "index": 1,
            "uid": "uid-99",
            "name": "test.dat",
            "type": "DataGroup",
            "experiment": "FCS",
        })
        assert d.index == 1
        assert d.name == "test.dat"

    def test_json_serializable(self):
        d = DatasetSummary(
            index=0,
            uid="abc-123",
            name="mydata.ptu",
            type="DataCurve",
            experiment="TCSPC",
        )
        text = json.dumps(d.to_dict())
        assert isinstance(text, str)
        parsed = json.loads(text)
        assert parsed["name"] == "mydata.ptu"


class TestParameterDTO:
    def test_creation(self):
        p = ParameterDTO(
            name="tau1",
            value=3.8,
            fixed=False,
            bounds_on=True,
            bounds=(0.0, 100.0),
            linked_to=None,
            error_estimate=0.1,
        )
        assert p.name == "tau1"
        assert p.value == 3.8
        assert p.fixed is False
        assert p.bounds_on is True
        assert p.bounds == (0.0, 100.0)
        assert p.linked_to is None

    def test_to_dict(self):
        p = ParameterDTO(
            name="tau1",
            value=3.8,
            fixed=False,
            bounds_on=True,
            bounds=(0.0, 100.0),
            linked_to=None,
            error_estimate=0.1,
        )
        d = p.to_dict()
        assert d["name"] == "tau1"
        assert d["value"] == 3.8
        assert d["bounds"] == (0.0, 100.0)
        assert d["linked_to"] is None

    def test_from_dict(self):
        p = ParameterDTO.from_dict({
            "name": "tau2",
            "value": 1.5,
            "fixed": True,
            "bounds_on": False,
            "bounds": (0.0, 10.0),
            "linked_to": None,
            "error_estimate": 0.05,
        })
        assert p.name == "tau2"
        assert p.value == 1.5
        assert p.fixed is True
        assert p.bounds_on is False

    def test_json_serializable(self):
        p = ParameterDTO(
            name="tau1",
            value=3.8,
            fixed=False,
            bounds_on=True,
            bounds=(0.0, 100.0),
            linked_to=None,
            error_estimate=0.1,
        )
        text = json.dumps(p.to_dict())
        assert isinstance(text, str)
        parsed = json.loads(text)
        assert parsed["name"] == "tau1"
        assert parsed["bounds"] == [0.0, 100.0]


class TestFitSummary:
    def test_creation(self):
        f = FitSummary(
            index=0,
            uid="fit-1",
            name="Fit 1",
            type="FitGroup",
            chi2=1.23,
            dataset_uid="ds-1",
            dataset_name="sample.ptu",
            model_name="LifetimeModel",
            parameter_count=12,
        )
        assert f.index == 0
        assert f.chi2 == 1.23
        assert f.parameter_count == 12

    def test_to_dict(self):
        f = FitSummary(
            index=0,
            uid="fit-1",
            name="Fit 1",
            type="FitGroup",
            chi2=1.23,
            dataset_uid="ds-1",
            dataset_name="sample.ptu",
            model_name="LifetimeModel",
            parameter_count=12,
        )
        d = f.to_dict()
        assert d["uid"] == "fit-1"
        assert d["chi2"] == 1.23
        assert d["dataset_name"] == "sample.ptu"
        assert d["model_name"] == "LifetimeModel"
        assert d["parameter_count"] == 12

    def test_from_dict(self):
        f = FitSummary.from_dict({
            "index": 2,
            "uid": "fit-3",
            "name": "Fit 3",
            "type": "FitGroup",
            "chi2": 0.98,
            "dataset_uid": "ds-3",
            "dataset_name": "test.ptu",
            "model_name": "AnisotropyModel",
            "parameter_count": 8,
        })
        assert f.name == "Fit 3"
        assert f.chi2 == 0.98

    def test_json_serializable(self):
        f = FitSummary(
            index=0,
            uid="fit-1",
            name="Fit 1",
            type="FitGroup",
            chi2=1.23,
            dataset_uid="ds-1",
            dataset_name="sample.ptu",
            model_name="LifetimeModel",
            parameter_count=12,
        )
        text = json.dumps(f.to_dict())
        assert isinstance(text, str)
        parsed = json.loads(text)
        assert parsed["chi2"] == 1.23


class TestProjectInfoDTO:
    def test_creation(self):
        p = ProjectInfoDTO(
            project_path="/path/to/project",
            project_name="MyProject",
            fit_count=3,
            dataset_count=5,
            experiment_names=["TCSPC", "FCS"],
        )
        assert p.project_path == "/path/to/project"
        assert p.fit_count == 3
        assert p.dataset_count == 5

    def test_from_dict(self):
        p = ProjectInfoDTO.from_dict({
            "project_path": "/tmp",
            "project_name": "Test",
            "fit_count": 0,
            "dataset_count": 2,
            "experiment_names": ["FCS"],
        })
        assert p.project_name == "Test"
        assert p.dataset_count == 2
        assert p.experiment_names == ["FCS"]

    def test_json_serializable(self):
        p = ProjectInfoDTO(
            project_path="/path/to/project",
            project_name="MyProject",
            fit_count=3,
            dataset_count=5,
        )
        text = json.dumps(p.to_dict())
        assert isinstance(text, str)
        parsed = json.loads(text)
        assert parsed["project_path"] == "/path/to/project"
        assert parsed["fit_count"] == 3


class TestActionResultDTO:
    def test_success(self):
        r = ActionResultDTO(ok=True, message="done", data={"count": 5})
        assert r.ok is True
        assert r.message == "done"
        assert r.data["count"] == 5

    def test_failure(self):
        r = ActionResultDTO(ok=False, message="not found")
        assert r.ok is False

    def test_to_dict(self):
        r = ActionResultDTO(ok=True, message="done", data={"count": 5})
        d = r.to_dict()
        assert d["ok"] is True
        assert d["message"] == "done"
        assert d["data"] == {"count": 5}

    def test_json_serializable(self):
        r = ActionResultDTO(ok=True, message="done")
        text = json.dumps(r.to_dict())
        assert isinstance(text, str)


class TestSetupDTO:
    def test_creation(self):
        s = SetupDTO(
            name="TCSPC",
            experiment_name="TCSPC",
            properties={"channels": 4096},
        )
        assert s.name == "TCSPC"
        assert s.properties == {"channels": 4096}

    def test_json_serializable(self):
        s = SetupDTO(
            name="TCSPC",
            experiment_name="TCSPC",
            properties={"channels": 4096},
        )
        text = json.dumps(s.to_dict())
        assert isinstance(text, str)


class TestToJsonSafe:
    def test_none(self):
        assert to_json_safe(None) is None

    def test_primitives(self):
        assert to_json_safe(42) == 42
        assert to_json_safe(3.14) == 3.14
        assert to_json_safe("hello") == "hello"
        assert to_json_safe(True) is True

    def test_list(self):
        assert to_json_safe([1, 2, 3]) == [1, 2, 3]

    def test_tuple(self):
        assert to_json_safe((1, 2)) == [1, 2]

    def test_dict(self):
        assert to_json_safe({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_dto_object(self):
        dto = DatasetSummary(
            index=0,
            uid="abc",
            name="data",
            type="DataCurve",
            experiment="TCSPC",
        )
        result = to_json_safe(dto)
        assert result == dto.to_dict()

    def test_unknown_type(self):
        assert to_json_safe(set()) == "set()"


class TestDtoToJson:
    def test_produces_json_string(self):
        d = DatasetSummary(
            index=0,
            uid="abc",
            name="data.ptu",
            type="DataCurve",
            experiment="TCSPC",
        )
        text = dto_to_json(d)
        assert isinstance(text, str)
        parsed = json.loads(text)
        assert parsed["name"] == "data.ptu"
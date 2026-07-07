from __future__ import annotations

import json

import numpy as np
import pytest

from chisurf.core.data import DataCurve
from chisurf.core.fitting.fit import Fit
from chisurf.core.fitting.parameter import FittingParameter
from chisurf.core.mfdb.adapters.chinet import archive_fit_to_mfdb
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.models.model import ModelCurve


class DummyLinearModelForMFDB(ModelCurve):
    """Small concrete model used for MFDB fit archive tests."""

    name = "DummyLinearModelForMFDB"

    def __init__(self, fit: Fit, **kwargs):
        super().__init__(fit, **kwargs)
        self.p0 = FittingParameter(name="p0", value=0.5)
        self.p1 = FittingParameter(name="p1", value=1.5)
        self.find_parameters()

    def update_model(self, **kwargs):
        x = self.fit.data.x
        if x is None:
            x = np.arange(self.fit.data.y.size, dtype=float)
        self.x = x
        self.y = float(self.p0.value) + float(self.p1.value) * x

    def update(self, **kwargs) -> None:
        super().update(**kwargs)


def _make_dummy_fit() -> Fit:
    data = DataCurve(x=np.arange(4, dtype=float), y=np.ones(4))
    return Fit(model_class=DummyLinearModelForMFDB, data=data)


def _json(value: str | None):
    return json.loads(value) if value else None


def test_archive_fit_to_mfdb_does_not_mutate_live_parameter_ports(tmp_path) -> None:
    fit = _make_dummy_fit()
    params = fit.model.parameters_all_dict
    p0_port = params["p0"]._port
    p1_port = params["p1"]._port
    p0_owner = p0_port.get_node()
    p1_owner = p1_port.get_node()

    db_path = tmp_path / "fit_archive_no_mutation.db"
    with MFDatabase(db_path) as db:
        archive_fit_to_mfdb(
            db,
            fit,
            operation_id="op_fit_no_mutation",
            experiment_id="exp_fit_no_mutation",
            fit_id="fit-no-mutation",
        )

    assert p0_port.get_node() is p0_owner
    assert p1_port.get_node() is p1_owner


def test_archive_fit_to_mfdb_stores_fit_state_chinet_session_and_parameters(tmp_path) -> None:
    fit = _make_dummy_fit()
    params = fit.model.parameters_all_dict
    params["p0"].value = 2.0
    params["p0"].bounds = (0.0, 5.0)
    params["p0"].bounds_on = True
    params["p1"].value = -0.5
    params["p1"].fixed = True
    params["p1"].link = params["p0"]

    db_path = tmp_path / "fit_archive.db"
    with MFDatabase(db_path) as db:
        result = archive_fit_to_mfdb(
            db,
            fit,
            operation_id="op_fit",
            experiment_id="exp_fit",
            fit_id="fit-1",
            dataset_id="dataset-1",
        )

        assert result["operation_id"] == "op_fit"
        assert result["fit_state_artifact_id"].startswith("fit_result:")
        assert result["fit_parameter_count"] >= 2
        assert result["fit_dependency_edge_count"] == 1

        operation = db.get_operation("op_fit")
        assert operation["operation_type"] == "local_fit"
        assert _json(operation["metadata_json"])["fit_id"] == "fit-1"

        artifacts = db.list_artifacts(experiment_id="exp_fit")
        artifact_kinds = {row["artifact_kind"] for row in artifacts}
        assert {
            "chinet_session",
            "chinet_node",
            "fit_result",
            "processed_data",
        }.issubset(artifact_kinds)

        op_links = db.conn.execute(
            "SELECT artifact_id, direction, role "
            "FROM mfdb_operation_artifact WHERE operation_id = ?",
            ("op_fit",),
        ).fetchall()
        roles = {row["role"] for row in op_links}
        assert {"chinet_session", "fit_state"}.issubset(roles)
        assert any(
            row["direction"] == "input" and row["artifact_id"] == "dataset-1"
            for row in op_links
        )

        parameters = db.list_parameters(operation_id="op_fit")
        by_name = {row["name"]: row for row in parameters}
        assert "p0" in by_name
        assert "p1" in by_name
        assert by_name["p0"]["value"] == 2.0
        assert by_name["p0"]["lower_bound"] == 0.0
        assert by_name["p0"]["upper_bound"] == 5.0
        assert by_name["p0"]["bounds_on"] == 1
        assert by_name["p1"]["parameter_type"] == "linked"
        assert _json(by_name["p1"]["metadata_json"])["schema_name"] == "chisurf.fit_state.v1"

        edges = db.conn.execute(
            "SELECT source_node_id, target_node_id, relationship_type FROM mfdb_edge "
            "WHERE relationship_type = 'parameter_depends_on' AND operation_id = ?",
            ("op_fit",),
        ).fetchall()
        assert any(edge["relationship_type"] == "parameter_depends_on" for edge in edges)

        run = db.get_analysis_run_full("op_fit")
        assert run is not None
        assert run["analysis_id"] == "op_fit"
        assert len(run["parameters"]) >= 2


def test_archive_fit_to_mfdb_rolls_back_after_fit_state_write_failure(
    tmp_path, monkeypatch
) -> None:
    fit = _make_dummy_fit()
    db_path = tmp_path / "fit_archive_rollback.db"
    with MFDatabase(db_path) as db:
        original_register_artifact = db.register_artifact

        def fail_fit_result_artifact(*args, **kwargs):
            if kwargs.get("artifact_kind") == "fit_result":
                raise RuntimeError("fit result failure")
            return original_register_artifact(*args, **kwargs)

        monkeypatch.setattr(db, "register_artifact", fail_fit_result_artifact)
        with pytest.raises(RuntimeError, match="fit result failure"):
            archive_fit_to_mfdb(
                db,
                fit,
                operation_id="op_fit_rollback",
                experiment_id="exp_fit_rollback",
                fit_id="fit-rollback",
                dataset_id="dataset-rollback",
            )

        for table in (
            "mfdb_operation",
            "mfdb_artifact",
            "mfdb_operation_artifact",
            "mfdb_parameter",
            "mfdb_edge",
        ):
            assert db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0

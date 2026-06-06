from __future__ import annotations

"""Tests for chisurf.server.services.fits."""

from unittest.mock import MagicMock

from chisurf.server.services.fits import (
    list_fits,
    get_fit_info,
    run_fit,
    ping,
    fit_create,
    fit_save,
    fit_set_dataset,
    fit_set_result_idx,
    fit_set_fit_range,
)
from chisurf.server.services.models import model_finalize, model_set_parse_function
from chisurf.server.session import SessionState


class TestFitsService:

    def test_list_fits_empty(self):
        state = SessionState()
        result = list_fits(state)
        assert result["fits"] == []

    def test_list_fits(self):
        fit = MagicMock()
        fit.unique_identifier = "fit-uid-1"
        fit.name = "Fit1"
        fit.chi2 = 1.5
        fit.data = MagicMock()
        fit.data.name = "Data1"

        state = SessionState(fits=[fit])
        result = list_fits(state)
        assert len(result["fits"]) == 1
        assert result["fits"][0]["name"] == "Fit1"
        assert result["fits"][0]["chi2"] == 1.5

    def test_get_fit_info_by_index(self):
        fit = MagicMock()
        fit.unique_identifier = "fit-uid"
        fit.name = "MyFit"
        fit.chi2 = 1.2
        fit.data = MagicMock()
        fit.data.name = "MyData"
        fit.model = MagicMock()
        fit.model.parameters_all_dict = {}

        state = SessionState(fits=[fit])
        result = get_fit_info(state, fit_index=0)
        assert result["fit"]["name"] == "MyFit"

    def test_get_fit_info_not_found(self):
        state = SessionState()
        result = get_fit_info(state, fit_index=0)
        assert not result["ok"]

    def test_run_fit_calls_method(self):
        fit = MagicMock()
        fit.unique_identifier = "fit-uid"
        fit.name = "MyFit"
        fit.chi2 = 3.0
        fit.data = MagicMock()
        fit.data.name = "D"
        fit.model = MagicMock()
        fit.model.parameters_all_dict = {}
        fit.model.parameter_values = []
        fit.model.parameter_bounds = []
        fit.grouped_fits = []

        state = SessionState(fits=[fit])
        result = run_fit(state, fit_index=0)
        assert fit.run.called
        assert result["ok"]

    def test_run_fit_nonexistent(self):
        state = SessionState()
        result = run_fit(state, fit_index=0)
        assert not result["ok"]

    def test_run_fit_by_uid(self):
        fit = MagicMock()
        fit.unique_identifier = "target-uid"
        fit.name = "Target"
        fit.chi2 = 1.0
        fit.data = MagicMock()
        fit.data.name = "D"
        fit.run = MagicMock()
        fit.model = MagicMock()
        fit.model.parameters_all_dict = {}

        state = SessionState(fits=[fit])
        result = run_fit(state, fit_uid="target-uid")
        assert result["ok"]
        fit.run.assert_called_once()

    def test_ping_returns_alive(self):
        from chisurf.server.services.fits import ping
        state = SessionState()
        result = ping(state)
        assert result["ok"]
        assert result["status"] == "alive"

    def test_list_fits_with_parameters_all(self):
        fit = MagicMock()
        fit.unique_identifier = "fit-p1"
        fit.name = "ParamFit"
        fit.chi2 = 2.0
        fit.data = MagicMock()
        fit.data.name = "Data1"
        fit.data.filename = "test.dat"
        fit.data.unique_identifier = "ds-1"
        fit.model = MagicMock()
        fit.model.name = "LifetimeModel"
        fit.model.n_points = 100
        fit.model.n_free = 3
        fit.model.chi2r = 2.0
        fit.model.parameters_all_dict = {"tau1": MagicMock(), "tau2": MagicMock()}
        p1 = MagicMock()
        p1.name = "tau1"
        p1.value = 3.5
        p1.fixed = True
        p1.bounds = (0, 10)
        p1.bounds_on = True
        p1.is_linked = False
        p1.link = None
        p1.error_estimate = 0.1
        p2 = MagicMock()
        p2.name = "tau2"
        p2.value = 1.2
        p2.fixed = False
        p2.bounds = (0, 5)
        p2.bounds_on = True
        p2.is_linked = False
        p2.link = None
        p2.error_estimate = None
        fit.model.parameters_all = [p1, p2]
        fit.model.chi2r = 2.0

        state = SessionState(fits=[fit])
        result = list_fits(state)
        assert len(result["fits"]) == 1
        model = result["fits"][0].get("model", {})
        params = model.get("parameters_all", [])
        assert len(params) == 2
        assert params[0]["name"] == "tau1"
        assert params[0]["value"] == 3.5
        assert params[0]["fixed"] is True
        assert params[1]["name"] == "tau2"
        assert params[1]["value"] == 1.2

    def test_get_fit_info_with_parameters_all(self):
        fit = MagicMock()
        fit.unique_identifier = "fit-p2"
        fit.name = "ParamFit2"
        fit.chi2 = 1.5
        fit.data = MagicMock()
        fit.data.name = "Data2"
        fit.data.filename = "other.dat"
        fit.data.unique_identifier = "ds-2"
        fit.model = MagicMock()
        fit.model.name = "Exponential"
        fit.model.n_points = 50
        fit.model.n_free = 2
        fit.model.chi2r = 1.5
        fit.model.parameters_all_dict = {"amp": MagicMock()}
        p = MagicMock()
        p.name = "amp"
        p.value = 0.8
        p.fixed = False
        p.bounds = (0, 1)
        p.bounds_on = True
        p.is_linked = True
        p.link = MagicMock()
        p.link.name = "master_amp"
        p.error_estimate = 0.05
        fit.model.parameters_all = [p]
        fit.model.chi2r = 1.5

        state = SessionState(fits=[fit])
        result = get_fit_info(state, fit_index=0)
        assert result["ok"]
        model = result["fit"].get("model", {})
        params = model.get("parameters_all", [])
        assert len(params) == 1
        assert params[0]["name"] == "amp"
        assert params[0]["value"] == 0.8
        assert params[0]["linked_to"] == "master_amp"
        state = SessionState()
        result = ping(state)
        assert result["ok"]
        assert result["status"] == "alive"

    # ── fit_create ────────────────────────────────────────────────

    def test_fit_create_no_datasets(self):
        state = SessionState()
        result = fit_create(state, dataset_index=0, model_name="Lifetime fit")
        assert not result["ok"]
        assert "no datasets" in result.get("error", "")

    def test_fit_create_invalid_dataset_index(self):
        ds = MagicMock()
        ds.unique_identifier = "ds-1"
        ds.name = "D"
        state = SessionState(datasets=[ds])
        result = fit_create(state, dataset_index=5, model_name="Lifetime fit")
        assert not result["ok"]
        assert "out of range" in result.get("error", "")

    def test_fit_create_invalid_dataset_indices(self):
        ds = MagicMock()
        ds.unique_identifier = "ds-1"
        ds.name = "D"
        state = SessionState(datasets=[ds])
        result = fit_create(state, dataset_indices=[0, 3], model_name="Lifetime fit")
        assert not result["ok"]
        assert "out of range" in result.get("error", "")

    def test_fit_create_defaults_to_single_index(self):
        ds = MagicMock()
        ds.unique_identifier = "ds-1"
        ds.name = "D"
        state = SessionState(datasets=[ds])
        result = fit_create(state)
        # Without a real model class available, this should fail at import
        assert not result["ok"]


    def test_fit_set_dataset_without_fit(self):
        state = SessionState()
        result = fit_set_dataset(state, fit_index=0, dataset_index=0)
        assert not result["ok"]

    def test_fit_set_dataset_without_dataset(self):
        fit = MagicMock()
        fit.unique_identifier = "fit-1"
        state = SessionState(fits=[fit])
        result = fit_set_dataset(state, fit_index=0, dataset_index=0)
        assert not result["ok"]

    def test_fit_set_result_idx_without_fit(self):
        state = SessionState()
        result = fit_set_result_idx(state, fit_index=0, result_idx=1)
        assert not result["ok"]

    def test_fit_set_fit_range_without_fit(self):
        state = SessionState()
        result = fit_set_fit_range(state, fit_index=0, xmin=0, xmax=100)
        assert not result["ok"]

    def test_model_finalize_without_fit(self):
        state = SessionState()
        result = model_finalize(state, fit_index=0)
        assert not result["ok"]

    def test_model_set_parse_function_without_fit(self):
        state = SessionState()
        result = model_set_parse_function(state, parse_function="y = a*x + b", fit_index=0)
        assert not result["ok"]

    def test_fit_save_without_fit(self):
        state = SessionState()
        result = fit_save(state, filename="/tmp/out.csv", fit_index=0)
        assert not result["ok"]

    def test_fit_set_fit_range_updates_fit(self):
        fit = MagicMock()
        fit.unique_identifier = "fit-1"
        fit.fit_range = (0, 255)
        state = SessionState(fits=[fit])
        result = fit_set_fit_range(state, fit_index=0, xmin=10, xmax=200)
        assert result["ok"]
        assert fit.fit_range == (10, 200)

    def test_fit_set_result_idx_updates_fit(self):
        fit = MagicMock()
        fit.unique_identifier = "fit-1"
        state = SessionState(fits=[fit])
        result = fit_set_result_idx(state, fit_index=0, result_idx=2)
        assert result["ok"]
        fit.set_result_idx.assert_called_once_with(2)


class TestGraphService:

    def test_build_graph_empty_state(self):
        from chisurf.server.services.graph import build_fit_graph
        state = SessionState()
        result = build_fit_graph(state)
        assert result["ok"]
        assert result["graph"]["nodes"] == []
        assert result["graph"]["edges"] == []

    def test_build_graph_single_fit(self):
        from chisurf.server.services.graph import build_fit_graph
        fit = MagicMock()
        fit.unique_identifier = "fit-1"
        fit.name = "TestFit"
        fit.data = MagicMock()
        fit.data.filename = "/path/to/data.dat"
        fit.model = MagicMock()
        fit.model.__class__.__module__ = "chisurf.core.models.test"
        fit.model.__class__.__name__ = "TestModel"
        fit.model.parameters_all = []

        state = SessionState(fits=[fit])
        result = build_fit_graph(state)
        assert result["ok"]
        assert len(result["graph"]["nodes"]) == 1
        node = result["graph"]["nodes"][0]
        assert node["node_type"] == "fit"
        assert node["name"] == "TestFit"
        assert node["data_filename"] == "/path/to/data.dat"

    def test_build_graph_with_parameters(self):
        from chisurf.server.services.graph import build_fit_graph
        param_free = MagicMock()
        param_free.name = "amp"
        param_free.value = 1.5
        param_free.fixed = False
        param_free.is_linked = False
        param_free.link = None

        param_fixed = MagicMock()
        param_fixed.name = "tau"
        param_fixed.value = 3.0
        param_fixed.fixed = True
        param_fixed.is_linked = False
        param_fixed.link = None

        fit = MagicMock()
        fit.unique_identifier = "fit-1"
        fit.name = "Fit"
        fit.data = MagicMock()
        fit.data.filename = ""
        fit.model = MagicMock()
        fit.model.__class__.__module__ = "mod"
        fit.model.__class__.__name__ = "M"
        fit.model.parameters_all = [param_free, param_fixed]

        state = SessionState(fits=[fit])
        result = build_fit_graph(state, include_fixed=True)
        assert len(result["graph"]["nodes"]) == 3  # 1 fit + 2 params
        assert len(result["graph"]["edges"]) == 2

    def test_build_graph_excludes_fixed_when_requested(self):
        from chisurf.server.services.graph import build_fit_graph
        param_free = MagicMock()
        param_free.name = "amp"
        param_free.value = 1.0
        param_free.fixed = False
        param_free.is_linked = False
        param_free.link = None

        param_fixed = MagicMock()
        param_fixed.name = "tau"
        param_fixed.value = 3.0
        param_fixed.fixed = True
        param_fixed.is_linked = False
        param_fixed.link = None

        fit = MagicMock()
        fit.unique_identifier = "fit-1"
        fit.name = "Fit"
        fit.data = MagicMock()
        fit.data.filename = ""
        fit.model = MagicMock()
        fit.model.__class__.__module__ = "mod"
        fit.model.__class__.__name__ = "M"
        fit.model.parameters_all = [param_free, param_fixed]

        state = SessionState(fits=[fit])
        result = build_fit_graph(state, include_fixed=False)
        assert len(result["graph"]["nodes"]) == 2  # 1 fit + 1 free param
        assert len(result["graph"]["edges"]) == 1

    def test_build_graph_linked_parameters(self):
        from chisurf.server.services.graph import build_fit_graph
        link_target = MagicMock()
        link_target.name = "tau"

        linked_param = MagicMock()
        linked_param.name = "tau"
        linked_param.value = 3.0
        linked_param.fixed = False
        linked_param.is_linked = True
        linked_param.link = link_target

        source_param = MagicMock()
        source_param.name = "tau"
        source_param.value = 3.0
        source_param.fixed = False
        source_param.is_linked = False
        source_param.link = None

        fit1 = MagicMock()
        fit1.unique_identifier = "fit-1"
        fit1.name = "Fit1"
        fit1.data = MagicMock()
        fit1.data.filename = ""
        fit1.model = MagicMock()
        fit1.model.__class__.__module__ = "mod"
        fit1.model.__class__.__name__ = "M"
        fit1.model.parameters_all = [linked_param]

        fit2 = MagicMock()
        fit2.unique_identifier = "fit-2"
        fit2.name = "Fit2"
        fit2.data = MagicMock()
        fit2.data.filename = ""
        fit2.model = MagicMock()
        fit2.model.__class__.__module__ = "mod"
        fit2.model.__class__.__name__ = "M"
        fit2.model.parameters_all = [source_param]

        state = SessionState(fits=[fit1, fit2])
        result = build_fit_graph(state)
        assert result["ok"]
        nodes = result["graph"]["nodes"]
        edges = result["graph"]["edges"]

        # 2 fit nodes + 2 param nodes = 4 nodes total
        assert len(nodes) == 4
        fit_nodes = [n for n in nodes if n["node_type"] == "fit"]
        param_nodes = [n for n in nodes if n["node_type"] == "parameter"]
        assert len(fit_nodes) == 2
        assert len(param_nodes) == 2

        # Each param should connect to its fit, plus one cross-param link edge
        param_to_fit = [e for e in edges if any(
            n["node_idx"] == e["source"] and n["node_type"] == "parameter"
            for n in nodes
        ) and any(
            n["node_idx"] == e["target"] and n["node_type"] == "fit"
            for n in nodes
        )]
        assert len(param_to_fit) == 2

        # The linked param should produce a cross-param edge
        cross_param = [e for e in edges if any(
            n["node_idx"] == e["source"] and n["node_type"] == "parameter"
            for n in nodes
        ) and any(
            n["node_idx"] == e["target"] and n["node_type"] == "parameter"
            for n in nodes
        )]
        assert len(cross_param) >= 1

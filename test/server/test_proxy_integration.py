from __future__ import annotations

"""Integration-style tests for the clean typed proxy classes."""
import chisurf as cs
from unittest.mock import MagicMock, patch

from chisurf.core.api._client import ChisurfClient
from chisurf.core.api._proxies import (
    FitProxy,
    DatasetProxy,
    DataProxy,
    ParameterProxy,
    ProxyDatasetList,
    ProxyFitList,
)
from chisurf.macros.model_parse import _is_proxy_for


class TestIsinstancePatterns:
    """Plugins use isinstance checks like 'isinstance(fit, Fit)'."""

    def test_isinstance_fit_via_real_class(self):
        """FitProxy is a real class, isinstance works naturally."""
        proxy = FitProxy({"uid": "f1", "name": "TestFit"})
        assert isinstance(proxy, FitProxy)
        assert not isinstance(proxy, DatasetProxy)

    def test_isinstance_dataset_via_real_class(self):
        """DatasetProxy is a real class."""
        proxy = DatasetProxy({"uid": "ds1", "name": "TestData"})
        assert isinstance(proxy, DatasetProxy)

    def test_is_proxy_for_recognizes_fit(self):
        """_is_proxy_for checks _data.type field."""
        proxy = FitProxy({
            "uid": "f1", "name": "Fit1", "type": "FitGroup",
            "chi2": 1.5,
            "data": {"name": "Data", "uid": "ds1"},
            "model": {"name": "Lifetime", "parameters_all": []},
        })
        assert _is_proxy_for(proxy, "FitGroup")
        assert not _is_proxy_for(proxy, "DataCurve")

    def test_is_proxy_for_data_curve(self):
        """_is_proxy_for correctly identifies a DataCurve."""
        proxy = DatasetProxy({
            "uid": "ds1", "name": "DataCurve", "type": "DataCurve",
        })
        assert _is_proxy_for(proxy, "DataCurve")
        assert not _is_proxy_for(proxy, "FitGroup")


class TestPluginIterationPatterns:
    """Plugins iterate over fit lists and access attributes."""

    def test_iterate_fits_and_access_model(self):
        """Typical pattern: iterate fits, access .model.parameters_all."""
        client = MagicMock(spec=ChisurfClient)
        client.fit__list.return_value = [
            {
                "uid": "f1", "name": "Fit1", "chi2": 1.5,
                "model": {
                    "name": "Lifetime",
                    "parameters_all": [
                        {"name": "tau1", "value": 3.5, "fixed": False, "fit_uid": "f1"},
                        {"name": "tau2", "value": 8.1, "fixed": True, "fit_uid": "f1"},
                    ],
                },
                "data": {"name": "Data1", "uid": "ds1"},
            },
        ]
        flist = ProxyFitList(client)
        for fit in flist:
            assert isinstance(fit, FitProxy)
            assert fit.name == "Fit1"
            assert fit.chi2 == 1.5
            params = fit.model.parameters_all
            assert len(params) == 2
            assert params[0].name == "tau1"
            assert params[1].value == 8.1
            assert fit.data.name == "Data1"

    def test_parameters_all_dict_access(self):
        """Fit.model.parameters_all_dict provides name->proxy mapping."""
        client = MagicMock(spec=ChisurfClient)
        client.fit__list.return_value = [
            {
                "uid": "f1", "name": "GroupFit",
                "model": {
                    "name": "MultiExp",
                    "parameters_all": [
                        {"name": "tau1", "value": 3.0, "fixed": False, "fit_uid": "f1"},
                        {"name": "amp1", "value": 0.5, "fixed": False, "fit_uid": "f1"},
                    ],
                },
            },
        ]
        flist = ProxyFitList(client)
        fit = flist[0]
        pdict = fit.model.parameters_all_dict
        assert isinstance(pdict, dict)
        assert "tau1" in pdict
        assert "amp1" in pdict
        assert pdict["tau1"].value == 3.0

    def test_fit_data_access(self):
        """fit.data returns a DataProxy with metadata."""
        client = MagicMock(spec=ChisurfClient)
        client.fit__list.return_value = [
            {
                "uid": "f1", "name": "FitWithData",
                "chi2": 1.2,
                "data": {
                    "name": "MyData",
                    "filename": "/tmp/data.dat",
                    "uid": "ds-1",
                },
                "model": {"name": "Lifetime"},
            },
        ]
        flist = ProxyFitList(client)
        fit = flist[0]
        assert isinstance(fit.data, DataProxy)
        assert fit.data.name == "MyData"
        assert fit.data.uid == "ds-1"

    def test_mixed_fit_and_dataset_lists(self):
        """Plugin iterates fits and datasets side by side."""
        client = MagicMock(spec=ChisurfClient)
        client.fit__list.return_value = [
            {"uid": "f1", "name": "FitA", "chi2": 1.0},
        ]
        client.dataset__list.return_value = [
            {"uid": "ds1", "name": "DataA"},
        ]
        flist = ProxyFitList(client)
        dlist = ProxyDatasetList(client)
        assert len(flist) == 1
        assert len(dlist) == 1
        assert flist[0].uid == "f1"
        assert dlist[0].uid == "ds1"


class TestProxyMutationPatterns:
    """Plugins mutate parameters through explicit methods (no __setattr__)."""

    def test_param_set_value_via_method(self):
        """param.set_value(4.5) → server call."""
        client = MagicMock(spec=ChisurfClient)
        client.call.return_value = {"ok": True}
        param = ParameterProxy({
            "name": "tau1", "value": 3.0, "fit_uid": "f1", "fixed": False,
        }, client=client)
        param.set_value(4.5)
        client.call.assert_called_once_with(
            "parameter.set_value",
            {"parameter_name": "tau1", "value": 4.5, "fit_uid": "f1"},
        )

    def test_param_set_fixed_via_method(self):
        client = MagicMock(spec=ChisurfClient)
        client.call.return_value = {"ok": True}
        param = ParameterProxy({
            "name": "tau1", "fixed": False, "fit_uid": "f1",
        }, client=client)
        param.set_fixed(True)
        client.call.assert_called_once_with(
            "parameter.set_fixed",
            {"parameter_name": "tau1", "fixed": True, "fit_uid": "f1"},
        )

    def test_param_set_bounds_via_method(self):
        client = MagicMock(spec=ChisurfClient)
        client.call.return_value = {"ok": True}
        param = ParameterProxy({
            "name": "tau1", "bounds": (0.0, 10.0), "fit_uid": "f1",
        }, client=client)
        param.set_bounds((1.0, 20.0))
        client.call.assert_called_once_with(
            "parameter.set_bounds",
            {"parameter_name": "tau1", "bounds": [1.0, 20.0], "fit_uid": "f1"},
        )

    def test_param_set_bounds_on_via_method(self):
        client = MagicMock(spec=ChisurfClient)
        client.call.return_value = {"ok": True}
        param = ParameterProxy({
            "name": "tau1", "bounds_on": False, "fit_uid": "f1",
        }, client=client)
        param.set_bounds_on(True)
        client.call.assert_called_once_with(
            "parameter.set_bounds_on",
            {"parameter_name": "tau1", "bounds_on": True, "fit_uid": "f1"},
        )

    def test_param_link_to_via_method(self):
        client = MagicMock(spec=ChisurfClient)
        client.call.return_value = {"ok": True}
        param = ParameterProxy({
            "name": "tau1", "fit_uid": "f1",
        }, client=client)
        param.link_to("tau2")
        client.call.assert_called_once_with(
            "parameter.link",
            {"parameter_name": "tau1", "target_parameter_name": "tau2", "fit_uid": "f1"},
        )

    def test_param_unlink_via_method(self):
        client = MagicMock(spec=ChisurfClient)
        client.call.return_value = {"ok": True}
        param = ParameterProxy({
            "name": "tau1", "fit_uid": "f1",
        }, client=client)
        param.unlink()
        client.call.assert_called_once_with(
            "parameter.unlink",
            {"parameter_name": "tau1", "fit_uid": "f1"},
        )

    def test_fit_set_result_idx_via_method(self):
        client = MagicMock(spec=ChisurfClient)
        client.call.return_value = {"ok": True}
        fit = FitProxy({"uid": "f1", "name": "TestFit"}, client=client)
        result = fit.set_result_idx(2)
        client.call.assert_called_once_with(
            "fit.set_result_idx",
            {"fit_uid": "f1", "result_idx": 2},
        )
        assert result.get("ok") is True


class TestProxyUpdateMethod:
    """Plugins call .update(), .run() on FitProxy."""

    def test_fit_update_calls_server(self):
        client = MagicMock(spec=ChisurfClient)
        client.call.return_value = {"ok": True, "chi2": 1.5}
        fit = FitProxy({"uid": "f1", "name": "UpdateTest"}, client=client)
        result = fit.update()
        client.call.assert_called_once_with("fit.update", {"fit_uid": "f1"})
        assert result.get("ok") is True

    def test_fit_run_calls_server(self):
        client = MagicMock(spec=ChisurfClient)
        client.call.return_value = {"ok": True, "chi2_before": 3.0}
        fit = FitProxy({"uid": "f1", "name": "RunTest"}, client=client)
        result = fit.run()
        client.call.assert_called_once_with("fit.run", {"fit_uid": "f1"})

    def test_model_finalize_calls_server(self):
        client = MagicMock(spec=ChisurfClient)
        client.call.return_value = {"ok": True}
        fit = FitProxy({"uid": "f1", "name": "FinalizeTest"}, client=client)
        result = fit.model_finalize()
        client.call.assert_called_once_with("model.finalize", {"fit_uid": "f1"})

    def test_model_set_parse_function(self):
        client = MagicMock(spec=ChisurfClient)
        client.call.return_value = {"ok": True}
        fit = FitProxy({"uid": "f1", "name": "ParseFnTest"}, client=client)
        result = fit.model_set_parse_function("expr")
        client.call.assert_called_once_with(
            "model.set_parse_function",
            {"fit_uid": "f1", "function_name": "expr"},
        )


class TestProxyListMutations:
    """Plugins use pop/remove/clear on proxy lists."""

    def test_dataset_pop_and_verify(self):
        client = MagicMock(spec=ChisurfClient)
        client.dataset__list.return_value = [
            {"uid": "ds1", "name": "A"},
            {"uid": "ds2", "name": "B"},
        ]
        client.dataset__remove.return_value = {"ok": True}
        dlist = ProxyDatasetList(client)
        popped = dlist.pop(0)
        assert popped.uid == "ds1"
        client.dataset__remove.assert_called_once_with(
            dataset_indices=[], dataset_uids=["ds1"]
        )

    def test_fit_clear_then_list_empty(self):
        client = MagicMock(spec=ChisurfClient)
        client.fit__list.return_value = [
            {"uid": "f1", "name": "Fit1"},
        ]
        client.fit__clear.return_value = {"ok": True}
        flist = ProxyFitList(client)
        assert len(flist) == 1
        flist.clear()
        client.fit__list.return_value = []
        assert len(flist) == 0

    def test_proxy_list_contains(self):
        client = MagicMock(spec=ChisurfClient)
        client.dataset__list.return_value = [
            {"uid": "ds1", "name": "A"},
            {"uid": "ds2", "name": "B"},
        ]
        dlist = ProxyDatasetList(client)
        item = dlist[0]
        assert item in dlist
        missing = DatasetProxy({"uid": "ds99"})
        assert missing not in dlist


class TestModelParseIntegration:
    """Plugin pattern: _as_iterable_fits uses _is_proxy_for integration."""

    def test_as_iterable_fits_with_proxy(self):
        """_as_iterable_fits works with FitProxy objects."""
        from chisurf.macros.model_parse import _as_iterable_fits
        client = MagicMock(spec=ChisurfClient)
        client.fit__list.return_value = [
            {
                "uid": "f1", "name": "Fit1", "type": "FitGroup",
                "chi2": 1.5,
                "model": {"name": "Lifetime", "parameters_all": []},
                "data": {"name": "Data1", "uid": "ds1"},
            },
        ]
        flist = ProxyFitList(client)
        fit = flist[0]
        targets = _as_iterable_fits(fit)
        assert len(targets) == 1
        assert targets[0] is fit


class TestServerModeGuards:
    """Plugin server-mode guard patterns (wizard.py)."""

    def test_plugin_server_mode_guard_pattern(self):
        api = MagicMock()
        api.mode = "server"
        with patch("chisurf.core.api", api, create=True):
            assert cs.core.api.mode == "server"

    def test_plugin_local_mode_bypasses_guard(self):
        api = MagicMock()
        api.mode = "local"
        with patch("chisurf.core.api", api, create=True):
            assert cs.core.api.mode == "local"

    def test_server_mode_read_works(self):
        """Read-only access works on FitProxy."""
        fit = FitProxy({
            "uid": "f1", "name": "ReadOnly", "type": "FitGroup",
            "chi2": 1.5, "model": {"name": "Lifetime"},
        })
        assert fit.name == "ReadOnly"
        assert fit.chi2 == 1.5


class TestChisurfRunPattern:
    """cs.run() constructs Python strings and execs them against proxies."""

    def test_run_pattern_index_then_rpc_call(self):
        """cs.run('cs.fits[0].set_result_idx(2)') pattern works."""
        client = MagicMock(spec=ChisurfClient)
        client.fit__list.return_value = [
            {"uid": "f1", "name": "Fit1", "chi2": 1.0},
        ]
        client.call.return_value = {"ok": True}
        flist = ProxyFitList(client)
        fit = flist[0]
        assert fit.uid == "f1"
        result = fit.set_result_idx(result_idx=2)
        client.call.assert_called_once_with(
            "fit.set_result_idx",
            {"fit_uid": "f1", "result_idx": 2},
        )
        assert result.get("ok") is True

    def test_run_pattern_index_inside_run(self):
        """Construct the string that cs.run() would execute."""
        client = MagicMock(spec=ChisurfClient)
        client.fit__list.return_value = [
            {"uid": "f1", "name": "Fit1"},
            {"uid": "f2", "name": "Fit2"},
        ]
        client.call.return_value = {"ok": True}
        flist = ProxyFitList(client)
        fit_idx = 0
        result_idx = 2
        expr = f"cs.fits[{fit_idx}].set_result_idx({result_idx})"
        assert expr == "cs.fits[0].set_result_idx(2)"
        fit = flist[fit_idx]
        result = fit.set_result_idx(result_idx=result_idx)
        assert result.get("ok") is True

    def test_enumerate_pattern(self):
        """for fit_idx, f in enumerate(cs.fits) GUI pattern."""
        client = MagicMock(spec=ChisurfClient)
        client.fit__list.return_value = [
            {"uid": "f1", "name": "Fit1", "chi2": 1.0},
            {"uid": "f2", "name": "Fit2", "chi2": 2.0},
        ]
        flist = ProxyFitList(client)
        enumerated = list(enumerate(flist))
        assert len(enumerated) == 2
        assert enumerated[0][0] == 0
        assert enumerated[0][1].uid == "f1"
        assert enumerated[1][1].name == "Fit2"

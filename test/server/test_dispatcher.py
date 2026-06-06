from __future__ import annotations

"""Tests for chisurf.server.dispatcher."""

import pytest
from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.services import service_error, NOT_FOUND, INVALID_INPUT, OPERATION_FAILED
from chisurf.server.session import SessionState


def sample_handler(params: dict) -> dict:
    return {"result": params.get("value", 0) * 2}


def failing_handler(params: dict) -> dict:
    raise ValueError("handler error")


def invalid_params_handler(params: dict) -> dict:
    raise TypeError("bad params")


class TestServiceErrorHelper:

    def test_service_error_basic(self):
        result = service_error("something went wrong", error_code=OPERATION_FAILED)
        assert result["ok"] is False
        assert result["error"] == "something went wrong"
        assert result["error_code"] == OPERATION_FAILED
        assert result["jsonrpc_code"] == -32603

    def test_service_error_not_found(self):
        result = service_error("fit not found", error_code=NOT_FOUND)
        assert result["ok"] is False
        assert result["error_code"] == NOT_FOUND
        assert result["jsonrpc_code"] == -32601

    def test_service_error_invalid_input(self):
        result = service_error("bad params", error_code=INVALID_INPUT)
        assert result["ok"] is False
        assert result["error_code"] == INVALID_INPUT
        assert result["jsonrpc_code"] == -32602

    def test_service_error_with_exception(self):
        try:
            raise ValueError("test exception")
        except ValueError as e:
            result = service_error(str(e), error_code=OPERATION_FAILED, exception=e)
        assert result["ok"] is False
        assert result["error_code"] == OPERATION_FAILED
        assert result["jsonrpc_code"] == -32603
        assert result["exception_type"] == "ValueError"

    def test_service_error_custom_jsonrpc_code(self):
        result = service_error("custom", error_code="CUSTOM", jsonrpc_code=-32000)
        assert result["error_code"] == "CUSTOM"
        assert result["jsonrpc_code"] == -32000

    def test_service_error_unknown_code_defaults_internal(self):
        result = service_error("unknown", error_code="UNKNOWN_CODE")
        assert result["jsonrpc_code"] == -32603


class TestServiceDispatcher:

    def test_register_and_dispatch(self):
        d = ServiceDispatcher(SessionState())
        d.register("double", sample_handler)
        result = d.dispatch("double", {"value": 21})
        assert result == {"result": 42}

    def test_dispatch_unknown(self):
        d = ServiceDispatcher(SessionState())
        result = d.dispatch("nonexistent", {})
        assert "error" in result
        assert "not found" in result["error"]
        assert result["error_code"] == "METHOD_NOT_FOUND"
        assert result["jsonrpc_code"] == -32601

    def test_dispatch_handler_error(self):
        d = ServiceDispatcher(SessionState())
        d.register("fail", failing_handler)
        result = d.dispatch("fail", {})
        assert "error" in result
        assert result["error_code"] == "INTERNAL_ERROR"
        assert result["jsonrpc_code"] == -32603
        assert result["exception_type"] == "ValueError"

    def test_dispatch_invalid_params_error(self):
        d = ServiceDispatcher(SessionState())
        d.register("bad_params", invalid_params_handler)
        result = d.dispatch("bad_params", {})
        assert result["ok"] is False
        assert result["error_code"] == "INVALID_PARAMS"
        assert result["jsonrpc_code"] == -32602
        assert result["exception_type"] == "TypeError"

    def test_has_method(self):
        d = ServiceDispatcher(SessionState())
        d.register("ping", sample_handler)
        assert d.has_method("ping")
        assert not d.has_method("nope")

    def test_list_methods(self):
        d = ServiceDispatcher(SessionState())
        d.register("a", sample_handler)
        d.register("b", sample_handler)
        methods = d.list_methods()
        assert "a" in methods
        assert "b" in methods

    def test_dispatch_returns_service_result(self):
        d = ServiceDispatcher(SessionState())

        def my_handler(params):
            return {"ok": True, "value": params["x"] + params["y"]}

        d.register("add", my_handler)
        result = d.dispatch("add", {"x": 3, "y": 4})
        assert result["ok"]
        assert result["value"] == 7

    def test_build_default_registry_has_ping(self):
        d = ServiceDispatcher(SessionState())
        d._build_default_registry()
        assert d.has_method("ping")
        assert d.has_method("list_datasets")
        assert d.has_method("list_fits")
        assert d.has_method("list_methods")
        assert d.has_method("graph.build")
        assert d.has_method("graph.build_fits")
        assert d.has_method("fit.create")
        assert d.has_method("fit.add")

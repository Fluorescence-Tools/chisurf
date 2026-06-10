"""Tests for PluginClient protocol and InProcessClient."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chisurf.core.plugin.client import InProcessClient


class TestInProcessClient:
    """InProcessClient wraps a ServiceDispatcher."""

    def test_call_returns_dispatcher_result(self):
        dispatcher = MagicMock()
        dispatcher.dispatch.return_value = {"ok": True, "result": "done"}

        client = InProcessClient(dispatcher)
        result = client.call("test.ping", {"key": "value"})

        assert result == {"ok": True, "result": "done"}
        dispatcher.dispatch.assert_called_once_with("test.ping", {"key": "value"})

    def test_call_default_params(self):
        dispatcher = MagicMock()
        dispatcher.dispatch.return_value = {"ok": True}

        client = InProcessClient(dispatcher)
        result = client.call("test.ping")

        assert result == {"ok": True}
        dispatcher.dispatch.assert_called_once_with("test.ping", None)

    def test_subscribe_and_unsubscribe(self):
        dispatcher = MagicMock()
        client = InProcessClient(dispatcher)

        callback = MagicMock()
        token = client.subscribe("test.topic", callback)
        assert token is not None

        # Unsubscribe should not raise
        client.unsubscribe(token)

    def test_is_connected(self):
        client = InProcessClient(MagicMock())
        assert client.is_connected is True

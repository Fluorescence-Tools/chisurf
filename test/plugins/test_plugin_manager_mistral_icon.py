"""Tests for Mistral-backed plugin icon generation."""

import sys
from types import SimpleNamespace

import pytest

from chisurf.plugins.core.plugin_manager import AIIconRateLimitError
from chisurf.plugins.core.plugin_manager import PluginManagerWidget


class _TextField:
    """Minimal text field stand-in for plugin manager settings controls."""

    def __init__(self, value):
        """Store the text value returned by the fake control."""
        self._value = value

    def text(self):
        """Return the configured control text."""
        return self._value


class _Response:
    """Small requests response double used by icon generation tests."""

    def __init__(self, status_code=200, payload=None, content=b"", text="", headers=None):
        """Store response attributes consumed by the production code."""
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content
        self.text = text
        self.headers = headers or {}

    def json(self):
        """Return the configured JSON payload."""
        return self._payload

    def raise_for_status(self):
        """Raise an HTTP-like error for failing status codes."""
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code}: {self.text}")


def _manager(endpoint="https://api.mistral.ai/v1/beta"):
    """Create a lightweight manager object with bound Mistral helpers."""
    manager = SimpleNamespace(
        icon_endpoint_edit=_TextField(endpoint),
        icon_model_edit=_TextField("mistral-medium-latest"),
    )
    manager._api_key_for_provider = lambda _provider: "test-key"
    manager._ai_icon_prompt = lambda _plugin_info: "make an icon"
    manager._mistral_endpoint_url = lambda path: PluginManagerWidget._mistral_endpoint_url(manager, path)
    manager._extract_mistral_file_id = lambda data: PluginManagerWidget._extract_mistral_file_id(manager, data)
    manager._post_mistral_json_with_retries = (
        lambda requests_module, path, headers, payload, timeout: PluginManagerWidget._post_mistral_json_with_retries(
            manager,
            requests_module,
            path,
            headers,
            payload,
            timeout,
        )
    )
    manager._retry_after_seconds = lambda response: PluginManagerWidget._retry_after_seconds(manager, response)
    return manager


def test_mistral_icon_generation_uses_stable_v1_endpoints(monkeypatch):
    """Mistral icon generation should strip stale beta suffixes from endpoints."""
    calls = []

    def post(url, **kwargs):
        calls.append(("POST", url, kwargs.get("json")))
        if url.endswith("/agents"):
            return _Response(payload={"id": "agent-123"})
        return _Response(
            payload={
                "outputs": [
                    {
                        "content": [
                            {
                                "type": "tool_file",
                                "file_id": "file-123",
                                "file_type": "png",
                            }
                        ]
                    }
                ]
            }
        )

    def get(url, **kwargs):
        calls.append(("GET", url, None))
        return _Response(content=b"png-bytes")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post, get=get))

    image_bytes = PluginManagerWidget._request_mistral_generated_icon_bytes(_manager(), {"name": "Demo"})

    assert image_bytes == b"png-bytes"
    assert [call[1] for call in calls] == [
        "https://api.mistral.ai/v1/agents",
        "https://api.mistral.ai/v1/conversations",
        "https://api.mistral.ai/v1/files/file-123/content",
    ]
    assert calls[1][2]["stream"] is False


def test_openai_compatible_icon_generation_retries_without_response_format(monkeypatch):
    """OpenAI-compatible image fallback should not force legacy response_format."""
    calls = []
    manager = SimpleNamespace(
        icon_endpoint_edit=_TextField("https://api.example.test/v1"),
        icon_model_edit=_TextField("image-model"),
    )
    manager._selected_icon_provider = lambda: "custom"
    manager._api_key_for_provider = lambda _provider: "test-key"
    manager._ai_icon_prompt = lambda _plugin_info: "make an icon"

    def post(url, **kwargs):
        payload = kwargs.get("json") or {}
        calls.append(payload)
        if len(calls) == 1:
            return _Response(status_code=400, text="unknown output_format")
        if "response_format" in payload:
            return _Response(status_code=400, text="Unknown parameter: response_format")
        return _Response(payload={"data": [{"b64_json": "cG5nLWJ5dGVz"}]})

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))

    image_bytes = PluginManagerWidget._request_openai_compatible_icon_bytes(manager, {"name": "Demo"})

    assert image_bytes == b"png-bytes"
    assert calls[0]["output_format"] == "png"
    assert "output_format" not in calls[1]
    assert "response_format" not in calls[1]


def test_openai_compatible_icon_generation_passes_api_key(monkeypatch):
    """OpenAI-compatible image requests should include the configured API key."""
    captured = {}
    manager = SimpleNamespace(
        icon_endpoint_edit=_TextField("https://api.example.test/v1"),
        icon_model_edit=_TextField("gpt-image-2"),
    )
    manager._selected_icon_provider = lambda: "openai"
    manager._api_key_for_provider = lambda _provider: "sk-test-key"
    manager._ai_icon_prompt = lambda _plugin_info: "make an icon"

    def post(url, **kwargs):
        captured.update(kwargs)
        return _Response(payload={"data": [{"b64_json": "cG5nLWJ5dGVz"}]})

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))

    image_bytes = PluginManagerWidget._request_openai_compatible_icon_bytes(manager, {"name": "Demo"})

    assert image_bytes == b"png-bytes"
    assert captured["headers"]["Authorization"] == "Bearer sk-test-key"


def test_openai_icon_defaults_ignore_text_model_for_image_generation():
    """OpenAI image defaults should not reuse a text model as the image model."""
    manager = SimpleNamespace(
        icon_generation_settings={
            "provider": "openai",
            "endpoint": "https://api.openai.com/v1",
            "image_model": "gpt-5.5-2026-04-23",
        },
    )

    endpoint, model = PluginManagerWidget._default_icon_generation_values(manager, "openai")

    assert endpoint == "https://api.openai.com/v1"
    assert model == "gpt-image-2"


def test_mistral_icon_generation_falls_back_to_file_download(monkeypatch):
    """Mistral file retrieval should tolerate download endpoints."""
    calls = []

    def post(url, **kwargs):
        calls.append(("POST", url))
        if url.endswith("/agents"):
            return _Response(payload={"id": "agent-123"})
        return _Response(payload={"file_id": "file-123", "file_type": "png"})

    def get(url, **kwargs):
        calls.append(("GET", url))
        if url.endswith("/content"):
            return _Response(status_code=404, text="not found")
        return _Response(content=b"downloaded-png")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post, get=get))

    image_bytes = PluginManagerWidget._request_mistral_generated_icon_bytes(
        _manager("https://api.mistral.ai/v1"),
        {"name": "Demo"},
        prompt="make an icon",
    )

    assert image_bytes == b"downloaded-png"
    assert calls[-2:] == [
        ("GET", "https://api.mistral.ai/v1/files/file-123/content"),
        ("GET", "https://api.mistral.ai/v1/files/file-123/download"),
    ]


def test_ai_icon_prompt_uses_scientific_template():
    """AI icon prompts should follow the configured scientific icon template."""
    manager = SimpleNamespace()
    manager._icon_visual_elements = lambda name, description: PluginManagerWidget._icon_visual_elements(
        manager,
        name,
        description,
    )
    manager._icon_symbol_pair = lambda name, description: PluginManagerWidget._icon_symbol_pair(
        manager,
        name,
        description,
    )

    prompt = PluginManagerWidget._ai_icon_prompt(
        manager,
        {
            "name": "FCS:Correlation",
            "doc": "Analyze fluorescence correlation spectroscopy curves.\nAdditional details.",
        },
    )

    assert prompt.startswith("Scientific Software Icon Template")
    assert "Square icon, 128x128 pixels" in prompt
    assert "Plugin: FCS:Correlation" in prompt
    assert "Analyze fluorescence correlation spectroscopy curves." in prompt
    assert "correlation curve + confocal detection spot" in prompt
    assert "Negative prompts" in prompt


def test_mistral_post_retries_429_with_retry_after(monkeypatch):
    """Mistral POST helper should retry rate limits before returning."""
    manager = _manager("https://api.mistral.ai/v1")
    calls = []
    sleeps = []

    def post(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            return _Response(status_code=429, text="rate limited", headers={"Retry-After": "0.25"})
        return _Response(payload={"ok": True})

    monkeypatch.setattr("time.sleep", lambda delay: sleeps.append(delay))

    response = PluginManagerWidget._post_mistral_json_with_retries(
        manager,
        SimpleNamespace(post=post),
        "conversations",
        headers={},
        payload={},
        timeout=1,
    )

    assert response.json() == {"ok": True}
    assert calls == [
        "https://api.mistral.ai/v1/conversations",
        "https://api.mistral.ai/v1/conversations",
    ]
    assert sleeps == [0.25]


def test_mistral_post_raises_clear_error_after_429_retries(monkeypatch):
    """Mistral POST helper should raise a provider-specific error after retrying."""
    manager = _manager("https://api.mistral.ai/v1")
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return _Response(status_code=429, text="rate limited")

    monkeypatch.setattr("time.sleep", lambda _delay: None)

    with pytest.raises(AIIconRateLimitError, match="429 Too Many Requests"):
        PluginManagerWidget._post_mistral_json_with_retries(
            manager,
            SimpleNamespace(post=post),
            "conversations",
            headers={},
            payload={},
            timeout=1,
        )

    assert len(calls) == 3

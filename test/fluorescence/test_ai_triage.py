"""Tests for AI-assisted triage (PRD-06 Task 9)."""
from __future__ import annotations

import json
from unittest import mock

import pytest

from chisurf.core.fluorescence.curation import ai_triage
from chisurf.core.fluorescence.curation.ai_triage import (
    _call_llm,
    _parse_llm_reply,
    run_deterministic_checks,
)


def test_deterministic_qy_out_of_range():
    probe = {"qy": "1.5", "abs_max": "500", "em_max": "550", "ext_coeff": "50000"}
    result = run_deterministic_checks(probe)
    assert any("QY > 1" in i for i in result["issues"])


def test_deterministic_negative_stokes_shift():
    probe = {"abs_max": "550", "em_max": "500", "qy": "0.5", "ext_coeff": "50000"}
    result = run_deterministic_checks(probe)
    assert any("negative Stokes" in i for i in result["issues"])


def test_deterministic_missing_spectra():
    probe = {"abs_max": "500", "em_max": "550", "qy": "0.5", "ext_coeff": "50000"}
    result = run_deterministic_checks(probe)
    assert any("missing" in i and "spectrum" in i for i in result["issues"])


def test_deterministic_clean_probe():
    probe = {
        "abs_max": "500", "em_max": "550", "qy": "0.8", "ext_coeff": "80000",
        "has_abs": True, "has_em": True,
    }
    result = run_deterministic_checks(probe)
    assert len(result["issues"]) == 0
    assert result["proposed_quality"] == "high"


def test_deterministic_quality_scoring():
    """Probe with only metadata issues (no missing spectra) gets medium."""
    probe = {
        "abs_max": "500", "em_max": "495", "qy": "0.8", "ext_coeff": "80000",
        "has_abs": True, "has_em": True,
    }
    result = run_deterministic_checks(probe)
    assert result["proposed_quality"] == "medium"


def test_parse_llm_reply_valid_json():
    reply = '{"category": "organic_dye", "recommendation": "approve", "rationale": "Looks good"}'
    parsed = _parse_llm_reply(reply)
    assert parsed is not None
    assert parsed["category"] == "organic_dye"


def test_parse_llm_reply_markdown_fence():
    reply = '```json\n{"category": "protein", "recommendation": "needs_review"}\n```'
    parsed = _parse_llm_reply(reply)
    assert parsed is not None
    assert parsed["category"] == "protein"


def test_parse_llm_reply_invalid():
    reply = "I think this is a dye. Not sure."
    parsed = _parse_llm_reply(reply)
    assert parsed is None


# ── _call_llm: provider-neutral, no live network (PRD-06 Task 9) ──────────────

def _fake_settings(**overrides):
    base = {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "text_model": "gpt-4o",
        "model": "gpt-4o",
        "api_key": "",
        "temperature": 0.3,
        "max_tokens": 1024,
    }
    base.update(overrides)
    return base


def test_call_llm_posts_to_chat_completions_when_configured():
    """A configured provider POSTs to the OpenAI-compatible endpoint and returns text."""
    settings = _fake_settings(api_key="sk-test", base_url="https://api.mistral.ai/v1",
                              text_model="mistral-small-latest", model="mistral-small-latest")
    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    resp.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
    with mock.patch("chisurf.core.settings.ai_settings.get_api_settings", return_value=settings), \
         mock.patch("requests.post", return_value=resp) as post:
        out = _call_llm("prompt", provider="mistral")
    assert out == "hello"
    assert post.call_count == 1
    url = post.call_args[0][0] if post.call_args[0] else post.call_args.kwargs.get("url")
    assert url.endswith("/chat/completions")
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"


def test_call_llm_skips_when_no_key_and_remote():
    """Unconfigured default (cloud URL, no key) must NOT hit the network."""
    settings = _fake_settings(api_key="")  # openai cloud, no key
    with mock.patch("chisurf.core.settings.ai_settings.get_api_settings", return_value=settings), \
         mock.patch("requests.post", side_effect=AssertionError("network hit")) as post:
        out = _call_llm("prompt")
    assert out is None
    assert post.call_count == 0


def test_call_llm_allows_local_without_key():
    """A local provider (Ollama/LMStudio) works keyless."""
    settings = _fake_settings(provider="local", base_url="http://localhost:11434/v1",
                              text_model="llama3.2", model="llama3.2", api_key="")
    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    with mock.patch("chisurf.core.settings.ai_settings.get_api_settings", return_value=settings), \
         mock.patch("requests.post", return_value=resp) as post:
        out = _call_llm("prompt", provider="local")
    assert out == "{}"
    assert post.call_count == 1
    assert "Authorization" not in post.call_args.kwargs["headers"]


def test_call_llm_no_model_skips():
    settings = _fake_settings(text_model="", model="")
    with mock.patch("chisurf.core.settings.ai_settings.get_api_settings", return_value=settings), \
         mock.patch("requests.post", side_effect=AssertionError("network hit")):
        assert _call_llm("prompt") is None

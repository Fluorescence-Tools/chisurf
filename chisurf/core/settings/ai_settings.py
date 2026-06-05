from __future__ import annotations

import os
import json
import pathlib

from chisurf.core.settings.path_utils import get_path

# Mistral-only defaults
DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

DEFAULT_SETTINGS = {
    "base_url": DEFAULT_BASE_URL,
    "model": "mistral-small-latest",
    "api_key": "",
    "text_embed_model": "mistral-embed",
    "code_embed_model": "codestral-embed",
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 4096,
    "provider": "local_llm",
    "hf_repo": "",
    "hf_file": "",
    "persistent_ingest": True,
    "log_level": 20,
    "rag_debug": False,
    "rag_debug_max_hits": 8,
    "rag_debug_preview_chars": 220,
    "graphrag_enable": True,
    "graphrag_mode": "per_chunk",
}

# Chat models
MISTRAL_CHAT_MODELS = [
    "mistral-small-latest",
    "mistral-medium-latest", 
    "mistral-large-latest",
]

# Embedding models
MISTRAL_EMBED_MODELS = {
    "text": ["mistral-embed"],
    "code": ["codestral-embed"],
}


def _get_settings_path() -> pathlib.Path:
    """Return path to AI settings JSON file."""
    return get_path('settings') / 'ai_api_settings.json'


def get_api_settings() -> dict:
    """
    Load AI API settings from JSON file with env var fallbacks.
    Priority: settings file > environment variables
    """
    settings_path = _get_settings_path()

    if settings_path.is_file():
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            if isinstance(settings, dict):
                return {**DEFAULT_SETTINGS, **settings}
        except Exception:
            pass

    return {**DEFAULT_SETTINGS}


def save_api_settings(settings: dict) -> bool:
    """Save AI API settings to JSON file."""
    settings_path = _get_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception:
        return False


def get_api_key() -> str:
    """Get Mistral API key."""
    settings = get_api_settings()
    api_key = settings.get('api_key', '')

    if not api_key:
        api_key = os.environ.get('MISTRAL_API_KEY', '')

    return api_key.strip()


def get_base_url() -> str:
    """Get base URL for Mistral API."""
    settings = get_api_settings()
    base_url = settings.get('base_url', '')

    if not base_url:
        base_url = DEFAULT_BASE_URL

    return base_url.strip().rstrip('/')


def get_model() -> str:
    """Get chat model name."""
    settings = get_api_settings()
    model = settings.get('model', '')

    if not model:
        model = "mistral-small-latest"

    return model.strip()


def get_text_embed_model() -> str:
    """Get text embedding model name."""
    settings = get_api_settings()
    return settings.get('text_embed_model', 'mistral-embed')


def get_code_embed_model() -> str:
    """Get code embedding model name."""
    settings = get_api_settings()
    return settings.get('code_embed_model', 'codestral-embed')


def get_embed_model() -> str:
    """Get default embedding model (text)."""
    return get_text_embed_model()


def get_temperature() -> float:
    """Get temperature setting."""
    settings = get_api_settings()
    return float(settings.get('temperature', 0.3))


def get_top_p() -> float:
    """Get top_p setting."""
    settings = get_api_settings()
    return float(settings.get('top_p', 0.9))


def get_max_tokens() -> int:
    """Get max tokens setting."""
    settings = get_api_settings()
    return int(settings.get('max_tokens', 4096))


def get_provider() -> str:
    """Get the LLM provider."""
    settings = get_api_settings()
    return settings.get('provider', 'local_llm')


def get_hf_repo() -> str:
    """Get the Hugging Face repo for local LLM."""
    settings = get_api_settings()
    return settings.get('hf_repo', '')


def get_hf_file() -> str:
    """Get the Hugging Face file for local LLM."""
    settings = get_api_settings()
    return settings.get('hf_file', '')


def get_persistent_ingest() -> bool:
    """Get the persistent ingest setting."""
    settings = get_api_settings()
    return bool(settings.get('persistent_ingest', True))


def get_log_level() -> int:
    """Get the log level for Chato/AI."""
    settings = get_api_settings()
    return int(settings.get('log_level', 20))


def get_rag_debug() -> bool:
    """Get the RAG debug setting."""
    settings = get_api_settings()
    return bool(settings.get('rag_debug', False))


def get_rag_debug_max_hits() -> int:
    """Get the RAG debug max hits."""
    settings = get_api_settings()
    return int(settings.get('rag_debug_max_hits', 8))


def get_rag_debug_preview_chars() -> int:
    """Get the RAG debug preview characters."""
    settings = get_api_settings()
    return int(settings.get('rag_debug_preview_chars', 220))


def get_graphrag_enable() -> bool:
    """Get the GraphRAG enable setting."""
    settings = get_api_settings()
    return bool(settings.get('graphrag_enable', True))


def get_graphrag_mode() -> str:
    """Get the GraphRAG mode."""
    settings = get_api_settings()
    return settings.get('graphrag_mode', 'per_chunk')

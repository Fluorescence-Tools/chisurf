from __future__ import annotations

import json
import logging
import os
import pathlib

from chisurf.core.settings.path_utils import get_path

_LOG = logging.getLogger(__name__)

# Provider definitions: display_name -> (key, default_base_url, api_key_url, env_var)
PROVIDERS: dict[str, tuple[str, str, str, str]] = {
    "OpenAI (ChatGPT)": ("openai", "https://api.openai.com/v1", "https://platform.openai.com/api-keys", "OPENAI_API_KEY"),
    "Mistral": ("mistral", "https://api.mistral.ai/v1", "https://console.mistral.ai/api-keys/", "MISTRAL_API_KEY"),
    "Local (Ollama, LMStudio, ...)": ("local", "http://localhost:11434/v1", "", ""),
    "Custom (OpenAI-compatible)": ("custom", "", "", ""),
}

# Default settings for each provider
DEFAULT_PROVIDER_SETTINGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": "",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
        "api_key": "",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
    },
    "local": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2",
        "api_key": "",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
    },
    "custom": {
        "base_url": "",
        "model": "",
        "api_key": "",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
    },
}


def _get_settings_path() -> pathlib.Path:
    """Return path to AI settings JSON file."""
    return get_path('settings') / 'ai_api_settings.json'


def get_api_settings(provider: str | None = None) -> dict:
    """
    Load AI API settings from JSON file with env var fallbacks.

    Priority: settings file > environment variables > defaults.
    
    Args:
        provider: If specified, return settings for this provider.
                 If None, return settings for the currently selected provider.
    
    Returns:
        dict: Settings for the specified/provider
    """
    settings_path = _get_settings_path()

    # Load all settings from file
    all_settings = {}
    if settings_path.is_file():
        try:
            with open(settings_path, encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    all_settings = data
        except Exception:
            pass

    # Determine which provider to get settings for
    if provider is None:
        # Get currently selected provider from settings or default to openai
        provider = all_settings.get('selected_provider', 'openai')
    
    # Get settings for the specified provider, falling back to defaults
    provider_settings = all_settings.get(provider, {})
    
    # Merge with defaults for this provider
    defaults = DEFAULT_PROVIDER_SETTINGS.get(provider, {})
    result = {**defaults, **provider_settings}
    
    # Ensure we have the provider field
    result['provider'] = provider
    
    return result


def save_api_settings(settings: dict, provider: str | None = None) -> bool:
    """Save AI API settings for a specific provider to JSON file."""
    settings_path = _get_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing settings
    all_settings = {}
    if settings_path.is_file():
        try:
            with open(settings_path, encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    all_settings = data
        except Exception:
            pass

    # Determine which provider to save settings for
    if provider is None:
        provider = settings.get('provider', 'openai')
    
    # Update settings for this provider
    all_settings[provider] = {
        k: v for k, v in settings.items() 
        if k in ['base_url', 'model', 'api_key', 'temperature', 'top_p', 'max_tokens']
    }
    
    # Update selected provider
    all_settings['selected_provider'] = provider

    try:
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(all_settings, f, indent=2)
        return True
    except Exception:
        return False


def get_api_key() -> str:
    """Get API key from settings or environment variable."""
    settings = get_api_settings()
    api_key = settings.get('api_key', '')

    if not api_key:
        provider = settings.get('provider', '')
        for _display, (key, _url, _api_url, env_var) in PROVIDERS.items():
            if key == provider and env_var:
                api_key = os.environ.get(env_var, '')
                break

    return api_key.strip()


def get_base_url() -> str:
    """Get base URL for API."""
    settings = get_api_settings()
    return settings.get('base_url', '').strip().rstrip('/')


def get_model() -> str:
    """Get chat model name."""
    settings = get_api_settings()
    return settings.get('model', '').strip()


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
    """Get the LLM provider key."""
    settings = get_api_settings()
    return settings.get('provider', 'openai')


def get_available_providers() -> list[str]:
    """Get list of providers that have saved settings."""
    settings_path = _get_settings_path()
    if not settings_path.is_file():
        return list(DEFAULT_PROVIDER_SETTINGS.keys())
    
    try:
        with open(settings_path, encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                # Return providers that have settings plus the selected one
                providers = set(data.keys()) - {'selected_provider'}
                selected = data.get('selected_provider', 'openai')
                providers.add(selected)
                return list(providers)
    except Exception:
        pass
    
    return list(DEFAULT_PROVIDER_SETTINGS.keys())

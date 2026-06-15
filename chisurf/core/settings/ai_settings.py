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

LEGACY_PROVIDER_KEYS = {
    "openai_api": "openai",
    "mistral_api": "mistral",
    "local_llm": "local",
}

# Default settings for each provider
DEFAULT_PROVIDER_SETTINGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "text_model": "gpt-4o",
        "model": "gpt-4o",
        "image_model": "gpt-image-2",
        "api_key": "",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "text_model": "mistral-small-latest",
        "model": "mistral-small-latest",
        "image_model": "mistral-medium-latest",
        "api_key": "",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
    },
    "local": {
        "base_url": "http://localhost:11434/v1",
        "text_model": "llama3.2",
        "model": "llama3.2",
        "image_model": "",
        "api_key": "",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
    },
    "custom": {
        "base_url": "",
        "text_model": "",
        "model": "",
        "image_model": "",
        "api_key": "",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
    },
}

DEFAULT_SETTINGS = {
    "provider": "openai",
    **DEFAULT_PROVIDER_SETTINGS["openai"],
}


def normalize_provider_key(provider: str | None) -> str:
    """Return the canonical provider key for saved or legacy settings."""
    if not provider:
        return "openai"
    return LEGACY_PROVIDER_KEYS.get(provider, provider)


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

    if 'base_url' in all_settings and not any(
        key in all_settings for key in DEFAULT_PROVIDER_SETTINGS
    ):
        legacy_provider = normalize_provider_key(all_settings.get('provider'))
        all_settings = {
            legacy_provider: all_settings,
            'selected_provider': legacy_provider,
        }

    # Determine which provider to get settings for
    if provider is None:
        # Get currently selected provider from settings or default to openai
        provider = all_settings.get('selected_provider', 'openai')
    provider = normalize_provider_key(provider)

    # Get settings for the specified provider, falling back to defaults
    provider_settings = all_settings.get(provider, {})

    # Merge with defaults for this provider
    defaults = DEFAULT_PROVIDER_SETTINGS.get(provider, {})
    result = {**defaults, **provider_settings}
    text_model = str(
        provider_settings.get('text_model')
        or provider_settings.get('model')
        or defaults.get('text_model')
        or defaults.get('model')
        or ''
    ).strip()
    image_model = str(result.get('image_model') or '').strip()
    result['text_model'] = text_model
    result['model'] = text_model
    result['image_model'] = image_model

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
    provider = normalize_provider_key(provider)

    text_model = str(settings.get('text_model') or settings.get('model') or '').strip()
    settings = {**settings, 'text_model': text_model, 'model': text_model}

    # Update settings for this provider
    all_settings[provider] = {
        k: v for k, v in settings.items()
        if k in ['base_url', 'text_model', 'model', 'image_model', 'api_key', 'temperature', 'top_p', 'max_tokens']
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
    return settings.get('text_model', settings.get('model', '')).strip()


def get_image_model() -> str:
    """Get image generation model name."""
    settings = get_api_settings()
    model = settings.get('image_model', '').strip()
    provider = settings.get('provider', 'openai')
    if model and (provider != 'openai' or _looks_like_image_model(model)):
        return model
    if provider == 'openai':
        return 'gpt-image-2'
    return ''


def _looks_like_image_model(model: str) -> bool:
    """Return whether a model name looks suitable for image generation."""
    normalized = model.lower()
    return any(marker in normalized for marker in ('image', 'dall-e', 'gpt-image'))


def model_capabilities(model: dict | str, provider: str = "") -> set[str]:
    """Infer model capabilities from endpoint metadata and model id."""
    if isinstance(model, str):
        model_id = model
        metadata = {}
    else:
        model_id = str(model.get('id') or model.get('name') or '')
        metadata = model
    normalized_id = model_id.lower()
    capabilities = set()

    metadata_values = _flatten_model_metadata(metadata)
    if any(_metadata_mentions(value, 'image') for value in metadata_values):
        capabilities.add('image')
    if any(_metadata_mentions(value, 'text', 'chat', 'completion', 'response') for value in metadata_values):
        capabilities.add('text')

    if _looks_like_image_model(model_id):
        capabilities.add('image')

    if provider == 'mistral' and normalized_id.startswith('mistral-'):
        capabilities.update({'text', 'image'})

    non_text_markers = (
        'embedding',
        'embed',
        'moderation',
        'whisper',
        'tts',
        'audio',
        'realtime',
        'transcribe',
        'video',
    )
    is_non_text_model = any(marker in normalized_id for marker in non_text_markers) or any(
        any(marker in value for marker in non_text_markers)
        for value in metadata_values
    )
    if not capabilities and not is_non_text_model:
        capabilities.add('text')
    if 'image' in capabilities and not is_non_text_model:
        capabilities.add('text') if provider == 'mistral' else None
    return capabilities


def split_models_by_capability(models: list[dict | str], provider: str = "") -> tuple[list[str], list[str]]:
    """Split model metadata into text and image model id lists."""
    text_models = []
    image_models = []
    for model in models:
        model_id = model if isinstance(model, str) else model.get('id') or model.get('name') or ''
        model_id = str(model_id).strip()
        if not model_id:
            continue
        capabilities = model_capabilities(model, provider=provider)
        if 'text' in capabilities:
            text_models.append(model_id)
        if 'image' in capabilities:
            image_models.append(model_id)
    return sorted(set(text_models)), sorted(set(image_models))


def _flatten_model_metadata(value):
    """Yield scalar metadata values from nested model metadata."""
    if isinstance(value, dict):
        for nested in value.values():
            yield from _flatten_model_metadata(nested)
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _flatten_model_metadata(nested)
    else:
        yield str(value).lower()


def _metadata_mentions(value: str, *needles: str) -> bool:
    """Return whether a metadata value mentions any capability keyword."""
    return any(needle in value for needle in needles)


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

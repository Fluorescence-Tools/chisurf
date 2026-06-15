"""Tests for AI model capability classification."""

from chisurf.core.settings import ai_settings


def test_split_models_by_capability_uses_model_ids():
    """Model ids should separate common text and image models."""
    text_models, image_models = ai_settings.split_models_by_capability(
        [
            {"id": "gpt-4o"},
            {"id": "gpt-image-2"},
            {"id": "text-embedding-3-large"},
            {"id": "dall-e-3"},
        ],
        provider="openai",
    )

    assert "gpt-4o" in text_models
    assert "text-embedding-3-large" not in text_models
    assert image_models == ["dall-e-3", "gpt-image-2"]


def test_split_models_by_capability_uses_metadata():
    """Capability metadata should override weak id heuristics."""
    text_models, image_models = ai_settings.split_models_by_capability(
        [
            {"id": "provider-chat", "capabilities": {"chat": True}},
            {"id": "provider-picture", "modalities": ["image"]},
            {"id": "provider-embed", "capabilities": ["embeddings"]},
        ],
        provider="custom",
    )

    assert "provider-chat" in text_models
    assert "provider-picture" in image_models
    assert "provider-embed" not in text_models


def test_mistral_models_can_drive_image_generation_agents():
    """Mistral chat models can be used with the Mistral image-generation tool."""
    text_models, image_models = ai_settings.split_models_by_capability(
        [{"id": "mistral-medium-latest"}],
        provider="mistral",
    )

    assert text_models == ["mistral-medium-latest"]
    assert image_models == ["mistral-medium-latest"]

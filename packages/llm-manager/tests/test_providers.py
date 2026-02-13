"""Tests for provider registry and factory."""

import pytest

from llm_manager.providers import (
    PROVIDER_CONFIGS,
    ProviderConfig,
    get_provider_config,
    list_providers,
    register_provider,
)


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_list_providers(self):
        """Test listing all providers."""
        providers = list_providers()
        assert len(providers) > 0
        provider_names = [p.name for p in providers]
        assert "openai" in provider_names
        assert "anthropic" in provider_names
        assert "ollama" in provider_names

    def test_get_provider_config(self):
        """Test getting provider configuration."""
        config = get_provider_config("openai")
        assert config is not None
        assert config.name == "openai"
        assert config.display_name == "OpenAI"
        assert config.api_key_env_var == "OPENAI_API_KEY"

    def test_get_provider_config_not_found(self):
        """Test getting non-existent provider."""
        config = get_provider_config("nonexistent")
        assert config is None

    def test_get_provider_config_case_insensitive(self):
        """Test case-insensitive provider lookup."""
        config = get_provider_config("OpenAI")
        assert config is not None
        assert config.name == "openai"

    def test_provider_capabilities(self):
        """Test provider capability flags."""
        openai = get_provider_config("openai")
        assert openai.supports_language is True
        assert openai.supports_embedding is True
        assert openai.supports_vision is True
        assert openai.supports_tools is True
        assert openai.supports_speech_to_text is True
        assert openai.supports_text_to_speech is True
        assert openai.is_local is False

        ollama = get_provider_config("ollama")
        assert ollama.is_local is True
        assert ollama.default_base_url == "http://localhost:11434"

    def test_register_provider(self):
        """Test registering a custom provider."""
        custom = ProviderConfig(
            name="custom_provider",
            display_name="Custom Provider",
            env_var_prefix="CUSTOM_",
            api_key_env_var="CUSTOM_API_KEY",
        )
        register_provider(custom)

        retrieved = get_provider_config("custom_provider")
        assert retrieved is not None
        assert retrieved.name == "custom_provider"
        assert retrieved.display_name == "Custom Provider"

    def test_provider_config_model(self):
        """Test ProviderConfig model."""
        config = ProviderConfig(
            name="test",
            display_name="Test Provider",
            env_var_prefix="TEST_",
            api_key_env_var="TEST_API_KEY",
            supports_vision=True,
            supports_tools=False,
            is_local=True,
        )
        assert config.name == "test"
        assert config.supports_vision is True
        assert config.supports_tools is False
        assert config.is_local is True
        # Defaults
        assert config.supports_language is True
        assert config.supports_embedding is True


class TestProviderConfigs:
    """Tests for predefined provider configurations."""

    @pytest.mark.parametrize("provider_name", [
        "openai", "anthropic", "google", "groq", "mistral",
        "ollama", "llamacpp", "openrouter", "together", "deepseek",
    ])
    def test_provider_exists(self, provider_name):
        """Test that common providers are registered."""
        assert provider_name in PROVIDER_CONFIGS

    def test_openai_config(self):
        """Test OpenAI configuration."""
        config = PROVIDER_CONFIGS["openai"]
        assert config.api_key_env_var == "OPENAI_API_KEY"
        assert config.supports_speech_to_text is True
        assert config.supports_text_to_speech is True

    def test_anthropic_config(self):
        """Test Anthropic configuration."""
        config = PROVIDER_CONFIGS["anthropic"]
        assert config.api_key_env_var == "ANTHROPIC_API_KEY"
        assert config.supports_speech_to_text is False
        assert config.supports_text_to_speech is False

    def test_local_providers(self):
        """Test local provider configurations."""
        ollama = PROVIDER_CONFIGS["ollama"]
        assert ollama.is_local is True
        assert ollama.default_base_url is not None

        llamacpp = PROVIDER_CONFIGS["llamacpp"]
        assert llamacpp.is_local is True
        assert llamacpp.default_base_url is not None

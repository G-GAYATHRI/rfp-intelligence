import pytest
import config
from src.agents.llm import _provider, _require_config, LLMConfigError


def test_explicit_anthropic_provider_uses_anthropic_key(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "LLM_API_KEY", None)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "ant-test")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "openai-test")
    monkeypatch.setattr(config, "LLM_MODEL", "claude-test")
    assert _require_config() == ("anthropic", "ant-test", "claude-test")


def test_unsupported_provider_is_actionable(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "unknown")
    with pytest.raises(LLMConfigError, match="Unsupported LLM_PROVIDER"):
        _provider()

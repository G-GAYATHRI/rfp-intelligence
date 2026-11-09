
"""Provider-explicit LLM calls with actionable configuration errors.

Supported providers:
- Anthropic Claude
- OpenAI
- Google Gemini (OpenAI-compatible API)
"""

from __future__ import annotations

import json
import os
import re

import httpx
import config


class LLMConfigError(RuntimeError):
    """Raised when the LLM configuration is invalid."""


class LLMCallError(RuntimeError):
    """Raised when an LLM request fails."""


GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai"
)


def _provider() -> str:
    """Resolve and validate the configured LLM provider."""

    explicit = (
        config.LLM_PROVIDER or ""
    ).strip().lower()

    if explicit:
        supported = {"anthropic", "openai", "gemini"}

        if explicit not in supported:
            raise LLMConfigError(
                f"Unsupported LLM_PROVIDER '{explicit}'. "
                "Use anthropic, openai, or gemini."
            )

        return explicit

    # Automatic detection only when no provider is configured.
    if config.GEMINI_API_KEY:
        return "gemini"

    if config.ANTHROPIC_API_KEY:
        return "anthropic"

    if config.OPENAI_API_KEY:
        return "openai"

    generic = config.LLM_API_KEY or ""

    if generic.startswith("sk-ant"):
        return "anthropic"

    if generic:
        return "openai"

    raise LLMConfigError(
        "Set LLM_PROVIDER and the matching API key in .env."
    )

       

 

def _require_config() -> tuple[str, str, str]:
    """Validate provider, API key, and model configuration."""

    provider = _provider()

    provider_keys = {
        "anthropic": config.ANTHROPIC_API_KEY,
        "openai": config.OPENAI_API_KEY,
        "gemini": config.GEMINI_API_KEY,
    }

    env_names = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }

    key = provider_keys[provider] or config.LLM_API_KEY
    model = config.LLM_MODEL

    if not key:
        raise LLMConfigError(
            f"LLM_PROVIDER={provider} requires "
            f"{env_names[provider]} (or LLM_API_KEY)."
        )

    if not model:
        raise LLMConfigError(
            "LLM_MODEL is required. Set a model supported "
            "by the selected provider."
        )

    return provider, key, model


def parse_json(text: str) -> dict:
    """Parse a JSON object from an LLM response."""

    raw = (text or "").strip()

    if not raw:
        raise LLMCallError("Model returned an empty response.")

    # Remove Markdown code fences.
    if raw.startswith("```"):
        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        raw = re.sub(r"\s*```$", "", raw)

    try:
        value = json.loads(raw)

    except json.JSONDecodeError:
        # Fallback: find a JSON object in surrounding text.
        match = re.search(r"\{.*\}", raw, re.S)

        if not match:
            raise LLMCallError(
                "Model did not return a valid JSON object."
            )

        try:
            value = json.loads(match.group(0))

        except json.JSONDecodeError as exc:
            raise LLMCallError(
                f"Invalid JSON from model: {exc}"
            ) from exc

    if not isinstance(value, dict):
        raise LLMCallError(
            "Model response must be a JSON object."
        )

    return value


def chat_json(system: str, user: str) -> tuple[dict, dict]:
    """Send a request to the configured LLM provider."""

    provider, key, model = _require_config()

    if provider == "anthropic":
        return _anthropic(system, user, key, model)

    if provider == "gemini":
        return _openai_compatible(
            system,
            user,
            key,
            model,
            base_url=GEMINI_BASE_URL,
        )

    return _openai_compatible(
        system,
        user,
        key,
        model,
        base_url=config.LLM_BASE_URL,
    )


def _anthropic(
    system: str,
    user: str,
    key: str,
    model: str,
) -> tuple[dict, dict]:
    """Call Anthropic Claude."""

    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=key,
            timeout=90,
        )

        msg = client.messages.create(
            model=model,
            max_tokens=3000,
            temperature=0,
            system=system,
            messages=[
                {"role": "user", "content": user}
            ],
        )

        content = "".join(
            getattr(block, "text", "")
            for block in msg.content
        )

        usage = {
            "input_tokens": getattr(
                msg.usage, "input_tokens", None
            ),
            "output_tokens": getattr(
                msg.usage, "output_tokens", None
            ),
        }

        return parse_json(content), usage

    except LLMCallError:
        raise

    except Exception as exc:
        raise LLMCallError(
            f"Anthropic API request failed: {exc}"
        ) from exc


def _openai_compatible(
    system: str,
    user: str,
    key: str,
    model: str,
    base_url: str,
) -> tuple[dict, dict]:
    """Call an OpenAI-compatible chat completion API.

    Used for OpenAI and Google's Gemini API.
    """

    url = f"{base_url.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    try:
        response = httpx.post(
            url,
            headers=headers,
            json=payload,
            timeout=90,
        )

        response.raise_for_status()

        body = response.json()

        content = (
            body["choices"][0]["message"]["content"]
        )

        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )

        if not isinstance(content, str):
            raise LLMCallError(
                "Provider returned an unsupported response format."
            )

        usage_data = body.get("usage") or {}

        usage = {
            "input_tokens": usage_data.get("prompt_tokens"),
            "output_tokens": usage_data.get("completion_tokens"),
        }

        return parse_json(content), usage

    except LLMCallError:
        raise

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code

        try:
            details = exc.response.json()
        except ValueError:
            details = exc.response.text[:1000]

        raise LLMCallError(
            f"LLM API returned HTTP {status}: {details}"
        ) from exc

    except httpx.RequestError as exc:
        raise LLMCallError(
            f"Could not connect to LLM API: {exc}"
        ) from exc

    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMCallError(
            f"Unexpected LLM API response: {exc}"
        ) from exc

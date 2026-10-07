"""
LLM Provider Service
---------------------
Thin wrapper around the OpenAI-compatible Chat Completions API.
All credentials are read from environment variables — nothing is hardcoded.

Provider: OpenRouter (https://openrouter.ai) via the OpenAI Python SDK.

Required environment variable
------------------------------
    ECO_LLM_API_KEY      OpenRouter API key

Optional environment variables (defaults shown)
-------------------------------------------------
    ECO_LLM_BASE_URL     https://openrouter.ai/api/v1
    ECO_LLM_MODEL        openai/gpt-4o-mini
    ECO_LLM_TIMEOUT      30   (seconds)
    ECO_LLM_MAX_TOKENS   512
    ECO_LLM_TEMPERATURE  0.3

Any OpenRouter-listed model ID works for ECO_LLM_MODEL.
Swap to a different OpenAI-compatible provider by changing ECO_LLM_BASE_URL.
"""

from __future__ import annotations

import os
from typing import Optional

# Auto-load .env if present (never raises if file is missing)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # env vars already set in shell take precedence
except ImportError:
    pass   # python-dotenv optional; rely on shell env if not installed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL    = "openai/gpt-4o-mini"


class LLMConfig:
    """Reads LLM configuration from environment at access time (no caching)."""

    @property
    def api_key(self) -> str:
        return os.environ.get("ECO_LLM_API_KEY", "").strip()

    @property
    def base_url(self) -> str:
        return os.environ.get("ECO_LLM_BASE_URL", _DEFAULT_BASE_URL).strip()

    @property
    def model(self) -> str:
        return os.environ.get("ECO_LLM_MODEL", _DEFAULT_MODEL).strip()

    @property
    def timeout(self) -> float:
        try:
            return float(os.environ.get("ECO_LLM_TIMEOUT", "30"))
        except ValueError:
            return 30.0

    @property
    def max_tokens(self) -> int:
        try:
            return int(os.environ.get("ECO_LLM_MAX_TOKENS", "512"))
        except ValueError:
            return 512

    @property
    def temperature(self) -> float:
        try:
            return float(os.environ.get("ECO_LLM_TEMPERATURE", "0.3"))
        except ValueError:
            return 0.3

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


# Module-level singleton
config = LLMConfig()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_completion(
    system_prompt: str,
    user_message: str,
    cfg: Optional[LLMConfig] = None,
) -> str:
    """
    Send a chat completion request to OpenRouter (or any OpenAI-compatible API).

    Parameters
    ----------
    system_prompt : str   Instructions / context for the LLM.
    user_message  : str   The user's question.
    cfg           : LLMConfig, optional  Override config (uses singleton by default).

    Returns
    -------
    str   The assistant's reply text.

    Raises
    ------
    LLMNotConfiguredError  ECO_LLM_API_KEY is missing.
    LLMRequestError        API call failed for any reason.
    """
    if cfg is None:
        cfg = config

    if not cfg.is_configured:
        raise LLMNotConfiguredError(
            "ECO_LLM_API_KEY is not set. "
            "Add it to your .env file to enable the AI assistant."
        )

    try:
        import openai

        client = openai.OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )

        response = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            timeout=cfg.timeout,
        )

        return response.choices[0].message.content or ""

    except LLMNotConfiguredError:
        raise
    except ImportError as exc:
        raise LLMRequestError(
            "openai package is not installed. Run: pip install openai"
        ) from exc
    except openai.AuthenticationError as exc:
        raise LLMRequestError(
            "Authentication failed. Check your ECO_LLM_API_KEY."
        ) from exc
    except openai.RateLimitError as exc:
        raise LLMRequestError(
            "API rate limit reached. Please wait and try again."
        ) from exc
    except openai.APIConnectionError as exc:
        raise LLMRequestError(
            "Cannot reach the LLM API. Check your internet connection."
        ) from exc
    except openai.APITimeoutError as exc:
        raise LLMRequestError(
            f"Request timed out after {cfg.timeout}s. Please try again."
        ) from exc
    except openai.APIStatusError as exc:
        raise LLMRequestError(
            f"API error {exc.status_code}: {exc.message}"
        ) from exc
    except Exception as exc:
        raise LLMRequestError(f"Unexpected error calling LLM: {exc}") from exc


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMNotConfiguredError(RuntimeError):
    """Raised when ECO_LLM_API_KEY is not present in the environment."""


class LLMRequestError(RuntimeError):
    """Raised when the API call fails for any reason."""

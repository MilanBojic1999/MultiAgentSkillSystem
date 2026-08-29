import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_TOKENS = 4096
DEFAULT_API_KEY_ENV = "LLM_KEY"


@lru_cache(maxsize=None)
def create_llm(
    model: Optional[str] = None,
    url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: Optional[bool] = False
) -> ChatOpenAI:
    """Build (and cache) a ChatOpenAI client for an OpenAI-compatible endpoint.

    model/url fall back to LLM_MODEL/LLM_URL from the environment. api_key_env
    names the environment variable holding the API key (defaults to LLM_KEY),
    so per-agent configs (Phase 4.3) can point at a different key without
    ever storing the key itself in agent_config.json. Instances are cached by
    the exact parameter tuple, so repeated calls with the same config reuse
    one client.
    """
    model = model or os.getenv("LLM_MODEL")
    url = url or os.getenv("LLM_URL")
    api_key_env = api_key_env or DEFAULT_API_KEY_ENV
    api_key = os.getenv(api_key_env, "")

    missing = [name for name, value in (("LLM_MODEL", model), ("LLM_URL", url)) if not value]
    if missing:
        raise EnvironmentError(
            f"Missing required LLM configuration: {', '.join(missing)}. "
            f"Set them in .env, or pass model=/url= explicitly."
        )

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=url,
        temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
        max_tokens=DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
        streaming=streaming
    )

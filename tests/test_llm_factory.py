"""llm_factory.create_llm — env fallbacks, caching, defaults, missing-var errors.

Constructing ChatOpenAI is offline-safe (no connection is opened), so these run
without a network. ``fresh_llm_cache`` clears the lru_cache around each test.
"""

import pytest


def test_env_fallback_uses_conftest_dummies(fresh_llm_cache):
    from llm_factory import create_llm

    assert create_llm().model_name == "test-model"


def test_missing_model_and_url_raise_naming_them(fresh_llm_cache, monkeypatch):
    from llm_factory import create_llm

    # load_dotenv already ran at import, so delete from the live environment.
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_URL", raising=False)
    with pytest.raises(EnvironmentError, match="LLM_MODEL, LLM_URL"):
        create_llm(model=None, url=None)


def test_same_args_are_cached_identical(fresh_llm_cache):
    from llm_factory import create_llm

    assert create_llm() is create_llm()


def test_different_temperature_is_a_different_object(fresh_llm_cache):
    from llm_factory import create_llm

    assert create_llm(temperature=0.1) is not create_llm(temperature=0.2)


def test_defaults_are_applied(fresh_llm_cache):
    from llm_factory import create_llm

    client = create_llm()
    assert client.temperature == 0.9
    assert client.max_tokens == 4096


def test_explicit_zero_temperature_is_respected(fresh_llm_cache):
    # Guards against an `or`-style falsy bug that would coerce 0.0 to the default.
    from llm_factory import create_llm

    assert create_llm(temperature=0.0).temperature == 0.0

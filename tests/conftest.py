"""Shared test infrastructure.

The env vars must be set *before* the first pipeline import, and the CWD must
be the repo root *before* any module calls ``load_skills()``. Both are
import-time concerns (see ``TESTING_GUIDE.md`` → "The import-time side-effect
problem"), so both happen here at module top level, not inside fixtures.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Must precede every pipeline import. Port 9 (discard) makes any accidental
# live LLM call fail instantly instead of silently hitting a real server.
# ``load_dotenv`` (called at import in several modules) never overrides an
# already-set variable, so these beat a developer's real ``.env``.
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("LLM_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("LLM_KEY", "test-key")
os.environ.setdefault("CONFIG_PATH", str(REPO_ROOT / "agents" / "agent_config.json"))

# skill_loader.root_dir is CWD-relative; agents/ modules call load_skills() at
# import, which can happen during collection — so pin CWD here.
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))  # redundant after `pip install -e .`, harmless

import pytest  # noqa: E402


@pytest.fixture()
def fresh_llm_cache():
    """Clear the ``create_llm`` lru_cache around a test that varies env vars."""
    from llm_factory import create_llm

    create_llm.cache_clear()
    yield
    create_llm.cache_clear()

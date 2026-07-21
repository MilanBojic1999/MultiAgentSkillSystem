"""config_loader runs at import, so error paths need a fresh import.

Reloading config_loader orphans the already-imported ``agents`` package (it
keeps its old reference) — so these tests are self-contained and never mix with
graph tests. The fixture restores the original module afterwards so later tests
that touch the real config are unaffected.
"""

import importlib
import json
import sys

import pytest


@pytest.fixture()
def load_config(monkeypatch):
    original = sys.modules.get("config_loader")

    def _load(path):
        monkeypatch.setenv("CONFIG_PATH", str(path))
        sys.modules.pop("config_loader", None)
        return importlib.import_module("config_loader")

    yield _load

    sys.modules.pop("config_loader", None)
    if original is not None:
        sys.modules["config_loader"] = original


def _write(tmp_path, data):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    return p


def test_missing_file_raises_naming_path_and_example(load_config, tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match=r"\.env\.example"):
        load_config(missing)


def test_duplicate_mcp_owner_raises_naming_both_agents(load_config, tmp_path):
    data = {
        "alice": {"description": "a", "tools": [], "mcp_servers": {"shared": "u1"}},
        "bob": {"description": "b", "tools": [], "mcp_servers": {"shared": "u2"}},
    }
    with pytest.raises(ValueError, match="multiple owners"):
        load_config(_write(tmp_path, data))


def test_valid_minimal_config_is_keyed_by_agent(load_config, tmp_path):
    data = {"solo": {"description": "d", "tools": [], "mcp_servers": {}}}
    mod = load_config(_write(tmp_path, data))
    assert set(mod.AGENT_CONFIG) == {"solo"}


def test_real_repo_config_loads_non_empty(load_config):
    from pathlib import Path

    repo_config = Path(__file__).resolve().parent.parent / "agents" / "agent_config.json"
    mod = load_config(repo_config)
    assert mod.AGENT_CONFIG
    assert {"mathematician", "researcher", "writer"} <= set(mod.AGENT_CONFIG)

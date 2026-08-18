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


def test_real_repo_config_reads_back_non_default_example(load_config):
    """The shipped config's ``execution`` example reads back through the helper.

    Researcher declares ``max_attempts=3``; the other agents have no block and
    normalize to the default of 2 — proving a per-agent override needs no
    Python changes.
    """
    from pathlib import Path

    repo_config = Path(__file__).resolve().parent.parent / "agents" / "agent_config.json"
    mod = load_config(repo_config)
    assert mod.get_max_attempts("researcher") == 3
    assert mod.get_max_attempts("mathematician") == 2
    assert mod.get_max_attempts("writer") == 2


# ---------------------------------------------------------------------------
# Slice 2: execution block validation and normalization
# ---------------------------------------------------------------------------

def test_absent_execution_block_defaults_to_two(load_config, tmp_path):
    """No ``execution`` block → ``get_max_attempts`` normalizes to 2."""
    data = {"solo": {"description": "d", "tools": [], "mcp_servers": {}}}
    mod = load_config(_write(tmp_path, data))
    assert mod.get_max_attempts("solo") == 2


@pytest.mark.parametrize("value", [1, 2, 10])
def test_valid_max_attempts_values_load(load_config, tmp_path, value):
    """Boundary values 1, 2, and 10 are accepted and read back."""
    data = {
        "solo": {
            "description": "d",
            "tools": [],
            "mcp_servers": {},
            "execution": {"max_attempts": value},
        }
    }
    mod = load_config(_write(tmp_path, data))
    assert mod.get_max_attempts("solo") == value


@pytest.mark.parametrize("value", [0, 11, -1, 3.5, "2"])
def test_invalid_max_attempts_values_fail(load_config, tmp_path, value):
    """Out-of-range, float, and string values fail at load, naming the field."""
    data = {
        "solo": {
            "description": "d",
            "tools": [],
            "mcp_servers": {},
            "execution": {"max_attempts": value},
        }
    }
    with pytest.raises(ValueError, match=r"execution\.max_attempts"):
        load_config(_write(tmp_path, data))


@pytest.mark.parametrize("value", [True, False])
def test_boolean_max_attempts_fails(load_config, tmp_path, value):
    """Booleans are invalid even though ``bool`` subclasses ``int``."""
    data = {
        "solo": {
            "description": "d",
            "tools": [],
            "mcp_servers": {},
            "execution": {"max_attempts": value},
        }
    }
    with pytest.raises(ValueError, match="boolean"):
        load_config(_write(tmp_path, data))


def test_unknown_execution_key_fails_naming_agent_and_key(load_config, tmp_path):
    """A typo inside ``execution`` fails at load, naming agent and offending key."""
    data = {
        "solo": {
            "description": "d",
            "tools": [],
            "mcp_servers": {},
            "execution": {"max_attempts": 2, "retry": True},
        }
    }
    with pytest.raises(ValueError) as excinfo:
        load_config(_write(tmp_path, data))
    assert "solo" in str(excinfo.value)
    assert "retry" in str(excinfo.value)


def test_non_dict_execution_block_fails_naming_agent(load_config, tmp_path):
    data = {
        "solo": {
            "description": "d",
            "tools": [],
            "mcp_servers": {},
            "execution": 5,
        }
    }
    with pytest.raises(TypeError, match="solo"):
        load_config(_write(tmp_path, data))

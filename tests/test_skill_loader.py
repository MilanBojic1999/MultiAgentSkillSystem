"""skill_loader — frontmatter parsing against a temp skills tree."""

import pytest

from skill_loader import _split_frontmatter, load_skills, load_skills_body


@pytest.fixture()
def skills_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("skill_loader.root_dir", str(tmp_path))

    def make(name, frontmatter, body):
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}")

    return make


def test_wellformed_file_is_parsed_and_indexed(skills_dir):
    skills_dir("alpha", "name: alpha\ndescription: does things", "  body text  ")
    index, pairs = load_skills()
    assert index["alpha"]["description"] == "does things"
    assert pairs["alpha"].endswith("alpha")
    assert load_skills_body(pairs, "alpha") == "body text"


def test_missing_frontmatter_raises_naming_the_path():
    with pytest.raises(ValueError, match="frontmatter"):
        _split_frontmatter("no frontmatter here", "some/path/SKILL.md")


def test_body_containing_separators_is_not_truncated(skills_dir):
    body = "intro line\n---\nmiddle\n---\nend line"
    skills_dir("beta", "name: beta\ndescription: d", body)
    _, pairs = load_skills()
    loaded = load_skills_body(pairs, "beta")
    assert "intro line" in loaded
    assert "end line" in loaded
    assert loaded.count("---") == 2


def test_directory_without_skill_md_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr("skill_loader.root_dir", str(tmp_path))
    (tmp_path / "empty").mkdir()
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "SKILL.md").write_text("---\nname: real\ndescription: d\n---\nbody")
    index, _ = load_skills()
    assert set(index) == {"real"}


def test_unknown_skill_body_raises(skills_dir):
    skills_dir("alpha", "name: alpha\ndescription: d", "body")
    _, pairs = load_skills()
    with pytest.raises(ValueError, match="not found"):
        load_skills_body(pairs, "does-not-exist")


def test_real_repo_skills_load(monkeypatch):
    # No fixture: exercise the checked-in skills/ tree (CWD pinned by conftest).
    monkeypatch.setattr("skill_loader.root_dir", "skills")
    index, _ = load_skills()
    assert {"answer-writer", "frontend-design", "roll-dice", "yotta-researcher"} <= set(index)

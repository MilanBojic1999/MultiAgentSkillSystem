
import os
import re
import yaml

root_dir = "skills"

# Matches a leading YAML frontmatter block: ---\n<yaml>\n---
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)", re.DOTALL)


def _split_frontmatter(skill_content: str, skill_path: str) -> tuple[str, str]:
    match = _FRONTMATTER_RE.match(skill_content)
    if not match:
        raise ValueError(f"'{skill_path}' does not start with a '---' YAML frontmatter block.")
    yaml_str, body = match.groups()
    return yaml_str.strip(), body.strip()


def load_skills():
    skills = {}
    skills_directory_pairs = {}
    for skills_dir in os.listdir(root_dir):
        # Find only SKILL.md file in directory
        skill_path = os.path.join(root_dir, skills_dir, "SKILL.md")
        if os.path.isfile(skill_path):
            with open(skill_path, "r") as f:
                skill_content = f.read()
                yaml_str, _ = _split_frontmatter(skill_content, skill_path)
                skill_data = yaml.safe_load(yaml_str)
                skill_name = skill_data.get("name")
                skills[skill_name] = skill_data
                skills_directory_pairs[skill_name] = os.path.join(root_dir, skills_dir)


    return skills, skills_directory_pairs

def load_skills_body(skills_directory_pairs, skill_name):
    skill_dir = skills_directory_pairs.get(skill_name)
    if not skill_dir:
        raise ValueError(f"Skill '{skill_name}' not found in directory pairs.")


    skill_path = os.path.join(skill_dir, "SKILL.md")
    if not os.path.isfile(skill_path):
        raise ValueError(f"SKILL.md not found for skill '{skill_name}' at path: {skill_path}")

    with open(skill_path, "r") as f:
        skill_content = f.read()
        _, body_str = _split_frontmatter(skill_content, skill_path)
        return body_str
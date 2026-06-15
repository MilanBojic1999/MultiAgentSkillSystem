
import os
import yaml

root_dir = "skills"

def load_skills():
    skills = {}
    skills_directory_pairs = {}
    for skills_dir in os.listdir(root_dir):
        # Find only SKILL.md file in directory
        skill_path = os.path.join(root_dir, skills_dir, "SKILL.md")
        if os.path.isfile(skill_path):
            with open(skill_path, "r") as f:
                skill_content = f.read()
                yaml_str = skill_content.split("---")[1].strip()
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
        body_str = skill_content.split("---", maxsplit=2)[2].strip()
        return body_str
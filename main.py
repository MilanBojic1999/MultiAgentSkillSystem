import os
import sys
import yaml
from openai import OpenAI

import subprocess
import io
import contextlib
import pwd
import grp

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
        # print(skill_content.split("---"))
        body_str = skill_content.split("---")[2].strip()
        return body_str


START_PROMPT = """
You are an **Orchestrator Agent** responsible for decomposing user requests into actionable steps using a set of available skills.

## Your Task
Given a user query, select the appropriate skill(s) and create a clear, step-by-step execution plan.

## Available Skills
{skills}

## Instructions
1. Analyze the user query to understand the goal and any sub-tasks
2. Select one or more skills from the available skills list that can solve the problem
3. Order the skills logically — earlier steps may produce outputs that later steps depend on
4. Write a focused, specific query for each skill that describes exactly what it needs to do
5. If no available skill can address the user query, respond: `No suitable skill found for this request.`

## Output Format
For each step, output the skill name and its query using the following structure:

```
<skill>skill_name</skill>
<query>Specific description of what this skill should do, including any relevant context or inputs from previous steps</query>
```

Repeat this block for each step in sequence.

## Rules
- Only use skills from the provided list — do not invent or assume unlisted skills exist
- Each `<query>` should be self-contained and unambiguous
- If a step depends on output from a prior step, reference it explicitly (e.g., *"Using the data extracted in Step 1..."*)
- Prefer the fewest steps necessary to solve the problem
""".strip()

url = "https://api.deepseek.com"
# modelName = "deepseek-v4-flash"
modelName = "deepseek-v4-pro"
api_key = "sk-"

client = OpenAI(
    api_key=api_key,
    base_url=url,
)

def call_llm(system_prompt, user_query):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",  "content": user_query},
    ]

    response = client.chat.completions.create(
            model=modelName,
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=64048,
            extra_body={
                "repetition_penalty": 1,
                "top_k": 40
            },
        )

    return response.choices[0].message.model_dump()['content']


SANDBOX_USER = "nobody"  # or a dedicated "sandbox" user you create

def _drop_privileges():
    """Called in child process before exec — drops to unprivileged user."""

    if os.getuid() != 0:
        return
    try:
        user = pwd.getpwnam(SANDBOX_USER)
        os.setgid(user.pw_gid)              # drop group first
        os.setgroups([])                     # strip supplementary groups
        os.setuid(user.pw_uid)              # then drop user
    except (KeyError, PermissionError) as e:
        raise RuntimeError(f"Failed to drop privileges to '{SANDBOX_USER}': {e}")

def parse_and_run_bash(command):
    if "```bash" in command and "```" in command:
        bash_command = command.split("```bash")[1].split("```")[0].strip()
        print(f"Executing bash command: {bash_command}")
        result = subprocess.run(
            bash_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            preexec_fn=_drop_privileges      # drops privileges in child before exec
        )
        return result.stdout.strip(), result.stderr.strip()
    return "", "No bash block found"

def parse_and_run_python(code):
    if "```python" in code:
        python_code = code.split("```python")[1].lstrip("3\n").strip()
        python_code = python_code.split("```")[0].strip()
        print(f"Executing python code: {python_code}")

        # Write code to a temp file and run it in a subprocess with dropped privs
        # exec() is in-process, so we fork out instead
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
                preexec_fn=_drop_privileges  # drops privileges in child before exec
            )
            return result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return "", "Timeout: execution exceeded 10 seconds"
        finally:
            os.unlink(tmp_path)             # always clean up the temp file

    return "", "No python block found"

if __name__ == "__main__":
    skills, skills_directory_pairs = load_skills()
    print(f"Loaded skills: {list(skills.keys())}")
    print(f"Content of a skill: {skills['roll-dice']}")
    skills_str = "\n".join([f"{i+1}. {k}: {skills[k]['description']}" for i,k in enumerate(skills.keys())]) 

    # print(skills_str)

    print(START_PROMPT.format(skills=skills_str))
    print("-"*50)

    response = call_llm(START_PROMPT.format(skills=skills_str), "Design a website for a digital album sticker collector, where users can view, organize, and trade their digital stickers. The website should have a visually appealing interface, user profiles, and a marketplace for trading stickers.")

    print("LLM Response:")
    print(response)

    # Extract skill names and queris from response
    reponse_skills = []
    for block in response.split("</query>"):
        if "<skill>" in block and "<query>" in block:
            skill_name = block.split("<skill>")[1].split("</skill>")[0].strip()
            query = block.split("<query>")[1].strip()
            reponse_skills.append((skill_name, query))

    
    print("Extracted Skills and Queries:")
    print(reponse_skills)

    for skill_name, query in reponse_skills:
        skill_body = load_skills_body(skills_directory_pairs, skill_name)
        print(f"Executing skill: {skill_name} with query: {query}")
        print(f"Skill body: {skill_body}")
        print("-"*50)
        print("-"*50)
        new_response = call_llm(skill_body, query)
        print(f"Response from skill '{skill_name}': {new_response}")
        print("-"*50)
        stdout, stderr = "", ""
        if "```bash" in new_response:
            stdout, stderr = parse_and_run_bash(new_response)
            print(f"Output from bash command:\n{stdout}\nErrors:\n{stderr}")
        elif "```python" in new_response:
            stdout, stderr = parse_and_run_python(new_response)
            print(f"Output from python code:\n{stdout}\nErrors:\n{stderr}")


        with open(f"{skill_name}_output.txt", "w") as f:
            f.write(f"Response from skill '{skill_name}':\n{new_response}\n")
            if stdout:
                f.write(f"\nOutput from execution:\n{stdout}\n")
            if stderr:
                f.write(f"\nErrors from execution:\n{stderr}\n")
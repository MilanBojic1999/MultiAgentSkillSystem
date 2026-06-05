"""
Bash execution tool for LangGraph agents.
Safely executes bash commands inside a subprocess with a timeout and no-privilege model.
"""

import subprocess
from langchain_core.tools import tool


@tool
def run_bash(command: str, timeout: int = 10) -> str:
    """
    Execute a bash command and return stdout + stderr.

    Args:
        command: The bash command to execute (e.g. 'echo $((RANDOM % 6 + 1))')
        timeout: Max seconds to wait before killing the process (default 10).

    Returns:
        Combined stdout and stderr output.
    """
    import os

    def drop_privileges():
        """Drop to 'nobody' user if running as root."""
        if os.getuid() != 0:
            return
        try:
            import pwd
            nobody = pwd.getpwnam("nobody")
            os.setgid(nobody.pw_gid)
            os.setgroups([])
            os.setuid(nobody.pw_uid)
        except (KeyError, PermissionError):
            pass  # best-effort, ignore if nobody doesn't exist

    try:
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=drop_privileges,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        return {
            "stdout": out,
            "stderr": err,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Error: command timed out after {timeout}s",
            "returncode": 0
        }

@tool
def run_bash_with_approval(command: str) -> str:
    """
    Execute a bash command with human feedback.

    Args:
        command: The bash command to execute (e.g. 'echo $((RANDOM % 6 + 1))')
    Returns:
        Combined stdout and stderr output.
    """
    print(f"\n⚠️  Agent wants to run:\n  $ {command}")
    confirm = input("Approve? [y/N]: ").strip().lower()
    if confirm == "y":
        return run_bash(command)
    return {
            "stdout": "",
            "stderr": "Command rejected by user.",
            "returncode": 0
        }
"""Terminal command execution tools for AugAgent."""

import subprocess
import shlex
from pydantic import BaseModel, Field
from augagent.tools import aug_tool

ALLOWED_COMMANDS = {"ls", "cat", "echo", "python", "node", "dir", "type", "git"}

class RunCommandArgs(BaseModel):
    command: str = Field(description="The shell command to execute.")
    cwd: str = Field(description="Current working directory to run the command in.", default=".")

@aug_tool(args_schema=RunCommandArgs)
def run_terminal_command(command: str, cwd: str) -> str:
    """Run a terminal command securely in a subprocess and return its output."""
    try:
        parsed_cmd = shlex.split(command)
        if not parsed_cmd or parsed_cmd[0] not in ALLOWED_COMMANDS:
            return f"Error: Command not permitted by allowlist. Attempted to run: {parsed_cmd[0] if parsed_cmd else 'empty command'}"

        result = subprocess.run(
            parsed_cmd,
            cwd=cwd,
            shell=False,
            capture_output=True,
            text=True,
            timeout=120
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        
        if result.returncode == 0:
            return f"Command succeeded:\n{out}"
        else:
            return f"Command failed (Code {result.returncode}):\nSTDOUT: {out}\nSTDERR: {err}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120 seconds."
    except Exception as e:
        return f"Error executing command: {e}"

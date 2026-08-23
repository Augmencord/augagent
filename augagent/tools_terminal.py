"""Terminal command execution tools for AugAgent."""

import subprocess
import shlex
from pydantic import BaseModel, Field
from augagent.tools import aug_tool

ALLOWED_COMMANDS = {"ls", "cat", "echo", "python", "node", "dir", "type", "git"}
BLOCKED_PATHS = {".env", ".ssh", "id_rsa", "secrets"}
MAX_OUTPUT_SIZE = 1_000_000

class RunCommandArgs(BaseModel):
    command: str = Field(description="The shell command to execute.")
    cwd: str = Field(description="Current working directory to run the command in.", default=".")

@aug_tool(args_schema=RunCommandArgs)  # type: ignore
def run_terminal_command(command: str, cwd: str) -> str:
    """Run a terminal command securely in a subprocess and return its output."""
    try:
        if not command.strip():
            return "Error: Empty command"
            
        import sys
        if sys.platform == "win32":
            # Use powershell on Windows without cmd.exe interpreting pipes
            exec_cmd = ["powershell.exe", "-NoProfile", "-Command", command]
            use_shell = False
        else:
            exec_cmd = command
            use_shell = True
            
        result = subprocess.run(
            exec_cmd,
            cwd=cwd,
            shell=use_shell,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        out = result.stdout
        err = result.stderr
        
        if len(out) > MAX_OUTPUT_SIZE:
            out = out[:MAX_OUTPUT_SIZE] + f"\n... [Output truncated at {MAX_OUTPUT_SIZE} bytes]"
        if len(err) > MAX_OUTPUT_SIZE:
            err = err[:MAX_OUTPUT_SIZE] + f"\n... [Error output truncated at {MAX_OUTPUT_SIZE} bytes]"
            
        out = out.strip()
        err = err.strip()
        
        if result.returncode == 0:
            return f"Command succeeded:\n{out}"
        else:
            return f"Command failed (Code {result.returncode}):\nSTDOUT: {out}\nSTDERR: {err}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {e}"

class ListWindowsArgs(BaseModel):
    app_name_filter: str = Field(default="", description="Optional filter to only return windows matching this app name (e.g., 'Google Chrome' or 'Code')")

@aug_tool(args_schema=ListWindowsArgs)  # type: ignore
def list_open_windows(app_name_filter: str = "") -> str:
    """List all visible open window titles on the desktop. Highly useful for finding active browser tabs or applications without terminal scripts."""
    import sys
    if sys.platform != "win32":
        return "Error: This tool is currently only implemented for Windows."
    
    import ctypes
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    GetWindowText = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible
    
    titles = []
    def foreach_window(hwnd, lParam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLength(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                GetWindowText(hwnd, buff, length + 1)
                titles.append(buff.value)
        return True
    
    EnumWindows(EnumWindowsProc(foreach_window), 0)
    
    if app_name_filter:
        filter_lower = app_name_filter.lower()
        titles = [t for t in titles if filter_lower in t.lower()]
        
    if not titles:
        return f"No open windows found matching '{app_name_filter}'."
    return "Open Windows:\n" + "\n".join(f"- {t}" for t in titles)

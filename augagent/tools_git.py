"""Git integration tools for AugAgent with structured output and sandbox validation.

Tools:
- git_status: Working tree and staging status.
- git_diff: Unified diff of working directory or staged index.
- git_commit: Record changes to repository.
- git_log: Commit history log.
- git_branch: List or create branches.
- git_add: Stage files for commit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field

from augagent.tools import aug_tool
from augagent.diff_engine import resolve_sandboxed_path, get_workspace_root


def _run_git_command(args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Execute git command safely in target directory without shell=True."""
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 1, "", "Error: 'git' executable not found on system PATH."
    except Exception as exc:
        return 1, "", f"Git execution error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# 1. git_status
# ═══════════════════════════════════════════════════════════════════════════

class GitStatusArgs(BaseModel):
    cwd: str = Field(default=".", description="Sandbox directory containing the git repository.")


@aug_tool(args_schema=GitStatusArgs)
def git_status(cwd: str = ".") -> str:
    """Show the working tree status with staged, unstaged, and untracked files."""
    try:
        target_dir = resolve_sandboxed_path(cwd)
    except PermissionError as pe:
        return f"Error: {pe}"
    except Exception as exc:
        return f"Error resolving cwd '{cwd}': {exc}"

    if not target_dir.exists() or not target_dir.is_dir():
        return f"Error: Directory '{cwd}' does not exist."

    code, stdout, stderr = _run_git_command(["status", "--porcelain=v1", "-b"], target_dir)
    if code != 0:
        return f"Git error ({code}): {stderr or stdout}"

    lines = stdout.splitlines() if stdout else []
    branch_header = lines[0] if lines and lines[0].startswith("## ") else "## unknown"
    branch_name = branch_header[3:].split("...")[0].strip()

    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []

    for line in lines[1:]:
        if len(line) < 3:
            continue
        index_status = line[0]
        worktree_status = line[1]
        filepath = line[3:].strip()

        if index_status in ("A", "M", "D", "R", "C"):
            staged.append(f"[{index_status}] {filepath}")
        if worktree_status in ("M", "D"):
            unstaged.append(f"[{worktree_status}] {filepath}")
        elif index_status == "?" and worktree_status == "?":
            untracked.append(filepath)

    output = [f"Branch: {branch_name}"]
    if not staged and not unstaged and not untracked:
        output.append("Working tree clean (nothing to commit).")
    else:
        if staged:
            output.append(f"Staged ({len(staged)} files):")
            output.extend([f"  {f}" for f in staged])
        if unstaged:
            output.append(f"Unstaged changes ({len(unstaged)} files):")
            output.extend([f"  {f}" for f in unstaged])
        if untracked:
            output.append(f"Untracked files ({len(untracked)} files):")
            output.extend([f"  ? {f}" for f in untracked])

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════
# 2. git_diff
# ═══════════════════════════════════════════════════════════════════════════

class GitDiffArgs(BaseModel):
    cwd: str = Field(default=".", description="Sandbox directory containing the git repository.")
    staged: bool = Field(default=False, description="If True, shows diff of staged changes against HEAD.")
    filepath: str = Field(default="", description="Optional relative file path to restrict diff output.")


@aug_tool(args_schema=GitDiffArgs)
def git_diff(cwd: str = ".", staged: bool = False, filepath: str = "") -> str:
    """Show changes between commits, commit and working tree, etc."""
    try:
        target_dir = resolve_sandboxed_path(cwd)
    except PermissionError as pe:
        return f"Error: {pe}"

    cmd = ["diff"]
    if staged:
        cmd.append("--cached")

    if filepath:
        try:
            target_file = resolve_sandboxed_path(filepath, workspace_root=target_dir)
            rel_file = target_file.relative_to(target_dir).as_posix()
            cmd.extend(["--", rel_file])
        except PermissionError as pe:
            return f"Error: {pe}"
        except Exception:
            cmd.extend(["--", filepath])

    code, stdout, stderr = _run_git_command(cmd, target_dir)
    if code != 0:
        return f"Git error ({code}): {stderr or stdout}"

    if not stdout.strip():
        scope = "staged changes" if staged else "working tree changes"
        target_info = f" for '{filepath}'" if filepath else ""
        return f"No {scope}{target_info}."

    return stdout


# ═══════════════════════════════════════════════════════════════════════════
# 3. git_commit
# ═══════════════════════════════════════════════════════════════════════════

class GitCommitArgs(BaseModel):
    message: str = Field(description="Commit message describing changes.")
    cwd: str = Field(default=".", description="Sandbox directory containing the git repository.")


@aug_tool(args_schema=GitCommitArgs)
def git_commit(message: str, cwd: str = ".") -> str:
    """Record staged changes to the repository."""
    if not message.strip():
        return "Error: Commit message cannot be empty."

    try:
        target_dir = resolve_sandboxed_path(cwd)
    except PermissionError as pe:
        return f"Error: {pe}"

    code, stdout, stderr = _run_git_command(["commit", "-m", message], target_dir)
    if code != 0:
        return f"Git commit failed ({code}): {stderr or stdout}"

    return stdout or "Commit successful."


# ═══════════════════════════════════════════════════════════════════════════
# 4. git_log
# ═══════════════════════════════════════════════════════════════════════════

class GitLogArgs(BaseModel):
    cwd: str = Field(default=".", description="Sandbox directory containing the git repository.")
    max_count: int = Field(default=10, description="Maximum number of commits to list.")


@aug_tool(args_schema=GitLogArgs)
def git_log(cwd: str = ".", max_count: int = 10) -> str:
    """Show commit logs in a structured, concise format."""
    try:
        target_dir = resolve_sandboxed_path(cwd)
    except PermissionError as pe:
        return f"Error: {pe}"

    cmd = [
        "log",
        f"-n{max(1, max_count)}",
        "--pretty=format:%h | %ad | %an | %s",
        "--date=short",
    ]
    code, stdout, stderr = _run_git_command(cmd, target_dir)
    if code != 0:
        return f"Git error ({code}): {stderr or stdout}"

    if not stdout.strip():
        return "No commits found in repository."

    lines = stdout.splitlines()
    formatted = ["Hash    | Date       | Author          | Message", "-" * 60]
    formatted.extend(lines)
    return "\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════════
# 5. git_branch
# ═══════════════════════════════════════════════════════════════════════════

class GitBranchArgs(BaseModel):
    cwd: str = Field(default=".", description="Sandbox directory containing the git repository.")
    action: str = Field(default="list", description="'list' to show branches, 'create' to create a new branch.")
    branch_name: str = Field(default="", description="Name of branch to create when action is 'create'.")


@aug_tool(args_schema=GitBranchArgs)
def git_branch(cwd: str = ".", action: str = "list", branch_name: str = "") -> str:
    """List local branches or create a new branch."""
    try:
        target_dir = resolve_sandboxed_path(cwd)
    except PermissionError as pe:
        return f"Error: {pe}"

    act = action.strip().lower()
    if act == "create":
        if not branch_name.strip():
            return "Error: branch_name must be provided when action is 'create'."
        code, stdout, stderr = _run_git_command(["branch", branch_name.strip()], target_dir)
        if code != 0:
            return f"Failed to create branch '{branch_name}': {stderr or stdout}"
        return f"Successfully created branch '{branch_name}'."
    else:
        code, stdout, stderr = _run_git_command(["branch"], target_dir)
        if code != 0:
            return f"Git error ({code}): {stderr or stdout}"
        return stdout or "No branches found."


# ═══════════════════════════════════════════════════════════════════════════
# 6. git_add
# ═══════════════════════════════════════════════════════════════════════════

class GitAddArgs(BaseModel):
    files: str = Field(default=".", description="Path or space-separated paths to stage (e.g. '.' or 'src/main.py').")
    cwd: str = Field(default=".", description="Sandbox directory containing the git repository.")


@aug_tool(args_schema=GitAddArgs)
def git_add(files: str = ".", cwd: str = ".") -> str:
    """Add file contents to the staging index."""
    try:
        target_dir = resolve_sandboxed_path(cwd)
    except PermissionError as pe:
        return f"Error: {pe}"

    file_items = files.split() if files.strip() else ["."]
    # Validate each item to ensure no escape outside sandbox
    for item in file_items:
        try:
            resolve_sandboxed_path(item, workspace_root=target_dir)
        except PermissionError as pe:
            return f"Error: Cannot stage '{item}': {pe}"

    code, stdout, stderr = _run_git_command(["add"] + file_items, target_dir)
    if code != 0:
        return f"Git add failed ({code}): {stderr or stdout}"

    return f"Successfully staged {files}."

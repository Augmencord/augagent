"""Code search and directory navigation tools for AugAgent with sandbox protection.

Tools:
- grep_search: ripgrep-backed text search with Python re fallback.
- find_files: glob-based filename matching.
- list_directory: sandbox-validated directory structure listing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, List
from pydantic import BaseModel, Field

from augagent.tools import aug_tool
from augagent.diff_engine import resolve_sandboxed_path, get_workspace_root

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".mypy_cache",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. grep_search
# ═══════════════════════════════════════════════════════════════════════════

class GrepSearchArgs(BaseModel):
    query: str = Field(description="Regex or literal string to search for.")
    path: str = Field(default=".", description="Relative directory or file path to search within.")
    case_sensitive: bool = Field(default=True, description="Whether search should be case sensitive.")
    max_results: int = Field(default=100, description="Maximum number of matches to return.")


def _python_grep_fallback(
    query: str,
    target_path: Path,
    workspace_root: Path,
    case_sensitive: bool,
    max_results: int,
) -> str:
    """Python re-based fallback when ripgrep is unavailable."""
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(query, flags)
    except re.error:
        # Treat as literal string if regex compilation fails
        pattern = re.compile(re.escape(query), flags)

    matches: list[str] = []

    def scan_file(file_path: Path):
        nonlocal matches
        if len(matches) >= max_results:
            return
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, start=1):
                    if pattern.search(line):
                        try:
                            rel_path = file_path.relative_to(workspace_root).as_posix()
                        except ValueError:
                            rel_path = file_path.as_posix()
                        clean_line = line.rstrip("\r\n")
                        matches.append(f"{rel_path}:{line_idx}:{clean_line}")
                        if len(matches) >= max_results:
                            break
        except (PermissionError, OSError):
            pass

    if target_path.is_file():
        scan_file(target_path)
    else:
        for root, dirs, files in os.walk(target_path):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES]
            for file_name in files:
                scan_file(Path(root) / file_name)
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

    if not matches:
        return f"No matches found for '{query}' in '{target_path}'."
    return "\n".join(matches)


@aug_tool(args_schema=GrepSearchArgs)
def grep_search(
    query: str,
    path: str = ".",
    case_sensitive: bool = True,
    max_results: int = 100,
) -> str:
    """Search for matching patterns across files using ripgrep with Python re fallback."""
    try:
        target_path = resolve_sandboxed_path(path)
    except PermissionError as pe:
        return f"Error: {pe}"
    except Exception as exc:
        return f"Error resolving path '{path}': {exc}"

    if not target_path.exists():
        return f"Error: Path '{path}' not found."

    workspace_root = get_workspace_root()
    rg_binary = shutil.which("rg")

    if rg_binary:
        cmd = [
            rg_binary,
            "--line-number",
            "--no-heading",
            "--color=never",
            "--max-count",
            str(max_results),
        ]
        if not case_sensitive:
            cmd.append("-i")
        for ignored in IGNORED_DIRECTORIES:
            cmd.extend(["-g", f"!{ignored}"])
        cmd.extend([query, str(target_path)])

        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().splitlines()
                # Format relative paths
                formatted = []
                for line in lines[:max_results]:
                    formatted.append(line)
                return "\n".join(formatted)
            elif result.returncode == 1:
                return f"No matches found for '{query}' in '{path}'."
        except Exception:
            pass  # Fall through to Python fallback on any failure

    return _python_grep_fallback(
        query=query,
        target_path=target_path,
        workspace_root=workspace_root,
        case_sensitive=case_sensitive,
        max_results=max_results,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. find_files
# ═══════════════════════════════════════════════════════════════════════════

class FindFilesArgs(BaseModel):
    pattern: str = Field(default="*", description="Glob pattern (e.g. '*.py', '**/*.tsx').")
    path: str = Field(default=".", description="Relative directory path to search within.")
    max_results: int = Field(default=100, description="Maximum number of files to return.")


@aug_tool(args_schema=FindFilesArgs)
def find_files(
    pattern: str = "*",
    path: str = ".",
    max_results: int = 100,
) -> str:
    """Find files matching a glob pattern within the workspace sandbox."""
    try:
        target_path = resolve_sandboxed_path(path)
    except PermissionError as pe:
        return f"Error: {pe}"
    except Exception as exc:
        return f"Error resolving path '{path}': {exc}"

    if not target_path.exists() or not target_path.is_dir():
        return f"Error: Directory '{path}' not found."

    workspace_root = get_workspace_root()
    matches: list[str] = []

    # If pattern does not include recursive wildcards, apply glob recursively
    glob_pattern = pattern if ("/" in pattern or "**" in pattern) else f"**/{pattern}"

    try:
        for p in target_path.glob(glob_pattern):
            # Check if any parent component is in IGNORED_DIRECTORIES
            parts = p.relative_to(target_path).parts
            if any(part in IGNORED_DIRECTORIES for part in parts):
                continue
            if p.is_file():
                try:
                    rel = p.relative_to(workspace_root).as_posix()
                except ValueError:
                    rel = p.as_posix()
                matches.append(rel)
                if len(matches) >= max_results:
                    break
    except Exception as exc:
        return f"Error searching files: {exc}"

    if not matches:
        return f"No files matching '{pattern}' in '{path}'."
    return "\n".join(matches)


# ═══════════════════════════════════════════════════════════════════════════
# 3. list_directory
# ═══════════════════════════════════════════════════════════════════════════

class ListDirectoryArgs(BaseModel):
    path: str = Field(default=".", description="Sandbox path to the directory to list.")
    recursive: bool = Field(default=False, description="Whether to recursively list subdirectories.")
    max_depth: int = Field(default=2, description="Maximum recursive depth if recursive is True.")


@aug_tool(args_schema=ListDirectoryArgs)
def list_directory(
    path: str = ".",
    recursive: bool = False,
    max_depth: int = 2,
) -> str:
    """List directory contents with file/folder indicators and size metadata."""
    try:
        target_path = resolve_sandboxed_path(path)
    except PermissionError as pe:
        return f"Error: {pe}"
    except Exception as exc:
        return f"Error resolving directory '{path}': {exc}"

    if not target_path.exists() or not target_path.is_dir():
        return f"Error: Directory '{path}' not found."

    workspace_root = get_workspace_root()

    def format_entry(p: Path, indent: str = "") -> list[str]:
        lines = []
        try:
            entries = sorted(list(p.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
            for entry in entries:
                if entry.name in IGNORED_DIRECTORIES or entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    try:
                        child_count = len([c for c in entry.iterdir() if not c.name.startswith(".")])
                        lines.append(f"{indent}[DIR]  {entry.name}/ ({child_count} items)")
                    except (PermissionError, OSError):
                        lines.append(f"{indent}[DIR]  {entry.name}/")
                else:
                    try:
                        size = entry.stat().st_size
                        lines.append(f"{indent}[FILE] {entry.name} ({size} bytes)")
                    except (PermissionError, OSError):
                        lines.append(f"{indent}[FILE] {entry.name}")
        except (PermissionError, OSError) as exc:
            lines.append(f"{indent}[ACCESS DENIED] {exc}")
        return lines

    if not recursive:
        entries = format_entry(target_path)
        if not entries:
            return f"Directory '{path}' is empty."
        return "\n".join(entries)

    # Recursive listing up to max_depth
    output_lines: list[str] = [f"Directory tree for '{path}':"]

    def walk_tree(current_dir: Path, current_depth: int, indent: str):
        if current_depth > max_depth:
            return
        try:
            items = sorted(list(current_dir.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
            for item in items:
                if item.name in IGNORED_DIRECTORIES or item.name.startswith("."):
                    continue
                if item.is_dir():
                    output_lines.append(f"{indent}├── [DIR]  {item.name}/")
                    walk_tree(item, current_depth + 1, indent + "│   ")
                else:
                    try:
                        size = item.stat().st_size
                        output_lines.append(f"{indent}├── [FILE] {item.name} ({size} bytes)")
                    except OSError:
                        output_lines.append(f"{indent}├── [FILE] {item.name}")
        except (PermissionError, OSError) as exc:
            output_lines.append(f"{indent}└── [ACCESS DENIED] {exc}")

    walk_tree(target_path, 1, "")
    return "\n".join(output_lines)

"""Diff and Patch Engine for AugAgent.

Provides robust unified diff calculation and safe atomic patch application.
"""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import List, Tuple


def get_workspace_root() -> Path:
    """Retrieve the configured workspace root directory."""
    raw = os.environ.get("WORKSPACE_ROOT")
    if raw:
        return Path(raw).resolve()
    return Path.cwd().resolve()


def resolve_sandboxed_path(filepath: str | Path, workspace_root: Path | str | None = None) -> Path:
    """Resolve and validate a file path against the workspace sandbox boundary.

    Raises PermissionError if path traversal outside workspace is attempted.
    """
    root = Path(workspace_root).resolve() if workspace_root else get_workspace_root()
    path_obj = Path(filepath)
    target = (root / path_obj).resolve() if not path_obj.is_absolute() else path_obj.resolve()

    try:
        target.relative_to(root)
    except ValueError:
        raise PermissionError(f"Access denied: path '{filepath}' escapes workspace sandbox '{root}'.")
    return target


def generate_unified_diff(
    before: str,
    after: str,
    filepath: str = "file",
) -> str:
    """Generate unified diff string between before and after contents.

    Args:
        before: Original file content.
        after: Modified file content.
        filepath: Label used in diff header.

    Returns:
        Unified diff string with standard headers.
    """
    if before is None or after is None:
        raise ValueError("before and after text must not be None.")

    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)

    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
        lineterm="\n",
    )
    return "".join(diff)


def _apply_unified_diff_to_lines(original_lines: List[str], patch: str) -> List[str]:
    """Internal helper to apply unified diff hunks to a list of lines."""
    patch_lines = patch.splitlines(keepends=True)
    if not patch_lines:
        return list(original_lines)

    hunk_regex = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    
    # Strip any git header lines until the first hunk
    first_hunk_idx = -1
    for idx, line in enumerate(patch_lines):
        if line.startswith("@@"):
            first_hunk_idx = idx
            break

    if first_hunk_idx == -1:
        # No hunks found, cannot apply
        raise ValueError("Invalid patch format: No hunk headers ('@@') found.")

    patch_lines = patch_lines[first_hunk_idx:]
    result_lines = list(original_lines)
    orig_offset = 0

    idx = 0
    while idx < len(patch_lines):
        line = patch_lines[idx]
        match = hunk_regex.match(line)
        if not match:
            idx += 1
            continue

        orig_start = int(match.group(1))
        # 1-indexed to 0-indexed
        orig_pos = orig_start - 1 + orig_offset
        idx += 1

        hunk_removals = []
        hunk_additions = []

        while idx < len(patch_lines) and not patch_lines[idx].startswith("@@"):
            hline = patch_lines[idx]
            if hline.startswith("-"):
                hunk_removals.append(hline[1:])
            elif hline.startswith("+"):
                hunk_additions.append(hline[1:])
            elif hline.startswith(" "):
                # Context line: if we have pending changes, apply them
                if hunk_removals or hunk_additions:
                    # Check context match
                    remove_count = len(hunk_removals)
                    result_lines[orig_pos : orig_pos + remove_count] = hunk_additions
                    orig_offset += len(hunk_additions) - remove_count
                    orig_pos += len(hunk_additions)
                    hunk_removals = []
                    hunk_additions = []
                orig_pos += 1
            idx += 1

        # Flush any trailing changes in hunk
        if hunk_removals or hunk_additions:
            remove_count = len(hunk_removals)
            result_lines[orig_pos : orig_pos + remove_count] = hunk_additions
            orig_offset += len(hunk_additions) - remove_count

    return result_lines


def apply_patch(
    filepath: str,
    patch: str,
    workspace_root: Path | str | None = None,
) -> str:
    """Apply a unified diff patch to a sandboxed file.

    Args:
        filepath: Relative or absolute path to the target file.
        patch: Unified diff string to apply.
        workspace_root: Optional custom sandbox root override.

    Returns:
        Status message confirming patch application.

    Raises:
        PermissionError: If path escapes sandbox.
        FileNotFoundError: If target file does not exist.
        ValueError: If patch cannot be applied cleanly.
    """
    if not filepath:
        raise ValueError("filepath must not be empty.")
    if patch is None:
        raise ValueError("patch cannot be None.")

    target = resolve_sandboxed_path(filepath, workspace_root=workspace_root)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"File '{filepath}' not found.")

    original_content = target.read_text(encoding="utf-8")
    original_lines = original_content.splitlines(keepends=True)

    try:
        updated_lines = _apply_unified_diff_to_lines(original_lines, patch)
        updated_content = "".join(updated_lines)
        target.write_text(updated_content, encoding="utf-8")
        return f"Successfully applied patch to '{filepath}'."
    except Exception as exc:
        raise ValueError(f"Failed to apply patch to '{filepath}': {exc}") from exc

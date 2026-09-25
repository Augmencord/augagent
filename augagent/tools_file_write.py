"""Advanced file writing and editing tools with sandbox validation and diff generation.

Tools:
- write_file: Write or overwrite file contents in the sandbox.
- create_file: Create a new file only; error if file already exists.
- replace_in_file: Targeted substring/block replacement with diff return.
- insert_at_line: Line insertion at specific 1-indexed position.
- delete_lines: Line range deletion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from augagent.tools import aug_tool
from augagent.diff_engine import (
    generate_unified_diff,
    resolve_sandboxed_path,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. write_file
# ═══════════════════════════════════════════════════════════════════════════

class WriteFileArgs(BaseModel):
    filepath: str = Field(description="Sandbox-relative or absolute path to the file.")
    content: str = Field(description="Content to write or overwrite into the file.")


@aug_tool(args_schema=WriteFileArgs)
def write_file(filepath: str, content: str) -> str:
    """Create or overwrite a file within the workspace sandbox."""
    try:
        target = resolve_sandboxed_path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{filepath}'."
    except PermissionError as pe:
        return f"Error: {pe}"
    except Exception as exc:
        return f"Error writing to file '{filepath}': {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. create_file
# ═══════════════════════════════════════════════════════════════════════════

class CreateFileArgs(BaseModel):
    filepath: str = Field(description="Sandbox-relative or absolute path to the file to create.")
    content: str = Field(default="", description="Initial content to write into the new file.")


@aug_tool(args_schema=CreateFileArgs)
def create_file(filepath: str, content: str = "") -> str:
    """Create a new file with optional content. Fails if the file already exists."""
    try:
        target = resolve_sandboxed_path(filepath)
        if target.exists():
            return f"Error: File '{filepath}' already exists. Use write_file or replace_in_file to edit it."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully created new file '{filepath}'."
    except PermissionError as pe:
        return f"Error: {pe}"
    except Exception as exc:
        return f"Error creating file '{filepath}': {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. replace_in_file
# ═══════════════════════════════════════════════════════════════════════════

class ReplaceInFileArgs(BaseModel):
    filepath: str = Field(description="Path to the file to edit.")
    target: str = Field(description="Exact block of code or text to find and replace.")
    replacement: str = Field(description="New content to substitute in place of target.")
    start_line: Optional[int] = Field(default=None, description="Optional 1-indexed start line constraint.")
    end_line: Optional[int] = Field(default=None, description="Optional 1-indexed end line constraint.")


@aug_tool(args_schema=ReplaceInFileArgs)
def replace_in_file(
    filepath: str,
    target: str,
    replacement: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """Replace target text with replacement content in a file and return the unified diff."""
    try:
        path_obj = resolve_sandboxed_path(filepath)
        if not path_obj.exists() or not path_obj.is_file():
            return f"Error: File '{filepath}' not found."

        before_content = path_obj.read_text(encoding="utf-8")

        if start_line is not None or end_line is not None:
            lines = before_content.splitlines(keepends=True)
            total_lines = len(lines)

            s_line = start_line if start_line is not None else 1
            e_line = end_line if end_line is not None else total_lines

            if s_line < 1:
                return "Error: start_line must be >= 1."
            if e_line < s_line:
                return f"Error: end_line ({e_line}) cannot be less than start_line ({s_line})."
            if s_line > total_lines:
                return f"Error: start_line ({s_line}) exceeds total lines in file ({total_lines})."

            effective_end = min(e_line, total_lines)
            target_slice = "".join(lines[s_line - 1 : effective_end])

            if target not in target_slice:
                return (
                    f"Error: Target text not found in '{filepath}' between lines "
                    f"{s_line} and {effective_end}."
                )

            new_slice = target_slice.replace(target, replacement, 1)
            after_content = "".join(lines[: s_line - 1]) + new_slice + "".join(lines[effective_end:])
        else:
            if target not in before_content:
                return f"Error: Target text not found in '{filepath}'. Check exact whitespace and indentation."
            if before_content.count(target) > 1:
                return (
                    f"Error: Target occurs {before_content.count(target)} times in '{filepath}'. "
                    "Specify start_line and end_line or provide a more unique target block."
                )
            after_content = before_content.replace(target, replacement, 1)

        diff = generate_unified_diff(before_content, after_content, filepath=filepath)
        path_obj.write_text(after_content, encoding="utf-8")
        return f"Successfully replaced content in '{filepath}'.\n\n--- Unified Diff ---\n{diff}"
    except PermissionError as pe:
        return f"Error: {pe}"
    except Exception as exc:
        return f"Error modifying file '{filepath}': {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# 4. insert_at_line
# ═══════════════════════════════════════════════════════════════════════════

class InsertAtLineArgs(BaseModel):
    filepath: str = Field(description="Path to the file to modify.")
    line_number: int = Field(ge=1, description="1-indexed line number where content will be inserted.")
    content: str = Field(description="Content to insert.")


@aug_tool(args_schema=InsertAtLineArgs)
def insert_at_line(filepath: str, line_number: int, content: str) -> str:
    """Insert text at a specific 1-indexed line position in the file."""
    try:
        path_obj = resolve_sandboxed_path(filepath)
        if not path_obj.exists() or not path_obj.is_file():
            return f"Error: File '{filepath}' not found."

        if line_number < 1:
            return "Error: line_number must be >= 1."

        before_content = path_obj.read_text(encoding="utf-8")
        lines = before_content.splitlines(keepends=True)

        # Normalize content insertion with trailing newline if not present
        to_insert = content if content.endswith("\n") else content + "\n"

        insert_idx = line_number - 1
        if insert_idx >= len(lines):
            lines.append(to_insert)
        else:
            lines.insert(insert_idx, to_insert)

        after_content = "".join(lines)
        diff = generate_unified_diff(before_content, after_content, filepath=filepath)
        path_obj.write_text(after_content, encoding="utf-8")
        return f"Successfully inserted content at line {line_number} in '{filepath}'.\n\n--- Unified Diff ---\n{diff}"
    except PermissionError as pe:
        return f"Error: {pe}"
    except Exception as exc:
        return f"Error inserting into file '{filepath}': {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# 5. delete_lines
# ═══════════════════════════════════════════════════════════════════════════

class DeleteLinesArgs(BaseModel):
    filepath: str = Field(description="Path to the file to modify.")
    start_line: int = Field(ge=1, description="1-indexed start line to delete (inclusive).")
    end_line: int = Field(ge=1, description="1-indexed end line to delete (inclusive).")


@aug_tool(args_schema=DeleteLinesArgs)
def delete_lines(filepath: str, start_line: int, end_line: int) -> str:
    """Delete a range of lines from start_line to end_line (inclusive) in a file."""
    try:
        path_obj = resolve_sandboxed_path(filepath)
        if not path_obj.exists() or not path_obj.is_file():
            return f"Error: File '{filepath}' not found."

        if start_line < 1:
            return "Error: start_line must be >= 1."
        if end_line < start_line:
            return f"Error: end_line ({end_line}) cannot be less than start_line ({start_line})."

        before_content = path_obj.read_text(encoding="utf-8")
        lines = before_content.splitlines(keepends=True)
        total_lines = len(lines)

        if start_line > total_lines:
            return f"Error: start_line ({start_line}) exceeds total lines in file ({total_lines})."

        effective_end = min(end_line, total_lines)
        del lines[start_line - 1 : effective_end]

        after_content = "".join(lines)
        diff = generate_unified_diff(before_content, after_content, filepath=filepath)
        path_obj.write_text(after_content, encoding="utf-8")
        return f"Successfully deleted lines {start_line}-{effective_end} in '{filepath}'.\n\n--- Unified Diff ---\n{diff}"
    except PermissionError as pe:
        return f"Error: {pe}"
    except Exception as exc:
        return f"Error deleting lines in file '{filepath}': {exc}"

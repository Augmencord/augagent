"""File manipulation tools for AugAgent."""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from augagent.tools import aug_tool

class ViewFileArgs(BaseModel):
    filepath: str = Field(description="Absolute or relative path to the file to view.")
    start_line: int = Field(description="Optional 1-indexed start line.", default=1)
    end_line: int | None = Field(description="Optional 1-indexed end line. Max 500 lines will be returned.", default=None)

@aug_tool(args_schema=ViewFileArgs)  # type: ignore
def view_file(filepath: str, start_line: int = 1, end_line: int | None = None) -> str:
    """View the contents of a file with pagination to prevent context window bloat."""
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        return f"Error: File '{filepath}' not found."
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
        total_lines = len(lines)
        if start_line < 1: start_line = 1
        
        effective_end = end_line if end_line is not None else start_line + 499
        if effective_end > start_line + 499:
            effective_end = start_line + 499
        if effective_end > total_lines:
            effective_end = total_lines
            
        view_lines = lines[start_line-1 : effective_end]
        output = [f"File: {filepath} (Lines {start_line}-{effective_end} of {total_lines})", "-"*40]
        for i, line in enumerate(view_lines, start_line):
            output.append(f"{i:4d} | {line}")
        
        if effective_end < total_lines:
            output.append("... (file truncated, use view_file with a higher start_line to read more) ...")
            
        return "\n".join(output)
    except Exception as e:
        return f"Error viewing file: {e}"

class CreateFileArgs(BaseModel):
    filepath: str = Field(description="Path to the file to create.")
    content: str = Field(description="Content to write to the file.")

@aug_tool(args_schema=CreateFileArgs)  # type: ignore
def create_file(filepath: str, content: str) -> str:
    """Create a new file with content. Fails if the file already exists."""
    path = Path(filepath)
    if path.exists():
        return f"Error: File '{filepath}' already exists. Use replace_file_content to edit it."
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return f"Successfully created '{filepath}'."
    except Exception as e:
        return f"Error creating file: {e}"

class ListDirArgs(BaseModel):
    dirpath: str = Field(description="Path to the directory to list.", default=".")

@aug_tool(args_schema=ListDirArgs)  # type: ignore
def list_directory(dirpath: str) -> str:
    """List contents of a directory."""
    path = Path(dirpath)
    if not path.exists() or not path.is_dir():
        return f"Error: Directory '{dirpath}' not found."
    
    try:
        items = list(path.iterdir())
        res = []
        for p in items:
            prefix = "[DIR] " if p.is_dir() else "[FILE]"
            res.append(f"{prefix} {p.name}")
        return "\n".join(res)
    except Exception as e:
        return f"Error listing directory: {e}"

class ReplaceFileContentArgs(BaseModel):
    filepath: str = Field(description="Path to the file to modify.")
    target_content: str = Field(description="The exact block of code to remove. Must match exactly.")
    replacement_content: str = Field(description="The new block of code to insert in its place.")

@aug_tool(args_schema=ReplaceFileContentArgs)  # type: ignore
def replace_file_content(filepath: str, target_content: str, replacement_content: str) -> str:
    """Replace a specific block of text in a file with new text."""
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        return f"Error: File '{filepath}' not found."
        
    try:
        content = path.read_text(encoding='utf-8')
        
        if target_content not in content:
            return "Error: target_content not found in the file. Ensure exact whitespace matching."
            
        if content.count(target_content) > 1:
            return "Error: target_content occurs multiple times in the file. Make it more specific."
            
        new_content = content.replace(target_content, replacement_content)
        path.write_text(new_content, encoding='utf-8')
        return f"Successfully replaced content in '{filepath}'."
    except Exception as e:
        return f"Error modifying file: {e}"

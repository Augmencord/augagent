"""File manipulation tools for AugAgent."""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from augagent.tools import aug_tool

class ReadFileArgs(BaseModel):
    filepath: str = Field(description="Absolute or relative path to the file to read.")

@aug_tool(args_schema=ReadFileArgs)
def read_file(filepath: str) -> str:
    """Read the contents of a file."""
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        return f"Error: File '{filepath}' not found."
    try:
        return path.read_text(encoding='utf-8')
    except Exception as e:
        return f"Error reading file: {e}"

class WriteFileArgs(BaseModel):
    filepath: str = Field(description="Path to the file to write.")
    content: str = Field(description="Content to write to the file.")

@aug_tool(args_schema=WriteFileArgs)
def write_file(filepath: str, content: str) -> str:
    """Write content to a file, overwriting if it exists."""
    path = Path(filepath)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return f"Successfully wrote to '{filepath}'."
    except Exception as e:
        return f"Error writing file: {e}"

class ListDirArgs(BaseModel):
    dirpath: str = Field(description="Path to the directory to list.", default=".")

@aug_tool(args_schema=ListDirArgs)
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

@aug_tool(args_schema=ReplaceFileContentArgs)
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

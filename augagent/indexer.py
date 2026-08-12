"""Codebase Indexer for AugAgent.

Scans local directories, chunks code files, and embeddings them into LongTermMemory.
"""

import os
import ast
from pathlib import Path
from augagent.memory import global_long_term_memory

def chunk_text_naive(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def chunk_python_ast(content: str) -> list[str]:
    chunks = []
    try:
        tree = ast.parse(content)
        lines = content.splitlines(keepends=True)
        
        # Track which lines are already part of a chunk
        chunked_lines = set()
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = getattr(node, 'end_lineno', start + 1)
                chunk = "".join(lines[start:end])
                chunks.append(chunk)
                for i in range(start, end):
                    chunked_lines.add(i)
                    
        # Group remaining lines into naive chunks
        remaining = []
        for i, line in enumerate(lines):
            if i not in chunked_lines:
                remaining.append(line)
        
        if remaining:
            chunks.extend(chunk_text_naive("".join(remaining)))
            
    except SyntaxError:
        # Fallback if there is a syntax error
        chunks = chunk_text_naive(content)
        
    return chunks

def index_codebase(root_path: str, max_files: int = 100):
    """Recursively index the codebase, skipping common binary/hidden files."""
    skip_dirs = {'.git', 'node_modules', 'venv', '__pycache__', '.chroma_db'}
    skip_exts = {'.png', '.jpg', '.pdf', '.exe', '.dll', '.zip', '.tar'}
    
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Modify dirnames in-place to prune skip_dirs
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        
        for file in filenames:
            if count >= max_files:
                break
                
            path = Path(dirpath) / file
            if path.suffix in skip_exts:
                continue
                
            try:
                content = path.read_text(encoding='utf-8')
                
                if path.suffix == '.py':
                    chunks = chunk_python_ast(content)
                else:
                    chunks = chunk_text_naive(content)
                    
                for i, chunk in enumerate(chunks):
                    doc_id = f"{path.as_posix()}_{i}"
                    global_long_term_memory.add_document(
                        text=chunk,
                        metadata={"filepath": path.as_posix(), "chunk": i},
                        doc_id=doc_id
                    )
                count += 1
            except Exception:
                # Skip files that can't be read as text
                continue
                
    return count

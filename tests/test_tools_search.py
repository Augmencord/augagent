import os
from pathlib import Path
import pytest

from augagent.tools_search import (
    grep_search,
    find_files,
    list_directory,
    _python_grep_fallback,
)
from augagent.diff_engine import get_workspace_root


@pytest.fixture
def search_sandbox(tmp_path, monkeypatch):
    """Fixture to set up a dedicated test directory as WORKSPACE_ROOT."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))

    # Populate test files
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "app.py").write_text("def run():\n    print('Hello world!')\n    return 42\n", encoding="utf-8")
    (src / "utils.py").write_text("def helper():\n    # HELPER FUNCTION\n    return 'HELP'\n", encoding="utf-8")

    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "README.md").write_text("# Project Docs\nWelcome to AugHome IDE.\n", encoding="utf-8")

    # Noise directory to be ignored
    noise = tmp_path / "node_modules" / "pkg"
    noise.mkdir(parents=True, exist_ok=True)
    (noise / "index.js").write_text("console.log('Ignore me');", encoding="utf-8")

    return tmp_path


def test_grep_search_normal(search_sandbox):
    """Test normal grep_search finding occurrences in test files."""
    res = grep_search(query="Hello world", path="src")
    assert "app.py" in res
    assert "Hello world" in res


def test_grep_search_case_insensitive(search_sandbox):
    """Test case-insensitive grep matching."""
    res = grep_search(query="helper function", path="src", case_sensitive=False)
    assert "utils.py" in res
    assert "HELPER FUNCTION" in res


def test_grep_search_python_fallback(search_sandbox, monkeypatch):
    """Test Python re fallback when ripgrep (rg) is not found."""
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    res = grep_search(query="return 42", path="src")
    assert "app.py" in res
    assert "return 42" in res

    no_match = grep_search(query="nonexistent_pattern_xyz", path="src")
    assert "No matches found" in no_match


def test_grep_search_sandbox_traversal(search_sandbox):
    """Test that grep_search blocks path traversal out of sandbox."""
    res = grep_search(query="secret", path="../../etc")
    assert "Error" in res or "Sandbox violation" in res or "Access denied" in res


def test_find_files_normal(search_sandbox):
    """Test find_files with glob pattern."""
    res = find_files(pattern="*.py", path=".")
    assert "src/app.py" in res or "app.py" in res
    assert "src/utils.py" in res or "utils.py" in res
    # Ensure node_modules was ignored
    assert "index.js" not in res


def test_find_files_recursive(search_sandbox):
    """Test find_files matching markdown in docs."""
    res = find_files(pattern="*.md", path=".")
    assert "README.md" in res


def test_find_files_sandbox_traversal(search_sandbox):
    """Test find_files rejects path traversal."""
    res = find_files(pattern="*", path="../..")
    assert "Error" in res


def test_list_directory_non_recursive(search_sandbox):
    """Test list_directory basic output."""
    res = list_directory(path=".")
    assert "[DIR]  src/" in res
    assert "[DIR]  docs/" in res
    # node_modules should be ignored
    assert "node_modules" not in res


def test_list_directory_recursive(search_sandbox):
    """Test recursive directory tree listing."""
    res = list_directory(path=".", recursive=True, max_depth=2)
    assert "Directory tree" in res
    assert "app.py" in res
    assert "utils.py" in res


def test_list_directory_nonexistent(search_sandbox):
    """Test list_directory on nonexistent path."""
    res = list_directory(path="does_not_exist")
    assert "Error" in res

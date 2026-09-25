import subprocess
from pathlib import Path
import pytest

from augagent.tools_git import (
    git_status,
    git_diff,
    git_commit,
    git_log,
    git_branch,
    git_add,
)


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """Fixture that initializes a clean git repository inside tmp_path."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))

    # Initialize git repo in tmp_path
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@augmencord.com"], cwd=str(tmp_path), capture_output=True, check=True)

    # Create initial file and commit
    init_file = tmp_path / "hello.txt"
    init_file.write_text("initial content\n", encoding="utf-8")
    subprocess.run(["git", "add", "hello.txt"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(tmp_path), capture_output=True, check=True)

    return tmp_path


def test_git_status_clean(git_repo):
    """Test git_status reports clean working tree."""
    res = git_status(cwd=".")
    assert "Branch:" in res
    assert "clean" in res.lower()


def test_git_add_and_status(git_repo):
    """Test git_add stages files and git_status shows them."""
    new_file = git_repo / "feature.py"
    new_file.write_text("print('new feature')", encoding="utf-8")

    # Before staging: untracked
    stat_before = git_status(cwd=".")
    assert "Untracked files" in stat_before
    assert "feature.py" in stat_before

    # Stage
    add_res = git_add(files="feature.py", cwd=".")
    assert "Successfully staged" in add_res

    # After staging: staged
    stat_after = git_status(cwd=".")
    assert "Staged" in stat_after
    assert "feature.py" in stat_after


def test_git_diff_working_and_staged(git_repo):
    """Test git_diff for unstaged and staged changes."""
    target_file = git_repo / "hello.txt"
    target_file.write_text("initial content\nnew uncommitted line\n", encoding="utf-8")

    # Unstaged diff
    diff_unstaged = git_diff(cwd=".")
    assert "+new uncommitted line" in diff_unstaged

    # Stage changes
    git_add(files="hello.txt", cwd=".")

    # Staged diff
    diff_staged = git_diff(cwd=".", staged=True)
    assert "+new uncommitted line" in diff_staged


def test_git_commit_and_log(git_repo):
    """Test committing changes and viewing git_log."""
    doc = git_repo / "notes.txt"
    doc.write_text("My notes", encoding="utf-8")
    git_add(files="notes.txt", cwd=".")

    commit_res = git_commit(message="Add notes file", cwd=".")
    assert "Add notes file" in commit_res or "1 file changed" in commit_res or "commit" in commit_res.lower()

    # Verify log
    log_res = git_log(cwd=".", max_count=5)
    assert "Add notes file" in log_res
    assert "Initial commit" in log_res


def test_git_branch_list_and_create(git_repo):
    """Test listing and creating git branches."""
    # List branches
    branches = git_branch(cwd=".", action="list")
    assert "master" in branches or "main" in branches

    # Create branch
    create_res = git_branch(cwd=".", action="create", branch_name="feature-branch")
    assert "Successfully created branch 'feature-branch'" in create_res

    # Verify listed
    branches_after = git_branch(cwd=".", action="list")
    assert "feature-branch" in branches_after


def test_git_sandbox_traversal(git_repo):
    """Test git tools prevent path traversal outside workspace."""
    res_status = git_status(cwd="../../outside")
    assert "Error" in res_status

    res_add = git_add(files="../../escape.py", cwd=".")
    assert "Error" in res_add


def test_git_non_git_directory(tmp_path, monkeypatch):
    """Test git commands fail gracefully in non-git directory."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    res = git_status(cwd=".")
    assert "Git error" in res or "not a git repository" in res.lower()

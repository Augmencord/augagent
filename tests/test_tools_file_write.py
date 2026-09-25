"""Comprehensive test suite for augagent file write, edit, and diff tools.

Covers:
- Normal operations (write, create, replace, insert, delete, diff, patch)
- Edge cases (empty files, unicode/emoji strings, single-line boundary edits)
- Failure cases (path traversal attempts, missing files, invalid line bounds, duplicate targets)
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from augagent.diff_engine import (
    apply_patch,
    generate_unified_diff,
    get_workspace_root,
    resolve_sandboxed_path,
)
from augagent.tools_file_write import (
    create_file,
    delete_lines,
    insert_at_line,
    replace_in_file,
    write_file,
)


class TestToolsFileWrite(unittest.TestCase):
    """Test suite for file writing, editing, diffing, and sandbox enforcement."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="augagent_sandbox_")
        self.orig_workspace_root = os.environ.get("WORKSPACE_ROOT")
        os.environ["WORKSPACE_ROOT"] = self.temp_dir

    def tearDown(self) -> None:
        if self.orig_workspace_root is not None:
            os.environ["WORKSPACE_ROOT"] = self.orig_workspace_root
        else:
            os.environ.pop("WORKSPACE_ROOT", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Normal Cases
    # ═══════════════════════════════════════════════════════════════════════

    def test_write_file_normal(self) -> None:
        """Verify write_file creates and overwrites a file."""
        res1 = write_file("hello.txt", "Initial content")
        self.assertIn("Successfully wrote", res1)
        self.assertEqual((Path(self.temp_dir) / "hello.txt").read_text(encoding="utf-8"), "Initial content")

        # Overwrite
        res2 = write_file("hello.txt", "Updated content")
        self.assertIn("Successfully wrote", res2)
        self.assertEqual((Path(self.temp_dir) / "hello.txt").read_text(encoding="utf-8"), "Updated content")

    def test_create_file_normal(self) -> None:
        """Verify create_file creates a new file and rejects existing file."""
        res1 = create_file("new_file.py", "print('hello')\n")
        self.assertIn("Successfully created", res1)
        self.assertTrue((Path(self.temp_dir) / "new_file.py").exists())

        # Attempting to recreate should fail
        res2 = create_file("new_file.py", "different content")
        self.assertIn("Error: File 'new_file.py' already exists", res2)

    def test_replace_in_file_whole_file_normal(self) -> None:
        """Verify replace_in_file substitutes unique target and returns diff."""
        write_file("app.py", "def old_name():\n    return 42\n")
        res = replace_in_file("app.py", "old_name", "new_name")
        self.assertIn("Successfully replaced content", res)
        self.assertIn("--- Unified Diff ---", res)
        self.assertIn("-def old_name():", res)
        self.assertIn("+def new_name():", res)

        content = (Path(self.temp_dir) / "app.py").read_text(encoding="utf-8")
        self.assertIn("new_name", content)
        self.assertNotIn("old_name", content)

    def test_replace_in_file_targeted_lines(self) -> None:
        """Verify replace_in_file with start_line and end_line bounds."""
        initial = "item = 1\nitem = 2\nitem = 3\n"
        write_file("items.py", initial)

        # Replace only the second occurrence using line bounding
        res = replace_in_file("items.py", "item = 2", "item = 200", start_line=2, end_line=2)
        self.assertIn("Successfully replaced", res)

        content = (Path(self.temp_dir) / "items.py").read_text(encoding="utf-8")
        self.assertEqual(content, "item = 1\nitem = 200\nitem = 3\n")

    def test_insert_at_line_normal(self) -> None:
        """Verify insert_at_line inserts content at 1-indexed line position."""
        write_file("script.py", "line 1\nline 3\n")
        res = insert_at_line("script.py", line_number=2, content="line 2")
        self.assertIn("Successfully inserted", res)
        self.assertIn("+line 2", res)

        content = (Path(self.temp_dir) / "script.py").read_text(encoding="utf-8")
        self.assertEqual(content, "line 1\nline 2\nline 3\n")

    def test_delete_lines_normal(self) -> None:
        """Verify delete_lines removes the specified line range."""
        write_file("data.txt", "one\ntwo\nthree\nfour\n")
        res = delete_lines("data.txt", start_line=2, end_line=3)
        self.assertIn("Successfully deleted lines 2-3", res)

        content = (Path(self.temp_dir) / "data.txt").read_text(encoding="utf-8")
        self.assertEqual(content, "one\nfour\n")

    def test_diff_engine_generate_and_apply_patch(self) -> None:
        """Verify diff generation and round-trip patch application."""
        original = "alpha\nbeta\ngamma\n"
        modified = "alpha\nbeta prime\ngamma\ndelta\n"

        write_file("sample.txt", original)
        diff = generate_unified_diff(original, modified, filepath="sample.txt")
        self.assertIn("-beta", diff)
        self.assertIn("+beta prime", diff)
        self.assertIn("+delta", diff)

        patch_res = apply_patch("sample.txt", diff)
        self.assertIn("Successfully applied patch", patch_res)

        updated = (Path(self.temp_dir) / "sample.txt").read_text(encoding="utf-8")
        self.assertEqual(updated, modified)

    def test_tools_openai_schema_generation(self) -> None:
        """Verify OpenAI-compatible function schema derivation for all 5 tools."""
        tools = [write_file, create_file, replace_in_file, insert_at_line, delete_lines]
        for t in tools:
            schema = t.to_openai_schema()
            self.assertEqual(schema["type"], "function")
            self.assertIn("name", schema["function"])
            self.assertIn("parameters", schema["function"])
            self.assertIn("properties", schema["function"]["parameters"])

    # ═══════════════════════════════════════════════════════════════════════
    # Edge Cases
    # ═══════════════════════════════════════════════════════════════════════

    def test_empty_content_and_file(self) -> None:
        """Verify creation of an empty file and writing empty string."""
        res = create_file("empty.txt")
        self.assertIn("Successfully created", res)
        self.assertEqual((Path(self.temp_dir) / "empty.txt").read_text(encoding="utf-8"), "")

        # Write empty content
        res2 = write_file("empty.txt", "")
        self.assertIn("Successfully wrote 0 characters", res2)

    def test_unicode_and_emoji_content(self) -> None:
        """Verify unicode, emoji, and CJK multi-byte character preservation."""
        unicode_text = "AI 🤖 智能 / Привет мир / 🚀\n"
        res = write_file("unicode.txt", unicode_text)
        self.assertIn("Successfully wrote", res)

        content = (Path(self.temp_dir) / "unicode.txt").read_text(encoding="utf-8")
        self.assertEqual(content, unicode_text)

        # Replace unicode with another unicode
        res2 = replace_in_file("unicode.txt", "Привет мир", "Hello World 🌍")
        self.assertIn("Successfully replaced", res2)
        content2 = (Path(self.temp_dir) / "unicode.txt").read_text(encoding="utf-8")
        self.assertIn("Hello World 🌍", content2)

    def test_insert_at_line_beyond_eof(self) -> None:
        """Verify insert_at_line appends cleanly when line_number exceeds file line count."""
        write_file("short.txt", "line 1\n")
        res = insert_at_line("short.txt", line_number=100, content="line 2 appended")
        self.assertIn("Successfully inserted", res)

        content = (Path(self.temp_dir) / "short.txt").read_text(encoding="utf-8")
        self.assertEqual(content, "line 1\nline 2 appended\n")

    def test_diff_identical_strings(self) -> None:
        """Verify diff of identical strings produces empty string."""
        diff = generate_unified_diff("same", "same")
        self.assertEqual(diff, "")

    # ═══════════════════════════════════════════════════════════════════════
    # Failure Cases
    # ═══════════════════════════════════════════════════════════════════════

    def test_path_traversal_detection(self) -> None:
        """Verify sandbox prevents directory traversal attacks."""
        traversal_paths = [
            "../outside.txt",
            "../../etc/passwd",
            "..\\..\\windows\\system32\\calc.exe",
            f"{self.temp_dir}/../secret.env",
        ]

        for bad_path in traversal_paths:
            with self.subTest(path=bad_path):
                res_write = write_file(bad_path, "malicious")
                self.assertIn("Error:", res_write)
                self.assertIn("escapes workspace sandbox", res_write)

                res_create = create_file(bad_path, "malicious")
                self.assertIn("Error:", res_create)
                self.assertIn("escapes workspace sandbox", res_create)

                with self.assertRaises(PermissionError):
                    resolve_sandboxed_path(bad_path)

    def test_file_not_found_errors(self) -> None:
        """Verify non-existent files return descriptive errors."""
        res_replace = replace_in_file("missing.py", "x", "y")
        self.assertIn("Error: File 'missing.py' not found", res_replace)

        res_insert = insert_at_line("missing.py", 1, "code")
        self.assertIn("Error: File 'missing.py' not found", res_insert)

        res_delete = delete_lines("missing.py", 1, 2)
        self.assertIn("Error: File 'missing.py' not found", res_delete)

        with self.assertRaises(FileNotFoundError):
            apply_patch("missing.py", "@@ -1 +1 @@\n-a\n+b\n")

    def test_replace_target_not_found(self) -> None:
        """Verify error when target content does not exist in file."""
        write_file("foo.py", "print('hello')\n")
        res = replace_in_file("foo.py", "non_existent", "replacement")
        self.assertIn("Error: Target text not found", res)

    def test_replace_duplicate_target_without_bounds(self) -> None:
        """Verify error when target occurs multiple times without line bounds."""
        write_file("foo.py", "var x = 1;\nvar x = 2;\n")
        res = replace_in_file("foo.py", "var x", "let y")
        self.assertIn("Error: Target occurs 2 times", res)

    def test_invalid_line_bounds(self) -> None:
        """Verify invalid line numbers return explicit errors."""
        write_file("bounds.py", "line 1\nline 2\n")

        # start_line > end_line
        res_del = delete_lines("bounds.py", start_line=5, end_line=2)
        self.assertIn("cannot be less than start_line", res_del)

        # start_line beyond file
        res_del2 = delete_lines("bounds.py", start_line=50, end_line=60)
        self.assertIn("exceeds total lines in file", res_del2)

        # start_line < 1 in replace
        res_rep = replace_in_file("bounds.py", "line", "line", start_line=0, end_line=2)
        self.assertIn("start_line must be >= 1", res_rep)

    def test_apply_patch_invalid_format(self) -> None:
        """Verify apply_patch rejects malformed patch strings."""
        write_file("patch_target.txt", "line 1\n")
        with self.assertRaises(ValueError):
            apply_patch("patch_target.txt", "not a valid unified diff")


if __name__ == "__main__":
    unittest.main()

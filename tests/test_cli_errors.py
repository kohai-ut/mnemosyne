"""CLI error handling regression tests."""

import os
import subprocess
import sys


COMMANDS = [
    (
        ["store", "hello", "cli", "not-a-float"],
        "importance must be a number",
    ),
    (
        ["recall", "hello", "not-an-int"],
        "top_k must be an integer",
    ),
    (
        ["update", "missing-id", "new content", "not-a-float"],
        "importance must be a number",
    ),
    (
        ["import", "missing-file.json"],
        "Import file not found",
    ),
]

USAGE_COMMANDS = [
    (["store"], "Usage: mnemosyne store <content> [source] [importance]"),
    (["recall"], "Usage: mnemosyne recall <query> [top_k]"),
    (["update", "missing-id"], "Usage: mnemosyne update <memory_id> <new_content> [importance]"),
    (["delete"], "Usage: mnemosyne delete <memory_id>"),
    (["import"], "Usage: mnemosyne import <file.json>"),
    (["import-hindsight"], "Usage: mnemosyne import-hindsight <file.json|base_url> [bank]"),
    (["bank"], "Usage: mnemosyne bank <list|create|delete> [name]"),
]


def run_cli(args, tmp_path):
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "mnemosyne-data")
    return subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", *args],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_invalid_cli_input_reports_error_without_traceback(tmp_path):
    for args, expected_error in COMMANDS:
        result = run_cli(args, tmp_path)

        assert result.returncode != 0, args
        assert expected_error in result.stderr, result.stderr
        assert "Traceback" not in result.stderr


def test_import_malformed_json_reports_error_without_traceback(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")

    result = run_cli(["import", str(bad_json)], tmp_path)

    assert result.returncode != 0
    assert "Invalid JSON" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_required_args_report_usage_error_without_traceback(tmp_path):
    for args, expected_usage in USAGE_COMMANDS:
        result = run_cli(args, tmp_path)

        assert result.returncode != 0, args
        assert result.stdout == ""
        assert expected_usage in result.stderr
        assert "Traceback" not in result.stderr


def test_unknown_command_reports_error_without_traceback(tmp_path):
    result = run_cli(["definitely-not-a-command"], tmp_path)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "Unknown command: definitely-not-a-command" in result.stderr
    assert "Run 'mnemosyne --help' for usage." in result.stderr
    assert "Traceback" not in result.stderr


def test_help_exits_successfully(tmp_path):
    result = run_cli(["--help"], tmp_path)

    assert result.returncode == 0
    assert "Usage: mnemosyne <command> [args]" in result.stdout
    assert "Traceback" not in result.stderr

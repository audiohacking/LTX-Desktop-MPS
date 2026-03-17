"""Static type checking gate for backend sources."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _format_diagnostics(payload: dict[str, Any], limit: int = 15) -> str:
    diagnostics = payload.get("generalDiagnostics", [])
    if not isinstance(diagnostics, list):
        return "No diagnostics available."

    lines: list[str] = []
    for item in diagnostics[:limit]:
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file", "<unknown>"))
        message = str(item.get("message", "<no message>")).splitlines()[0]
        range_data = item.get("range", {})
        start = range_data.get("start", {}) if isinstance(range_data, dict) else {}
        line = int(start.get("line", 0)) + 1 if isinstance(start, dict) else 0
        severity = str(item.get("severity", "unknown"))
        lines.append(f"{severity}: {file_name}:{line}: {message}")

    if not lines:
        return "No diagnostics available."
    return "\n".join(lines)


def test_pyright_has_no_errors_or_warnings() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    # Use pyright from the backend venv (sys.executable can resolve to uv's Python)
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    pyright_name = "pyright.cmd" if sys.platform == "win32" else "pyright"
    venv_pyright = backend_root / ".venv" / bin_dir / pyright_name
    cmd = [str(venv_pyright), "--outputjson"] if venv_pyright.exists() else ["pyright", "--outputjson"]
    result = subprocess.run(
        cmd,
        cwd=backend_root,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout.strip() or result.stderr.strip()
    assert output, "pyright produced no output"

    try:
        payload: dict[str, Any] = json.loads(output)
    except json.JSONDecodeError:
        normalized = output.lower()
        if "command not found" in normalized or "not found" in normalized:
            pytest.skip("pyright is not available on PATH; skipping type-check gate")
        raise AssertionError(
            "pyright did not produce valid JSON output.\n"
            f"returnCode={result.returncode}\n"
            f"{output}"
        ) from None
    summary = payload.get("summary", {})
    assert isinstance(summary, dict), "pyright JSON output missing summary"

    error_count = int(summary.get("errorCount", -1))
    warning_count = int(summary.get("warningCount", -1))

    assert error_count == 0 and warning_count == 0, (
        "pyright reported issues:\n"
        f"errorCount={error_count}, warningCount={warning_count}, returnCode={result.returncode}\n"
        f"{_format_diagnostics(payload)}"
    )

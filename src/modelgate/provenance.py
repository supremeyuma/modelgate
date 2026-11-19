"""Small provenance helpers for records and manifests."""

from __future__ import annotations

import subprocess


def git_sha(short: bool = False) -> str:
    """Best-effort HEAD sha; records stay valid outside a git checkout."""
    command = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unknown"

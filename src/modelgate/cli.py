"""Thin command line entry point; real CLIs live next to their modules."""

from __future__ import annotations


def main() -> int:
    """Print the installed version."""
    from modelgate import __version__

    print(f"modelgate {__version__}")
    return 0

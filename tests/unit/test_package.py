import pytest

from modelgate import __version__


def test_version_is_exposed() -> None:
    assert __version__


def test_cli_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    from modelgate.cli import main

    assert main() == 0
    captured = capsys.readouterr()
    assert "modelgate" in captured.out

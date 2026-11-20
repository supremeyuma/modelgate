from __future__ import annotations

import pytest

from modelgate.promote.__main__ import main


def test_cli_demo_reject_runs_without_writing() -> None:
    assert main(["--demo", "1", "--dry-run"]) == 0


def test_cli_demo_promote_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--demo", "2", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "PROMOTE" in output


def test_cli_requires_arguments() -> None:
    with pytest.raises(SystemExit):
        main([])

"""The readiness check.

Its whole job is to be right when something is missing, so the tests are about
failure paths rather than the happy one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pharmadt.doctor import DATASETS, OPTIONAL_DATASETS, Result, _files, run_checks


def test_a_failure_carries_the_command_that_fixes_it() -> None:
    """A check that says FAIL without a fix just relocates the confusion."""
    rendered = Result("schema", False, "3/8 tables", "make migrate").render(20)
    assert "FAIL" in rendered
    assert "make migrate" in rendered


def test_a_passing_check_does_not_shout_a_fix() -> None:
    rendered = Result("schema", True, "8/8 tables", "make migrate").render(20)
    assert "ok" in rendered
    assert "make migrate" not in rendered


def test_an_optional_gap_warns_rather_than_fails() -> None:
    """CMS validates the demand mix; nothing depends on it to run."""
    rendered = Result("dataset cms", False, "missing", "make data", optional=True).render(20)
    assert "warn" in rendered
    assert "FAIL" not in rendered


def test_every_required_check_names_a_fix() -> None:
    for result in run_checks():
        if not result.passed:
            assert result.fix, f"{result.name} fails without telling anyone what to run"


def test_the_datasets_checked_are_the_ones_make_data_fetches() -> None:
    from pharmadt.ml.preprocessing import DATASETS as PIPELINE

    checked = set(DATASETS) | set(OPTIONAL_DATASETS)
    # demand_profiles is derived from rossmann rather than downloaded, so it has
    # no directory under data/raw to look for.
    assert checked == set(PIPELINE) - {"demand_profiles"}


def test_cms_is_the_only_optional_dataset() -> None:
    assert OPTIONAL_DATASETS == ("cms",)


def test_a_missing_env_is_reported(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    by_name = {r.name: r for r in _files()}

    assert by_name[".env"].passed is False
    assert by_name["kaggle token"].passed is False
    assert "cp .env.example .env" in by_name[".env"].fix


def test_an_env_without_a_token_is_reported(monkeypatch, tmp_path: Path) -> None:
    """The file existing is not the same as the token being in it."""
    monkeypatch.chdir(tmp_path)
    Path(".env").write_text("DATABASE_URL=postgresql://x\n", encoding="utf-8")
    by_name = {r.name: r for r in _files()}

    assert by_name[".env"].passed is True
    assert by_name["kaggle token"].passed is False


def test_missing_keys_are_reported(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = next(r for r in _files() if r.name == "private keys on disk")
    assert result.passed is False
    assert result.fix == "make keys"


@pytest.mark.slow
def test_the_check_runs_without_raising_on_this_machine() -> None:
    """However configured, it must report rather than crash."""
    results = run_checks()
    assert results
    assert all(isinstance(r.passed, bool) for r in results)

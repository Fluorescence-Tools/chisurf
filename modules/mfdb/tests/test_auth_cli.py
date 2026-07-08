"""Headless CLI tests for `mfdb-admin auth …` (PRD-59 Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from mfdb.admin.cli import cli
from mfdb.repository import MFDatabase


def _prepare_db(tmp_path: Path, monkeypatch) -> Path:
    db_path = tmp_path / "cli_auth.db"
    MFDatabase(db_path).close()  # bootstrap admin/guest/groups
    monkeypatch.setattr(
        "mfdb.store.database_resolver.resolve_database_path",
        lambda *a, **k: db_path,
    )
    # Deterministic provider config (local) regardless of ambient env/resolver.
    import mfdb.config as config

    monkeypatch.setattr(config, "_AUTH_CONFIG_RESOLVER", None)
    monkeypatch.delenv("MFDB_AUTH_PROVIDER", raising=False)
    return db_path


def test_cli_auth_login_and_whoami(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    runner = CliRunner()

    r = runner.invoke(
        cli, ["auth", "login", "--user", "user_default", "--password", "admin", "--json"]
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["ok"] and data["authenticated"]
    token = data["token"]
    assert data["user"]["user_id"] == "user_default"

    r2 = runner.invoke(cli, ["auth", "whoami", "--token", token, "--json"])
    assert r2.exit_code == 0, r2.output
    who = json.loads(r2.output)
    assert who["authenticated"] is True
    assert who["user_id"] == "user_default"
    assert who["is_admin"] is True


def test_cli_auth_login_bad_password_aborts(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    runner = CliRunner()
    r = runner.invoke(
        cli, ["auth", "login", "--user", "user_default", "--password", "wrong"]
    )
    assert r.exit_code != 0
    assert "Login failed" in r.output


def test_cli_auth_whoami_invalid_token(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    runner = CliRunner()
    r = runner.invoke(cli, ["auth", "whoami", "--token", "not-a-real-token"])
    assert r.exit_code == 0
    assert "anonymous" in r.output


def test_cli_auth_status_defaults_local(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    runner = CliRunner()
    r = runner.invoke(cli, ["auth", "status", "--json"])
    assert r.exit_code == 0, r.output
    status = json.loads(r.output)
    assert status["provider"] == "local"
    assert status["local_always_available"] is True

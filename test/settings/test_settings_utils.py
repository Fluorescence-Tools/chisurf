from __future__ import annotations

import yaml


def test_set_mfdb_login_settings_persists_values(tmp_path, monkeypatch) -> None:
    """MFDB login settings are merged into the user settings YAML."""
    from chisurf.core.settings import settings_utils

    settings_file = tmp_path / "settings_chisurf.yaml"
    settings_file.write_text(
        yaml.safe_dump(
            {
                "existing": True,
                "mfdb": {
                    "autologin": False,
                    "default_user_id": "old_user",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_utils, "get_path", lambda name: tmp_path)

    saved = settings_utils.set_mfdb_login_settings(
        {
            "autologin": True,
            "default_user_id": "admin_user",
            "save_login": True,
        }
    )

    data = yaml.safe_load(settings_file.read_text(encoding="utf-8"))
    assert saved is True
    assert data["existing"] is True
    assert data["mfdb"]["autologin"] is True
    assert data["mfdb"]["default_user_id"] == "admin_user"
    assert data["mfdb"]["save_login"] is True

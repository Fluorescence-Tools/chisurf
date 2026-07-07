from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from mfdb.security.auth import create_session
from mfdb.repository import MFDatabase
from mfdb.config import configure_runtime, reset_runtime_config


def _admin_session(db_path: Path) -> dict[str, Any]:
    """Create an admin session for testing (admin_user with password)."""
    db = MFDatabase(db_path)
    try:
        row = db.conn.execute(
            "SELECT user_id FROM flr_sample_users WHERE is_admin = 1 ORDER BY user_id LIMIT 1"
        ).fetchone()
        user_id = row[0] if row else "user_default"
        session = create_session(db.conn, user_id=user_id)
        db.conn.commit()
        return {"token": session["token"]}
    finally:
        db.close()


def _default_auth(db_path: Path) -> dict[str, Any]:
    """Create a session for user_default."""
    db = MFDatabase(db_path)
    try:
        session = create_session(db.conn, user_id="user_default")
        db.conn.commit()
        return {"token": session["token"]}
    finally:
        db.close()


def test_mfdb_default_user_creation_and_fallbacks(tmp_path: Path) -> None:
    db_path = tmp_path / "user_test.db"

    configure_runtime(default_user_id="user_default")
    try:
        # 1. Opening MFDatabase should automatically bootstrap 'user_default' in flr_sample_users
        with MFDatabase(db_path) as db:
            users = db.conn.execute("SELECT user_id, display_name FROM flr_sample_users").fetchall()
            user_ids = {row["user_id"]: row["display_name"] for row in users}
            assert "user_default" in user_ids
            assert user_ids["user_default"] == "Default User"

            # 2. Add experiment and sample without specifying measured_by_user_id.
            # They should fall back to user_default.
            db.add_sample("sample_test")
            sample = db.get_sample("sample_test")
            assert sample["measured_by_user_id"] == "user_default"

            # 3. Record operation without operator_user_id.
            # It should fall back to user_default.
            db.record_operation("op_test", "measurement_import")
            op = db.get_operation("op_test")
            assert op["operator_user_id"] == "user_default"

            # 4. If default_user_id is changed in settings, it should use the new value.
            configure_runtime(default_user_id="test_user_id")
            db.add_sample("sample_test_2")
            sample2 = db.get_sample("sample_test_2")
            assert sample2["measured_by_user_id"] == "test_user_id"

            db.add_experiment("exp_test_2", sample_id="sample_test")
            exp2 = db.get_experiment("exp_test_2")
            assert exp2["measured_by_user_id"] == "test_user_id"

            db.record_operation("op_test_2", "measurement_import")
            op2 = db.get_operation("op_test_2")
            assert op2["operator_user_id"] == "test_user_id"
    finally:
        reset_runtime_config()


def test_delete_user_handler_safety(tmp_path: Path) -> None:
    db_path = tmp_path / "user_delete_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.services import (
        delete_user_handler,
        list_users_handler,
        save_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path):
        # 1. Initialize DB by listing users (calls FluorescenceDatabase which migrates the DB)
        res = list_users_handler()
        user_ids = {u["user_id"] for u in res["users"]}
        assert "user_default" in user_ids

        auth = _default_auth(db_path)

        # 2. Try to delete 'user_default' -> should raise ValueError
        with pytest.raises(ValueError, match="cannot be deleted"):
            delete_user_handler("user_default", auth=auth)

        # 3. Create a new user 'unused_user'
        save_user_handler({"user_id": "unused_user", "display_name": "Unused User"}, auth=auth)
        res = list_users_handler()
        user_ids = {u["user_id"] for u in res["users"]}
        assert "unused_user" in user_ids

        # 4. Deleting 'unused_user' should succeed since they haven't committed any data
        delete_user_handler("unused_user", auth=auth)
        res = list_users_handler()
        user_ids = {u["user_id"] for u in res["users"]}
        assert "unused_user" not in user_ids

        # 5. Create a user 'used_user' and record an operation under them
        save_user_handler({"user_id": "used_user", "display_name": "Used User"}, auth=auth)

        # Open repository and record operation
        with MFDatabase(db_path) as db:
            db.record_operation("op_1", "measurement_import", operator_user_id="used_user")

        # 6. Try to delete 'used_user' -> should raise ValueError because of committed data
        with pytest.raises(ValueError, match="committed data"):
            delete_user_handler("used_user", auth=auth)


def test_mfdb_user_attributes_and_validation(tmp_path: Path) -> None:
    db_path = tmp_path / "user_attr_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.services import (
        save_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path):
        # Initialize DB and get auth
        from mfdb.admin.backend.services import list_users_handler
        list_users_handler()
        auth = _default_auth(db_path)

        # 1. Invalid email should raise ValueError
        with pytest.raises(ValueError, match="Invalid email format"):
            save_user_handler({
                "user_id": "test_user",
                "display_name": "Test User",
                "email": "invalid_email_no_at",
            }, auth=auth)

        # 2. Save user with valid email and all new attributes
        res = save_user_handler({
            "user_id": "test_user",
            "display_name": "Test User",
            "email": "test@example.com",
            "role": "Scientist",
            "affiliation": "Institute A",
            "department": "Biophysics",
            "phone": "+123456789",
            "website": "https://example.com",
            "address": "123 Science Rd, Tech City",
            "details": "Research scientist profile."
        }, auth=auth)

        # 3. Verify user attributes are loaded correctly
        users = res["users"]
        test_user = next(u for u in users if u["user_id"] == "test_user")

        assert test_user["user_uuid"] is not None
        assert len(test_user["user_uuid"]) > 0
        assert test_user["email"] == "test@example.com"
        assert test_user["role"] == "Scientist"
        assert test_user["affiliation"] == "Institute A"
        assert test_user["department"] == "Biophysics"
        assert test_user["phone"] == "+123456789"
        assert test_user["website"] == "https://example.com"
        assert test_user["address"] == "123 Science Rd, Tech City"
        assert test_user["details"] == "Research scientist profile."


def test_mfdb_user_passwords_and_login(tmp_path: Path) -> None:
    db_path = tmp_path / "user_pw_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.password_services import login_handler
    from mfdb.admin.backend.services import (
        list_users_handler,
        save_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path), patch(
        "mfdb.store.database_resolver.resolve_database_path", return_value=db_path
    ):
        # Initialize DB
        list_users_handler()
        auth = _default_auth(db_path)

        # 1. Create a user without a password
        save_user_handler({
            "user_id": "no_pw_user",
            "display_name": "No Password User"
        }, auth=auth)

        # Login without password should succeed
        res = login_handler("no_pw_user")
        assert res["authenticated"] is True
        assert res["user"]["user_id"] == "no_pw_user"
        assert res["user"]["is_admin"] is False

        # 2. Create a user with a password
        save_user_handler({
            "user_id": "pw_user",
            "display_name": "Password User",
            "password": "secret_password"
        }, auth=auth)

        # Login without password should fail
        res = login_handler("pw_user")
        assert res["authenticated"] is False

        # Login with incorrect password should fail
        res = login_handler("pw_user", "wrong_password")
        assert res["authenticated"] is False

        # Login with correct password should succeed
        res = login_handler("pw_user", "secret_password")
        assert res["authenticated"] is True
        assert res["user"]["user_id"] == "pw_user"


def test_mfdb_save_user_permissions(tmp_path: Path) -> None:
    db_path = tmp_path / "user_perms_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.services import (
        save_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path):
        # Initialize DB — user_default is now admin
        from mfdb.admin.backend.services import list_users_handler
        list_users_handler()

        # Get an admin session as user_default
        admin_auth = _default_auth(db_path)

        # 1. user_default is already admin. Create normal_user via admin
        res = save_user_handler(
            user={
                "user_id": "normal_user",
                "display_name": "Normal User",
                "is_admin": 0,
            },
            auth=admin_auth,
        )
        users = res["users"]
        normal = next(u for u in users if u["user_id"] == "normal_user")
        assert normal["is_admin"] == 0

        # 2. Non-admin trying to edit another user should raise ValueError
        normal_db = MFDatabase(db_path)
        normal_session = create_session(normal_db.conn, user_id="normal_user")
        normal_db.conn.commit()
        normal_auth = {"token": normal_session["token"]}
        with pytest.raises(ValueError, match="Non-admin users can only edit their own profile"):
            save_user_handler(
                user={
                    "user_id": "user_default",
                    "display_name": "Hacked Default Display Name",
                },
                auth=normal_auth,
            )

        # 3. Non-admin trying to promote themselves (set is_admin to 1) should raise ValueError
        with pytest.raises(ValueError, match="Non-admin users cannot grant admin privileges"):
            save_user_handler(
                user={
                    "user_id": "normal_user",
                    "display_name": "Normal User",
                    "is_admin": 1,
                },
                auth=normal_auth,
            )


def test_mfdb_save_user_renames_username_and_references(tmp_path: Path) -> None:
    db_path = tmp_path / "user_rename_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.services import (
        list_users_handler,
        save_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path):
        list_users_handler()
        admin_auth = _default_auth(db_path)

        created = save_user_handler(
            user={
                "user_id": "old_name",
                "display_name": "Old Name",
            },
            auth=admin_auth,
        )
        user_uuid = next(u["user_uuid"] for u in created["users"] if u["user_id"] == "old_name")

        db = MFDatabase(db_path)
        try:
            db.add_sample("rename_sample", measured_by_user_id="old_name")
            db.add_experiment("rename_experiment", sample_id="rename_sample", measured_by_user_id="old_name")
            session = create_session(db.conn, "old_name")
            db.conn.execute(
                "INSERT OR IGNORE INTO mfdb_group_member (group_id, user_id, role) VALUES ('users', ?, 'member')",
                ("old_name",),
            )
            db.conn.execute(
                "INSERT INTO mfdb_object_acl (object_type, object_id, owner_user_id) VALUES (?, ?, ?)",
                ("sample", "rename_sample", "old_name"),
            )
            db.conn.commit()
        finally:
            db.close()

        res = save_user_handler(
            user={
                "user_uuid": user_uuid,
                "user_id": "new_name",
                "display_name": "New Name",
            },
            auth=admin_auth,
        )
        user_ids = {u["user_id"] for u in res["users"]}
        assert "old_name" not in user_ids
        assert "new_name" in user_ids

        db = MFDatabase(db_path)
        try:
            assert db.conn.execute(
                "SELECT measured_by_user_id FROM flr_sample WHERE sample_id = 'rename_sample'"
            ).fetchone()[0] == "new_name"
            assert db.conn.execute(
                "SELECT measured_by_user_id FROM flr_experiment WHERE experiment_id = 'rename_experiment'"
            ).fetchone()[0] == "new_name"
            assert db.conn.execute(
                "SELECT user_id FROM mfdb_session WHERE session_id = ?",
                (db.conn.execute(
                    "SELECT session_id FROM mfdb_session WHERE token_hash = ?",
                    (hashlib.sha256(session["token"].encode("utf-8")).hexdigest(),),
                ).fetchone()[0],),
            ).fetchone()[0] == "new_name"
            assert db.conn.execute(
                "SELECT user_id FROM mfdb_group_member WHERE user_id = 'new_name'"
            ).fetchone()[0] == "new_name"
            assert db.conn.execute(
                "SELECT owner_user_id FROM mfdb_object_acl WHERE object_type = 'sample' AND object_id = 'rename_sample'"
            ).fetchone()[0] == "new_name"
        finally:
            db.close()


def test_mfdb_delete_user_admin_override(tmp_path: Path) -> None:
    db_path = tmp_path / "user_override_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.services import (
        delete_user_handler,
        save_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path):
        # 1. Initialize DB — user_default is already admin
        from mfdb.admin.backend.services import list_users_handler
        list_users_handler()
        admin_auth = _default_auth(db_path)

        # Create normal user via admin
        save_user_handler(
            user={
                "user_id": "normal_user",
                "display_name": "Normal User",
            },
            auth=admin_auth,
        )

        # 2. Add operation under normal_user
        with MFDatabase(db_path) as db:
            db.record_operation("op_override", "measurement_import", operator_user_id="normal_user")

        # 3. Normal delete without force should fail
        with pytest.raises(ValueError, match="committed data"):
            delete_user_handler("normal_user", force=False, auth=admin_auth)

        # 4. Non-admin user trying to force-delete should fail
        normal_db2 = MFDatabase(db_path)
        normal_session2 = create_session(normal_db2.conn, user_id="normal_user")
        normal_db2.conn.commit()
        normal_auth = {"token": normal_session2["token"]}
        with pytest.raises(ValueError, match="Only administrators can force-delete users"):
            delete_user_handler("normal_user", force=True, auth=normal_auth)

        # 5. Admin force-delete should succeed
        res = delete_user_handler("normal_user", force=True, auth=admin_auth)
        user_ids = {u["user_id"] for u in res["users"]}
        assert "normal_user" not in user_ids


def test_autologin_conditions() -> None:
    # Mimic user data structures
    user_no_pw = {"user_id": "default", "has_password": False, "is_admin": False, "allow_passwordless_login": False}
    user_pw = {"user_id": "default", "has_password": True, "is_admin": False, "allow_passwordless_login": False}
    user_admin_no_pw = {"user_id": "default", "has_password": False, "is_admin": True, "allow_passwordless_login": False}
    user_passwdless = {"user_id": "default", "has_password": False, "is_admin": False, "allow_passwordless_login": True}
    user_admin_pw = {"user_id": "default", "has_password": True, "is_admin": True, "allow_passwordless_login": False}
    user_admin_passwdless = {"user_id": "default", "has_password": True, "is_admin": True, "allow_passwordless_login": True}

    # Helper simulating the logic in chisurf.gui._run_startup_auth: admin
    # accounts are never silently auto-logged-in, so they always trigger the
    # LoginDialog regardless of autologin / stored tokens.
    def should_trigger_login(autologin, user_data, autologin_succeeded=False):
        if not autologin:
            return True
        if user_data is None:
            return True
        if user_data.get("is_admin"):
            return True
        return not autologin_succeeded

    # 1. autologin is OFF: always triggers login
    assert should_trigger_login(autologin=False, user_data=user_no_pw) is True
    assert should_trigger_login(autologin=False, user_data=user_pw) is True
    assert should_trigger_login(autologin=False, user_data=user_admin_no_pw) is True
    assert should_trigger_login(autologin=False, user_data=user_passwdless) is True
    assert should_trigger_login(autologin=False, user_data=user_admin_pw) is True
    assert should_trigger_login(autologin=False, user_data=user_admin_passwdless) is True

    # 2. autologin is ON:
    # - non-admin: the login screen is skipped only after backend autologin succeeds.
    assert should_trigger_login(autologin=True, user_data=user_no_pw, autologin_succeeded=True) is False
    assert should_trigger_login(autologin=True, user_data=user_pw, autologin_succeeded=False) is True
    assert should_trigger_login(autologin=True, user_data=user_passwdless, autologin_succeeded=True) is False
    # - admin: always triggers login, even when a stored token/passwordless would succeed.
    assert should_trigger_login(autologin=True, user_data=user_admin_no_pw, autologin_succeeded=True) is True
    assert should_trigger_login(autologin=True, user_data=user_admin_pw, autologin_succeeded=False) is True
    assert should_trigger_login(autologin=True, user_data=user_admin_passwdless, autologin_succeeded=True) is True
    assert should_trigger_login(autologin=True, user_data=None) is True


def test_admin_password_strength_enforcement(tmp_path: Path) -> None:
    db_path = tmp_path / "user_strength_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.password_services import change_password_handler
    from mfdb.admin.backend.services import (
        list_users_handler,
        save_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path), patch(
        "mfdb.store.database_resolver.resolve_database_path", return_value=db_path
    ):
        # Initialize DB
        list_users_handler()
        auth = _default_auth(db_path)

        # 1. Normal user with weak password should succeed
        save_user_handler({
            "user_id": "normal_user",
            "display_name": "Normal User",
            "password": "123",
            "is_admin": 0
        }, auth=auth)

        # 2. Admin user with weak password ("123") should fail (score 1)
        with pytest.raises(ValueError, match="Admin password is too weak"):
            save_user_handler({
                "user_id": "admin_user",
                "display_name": "Admin User",
                "password": "123",
                "is_admin": 1
            }, auth=auth)

        # 3. Admin user with score 3 password ("P@ss") should fail
        with pytest.raises(ValueError, match="Admin password is too weak"):
            save_user_handler({
                "user_id": "admin_user",
                "display_name": "Admin User",
                "password": "P@ss",
                "is_admin": 1
            }, auth=auth)

        # 4. Admin user with score 4 password ("Password123") should succeed
        save_user_handler({
            "user_id": "admin_user",
            "display_name": "Admin User",
            "password": "Password123",
            "is_admin": 1
        }, auth=auth)

        admin_auth = _admin_session(db_path)

        # 5. Changing admin password to a weak one via change_password_handler should fail
        with pytest.raises(ValueError, match="Admin password is too weak"):
            change_password_handler(
                user_id="admin_user",
                password="123",
                requester_id="admin_user"
            )

        # 6. Clearing an admin password should fail
        with pytest.raises(ValueError, match="Admin password cannot be empty"):
            change_password_handler(
                user_id="admin_user",
                password="",
                requester_id="admin_user"
            )

        # 7. Saving an empty admin password should fail
        with pytest.raises(ValueError, match="Admin password cannot be empty"):
            save_user_handler(
                user={
                    "user_id": "admin_user",
                    "display_name": "Admin User",
                    "password": "",
                    "is_admin": 1,
                    "requester_id": "admin_user",
                },
                auth=admin_auth,
            )

        # 8. Changing admin password to a strong one via change_password_handler should succeed
        change_password_handler(
            user_id="admin_user",
            password="StrongPassword!123",
            requester_id="admin_user"
        )


def test_save_user_handler_admin_forces_no_passwordless(tmp_path: Path) -> None:
    """Admin accounts can never be flagged for passwordless login: saving an
    admin with allow_passwordless_login=1 must persist 0."""
    db_path = tmp_path / "admin_passwordless_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.services import (
        list_users_handler,
        save_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path), patch(
        "mfdb.store.database_resolver.resolve_database_path", return_value=db_path
    ):
        list_users_handler()
        auth = _default_auth(db_path)

        # Admin created with the passwordless flag set -> flag is forced off.
        res = save_user_handler({
            "user_id": "admin2",
            "display_name": "Admin Two",
            "password": "Password123",
            "is_admin": 1,
            "allow_passwordless_login": 1,
        }, auth=auth)
        admin2 = next(u for u in res["users"] if u["user_id"] == "admin2")
        assert not admin2.get("allow_passwordless_login")

        # A non-admin keeps the passwordless flag.
        res = save_user_handler({
            "user_id": "kiosk",
            "display_name": "Kiosk",
            "is_admin": 0,
            "allow_passwordless_login": 1,
        }, auth=auth)
        kiosk = next(u for u in res["users"] if u["user_id"] == "kiosk")
        assert kiosk.get("allow_passwordless_login")


def test_change_password_permissions(tmp_path: Path) -> None:
    db_path = tmp_path / "user_perms_pw_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.password_services import change_password_handler
    from mfdb.admin.backend.services import (
        save_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path), patch(
        "mfdb.store.database_resolver.resolve_database_path", return_value=db_path
    ):
        # Initialize DB — user_default is already admin with password "admin"
        from mfdb.admin.backend.services import list_users_handler
        list_users_handler()
        admin_auth = _default_auth(db_path)

        # Create two normal users via admin
        save_user_handler(
            user={
                "user_id": "user1",
                "display_name": "User One",
                "password": "User1Password",
                "is_admin": 0,
            },
            auth=admin_auth,
        )
        save_user_handler(
            user={
                "user_id": "user2",
                "display_name": "User Two",
                "password": "User2Password",
                "is_admin": 0,
            },
            auth=admin_auth,
        )

        # 1. Non-admin user can change their own password
        change_password_handler(
            user_id="user1",
            password="NewUser1Password",
            requester_id="user1"
        )

        # 2. Non-admin user cannot change another user's password
        with pytest.raises(ValueError, match="Non-admin users can only change their own password"):
            change_password_handler(
                user_id="user2",
                password="HackedUser2Password",
                requester_id="user1"
            )

        # 3. Admin can change another user's password
        change_password_handler(
            user_id="user2",
            password="AdminResetPassword",
            requester_id="user_default"
        )


def test_guest_user_exists(tmp_path: Path) -> None:
    """Guest user is bootstrapped with allow_passwordless_login=1."""
    db_path = tmp_path / "guest_test.db"
    with MFDatabase(db_path) as db:
        row = db.conn.execute(
            "SELECT user_id, is_admin, allow_passwordless_login, password_hash FROM flr_sample_users WHERE user_id = 'guest'"
        ).fetchone()
        assert row is not None, "guest user should exist"
        assert row["is_admin"] == 0
        assert row["allow_passwordless_login"] == 1
        assert row["password_hash"] is None


def test_guest_login_passwordless(tmp_path: Path) -> None:
    """Guest can log in without a password."""
    db_path = tmp_path / "guest_login.db"

    from unittest.mock import patch

    from mfdb.admin.backend.password_services import login_handler

    # Initialize DB so guest user exists
    with MFDatabase(db_path) as db:
        pass

    with patch("mfdb.store.database_resolver.resolve_database_path", return_value=db_path):
        res = login_handler("guest")
        assert res["authenticated"] is True
        assert res["user"]["user_id"] == "guest"
        assert res["user"]["is_admin"] is False


def test_allow_passwordless_login_flag(tmp_path: Path) -> None:
    """Non-admin user with allow_passwordless_login=1 can login without password."""
    db_path = tmp_path / "passwdless_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.password_services import login_handler
    from mfdb.admin.backend.services import (
        list_users_handler,
        save_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path), patch(
        "mfdb.store.database_resolver.resolve_database_path", return_value=db_path
    ):
        list_users_handler()
        auth = _default_auth(db_path)

        # Create a user with allow_passwordless_login=1 and a password
        save_user_handler({
            "user_id": "passwdless_user",
            "display_name": "Passwordless User",
            "password": "some_password",
            "is_admin": 0,
            "allow_passwordless_login": 1,
        }, auth=auth)

        # Should be able to login without password
        res = login_handler("passwdless_user")
        assert res["authenticated"] is True
        assert res["user"]["user_id"] == "passwdless_user"

        # Should also login with password
        res = login_handler("passwdless_user", "some_password")
        assert res["authenticated"] is True


def test_allow_passwordless_login_can_be_disabled(tmp_path: Path) -> None:
    """Saving allow_passwordless_login=0 disables passwordless login."""
    db_path = tmp_path / "passwdless_disable_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.password_services import login_handler
    from mfdb.admin.backend.services import (
        list_users_handler,
        save_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path), patch(
        "mfdb.store.database_resolver.resolve_database_path", return_value=db_path
    ):
        list_users_handler()
        auth = _default_auth(db_path)

        save_user_handler({
            "user_id": "toggle_autologin_user",
            "display_name": "Toggle Autologin User",
            "password": "some_password",
            "allow_passwordless_login": 1,
        }, auth=auth)
        assert login_handler("toggle_autologin_user")["authenticated"] is True

        save_user_handler({
            "user_id": "toggle_autologin_user",
            "allow_passwordless_login": 0,
        }, auth=auth)

        res = login_handler("toggle_autologin_user")
        assert res["authenticated"] is False
        assert login_handler("toggle_autologin_user", "some_password")["authenticated"] is True


def test_allow_passwordless_login_admin_denied_without_password(tmp_path: Path) -> None:
    """Admins can never log in without a password, even with the passwordless
    flag; the flag itself is also forced off when an admin is saved."""
    db_path = tmp_path / "admin_passwdless_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.password_services import login_handler
    from mfdb.admin.backend.services import (
        save_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path), patch(
        "mfdb.store.database_resolver.resolve_database_path", return_value=db_path
    ):
        from mfdb.admin.backend.services import list_users_handler
        list_users_handler()
        auth = _default_auth(db_path)

        # Create an admin user; the requested allow_passwordless_login=1 is
        # forced off by save_user_handler.
        res = save_user_handler({
            "user_id": "admin_passwdless",
            "display_name": "Admin Passwordless",
            "password": "StrongPassword123!",
            "is_admin": 1,
            "allow_passwordless_login": 1,
        }, auth=auth)
        saved = next(u for u in res["users"] if u["user_id"] == "admin_passwdless")
        assert not saved.get("allow_passwordless_login")

        # Login without a password is denied for admins.
        res = login_handler("admin_passwdless")
        assert res["authenticated"] is False

        # Admin with correct password should succeed
        res = login_handler("admin_passwdless", "StrongPassword123!")
        assert res["authenticated"] is True
        assert res["user"]["is_admin"] is True


def test_guest_user_cannot_be_deleted(tmp_path: Path) -> None:
    """Guest user is protected from deletion."""
    db_path = tmp_path / "guest_delete_test.db"

    from unittest.mock import patch

    from mfdb.admin.backend.services import (
        delete_user_handler,
    )

    with patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path):
        from mfdb.admin.backend.services import list_users_handler
        list_users_handler()
        auth = _default_auth(db_path)

        with pytest.raises(ValueError, match="built-in user.*guest"):
            delete_user_handler("guest", auth=auth)


def test_migration_repairs_legacy_schema_marked_without_deleted_at(tmp_path: Path) -> None:
    """A DB marked v23 but missing deleted_at is repaired on open."""
    import sqlite3

    from mfdb.repository import MFDatabase

    db_path = tmp_path / "legacy_user_schema.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE flr_sample_users (user_id TEXT PRIMARY KEY, display_name TEXT, user_uuid TEXT)")
        conn.execute("INSERT INTO flr_sample_users (user_id, display_name) VALUES (?, ?)", ("legacy", "Legacy User"))
        conn.execute(
            "CREATE TABLE mfdb_vocabulary ("
            "field_name TEXT NOT NULL, value TEXT NOT NULL, display_name TEXT, "
            "description TEXT, is_builtin INTEGER DEFAULT 0, is_active INTEGER DEFAULT 1, "
            "created_at TEXT, updated_at TEXT, deleted_at TEXT, PRIMARY KEY (field_name, value))"
        )
        conn.execute("CREATE TABLE mfdb_schema_version (version INTEGER)")
        conn.execute("INSERT INTO mfdb_schema_version (version) VALUES (23)")
        conn.commit()
    finally:
        conn.close()

    db = MFDatabase(db_path)
    try:
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(flr_sample_users)")}
        assert "deleted_at" in cols, sorted(cols)
        users = db.get_users()
        assert [row["user_id"] for row in users] == ["legacy"]
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(flr_sample_users)")}
        assert "deleted_at" in cols
    finally:
        db.close()

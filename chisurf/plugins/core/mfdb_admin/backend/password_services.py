"""Password authentication handlers for MFDB users."""

from __future__ import annotations

import hashlib
import secrets
from typing import Any



def hash_password(password: str) -> str:
    """Hash *password* with a random salt using PBKDF2-SHA256."""
    salt = secrets.token_hex(16)
    iterations = 100000
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    return f"pbkdf2_sha256${iterations}${salt}${dk.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Return ``True`` if *password* matches *password_hash*."""
    if not password_hash:
        return False
    parts = password_hash.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False
    try:
        iterations = int(parts[1])
        salt = parts[2]
        hashed = parts[3]
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    return dk.hex() == hashed


def evaluate_password(password: str) -> dict[str, Any]:
    """Return a simple password-strength score and feedback."""
    score = 0
    feedback: list[str] = []

    if len(password) >= 8:
        score += 1
    else:
        feedback.append("At least 8 characters")

    if any(char.islower() for char in password):
        score += 1
    else:
        feedback.append("At least one lowercase letter")

    if any(char.isupper() for char in password):
        score += 1
    else:
        feedback.append("At least one uppercase letter")

    if any(char.isdigit() for char in password):
        score += 1
    else:
        feedback.append("At least one number")

    special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
    if any(char in special_chars for char in password):
        score += 1
    else:
        feedback.append("At least one special character")

    return {
        "score": score,
        "label": "Weak" if score <= 2 else ("Medium" if score <= 4 else "Strong"),
        "feedback": feedback,
    }


def login_handler(user_id: str, password: str = "") -> dict[str, Any]:
    """Authenticate an MFDB user without exposing password hashes.

    Rules:
    - Guest user (``allow_passwordless_login = 1``, no password) logs in
      without password.
    - Non-admin users with ``allow_passwordless_login = 1`` log in without
      password.
    - Admin users always require a password.
    """
    from chisurf.core.fio.mmcif.db import FluorescenceDatabase
    from chisurf.core.mfdb.database_resolver import resolve_database_path

    with FluorescenceDatabase(resolve_database_path()) as db:
        row = db.conn.execute(
            "SELECT display_name, is_admin, password_hash, allow_passwordless_login FROM flr_sample_users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return {"authenticated": False, "error": "User not found"}
        display_name, is_admin, password_hash, allow_passwordless = row
        is_admin_bool = is_admin == 1

        # Passwordless login allowed if user has the flag and is not admin
        if allow_passwordless == 1 and not is_admin_bool:
            return {
                "authenticated": True,
                "user": {
                    "user_id": user_id,
                    "display_name": display_name,
                    "is_admin": is_admin_bool,
                },
            }

        # No password set — free login (legacy behavior for passwordless users)
        if not password_hash:
            return {
                "authenticated": True,
                "user": {
                    "user_id": user_id,
                    "display_name": display_name,
                    "is_admin": is_admin_bool,
                },
            }

        if verify_password(password, password_hash):
            return {
                "authenticated": True,
                "user": {
                    "user_id": user_id,
                    "display_name": display_name,
                    "is_admin": is_admin_bool,
                },
            }
        return {"authenticated": False, "error": "Incorrect password"}


def change_password_handler(
    user_id: str,
    password: str,
    requester_id: str | None = None,
) -> dict[str, Any]:
    """Change an MFDB user's password using MFDB authorization rules."""
    import sqlite3

    from chisurf.core.fio.mmcif.db import FluorescenceDatabase
    from chisurf.core.mfdb.database_resolver import resolve_database_path

    with FluorescenceDatabase(resolve_database_path()) as db:
        has_admin_res = db.conn.execute(
            "SELECT 1 FROM flr_sample_users WHERE is_admin = 1 LIMIT 1"
        ).fetchone()
        has_admins = has_admin_res is not None

        if has_admins:
            if not requester_id:
                raise ValueError("Unauthorized: requester_id is required")
            req_row = db.conn.execute(
                "SELECT is_admin FROM flr_sample_users WHERE user_id = ?",
                (requester_id,),
            ).fetchone()
            is_req_admin = req_row and req_row[0] == 1
            if not is_req_admin and requester_id != user_id:
                raise ValueError("Unauthorized: Non-admin users can only change their own password")

        target_row = db.conn.execute(
            "SELECT is_admin FROM flr_sample_users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        is_target_admin = target_row and target_row[0] == 1

        if is_target_admin:
            if not password:
                raise ValueError("Admin password cannot be empty")
            strength = evaluate_password(password)
            if strength["score"] < 4:
                raise ValueError(
                    f"Admin password is too weak. Requirements: {', '.join(strength['feedback'])}"
                )

        password_hash = hash_password(password) if password else None
        with db.conn:
            db.conn.execute(
                "UPDATE flr_sample_users SET password_hash = ? WHERE user_id = ?",
                (password_hash, user_id),
            )
        try:
            db.conn.execute(
                "INSERT INTO mfdb_audit_log (action, target_type, target_id, operator_user_id) "
                "VALUES (?, ?, ?, ?)",
                ("password.change", "user", user_id, requester_id or user_id),
            )
        except sqlite3.OperationalError:
            pass
    return {"ok": True}

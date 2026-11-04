"""Application configuration loaded exclusively from environment variables.

No secret is read at import time.  This makes the package safe to import in
tests and lets deployments provide credentials through their normal secret
manager.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    value = default if raw in (None, "") else int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def load_config(instance_path: Path) -> dict[str, Any]:
    """Build Flask configuration without exposing LLM credentials."""

    environment = os.getenv("APP_ENV", "development").strip().lower()
    if environment not in {"development", "production"}:
        raise RuntimeError("APP_ENV must be either development or production")

    raw_secret = os.getenv("APP_SECRET_KEY")
    secret = raw_secret.strip() if raw_secret and raw_secret.strip() else "development-only-change-me"
    insecure_secrets = {
        "development-only-change-me",
        "replace-with-a-long-random-value",
    }
    if environment == "production" and (
        secret in insecure_secrets or len(secret) < 32
    ):
        raise RuntimeError(
            "APP_SECRET_KEY must be a non-placeholder value of at least 32 characters "
            "when APP_ENV=production"
        )

    session_cookie_secure = _as_bool(
        os.getenv("SESSION_COOKIE_SECURE"), environment == "production"
    )
    if environment == "production" and not session_cookie_secure:
        raise RuntimeError(
            "SESSION_COOKIE_SECURE must be true when APP_ENV=production"
        )

    database_path = Path(
        os.getenv("DATABASE_PATH", str(instance_path / "venturebridge.sqlite3"))
    ).expanduser()
    anchor_plan_path = (os.getenv("ANCHOR_PLAN_PATH") or "").strip()
    anchor_plan_expected_hash = (
        os.getenv("ANCHOR_PLAN_EXPECTED_HASH") or ""
    ).strip()
    anchor_plan_required = _as_bool(
        os.getenv("ANCHOR_PLAN_REQUIRED"), environment == "production"
    )
    if environment == "production" and not anchor_plan_required:
        raise RuntimeError(
            "ANCHOR_PLAN_REQUIRED cannot be false when APP_ENV=production"
        )
    if anchor_plan_path and not anchor_plan_expected_hash:
        raise RuntimeError(
            "ANCHOR_PLAN_EXPECTED_HASH is required whenever ANCHOR_PLAN_PATH is set"
        )
    if anchor_plan_expected_hash and not anchor_plan_path:
        raise RuntimeError(
            "ANCHOR_PLAN_PATH is required whenever ANCHOR_PLAN_EXPECTED_HASH is set"
        )
    if anchor_plan_expected_hash and (
        len(anchor_plan_expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in anchor_plan_expected_hash)
    ):
        raise RuntimeError(
            "ANCHOR_PLAN_EXPECTED_HASH must be a lowercase 64-character SHA-256 digest"
        )
    # A production deployment may start without either value only as a
    # fail-closed commissioning instance.  This is needed to create consented
    # pseudonymous workspaces and export their allocation-unit IDs before the
    # first queue plan exists.  Routes and the scheduler block all simulation
    # delivery until a path/hash pair is registered and the app is restarted.

    return {
        "ENVIRONMENT": environment,
        "SECRET_KEY": secret,
        "DATABASE": str(database_path),
        "MAX_CONTENT_LENGTH": _as_int("MAX_REQUEST_BYTES", 1_000_000, 10_000),
        "MAX_PROFILE_TEXT_CHARS": _as_int("MAX_PROFILE_TEXT_CHARS", 8_000, 100),
        "MAX_DECISION_TEXT_CHARS": _as_int("MAX_DECISION_TEXT_CHARS", 6_000, 100),
        "TRAINING_INTERVAL_HOURS": _as_int("TRAINING_INTERVAL_HOURS", 0, 0),
        "ANCHOR_EVERY_TURNS": _as_int("ANCHOR_EVERY_TURNS", 4, 2),
        "ANCHOR_PLAN_PATH": anchor_plan_path,
        "ANCHOR_PLAN_EXPECTED_HASH": anchor_plan_expected_hash,
        "ANCHOR_PLAN_REQUIRED": anchor_plan_required,
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": session_cookie_secure,
        "JSON_AS_ASCII": False,
    }

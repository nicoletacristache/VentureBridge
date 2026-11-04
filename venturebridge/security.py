"""Small security controls that do not require a third-party auth framework."""

from __future__ import annotations

import hmac
import os
import secrets
import stat
from pathlib import Path
from typing import Any

from flask import Flask, abort, request, session


def enforce_private_directory(path: str | Path) -> Path:
    """Create/harden an application-owned directory to POSIX mode 0700."""

    directory = Path(path).expanduser()
    if directory.is_symlink():
        raise RuntimeError(f"Private storage directory cannot be a symlink: {directory}")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not directory.is_dir():
        raise RuntimeError(f"Private storage path is not a directory: {directory}")
    if os.name == "posix":
        os.chmod(directory, stat.S_IRWXU)
        actual = stat.S_IMODE(directory.stat().st_mode)
        if actual & 0o077:
            raise RuntimeError(f"Cannot enforce private directory permissions: {directory}")
    return directory


def prepare_private_database_path(
    database: str | Path, *, private_root: str | Path
) -> Path | None:
    """Prepare a SQLite parent without changing permissions on shared parents.

    The configured database parent is hardened when it is inside the private
    instance root or when this function creates it.  A shared existing parent
    such as ``/tmp`` is never chmod'ed.
    """

    if str(database) == ":memory:":
        return None
    database_path = Path(database).expanduser()
    root = Path(private_root).expanduser().resolve(strict=False)
    parent = database_path.parent
    parent_existed = parent.exists()
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    resolved_parent = parent.resolve(strict=False)
    try:
        inside_private_root = resolved_parent == root or resolved_parent.is_relative_to(root)
    except AttributeError:  # pragma: no cover - supported Python has is_relative_to
        inside_private_root = str(resolved_parent).startswith(str(root) + os.sep)
    if inside_private_root or not parent_existed:
        enforce_private_directory(parent)
    if database_path.exists():
        enforce_private_file(database_path)
    return database_path


def enforce_private_file(path: str | Path) -> Path:
    """Harden an existing regular file to POSIX mode 0600."""

    file_path = Path(path).expanduser()
    if file_path.is_symlink():
        raise RuntimeError(f"Private data file cannot be a symlink: {file_path}")
    if not file_path.exists() or not file_path.is_file():
        raise RuntimeError(f"Private data file does not exist or is not regular: {file_path}")
    if os.name == "posix":
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        actual = stat.S_IMODE(file_path.stat().st_mode)
        if actual & 0o177:
            raise RuntimeError(f"Cannot enforce private file permissions: {file_path}")
    return file_path


def csrf_token() -> str:
    token = session.get("csrf_token")
    if not isinstance(token, str):
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def _validate_csrf() -> None:
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return
    expected = session.get("csrf_token")
    supplied = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
    if not isinstance(expected, str) or not isinstance(supplied, str):
        abort(400, description="A valid request-verification token is missing. Refresh the page and try again.")
    if not hmac.compare_digest(expected, supplied):
        abort(400, description="The request-verification token does not match. Refresh the page and try again.")


def _security_headers(response: Any) -> Any:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; style-src 'self'; script-src 'self'; "
        "img-src 'self' data:; form-action 'self'; frame-ancestors 'none'",
    )
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if response.mimetype == "text/html":
        response.headers.setdefault("Cache-Control", "no-store")
    return response


def init_app(app: Flask) -> None:
    app.jinja_env.globals["csrf_token"] = csrf_token
    app.before_request(_validate_csrf)
    app.after_request(_security_headers)

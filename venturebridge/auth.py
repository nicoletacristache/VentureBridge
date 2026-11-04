"""Minimal pseudonymous workspace authentication for the teaching prototype."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar, cast

from flask import g, redirect, request, session, url_for

from .repository import get_user


View = TypeVar("View", bound=Callable[..., Any])


def load_logged_in_user() -> None:
    user_id = session.get("user_id")
    g.user = get_user(user_id) if isinstance(user_id, str) else None
    if g.user is None:
        session.pop("user_id", None)


def login_required(view: View) -> View:
    @wraps(view)
    def wrapped_view(*args: Any, **kwargs: Any) -> Any:
        if g.get("user") is None:
            return redirect(url_for("home", next=request.path))
        return view(*args, **kwargs)

    return cast(View, wrapped_view)


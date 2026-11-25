"""Accounts for running Lesson Talk as a shared team server.

Off by default: on a single computer the app binds to 127.0.0.1 and needs no
login. When it is reachable by other people, every request needs HTTP Basic
credentials for an account created with `lesson-talk add-user NAME`.

Passwords are generated, never chosen, and stored only as PBKDF2-SHA256
hashes in $LESSONTALK_USERS (default ~/.config/lesson-talk/users.json, mode
600). Basic auth is only safe over HTTPS; the deployment puts the app behind
Tailscale Funnel, which terminates TLS.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

USERS = os.path.expanduser(
    os.environ.get("LESSONTALK_USERS", "~/.config/lesson-talk/users.json"))
ITER = 240_000
_lock = threading.Lock()
_ok_cache: dict[str, float] = {}      # digest of a verified header -> expiry
_fails: dict[str, list[float]] = {}   # client -> failure times
MAX_FAILS, WINDOW_S, CACHE_S = 10, 600, 600


def _load() -> dict:
    try:
        with open(USERS) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def _save(users: dict) -> None:
    os.makedirs(os.path.dirname(USERS), mode=0o700, exist_ok=True)
    tmp = USERS + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(users, fh, indent=1)
    os.replace(tmp, USERS)
    with _lock:
        _ok_cache.clear()  # a removed user must stop working at once


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITER).hex()


def enabled() -> bool:
    return bool(_load())


def users() -> list[str]:
    return sorted(_load())


def add_user(name: str) -> str:
    name = name.strip().lower()
    if not name or not name.replace("-", "").replace(".", "").isalnum() \
            or len(name) > 40:
        raise ValueError("use letters, digits, dots or hyphens for the name")
    pw = secrets.token_urlsafe(15)
    salt = secrets.token_bytes(16)
    u = _load()
    u[name] = {"salt": salt.hex(), "hash": _hash(pw, salt),
               "created": dt.datetime.now().isoformat(timespec="seconds")}
    _save(u)
    return pw


def remove_user(name: str) -> bool:
    u = _load()
    if u.pop(name.strip().lower(), None) is None:
        return False
    _save(u)
    return True


def check(header: str | None, client: str) -> tuple[str | None, bool]:
    """(username or None, locked_out). Verifying a PBKDF2 hash takes ~0.1 s,
    and the browser re-sends the header on every request (audio seeking
    makes many), so a verified header is cached for ten minutes."""
    now = time.time()
    with _lock:
        recent = [t for t in _fails.get(client, []) if now - t < WINDOW_S]
        _fails[client] = recent
        if len(recent) >= MAX_FAILS:
            return None, True
    if not header or not header.startswith("Basic "):
        return None, False
    key = hashlib.sha256(header.encode()).hexdigest()
    try:
        name, _, pw = base64.b64decode(header[6:]).decode().partition(":")
    except (ValueError, UnicodeDecodeError):
        name, pw = "", ""
    name = name.strip().lower()
    with _lock:
        if _ok_cache.get(key, 0) > now:
            return name, False
    rec = _load().get(name)
    good = False
    if rec:
        good = hmac.compare_digest(
            _hash(pw, bytes.fromhex(rec["salt"])), rec["hash"])
    else:
        _hash(pw, b"0" * 16)  # same cost whether or not the name exists
    with _lock:
        if good:
            _ok_cache[key] = now + CACHE_S
        else:
            _fails.setdefault(client, []).append(now)
    return (name if good else None), False

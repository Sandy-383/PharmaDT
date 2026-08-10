"""OAuth2 password flow — NFR-04's API authentication requirement.

Tokens are signed with the same HMAC secret the app is configured with, and
carry an expiry. Deliberately *not* a rolling opaque token in a dict: an
in-memory session store would not survive the reload that `make api` runs with,
so every code edit would silently log the demo out mid-presentation.

The demo credentials exist because this is a simulation with no user directory.
That is stated plainly rather than dressed up as a security model — in
deployment this delegates to the consortium's identity provider, and the
`require_user` dependency is the only thing that would change.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from pharmadt.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)

#: Demo operators. A real deployment resolves these against the consortium's
#: identity provider; only this mapping and `authenticate` would change.
DEMO_USERS: dict[str, dict[str, str]] = {
    "operator": {"password": "pharmadt", "role": "operator"},
    "auditor": {"password": "pharmadt", "role": "auditor"},
}

TOKEN_TTL_SECONDS = 8 * 3600


def _secret() -> bytes:
    return settings.api_secret_key.get_secret_value().encode()


def _sign(payload: bytes) -> str:
    return urlsafe_b64encode(
        hmac.new(_secret(), payload, hashlib.sha256).digest()
    ).decode().rstrip("=")


def create_token(username: str, role: str) -> str:
    """Issue a signed, expiring bearer token."""
    body = json.dumps(
        {"sub": username, "role": role, "exp": int(time.time()) + TOKEN_TTL_SECONDS},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"{urlsafe_b64encode(body).decode().rstrip('=')}.{_sign(body)}"


def decode_token(token: str) -> dict[str, Any] | None:
    """Verify and decode. Returns None for anything not currently valid."""
    try:
        encoded, signature = token.split(".", 1)
        body = urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, TypeError):
        return None

    # Constant-time compare: a plain == leaks how much of the signature matched.
    if not hmac.compare_digest(signature, _sign(body)):
        return None

    claims = json.loads(body)
    if claims.get("exp", 0) < time.time():
        return None
    return claims


def authenticate(username: str, password: str) -> dict[str, str] | None:
    user = DEMO_USERS.get(username)
    if user is None or not hmac.compare_digest(user["password"], password):
        return None
    return {"username": username, "role": user["role"]}


async def require_user(token: str | None = Depends(oauth2_scheme)) -> dict[str, Any]:
    """Dependency for every mutating or sensitive route."""
    claims = decode_token(token) if token else None
    if claims is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to use this endpoint.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return claims

"""Role-based access for the four stakeholder groups in the project scope.

    authority  Municipal corporations / road maintenance - full dashboard,
               may change defect status and trigger ingestion.
    transport  Traffic & transport departments - read + analytics.
    planner    Smart-city infrastructure planners - read + analytics.
    citizen    Drivers & commuters - may submit a crowdsourced photo only.

HONESTY NOTE (also in docs/SECURITY.md): this is demo-grade authentication -
static bearer tokens compared in constant time, no user store, no password
hashing, no token expiry, no refresh. It exists to demonstrate that the system
is *designed* around distinct stakeholder roles, which is a named scope
requirement. A production deployment must replace it with a real identity
provider (OAuth2/OIDC against the municipality's directory). Auth is OFF by
default (`AUTH_ENABLED=false`) so the demo and the tests need no setup.
"""

from __future__ import annotations

import hmac
from typing import Iterable

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings

ROLE_ANY = ("authority", "transport", "planner", "citizen")
ROLE_READ = ("authority", "transport", "planner")
ROLE_WRITE = ("authority",)


def _extract_token(authorization: str | None, api_key: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return api_key


def resolve_role(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> str:
    """Resolve the caller's role, or ``authority`` when auth is disabled."""
    if not settings.auth_enabled:
        return "authority"

    token = _extract_token(authorization, x_api_key)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing credentials. Send `Authorization: Bearer <token>`.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison against every known token, so a timing side
    # channel cannot reveal which prefix was correct.
    for known, role in settings.api_tokens.items():
        if hmac.compare_digest(token, known):
            return role

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials."
    )


def require_roles(allowed: Iterable[str]):
    """Dependency factory gating an endpoint on a set of roles."""
    allowed = tuple(allowed)

    def _guard(role: str = Depends(resolve_role)) -> str:
        if role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{role}' is not permitted here. "
                    f"Required: {', '.join(allowed)}."
                ),
            )
        return role

    return _guard


require_read = require_roles(ROLE_READ)
require_write = require_roles(ROLE_WRITE)
require_any = require_roles(ROLE_ANY)

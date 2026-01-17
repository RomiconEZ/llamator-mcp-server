from __future__ import annotations

from fastapi import HTTPException

from llamator_mcp_server.config.settings import Settings


async def require_api_key(settings: Settings, x_api_key: str | None) -> None:
    """
    Enforce API key access control.

    If ``settings.api_key`` is empty, authentication is disabled.

    :param settings: Application settings.
    :param x_api_key: Value of the ``X-API-Key`` header.
    :return: None.
    :raises HTTPException: If a key is configured and does not match.
    """
    expected: str | None = settings.api_key
    if expected is None or expected == "":
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

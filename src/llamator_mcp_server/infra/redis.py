from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from arq.connections import RedisSettings
from redis.asyncio import Redis


@dataclass(frozen=True, slots=True)
class RedisConfig:
    """
    Normalized Redis configuration.

    :param dsn: Redis DSN (redis://host:port/db).
    """

    dsn: str


def parse_redis_settings(dsn: str) -> RedisSettings:
    """
    Convert Redis DSN into ARQ RedisSettings.

    :param dsn: Redis DSN.
    :return: RedisSettings for ARQ.
    :raises ValueError: If DSN is invalid.
    """
    parsed = urlparse(dsn)
    if parsed.scheme not in ("redis", "rediss"):
        raise ValueError("Only redis:// and rediss:// schemes are supported.")
    if not parsed.hostname:
        raise ValueError("Redis DSN must include hostname.")

    database: int = 0
    if parsed.path and parsed.path != "/":
        raw_db: str = parsed.path.lstrip("/")
        if not raw_db.isdigit():
            raise ValueError("Redis DSN database must be an integer.")
        database = int(raw_db)

    return RedisSettings(
        host=parsed.hostname,
        port=parsed.port or 6379,
        database=database,
        password=parsed.password,
        username=parsed.username,
        ssl=(parsed.scheme == "rediss"),
    )


def create_redis_client(dsn: str) -> Redis:
    """
    Create an asyncio Redis client.

    :param dsn: Redis DSN.
    :return: Redis client.
    """
    return Redis.from_url(dsn, decode_responses=True)

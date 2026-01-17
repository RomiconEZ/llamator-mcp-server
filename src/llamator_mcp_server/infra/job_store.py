from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from llamator_mcp_server.domain.models import JobStatus, LlamatorJobError, LlamatorJobInfo, LlamatorJobResult


def _utcnow() -> datetime:
    """
    Return the current UTC time.

    :return: Current datetime in UTC.
    """
    return datetime.now(timezone.utc)


def _build_error_notice(error_type: str, message: str) -> str:
    """
    Build a compact user-facing error notice.

    :param error_type: Exception type name.
    :param message: Exception message.
    :return: Formatted notice string.
    """
    msg: str = str(message)
    if msg:
        return f"{error_type}: {msg}"
    return f"{error_type}"


class JobStore:
    """
    Redis-backed job state store.

    :param redis: Redis client.
    :param ttl_seconds: Key TTL in seconds.
    """

    def __init__(self, redis: Redis, ttl_seconds: int) -> None:
        self._redis: Redis = redis
        self._ttl_seconds: int = ttl_seconds

    async def create(self, job_id: str, request_redacted: dict[str, object]) -> LlamatorJobInfo:
        """
        Create a job record.

        :param job_id: Job identifier.
        :param request_redacted: Request payload with redacted secrets.
        :return: LlamatorJobInfo.
        """
        now: datetime = _utcnow()
        info: LlamatorJobInfo = LlamatorJobInfo(
            job_id=job_id,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            request=request_redacted,
            result=None,
            error=None,
            error_notice=None,
        )
        await self._set(job_id, info)
        return info

    async def update_status(self, job_id: str, status: JobStatus) -> None:
        """
        Update job status.

        :param job_id: Job identifier.
        :param status: New status.
        :return: None.
        :raises KeyError: If job does not exist.
        """
        info: LlamatorJobInfo = await self.get(job_id)
        updated: LlamatorJobInfo = info.model_copy(update={"status": status, "updated_at": _utcnow()})
        await self._set(job_id, updated)

    async def set_result(self, job_id: str, aggregated: dict[str, dict[str, int]]) -> None:
        """
        Persist job successful result.

        :param job_id: Job identifier.
        :param aggregated: Aggregated metrics.
        :return: None.
        :raises KeyError: If job does not exist.
        """
        info: LlamatorJobInfo = await self.get(job_id)
        result: LlamatorJobResult = LlamatorJobResult(aggregated=aggregated, finished_at=_utcnow())
        updated: LlamatorJobInfo = info.model_copy(
            update={
                "status": JobStatus.SUCCEEDED,
                "updated_at": _utcnow(),
                "result": result,
                "error": None,
                "error_notice": None,
            }
        )
        await self._set(job_id, updated)

    async def set_error(self, job_id: str, error_type: str, message: str) -> None:
        """
        Persist job error result.

        :param job_id: Job identifier.
        :param error_type: Exception type name.
        :param message: Exception message.
        :return: None.
        :raises KeyError: If job does not exist.
        """
        info: LlamatorJobInfo = await self.get(job_id)
        error: LlamatorJobError = LlamatorJobError(error_type=error_type, message=message, occurred_at=_utcnow())
        error_notice: str = _build_error_notice(error_type=error_type, message=message)
        updated: LlamatorJobInfo = info.model_copy(
            update={
                "status": JobStatus.FAILED,
                "updated_at": _utcnow(),
                "error": error,
                "error_notice": error_notice,
            }
        )
        await self._set(job_id, updated)

    async def get(self, job_id: str) -> LlamatorJobInfo:
        """
        Get job state by id.

        :param job_id: Job identifier.
        :return: LlamatorJobInfo.
        :raises KeyError: If job does not exist.
        """
        key: str = self._key(job_id)
        raw: str | None = await self._redis.get(key)
        if raw is None:
            raise KeyError(f"Job not found: {job_id}")
        payload: dict[str, Any] = json.loads(raw)
        return LlamatorJobInfo.model_validate(payload)

    async def _set(self, job_id: str, info: LlamatorJobInfo) -> None:
        """
        Persist job state to Redis.

        :param job_id: Job identifier.
        :param info: LlamatorJobInfo to persist.
        :return: None.
        """
        key: str = self._key(job_id)
        raw: str = info.model_dump_json()
        await self._redis.set(key, raw, ex=self._ttl_seconds)

    @staticmethod
    def _key(job_id: str) -> str:
        """
        Build Redis key for a job.

        :param job_id: Job identifier.
        :return: Redis key string.
        """
        return f"llamator:job:{job_id}"

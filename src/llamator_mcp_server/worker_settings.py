from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from arq.connections import RedisSettings
from pydantic import TypeAdapter

from llamator_mcp_server.config.settings import Settings, settings
from llamator_mcp_server.domain.models import JobStatus, LlamatorRunConfig, OpenAIClientConfig, TestPlan
from llamator_mcp_server.domain.ports.artifacts_storage import ArtifactsStorage
from llamator_mcp_server.infra.artifacts import MinioArtifactsStorage, create_artifacts_storage
from llamator_mcp_server.infra.job_store import JobStore
from llamator_mcp_server.infra.llamator_runner import LlamatorRunner, ResolvedRun
from llamator_mcp_server.infra.redis import create_redis_client, parse_redis_settings
from llamator_mcp_server.utils.logging import LOGGER_NAME, configure_logging


def _utcnow() -> datetime:
    """
    Return the current UTC time.

    :return: Current datetime in UTC.
    """
    return datetime.now(timezone.utc)


def _validate_client_config(val: Any) -> OpenAIClientConfig:
    """
    Validate and parse an OpenAIClientConfig from an untyped payload.

    :param val: Raw payload value.
    :return: Parsed OpenAIClientConfig.
    :raises ValueError: If payload is not a dict or schema validation fails.
    """
    if not isinstance(val, dict):
        raise ValueError("ClientConfig payload must be an object.")
    adapter: TypeAdapter[Any] = TypeAdapter(OpenAIClientConfig)
    parsed: Any = adapter.validate_python(val)
    return parsed  # type: ignore[return-value]


def _validate_start_testing_result(val: Any) -> dict[str, dict[str, int]]:
    """
    Validate and parse LLAMATOR start_testing result payload.

    :param val: Raw payload value.
    :return: Parsed result.
    :raises ValueError: If payload is not a dict or schema validation fails.
    """
    if not isinstance(val, dict):
        raise ValueError("start_testing result must be an object.")
    adapter: TypeAdapter[Any] = TypeAdapter(dict[str, dict[str, int]])
    parsed: Any = adapter.validate_python(val)
    return parsed  # type: ignore[return-value]


def _is_empty_aggregated_result(aggregated: dict[str, dict[str, int]]) -> bool:
    """
    Check whether aggregated result is empty.

    :param aggregated: Aggregated result dict.
    :return: True if empty.
    """
    return not aggregated


def _safe_posix_relpath(val: str) -> PurePosixPath:
    """
    Validate a safe relative POSIX path.

    :param val: Raw path value.
    :return: Normalized PurePosixPath.
    :raises ValueError: If path is unsafe.
    """
    raw: str = str(val).strip()
    if not raw:
        raise ValueError("artifacts_path must be non-empty when provided.")
    p: PurePosixPath = PurePosixPath(raw)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("artifacts_path must be a safe relative path.")
    if str(p) in (".", ""):
        raise ValueError("artifacts_path must be a safe relative path.")
    return p


def _resolve_job_artifacts_root(settings_obj: Settings, job_id: str) -> Path:
    """
    Resolve base local artifacts directory for a job.

    :param settings_obj: Application settings.
    :param job_id: Job identifier.
    :return: Absolute base directory for the job.
    """
    base: Path = (Path(settings_obj.artifacts_root) / str(job_id)).resolve(strict=False)
    return base


def _resolve_local_artifacts_dir(settings_obj: Settings, job_id: str, user_cfg: LlamatorRunConfig | None) -> Path:
    """
    Resolve local artifacts directory for a job, optionally using a user-provided relative path.

    The returned directory is always inside the job artifacts root.

    :param settings_obj: Application settings.
    :param job_id: Job identifier.
    :param user_cfg: Optional user run configuration.
    :return: Absolute local artifacts directory.
    :raises ValueError: If user artifacts_path escapes the job directory.
    """
    base: Path = _resolve_job_artifacts_root(settings_obj, job_id)

    if user_cfg is None or user_cfg.artifacts_path is None:
        return base

    rel: PurePosixPath = _safe_posix_relpath(user_cfg.artifacts_path)
    candidate: Path = (base / Path(*rel.parts)).resolve(strict=False)

    if base != candidate and base not in candidate.parents:
        raise ValueError("artifacts_path escaped job artifacts root.")

    return candidate


def _merge_llamator_run_config(
    settings_obj: Settings, job_id: str, user_cfg: LlamatorRunConfig | None
) -> dict[str, Any]:
    """
    Merge user run configuration with defaults and resolve local artifacts path.

    :param settings_obj: Application settings.
    :param job_id: Job identifier.
    :param user_cfg: Optional user config.
    :return: Effective LLAMATOR config dict.
    """
    effective: dict[str, Any] = {}

    enable_logging: bool = (
        True if user_cfg is None or user_cfg.enable_logging is None else bool(user_cfg.enable_logging)
    )
    enable_reports: bool = (
        False if user_cfg is None or user_cfg.enable_reports is None else bool(user_cfg.enable_reports)
    )
    debug_level: int = 1 if user_cfg is None or user_cfg.debug_level is None else int(user_cfg.debug_level)
    report_language: str = (
        settings_obj.report_language
        if user_cfg is None or user_cfg.report_language is None
        else user_cfg.report_language
    )

    effective["enable_logging"] = enable_logging
    effective["enable_reports"] = enable_reports
    effective["debug_level"] = debug_level
    effective["report_language"] = report_language

    artifacts_dir: Path = _resolve_local_artifacts_dir(settings_obj=settings_obj, job_id=job_id, user_cfg=user_cfg)
    effective["artifacts_path"] = str(artifacts_dir)

    return effective


def _try_parse_run_config(val: Any) -> LlamatorRunConfig | None:
    """
    Validate and parse a LlamatorRunConfig from an untyped payload.

    :param val: Raw payload value.
    :return: Parsed LlamatorRunConfig or None.
    :raises ValueError: If payload is not a dict when provided or schema validation fails.
    """
    if val is None:
        return None
    if not isinstance(val, dict):
        raise ValueError("run_config payload must be an object.")
    return LlamatorRunConfig.model_validate(val)


def _cleanup_expired_local_artifacts(settings_obj: Settings, logger: logging.Logger) -> None:
    """
    Cleanup expired local artifacts directories.

    :param settings_obj: Application settings.
    :param logger: Logger.
    :return: None.
    """
    root: Path = Path(settings_obj.artifacts_root).resolve(strict=False)
    ttl_s: int = int(settings_obj.artifacts_local_ttl_seconds)
    if ttl_s < 1:
        return

    now: float = time.time()
    try:
        entries: list[Path] = [p for p in root.iterdir()]
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.warning(f"Worker local artifacts cleanup failed root={root} error={type(exc).__name__}: {exc}")
        return

    for p in entries:
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        except Exception:
            continue

        age_s: float = now - float(st.st_mtime)
        if age_s < float(ttl_s):
            continue

        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
            logger.info(f"Worker local artifacts cleaned path={p} age_seconds={age_s:.2f}")
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning(f"Worker local artifacts cleanup failed path={p} error={type(exc).__name__}: {exc}")


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    """
    Immutable typed view over ARQ worker context.

    This class centralizes access to required dependencies and prevents
    scattered string-key lookups across the code.
    """

    settings: Settings
    logger: logging.Logger
    store: JobStore
    artifacts: ArtifactsStorage

    @classmethod
    def from_ctx(cls, ctx: dict[str, Any]) -> _ExecutionContext:
        """
        Build _ExecutionContext from ARQ ctx dict.

        :param ctx: ARQ worker context dict.
        :return: _ExecutionContext.
        """
        return cls(
            settings=ctx["settings"],
            logger=ctx["logger"],
            store=ctx["store"],
            artifacts=ctx["artifacts_storage"],
        )


@dataclass(frozen=True, slots=True)
class _RunInputs:
    """
    Validated job inputs required to run LLAMATOR.
    """

    job_id: str
    attack_model: OpenAIClientConfig
    tested_model: OpenAIClientConfig
    judge_model: OpenAIClientConfig
    plan: TestPlan
    user_run_config: LlamatorRunConfig | None

    @classmethod
    def from_payload(cls, job_id: str, payload: dict[str, Any]) -> _RunInputs:
        """
        Parse and validate inputs from a job payload.

        :param job_id: Job identifier.
        :param payload: Job payload dict.
        :return: _RunInputs.
        :raises ValueError: If schema validation fails.
        """
        attack_model: OpenAIClientConfig = _validate_client_config(payload["attack_model"])
        tested_model: OpenAIClientConfig = _validate_client_config(payload["tested_model"])
        judge_model: OpenAIClientConfig = _validate_client_config(payload["judge_model"])
        plan: TestPlan = TestPlan.model_validate(payload["plan"])
        user_run_config: LlamatorRunConfig | None = _try_parse_run_config(payload.get("run_config"))

        return cls(
            job_id=job_id,
            attack_model=attack_model,
            tested_model=tested_model,
            judge_model=judge_model,
            plan=plan,
            user_run_config=user_run_config,
        )

    def to_resolved_run(self, settings_obj: Settings) -> ResolvedRun:
        """
        Convert inputs into ResolvedRun.

        :param settings_obj: Application settings.
        :return: ResolvedRun.
        """
        run_config: dict[str, Any] = _merge_llamator_run_config(
            settings_obj=settings_obj,
            job_id=self.job_id,
            user_cfg=self.user_run_config,
        )
        artifacts_root: Path = Path(str(run_config["artifacts_path"])).resolve(strict=False)

        return ResolvedRun(
            job_id=self.job_id,
            attack_model=self.attack_model,
            tested_model=self.tested_model,
            judge_model=self.judge_model,
            plan=self.plan,
            run_config=run_config,
            artifacts_root=artifacts_root,
        )


class _ArtifactsLifecycle:
    """
    Job artifacts lifecycle operations: upload to backend and local cleanup.
    """

    def __init__(
        self,
        logger: logging.Logger,
        artifacts: ArtifactsStorage,
        job_id: str,
        local_root: Path,
        local_job_root: Path,
        upload_max_retries: int,
        upload_retry_delay_seconds: float,
    ) -> None:
        self._logger: logging.Logger = logger
        self._artifacts: ArtifactsStorage = artifacts
        self._job_id: str = job_id
        self._local_root: Path = local_root
        self._local_job_root: Path = local_job_root
        self._upload_max_retries: int = int(upload_max_retries)
        self._upload_retry_delay_seconds: float = float(upload_retry_delay_seconds)

    async def upload(self, job_status: str) -> bool:
        """
        Upload job artifacts to the configured backend.

        :param job_status: Job status marker for logs (e.g. "succeeded", "failed").
        :return: True if upload succeeded; otherwise False.
        """
        attempts: int = max(1, self._upload_max_retries)

        for attempt in range(1, attempts + 1):
            try:
                self._logger.info(
                    f"Worker job_id={self._job_id} status=artifacts_uploading job_status={job_status} "
                    f"attempt={attempt}/{attempts} path={self._local_root}"
                )
                await self._artifacts.upload_job_artifacts(job_id=self._job_id, local_root=self._local_root)
                self._logger.info(
                    f"Worker job_id={self._job_id} status=artifacts_uploaded job_status={job_status} "
                    f"attempt={attempt}/{attempts} path={self._local_root}"
                )
                return True
            except Exception as exc:
                self._logger.error(
                    f"Worker job_id={self._job_id} status=artifacts_upload_failed job_status={job_status} "
                    f"attempt={attempt}/{attempts} error={type(exc).__name__}: {exc}"
                )
                if attempt < attempts and self._upload_retry_delay_seconds > 0.0:
                    await asyncio.sleep(self._upload_retry_delay_seconds)

        return False

    async def cleanup_local(self, uploaded: bool) -> None:
        """
        Remove local artifacts directory after a successful upload.

        :param uploaded: Indicates whether upload succeeded.
        :return: None.
        """
        if not uploaded:
            return

        try:
            await asyncio.to_thread(shutil.rmtree, self._local_job_root)
            self._logger.info(
                f"Worker job_id={self._job_id} status=artifacts_local_cleaned path={self._local_job_root}"
            )
        except FileNotFoundError:
            return
        except Exception as exc:
            self._logger.warning(
                f"Worker job_id={self._job_id} status=artifacts_local_cleanup_failed error={type(exc).__name__}: {exc}"
            )


class _JobExecutor:
    """
    Orchestrates job execution, artifacts handling, and job state persistence.
    """

    def __init__(self, exec_ctx: _ExecutionContext) -> None:
        self._settings: Settings = exec_ctx.settings
        self._logger: logging.Logger = exec_ctx.logger
        self._store: JobStore = exec_ctx.store
        self._artifacts: ArtifactsStorage = exec_ctx.artifacts

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a single job payload.

        :param payload: Job payload dict.
        :return: Result dict containing job_id, aggregated and finished_at.
        :raises Exception: Propagates execution error after persisting job state.
        """
        job_id: str = str(payload["job_id"])

        await self._store.update_status(job_id, JobStatus.RUNNING)
        self._logger.info(f"Worker started job_id={job_id} status=running")

        lifecycle: _ArtifactsLifecycle | None = None
        uploaded: bool = False

        try:
            inputs: _RunInputs = _RunInputs.from_payload(job_id=job_id, payload=payload)
            resolved: ResolvedRun = inputs.to_resolved_run(settings_obj=self._settings)

            job_root: Path = _resolve_job_artifacts_root(self._settings, job_id)

            lifecycle = _ArtifactsLifecycle(
                logger=self._logger,
                artifacts=self._artifacts,
                job_id=job_id,
                local_root=resolved.artifacts_root,
                local_job_root=job_root,
                upload_max_retries=self._settings.artifacts_upload_max_retries,
                upload_retry_delay_seconds=self._settings.artifacts_upload_retry_delay_seconds,
            )

            runner: LlamatorRunner = LlamatorRunner(logger=self._logger)
            aggregated_raw: Any = await asyncio.to_thread(runner.run, resolved)
            aggregated: dict[str, dict[str, int]] = _validate_start_testing_result(aggregated_raw)

            if _is_empty_aggregated_result(aggregated):
                err_type: str = "EmptyAggregatedResultError"
                err_msg: str = "No tests were executed; aggregated results are empty."

                uploaded = await lifecycle.upload(job_status="failed")
                await self._store.set_error(job_id, err_type, err_msg)
                self._logger.error(f"Worker finished job_id={job_id} status=failed error={err_type}: {err_msg}")

                return {"job_id": job_id, "aggregated": {}, "finished_at": _utcnow().isoformat()}

            uploaded = await lifecycle.upload(job_status="succeeded")

            await self._store.set_result(job_id, aggregated)
            self._logger.info(f"Worker finished job_id={job_id} status=succeeded")
            return {"job_id": job_id, "aggregated": aggregated, "finished_at": _utcnow().isoformat()}
        except Exception as exc:
            if lifecycle is not None:
                uploaded = await lifecycle.upload(job_status="failed")

            await self._store.set_error(job_id, type(exc).__name__, str(exc))
            self._logger.exception(f"Worker failed job_id={job_id} status=failed error={type(exc).__name__}: {exc}")
            raise
        finally:
            if lifecycle is not None:
                await lifecycle.cleanup_local(uploaded=uploaded)


async def worker_startup(ctx: dict[str, Any]) -> None:
    """
    ARQ worker startup hook.

    :param ctx: ARQ worker context dict.
    :return: None.
    """
    configure_logging(settings.log_level)
    logger: logging.Logger = logging.getLogger(LOGGER_NAME)

    redis = create_redis_client(settings.redis_dsn)
    await redis.ping()

    artifacts: ArtifactsStorage = create_artifacts_storage(
        settings=settings,
        list_max_keys=1000,
    )
    if isinstance(artifacts, MinioArtifactsStorage):
        await artifacts.ensure_ready()

    logger.info(
        f"Artifacts backend initialized provider=minio endpoint={settings.minio_endpoint_url} bucket={settings.minio_bucket}"
    )

    await asyncio.to_thread(_cleanup_expired_local_artifacts, settings, logger)

    ctx["settings"] = settings
    ctx["logger"] = logger
    ctx["redis_client"] = redis
    ctx["store"] = JobStore(redis=redis, ttl_seconds=settings.job_ttl_seconds)
    ctx["artifacts_storage"] = artifacts

    logger.info("ARQ worker startup completed status=ready")


async def worker_shutdown(ctx: dict[str, Any]) -> None:
    """
    ARQ worker shutdown hook.

    :param ctx: ARQ worker context dict.
    :return: None.
    """
    logger = ctx.get("logger")

    redis = ctx.get("redis_client")
    if redis is not None:
        await redis.aclose()

    if logger is not None:
        logger.info("ARQ worker shutdown completed status=stopped")


async def run_llamator_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """
    ARQ task: execute LLAMATOR test run.

    :param ctx: Worker context.
    :param payload: Job payload (job_id and run configuration).
    :return: Result dict (aggregated metrics).
    :raises Exception: Re-raises execution exception to let ARQ handle it.
    """
    exec_ctx: _ExecutionContext = _ExecutionContext.from_ctx(ctx)
    executor: _JobExecutor = _JobExecutor(exec_ctx=exec_ctx)
    return await executor.execute(payload)


class WorkerSettings:
    on_startup = worker_startup
    on_shutdown = worker_shutdown
    functions = [run_llamator_job]
    redis_settings: RedisSettings = parse_redis_settings(settings.redis_dsn)
    job_timeout: int = settings.run_timeout_seconds
    max_tries: int = 1

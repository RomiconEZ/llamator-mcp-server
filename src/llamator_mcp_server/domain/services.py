from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from arq.connections import ArqRedis

from llamator_mcp_server.config.settings import Settings
from llamator_mcp_server.domain.models import (
    BasicTestSpec,
    CustomTestSpec,
    JobStatus,
    LlamatorTestRunRequest,
    OpenAIClientConfig,
    TestParameter,
)
from llamator_mcp_server.infra.job_store import JobStore


def _utcnow() -> datetime:
    """
    Return the current UTC time.

    :return: Current datetime in UTC.
    """
    return datetime.now(timezone.utc)


def _redact_client(cfg: OpenAIClientConfig) -> dict[str, Any]:
    """
    Redact sensitive fields from a client configuration.

    :param cfg: Client configuration.
    :return: Safe-to-store representation.
    """
    return {
        "kind": "openai",
        "base_url": str(cfg.base_url),
        "model": cfg.model,
        "temperature": cfg.temperature,
        "system_prompts": list(cfg.system_prompts) if cfg.system_prompts is not None else None,
        "model_description": cfg.model_description,
        "api_key_present": bool(cfg.api_key),
    }


def _redact_request(
    req: LlamatorTestRunRequest,
    attack: OpenAIClientConfig,
    judge: OpenAIClientConfig,
) -> dict[str, Any]:
    """
    Redact secrets from a test run request before persisting.

    :param req: Original request.
    :param attack: Attack model configuration.
    :param judge: Judge model configuration.
    :return: Safe-to-store request representation.
    """
    plan: dict[str, Any] = {
        "preset_name": req.plan.preset_name,
        "num_threads": req.plan.num_threads,
        "basic_tests": [
            {"code_name": t.code_name, "params": [{"name": p.name, "value": p.value} for p in t.params]}
            for t in (req.plan.basic_tests or ())
        ],
        "custom_tests": [
            {"import_path": t.import_path, "params": [{"name": p.name, "value": p.value} for p in t.params]}
            for t in (req.plan.custom_tests or ())
        ],
    }
    return {
        "tested_model": _redact_client(req.tested_model),
        "attack_model": _redact_client(attack),
        "judge_model": _redact_client(judge),
        "run_config": (req.run_config.model_dump() if req.run_config is not None else None),
        "plan": plan,
    }


def _build_attack_client(settings: Settings) -> OpenAIClientConfig:
    """
    Build attack model configuration from application settings.

    :param settings: Application settings.
    :return: OpenAIClientConfig for the attack model.
    :raises ValueError: If settings are invalid.
    """
    api_key_val: str | None = settings.attack_openai_api_key or None
    return OpenAIClientConfig(
        api_key=api_key_val,
        base_url=settings.attack_openai_base_url,
        model=settings.attack_openai_model,
        temperature=settings.attack_openai_temperature,
        system_prompts=settings.attack_openai_system_prompts,
        model_description=None,
    )


def _build_judge_client(settings: Settings) -> OpenAIClientConfig:
    """
    Build judge model configuration from application settings.

    :param settings: Application settings.
    :return: OpenAIClientConfig for the judge model.
    :raises ValueError: If settings are invalid.
    """
    api_key_val: str | None = settings.judge_openai_api_key or None
    return OpenAIClientConfig(
        api_key=api_key_val,
        base_url=settings.judge_openai_base_url,
        model=settings.judge_openai_model,
        temperature=settings.judge_openai_temperature,
        system_prompts=settings.judge_openai_system_prompts,
        model_description=None,
    )


@dataclass(frozen=True, slots=True)
class SubmitResult:
    """
    Result of submitting a job.

    :param job_id: Job identifier.
    :param created_at: Creation time.
    :param status: Current status.
    """

    job_id: str
    created_at: datetime
    status: JobStatus


class TestRunService:
    """
    Service responsible for submitting LLAMATOR test runs.

    :param arq: ARQ redis pool used to enqueue jobs.
    :param store: Job store for persisting statuses/results.
    :param settings: Application settings.
    :param logger: Logger.
    """

    def __init__(self, arq: ArqRedis, store: JobStore, settings: Settings, logger: logging.Logger) -> None:
        self._arq: ArqRedis = arq
        self._store: JobStore = store
        self._settings: Settings = settings
        self._logger: logging.Logger = logger

    async def submit(self, req: LlamatorTestRunRequest) -> SubmitResult:
        """
        Submit a test run job.

        :param req: Run request payload.
        :return: SubmitResult.
        :raises ValueError: If inputs are invalid.
        """
        job_id: str = uuid.uuid4().hex

        attack: OpenAIClientConfig = _build_attack_client(self._settings)
        judge: OpenAIClientConfig = _build_judge_client(self._settings)

        request_redacted: dict[str, Any] = _redact_request(req, attack=attack, judge=judge)
        await self._store.create(job_id=job_id, request_redacted=request_redacted)

        payload: dict[str, Any] = {
            "job_id": job_id,
            "created_at": _utcnow().isoformat(),
            "attack_model": attack.model_dump(mode="json"),
            "tested_model": req.tested_model.model_dump(mode="json"),
            "judge_model": judge.model_dump(mode="json"),
            "plan": req.plan.model_dump(mode="json"),
            "run_config": (req.run_config.model_dump(mode="json") if req.run_config is not None else None),
        }

        await self._arq.enqueue_job("run_llamator_job", payload, _job_id=job_id)
        self._logger.info(f"Enqueued LLAMATOR job_id={job_id}")
        return SubmitResult(job_id=job_id, created_at=_utcnow(), status=JobStatus.QUEUED)


def validate_unique_param_names(params: tuple[TestParameter, ...]) -> None:
    """
    Validate uniqueness of parameter names.

    :param params: Test parameters tuple.
    :return: None.
    :raises ValueError: If parameter names are duplicated.
    """
    names: set[str] = set()
    for p in params:
        if p.name in names:
            raise ValueError(f"Duplicate parameter name: {p.name}")
        names.add(p.name)


def validate_test_specs(
    basic_tests: tuple[BasicTestSpec, ...] | None,
    custom_tests: tuple[CustomTestSpec, ...] | None,
) -> None:
    """
    Validate basic and custom test specs.

    :param basic_tests: Basic tests list (or None).
    :param custom_tests: Custom tests list (or None).
    :return: None.
    :raises ValueError: If test parameters are invalid (e.g. duplicate names).
    """
    if basic_tests is not None:
        for t in basic_tests:
            validate_unique_param_names(t.params)
    if custom_tests is not None:
        for t in custom_tests:
            validate_unique_param_names(t.params)

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class JobStatus(str, Enum):
    """
    Job execution status.

    :cvar QUEUED: Job is enqueued.
    :cvar RUNNING: Job is running in a worker.
    :cvar SUCCEEDED: Job finished successfully.
    :cvar FAILED: Job finished with an error.
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ClientKind(str, Enum):
    """
    LLM client type.

    :cvar OPENAI: OpenAI-compatible API.
    """

    OPENAI = "openai"


class TestParameter(BaseModel):
    """
    A single test parameter.

    :param name: Parameter name.
    :param value: Parameter value (JSON-compatible).
    :raises ValueError: If name is empty.
    """

    model_config = ConfigDict(frozen=True)
    name: str = Field(min_length=1, max_length=200)
    value: object

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        name: str = v.strip()
        if not name:
            raise ValueError("Parameter name must be non-empty.")
        return name


class BasicTestSpec(BaseModel):
    """
    LLAMATOR built-in test specification.

    :param code_name: Attack code name.
    :param params: Test parameters.
    :raises ValueError: If code_name is empty.
    """

    model_config = ConfigDict(frozen=True)
    code_name: str = Field(min_length=1, max_length=200)
    params: tuple[TestParameter, ...] = Field(default_factory=tuple)

    @field_validator("code_name")
    @classmethod
    def _validate_code_name(cls, v: str) -> str:
        code_name: str = v.strip()
        if not code_name:
            raise ValueError("Test code_name must be non-empty.")
        return code_name


class CustomTestSpec(BaseModel):
    """
    Custom test specification (a class importable in the worker environment).

    import_path must point to an importable class inheriting
    ``llamator.attack_provider.test_base.TestBase``.

    :param import_path: Fully qualified import path to the test class.
    :param params: Test parameters.
    :raises ValueError: If import_path is empty or violates import policy.
    """

    model_config = ConfigDict(frozen=True)
    import_path: str = Field(min_length=1, max_length=500)
    params: tuple[TestParameter, ...] = Field(default_factory=tuple)

    @field_validator("import_path")
    @classmethod
    def _validate_import_path(cls, v: str) -> str:
        path: str = v.strip()
        if not path:
            raise ValueError("Custom test import_path must be non-empty.")
        allowed_prefixes: tuple[str, ...] = ("llamator.", "llamator_mcp_server.")
        if not any(path.startswith(prefix) for prefix in allowed_prefixes):
            raise ValueError("Custom test import_path is not allowed by import policy.")
        return path


class LlamatorRunConfig(BaseModel):
    """
    LLAMATOR run configuration (``config`` argument for start_testing).

    :param enable_logging: Enable LLAMATOR logging.
    :param enable_reports: Enable report generation (xlsx/docx).
    :param artifacts_path: Relative path inside server artifacts root.
    :param debug_level: LLAMATOR log verbosity (0=WARNING, 1=INFO, 2=DEBUG).
    :param report_language: Report language (en/ru).
    :raises ValueError: On invalid values.
    """

    model_config = ConfigDict(frozen=True)
    enable_logging: bool | None = None
    enable_reports: bool | None = None
    artifacts_path: str | None = None
    debug_level: int | None = None
    report_language: Literal["en", "ru"] | None = None

    @field_validator("artifacts_path")
    @classmethod
    def _validate_artifacts_path(cls, v: str | None) -> str | None:
        if v is None:
            return None
        normalized: PurePosixPath = PurePosixPath(v)
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError("artifacts_path must be a safe relative path.")
        return str(normalized)

    @field_validator("debug_level")
    @classmethod
    def _validate_debug_level(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if v not in (0, 1, 2):
            raise ValueError("debug_level must be one of: 0, 1, 2.")
        return v


class OpenAIClientConfig(BaseModel):
    """
    OpenAI-compatible client configuration for LLAMATOR.

    :param kind: Client kind (``openai``).
    :param api_key: Optional API key.
    :param base_url: OpenAI-compatible base URL (e.g. http://host:port/v1).
    :param model: Model identifier.
    :param temperature: Sampling temperature.
    :param system_prompts: Optional system prompts.
    :param model_description: Optional model description.
    :raises ValueError: On invalid values.
    """

    model_config = ConfigDict(frozen=True)
    kind: ClientKind = Field(default=ClientKind.OPENAI)
    api_key: str | None = Field(default=None, min_length=1)
    base_url: HttpUrl
    model: str = Field(min_length=1, max_length=300)
    temperature: float | None = None
    system_prompts: tuple[str, ...] | None = None
    model_description: str | None = None

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, v: ClientKind) -> ClientKind:
        if v != ClientKind.OPENAI:
            raise ValueError("Only 'openai' client kind is supported.")
        return v

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if v < 0.0 or v > 2.0:
            raise ValueError("temperature must be in [0.0, 2.0].")
        return v

    @field_validator("system_prompts")
    @classmethod
    def _validate_system_prompts(cls, v: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if v is None:
            return None
        cleaned: list[str] = [p.strip() for p in v if p.strip()]
        if not cleaned:
            return None
        return tuple(cleaned)


class TestPlan(BaseModel):
    """
    LLAMATOR test plan.

    :param preset_name: Built-in preset name (e.g. ``all``, ``rus``, ``owasp:llm01``).
    :param num_threads: Number of parallel threads.
    :param basic_tests: Explicit basic tests list.
    :param custom_tests: Explicit custom tests list (by import_path).
    :raises ValueError: On invalid combinations.
    """

    model_config = ConfigDict(frozen=True)
    preset_name: str | None = None
    num_threads: int | None = None
    basic_tests: tuple[BasicTestSpec, ...] | None = None
    custom_tests: tuple[CustomTestSpec, ...] | None = None

    @model_validator(mode="after")
    def _validate_plan(self) -> TestPlan:
        if self.num_threads is not None and self.num_threads < 1:
            raise ValueError("num_threads must be >= 1.")
        if self.preset_name is not None and not self.preset_name.strip():
            raise ValueError("preset_name must be non-empty when provided.")
        return self


class LlamatorTestRunRequest(BaseModel):
    """
    Request payload for starting LLAMATOR tests against an LLM endpoint.

    :param tested_model: Tested model configuration.
    :param run_config: Optional LLAMATOR run configuration.
    :param plan: Test plan.
    :raises ValueError: On invalid data.
    """

    model_config = ConfigDict(frozen=True)
    tested_model: OpenAIClientConfig
    run_config: LlamatorRunConfig | None = None
    plan: TestPlan


class LlamatorTestRunResponse(BaseModel):
    """
    Response payload for job creation.

    :param job_id: Job identifier.
    :param status: Current job status.
    :param created_at: Job creation time.
    """

    model_config = ConfigDict(frozen=True)
    job_id: str
    status: JobStatus
    created_at: datetime


class LlamatorJobError(BaseModel):
    """
    Job execution error details.

    :param error_type: Exception type name.
    :param message: Exception message.
    :param occurred_at: Error timestamp.
    """

    model_config = ConfigDict(frozen=True)
    error_type: str
    message: str
    occurred_at: datetime


class LlamatorJobResult(BaseModel):
    """
    Job execution result.

    :param aggregated: Aggregated metrics by attacks.
    :param finished_at: Completion timestamp.
    """

    model_config = ConfigDict(frozen=True)
    aggregated: dict[str, dict[str, int]]
    finished_at: datetime


class LlamatorJobInfo(BaseModel):
    """
    Job state.

    :param job_id: Job identifier.
    :param status: Job status.
    :param created_at: Creation timestamp.
    :param updated_at: Last update timestamp.
    :param request: Request payload with redacted secrets.
    :param result: Result payload (if any).
    :param error: Error payload (if any).
    :param error_notice: A user-facing error message (if any).
    """

    model_config = ConfigDict(frozen=True)
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    request: dict[str, object]
    result: LlamatorJobResult | None = None
    error: LlamatorJobError | None = None
    error_notice: str | None = None


class ArtifactFileInfo(BaseModel):
    """
    Artifact file metadata.

    :param path: Relative artifact path inside the job artifacts root.
    :param size_bytes: File size in bytes.
    :param mtime: File modification time as a Unix timestamp in seconds.
    """

    model_config = ConfigDict(frozen=True)

    path: str = Field(min_length=1, max_length=4000, description="Relative path inside job artifacts root.")
    size_bytes: int = Field(ge=0, description="File size in bytes.")
    mtime: float = Field(description="Unix timestamp in seconds.")


class ArtifactsListResponse(BaseModel):
    """
    Artifacts list endpoint response.

    :param job_id: Job identifier.
    :param files: Artifact files metadata list.
    """

    model_config = ConfigDict(frozen=True)

    job_id: str = Field(min_length=1, max_length=200, description="Job identifier.")
    files: list[ArtifactFileInfo] = Field(default_factory=list, description="Artifact files metadata list.")


class ArtifactDownloadResponse(BaseModel):
    """
    Artifact download link response.

    :param job_id: Job identifier.
    :param path: Relative artifact path inside the job artifacts prefix.
    :param download_url: Temporary presigned URL for downloading.
    """

    model_config = ConfigDict(frozen=True)

    job_id: str = Field(min_length=1, max_length=200, description="Job identifier.")
    path: str = Field(min_length=1, max_length=4000, description="Relative path inside job artifacts prefix.")
    download_url: str = Field(min_length=1, max_length=8000, description="Temporary presigned download URL.")


class HealthResponse(BaseModel):
    """
    Healthcheck response.

    :param status: Service status.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = Field(description="Service status.")

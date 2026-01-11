import logging
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Security
from fastapi.security.api_key import APIKeyHeader
from llamator_mcp_server.config.settings import Settings
from llamator_mcp_server.domain.models import ArtifactDownloadResponse
from llamator_mcp_server.domain.models import ArtifactFileInfo
from llamator_mcp_server.domain.models import ArtifactsListResponse
from llamator_mcp_server.domain.models import HealthResponse
from llamator_mcp_server.domain.models import LlamatorJobInfo
from llamator_mcp_server.domain.models import LlamatorTestRunRequest
from llamator_mcp_server.domain.models import LlamatorTestRunResponse
from llamator_mcp_server.domain.ports.artifacts_storage import ArtifactDownloadLink
from llamator_mcp_server.domain.ports.artifacts_storage import ArtifactFileRecord
from llamator_mcp_server.domain.ports.artifacts_storage import ArtifactsStorage
from llamator_mcp_server.domain.ports.artifacts_storage import ArtifactsStorageError
from llamator_mcp_server.domain.services import TestRunService
from llamator_mcp_server.domain.services import validate_test_specs
from llamator_mcp_server.infra.job_store import JobStore
from redis.asyncio import Redis

from .security import require_api_key

_API_KEY_SCHEME: APIKeyHeader = APIKeyHeader(
        name="X-API-Key",
        auto_error=False,
        scheme_name="McpApiKey",
)


class _ApiKeyDependency:
    """
    FastAPI dependency enforcing X-API-Key authentication.

    This dependency is intended to be attached to a protected router, while
    public routes (e.g. healthchecks) are mounted on a separate router without it.

    :param settings: Application settings.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings

    async def __call__(self, x_api_key: str | None = Security(_API_KEY_SCHEME)) -> None:
        await require_api_key(settings=self._settings, x_api_key=x_api_key)


def build_router(
        settings: Settings,
        redis: Redis,
        arq: Any,
        logger: logging.Logger,
        artifacts: ArtifactsStorage,
) -> APIRouter:
    """
    Построить HTTP роутер API.

    :param settings: Настройки приложения.
    :param redis: Redis-клиент.
    :param arq: ARQ pool.
    :param logger: Логгер.
    :param artifacts: Backend хранения артефактов.
    :return: Роутер FastAPI.
    """
    root_router: APIRouter = APIRouter()

    public_router: APIRouter = APIRouter()
    api_key_dep: _ApiKeyDependency = _ApiKeyDependency(settings=settings)
    protected_router: APIRouter = APIRouter(dependencies=[Depends(api_key_dep)])

    store: JobStore = JobStore(redis=redis, ttl_seconds=settings.job_ttl_seconds)
    service: TestRunService = TestRunService(arq=arq, store=store, settings=settings, logger=logger)

    @public_router.get("/v1/health", response_model=HealthResponse)
    @public_router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """
        Проверка здоровья сервиса.

        :return: Статус сервера.
        """
        return HealthResponse(status="ok")

    @protected_router.post("/v1/tests/runs", response_model=LlamatorTestRunResponse)
    async def create_run(req: LlamatorTestRunRequest) -> LlamatorTestRunResponse:
        """
        Создать задание на тестирование.

        :param req: Запрос запуска.
        :return: Ответ с job_id.
        :raises HTTPException: При ошибке валидации входных данных.
        """
        try:
            validate_test_specs(req.plan.basic_tests, req.plan.custom_tests)
        except ValueError as e:
            logger.warning(f"Validation error in create_run: {e}")
            raise HTTPException(status_code=400, detail=str(e)) from e
        result = await service.submit(req)
        return LlamatorTestRunResponse(job_id=result.job_id, status=result.status, created_at=result.created_at)

    @protected_router.get("/v1/tests/runs/{job_id}", response_model=LlamatorJobInfo)
    async def get_run(job_id: str) -> LlamatorJobInfo:
        """
        Получить состояние задания.

        :param job_id: Идентификатор задания.
        :return: Состояние задания.
        :raises HTTPException: Если задание не найдено.
        """
        try:
            return await store.get(job_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail="Not found") from e

    @protected_router.get("/v1/tests/runs/{job_id}/artifacts", response_model=ArtifactsListResponse)
    async def list_artifacts(job_id: str) -> ArtifactsListResponse:
        """
        Получить список файлов артефактов по заданию.

        :param job_id: Идентификатор задания.
        :return: Список файлов (путь, размер, mtime).
        :raises HTTPException: Если задание не найдено или backend артефактов недоступен.
        """
        try:
            await store.get(job_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail="Not found") from e

        try:
            files: list[ArtifactFileRecord] = await artifacts.list_files(job_id)
        except ArtifactsStorageError as e:
            logger.error(f"Artifacts list failed job_id={job_id} error={type(e).__name__}: {e}")
            raise HTTPException(status_code=502, detail="Artifacts backend error") from e

        parsed_files: list[ArtifactFileInfo] = [
            ArtifactFileInfo(path=f.path, size_bytes=f.size_bytes, mtime=f.mtime) for f in files
        ]
        return ArtifactsListResponse(job_id=job_id, files=parsed_files)

    @protected_router.get(
            "/v1/tests/runs/{job_id}/artifacts/{path:path}",
            response_model=ArtifactDownloadResponse,
            responses={
                200: {"description": "Artifact download link."},
                400: {"description": "Invalid path."},
                404: {"description": "Job or file not found."},
                502: {"description": "Artifacts backend error."},
            },
    )
    async def download_artifact(job_id: str, path: str) -> ArtifactDownloadResponse:
        """
        Скачать конкретный файл артефакта.

        :param job_id: Идентификатор задания.
        :param path: Относительный путь файла внутри артефактов задания.
        :return: ArtifactDownloadResponse с временной ссылкой на скачивание.
        :raises HTTPException: Если задание/файл не найден, путь небезопасен или backend артефактов недоступен.
        :raises RuntimeError: Если backend вернул некорректную цель загрузки.
        """
        try:
            await store.get(job_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail="Not found") from e

        try:
            link: ArtifactDownloadLink = await artifacts.get_download_link(
                    job_id=job_id,
                    rel_path=path,
                    expires_seconds=settings.artifacts_presign_expires_seconds,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid path") from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail="File not found") from e
        except ArtifactsStorageError as e:
            logger.error(
                    f"Artifacts download resolve failed job_id={job_id} path={path} error={type(e).__name__}: {e}"
            )
            raise HTTPException(status_code=502, detail="Artifacts backend error") from e

        return ArtifactDownloadResponse(job_id=job_id, path=path, download_url=link.url)

    root_router.include_router(public_router)
    root_router.include_router(protected_router)
    return root_router
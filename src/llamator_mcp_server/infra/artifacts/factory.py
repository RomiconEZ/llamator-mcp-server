from __future__ import annotations

from llamator_mcp_server.config.settings import Settings
from llamator_mcp_server.domain.ports.artifacts_storage import ArtifactsStorage
from llamator_mcp_server.infra.artifacts.minio import MinioArtifactsStorage, MinioConfig


def create_artifacts_storage(
    settings: Settings,
    list_max_keys: int,
) -> ArtifactsStorage:
    """
    Create artifacts storage based on configuration.

    :param settings: The application settings object.
    :param list_max_keys: Max keys returned by list operation.
    :return: ArtifactsStorage implementation.
    :raises ValueError: If configuration is invalid.
    """
    storage: MinioArtifactsStorage = MinioArtifactsStorage(
        MinioConfig(
            endpoint_url=settings.minio_endpoint_url,
            public_endpoint_url=settings.minio_public_endpoint_url,
            access_key_id=settings.minio_access_key_id,
            secret_access_key=settings.minio_secret_access_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
        ),
        list_max_keys=list_max_keys,
        retention_seconds=settings.artifacts_minio_ttl_seconds,
    )
    return storage

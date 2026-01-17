from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ARTIFACTS_ARCHIVE_NAME: Final[str] = "artifacts.zip"


class ArtifactsStorageError(RuntimeError):
    """
    Artifacts storage operation error.

    This exception is raised for backend communication/parsing errors.
    """


@dataclass(frozen=True, slots=True)
class ArtifactFileRecord:
    """
    Artifact file metadata record.

    :param path: Relative path inside job artifacts prefix.
    :param size_bytes: Object size in bytes.
    :param mtime: Unix timestamp in seconds.
    """

    path: str
    size_bytes: int
    mtime: float


@dataclass(frozen=True, slots=True)
class ArtifactDownloadLink:
    """
    Resolved download link for an artifact.

    :param url: A presigned URL usable for downloading.
    """

    url: str


class ArtifactsStorage(ABC):
    """
    Abstract artifacts storage interface.

    This port hides details of the underlying storage implementation (e.g. MinIO).
    """

    @abstractmethod
    async def list_files(self, job_id: str) -> list[ArtifactFileRecord]:
        """
        List all artifact files for a job.

        :param job_id: Job identifier.
        :return: List of artifact file records.
        :raises ArtifactsStorageError: On backend errors.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_download_link(self, job_id: str, rel_path: str, expires_seconds: int) -> ArtifactDownloadLink:
        """
        Build a temporary download link for a job artifact.

        :param job_id: Job identifier.
        :param rel_path: Relative path inside the job artifacts prefix.
        :param expires_seconds: Presigned URL TTL in seconds.
        :return: ArtifactDownloadLink with a presigned URL.
        :raises ValueError: If rel_path is unsafe.
        :raises FileNotFoundError: If the artifact does not exist.
        :raises ArtifactsStorageError: On backend errors.
        """
        raise NotImplementedError

    @abstractmethod
    async def upload_job_artifacts(self, job_id: str, local_root: Path) -> None:
        """
        Upload local job artifacts directory content into the storage.

        Implementations should upload the full directory tree preserving relative paths
        and also upload an archive named ``ARTIFACTS_ARCHIVE_NAME``.

        :param job_id: Job identifier.
        :param local_root: Local root directory with artifacts.
        :return: None.
        :raises ArtifactsStorageError: On backend errors.
        """
        raise NotImplementedError

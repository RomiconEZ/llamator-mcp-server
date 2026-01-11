from __future__ import annotations

import asyncio
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterable
from urllib.parse import ParseResult
from urllib.parse import urlparse
from urllib.parse import urlunparse

from llamator_mcp_server.domain.ports.artifacts_storage import ARTIFACTS_ARCHIVE_NAME
from llamator_mcp_server.domain.ports.artifacts_storage import ArtifactDownloadLink
from llamator_mcp_server.domain.ports.artifacts_storage import ArtifactFileRecord
from llamator_mcp_server.domain.ports.artifacts_storage import ArtifactsStorage
from llamator_mcp_server.domain.ports.artifacts_storage import ArtifactsStorageError
from minio.error import S3Error

from minio import Minio


def _utc_ts(dt: datetime | None) -> float:
    if dt is None:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _safe_posix_relpath(path: str) -> str:
    normalized: PurePosixPath = PurePosixPath(str(path))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError("Invalid path.")
    if str(normalized) in ("", "."):
        raise ValueError("Invalid path.")
    return str(normalized)


def _job_prefix(job_id: str) -> str:
    j: str = str(job_id).strip()
    if not j:
        raise ValueError("job_id must be non-empty.")
    return f"{j}/"


def _object_key(job_id: str, rel_path: str) -> str:
    rel: str = _safe_posix_relpath(rel_path)
    return f"{_job_prefix(job_id)}{rel}"


def _collect_files(root: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    root_resolved: Path = root.resolve(strict=False)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p: Path = Path(dirpath) / name
            if p.is_symlink():
                continue
            if not p.is_file():
                continue
            p_resolved: Path = p.resolve(strict=False)
            if p_resolved != root_resolved and root_resolved not in p_resolved.parents:
                continue
            rel: str = str(p.relative_to(root))
            rel_posix: str = str(PurePosixPath(Path(rel).as_posix()))
            out.append((p, rel_posix))
    out.sort(key=lambda x: x[1])
    return out


def _build_zip_archive(root: Path, out_file: Path) -> None:
    files: list[tuple[Path, str]] = _collect_files(root)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_file, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path, arc_name in files:
            zf.write(file_path, arcname=arc_name)


def _rewrite_presigned_url(url: str, public_endpoint_url: str | None) -> str:
    if public_endpoint_url is None:
        return url

    pub: str = str(public_endpoint_url).strip()
    if not pub:
        return url

    parsed_pub: ParseResult = urlparse(pub)
    if parsed_pub.scheme not in ("http", "https") or not parsed_pub.netloc:
        raise ValueError("minio_public_endpoint_url must be a valid http(s) URL with a host.")

    parsed: ParseResult = urlparse(url)
    rewritten: ParseResult = parsed._replace(scheme=parsed_pub.scheme, netloc=parsed_pub.netloc)
    return urlunparse(rewritten)


@dataclass(frozen=True, slots=True)
class MinioConfig:
    """
    MinIO connection configuration.

    :param endpoint_url: MinIO endpoint URL, e.g. http://minio:9000
    :param public_endpoint_url: Public endpoint URL used for presigned links rewriting (optional).
    :param access_key_id: Access key id.
    :param secret_access_key: Secret access key.
    :param bucket: Bucket name for artifacts.
    :param secure: Whether to use TLS for the internal MinIO client.
    """

    endpoint_url: str
    public_endpoint_url: str | None
    access_key_id: str
    secret_access_key: str
    bucket: str
    secure: bool


class MinioArtifactsStorage(ArtifactsStorage):
    """
    MinIO-based artifacts storage.

    This implementation uploads each artifact file as an object and also uploads a zip archive
    named ``ARTIFACTS_ARCHIVE_NAME``. Downloads are served via presigned URLs returned as links.
    """

    def __init__(self, cfg: MinioConfig, *, list_max_keys: int) -> None:
        if list_max_keys < 1:
            raise ValueError("list_max_keys must be >= 1.")

        endpoint_raw: str = str(cfg.endpoint_url).strip()
        if not endpoint_raw:
            raise ValueError("endpoint_url must be non-empty.")

        parsed: ParseResult = urlparse(endpoint_raw)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("endpoint_url must be a valid http(s) URL with a host.")

        if parsed.scheme == "https" and cfg.secure is False:
            raise ValueError("secure must be true when endpoint_url uses https.")
        if parsed.scheme == "http" and cfg.secure is True:
            raise ValueError("secure must be false when endpoint_url uses http.")

        public_raw: str | None = None
        if cfg.public_endpoint_url is not None:
            candidate: str = str(cfg.public_endpoint_url).strip()
            if candidate:
                parsed_pub: ParseResult = urlparse(candidate)
                if parsed_pub.scheme not in ("http", "https") or not parsed_pub.netloc:
                    raise ValueError("public_endpoint_url must be a valid http(s) URL with a host.")
                public_raw = candidate

        self._public_endpoint_url: str | None = public_raw
        self._bucket: str = str(cfg.bucket).strip()
        if not self._bucket:
            raise ValueError("bucket must be non-empty.")

        self._list_max_keys: int = int(list_max_keys)
        self._client: Minio = Minio(
                endpoint=parsed.netloc,
                access_key=str(cfg.access_key_id),
                secret_key=str(cfg.secret_access_key),
                secure=bool(cfg.secure),
        )

    async def ensure_ready(self) -> None:
        """
        Ensure MinIO bucket exists.

        :return: None.
        :raises ArtifactsStorageError: On backend errors.
        """
        try:
            exists: bool = await asyncio.to_thread(self._client.bucket_exists, self._bucket)
            if not exists:
                await asyncio.to_thread(self._client.make_bucket, self._bucket)
        except S3Error as e:
            raise ArtifactsStorageError(f"MinIO bucket init failed bucket={self._bucket!r}") from e

    async def list_files(self, job_id: str) -> list[ArtifactFileRecord]:
        prefix: str = _job_prefix(job_id)

        def _iter() -> list[ArtifactFileRecord]:
            items: list[ArtifactFileRecord] = []
            try:
                it: Iterable = self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
                for obj in it:
                    name: str = str(getattr(obj, "object_name", "") or "")
                    if not name.startswith(prefix):
                        continue
                    rel: str = name[len(prefix):]
                    if not rel:
                        continue
                    size: int = int(getattr(obj, "size", 0) or 0)
                    mtime: float = _utc_ts(getattr(obj, "last_modified", None))
                    items.append(ArtifactFileRecord(path=rel, size_bytes=size, mtime=mtime))
                    if len(items) >= self._list_max_keys:
                        break
            except S3Error as e:
                raise ArtifactsStorageError(f"MinIO list_objects failed prefix={prefix!r}") from e

            items.sort(key=lambda x: x.path)
            return items

        return await asyncio.to_thread(_iter)

    async def get_download_link(self, job_id: str, rel_path: str, expires_seconds: int) -> ArtifactDownloadLink:
        if expires_seconds < 1:
            raise ValueError("expires_seconds must be >= 1.")

        key: str = _object_key(job_id, rel_path)

        def _build() -> ArtifactDownloadLink:
            try:
                self._client.stat_object(self._bucket, key)
            except S3Error as e:
                code: str = str(getattr(e, "code", "") or "")
                if code in ("NoSuchKey", "NoSuchObject", "NoSuchBucket", "NotFound"):
                    raise FileNotFoundError("File not found")
                raise ArtifactsStorageError(f"MinIO stat_object failed key={key!r}") from e

            try:
                url: str = self._client.presigned_get_object(
                        self._bucket,
                        key,
                        expires=timedelta(seconds=expires_seconds),
                )
            except S3Error as e:
                raise ArtifactsStorageError(f"MinIO presigned_get_object failed key={key!r}") from e

            rewritten: str = _rewrite_presigned_url(url, self._public_endpoint_url)
            return ArtifactDownloadLink(url=rewritten)

        return await asyncio.to_thread(_build)

    async def upload_job_artifacts(self, job_id: str, local_root: Path) -> None:
        root: Path = Path(local_root).resolve(strict=False)

        if not root.exists():
            return

        prefix: str = _job_prefix(job_id)

        def _upload_all() -> None:
            try:
                self._client.bucket_exists(self._bucket)
            except S3Error as e:
                raise ArtifactsStorageError(f"MinIO bucket check failed bucket={self._bucket!r}") from e

            files: list[tuple[Path, str]] = _collect_files(root)
            for abs_path, rel_posix in files:
                key: str = f"{prefix}{_safe_posix_relpath(rel_posix)}"
                try:
                    self._client.fput_object(self._bucket, key, str(abs_path))
                except S3Error as e:
                    raise ArtifactsStorageError(f"MinIO upload failed key={key!r}") from e

            tmp_path: Path | None = None
            try:
                fd, tmp_name = tempfile.mkstemp(prefix="llamator-artifacts-", suffix=".zip")
                os.close(fd)
                tmp_path = Path(tmp_name)
                _build_zip_archive(root, tmp_path)

                zip_key: str = f"{prefix}{ARTIFACTS_ARCHIVE_NAME}"
                self._client.fput_object(self._bucket, zip_key, str(tmp_path))
            except S3Error as e:
                raise ArtifactsStorageError(f"MinIO upload archive failed job_id={job_id!r}") from e
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass

        await asyncio.to_thread(_upload_all)
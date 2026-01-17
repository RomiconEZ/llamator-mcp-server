from __future__ import annotations

import asyncio
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import ParseResult, urlparse

from minio import Minio
from minio.error import S3Error

from llamator_mcp_server.domain.ports.artifacts_storage import (
    ARTIFACTS_ARCHIVE_NAME,
    ArtifactDownloadLink,
    ArtifactFileRecord,
    ArtifactsStorage,
    ArtifactsStorageError,
)


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


def _validate_endpoint_url(raw_url: str, *, field_name: str) -> ParseResult:
    raw: str = str(raw_url).strip()
    if not raw:
        raise ValueError(f"{field_name} must be non-empty.")

    parsed: ParseResult = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{field_name} must be a valid http(s) URL with a host.")

    if parsed.path not in ("", "/"):
        raise ValueError(f"{field_name} must not contain a path component.")

    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not contain params/query/fragment.")

    return parsed


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _is_expired(last_modified: datetime | None, retention_seconds: int, now: datetime) -> bool:
    if retention_seconds < 1:
        return False
    if last_modified is None:
        return False
    lm: datetime = last_modified if last_modified.tzinfo is not None else last_modified.replace(tzinfo=timezone.utc)
    return lm <= (now - timedelta(seconds=int(retention_seconds)))


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

    This implementation uploads a single zip archive named ``ARTIFACTS_ARCHIVE_NAME``.
    Downloads are served via presigned URLs returned as links.
    """

    def __init__(self, cfg: MinioConfig, *, list_max_keys: int, retention_seconds: int) -> None:
        if list_max_keys < 1:
            raise ValueError("list_max_keys must be >= 1.")
        if retention_seconds < 1:
            raise ValueError("retention_seconds must be >= 1.")

        endpoint_raw: str = str(cfg.endpoint_url).strip()
        parsed: ParseResult = _validate_endpoint_url(endpoint_raw, field_name="endpoint_url")

        if parsed.scheme == "https" and cfg.secure is False:
            raise ValueError("secure must be true when endpoint_url uses https.")
        if parsed.scheme == "http" and cfg.secure is True:
            raise ValueError("secure must be false when endpoint_url uses http.")

        public_parsed: ParseResult | None = None
        if cfg.public_endpoint_url is not None:
            candidate: str = str(cfg.public_endpoint_url).strip()
            if candidate:
                public_parsed = _validate_endpoint_url(candidate, field_name="public_endpoint_url")

        self._bucket: str = str(cfg.bucket).strip()
        if not self._bucket:
            raise ValueError("bucket must be non-empty.")

        self._list_max_keys: int = int(list_max_keys)
        self._retention_seconds: int = int(retention_seconds)

        self._client: Minio = Minio(
            endpoint=parsed.netloc,
            access_key=str(cfg.access_key_id),
            secret_key=str(cfg.secret_access_key),
            secure=bool(cfg.secure),
        )

        if public_parsed is None:
            self._presign_endpoint: str = parsed.netloc
            self._presign_secure: bool = bool(cfg.secure)
        else:
            self._presign_endpoint = public_parsed.netloc
            self._presign_secure = public_parsed.scheme == "https"

        self._presign_access_key: str = str(cfg.access_key_id)
        self._presign_secret_key: str = str(cfg.secret_access_key)

        self._presign_client: Minio | None = None
        self._presign_lock: asyncio.Lock = asyncio.Lock()

    async def _get_presign_client(self) -> Minio:
        async with self._presign_lock:
            if self._presign_client is not None:
                return self._presign_client

            try:
                region: str = await asyncio.to_thread(self._client._get_region, self._bucket)
            except S3Error as e:
                raise ArtifactsStorageError(f"MinIO get_region failed bucket={self._bucket!r}") from e

            self._presign_client = Minio(
                endpoint=self._presign_endpoint,
                access_key=self._presign_access_key,
                secret_key=self._presign_secret_key,
                secure=self._presign_secure,
                region=region,
            )
            return self._presign_client

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

        await self._get_presign_client()

    async def list_files(self, job_id: str) -> list[ArtifactFileRecord]:
        prefix: str = _job_prefix(job_id)

        def _iter() -> list[ArtifactFileRecord]:
            now: datetime = _now_utc()
            items: list[ArtifactFileRecord] = []
            to_delete: list[str] = []

            try:
                it: Iterable = self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
                for obj in it:
                    name: str = str(getattr(obj, "object_name", "") or "")
                    if not name.startswith(prefix):
                        continue
                    rel: str = name[len(prefix) :]
                    if not rel:
                        continue

                    last_modified: datetime | None = getattr(obj, "last_modified", None)
                    if _is_expired(last_modified, self._retention_seconds, now):
                        to_delete.append(name)
                        continue

                    size: int = int(getattr(obj, "size", 0) or 0)
                    mtime: float = _utc_ts(last_modified)
                    items.append(ArtifactFileRecord(path=rel, size_bytes=size, mtime=mtime))
                    if len(items) >= self._list_max_keys:
                        break
            except S3Error as e:
                raise ArtifactsStorageError(f"MinIO list_objects failed prefix={prefix!r}") from e

            for key in to_delete:
                try:
                    self._client.remove_object(self._bucket, key)
                except S3Error:
                    continue

            items.sort(key=lambda x: x.path)
            return items

        return await asyncio.to_thread(_iter)

    async def get_download_link(self, job_id: str, rel_path: str, expires_seconds: int) -> ArtifactDownloadLink:
        if expires_seconds < 1:
            raise ValueError("expires_seconds must be >= 1.")

        key: str = _object_key(job_id, rel_path)
        presign_client: Minio = await self._get_presign_client()

        def _build() -> ArtifactDownloadLink:
            try:
                st = self._client.stat_object(self._bucket, key)
            except S3Error as e:
                code: str = str(getattr(e, "code", "") or "")
                if code in ("NoSuchKey", "NoSuchObject", "NoSuchBucket", "NotFound"):
                    raise FileNotFoundError("File not found")
                raise ArtifactsStorageError(f"MinIO stat_object failed key={key!r}") from e

            last_modified: datetime | None = getattr(st, "last_modified", None)
            if _is_expired(last_modified, self._retention_seconds, _now_utc()):
                try:
                    self._client.remove_object(self._bucket, key)
                except S3Error:
                    pass
                raise FileNotFoundError("File not found")

            try:
                url: str = presign_client.presigned_get_object(
                    self._bucket,
                    key,
                    expires=timedelta(seconds=expires_seconds),
                )
            except S3Error as e:
                raise ArtifactsStorageError(f"MinIO presigned_get_object failed key={key!r}") from e

            return ArtifactDownloadLink(url=url)

        return await asyncio.to_thread(_build)

    async def upload_job_artifacts(self, job_id: str, local_root: Path) -> None:
        root: Path = Path(local_root).resolve(strict=False)

        if not root.exists():
            return

        prefix: str = _job_prefix(job_id)

        def _cleanup_prefix() -> None:
            try:
                it: Iterable = self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
                for obj in it:
                    name: str = str(getattr(obj, "object_name", "") or "")
                    if not name.startswith(prefix):
                        continue
                    try:
                        self._client.remove_object(self._bucket, name)
                    except S3Error:
                        continue
            except S3Error as e:
                raise ArtifactsStorageError(f"MinIO list_objects cleanup failed prefix={prefix!r}") from e

        def _upload_archive(tmp_zip: Path) -> None:
            try:
                self._client.bucket_exists(self._bucket)
            except S3Error as e:
                raise ArtifactsStorageError(f"MinIO bucket check failed bucket={self._bucket!r}") from e

            _cleanup_prefix()

            try:
                zip_key: str = f"{prefix}{ARTIFACTS_ARCHIVE_NAME}"
                self._client.fput_object(self._bucket, zip_key, str(tmp_zip))
            except S3Error as e:
                raise ArtifactsStorageError(f"MinIO upload archive failed job_id={job_id!r}") from e

        tmp_path: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(prefix="llamator-artifacts-", suffix=".zip")
            os.close(fd)
            tmp_path = Path(tmp_name)
            _build_zip_archive(root, tmp_path)
            await asyncio.to_thread(_upload_archive, tmp_path)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

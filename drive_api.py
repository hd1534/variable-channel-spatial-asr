"""Personal Google Drive transport using Google's official Python libraries.

Each thread owns its HTTP client. Uploads use pre-generated IDs and resumable
requests, and verify Drive's size and MD5 before returning successfully.
Credentials and SDK imports are kept outside the training/model modules.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/drive.file",)
TOKEN_URI: str = "https://oauth2.googleapis.com/token"
FOLDER_MIME: str = "application/vnd.google-apps.folder"
FILE_FIELDS: str = "id,name,mimeType,size,md5Checksum,trashed"
CHUNK_BYTES: int = 8 * 1024 * 1024
RETRY_DELAYS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 60, 60)  # about 3 minutes before a backup fails
TOKEN_GRANT_ERRORS: frozenset[str] = frozenset({
    "invalid_request", "invalid_client", "invalid_grant", "unauthorized_client", "unsupported_grant_type",
    "invalid_scope"})
T = TypeVar("T")


class StorageError(RuntimeError):
    """Storage failed; callers must preserve local data and report an incomplete backup."""


def error_status(error: BaseException) -> tuple[int | None, set[str]]:
    """Extract status/reasons without logging response bodies or credentials."""
    status = getattr(getattr(error, "resp", None), "status", None)
    reasons: set[str] = set()
    try:
        body = json.loads(getattr(error, "content", b"{}"))
        reasons = {str(row.get("reason", "")).lower() for row in body.get("error", {}).get("errors", [])}
    except (ValueError, TypeError, AttributeError):
        pass
    return status, reasons


def network_errors() -> tuple[type[BaseException], ...]:
    """Connection failures a later attempt can survive, as the SDK stack raises them.

    Besides socket errors (OSError) these are a response cut short
    (http.client.IncompleteRead), a DNS failure (httplib2 turns socket.gaierror
    into ServerNotFoundError, not an OSError) and any of these during the hourly
    token refresh (google-auth wraps them in TransportError).
    """
    errors: list[type[BaseException]] = [OSError, http.client.HTTPException]
    try:
        import httplib2

        errors.append(httplib2.ServerNotFoundError)
    except ImportError:
        pass
    try:
        from google.auth.exceptions import TransportError

        errors.append(TransportError)
    except ImportError:
        pass
    return tuple(errors)


def authorization_revoked(error: BaseException) -> bool:
    """Whether the token endpoint rejected the stored grant itself, so only setup_drive.py can help.

    google-auth raises RefreshError(details, {"error": code, ...}) when the
    endpoint answers in JSON, and RefreshError(body) for anything else (an HTML
    502/503 page), marking the latter not retryable although it says nothing
    about the grant.  Only the RFC 6749 section 5.2 codes are final.
    """
    if type(error).__name__ == "InvalidGrantError":
        return True
    payload = next((arg for arg in getattr(error, "args", ()) if isinstance(arg, dict)), None)
    return (type(error).__name__ == "RefreshError" and payload is not None
            and payload.get("error") in TOKEN_GRANT_ERRORS)


def is_transient(error: BaseException) -> bool:
    """Quota, server and network failures; never permission, quota-exhausted or integrity errors."""
    status, reasons = error_status(error)
    return (status in (429, 500, 502, 503, 504)
            or (status == 403 and bool(reasons & {"ratelimitexceeded", "userratelimitexceeded"}))
            or (type(error).__name__ == "RefreshError" and not authorization_revoked(error))
            or isinstance(error, network_errors()))


def retry(operation: Callable[[], T]) -> T:
    """Retry transient transport/quota errors, keeping permission and storage failures explicit."""
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return operation()
        except Exception as error:
            if not is_transient(error) or attempt == len(RETRY_DELAYS):
                raise
            delay = RETRY_DELAYS[attempt] + random.random()
            print(f"[storage] transient Drive request failure; retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable retry state")


def execute(request: Any) -> Any:
    """Execute a metadata request with bounded backoff and sanitized errors."""
    try:
        return retry(lambda: request.execute(num_retries=0))
    except Exception as error:
        raise api_error(error) from None


def api_error(error: BaseException) -> StorageError:
    """Describe a failed API call without exposing tokens, URLs or server response content."""
    status, reasons = error_status(error)
    if status == 401 or authorization_revoked(error):
        return StorageError("Drive authorization expired or was revoked; rerun setup_drive.py and update DRIVE_AUTH")
    reason = ", ".join(sorted(reasons)) or type(error).__name__
    return StorageError(f"Drive API request failed (HTTP {status or 'unavailable'}, {reason}); local files were preserved")


def private_json(path: Path, value: dict[str, Any]) -> None:
    """Write authentication material atomically with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle)
    temporary.chmod(0o600)
    temporary.replace(path)


def validate_auth(value: Any) -> dict[str, Any]:
    """Validate the private setup package before any authorization network request."""
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise StorageError("invalid DRIVE_AUTH; use the JSON produced by setup_drive.py")
    folder = value.get("folder_id")
    if not isinstance(folder, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", folder) or folder == "root":
        raise StorageError("DRIVE_AUTH must select a project folder, not an entire Drive")
    token = value.get("credentials")
    if not isinstance(token, dict) or not all(isinstance(token.get(k), str) and token[k]
                                            for k in ("client_id", "client_secret", "refresh_token")):
        raise StorageError("DRIVE_AUTH has no reusable personal OAuth authorization")
    if token.get("token_uri", TOKEN_URI) != TOKEN_URI or set(token.get("scopes", [])) != set(SCOPES):
        raise StorageError("DRIVE_AUTH must use the official Google token endpoint and drive.file scope")
    return value


def build_service(credentials: dict[str, Any]) -> Any:
    """Build one bounded-timeout authorized HTTP client for the current thread.

    build_http() (60 s timeout) also removes 308 from httplib2's redirect codes.
    Drive answers every partial resumable chunk and every upload status query
    with "308 Resume Incomplete" and no Location; a plain httplib2.Http follows
    it as a redirect and fails, so no file larger than one chunk could upload.
    """
    from google.oauth2.credentials import Credentials
    from google_auth_httplib2 import AuthorizedHttp
    from googleapiclient.discovery import build
    from googleapiclient.http import build_http

    creds = Credentials.from_authorized_user_info(credentials, scopes=SCOPES)
    http = AuthorizedHttp(creds, http=build_http())
    return build("drive", "v3", http=http, cache_discovery=False, static_discovery=True)


def file_checksums(path: Path) -> tuple[int, str]:
    """Read a large file incrementally to compare it with Drive's stored MD5."""
    digest = hashlib.md5()
    size: int = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def create_folder(service: Any, name: str, parent: str = "root") -> dict[str, Any]:
    """Create one folder idempotently, including recovery after an ambiguous HTTP response."""
    identifier = execute(service.files().generateIds(count=1, space="drive", type="files"))["ids"][0]
    request = service.files().create(body={"id": identifier, "name": name, "mimeType": FOLDER_MIME,
                                           "parents": [parent]}, fields=FILE_FIELDS)
    try:
        retry(lambda: request.execute(num_retries=0))
    except Exception as error:
        if error_status(error)[0] != 409:
            raise api_error(error) from None
    return execute(service.files().get(fileId=identifier, fields=FILE_FIELDS))


class GoogleDriveTransport:
    """Immutable backup objects addressed by project-relative names and cached Drive IDs."""

    def __init__(self, folder_id: str, service_factory: Callable[[], Any]) -> None:
        """Scope every operation to one dedicated project folder."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", folder_id) or folder_id == "root":
            raise StorageError("a dedicated Drive folder ID is required")
        self.remote: str = "gdrive:" + folder_id
        self.folder_id: str = folder_id
        self.factory: Callable[[], Any] = service_factory
        self.clients: threading.local = threading.local()
        self.folders: dict[str, str] = {"": folder_id}
        self.files: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_auth(cls, auth: dict[str, Any]) -> GoogleDriveTransport:
        """Load the user-created OAuth client; each worker gets an independent HTTP connection."""
        package = validate_auth(auth)
        return cls(package["folder_id"], lambda: build_service(package["credentials"]))

    @property
    def service(self) -> Any:
        """Never share an httplib2 client across notebook and backup threads."""
        if not hasattr(self.clients, "service"):
            self.clients.service = self.factory()
        return self.clients.service

    def _list(self, parent: str, name: str | None = None) -> list[dict[str, Any]]:
        """Read every page and keep duplicate names visible to integrity checks."""
        query = f"'{parent}' in parents and trashed = false"
        if name is not None:
            escaped = name.replace("\\", "\\\\").replace("'", "\\'")
            query += f" and name = '{escaped}'"
        rows: list[dict[str, Any]] = []
        page: str | None = None
        while True:
            result = execute(self.service.files().list(q=query, spaces="drive", pageSize=1000, pageToken=page,
                                                       fields=f"nextPageToken,files({FILE_FIELDS})"))
            rows.extend(result.get("files", []))
            page = result.get("nextPageToken")
            if not page:
                return rows

    def _folder(self, path: str, *, create: bool = False) -> str:
        """Resolve/cache folder IDs without treating authentication errors as missing paths."""
        if path in self.folders:
            return self.folders[path]
        parent_path, _, name = path.rpartition("/")
        parent = self._folder(parent_path, create=create)
        rows = self._list(parent, name)
        if len(rows) > 1 or (rows and rows[0]["mimeType"] != FOLDER_MIME):
            raise StorageError(f"ambiguous Drive backup directory: {path}")
        if not rows:
            if not create:
                raise StorageError(f"Drive backup directory is missing: {path}")
            rows = [create_folder(self.service, name, parent)]
        self.folders[path] = rows[0]["id"]
        return self.folders[path]

    def _lookup(self, relative: str) -> dict[str, Any] | None:
        """Resolve one immutable file; duplicate names cannot silently choose different content."""
        if relative in self.files:
            return self.files[relative]
        parent, _, name = relative.rpartition("/")
        rows = self._list(self._folder(parent), name)
        if len(rows) > 1 or (rows and rows[0]["mimeType"] == FOLDER_MIME):
            raise StorageError(f"ambiguous Drive backup file: {relative}")
        if rows:
            self.files[relative] = rows[0]
            return rows[0]
        return None

    def prepare(self) -> None:
        """Validate access to the project folder and initialize only backup-owned subfolders."""
        root = execute(self.service.files().get(fileId=self.folder_id, fields=FILE_FIELDS))
        if root.get("trashed") or root.get("mimeType") != FOLDER_MIME:
            raise StorageError("the configured Drive project folder is unavailable")
        self._folder("commits", create=True)
        self._folder("objects", create=True)

    def commits(self) -> list[str]:
        """Refresh history from Drive rather than trusting a stale directory listing."""
        rows = self._list(self._folder("commits"))
        names: list[str] = []
        for row in rows:
            name = row["name"]
            if row["mimeType"] != FOLDER_MIME and re.fullmatch(r"(?:root|[0-9a-f]{32})--[0-9a-f]{32}\.json", name):
                if name in names:
                    raise StorageError("duplicate Drive backup manifest names")
                names.append(name)
                self.files["commits/" + name] = row
        return names

    def manifest(self, name: str) -> dict[str, Any]:
        """Read a small committed manifest through its Drive file ID."""
        entry = self._lookup("commits/" + name)
        if entry is None:
            raise StorageError("committed backup manifest is missing")
        data = execute(self.service.files().get_media(fileId=entry["id"]))
        if hashlib.md5(data).hexdigest() != entry.get("md5Checksum"):
            raise StorageError("backup manifest checksum mismatch")
        return json.loads(data)

    def upload(self, local: Path, relative: str) -> None:
        """Wait for a resumable upload and verify server metadata before accepting its content."""
        from googleapiclient.http import MediaFileUpload

        parent, _, name = relative.rpartition("/")
        folder = self._folder(parent, create=True)
        size, digest = file_checksums(local)
        entry = self._lookup(relative)
        if entry is None:
            identifier = execute(self.service.files().generateIds(count=1, space="drive", type="files"))["ids"][0]
            media = MediaFileUpload(str(local), mimetype="application/octet-stream", chunksize=CHUNK_BYTES, resumable=True)
            request = self.service.files().create(body={"id": identifier, "name": name, "parents": [folder]},
                                                  media_body=media, fields=FILE_FIELDS)
            response: Any = None
            try:
                while response is None:
                    _, response = retry(lambda: request.next_chunk(num_retries=0))
            except Exception as error:
                if error_status(error)[0] != 409:
                    raise api_error(error) from None
            finally:
                media.stream().close()
            # A direct ID read avoids depending on newly uploaded files appearing in searches.
            entry = execute(self.service.files().get(fileId=identifier, fields=FILE_FIELDS))
        if int(entry.get("size", -1)) != size or entry.get("md5Checksum") != digest:
            raise StorageError(f"uploaded Drive object checksum mismatch: {relative}")
        self.files[relative] = entry

    def download(self, relative: str, local: Path) -> None:
        """Stream an object into the caller's staging file with bounded chunk retries."""
        from googleapiclient.http import MediaIoBaseDownload

        entry = self._lookup(relative)
        if entry is None:
            raise StorageError(f"committed Drive object is missing: {relative}")
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("wb") as handle:
            request = self.service.files().get_media(fileId=entry["id"])
            downloader = MediaIoBaseDownload(handle, request, chunksize=CHUNK_BYTES)
            done: bool = False
            try:
                while not done:
                    _, done = retry(lambda: downloader.next_chunk(num_retries=0))
            except Exception as error:
                raise api_error(error) from None
        size, digest = file_checksums(local)
        if int(entry.get("size", -1)) != size or entry.get("md5Checksum") != digest:
            raise StorageError(f"downloaded Drive object checksum mismatch: {relative}")

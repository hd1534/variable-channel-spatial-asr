"""Personal Google Drive transport using Google's official Python libraries.

Each thread owns its HTTP client. Uploads use pre-generated IDs and resumable
requests, and verify Drive's size and MD5 before returning the file's Drive ID.
A download given that ID reads the file directly, so it never waits for Drive's
search index. Runtimes writing at once can leave same-named folders and
identical copies of one immutable file; both are read as one, while copies with
different content stay an integrity error. Local disk failures are reported as
local and never retried as network errors. Credentials and SDK imports are kept
outside the training/model modules.
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
from typing import Any, BinaryIO, Callable, TypeVar

SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/drive.file",)
TOKEN_URI: str = "https://oauth2.googleapis.com/token"
FOLDER_MIME: str = "application/vnd.google-apps.folder"
FILE_FIELDS: str = "id,name,mimeType,size,md5Checksum,sha256Checksum,trashed"
"""Drive computes md5Checksum, sha1Checksum and sha256Checksum of every stored binary file itself
(output-only fields; sha256 "if available").  The backup names objects by their SHA-256, so Drive's
sha256Checksum verifies an object without the client reading the file again."""
# <parents>--<id>.json: "root", or one or more parent commit IDs joined by "+" (a merge).
COMMIT_NAME: re.Pattern[str] = re.compile(r"(?:root|[0-9a-f]{32}(?:\+[0-9a-f]{32})*)--[0-9a-f]{32}\.json")
DRIVE_ID: re.Pattern[str] = re.compile(r"[A-Za-z0-9_-]+")
CHUNK_BYTES: int = 1024 * 1024 * 1024
"""Bytes per resumable-upload chunk and per ranged download request (1 GiB, a
multiple of the 256 KiB Drive requires).  Every chunk is a separate HTTP request
with its own round trip, so small chunks throttle the multi-GB files this backup
moves -- 8 MB chunks restored at a few MB/s on Colab.  A failed chunk is resent
alone (network errors are rare), and the cost is one chunk of memory per transfer."""
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


def quoted(value: str) -> str:
    """Write a Drive search query string literal."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def one_copy(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the smallest-ID row when the rows are copies of one file, else None.

    A retried upload or two runtimes storing the same content-addressed object
    can leave several copies under one name. They count as one file only when
    Drive vouches for identical content (same size and MD5); a folder among them,
    or a copy whose content Drive cannot vouch for, keeps the name ambiguous.
    """
    if not rows or any(row.get("mimeType") == FOLDER_MIME for row in rows):
        return None
    if len(rows) > 1 and (not all(row.get("md5Checksum") for row in rows)
                          or len({(row.get("size"), row.get("md5Checksum")) for row in rows}) > 1):
        return None
    return min(rows, key=lambda row: row["id"])


def local_write_error(relative: str, error: OSError) -> StorageError:
    """Describe a failing local disk (usually a full Colab disk) so it is not mistaken for Drive."""
    return StorageError(f"local disk write failed while restoring {relative}: "
                        f"{error.strerror or type(error).__name__}; "
                        "free space on the Colab disk or add prefixes to RESTORE_SKIP")


def local_read_error(relative: str, local: Path, error: OSError) -> StorageError:
    """Describe a local file that cannot be read during an upload; retrying the network cannot help."""
    return StorageError(f"local disk read failed while backing up {relative} from {local}: "
                        f"{error.strerror or type(error).__name__}")


class DownloadSink:
    """MediaIoBaseDownload destination that hashes each chunk once it is on the local disk.

    The SDK advances its download offset before it writes a chunk, so after a
    failed write a retried request would skip that chunk. A disk error therefore
    ends the download as a StorageError, which retry() never repeats. Network
    failures happen before the SDK writes anything, so a retried chunk is still
    written exactly once. A chunk enters the size and MD5 only after all of it
    was written, so the hashes describe the file without reading it back.
    """

    def __init__(self, handle: BinaryIO, relative: str) -> None:
        """Wrap an open, unbuffered staging file for one committed object."""
        self.handle: BinaryIO = handle
        self.relative: str = relative
        self.size: int = 0
        self.md5: Any = hashlib.md5()

    def write(self, data: bytes) -> int:
        """Write the whole chunk (an unbuffered file may take it in parts) or fail as a local error."""
        view = memoryview(data)
        try:
            while view:
                view = view[self.handle.write(view):]
        except OSError as error:
            raise local_write_error(self.relative, error) from error
        self.size += len(data)
        self.md5.update(data)
        return len(data)


def chunk_media(handle: BinaryIO, relative: str, local: Path) -> Any:
    """Resumable media that reads each chunk into memory before sending it.

    MediaFileUpload streams the file while the request is on the wire, so a
    failing local disk surfaced as an OSError from the socket send: retry()
    took it for a network error and repeated it for minutes, and the request
    was left half sent. Reading the chunk first keeps a local error local and
    out of the connection; after a network error the chunk is read again from
    the offset Drive confirmed.
    """
    from googleapiclient.http import MediaIoBaseUpload

    class ChunkMedia(MediaIoBaseUpload):
        """MediaIoBaseUpload that hands the SDK bytes instead of a stream."""

        def has_stream(self) -> bool:
            """Make HttpRequest.next_chunk call getbytes() before it sends."""
            return False

        def getbytes(self, begin: int, length: int) -> bytes:
            """Read one chunk, reporting a disk failure as local."""
            try:
                return super().getbytes(begin, length)
            except OSError as error:
                raise local_read_error(relative, local, error) from error

    try:
        return ChunkMedia(handle, mimetype="application/octet-stream", chunksize=CHUNK_BYTES, resumable=True)
    except OSError as error:  # the constructor seeks to the end for the size
        raise local_read_error(relative, local, error) from error


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
    """Immutable backup objects addressed by project-relative names and cached Drive IDs.

    One project-relative directory can be several same-named Drive folders: two
    runtimes that both find it missing both create it. ``folders`` holds the one
    new files go into (the smallest ID, which every runtime seeing all of them
    picks alike) and ``duplicate_folders`` the others, which every lookup and
    listing searches as well.
    """

    def __init__(self, folder_id: str, service_factory: Callable[[], Any]) -> None:
        """Scope every operation to one dedicated project folder."""
        if not DRIVE_ID.fullmatch(folder_id) or folder_id == "root":
            raise StorageError("a dedicated Drive folder ID is required")
        self.remote: str = "gdrive:" + folder_id
        self.folder_id: str = folder_id
        self.factory: Callable[[], Any] = service_factory
        self.clients: threading.local = threading.local()
        self.folders: dict[str, str] = {"": folder_id}
        self.duplicate_folders: dict[str, list[str]] = {}
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

    def _list(self, parents: list[str], name: str | None = None) -> list[dict[str, Any]]:
        """Read every page under all folders of one directory; duplicate names stay visible."""
        query = " or ".join(f"{quoted(parent)} in parents" for parent in parents)
        query = (f"({query})" if len(parents) > 1 else query) + " and trashed = false"
        if name is not None:
            query += f" and name = {quoted(name)}"
        rows: list[dict[str, Any]] = []
        page: str | None = None
        while True:
            result = execute(self.service.files().list(q=query, spaces="drive", pageSize=1000, pageToken=page,
                                                       fields=f"nextPageToken,files({FILE_FIELDS})"))
            rows.extend(result.get("files", []))
            page = result.get("nextPageToken")
            if not page:
                return rows

    def _subfolders(self, parents: list[str], name: str, path: str) -> list[dict[str, Any]]:
        """List the folders named ``name`` under a directory; a file of that name is an integrity error."""
        rows = self._list(parents, name)
        if any(row["mimeType"] != FOLDER_MIME for row in rows):
            raise StorageError(f"ambiguous Drive backup directory: {path}")
        return rows

    def _folder_ids(self, path: str, *, create: bool = False) -> list[str]:
        """Resolve/cache every Drive folder of one directory, the creation target first.

        Authentication errors are never taken for missing paths. After creating
        a folder the parent is listed again, so a folder another runtime created
        at the same moment is found and joins the same directory.
        """
        if path not in self.folders:
            parent_path, _, name = path.rpartition("/")
            parents = self._folder_ids(parent_path, create=create)
            rows = self._subfolders(parents, name, path)
            if not rows:
                if not create:
                    raise StorageError(f"Drive backup directory is missing: {path}")
                created = create_folder(self.service, name, parents[0])
                rows = self._subfolders(parents, name, path)
                if created["id"] not in {row["id"] for row in rows}:
                    rows.append(created)  # searches may not list a new folder yet
            ids = sorted(row["id"] for row in rows)
            self.duplicate_folders[path] = ids[1:]
            self.folders[path] = ids[0]
        return [self.folders[path], *self.duplicate_folders.get(path, [])]

    def _folder(self, path: str, *, create: bool = False) -> str:
        """Return the folder new files of a directory are created in."""
        return self._folder_ids(path, create=create)[0]

    def _forget(self, path: str) -> None:
        """Drop a cached directory and its ancestors so the next lookup lists Drive again."""
        while path:
            self.folders.pop(path, None)
            self.duplicate_folders.pop(path, None)
            path = path.rpartition("/")[0]

    def _lookup(self, relative: str) -> dict[str, Any] | None:
        """Resolve one immutable file; identical copies are one file, differing copies an integrity error."""
        if relative in self.files:
            return self.files[relative]
        parent, _, name = relative.rpartition("/")
        rows = self._list(self._folder_ids(parent), name)
        if not rows:
            return None
        entry = one_copy(rows)
        if entry is None:
            raise StorageError(f"ambiguous Drive backup file: {relative}")
        self.files[relative] = entry
        return entry

    def _find(self, relative: str) -> dict[str, Any] | None:
        """Look up a committed file by name, listing its folders afresh once before calling it missing.

        The cached folders may predate a same-named folder another runtime created.
        """
        entry = self._lookup(relative)
        if entry is None:
            self._forget(relative.rpartition("/")[0])
            entry = self._lookup(relative)
        return entry

    def _metadata(self, identifier: str) -> dict[str, Any] | None:
        """Read one file's metadata by ID, which is strongly consistent; None when Drive has no such file."""
        request = self.service.files().get(fileId=identifier, fields=FILE_FIELDS)
        try:
            return retry(lambda: request.execute(num_retries=0))
        except Exception as error:
            if error_status(error)[0] == 404:
                return None
            raise api_error(error) from None

    def _referenced(self, relative: str, ref: str) -> dict[str, Any] | None:
        """Read the file a manifest recorded by ID and check it is that object; None when it is gone."""
        if not DRIVE_ID.fullmatch(ref):
            raise StorageError(f"invalid Drive file reference for {relative}")
        entry = self._metadata(ref)
        if entry is None or entry.get("trashed"):
            return None
        if entry.get("mimeType") == FOLDER_MIME or entry.get("name") != relative.rpartition("/")[2]:
            raise StorageError(f"Drive file reference does not point to {relative}")
        self.files[relative] = entry
        return entry

    def prepare(self) -> None:
        """Validate access to the project folder and initialize only backup-owned subfolders."""
        root = execute(self.service.files().get(fileId=self.folder_id, fields=FILE_FIELDS))
        if root.get("trashed") or root.get("mimeType") != FOLDER_MIME:
            raise StorageError("the configured Drive project folder is unavailable")
        self._folder("commits", create=True)
        self._folder("objects", create=True)

    def refresh(self) -> None:
        """Find commits folders created by another runtime since this one cached the directory."""
        self._forget("commits")

    def commits(self) -> list[str]:
        """List every manifest name in all commits folders, read from Drive rather than a stale cache.

        Identical copies of one manifest (same size and MD5) are one commit;
        copies that differ would make the history ambiguous.
        """
        copies: dict[str, list[dict[str, Any]]] = {}
        for row in self._list(self._folder_ids("commits")):
            if row["mimeType"] != FOLDER_MIME and COMMIT_NAME.fullmatch(row["name"]):
                copies.setdefault(row["name"], []).append(row)
        for name, rows in copies.items():
            entry = one_copy(rows)
            if entry is None:
                raise StorageError(f"duplicate Drive backup manifest names with different content: {name}")
            self.files["commits/" + name] = entry
        return list(copies)

    def manifest(self, name: str) -> dict[str, Any]:
        """Read a small committed manifest through its Drive file ID."""
        entry = self._find("commits/" + name)
        if entry is None:
            raise StorageError("committed backup manifest is missing")
        data = execute(self.service.files().get_media(fileId=entry["id"]))
        if hashlib.md5(data).hexdigest() != entry.get("md5Checksum"):
            raise StorageError("backup manifest checksum mismatch")
        return json.loads(data)

    def upload(self, local: Path, relative: str, sha256: str | None = None) -> str:
        """Store one immutable file unless it exists; return the Drive ID of the verified file.

        A new file counts as stored only after the resumable upload finished and
        Drive's checksum matches the local bytes; an existing one only when its
        checksum matches.  With ``sha256`` (the caller already hashed the bytes)
        Drive's own ``sha256Checksum`` is compared and the file is not read again;
        only when Drive reports none is the local file read once for its MD5.
        """
        folder = self._folder(relative.rpartition("/")[0], create=True)
        try:
            # Fail as a local error before any request: a missing, denied or unreadable file, or a directory.
            with local.open("rb") as handle:
                handle.read(1)
                size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise local_read_error(relative, local, error) from error
        entry = self._lookup(relative)
        if entry is None:
            entry = self._send(local, relative, folder)
        remote_sha = str(entry.get("sha256Checksum") or "").lower()
        if sha256 and remote_sha:
            matches = int(entry.get("size", -1)) == size and remote_sha == sha256.lower()
        else:
            try:
                size, digest = file_checksums(local)
            except OSError as error:
                raise local_read_error(relative, local, error) from error
            matches = int(entry.get("size", -1)) == size and entry.get("md5Checksum") == digest
        if not matches:
            raise StorageError(f"uploaded Drive object checksum mismatch: {relative}")
        self.files[relative] = entry
        return entry["id"]

    def _send(self, local: Path, relative: str, folder: str) -> dict[str, Any]:
        """Upload under a pre-generated ID and read that ID's metadata back (searches may lag)."""
        identifier = execute(self.service.files().generateIds(count=1, space="drive", type="files"))["ids"][0]
        try:
            handle = local.open("rb")
        except OSError as error:
            raise local_read_error(relative, local, error) from error
        with handle:
            media = chunk_media(handle, relative, local)
            request = self.service.files().create(
                body={"id": identifier, "name": relative.rpartition("/")[2], "parents": [folder]},
                media_body=media, fields=FILE_FIELDS)
            response: Any = None
            try:
                while response is None:
                    _, response = retry(lambda: request.next_chunk(num_retries=0))
            except StorageError:
                raise
            except Exception as error:
                if error_status(error)[0] != 409:  # 409: an earlier attempt already stored this ID
                    raise api_error(error) from None
        return execute(self.service.files().get(fileId=identifier, fields=FILE_FIELDS))

    def download(self, relative: str, local: Path, ref: str | None = None) -> str | None:
        """Stream one committed file into the caller's staging file and verify Drive's size and MD5.

        The MD5 is computed while the bytes are written, so the file is never read
        back.  Returns Drive's ``sha256Checksum`` of the same content (``None`` when
        Drive reports none): the bytes on disk match Drive's, so that is their SHA-256.

        ``ref`` is the Drive ID upload() returned. The file is then read by ID,
        so a file Drive has not indexed for searches yet is still found; without
        it the file is looked up by name.
        """
        entry = self._referenced(relative, ref) if ref is not None else self._find(relative)
        if entry is None:
            raise StorageError(f"committed Drive object is missing: {relative}")
        sink = self._receive(self.service.files().get_media(fileId=entry["id"]), local, relative)
        if int(entry.get("size", -1)) != sink.size or entry.get("md5Checksum") != sink.md5.hexdigest():
            raise StorageError(f"downloaded Drive object checksum mismatch: {relative}")
        return str(entry.get("sha256Checksum") or "").lower() or None

    def _receive(self, request: Any, local: Path, relative: str) -> DownloadSink:
        """Write every chunk through a hashing sink; only transport failures are retried."""
        from googleapiclient.http import MediaIoBaseDownload

        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open("wb", buffering=0) as handle:
                sink = DownloadSink(handle, relative)
                downloader = MediaIoBaseDownload(sink, request, chunksize=CHUNK_BYTES)
                done: bool = False
                try:
                    while not done:
                        _, done = retry(lambda: downloader.next_chunk(num_retries=0))
                except StorageError:
                    raise
                except Exception as error:
                    raise api_error(error) from None
        except OSError as error:  # creating, opening or closing the staging file
            raise local_write_error(relative, error) from error
        return sink

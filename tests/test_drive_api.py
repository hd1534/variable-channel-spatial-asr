"""Test the personal-Drive transport's v2 protocol without a Google account.

Covers what concurrent runtimes and the v2 store rely on: upload() returns the
stored file's Drive ID, download(ref=...) reads by that ID without a listing,
same-named folders and identical copies of one immutable file are read as one
(differing copies stay an integrity error), merge-commit names are listed, and
a failing local disk is reported as local instead of being retried as a network
error. Metadata tests use an in-memory Drive; the SDK tests drive Google's real
MediaIoBaseUpload/MediaIoBaseDownload through HttpMockSequence.

Run with::

    python3 -m unittest tests.test_drive_api -v
"""

from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest.mock import Mock, patch

import drive_api as api

SDK_AVAILABLE: bool = importlib.util.find_spec("googleapiclient") is not None
DIGEST: str = "ab" + "0" * 62
FIRST, SECOND = "1" * 32, "2" * 32


class ApiFailure(Exception):
    """Structured Drive error shaped like googleapiclient's HttpError."""

    def __init__(self, status: int, reason: str) -> None:
        """Expose the HTTP fields the transport inspects."""
        super().__init__(reason)
        self.resp: Any = Mock(status=status)
        self.content: bytes = json.dumps({"error": {"errors": [{"reason": reason}]}}).encode()


class Request:
    """One deferred Drive call, executed like an SDK HttpRequest."""

    def __init__(self, call: Callable[[], Any]) -> None:
        """Keep the call until execute()."""
        self.call: Callable[[], Any] = call

    def execute(self, num_retries: int = 0) -> Any:
        """Run the call."""
        return self.call()


class MemoryDrive:
    """Metadata-only Drive v3 over a dict, with lagging searches and concurrent-creation hooks."""

    def __init__(self) -> None:
        """Start with only the project folder."""
        self.rows: dict[str, dict[str, Any]] = {}
        self.queries: list[str] = []
        self.hidden: set[str] = set()  # IDs that searches do not list yet
        self.on_create: Callable[[dict[str, Any]], None] | None = None
        self.counter: int = 0
        self.add("project", "project", None, folder=True)

    def add(self, identifier: str, name: str, parent: str | None, content: bytes | None = None, *,
            folder: bool = False) -> dict[str, Any]:
        """Store a folder or a file whose size and MD5 Drive computed."""
        row: dict[str, Any] = {"id": identifier, "name": name, "trashed": False,
                               "mimeType": api.FOLDER_MIME if folder else "application/octet-stream",
                               "parents": [parent] if parent else []}
        if not folder:
            data = content or b""
            row.update(size=str(len(data)), md5Checksum=hashlib.md5(data).hexdigest())
        self.rows[identifier] = row
        return row

    def public(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return the fields FILE_FIELDS asks for."""
        return {key: value for key, value in row.items() if key != "parents"}

    def search(self, query: str) -> dict[str, Any]:
        """Answer the query shapes the transport sends; record every query."""
        self.queries.append(query)
        parents = set(re.findall(r"'([^']*)' in parents", query))
        named = re.search(r"name = '([^']*)'", query)
        rows = [self.public(row) for row in self.rows.values()
                if row["id"] not in self.hidden and not row["trashed"] and parents & set(row["parents"])
                and (named is None or row["name"] == named.group(1))]
        return {"files": rows}

    def get(self, identifier: str) -> dict[str, Any]:
        """Read by ID, which is strongly consistent (hidden rows are found)."""
        if identifier not in self.rows:
            raise ApiFailure(404, "notFound")
        return self.public(self.rows[identifier])

    def create(self, body: dict[str, Any]) -> dict[str, Any]:
        """Create a folder under a pre-generated ID."""
        row = self.add(body["id"], body["name"], body["parents"][0], folder=body.get("mimeType") == api.FOLDER_MIME)
        if self.on_create is not None:
            self.on_create(row)
        return self.public(row)

    def new_id(self) -> str:
        """Generate IDs that sort after the fixtures' hand-picked ones."""
        self.counter += 1
        return f"z{self.counter:04d}"


class MemoryFiles:
    """The ``service.files()`` resource of MemoryDrive."""

    def __init__(self, drive: MemoryDrive) -> None:
        """Bind to one in-memory Drive."""
        self.drive: MemoryDrive = drive

    def list(self, q: str, **_: Any) -> Request:
        """files.list (single page)."""
        return Request(lambda: self.drive.search(q))

    def get(self, fileId: str, **_: Any) -> Request:
        """files.get metadata."""
        return Request(lambda: self.drive.get(fileId))

    def create(self, body: dict[str, Any], **_: Any) -> Request:
        """files.create for folders."""
        return Request(lambda: self.drive.create(body))

    def generateIds(self, **_: Any) -> Request:
        """files.generateIds."""
        return Request(lambda: {"ids": [self.drive.new_id()]})


def memory_transport() -> tuple[api.GoogleDriveTransport, MemoryDrive]:
    """A transport whose service is the in-memory Drive."""
    drive = MemoryDrive()
    service = Mock()
    service.files.return_value = MemoryFiles(drive)
    return api.GoogleDriveTransport("project", lambda: service), drive


class TestDuplicates(unittest.TestCase):
    """Concurrent runtimes may duplicate folders and objects; identical ones must not break restores."""

    def test_identical_copies_of_an_object_resolve_to_the_smallest_id(self) -> None:
        """A retried upload that left two identical copies is one object (finding drive-dupobject)."""
        transport, drive = memory_transport()
        drive.add("objects", "objects", "project", folder=True)
        drive.add("prefix", "ab", "objects", folder=True)
        drive.add("copy-b", DIGEST, "prefix", b"weights")
        drive.add("copy-a", DIGEST, "prefix", b"weights")
        self.assertEqual(transport._lookup(f"objects/ab/{DIGEST}")["id"], "copy-a")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "blob"
            source.write_bytes(b"weights")
            self.assertEqual(transport.upload(source, f"objects/ab/{DIGEST}"), "copy-a")

    def test_differing_or_unverifiable_copies_stay_ambiguous(self) -> None:
        """Two contents under one immutable name, or copies without an MD5, are an integrity error."""
        for second in ({"content": b"other"}, {"folder": True}, {"content": b"weights", "md5": None}):
            transport, drive = memory_transport()
            drive.add("objects", "objects", "project", folder=True)
            drive.add("copy-a", "blob", "objects", b"weights")
            row = drive.add("copy-b", "blob", "objects", second.get("content"), folder=second.get("folder", False))
            if "md5" in second:
                row.pop("md5Checksum")
                drive.rows["copy-a"].pop("md5Checksum")
            with self.subTest(second=second), self.assertRaisesRegex(api.StorageError, "ambiguous Drive backup file"):
                transport._lookup("objects/blob")

    def test_same_named_folders_are_searched_together_and_the_smallest_id_receives_new_files(self) -> None:
        """Two runtimes that both created objects/ab still see each other's objects."""
        transport, drive = memory_transport()
        drive.add("objects", "objects", "project", folder=True)
        drive.add("prefix-b", "ab", "objects", folder=True)
        drive.add("prefix-a", "ab", "objects", folder=True)
        drive.add("object-id", DIGEST, "prefix-b", b"weights")
        self.assertEqual(transport._lookup(f"objects/ab/{DIGEST}")["id"], "object-id")
        self.assertIn("('prefix-a' in parents or 'prefix-b' in parents)", drive.queries[-1])
        self.assertEqual(transport._folder("objects/ab"), "prefix-a")
        self.assertEqual(transport._folder_ids("objects/ab"), ["prefix-a", "prefix-b"])

    def test_a_folder_created_concurrently_by_another_runtime_joins_the_directory(self) -> None:
        """After creating a folder the parent is listed again, so a concurrent twin is noticed."""
        transport, drive = memory_transport()
        drive.add("objects", "objects", "project", folder=True)
        drive.on_create = lambda row: drive.add("aaaa-twin", row["name"], row["parents"][0], folder=True)
        self.assertEqual(transport._folder_ids("objects/ab", create=True), ["aaaa-twin", "z0001"])
        self.assertEqual(sum(1 for query in drive.queries if "name = 'ab'" in query), 2)

    def test_a_new_folder_that_searches_do_not_list_yet_is_kept(self) -> None:
        """Listing lag right after creation must not lose the folder just created."""
        transport, drive = memory_transport()
        drive.add("objects", "objects", "project", folder=True)
        drive.on_create = lambda row: drive.hidden.add(row["id"])
        self.assertEqual(transport._folder_ids("objects/ab", create=True), ["z0001"])

    def test_a_file_named_like_a_backup_directory_is_an_integrity_error(self) -> None:
        """A non-folder where a directory belongs is never silently used or replaced."""
        transport, drive = memory_transport()
        drive.add("objects", "objects", "project", folder=True)
        drive.add("stray", "ab", "objects", b"not a folder")
        with self.assertRaisesRegex(api.StorageError, "ambiguous Drive backup directory"):
            transport._folder_ids("objects/ab", create=True)

    def test_name_lookup_lists_cached_folders_again_before_calling_a_file_missing(self) -> None:
        """A twin folder created after this process cached the directory is still searched."""
        transport, drive = memory_transport()
        drive.add("objects", "objects", "project", folder=True)
        drive.add("prefix-b", "ab", "objects", folder=True)
        self.assertEqual(transport._folder_ids("objects/ab"), ["prefix-b"])
        drive.add("prefix-a", "ab", "objects", folder=True)
        drive.add("object-id", DIGEST, "prefix-a", b"weights")
        self.assertEqual(transport._find(f"objects/ab/{DIGEST}")["id"], "object-id")
        self.assertEqual(transport._folder_ids("objects/ab"), ["prefix-a", "prefix-b"])


class TestCommits(unittest.TestCase):
    """History listing across duplicate commits folders, including v2 merge commits."""

    def test_merge_names_are_listed_and_identical_duplicates_across_folders_collapse(self) -> None:
        """Concurrent prepare() can leave two commits folders; history is their union."""
        transport, drive = memory_transport()
        drive.add("commits-b", "commits", "project", folder=True)
        drive.add("commits-a", "commits", "project", folder=True)
        names = [f"root--{FIRST}.json", f"{FIRST}--{SECOND}.json", f"{FIRST}+{SECOND}--{'3' * 32}.json"]
        for index, name in enumerate(names):
            drive.add(f"m{index}", name, "commits-a", name.encode())
        drive.add("m0-copy", names[0], "commits-b", names[0].encode())
        drive.add("junk", "notes.txt", "commits-b", b"x")
        drive.add("junk-merge", f"{FIRST}+--{SECOND}.json", "commits-b", b"x")
        self.assertEqual(sorted(transport.commits()), sorted(names))
        self.assertEqual(transport.files["commits/" + names[0]]["id"], "m0")

    def test_differing_manifests_under_one_name_raise(self) -> None:
        """One commit name with two contents would make the history ambiguous."""
        transport, drive = memory_transport()
        drive.add("commits", "commits", "project", folder=True)
        drive.add("m1", f"root--{FIRST}.json", "commits", b"one")
        drive.add("m2", f"root--{FIRST}.json", "commits", b"two")
        with self.assertRaisesRegex(api.StorageError, "duplicate Drive backup manifest names"):
            transport.commits()


class TestReferences(unittest.TestCase):
    """download(ref=...) reads the recorded file by ID and checks it is that object."""

    def test_reference_checks_happen_before_any_transfer(self) -> None:
        """A wrong, trashed, folder or forged reference never streams bytes into the staging file."""
        transport, drive = memory_transport()
        drive.add("other", "different-name", "project", b"x")
        drive.add("a-folder", DIGEST, "project", folder=True)
        drive.add("trashed", DIGEST, "project", b"x")["trashed"] = True
        cases = (("other", "does not point to"), ("a-folder", "does not point to"),
                 ("trashed", "is missing"), ("gone", "is missing"), ("../x", "invalid Drive file reference"))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "staged"
            for ref, message in cases:
                with self.subTest(ref=ref), self.assertRaisesRegex(api.StorageError, message):
                    transport.download(f"objects/ab/{DIGEST}", target, ref=ref)
            self.assertFalse(target.exists())
        self.assertEqual(drive.queries, [])  # never a name search

    def test_upload_of_an_existing_object_returns_its_id_without_uploading(self) -> None:
        """Content-addressed objects are stored once; the caller still gets a reference."""
        transport, drive = memory_transport()
        drive.add("objects", "objects", "project", folder=True)
        drive.add("existing", "blob", "objects", b"one")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "file"
            source.write_bytes(b"one")
            self.assertEqual(transport.upload(source, "objects/blob"), "existing")
        self.assertEqual(drive.counter, 0)  # no ID generated, so nothing was created


class ShortWrites:
    """Unbuffered file that accepts at most a few bytes per write() and can fail like a full disk."""

    def __init__(self, limit: int, fail_at_call: int | None = None) -> None:
        """Remember what was written."""
        self.data: bytearray = bytearray()
        self.limit: int = limit
        self.calls: int = 0
        self.fail_at_call: int | None = fail_at_call

    def write(self, view: memoryview) -> int:
        """Take a prefix of the data, or raise ENOSPC."""
        self.calls += 1
        if self.calls == self.fail_at_call:
            raise OSError(errno.ENOSPC, "No space left on device")
        taken = bytes(view[:self.limit])
        self.data.extend(taken)
        return len(taken)


class TestLocalDiskErrors(unittest.TestCase):
    """A full Colab disk is a local error, never a retried network error or a 'corrupt Drive object'."""

    def test_sink_writes_whole_chunks_and_hashes_only_what_reached_the_disk(self) -> None:
        """Partial writes are completed; a failed chunk never enters the size or MD5."""
        handle = ShortWrites(limit=3)
        sink = api.DownloadSink(handle, "objects/blob")  # type: ignore[arg-type]
        self.assertEqual(sink.write(b"0123456789"), 10)
        self.assertEqual(bytes(handle.data), b"0123456789")
        failing = ShortWrites(limit=3, fail_at_call=2)
        sink = api.DownloadSink(failing, "objects/ab/blob")  # type: ignore[arg-type]
        with self.assertRaises(api.StorageError) as caught:
            sink.write(b"0123456789")
        self.assertNotIsInstance(caught.exception, OSError)
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertEqual(str(caught.exception),
                         "local disk write failed while restoring objects/ab/blob: No space left on device; "
                         "free space on the Colab disk or add prefixes to RESTORE_SKIP")
        self.assertEqual((sink.size, sink.md5.hexdigest()), (0, hashlib.md5().hexdigest()))

    def test_local_disk_errors_are_not_retried(self) -> None:
        """retry() repeats network failures only."""
        for error in (api.local_write_error("objects/blob", OSError(errno.EIO, "Input/output error")),
                      api.local_read_error("objects/blob", Path("blob"), OSError(errno.EIO, "Input/output error"))):
            operation = Mock(side_effect=error)
            with self.subTest(error=str(error)), patch.object(api.time, "sleep") as sleep:
                with self.assertRaises(api.StorageError):
                    api.retry(operation)
                operation.assert_called_once()
                sleep.assert_not_called()

    def test_unreadable_source_fails_the_upload_as_local_before_anything_is_sent(self) -> None:
        """The checksum read before an upload maps a vanished or unreadable file to a local StorageError.

        A raw OSError would count as a network failure for retry(), and a raw
        FileNotFoundError is what the store's capture step takes for a pruned
        checkpoint, so the file would silently miss the backup.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a-directory").mkdir()
            unreadable = root / "unreadable"
            unreadable.write_bytes(b"weights")
            real_open = Path.open

            def failing_open(path: Path, *args: Any, **kwargs: Any) -> Any:
                """Deny one file and give another a bad sector."""
                if path.name == "denied":
                    raise PermissionError(errno.EACCES, "Permission denied", str(path))
                handle = real_open(path, *args, **kwargs)
                if path == unreadable:
                    handle.read = Mock(side_effect=OSError(errno.EIO, "Input/output error"))  # type: ignore[method-assign]
                return handle

            cases = ((root / "vanished", FileNotFoundError, "No such file or directory"),
                     (root / "denied", PermissionError, "Permission denied"),
                     (root / "a-directory", IsADirectoryError, "Is a directory"),
                     (unreadable, OSError, "Input/output error"))
            for source, kind, reason in cases:
                transport, drive = memory_transport()
                drive.add("objects", "objects", "project", folder=True)
                with self.subTest(error=kind.__name__):
                    with patch.object(Path, "open", failing_open), patch.object(api.time, "sleep") as sleep, \
                         self.assertRaises(api.StorageError) as caught:
                        api.retry(lambda: transport.upload(source, "objects/blob"))
                    self.assertNotIsInstance(caught.exception, OSError)
                    self.assertEqual(str(caught.exception),
                                     f"local disk read failed while backing up objects/blob from {source}: {reason}")
                    self.assertIsInstance(caught.exception.__cause__, kind)
                    sleep.assert_not_called()
                    self.assertEqual(drive.counter, 0)  # no ID generated, so no upload was started
                    self.assertFalse(any("name = 'blob'" in query for query in drive.queries))

    def test_source_vanishing_after_the_checksum_read_is_a_local_error(self) -> None:
        """A file deleted between the checksum and the upload's own open is still reported as local."""
        transport, drive = memory_transport()
        drive.add("objects", "objects", "project", folder=True)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "vanished"
            with patch.object(api, "file_checksums", return_value=(7, hashlib.md5(b"weights").hexdigest())), \
                 patch.object(api.time, "sleep") as sleep, self.assertRaises(api.StorageError) as caught:
                api.retry(lambda: transport.upload(source, "objects/blob"))
        self.assertEqual(str(caught.exception),
                         f"local disk read failed while backing up objects/blob from {source}: No such file or directory")
        self.assertIsInstance(caught.exception.__cause__, FileNotFoundError)
        sleep.assert_not_called()
        self.assertFalse(any(row["name"] == "blob" for row in drive.rows.values()))


def sdk_service(responses: list[Any]) -> tuple[Any, Any]:
    """Build the real Drive v3 client over HttpMockSequence; an Exception entry is raised instead."""
    from googleapiclient.discovery import build
    from googleapiclient.http import HttpMockSequence

    class FailingSequence(HttpMockSequence):
        """HttpMockSequence that can also drop a request like a broken connection."""

        def request(self, uri: str, method: str = "GET", body: Any = None, headers: Any = None,
                    *args: Any, **kwargs: Any) -> Any:
            """Raise an injected network error, or answer the next canned response."""
            if isinstance(self._iterable[0], BaseException):
                self.request_sequence.append((uri, method, body, headers))
                raise self._iterable.pop(0)
            return super().request(uri, method, body, headers, *args, **kwargs)

    http = FailingSequence(responses)
    return build("drive", "v3", http=http, cache_discovery=False, static_discovery=True), http


def metadata(identifier: str, name: str, data: bytes) -> tuple[dict[str, str], bytes]:
    """Canned files.get / final upload reply."""
    return {"status": "200"}, json.dumps({"id": identifier, "name": name, "mimeType": "application/octet-stream",
                                          "size": str(len(data)), "md5Checksum": hashlib.md5(data).hexdigest(),
                                          "trashed": False}).encode()


def media(data: bytes, start: int, stop: int) -> tuple[dict[str, str], bytes]:
    """Canned 206 reply for bytes [start, stop) of an object."""
    return {"status": "206", "content-range": f"bytes {start}-{stop - 1}/{len(data)}"}, data[start:stop]


def media_ranges(http: Any) -> list[str]:
    """Range headers of the media GETs the SDK sent."""
    return [headers["range"] for uri, method, _, headers in http.request_sequence if "alt=media" in uri]


@unittest.skipUnless(SDK_AVAILABLE, "install requirements-drive.txt to exercise Google's actual SDK")
class TestSdkTransfers(unittest.TestCase):
    """Real resumable upload / chunked download code paths with injected failures."""

    def setUp(self) -> None:
        """Small chunks keep the fixtures readable; sleeping is skipped."""
        self.patches = [patch.object(api, "CHUNK_BYTES", 4), patch.object(api.time, "sleep")]
        for active in self.patches:
            active.start()
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        """Undo the patches."""
        for active in self.patches:
            active.stop()
        self.directory.cleanup()

    def test_upload_returns_the_id_and_a_ref_download_needs_no_listing(self) -> None:
        """The ID upload() returns is all a later download needs."""
        data = b"0123456789"
        service, http = sdk_service([
            ({"status": "200"}, b'{"files": []}'),
            ({"status": "200"}, b'{"ids": ["new-id"]}'),
            ({"status": "200", "location": "https://www.googleapis.com/upload/drive/v3/files?upload_id=u"}, b""),
            ({"status": "308", "range": "bytes=0-3"}, b""),
            ({"status": "308", "range": "bytes=0-7"}, b""),
            metadata("new-id", "blob", data),
            metadata("new-id", "blob", data),
            metadata("new-id", "blob", data),
            media(data, 0, 4), media(data, 4, 8), media(data, 8, 10),
        ])
        transport = api.GoogleDriveTransport("project", lambda: service)
        transport.folders["objects"] = "objects-id"
        source, target = self.root / "source", self.root / "staged"
        source.write_bytes(data)
        self.assertEqual(transport.upload(source, "objects/blob"), "new-id")
        uploads = len(http.request_sequence)
        reader = api.GoogleDriveTransport("project", lambda: service)  # a fresh runtime: nothing cached
        reader.download("objects/blob", target, ref="new-id")
        self.assertEqual(target.read_bytes(), data)
        sent = http.request_sequence[uploads:]
        self.assertNotIn("q=", " ".join(uri for uri, *_ in sent))
        self.assertIn("/files/new-id?", sent[0][0])

    def test_network_failure_mid_download_resumes_at_the_same_offset(self) -> None:
        """A dropped chunk is fetched again and written exactly once; no re-read is needed to verify."""
        data = b"0123456789"
        service, http = sdk_service([
            metadata("id", "blob", data),
            media(data, 0, 4),
            ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"),
            ({"status": "503"}, b'{"error": {"errors": [{"reason": "backendError"}]}}'),
            media(data, 4, 8), media(data, 8, 10),
        ])
        transport = api.GoogleDriveTransport("project", lambda: service)
        target = self.root / "staged"
        with patch.object(api, "file_checksums", side_effect=AssertionError("re-read")):
            transport.download("objects/blob", target, ref="id")
        self.assertEqual(target.read_bytes(), data)
        self.assertEqual(media_ranges(http), ["bytes=0-3", "bytes=4-7", "bytes=4-7", "bytes=4-7", "bytes=8-11"])

    def test_full_disk_during_download_is_a_local_error_and_not_retried(self) -> None:
        """ENOSPC on the second chunk stops at once with advice about the Colab disk (finding api-5)."""
        data = b"0123456789"
        service, http = sdk_service([metadata("id", "blob", data), media(data, 0, 4), media(data, 4, 8)])
        transport = api.GoogleDriveTransport("project", lambda: service)
        target = self.root / "staged"
        real_open = Path.open

        def full_disk(path: Path, *args: Any, **kwargs: Any) -> Any:
            """Open the staging file as one that runs out of space after the first chunk."""
            if path != target:
                return real_open(path, *args, **kwargs)
            handle = real_open(path, *args, **kwargs)
            writes = {"count": 0}
            original = handle.write

            def write(view: Any) -> int:
                writes["count"] += 1
                if writes["count"] == 2:
                    raise OSError(errno.ENOSPC, "No space left on device")
                return original(view)

            handle.write = write  # type: ignore[method-assign]
            return handle

        with patch.object(Path, "open", full_disk), self.assertRaises(api.StorageError) as caught:
            transport.download("objects/ab/blob", target, ref="id")
        self.assertNotIsInstance(caught.exception, OSError)
        self.assertIn("local disk write failed while restoring objects/ab/blob: No space left on device",
                      str(caught.exception))
        self.assertIn("RESTORE_SKIP", str(caught.exception))
        self.assertEqual(media_ranges(http), ["bytes=0-3", "bytes=4-7"])
        api.time.sleep.assert_not_called()

    def test_network_failure_mid_upload_resends_the_confirmed_chunk_from_the_file(self) -> None:
        """After a dropped PUT the chunk is read again from the offset Drive confirmed."""
        data = b"0123456789"
        service, http = sdk_service([
            ({"status": "200"}, b'{"files": []}'),
            ({"status": "200"}, b'{"ids": ["new-id"]}'),
            ({"status": "200", "location": "https://www.googleapis.com/upload/drive/v3/files?upload_id=u"}, b""),
            ({"status": "308", "range": "bytes=0-3"}, b""),
            ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"),
            ({"status": "308", "range": "bytes=0-3"}, b""),  # status query: only the first chunk arrived
            ({"status": "308", "range": "bytes=0-7"}, b""),
            metadata("new-id", "blob", data),
            metadata("new-id", "blob", data),
        ])
        transport = api.GoogleDriveTransport("project", lambda: service)
        transport.folders["objects"] = "objects-id"
        source = self.root / "source"
        source.write_bytes(data)
        self.assertEqual(transport.upload(source, "objects/blob"), "new-id")
        puts = [(headers.get("Content-Range") or headers.get("content-range"), body)
                for uri, method, body, headers in http.request_sequence if method == "PUT"]
        self.assertEqual(puts, [("bytes 0-3/10", b"0123"), ("bytes 4-7/10", b"4567"), ("bytes */10", None),
                                ("bytes 4-7/10", b"4567"), ("bytes 8-9/10", b"89")])

    def test_unreadable_local_file_fails_the_upload_as_local_without_sending_it(self) -> None:
        """An EIO while reading a chunk is not a network failure, and no half-sent PUT is left behind."""
        service, http = sdk_service([
            ({"status": "200"}, b'{"files": []}'),
            ({"status": "200"}, b'{"ids": ["new-id"]}'),
            ({"status": "200", "location": "https://www.googleapis.com/upload/drive/v3/files?upload_id=u"}, b""),
        ])
        transport = api.GoogleDriveTransport("project", lambda: service)
        transport.folders["objects"] = "objects-id"
        source = self.root / "source"
        source.write_bytes(b"0123456789")
        real_open = Path.open

        def bad_sector(path: Path, *args: Any, **kwargs: Any) -> Any:
            """Open the source as a file whose reads fail."""
            handle = real_open(path, *args, **kwargs)
            if path == source:
                handle.read = Mock(side_effect=OSError(errno.EIO, "Input/output error"))  # type: ignore[method-assign]
            return handle

        with patch.object(api, "file_checksums", return_value=(10, hashlib.md5(b"0123456789").hexdigest())), \
             patch.object(Path, "open", bad_sector), self.assertRaises(api.StorageError) as caught:
            transport.upload(source, "objects/blob")
        self.assertEqual(str(caught.exception),
                         f"local disk read failed while backing up objects/blob from {source}: Input/output error")
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertEqual([method for _, method, _, _ in http.request_sequence], ["GET", "GET", "POST"])
        api.time.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()

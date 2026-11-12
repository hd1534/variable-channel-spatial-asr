"""Test transactional personal-Drive backups without needing a Google account."""

from __future__ import annotations

import hashlib
import http.client
import http.server
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import drive_storage as storage
import drive_api as api

SDK_AVAILABLE: bool = importlib.util.find_spec("googleapiclient") is not None


class MemoryTransport:
    """Object storage fixture with deterministic upload failure injection."""

    def __init__(self) -> None:
        """Start an empty remote with no account credentials."""
        self.remote: str = "gdrive:fixture-project"
        self.objects: dict[str, bytes] = {}
        self.fail_next: bool = False
        self.uploads: list[str] = []

    def prepare(self) -> None:
        """No remote account initialization is needed for this fixture."""

    def commits(self) -> list[str]:
        """List only complete manifests."""
        return [p.split("/")[-1] for p in self.objects if p.startswith("commits/")]

    def manifest(self, name: str) -> dict[str, Any]:
        """Decode one published manifest."""
        return json.loads(self.objects["commits/" + name])

    def upload(self, local: Path, relative: str) -> None:
        """Store bytes or simulate a dropped connection before publication."""
        if self.fail_next:
            self.fail_next = False
            raise storage.StorageError("simulated upload failure")
        self.objects[relative] = local.read_bytes()
        self.uploads.append(relative)

    def download(self, relative: str, local: Path) -> None:
        """Return stored bytes, including injected corruption for verification tests."""
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self.objects[relative])


class TestSnapshots(unittest.TestCase):
    """Exercise interrupted uploads, restore integrity and application dependencies."""

    def setUp(self) -> None:
        """Isolate source and restore directories for each scenario."""
        self.temporary: Any = tempfile.TemporaryDirectory()
        self.directory: Path = Path(self.temporary.name)
        self.remote: MemoryTransport = MemoryTransport()
        self.root: Path = self.directory / "source"
        self.store: storage.SnapshotStore = storage.SnapshotStore(self.root, self.remote, log=lambda _: None)
        self.store.restore()

    def tearDown(self) -> None:
        """Remove all synthetic snapshot data."""
        self.temporary.cleanup()

    def put(self, relative: str, value: bytes) -> Path:
        """Create a synthetic project file."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def restore_copy(self, name: str = "restored") -> Path:
        """Restore a fresh workspace through the public store API."""
        path = self.directory / name
        storage.SnapshotStore(path, self.remote, log=lambda _: None).restore()
        return path

    def test_round_trip_models_symlinks_and_secret_exclusion(self) -> None:
        """HF model links survive while tokens and temporary writes never leave the VM."""
        self.put("runs/model/config.json", b'{"setting":1}')
        self.put("models/asr/blobs/model", b"weights")
        link = self.root / "models/asr/snapshots/rev/weights.bin"
        link.parent.mkdir(parents=True)
        link.symlink_to("../../blobs/model")
        for path in ("token.json", "private/drive-auth.json", "private/rclone.conf", "runs/run/best.pt.tmp",
                     "models/asr/.locks/file.lock"):
            self.put(path, b"SECRET")
        self.store.backup()
        destination = self.restore_copy()
        self.assertEqual((destination / "runs/model/config.json").read_bytes(), b'{"setting":1}')
        self.assertTrue((destination / "models/asr/snapshots/rev/weights.bin").is_symlink())
        self.assertEqual((destination / "models/asr/snapshots/rev/weights.bin").read_bytes(), b"weights")
        self.assertFalse(any(b"SECRET" in data for data in self.remote.objects.values()))

    def test_failed_upload_does_not_publish_or_replace_previous_backup(self) -> None:
        """A failed checkpoint transfer leaves the last complete restore point intact."""
        path = self.put("runs/run/best.pt", b"first")
        first = self.store.backup()
        path.write_bytes(b"second")
        self.remote.fail_next = True
        with self.assertRaisesRegex(storage.StorageError, "upload failure"):
            self.store.backup()
        self.assertEqual(storage._head(self.remote.commits())[0], first)
        self.assertEqual((self.restore_copy() / "runs/run/best.pt").read_bytes(), b"first")
        self.store.backup()
        self.assertEqual((self.restore_copy("second-copy") / "runs/run/best.pt").read_bytes(), b"second")

    def test_corruption_is_detected_before_live_tree_is_exposed(self) -> None:
        """A damaged remote object cannot become an apparently successful restore."""
        self.put("runs/run/best.pt", b"valid weights")
        self.store.backup()
        digest = self.store.entries["runs/run/best.pt"]["sha256"]
        self.remote.objects[storage._object(digest)] = b"damaged"
        destination = self.directory / "restore"
        with self.assertRaisesRegex(storage.StorageError, "checksum mismatch"):
            storage.SnapshotStore(destination, self.remote, log=lambda _: None).restore()
        self.assertFalse(destination.exists())

    def test_live_file_changes_retain_its_previous_generation(self) -> None:
        """A file changed during copying is not uploaded under an incorrect content hash."""
        path = self.put("metrics.jsonl", b'{"step":1}\n')
        self.store.backup()
        path.write_bytes(b'{"step":2}\n')
        original = storage._signature
        count = 0

        def signature(source: Path) -> list[int]:
            """Simulate an append between the two generation checks."""
            nonlocal count
            if source.resolve() == path.resolve():
                count += 1
                if count == 2:
                    path.write_bytes(b'{"step":2}\n{"step":3}\n')
            return original(source)

        with patch.object(storage, "_signature", side_effect=signature):
            self.store.backup()
        self.assertEqual((self.restore_copy() / "metrics.jsonl").read_bytes(), b'{"step":1}\n')

    def test_checkpoint_index_is_committed_only_with_its_weights(self) -> None:
        """An incomplete save/prune transaction falls back as a whole and later recovers."""
        first = self.put("runs/run/ckpt/step_0000001.pt", b"one")
        index = self.put("runs/run/ckpt/LATEST.json", json.dumps([
            {"file": first.name, "bytes": 3, "sha256": hashlib.sha256(b"one").hexdigest()}]).encode())
        self.store.backup()
        first.unlink()
        index.write_text(json.dumps([{"file": "step_0000002.pt", "bytes": 3,
                                      "sha256": hashlib.sha256(b"two").hexdigest()}]))
        self.store.backup()
        restored = self.restore_copy()
        self.assertEqual(json.loads((restored / "runs/run/ckpt/LATEST.json").read_text())[0]["file"], first.name)
        self.assertTrue((restored / "runs/run/ckpt" / first.name).exists())
        self.put("runs/run/ckpt/step_0000002.pt", b"two")
        self.store.backup()
        restored = self.restore_copy("next-checkpoint")
        self.assertEqual(json.loads((restored / "runs/run/ckpt/LATEST.json").read_text())[0]["file"], "step_0000002.pt")
        self.assertEqual((restored / "runs/run/ckpt/step_0000002.pt").read_bytes(), b"two")

    def test_rir_orphans_and_stale_scores_are_not_complete_results(self) -> None:
        """Restore points contain RIR pairs and scores written after their predictions."""
        self.put("rirbanks/bank/rirs_0000.npy", b"responses")
        self.put("eval/run/full/score.json", b'{"result":1}')
        prediction = self.put("eval/run/full/predictions.jsonl", b'{"case_id":"new"}\n')
        os.utime(prediction, ns=(time.time_ns() + 1000000, time.time_ns() + 1000000))
        self.store.backup()
        restored = self.restore_copy()
        self.assertFalse((restored / "rirbanks/bank/rirs_0000.npy").exists())
        self.assertFalse((restored / "eval/run/full/score.json").exists())
        self.put("rirbanks/bank/rooms_0000.json", b'[{"room":0}]')
        score = self.put("eval/run/full/score.json", b'{"result":2}')
        now = prediction.stat().st_mtime_ns + 1000000
        os.utime(score, ns=(now, now))
        self.store.backup()
        restored = self.restore_copy("complete")
        self.assertTrue((restored / "rirbanks/bank/rirs_0000.npy").exists())
        self.assertEqual(json.loads((restored / "eval/run/full/score.json").read_text())["result"], 2)

    def test_fast_backup_retains_large_caches_without_uploading_new_ones(self) -> None:
        """Training state can be committed before frozen models and corpus archives."""
        self.put("cache/first.tar.gz", b"cached")
        self.store.backup()
        self.put("models/large/model.bin", b"new weights")
        self.put("runs/run/best.pt", b"checkpoint")
        self.store.backup(include_static=False)
        restored = self.restore_copy()
        self.assertEqual((restored / "runs/run/best.pt").read_bytes(), b"checkpoint")
        self.assertTrue((restored / "cache/first.tar.gz").exists())
        self.assertFalse((restored / "models/large/model.bin").exists())
        self.store.backup()
        self.assertTrue((self.restore_copy("full") / "models/large/model.bin").exists())

    def test_content_deduplication_and_remote_history_preservation(self) -> None:
        """Identical payloads upload once and later local pruning never deletes remote objects."""
        a = self.put("a.bin", b"identical")
        self.put("b.bin", b"identical")
        self.store.backup()
        objects = [name for name in self.remote.objects if name.startswith("objects/")]
        self.assertEqual(len(objects), 1)
        a.unlink()
        self.store.backup()
        self.assertTrue(all(name in self.remote.objects for name in objects))
        self.assertFalse((self.restore_copy() / "a.bin").exists())

    def test_missing_local_root_is_restored_despite_an_existing_state_file(self) -> None:
        """A deleted local cache cannot turn into a new empty remote backup on reconnect."""
        self.put("data.bin", b"preserved")
        self.store.backup()
        shutil.rmtree(self.root)
        storage.SnapshotStore(self.root, self.remote, log=lambda _: None).restore()
        self.assertEqual((self.root / "data.bin").read_bytes(), b"preserved")

    def test_unterminated_jsonl_and_invalid_json_keep_previous_versions(self) -> None:
        """Partially written control files never replace complete committed records."""
        path = self.put("cache.jsonl", b'{"done":1}\n')
        control = self.put("config.json", b'{"ok":1}')
        self.store.backup()
        path.write_bytes(b'{"done":1}\n{"next":')
        control.write_bytes(b'{"ok":')
        self.store.backup()
        restored = self.restore_copy()
        self.assertEqual((restored / "cache.jsonl").read_bytes(), b'{"done":1}\n')
        self.assertEqual((restored / "config.json").read_bytes(), b'{"ok":1}')

    def test_concurrent_writers_and_divergent_histories_fail_clearly(self) -> None:
        """Independent runtimes cannot silently overwrite each other's work."""
        self.put("file.bin", b"first")
        self.store.backup()
        other_root = self.directory / "other"
        other = storage.SnapshotStore(other_root, self.remote, log=lambda _: None)
        other.restore()
        self.put("file.bin", b"newer")
        self.store.backup()
        (other_root / "file.bin").write_bytes(b"other writer")
        with self.assertRaisesRegex(storage.StorageError, "another runtime"):
            other.backup()
        root = "a" * 32
        with self.assertRaisesRegex(storage.StorageError, "concurrent backup histories"):
            storage._head([f"root--{root}.json", f"{root}--{'b' * 32}.json", f"{root}--{'c' * 32}.json"])

    def test_path_traversal_external_symlinks_and_overlapping_paths_are_rejected(self) -> None:
        """Untrusted remote metadata cannot access files outside the project tree."""
        entry = {"type": "file", "sha256": "0" * 64, "size": 0, "mtime_ns": 0}
        for entries in ({"../secret": entry}, {"/absolute": entry},
                        {"dir": entry, "dir/file": entry},
                        {"link": {"type": "symlink", "target": "../secret"}},
                        {"token.json": entry}):
            with self.subTest(entries=entries), self.assertRaises(storage.StorageError):
                storage._validate_manifest({"schema": 1, "id": "test", "files": entries}, "test")
        (self.root / "external").symlink_to(self.directory / "secret")
        with self.assertRaisesRegex(storage.StorageError, "symlink escapes"):
            self.store.backup()

    def test_uploaded_commit_recovers_after_local_state_write_failure(self) -> None:
        """A successful remote commit remains recoverable after a local crash."""
        self.put("data.bin", b"data")
        with patch.object(self.store, "_save", side_effect=OSError("local write interrupted")):
            with self.assertRaises(OSError):
                self.store.backup()
        retry = storage.SnapshotStore(self.root, self.remote, log=lambda _: None)
        retry.restore()
        self.assertEqual((self.root / "data.bin").read_bytes(), b"data")
        retry.backup()

    def test_session_flushes_completed_commands_and_guards_duplicate_sessions(self) -> None:
        """Command completion means data has been committed, and one local writer owns it."""
        session = storage.StorageSession(self.store, interval_sec=60)
        try:
            with self.assertRaisesRegex(storage.StorageError, "another backup session"):
                storage.StorageSession(self.store, interval_sec=60)
            target = self.root / "result.txt"
            session.run([sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('done')", str(target)])
            self.assertEqual((self.restore_copy() / "result.txt").read_text(), "done")
        finally:
            session.close()
        with self.assertRaisesRegex(storage.StorageError, "closed"):
            session.run([sys.executable, "-c", "pass"])

    def test_publication_listing_delay_is_not_mistaken_for_another_writer(self) -> None:
        """A completed upload can appear in Drive's search results after a short delay."""
        self.put("result.txt", b"complete")
        original = self.remote.commits
        count: int = 0

        def delayed() -> list[str]:
            """Hide a newly published record twice, then expose the complete history."""
            nonlocal count
            names = original()
            if names:
                count += 1
                if count <= 2:
                    return []
            return names

        with patch.object(self.remote, "commits", side_effect=delayed), patch.object(storage.time, "sleep") as sleep:
            self.store.backup()
            self.assertEqual(sleep.call_count, 2)
        self.assertEqual((self.restore_copy() / "result.txt").read_bytes(), b"complete")

    def test_write_probe_creates_a_verified_generation_even_without_file_changes(self) -> None:
        """Connecting to a restored store still verifies that new remote writes are allowed."""
        self.put("result.txt", b"done")
        first = self.store.backup()
        self.assertEqual(self.store.backup(), first)
        second = self.store.backup(force=True, include_static=False)
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.remote.commits()), 2)
        self.assertEqual((self.restore_copy() / "result.txt").read_bytes(), b"done")

    def test_failed_background_backup_aborts_the_running_job(self) -> None:
        """A failed persistent backup cannot leave an unattended training subprocess running."""
        self.put("result.txt", b"done")
        self.remote.fail_next = True
        session = storage.StorageSession(self.store, interval_sec=0.02)
        try:
            with self.assertRaisesRegex(storage.StorageError, "backup failed"):
                session.run([sys.executable, "-c", "import time; time.sleep(30)"])
            self.assertFalse(session.thread.is_alive())
            self.assertTrue((self.root / "result.txt").exists())
        finally:
            with self.assertRaises(storage.StorageError):
                session.close()

    def test_visibility_timeout_preserves_local_files_and_allows_commit_recovery(self) -> None:
        """An unlisted upload is not reported complete, and reconnect can still recover it."""
        self.put("result.txt", b"done")
        with patch.object(self.remote, "commits", return_value=[]), patch.object(storage.time, "sleep"):
            with self.assertRaisesRegex(storage.StorageError, "not visible yet"):
                self.store.backup()
        self.assertIsNone(self.store.base)
        self.assertEqual((self.root / "result.txt").read_bytes(), b"done")
        retry = storage.SnapshotStore(self.root, self.remote, log=lambda _: None)
        retry.restore()
        self.assertIsNotNone(retry.base)
        self.assertEqual((self.restore_copy() / "result.txt").read_bytes(), b"done")


def auth_package() -> dict[str, Any]:
    """Return synthetic OAuth data; it is never used against a real account."""
    return {"schema": 1, "folder_id": "fixture-project", "account": "fixture@example.com", "credentials": {
        "client_id": "fixture.apps.googleusercontent.com", "client_secret": "SECRET", "refresh_token": "SECRET",
        "token_uri": api.TOKEN_URI, "scopes": list(api.SCOPES)}}


class TestCredentials(unittest.TestCase):
    """Check local credential boundaries and executable notebook integration."""

    def test_colab_secret_is_validated_and_saved_with_private_permissions(self) -> None:
        """Colab credentials never need to appear in a code cell or public backup."""
        package = Mock()
        package.userdata.get.return_value = json.dumps(auth_package())
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"google.colab": package}):
            path = Path(directory) / "drive-auth.json"
            with patch("builtins.print") as output:
                result = storage.auth_from_colab_secret("DRIVE_AUTH", path)
            self.assertEqual(result, auth_package())
            self.assertEqual(json.loads(path.read_text()), result)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            output.assert_not_called()

    def test_unsafe_endpoints_missing_offline_access_and_entire_drive_are_rejected(self) -> None:
        """A malformed secret cannot redirect refresh tokens or silently broaden access."""
        for key, value in (("token_uri", "https://example.com/token"), ("refresh_token", ""),
                           ("scopes", ["https://www.googleapis.com/auth/drive"])):
            package = auth_package()
            package["credentials"][key] = value
            with self.subTest(key=key), self.assertRaises(storage.StorageError):
                api.validate_auth(package)
        for folder in ("root", "../outside", ""):
            package = auth_package()
            package["folder_id"] = folder
            with self.subTest(folder=folder), self.assertRaises(storage.StorageError):
                api.validate_auth(package)

    def test_credentials_inside_data_root_are_rejected_before_network_access(self) -> None:
        """A convenience setting cannot cause private authorization to be uploaded."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(storage.StorageError, "outside LOCAL_STORE_ROOT"):
                storage.connect_personal_drive(root, auth_file=root / "drive-auth.json")

    def test_notebooks_use_official_api_and_preserve_the_two_asr_evaluations(self) -> None:
        """All notebook jobs share verified backups and retain the existing experiment choices."""
        repository = Path(__file__).resolve().parents[1]
        for name in ("train_template.ipynb", "mono_baselines.ipynb", "track_b_vibevoice.ipynb"):
            notebook = json.loads((repository / "notebooks" / name).read_text())
            sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
            for i, source in enumerate(sources):
                compile(source, f"{name}:{i}", "exec")
            code = "\n".join(sources)
            self.assertIn('STORAGE_BACKEND: str = "drive_api"', code)
            self.assertIn("from drive_storage import connect_personal_drive", code)
            self.assertIn("requirements-drive.txt", code)
            self.assertIn('os.environ["HF_HUB_CACHE"]', code)
            self.assertIn("storage.run(", code)
            self.assertIn("storage.flush()", code)
            self.assertNotIn("rclone", code)
            self.assertIn('elif STORAGE_BACKEND == "mounted_drive":', code)
            if name == "train_template.ipynb":
                self.assertIn('"all": ["qwen3-1.7b", "wav2vec2"]', code)
                self.assertIn("share_embeddings_across_asr", code)


class ApiFailure(Exception):
    """Structured Drive error fixture without requiring installed Google libraries."""

    def __init__(self, status: int, reason: str) -> None:
        """Expose the HTTP fields used by the official client's exceptions."""
        super().__init__(reason)
        self.resp: Any = Mock(status=status)
        self.content: bytes = json.dumps({"error": {"errors": [{"reason": reason}]}}).encode()


class TestRetries(unittest.TestCase):
    """Transient quota failures must not be confused with permanent storage/permission failures."""

    def test_rate_limits_server_errors_and_network_errors_use_bounded_backoff(self) -> None:
        """Retry precisely the errors that can succeed without changing user authorization."""
        for error in (ApiFailure(403, "rateLimitExceeded"), ApiFailure(429, "tooManyRequests"),
                      ApiFailure(503, "backendError"), TimeoutError()):
            operation = Mock(side_effect=[error, "ok"])
            with self.subTest(error=type(error).__name__), patch.object(api.time, "sleep") as sleep:
                self.assertEqual(api.retry(operation), "ok")
                self.assertEqual(operation.call_count, 2)
                sleep.assert_called_once()

    def test_dropped_responses_and_token_endpoint_outages_are_retried(self) -> None:
        """A cut-off response or a token endpoint 5xx is a network blip, not a dead backup."""
        class RefreshError(Exception):
            """Stand-in shaped like google-auth's: (details, JSON body) or (non-JSON body,)."""

        outages = (http.client.IncompleteRead(b"partial"),
                   RefreshError("temporarily_unavailable: retry", {"error": "temporarily_unavailable"}),
                   RefreshError("<html>502 Bad Gateway</html>"))  # google-auth marks this not retryable
        for error in outages:
            operation = Mock(side_effect=[error, "ok"])
            with self.subTest(error=str(error)), patch.object(api.time, "sleep"):
                self.assertEqual(api.retry(operation), "ok")
            self.assertNotIn("revoked", str(api.api_error(error)))
        revoked = RefreshError("invalid_grant: Token has been expired or revoked.", {"error": "invalid_grant"})
        self.assertFalse(api.is_transient(revoked))
        self.assertIn("revoked", str(api.api_error(revoked)))

    def test_storage_and_permission_errors_are_not_retried_or_logged_verbatim(self) -> None:
        """An unusable account fails promptly without printing raw API response content."""
        for reason in ("storageQuotaExceeded", "insufficientPermissions", "accessNotConfigured"):
            request = Mock()
            request.execute.side_effect = ApiFailure(403, reason)
            with self.subTest(reason=reason), patch.object(api.time, "sleep") as sleep:
                with self.assertRaises(storage.StorageError):
                    api.execute(request)
                request.execute.assert_called_once()
                sleep.assert_not_called()

    def test_transient_errors_eventually_fail_instead_of_retrying_forever(self) -> None:
        """A persistent quota outage is propagated after the finite backoff budget."""
        request = Mock()
        request.execute.side_effect = ApiFailure(429, "tooManyRequests")
        with patch.object(api.time, "sleep") as sleep, self.assertRaises(storage.StorageError):
            api.execute(request)
        self.assertEqual(sleep.call_count, len(api.RETRY_DELAYS))
        self.assertEqual(request.execute.call_count, len(api.RETRY_DELAYS) + 1)


@unittest.skipUnless(SDK_AVAILABLE, "install requirements-drive.txt to exercise Google's actual SDK")
class TestOfficialSdk(unittest.TestCase):
    """Exercise the real SDK's resumable upload and streaming download HTTP protocol."""

    def test_sdk_upload_download_and_server_checksum_verification(self) -> None:
        """Real MediaFileUpload/MediaIoBaseDownload operate through Drive v3 requests."""
        from googleapiclient.discovery import build
        from googleapiclient.http import HttpMockSequence

        data = b"checkpoint bytes\x00\xff"
        metadata = {"id": "uploaded-id", "name": "digest", "mimeType": "application/octet-stream",
                    "size": str(len(data)), "md5Checksum": hashlib.md5(data).hexdigest()}
        http = HttpMockSequence([
            ({"status": "200"}, json.dumps({"files": []}).encode()),
            ({"status": "200"}, b'{"ids":["uploaded-id"]}'),
            ({"status": "200", "location": "https://www.googleapis.com/upload/drive/v3/files?upload_id=fixture"}, b""),
            ({"status": "200"}, json.dumps(metadata).encode()),
            ({"status": "200"}, json.dumps(metadata).encode()),
            ({"status": "200", "content-length": str(len(data))}, data),
        ])
        service = build("drive", "v3", http=http, cache_discovery=False, static_discovery=True)
        transport = api.GoogleDriveTransport("project-id", lambda: service)
        transport.folders.update({"objects": "objects-id", "objects/ab": "prefix-id"})
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source", Path(directory) / "restored"
            source.write_bytes(data)
            transport.upload(source, "objects/ab/digest")
            transport.download("objects/ab/digest", destination)
            self.assertEqual(destination.read_bytes(), data)
            self.assertEqual(transport.files["objects/ab/digest"]["id"], "uploaded-id")

    def test_existing_immutable_object_is_checked_and_never_overwritten(self) -> None:
        """A wrong remote hash is an integrity failure, not permission to replace history."""
        service = Mock()
        transport = api.GoogleDriveTransport("project-id", lambda: service)
        transport.folders["objects"] = "objects-id"
        transport.files["objects/blob"] = {"id": "existing", "size": "3", "md5Checksum": "wrong"}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "file"
            source.write_bytes(b"one")
            with self.assertRaisesRegex(storage.StorageError, "checksum mismatch"):
                transport.upload(source, "objects/blob")
            service.files.assert_not_called()

    def test_resumable_upload_conflict_recovers_the_pre_generated_id(self) -> None:
        """A lost upload response cannot create duplicate files or publish unchecked bytes."""
        service = Mock()
        files = service.files.return_value
        files.list.return_value.execute.return_value = {"files": []}
        files.generateIds.return_value.execute.return_value = {"ids": ["known-id"]}
        files.create.return_value.next_chunk.side_effect = ApiFailure(409, "alreadyExists")
        files.get.return_value.execute.return_value = {
            "id": "known-id", "size": "3", "md5Checksum": hashlib.md5(b"one").hexdigest()}
        transport = api.GoogleDriveTransport("project-id", lambda: service)
        transport.folders["objects"] = "objects-id"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "file"
            source.write_bytes(b"one")
            transport.upload(source, "objects/blob")
        self.assertEqual(files.create.call_args.kwargs["body"]["id"], "known-id")
        self.assertEqual(files.get.call_args.kwargs["fileId"], "known-id")

    def test_http_clients_are_created_per_thread(self) -> None:
        """Periodic backups cannot accidentally share a non-thread-safe httplib2 session."""
        factory = Mock(side_effect=lambda: object())
        transport = api.GoogleDriveTransport("project-id", factory)
        main = transport.service
        observed: list[Any] = []
        thread = threading.Thread(target=lambda: observed.append(transport.service))
        thread.start()
        thread.join()
        self.assertIsNot(main, observed[0])
        self.assertIs(main, transport.service)
        self.assertEqual(factory.call_count, 2)

    def test_desktop_authorization_uses_personal_client_and_keeps_tokens_private(self) -> None:
        """Browser consent creates a scoped private package without shared client credentials."""
        from setup_drive import authorize

        package = auth_package()
        credentials = Mock(refresh_token="SECRET", client_id=package["credentials"]["client_id"])
        credentials.to_json.return_value = json.dumps(package["credentials"])
        with tempfile.TemporaryDirectory() as directory:
            client, output = Path(directory) / "credentials.json", Path(directory) / "drive-auth.json"
            client.write_text('{"installed":{"client_id":"personal-client"}}')
            with patch("google_auth_oauthlib.flow.InstalledAppFlow") as flow, \
                 patch("googleapiclient.discovery.build"), patch("setup_drive.execute") as execute, \
                 patch("setup_drive.create_folder", return_value={"id": package["folder_id"]}), \
                 patch("setup_drive.GoogleDriveTransport"), patch("builtins.print") as printed:
                flow.from_client_config.return_value.run_local_server.return_value = credentials
                execute.return_value = {"user": {"emailAddress": package["account"]}}
                self.assertEqual(authorize(client, output), package)
                self.assertEqual(flow.from_client_config.call_args.kwargs["scopes"], api.SCOPES)
                self.assertIn("select_account", flow.from_client_config.return_value.run_local_server.call_args.kwargs["prompt"])
                self.assertNotIn("SECRET", str(printed.call_args_list))
                self.assertEqual(output.stat().st_mode & 0o777, 0o600)
                with self.assertRaisesRegex(storage.StorageError, "already exists"):
                    authorize(client, output)

    def test_drive_listing_reads_all_pages_and_rejects_duplicate_manifest_names(self) -> None:
        """A long backup history cannot hide its head on a later API page."""
        first, second = "root--" + "a" * 32 + ".json", "a" * 32 + "--" + "b" * 32 + ".json"
        service = Mock()
        request = service.files.return_value.list.return_value
        rows = [{"id": str(i), "name": name, "mimeType": "application/octet-stream"}
                for i, name in enumerate((first, second))]
        request.execute.side_effect = [{"files": rows[:1], "nextPageToken": "next"}, {"files": rows[1:]}]
        transport = api.GoogleDriveTransport("project-id", lambda: service)
        transport.folders["commits"] = "commits-id"
        self.assertEqual(transport.commits(), [first, second])
        self.assertEqual(service.files.return_value.list.call_args.kwargs["pageToken"], "next")
        request.execute.side_effect = None
        request.execute.return_value = {"files": [rows[0], {**rows[0], "id": "duplicate"}]}
        with self.assertRaisesRegex(storage.StorageError, "duplicate"):
            transport.commits()

    def test_dns_and_token_refresh_network_failures_are_retried(self) -> None:
        """httplib2 and google-auth wrap these, so they are not OSError subclasses."""
        import httplib2
        from google.auth.exceptions import RefreshError, TransportError

        for error in (httplib2.ServerNotFoundError("Unable to find the server"), TransportError("refresh"),
                      RefreshError("<html>Bad Gateway</html>", retryable=False)):
            operation = Mock(side_effect=[error, "ok"])
            with self.subTest(error=type(error).__name__), patch.object(api.time, "sleep"):
                self.assertEqual(api.retry(operation), "ok")
        revoked = RefreshError("invalid_grant: Token has been expired or revoked.", {"error": "invalid_grant"})
        self.assertFalse(api.is_transient(revoked))
        self.assertIn("revoked", str(api.api_error(revoked)))

    def test_service_http_keeps_resumable_308_replies(self) -> None:
        """Drive's '308 Resume Incomplete' has no Location; following it as a redirect fails every chunk.

        Real httplib2 against a local server: HttpMockSequence bypasses the redirect code entirely.
        """
        import httplib2

        class ResumeIncomplete(http.server.BaseHTTPRequestHandler):
            """Answer every PUT like Drive answers a non-final resumable chunk."""

            def do_PUT(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(308)
                self.send_header("Range", "bytes=0-0")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ResumeIncomplete)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/upload"
        try:
            inner = api.build_service(auth_package()["credentials"])._http.http
            response, _ = inner.request(url, "PUT", body=b"x")
            self.assertEqual(response.status, 308)
            with self.assertRaises(httplib2.RedirectMissingLocation):  # what the plain client did
                httplib2.Http(timeout=5).request(url, "PUT", body=b"x")
        finally:
            server.shutdown()
            server.server_close()

    def test_large_objects_use_multiple_upload_and_download_chunks(self) -> None:
        """Model files larger than one chunk stay streamable in both transfer directions."""
        from googleapiclient.discovery import build
        from googleapiclient.http import HttpMockSequence

        data = b"x" * api.CHUNK_BYTES + b"end"
        metadata = {"id": "large-id", "size": str(len(data)), "md5Checksum": hashlib.md5(data).hexdigest()}
        http = HttpMockSequence([
            ({"status": "200"}, b'{"files":[]}'),
            ({"status": "200"}, b'{"ids":["large-id"]}'),
            ({"status": "200", "location": "https://www.googleapis.com/upload/drive/v3/files?upload_id=fixture"}, b""),
            ({"status": "308", "range": f"bytes=0-{api.CHUNK_BYTES - 1}"}, b""),
            ({"status": "200"}, json.dumps(metadata).encode()),
            ({"status": "200"}, json.dumps(metadata).encode()),
            ({"status": "206", "content-range": f"bytes 0-{api.CHUNK_BYTES - 1}/{len(data)}"}, data[:api.CHUNK_BYTES]),
            ({"status": "206", "content-range": f"bytes {api.CHUNK_BYTES}-{len(data) - 1}/{len(data)}"}, data[api.CHUNK_BYTES:]),
        ])
        service = build("drive", "v3", http=http, cache_discovery=False, static_discovery=True)
        transport = api.GoogleDriveTransport("project-id", lambda: service)
        transport.folders["objects"] = "objects-id"
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source", Path(directory) / "restored"
            source.write_bytes(data)
            transport.upload(source, "objects/large")
            transport.download("objects/large", destination)
            self.assertEqual(destination.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()



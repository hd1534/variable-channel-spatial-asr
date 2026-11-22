"""Test transactional personal-Drive backups without needing a Google account."""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import http.server
import importlib.util
import io
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
import transcripts

SDK_AVAILABLE: bool = importlib.util.find_spec("googleapiclient") is not None
# The real ASR provenance check (modern_asr) needs numpy, which the Drive-only environment may lack.
modern_asr: Any = importlib.import_module("modern_asr") if importlib.util.find_spec("numpy") else None
needs_modern_asr = unittest.skipUnless(modern_asr, "numpy is not installed")


class MemoryTransport:
    """Object storage fixture with upload failure injection, listing lag and direct references."""

    def __init__(self) -> None:
        """Start an empty remote with no account credentials."""
        self.remote: str = "gdrive:fixture-project"
        self.objects: dict[str, bytes] = {}
        self.fail_next: bool = False
        self.uploads: list[str] = []
        self.hidden: set[str] = set()
        self.downloads: list[tuple[str, str | None]] = []

    def prepare(self) -> None:
        """No remote account initialization is needed for this fixture."""

    def commits(self) -> list[str]:
        """List complete manifests, except those Drive's search index has not caught up with."""
        return [p.split("/")[-1] for p in self.objects if p.startswith("commits/") and p.split("/")[-1] not in self.hidden]

    def manifest(self, name: str) -> dict[str, Any]:
        """Decode one published manifest; like Drive's name lookup it cannot find an unlisted one."""
        if name in self.hidden:
            raise storage.StorageError("committed backup manifest is missing")
        return json.loads(self.objects["commits/" + name])

    def upload(self, local: Path, relative: str) -> str:
        """Store bytes or simulate a dropped connection before publication."""
        if self.fail_next:
            self.fail_next = False
            raise storage.StorageError("simulated upload failure")
        self.objects[relative] = local.read_bytes()
        self.uploads.append(relative)
        return "ref:" + relative

    def download(self, relative: str, local: Path, ref: str | None = None) -> None:
        """Return stored bytes, including injected corruption for verification tests."""
        if ref is not None and ref != "ref:" + relative:
            raise AssertionError("reference does not belong to the object")
        self.downloads.append((relative, ref))
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self.objects[relative])


def quiet(_: str) -> None:
    """Discard storage progress lines."""


class StoreCase(unittest.TestCase):
    """Shared workspace helpers."""

    def setUp(self) -> None:
        """Isolate source and restore directories for each scenario."""
        self.temporary: Any = tempfile.TemporaryDirectory()
        self.directory: Path = Path(self.temporary.name)
        self.remote: MemoryTransport = MemoryTransport()
        self.root: Path = self.directory / "source"
        self.store: storage.SnapshotStore = storage.SnapshotStore(self.root, self.remote, log=quiet)
        self.store.restore()

    def tearDown(self) -> None:
        """Remove all synthetic snapshot data."""
        self.temporary.cleanup()

    def put(self, relative: str, value: bytes, root: Path | None = None) -> Path:
        """Create a synthetic project file."""
        path = (root or self.root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def runtime(self, name: str, **options: Any) -> storage.SnapshotStore:
        """Connect another runtime (a fresh VM) to the same backup folder."""
        store = storage.SnapshotStore(self.directory / name, self.remote, log=quiet, **options)
        store.restore()
        return store

    def restore_copy(self, name: str = "restored") -> Path:
        """Restore a fresh workspace through the public store API."""
        path = self.directory / name
        storage.SnapshotStore(path, self.remote, log=quiet).restore()
        return path

    def heads(self) -> list[str]:
        """Current remote heads."""
        return storage.History(self.remote.commits()).heads


class TestSnapshots(StoreCase):
    """Exercise interrupted uploads, restore integrity and application dependencies."""

    def test_round_trip_models_symlinks_and_secret_exclusion(self) -> None:
        """HF model links survive while tokens and temporary writes never leave the VM."""
        self.put("runs/model/config.json", b'{"setting":1}')
        self.put("models/asr/blobs/model", b"weights")
        link = self.root / "models/asr/snapshots/rev/weights.bin"
        link.parent.mkdir(parents=True)
        link.symlink_to("../../blobs/model")
        for path in ("token.json", "private/drive-auth.json", "private/rclone.conf", "runs/run/best.pt.tmp",
                     "models/asr/.locks/file.lock", "runs/run/.train.lock"):
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
        self.assertEqual(self.heads(), [first])
        self.assertEqual((self.restore_copy() / "runs/run/best.pt").read_bytes(), b"first")
        self.store.backup()
        self.assertEqual((self.restore_copy("second-copy") / "runs/run/best.pt").read_bytes(), b"second")

    def test_corruption_is_detected_before_live_tree_is_exposed(self) -> None:
        """A damaged remote object cannot become an apparently successful restore."""
        self.put("runs/run/best.pt", b"valid weights")
        self.store.backup()
        digest = self.store.files["runs/run/best.pt"]["sha256"]
        self.remote.objects[storage._object(digest)] = b"damaged"
        destination = self.directory / "restore"
        with self.assertRaisesRegex(storage.StorageError, "checksum mismatch"):
            storage.SnapshotStore(destination, self.remote, log=quiet).restore()
        self.assertFalse(destination.exists())

    def test_downloads_use_the_direct_reference_recorded_at_upload(self) -> None:
        """Restores read objects by the id returned at upload, without depending on search listings."""
        self.put("runs/run/best.pt", b"weights")
        self.store.backup()
        self.assertTrue(self.store.files["runs/run/best.pt"]["ref"].startswith("ref:objects/"))
        self.restore_copy()
        self.assertTrue(self.remote.downloads and all(ref for _, ref in self.remote.downloads))

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
                if count == 3:
                    path.write_bytes(b'{"step":2}\n{"step":3}\n')
            return original(source)

        with patch.object(storage, "_signature", side_effect=signature):
            self.store.backup()
        self.assertEqual((self.restore_copy() / "metrics.jsonl").read_bytes(), b'{"step":1}\n')
        self.store.backup()
        self.assertEqual((self.restore_copy("later") / "metrics.jsonl").read_bytes(), b'{"step":2}\n{"step":3}\n')

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
        self.assertFalse((restored / "runs/run/ckpt" / first.name).exists(), "a pruned checkpoint must not come back")

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
        storage.SnapshotStore(self.root, self.remote, log=quiet).restore()
        self.assertEqual((self.root / "data.bin").read_bytes(), b"preserved")

    def test_emptied_workspace_is_restored_instead_of_deleting_the_backup(self) -> None:
        """Wiping the files but not the state directory must not publish a mass deletion."""
        self.put("runs/run/best.pt", b"keep")
        head = self.store.backup()
        for child in self.root.iterdir():
            shutil.rmtree(child)
        storage.SnapshotStore(self.root, self.remote, log=quiet).restore()
        self.assertEqual((self.root / "runs/run/best.pt").read_bytes(), b"keep")
        self.assertEqual(self.heads(), [head])

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

    def test_path_traversal_external_symlinks_and_overlapping_paths_are_rejected(self) -> None:
        """Untrusted remote metadata cannot access files outside the project tree."""
        entry = {"type": "file", "sha256": "0" * 64, "size": 0, "mtime_ns": 0}
        for entries in ({"../secret": entry}, {"/absolute": entry},
                        {"dir": entry, "dir/file": entry},
                        {"link": {"type": "symlink", "target": "../secret"}},
                        {"token.json": entry}, {"file": {**entry, "ref": 7}}):
            with self.subTest(entries=entries), self.assertRaises(storage.StorageError):
                storage._validate_manifest({"schema": 2, "id": "test", "files": entries}, "test")
        with self.assertRaisesRegex(storage.StorageError, "does not match its commit name"):
            storage._validate_manifest({"schema": 2, "id": "test", "parents": ["a" * 32], "files": {}}, "test", [])
        (self.root / "external").symlink_to(self.directory / "secret")
        with self.assertRaisesRegex(storage.StorageError, "symlink escapes"):
            self.store.backup()

    def test_uploaded_commit_recovers_after_local_state_write_failure(self) -> None:
        """A successful remote commit remains recoverable after a local crash."""
        self.put("data.bin", b"data")
        with patch.object(self.store, "_save", side_effect=OSError("local write interrupted")):
            with self.assertRaises(OSError):
                self.store.backup()
        retry = storage.SnapshotStore(self.root, self.remote, log=quiet)
        retry.restore()
        self.assertEqual((self.root / "data.bin").read_bytes(), b"data")
        retry.backup()
        self.assertEqual(len(self.heads()), 1)
        self.assertEqual((self.restore_copy() / "data.bin").read_bytes(), b"data")

    def test_interrupted_install_is_finished_instead_of_refused(self) -> None:
        """A stop between moving a restored tree into place and saving its state is completed later."""
        self.put("runs/run/best.pt", b"weights")
        self.store.backup()
        target = self.directory / "fresh-vm"
        first = storage.SnapshotStore(target, self.remote, log=quiet)
        with patch.object(storage.SnapshotStore, "_finish_install", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                first.restore()
        self.assertTrue((target / "runs/run/best.pt").exists())
        resumed = storage.SnapshotStore(target, self.remote, log=quiet)
        resumed.restore()
        self.assertIn("runs/run/best.pt", resumed.local)
        resumed.backup(force=True)
        self.assertEqual((self.restore_copy() / "runs/run/best.pt").read_bytes(), b"weights")

    def test_restore_refuses_untracked_files_but_seeds_an_empty_backup(self) -> None:
        """Existing files are never silently mixed into a backup, except to start an empty one."""
        seed = self.directory / "seed"
        self.put("runs/old/summary.json", b'{"old":1}', root=seed)
        other = MemoryTransport()
        store = storage.SnapshotStore(seed, other, log=quiet)
        store.restore()
        store.sync()
        self.assertEqual(len(storage.History(other.commits()).heads), 1)
        self.put("runs/x/a.txt", b"x")
        self.store.backup()
        crowded = self.directory / "crowded"
        self.put("stray.txt", b"untracked", root=crowded)
        with self.assertRaisesRegex(storage.StorageError, "import"):
            storage.SnapshotStore(crowded, self.remote, log=quiet).restore()

    def test_version_one_state_and_manifests_keep_working(self) -> None:
        """Backups and local state written before the merge-based history are adopted."""
        old = self.directory / "v1"
        data = b"v1 weights"
        digest = hashlib.sha256(data).hexdigest()
        self.remote.objects[storage._object(digest)] = data
        identifier = "a" * 32
        entry = {"type": "file", "sha256": digest, "size": len(data), "mtime_ns": 1}
        manifest = {"schema": 1, "id": identifier, "parent": None, "writer": "b" * 32,
                    "created_utc": "2026-11-12T00:00:00Z", "files": {"runs/r/best.pt": entry, "runs/r/gone.txt": entry}}
        self.remote.objects[f"commits/root--{identifier}.json"] = json.dumps(manifest).encode()
        self.put("runs/r/best.pt", data, root=old)
        state = storage._state_dir(old)
        state.mkdir(parents=True)
        (state / "state.json").write_text(json.dumps({
            "remote": self.remote.remote, "writer": "b" * 32, "base": identifier,
            "files": manifest["files"], "signatures": {}}))
        store = storage.SnapshotStore(old, self.remote, log=quiet)
        store.restore()
        self.put("runs/r/new.txt", b"new", root=old)
        head = store.backup()
        history = storage.History(self.remote.commits())
        self.assertEqual(history.heads, [head])
        self.assertIn(identifier, history.ancestors(head))
        restored = self.restore_copy()
        self.assertEqual((restored / "runs/r/best.pt").read_bytes(), data)
        self.assertEqual((restored / "runs/r/new.txt").read_bytes(), b"new")
        self.assertFalse((restored / "runs/r/gone.txt").exists(), "a file deleted under v1 must stay deleted")

    def test_session_flushes_completed_commands_and_guards_duplicate_sessions(self) -> None:
        """Command completion means data has been committed, and one local writer owns it."""
        session = storage.StorageSession(self.store, interval_sec=60)
        try:
            with self.assertRaisesRegex(storage.StorageError, "another backup session"):
                storage.StorageSession(self.store, interval_sec=60)
            with self.assertRaisesRegex(storage.StorageError, "storage.flush"):
                storage.acquire_lock(self.root)
            target = self.root / "result.txt"
            session.run([sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('done')", str(target)])
            self.assertEqual((self.restore_copy() / "result.txt").read_text(), "done")
        finally:
            session.close()
        with self.assertRaisesRegex(storage.StorageError, "closed"):
            session.run([sys.executable, "-c", "pass"])
        storage.acquire_lock(self.root).close()

    def test_flush_reports_an_up_to_date_backup(self) -> None:
        """The final notebook flush says so even when the previous command already backed everything up."""
        lines: list[str] = []
        self.store.log = lines.append
        session = storage.StorageSession(self.store, interval_sec=60)
        try:
            session.flush()
        finally:
            session.close()
        self.assertIn("[storage] backup up to date", lines)

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
        original = self.remote.upload

        def background_failure(local: Path, relative: str) -> str:
            """Fail only transfers started by the periodic backup thread."""
            if threading.current_thread().name == "vcsasr-backup":
                raise storage.StorageError("simulated upload failure")
            return original(local, relative)

        session = storage.StorageSession(self.store, interval_sec=0.05)
        try:
            with patch.object(self.remote, "upload", side_effect=background_failure):
                script = ("import pathlib, sys, time\n"
                          "for i in range(600):\n"
                          "    pathlib.Path(sys.argv[1]).write_text(str(i)); time.sleep(0.05)\n")
                with self.assertRaisesRegex(storage.StorageError, "backup failed"):
                    session.run([sys.executable, "-c", script, str(self.root / "progress.txt")])
            self.assertFalse(session.thread.is_alive())
            self.assertTrue((self.root / "result.txt").exists())
        finally:
            with self.assertRaises(storage.StorageError):
                session.close()


class TestHistories(StoreCase):
    """Several runtimes, CLI runs and delayed listings share one backup folder safely."""

    def test_a_second_runtime_never_stops_the_first_and_histories_merge(self) -> None:
        """Concurrent writers each commit, and every reader sees both sets of results."""
        self.put("runs/train/ckpt.pt", b"step 1")
        self.store.backup()
        mono = self.runtime("mono-vm")
        self.put("mono_baselines/vibevoice/score.json", b'{"acc":1}', root=mono.root)
        mono.backup(force=True)
        self.put("runs/train/ckpt.pt", b"step 2")
        self.store.backup()  # v1 raised "remote advanced in another runtime" and killed the job here
        self.assertEqual(len(self.heads()), 2)
        fresh = self.restore_copy("fresh-vm")
        self.assertEqual((fresh / "runs/train/ckpt.pt").read_bytes(), b"step 2")
        self.assertTrue((fresh / "mono_baselines/vibevoice/score.json").exists())
        self.store.sync()
        self.assertEqual(len(self.heads()), 1)
        self.assertTrue((self.root / "mono_baselines/vibevoice/score.json").exists())
        mono.sync()
        self.assertEqual((mono.root / "runs/train/ckpt.pt").read_bytes(), b"step 2")
        self.assertEqual(len(self.heads()), 1)

    def test_an_idle_runtime_that_is_behind_fast_forwards_without_committing(self) -> None:
        """A runtime left open while another one worked simply catches up on its next command."""
        self.put("runs/train/summary.json", b'{"done":1}')
        self.store.backup()
        other = self.runtime("other-vm")
        self.put("eval/train-qwen3-1.7b/eval-t2/full/score.json", b'{"acc":0.9}', root=other.root)
        latest = other.backup()
        count = len(self.remote.commits())
        self.store.sync()
        self.assertEqual(self.store.heads, [latest])
        self.assertEqual(len(self.remote.commits()), count)
        self.assertTrue((self.root / "eval/train-qwen3-1.7b/eval-t2/full/score.json").exists())
        storage.SnapshotStore(self.root, self.remote, log=quiet).restore()  # rerunning the connection cell

    def test_a_folder_changed_in_two_runtimes_keeps_the_newer_version_and_the_older_one_stays_recoverable(self) -> None:
        """The same run trained twice is resolved as a unit, and nothing committed is lost."""
        self.put("runs/carry/best.pt", b"base")  # trainers write weights before their summary
        self.put("runs/carry/summary.json", b'{"v":"base"}')
        self.store.backup()
        other = self.runtime("other-vm")
        self.put("runs/carry/extra.txt", b"only in the first")
        self.put("runs/carry/summary.json", b'{"v":"first"}')
        older = self.store.backup()
        time.sleep(0.01)
        self.put("runs/carry/best.pt", b"second", root=other.root)
        self.put("runs/carry/summary.json", b'{"v":"second"}', root=other.root)
        newer = other.backup()
        lines: list[str] = []
        self.store.log = lines.append
        self.store.sync()
        self.assertEqual(json.loads((self.root / "runs/carry/summary.json").read_text())["v"], "second")
        self.assertEqual((self.root / "runs/carry/best.pt").read_bytes(), b"second")
        self.assertFalse((self.root / "runs/carry/extra.txt").exists(), "a run is never mixed from two histories")
        self.assertTrue(any("runs/carry changed in two runtimes" in line and newer[:8] in line and older[:8] in line
                            for line in lines))
        out = self.directory / "recovered"
        storage.SnapshotStore(out, self.remote, log=quiet).export(older[:8])
        self.assertEqual(json.loads((out / "runs/carry/summary.json").read_text())["v"], "first")
        self.assertEqual((out / "runs/carry/extra.txt").read_bytes(), b"only in the first")

    def test_concurrent_transcript_cache_appends_are_combined(self) -> None:
        """Two runtimes transcribing different segments keep every cached transcript."""
        cache = "asr_cache/Qwen__Qwen3-ASR.jsonl"
        row = lambda key: (json.dumps({"key": key, "backend": "qwen", "text": key.upper()}) + "\n").encode()
        self.put(cache, row("a"))
        self.store.backup()
        other = self.runtime("other-vm")
        self.put(cache, row("a") + row("b"))
        self.store.backup()
        self.put(cache, row("a") + row("c"), root=other.root)
        other.backup()
        fresh = self.restore_copy("fresh-vm")
        self.assertEqual(transcripts.SegmentCache(fresh / cache, "qwen").text, {"a": "A", "b": "B", "c": "C"})
        storage.SnapshotStore(fresh, self.remote, log=quiet).backup(force=True)
        self.assertEqual(len(self.heads()), 1)
        self.store.sync()
        self.assertEqual(transcripts.SegmentCache(self.root / cache, "qwen").text, {"a": "A", "b": "B", "c": "C"})
        self.assertEqual(transcripts.SegmentCache(self.restore_copy("after") / cache, "qwen").text,
                         {"a": "A", "b": "B", "c": "C"})

    def test_an_unlisted_own_commit_is_never_forked(self) -> None:
        """Drive search lag cannot make a runtime branch from its own history (v1 'not visible yet')."""
        self.put("a.txt", b"1")
        first = self.store.backup()
        self.remote.hidden = set(self.remote.commits())
        self.put("a.txt", b"2")
        second = self.store.backup()
        reconnect = storage.SnapshotStore(self.root, self.remote, log=quiet)
        reconnect.restore()
        reconnect.backup(force=True)
        self.remote.hidden = set()
        history = storage.History(self.remote.commits())
        self.assertEqual(len(history.heads), 1)
        self.assertIn(first, history.ancestors(second))
        self.assertIn(second, history.ancestors(history.heads[0]))

    def test_an_unlisted_own_commit_still_merges_after_a_restart(self) -> None:
        """A reconnecting runtime merges its own not-yet-listed commit without reading it from Drive."""
        self.put("runs/a/x", b"base")
        self.store.backup()
        other = self.runtime("other-vm")
        self.put("runs/a/x", b"mine")
        mine = self.store.backup()
        self.remote.hidden = {n for n in self.remote.commits() if n.endswith(f"--{mine}.json")}
        self.put("runs/b/y", b"theirs", root=other.root)
        other.backup()
        reconnect = storage.SnapshotStore(self.root, self.remote, log=quiet)
        reconnect.restore()
        self.assertEqual((self.root / "runs/b/y").read_bytes(), b"theirs")
        self.assertEqual((self.root / "runs/a/x").read_bytes(), b"mine")
        self.remote.hidden = set()
        self.assertEqual(len(self.heads()), 1)
        fresh = self.restore_copy()
        self.assertEqual((fresh / "runs/a/x").read_bytes(), b"mine")

    def test_a_commit_uploaded_just_before_the_process_died_is_continued_not_forked(self) -> None:
        """A kernel killed between the manifest upload and the state save does not fork its own history."""
        self.put("runs/r/a.pt", b"a")
        self.store.backup()
        self.put("runs/r/b.pt", b"b")
        with patch.object(storage.SnapshotStore, "_save", side_effect=SystemExit("killed")):
            with self.assertRaises(SystemExit):
                self.store.backup()  # uploaded, but the state still names the previous commit
        (self.root / "runs/r/b.pt").unlink()  # the rerun after the restart deletes the newest file
        reconnect = storage.SnapshotStore(self.root, self.remote, log=quiet)
        reconnect.restore()
        reconnect.backup()
        history = storage.History(self.remote.commits())
        self.assertEqual(len(history.heads), 1)
        self.assertEqual(len(history.parents), 3, "a linear chain: no sibling of the lost commit")
        self.assertFalse((self.restore_copy() / "runs/r/b.pt").exists())

    def test_a_manifest_stored_although_its_upload_raised_is_not_forked_in_the_same_session(self) -> None:
        """A lost reply leaves a write-ahead copy; the next backup finishes that commit instead of forking."""
        self.put("eval/keep/score.json", b'{"v":1}')
        self.store.backup()
        self.put("eval/draft/score.json", b'{"v":"draft"}')
        upload = self.remote.upload

        def stored_then_lost(local: Path, relative: str) -> str:
            ref = upload(local, relative)
            if relative.startswith("commits/"):
                raise storage.StorageError("reply lost after Drive stored the file")
            return ref

        with patch.object(self.remote, "upload", side_effect=stored_then_lost), self.assertRaises(storage.StorageError):
            self.store.backup()
        shutil.rmtree(self.root / "eval/draft")  # the user removes the draft before the next command
        lines: list[str] = []
        self.store.log = lines.append
        self.store.backup()
        history = storage.History(self.remote.commits())
        self.assertEqual(len(history.heads), 1)
        self.assertEqual(len(history.parents), 3, "a linear chain")
        self.assertTrue(any("finished publishing" in line for line in lines), lines)
        self.assertFalse((self.restore_copy() / "eval/draft").exists())

    def test_finishing_an_interrupted_commit_never_deletes_files_this_vm_never_had(self) -> None:
        """Files a skipping runtime only carried forward stay in the backup when RESTORE_SKIP changes meanwhile."""
        self.put("runs/r1/best.pt", b"weights")
        self.store.backup()
        mono = self.runtime("mono-vm", skip=["runs"])
        self.put("mono_baselines/m1/score.json", b'{"acc":1}', root=mono.root)
        with patch.object(storage.SnapshotStore, "_save", side_effect=SystemExit("killed")), self.assertRaises(SystemExit):
            mono.backup()
        rerun = storage.SnapshotStore(mono.root, self.remote, log=quiet, skip=[])  # RESTORE_SKIP = [] this time
        rerun.restore()
        rerun.backup(force=True)
        self.assertEqual((self.restore_copy() / "runs/r1/best.pt").read_bytes(), b"weights")
        self.assertEqual((mono.root / "runs/r1/best.pt").read_bytes(), b"weights")

    def test_commits_whose_parents_are_not_listed_yet_wait(self) -> None:
        """A reader that sees a commit before its parent uses the older complete history."""
        self.put("a.txt", b"1")
        first = self.store.backup()
        self.put("a.txt", b"2")
        second = self.store.backup()
        self.put("a.txt", b"3")
        self.store.backup()
        self.remote.hidden = {name for name in self.remote.commits() if name.endswith(f"--{second}.json")}
        history = storage.History(self.remote.commits())
        self.assertEqual(history.heads, [first])
        self.assertEqual(len(history.pending), 1)
        self.assertEqual((self.restore_copy() / "a.txt").read_bytes(), b"1")

    def test_a_workspace_whose_own_backup_vanished_is_not_rolled_back(self) -> None:
        """If this workspace's last commit is gone from Drive, syncing stops instead of reverting local files."""
        self.put("runs/r/best.pt", b"only here")
        head = self.store.backup()
        self.store.sync()
        other = self.runtime("other-vm")
        self.put("runs/o/x", b"other", root=other.root)
        other.backup()
        for name in [n for n in self.remote.commits() if n.endswith(f"--{head}.json")]:
            del self.remote.objects["commits/" + name]
        with self.assertRaisesRegex(storage.StorageError, "missing from the Drive history"):
            self.store.sync()
        self.assertEqual((self.root / "runs/r/best.pt").read_bytes(), b"only here")

    def test_history_rejects_malformed_names_and_tolerates_forks(self) -> None:
        """Duplicate ids and unsorted parents are integrity errors; forks and cycles are not crashes."""
        a, b, c = "a" * 32, "b" * 32, "c" * 32
        history = storage.History([f"root--{a}.json", f"{a}--{b}.json", f"{a}--{c}.json"])
        self.assertEqual(history.heads, [b, c])
        self.assertEqual(history.newest(history.ancestors(b) & history.ancestors(c)), [a])
        merged = storage.History([f"root--{a}.json", f"{a}--{b}.json", f"{a}--{c}.json", f"{b}+{c}--{'d' * 32}.json"])
        self.assertEqual(merged.heads, ["d" * 32])
        self.assertEqual(storage.History([f"{b}--{c}.json", f"{c}--{b}.json"]).heads, [])
        for names in ([f"root--{a}.json", f"{b}--{a}.json"], [f"{c}+{b}--{a}.json"], ["nonsense.json"]):
            with self.subTest(names=names), self.assertRaises(storage.StorageError):
                storage.History(names)

    def test_every_reader_merges_heads_identically(self) -> None:
        """The merged view depends only on the commit graph, not on listing order."""
        a, b, c, d = ("a" * 32, "b" * 32, "c" * 32, "d" * 32)
        entry = lambda digest, mtime=1: {"type": "file", "sha256": digest * 64, "size": 1, "mtime_ns": mtime}
        files = {a: {"runs/a/x": entry("0"), "runs/b/y": entry("0")},
                 b: {"runs/a/x": entry("1", 5), "runs/b/y": entry("0")},
                 c: {"runs/a/x": entry("2", 9), "runs/b/y": entry("2")},
                 d: {"runs/a/x": entry("0"), "runs/b/y": entry("0"), "runs/c/z": entry("3")}}
        names = [f"root--{a}.json", f"{a}--{b}.json", f"{a}--{c}.json", f"{a}--{d}.json"]
        created = {a: 0, b: 1, c: 2, d: 3}
        results = [storage.merge_history(storage.History(order), sorted([b, c, d]) if i % 2 else [d, c, b],
                                         files.__getitem__, created.__getitem__, lambda _: b"")
                   for i, order in enumerate((names, names[::-1], names[2:] + names[:2]))]
        self.assertTrue(all(result[0] == results[0][0] and result[2] == results[0][2] for result in results))
        self.assertEqual(results[0][0]["runs/a/x"]["sha256"], "2" * 64)
        self.assertEqual(results[0][0]["runs/b/y"]["sha256"], "2" * 64)
        self.assertEqual(results[0][0]["runs/c/z"]["sha256"], "3" * 64)
        self.assertEqual(results[0][2], [("runs/a", (c,), False, [((b,), False)])])

    def test_a_virtual_base_leaves_its_conflicts_to_the_real_versions(self) -> None:
        """A folder two runtimes settled differently is decided again from the real versions (git's recursive rule)."""
        entry = lambda digest, mtime: {"type": "file", "sha256": digest * 64, "size": 1, "mtime_ns": mtime}
        r, a1, b2, a2, b3, ma, mb = ("0" * 32, "1" * 32, "2" * 32, "3" * 32, "4" * 32, "a" * 32, "b" * 32)
        group = "mono_baselines/vibevoice/score.json"
        # A wrote a1 then a2; B wrote b2 last. A's merge kept b2. B then deleted the folder and, not seeing A's
        # newer commits yet, merged only A's a1 with its deletion, so it holds a1.
        files = {r: {}, a1: {group: entry("1", 1)}, a2: {group: entry("2", 2)}, b2: {group: entry("3", 3)}, b3: {}}
        files[ma] = {group: entry("3", 3)}
        files[mb] = {group: entry("1", 1)}
        join = lambda *ids: "+".join(sorted(ids))
        names = [f"root--{r}.json", f"{r}--{a1}.json", f"{r}--{b2}.json", f"{a1}--{a2}.json", f"{b2}--{b3}.json",
                 f"{join(a2, b2)}--{ma}.json", f"{join(a1, b3)}--{mb}.json"]
        created = {r: 0, a1: 1, b2: 2, a2: 3, b3: 4, ma: 5, mb: 6}
        history = storage.History(names)
        merged, _, conflicts, _ = storage.merge_history(history, history.heads, files.__getitem__,
                                                     created.__getitem__, lambda _: b"")
        self.assertEqual(merged[group]["sha256"], "3" * 64, "a1, superseded by its own writer, never comes back")
        self.assertEqual([c[0] for c in conflicts], ["mono_baselines/vibevoice"])

    def test_a_changed_folder_is_always_stamped_newer_than_the_version_it_replaced(self) -> None:
        """Deleting a folder's newest file still makes a newer version, so a conflict never picks its ancestor."""
        entry = lambda mtime: {"type": "file", "sha256": "1" * 64, "size": 1, "mtime_ns": mtime}
        before = {"runs/r/a.pt": entry(5), "runs/r/b.pt": entry(9)}
        first = storage._restamp({}, storage.Stamps({}, {}), before)
        self.assertEqual(first, storage.Stamps({"runs/r": 9}, {}))
        after = storage._restamp(before, first, {"runs/r/a.pt": entry(5)})
        self.assertEqual(after, storage.Stamps({"runs/r": 10}, {}))
        self.assertEqual(storage._restamp(before, first, dict(before)), first)
        gone = storage._restamp(before, first, {})
        self.assertEqual(gone, storage.Stamps({}, {"runs/r": 9}), "a deletion remembers the version it deleted")
        self.assertEqual(storage._restamp({}, gone, {"runs/r/a.pt": entry(1)}).present, {"runs/r": 10})
        self.put("runs/r/a.pt", b"a")
        self.put("runs/r/b.pt", b"b")
        self.store.backup()
        stamp = self.store.stamps.present["runs/r"]
        (self.root / "runs/r/b.pt").unlink()
        os.utime(self.root / "runs/r/a.pt", ns=(1, 1))
        self.store.backup()
        self.assertGreater(self.store.stamps.present["runs/r"], stamp)

    def test_criss_cross_merges_use_a_virtual_base_whatever_the_commit_ids(self) -> None:
        """Two runtimes that merged the same heads still keep each other's later work (git's recursive merge)."""
        entry = lambda digest, mtime: {"type": "file", "sha256": digest * 64, "size": 1, "mtime_ns": mtime}
        for first, second in (("1", "2"), ("2", "1")):
            with self.subTest(order=(first, second)):
                v0, v1, v2, m1, m2, w1, w2 = ("0" * 32, first * 32, second * 32, "a" * 32, "b" * 32, "c" * 32, "d" * 32)
                files = {v0: {"runs/carry/best.pt": entry("0", 0)},
                         v1: {"runs/carry/best.pt": entry("1", 1)},
                         v2: {"runs/carry/best.pt": entry("0", 0), "mono_baselines/vibevoice/score.json": entry("2", 2)}}
                files[m1] = files[m2] = {**files[v1], **files[v2], "runs/carry/best.pt": entry("1", 1)}
                files[w1] = {**files[m1], "runs/carry/best.pt": entry("3", 3)}  # training goes on in runtime A
                files[w2] = {**files[m2], "mono_baselines/vibevoice/score.json": entry("4", 4)}  # mono goes on in B
                parents = {v1: v0, v2: v0, w1: m1, w2: m2}
                names = [f"root--{v0}.json", *(f"{parents[c]}--{c}.json" for c in (v1, v2, w1, w2)),
                         f"{'+'.join(sorted((v1, v2)))}--{m1}.json", f"{'+'.join(sorted((v1, v2)))}--{m2}.json"]
                created = {v0: 0, v1: 1, v2: 2, m1: 3, m2: 4, w1: 5, w2: 6}
                history = storage.History(names)
                self.assertEqual(history.heads, sorted([w1, w2]))
                merged, _, conflicts, _ = storage.merge_history(history, history.heads, files.__getitem__,
                                                             created.__getitem__, lambda _: b"")
                self.assertEqual(merged["runs/carry/best.pt"]["sha256"], "3" * 64)
                self.assertEqual(merged["mono_baselines/vibevoice/score.json"]["sha256"], "4" * 64)
                self.assertEqual(conflicts, [])

    def test_three_runtimes_never_revert_a_run_only_one_of_them_trained(self) -> None:
        """A notebook that skips runs/ and only carries an older copy forward cannot undo a finished run."""
        self.put("runs/carry/best.pt", b"step 0")
        self.store.backup()
        mono = self.runtime("mono-vm", skip=["runs"])
        track = self.runtime("track-vm", skip=["runs"])
        self.put("mono_baselines/nemotron/score.json", b'{"acc":1}', root=mono.root)
        mono.backup()
        self.put("runs/carry/best.pt", b"step 1000")
        self.store.backup()
        mono.sync()  # mono's merge carries the half-trained run
        self.put("mono_baselines/vibevoice/score.json", b'{"acc":2}', root=track.root)
        track.backup()  # a long VibeVoice job: periodic commits, no sync
        self.put("runs/carry/best.pt", b"final")
        self.put("runs/carry/summary.json", b'{"done":1}')
        self.store.backup()
        time.sleep(0.01)
        self.put("mono_baselines/nemotron/score.json", b'{"acc":3}', root=mono.root)
        mono.backup()
        self.assertEqual(len(self.heads()), 3)
        lines: list[str] = []
        mono.log = lines.append
        mono.sync()
        self.assertFalse([line for line in lines if "changed in two runtimes" in line], lines)
        self.store.sync()
        self.assertEqual((self.root / "runs/carry/best.pt").read_bytes(), b"final")
        self.assertTrue((self.root / "runs/carry/summary.json").exists())
        self.assertEqual((self.root / "mono_baselines/vibevoice/score.json").read_bytes(), b'{"acc":2}')
        fresh = self.restore_copy("fresh-vm")
        self.assertEqual((fresh / "runs/carry/best.pt").read_bytes(), b"final")
        self.assertEqual((fresh / "mono_baselines/nemotron/score.json").read_bytes(), b'{"acc":3}')

    def test_simultaneous_merges_of_the_same_heads_keep_later_work(self) -> None:
        """Two command-start syncs that cannot see each other's merge (criss-cross) lose nothing afterwards."""
        self.put("runs/carry/best.pt", b"v0")
        self.store.backup()
        other = self.runtime("other-vm", skip=["runs"])
        self.put("runs/carry/best.pt", b"v1")
        self.store.backup()
        self.put("mono_baselines/vibevoice/score.json", b'{"v":"m1"}', root=other.root)
        other.backup()
        self.store.sync()
        first_merge = set(self.remote.commits()) - set(self.remote.hidden)
        self.remote.hidden = {name for name in first_merge if name.count("+")}
        other.sync()
        self.remote.hidden = set()
        self.put("runs/carry/best.pt", b"v2")
        self.store.backup()
        time.sleep(0.01)
        self.put("mono_baselines/vibevoice/score.json", b'{"v":"m2"}', root=other.root)
        other.backup()
        self.assertEqual(len(self.heads()), 2)
        lines: list[str] = []
        self.store.log = lines.append
        self.store.sync()
        self.assertEqual((self.root / "runs/carry/best.pt").read_bytes(), b"v2")
        self.assertEqual((self.root / "mono_baselines/vibevoice/score.json").read_bytes(), b'{"v":"m2"}')
        self.assertFalse([line for line in lines if "changed in two runtimes" in line], lines)

    def test_a_conflict_keeps_the_most_recently_written_version_not_the_latest_commit(self) -> None:
        """A runtime that keeps committing other work does not win a folder it changed earlier."""
        self.put("eval/x/score.json", b'{"v":"base"}')
        self.store.backup()
        other = self.runtime("other-vm")
        os.utime(self.put("eval/x/score.json", b'{"v":"early"}'), ns=(10**18, 10**18))
        self.store.backup()
        os.utime(self.put("eval/x/score.json", b'{"v":"late"}', root=other.root), ns=(2 * 10**18, 2 * 10**18))
        late = other.backup()
        time.sleep(0.01)
        self.put("runs/r/metrics.jsonl", b'{"step":1}\n')
        self.store.backup()  # the newest commit, but its eval/x was written earlier
        lines: list[str] = []
        self.store.log = lines.append
        self.store.sync()
        self.assertEqual(json.loads((self.root / "eval/x/score.json").read_text())["v"], "late")
        self.assertTrue(any("eval/x changed in two runtimes" in line and late[:8] in line for line in lines), lines)

    def test_conflict_lines_name_the_commits_that_hold_each_version(self) -> None:
        """The ids in a conflict line export exactly the versions it describes, also with three runtimes."""
        self.put("eval/x/score.json", b'{"v":1}')
        self.put("eval/y/score.json", b'{"v":1}')
        self.store.backup()
        deleter, other, changer = self.runtime("deleter"), self.runtime("other"), self.runtime("changer")
        shutil.rmtree(deleter.root / "eval/x")
        deleted = deleter.backup()
        self.put("eval/y/score.json", b'{"v":2}', root=other.root)
        other.backup()
        self.put("eval/x/score.json", b'{"v":3}', root=changer.root)
        changed = changer.backup()
        lines: list[str] = []
        storage.SnapshotStore(self.directory / "fresh", self.remote, log=lines.append).restore()
        line = next(line for line in lines if "eval/x was deleted" in line)
        self.assertIn(f"kept the changed version (commit {changed[:8]})", line)
        self.assertIn(f"the deletion is commit {deleted[:8]}", line)

    def test_a_deletion_wins_over_the_very_version_it_deleted_after_a_criss_cross(self) -> None:
        """A virtual base leaves the folder open; the deleter had seen that version, so it stays deleted."""
        entry = lambda digest, mtime: {"type": "file", "sha256": digest * 64, "size": 1, "mtime_ns": mtime}
        r, a, b, ma, mb, d = ("0" * 32, "1" * 32, "2" * 32, "a" * 32, "b" * 32, "d" * 32)
        group = "eval/x/score.json"
        files = {r: {}, a: {group: entry("1", 1)}, b: {"eval/y/score.json": entry("2", 2)}}
        files[ma] = files[mb] = {**files[a], **files[b]}
        files[d] = dict(files[b])  # the second runtime deleted eval/x after merging it
        stamps = {r: storage.Stamps({}, {}), a: storage.Stamps({"eval/x": 1}, {}), b: storage.Stamps({"eval/y": 2}, {}),
                  ma: storage.Stamps({"eval/x": 1, "eval/y": 2}, {}), mb: storage.Stamps({"eval/x": 1, "eval/y": 2}, {}),
                  d: storage.Stamps({"eval/y": 2}, {"eval/x": 1})}
        join = lambda *ids: "+".join(sorted(ids))
        names = [f"root--{r}.json", f"{r}--{a}.json", f"{r}--{b}.json", f"{join(a, b)}--{ma}.json",
                 f"{join(a, b)}--{mb}.json", f"{mb}--{d}.json"]
        created = {r: 0, a: 1, b: 2, ma: 3, mb: 4, d: 5}
        history = storage.History(names)
        self.assertEqual(history.heads, sorted([ma, d]))
        merged, _, conflicts, _ = storage.merge_history(history, history.heads, files.__getitem__, created.__getitem__,
                                                        lambda _: b"", stamps.__getitem__)
        self.assertNotIn(group, merged)
        stamps[ma] = storage.Stamps({"eval/x": 7, "eval/y": 2}, {})  # changed again after the version d deleted
        files[ma] = {**files[ma], group: entry("7", 7)}
        merged, _, _, _ = storage.merge_history(history, history.heads, files.__getitem__, created.__getitem__,
                                                lambda _: b"", stamps.__getitem__)
        self.assertEqual(merged[group]["sha256"], "7" * 64, "a change newer than the deleted version is kept")

    def test_a_folder_deleted_in_one_runtime_and_changed_in_another_keeps_the_change(self) -> None:
        """A deletion never silently discards results another runtime produced meanwhile."""
        self.put("eval/x/score.json", b'{"v":1}')
        self.store.backup()
        other = self.runtime("other-vm")
        shutil.rmtree(self.root / "eval/x")
        self.store.backup()
        self.put("eval/x/score.json", b'{"v":2}', root=other.root)
        other.backup()
        lines: list[str] = []
        self.store.log = lines.append
        self.store.sync()
        self.assertEqual(json.loads((self.root / "eval/x/score.json").read_text())["v"], 2)
        self.assertTrue(any("eval/x was deleted in one runtime" in line for line in lines), lines)

    @needs_modern_asr
    def test_a_cache_kept_whole_once_is_not_mixed_in_by_a_later_merge(self) -> None:
        """After one merge kept one runtime's cache, the other runtime's later rows are not unioned under it."""
        self.put("seed.txt", b"seed")
        self.store.backup()
        mono = self.runtime("mono-vm")
        self.qwen_cache(self.root, {"t2_0000": b"zero"}, runtime=b'{"torch":"2.5.1"}\n')
        self.store.backup()
        time.sleep(0.01)
        self.qwen_cache(mono.root, {"t2_0001": b"one"}, runtime=b'{"torch":"2.6.0"}\n')
        mono.backup()
        mono.sync()  # keeps mono's whole cache and logs the conflict
        time.sleep(0.01)
        self.qwen_cache(self.root, {"t2_0002": b"two"}, runtime=b'{"torch":"2.5.1"}\n')
        self.store.backup(include_static=False)  # the train job goes on under its own contract
        lines: list[str] = []
        mono.log = lines.append
        mono.sync()
        self.assertTrue(any("asr_cache changed in two runtimes" in line for line in lines), lines)
        fresh = self.restore_copy("fresh-vm")
        runtime = json.loads((fresh / "asr_cache/qwen-runtime.json").read_text())["torch"]
        cache = transcripts.SegmentCache(modern_asr.cache_path(fresh / "asr_cache", modern_asr.QWEN_BACKEND),
                                         modern_asr.QWEN_BACKEND)
        made = {"2.5.1": {"t2_0000", "t2_0002"}, "2.6.0": {"t2_0001"}}[runtime]
        self.assertEqual(set(cache.text.values()), made, "every row was made under the recorded runtime")

    def qwen_cache(self, root: Path, sessions: dict[str, bytes], runtime: bytes = b'{"qwen":1}\n') -> list[dict[str, Any]]:
        """Write a Qwen transcript cache, its audio provenance and runtime contract as modern_asr does."""
        backend = modern_asr.QWEN_BACKEND
        cache = modern_asr.cache_path(root / "asr_cache", backend)
        cache.parent.mkdir(parents=True, exist_ok=True)
        inputs = json.loads(cache.with_suffix(".inputs.json").read_text()) if cache.with_suffix(".inputs.json").exists() else {}
        requests = []
        for sid, audio in sessions.items():
            key = transcripts.segment_key("eval-t2", sid, 0, 0.0, 1.0)
            requests.append({"key": key, "session_id": sid, "channel": 0, "start_sec": 0.0, "end_sec": 1.0})
            with cache.open("a") as handle:
                handle.write(json.dumps({"key": key, "backend": backend, "text": sid}) + "\n")
            inputs[f"eval-t2/{sid}"] = hashlib.sha256(audio).hexdigest()
        modern_asr.atomic_json(cache.with_suffix(".inputs.json"), inputs)
        (root / "asr_cache/qwen-runtime.json").write_bytes(runtime)
        return requests

    def fill(self, root: Path, requests: list[dict[str, Any]], audio: dict[str, bytes]) -> int:
        """Run modern_asr's real provenance check on a cache whose transcripts are all present (no model load)."""
        data = self.directory / "audio-data"
        for sid, content in audio.items():
            self.put(f"eval-t2/audio/{sid}.wav", content, root=data)
        listing = self.directory / "requests.jsonl"
        listing.write_text("".join(json.dumps(row) + "\n" for row in requests))
        return modern_asr.fill_requests([listing], data, root / "asr_cache", "cpu")

    @needs_modern_asr
    def test_a_transcript_cache_folder_merges_rows_and_audio_provenance_together(self) -> None:
        """Qwen transcripts made in two runtimes stay usable: rows and their provenance are combined."""
        audio = {"t2_0000": b"zero", "t2_0001": b"one", "t2_0002": b"two"}
        self.qwen_cache(self.root, {"t2_0000": audio["t2_0000"]})
        self.store.backup()
        mono = self.runtime("mono-vm")
        train_requests = self.qwen_cache(self.root, {"t2_0001": audio["t2_0001"]})
        self.store.backup()
        mono_requests = self.qwen_cache(mono.root, {"t2_0002": audio["t2_0002"]})
        mono.backup()
        self.store.sync()
        everything = [*self.qwen_cache(self.directory / "scratch", {"t2_0000": b""}), *train_requests, *mono_requests]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.fill(self.root, everything, audio), 0)
        cache = transcripts.SegmentCache(modern_asr.cache_path(self.root / "asr_cache", modern_asr.QWEN_BACKEND),
                                         modern_asr.QWEN_BACKEND)
        self.assertEqual(sorted(cache.text.values()), ["t2_0000", "t2_0001", "t2_0002"])

    @needs_modern_asr
    def test_transcript_caches_from_different_runtimes_are_never_mixed(self) -> None:
        """A different runtime contract or audio hash keeps one runtime's whole cache folder instead of a mix."""
        self.qwen_cache(self.root, {"t2_0000": b"zero"})
        self.store.backup()
        mono = self.runtime("mono-vm")
        self.qwen_cache(self.root, {"t2_0001": b"one"}, runtime=b'{"qwen":"train image"}\n')
        self.store.backup()
        time.sleep(0.01)
        self.qwen_cache(mono.root, {"t2_0002": b"two"}, runtime=b'{"qwen":"mono image"}\n')
        mono.backup()
        lines: list[str] = []
        self.store.log = lines.append
        self.store.sync()
        cache = modern_asr.cache_path(self.root / "asr_cache", modern_asr.QWEN_BACKEND)
        self.assertEqual(sorted(transcripts.SegmentCache(cache, modern_asr.QWEN_BACKEND).text.values()), ["t2_0000", "t2_0002"])
        self.assertEqual(sorted(json.loads(cache.with_suffix(".inputs.json").read_text())), ["eval-t2/t2_0000", "eval-t2/t2_0002"])
        self.assertEqual((self.root / "asr_cache/qwen-runtime.json").read_bytes(), b'{"qwen":"mono image"}\n')
        self.assertTrue(any("asr_cache changed in two runtimes" in line for line in lines), lines)

    def test_runtimes_that_start_on_an_empty_folder_at_once_are_merged(self) -> None:
        """Two first connections create two root commits; both survive."""
        first = self.runtime("first")
        second = self.runtime("second")
        self.put("runs/a/x", b"a", root=first.root)
        self.put("runs/b/y", b"b", root=second.root)
        first.backup(force=True)
        second.backup(force=True)
        self.assertEqual(len(self.heads()), 2)
        fresh = self.restore_copy("fresh")
        self.assertTrue((fresh / "runs/a/x").exists() and (fresh / "runs/b/y").exists())

    def test_command_start_brings_in_other_runtime_results(self) -> None:
        """A notebook command sees results another runtime backed up while this one was idle."""
        session = storage.StorageSession(self.store, interval_sec=60)
        try:
            other = self.runtime("other-vm")
            self.put("eval/voice_only-qwen3-1.7b/eval-t2/embeddings.npz", b"embeddings", root=other.root)
            other.backup()
            check = "import pathlib, sys; sys.exit(0 if pathlib.Path(sys.argv[1]).exists() else 3)"
            session.run([sys.executable, "-c", check, str(self.root / "eval/voice_only-qwen3-1.7b/eval-t2/embeddings.npz")])
        finally:
            session.close()

    def test_pull_removes_files_deleted_elsewhere_and_keeps_unsaved_local_edits(self) -> None:
        """Synchronization deletes only clean copies and never overwrites a file it could not back up."""
        self.put("runs/r/old.txt", b"old")
        self.put("runs/r/config.json", b'{"v":1}')
        self.store.backup()
        other = self.runtime("other-vm")
        (other.root / "runs/r/old.txt").unlink()
        self.put("runs/r/config.json", b'{"v":2}', root=other.root)
        other.backup()
        self.put("runs/r/config.json", b'{"v":')  # being written: cannot be backed up yet
        self.store.sync()
        self.assertFalse((self.root / "runs/r/old.txt").exists())
        self.assertEqual((self.root / "runs/r/config.json").read_bytes(), b'{"v":')

    def test_restore_skip_keeps_large_files_in_the_backup_without_downloading_them(self) -> None:
        """A training VM does not download another notebook's model weights, and never deletes them."""
        self.put("models/asr/models--microsoft--VibeVoice-ASR/blobs/w", b"17 GB of weights")
        self.put("runs/train/best.pt", b"weights")
        self.store.backup()
        skip = storage.model_cache_dirs(["microsoft/VibeVoice-ASR"])
        self.assertIn("models/asr/models--microsoft--VibeVoice-ASR", skip)
        train = self.runtime("train-vm", skip=skip)
        self.assertFalse((train.root / "models/asr/models--microsoft--VibeVoice-ASR").exists())
        self.put("models/asr/models--microsoft--VibeVoice-ASR/blobs/w2", b"more weights")
        self.store.backup()
        self.put("runs/train/best.pt", b"newer", root=train.root)
        train.sync()
        self.assertFalse((train.root / "models/asr/models--microsoft--VibeVoice-ASR").exists())
        fresh = self.restore_copy("fresh-vm")
        self.assertTrue((fresh / "models/asr/models--microsoft--VibeVoice-ASR/blobs/w").exists())
        self.assertTrue((fresh / "models/asr/models--microsoft--VibeVoice-ASR/blobs/w2").exists())
        self.assertEqual((fresh / "runs/train/best.pt").read_bytes(), b"newer")

    def test_insufficient_disk_space_fails_before_downloading(self) -> None:
        """A restore that cannot fit reports the disk, not a corrupted backup."""
        self.put("models/big.bin", b"x" * 1000)
        self.store.backup()
        usage = shutil.disk_usage(self.directory)._replace(free=100)
        with patch.object(storage.shutil, "disk_usage", return_value=usage):
            with self.assertRaisesRegex(storage.StorageError, "RESTORE_SKIP"):
                self.restore_copy()

    def test_an_interrupted_restore_resumes_after_other_runtimes_back_up(self) -> None:
        """Rerunning the connection cell keeps what was downloaded although the history moved meanwhile."""
        for name in "abc":
            self.put(f"models/{name}.bin", name.encode() * 100)
        self.store.backup()
        fresh = storage.SnapshotStore(self.directory / "fresh", self.remote, log=quiet)
        download = self.remote.download
        calls: list[str] = []

        def flaky(relative: str, local: Path, ref: str | None = None) -> None:
            calls.append(relative)
            if len(calls) == 2:
                raise storage.StorageError("connection reset")
            download(relative, local, ref)

        with patch.object(self.remote, "download", side_effect=flaky):
            with self.assertRaises(storage.StorageError):
                fresh.restore()
        first = calls[0]
        self.put("runs/r/metrics.jsonl", b'{"step":1}\n')
        self.store.backup()  # another runtime's periodic backup before the rerun
        self.remote.downloads.clear()
        storage.SnapshotStore(self.directory / "fresh", self.remote, log=quiet).restore()
        self.assertNotIn(first, [relative for relative, _ in self.remote.downloads])
        self.assertEqual((self.directory / "fresh/models/a.bin").read_bytes(), b"a" * 100)
        self.assertEqual(sorted(p.name for p in storage._state_dir(self.directory / "fresh").iterdir()
                                if p.name.startswith("restore")), [])

    def test_export_reruns_without_skip_fetch_everything_and_combined_caches_survive(self) -> None:
        """--skip belongs to one export run, and an unmerged combined cache can be exported again."""
        cache = "asr_cache/Qwen.jsonl"
        row = lambda key: (json.dumps({"key": key, "backend": "qwen", "text": key}) + "\n").encode()
        self.put(cache, row("a"))
        self.put("models/w.bin", b"weights")
        self.store.backup()
        other = self.runtime("other-vm")
        self.put(cache, row("a") + row("b"))
        self.store.backup()
        self.put(cache, row("a") + row("c"), root=other.root)
        other.backup()
        auth = self.directory / "auth.json"
        auth.write_text(json.dumps(auth_package()))
        out = self.directory / "export"
        with patch.object(storage.GoogleDriveTransport, "from_auth", return_value=self.remote):
            storage.main(["export", "--auth", str(auth), "--out", str(out), "--skip", "models"])
            self.assertFalse((out / "models/w.bin").exists())
            (out / cache).unlink()
            storage.main(["export", "--auth", str(auth), "--out", str(out)])
        self.assertEqual((out / "models/w.bin").read_bytes(), b"weights")
        self.assertEqual(transcripts.SegmentCache(out / cache, "qwen").text, {"a": "a", "b": "b", "c": "c"})
        self.assertEqual(len(self.heads()), 2, "an export never publishes")

    def test_temporary_copies_of_killed_operations_are_swept(self) -> None:
        """Partial downloads and copies left by a process killed before its cleanup do not stay on disk."""
        self.put("runs/r/x.txt", b"x")
        self.store.backup()
        state = self.store.state_dir
        (state / "pull-0").write_bytes(b"partial checkpoint")
        (state / "object-1").mkdir()
        (state / "object-1/data").write_bytes(b"copy")
        (state / "commit-2.json").write_text("{}")
        (state / "import/runs/a").mkdir(parents=True)
        storage.SnapshotStore(self.root, self.remote, log=quiet).restore()
        self.assertEqual(sorted(p.name for p in state.iterdir()), ["state.json"])

    def test_an_export_rerun_reads_the_same_stamps_as_every_other_reader(self) -> None:
        """The export's saved state keeps the stamps of the head it mirrors."""
        self.put("eval/foo/x.json", b'{"v":1}')
        self.store.backup()
        out = self.directory / "export"
        mirror = storage.SnapshotStore(out, self.remote, log=quiet)
        mirror.export()
        self.assertEqual(mirror.stamps, self.store.stamps)
        self.assertEqual(json.loads((storage._state_dir(out) / "state.json").read_text())["stamps"], self.store.stamps.present)

    def test_an_export_folder_made_by_the_previous_version_is_updated_in_place(self) -> None:
        """The 9123e53 export (a restore without a mode) is accepted by the documented rerun."""
        self.put("eval/x/score.json", b'{"acc":1}')
        head = self.store.backup()
        out = self.directory / "export"
        storage.SnapshotStore(out, self.remote, log=quiet).export()
        state = storage._state_dir(out) / "state.json"
        current = json.loads(state.read_text())
        state.write_text(json.dumps({"remote": current["remote"], "writer": current["writer"], "base": head,
                                     "files": current["files"], "signatures": {}}))
        self.put("eval/x/score.json", b'{"acc":2}')
        self.store.backup()
        storage.SnapshotStore(out, self.remote, log=quiet).export()
        self.assertEqual(json.loads((out / "eval/x/score.json").read_text())["acc"], 2)

    def test_export_can_be_rerun_into_the_same_folder(self) -> None:
        """A results mirror on the Mac is refreshed in place and local notes survive."""
        self.put("eval/x/score.json", b'{"acc":1}')
        self.put("runs/r/old.txt", b"old")
        self.store.backup()
        out = self.directory / "export"
        storage.SnapshotStore(out, self.remote, log=quiet).export()
        self.put("eval/x/score.json", b'{"acc":2}')
        (self.root / "runs/r/old.txt").unlink()
        self.store.backup()
        self.put("notes.md", b"mine", root=out)
        mirror = storage.SnapshotStore(out, self.remote, log=quiet)
        mirror.export()
        self.assertEqual(json.loads((out / "eval/x/score.json").read_text())["acc"], 2)
        self.assertFalse((out / "runs/r/old.txt").exists())
        self.assertEqual((out / "notes.md").read_bytes(), b"mine")
        with self.assertRaisesRegex(storage.StorageError, "export"):
            mirror.backup()
        self.assertEqual(len(self.heads()), 1)

    def test_import_adds_only_result_folders_the_backup_does_not_have(self) -> None:
        """School-Drive results are carried over once without mixing into existing runs."""
        self.put("runs/carry/summary.json", b'{"personal":1}')
        self.store.backup()
        school = self.directory / "school"
        self.put("runs/carry/summary.json", b'{"school":1}', root=school)
        self.put("runs/carry/stale.txt", b"must not be mixed in", root=school)
        self.put("runs/reset/summary.json", b'{"school":2}', root=school)
        self.put("cache/dev-clean.tar.gz", b"archive", root=school)
        self.put("private/drive-auth.json", b"SECRET", root=school)
        self.assertEqual(self.store.import_tree(school), 2)
        self.store.backup()
        fresh = self.restore_copy()
        self.assertEqual(json.loads((fresh / "runs/carry/summary.json").read_text()), {"personal": 1})
        self.assertFalse((fresh / "runs/carry/stale.txt").exists())
        self.assertEqual(json.loads((fresh / "runs/reset/summary.json").read_text()), {"school": 2})
        self.assertTrue((fresh / "cache/dev-clean.tar.gz").exists())
        self.assertFalse(any(b"SECRET" in data for data in self.remote.objects.values()))

    def test_an_interrupted_import_is_completed_by_rerunning_it(self) -> None:
        """A folder is copied whole or not at all, so the documented rerun finishes it."""
        school = self.directory / "school"
        names = ["a.pt", "ckpt/b.pt", "ckpt/c.pt", "notes.txt"]
        for name in names:
            self.put(f"runs/carry/{name}", name.encode(), root=school)
        copy = shutil.copy2
        calls: list[str] = []

        def flaky(source: Any, target: Any) -> Any:
            calls.append(str(source))
            if len(calls) == 3:
                raise OSError("Transport endpoint is not connected")
            return copy(source, target)

        with patch.object(storage.shutil, "copy2", side_effect=flaky), self.assertRaises(OSError):
            self.store.import_tree(school)
        self.assertFalse((self.root / "runs/carry").exists())
        self.store.backup()
        self.assertEqual(self.store.import_tree(school), 4)
        self.store.backup()
        run = self.restore_copy() / "runs/carry"
        self.assertEqual(sorted(p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()), names)

    def test_an_imported_folder_is_not_imported_again_after_it_was_renamed(self) -> None:
        """Rerunning with IMPORT_FROM still set, here or on a fresh VM, does not bring back a migrated folder."""
        school = self.directory / "school"
        self.put("eval/voice_only/eval-t2/full/score.json", b'{"legacy":1}', root=school)
        self.assertEqual(self.store.import_tree(school), 1)
        self.store.backup(force=True)  # connect_personal_drive backs up right after importing
        (self.root / "eval/voice_only").rename(self.root / "eval/voice_only-wav2vec2")
        self.store.backup()
        self.assertEqual(self.store.import_tree(school), 0)
        fresh = self.runtime("fresh-vm")
        self.assertEqual(fresh.import_tree(school), 0)
        self.assertFalse((fresh.root / "eval/voice_only").exists())
        self.assertTrue((fresh.root / "eval/voice_only-wav2vec2/eval-t2/full/score.json").exists())

    @needs_modern_asr
    def test_import_combines_a_transcript_cache_folder_the_backup_already_has(self) -> None:
        """School-Drive Qwen transcripts are not left behind because the backup already has a wav2vec2 cache."""
        row = lambda key, backend="w2v": (json.dumps({"key": key, "backend": backend, "text": key}) + "\n").encode()
        self.put("asr_cache/w2v.jsonl", row("a"))
        self.store.backup()
        school = self.directory / "school"
        self.put("asr_cache/w2v.jsonl", row("b"), root=school)
        requests = self.qwen_cache(school, {"t2_0000": b"zero"})
        self.assertEqual(self.store.import_tree(school), 4)
        self.store.backup()
        fresh = self.restore_copy()
        self.assertEqual(transcripts.SegmentCache(fresh / "asr_cache/w2v.jsonl", "w2v").text, {"a": "a", "b": "b"})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.fill(fresh, requests, {"t2_0000": b"zero"}), 0)
        self.assertEqual(self.store.import_tree(school), 0, "a folder is imported at most once")

    @needs_modern_asr
    def test_import_never_mixes_transcripts_made_with_another_runtime(self) -> None:
        """A cache recorded with a different Qwen runtime contract is left out instead of being mixed in."""
        self.qwen_cache(self.root, {"t2_0000": b"zero"}, runtime=b'{"qwen":"personal"}\n')
        self.store.backup()
        school = self.directory / "school"
        self.qwen_cache(school, {"t2_0001": b"one"}, runtime=b'{"qwen":"school"}\n')
        lines: list[str] = []
        self.store.log = lines.append
        self.assertEqual(self.store.import_tree(school), 0)
        self.assertTrue(any("kept this backup's asr_cache" in line for line in lines), lines)
        self.assertEqual((self.root / "asr_cache/qwen-runtime.json").read_bytes(), b'{"qwen":"personal"}\n')

    def checkpoint_run(self, root: Path, run: str, missing: bool) -> None:
        """A run as Checkpoints.save leaves it, optionally with an index entry whose upload never arrived."""
        entries = []
        for step in (200, 100) if missing else (100,):
            data = f"weights {step}".encode()
            entries.append({"file": f"step_{step:07d}.pt", "step": step, "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest()})
            if not (missing and step == 200):
                self.put(f"runs/{run}/ckpt/step_{step:07d}.pt", data, root=root)
        self.put(f"runs/{run}/ckpt/LATEST.json", json.dumps(entries, indent=1).encode(), root=root)
        self.put(f"runs/{run}/metrics.jsonl", b'{"step":100}\n{"step":', root=root)

    def test_an_imported_run_with_an_upload_the_source_never_received_is_backed_up(self) -> None:
        """Index entries and JSONL tails that readers skip anyway do not keep an imported run out of the backup."""
        school = self.directory / "school"
        self.checkpoint_run(school, "reset", missing=True)
        self.assertEqual(self.store.import_tree(school), 3)
        self.store.backup()
        run = self.restore_copy() / "runs/reset"
        self.assertEqual([e["step"] for e in json.loads((run / "ckpt/LATEST.json").read_text())], [100])
        self.assertEqual((run / "metrics.jsonl").read_bytes(), b'{"step":100}\n')
        self.assertEqual(self.store.imported, {"runs/reset"})

    def test_an_imported_folder_the_backup_holds_back_is_imported_again_later(self) -> None:
        """A folder recorded as imported is always in the backup; one that never got there is retried and named."""
        school = self.directory / "school"
        self.put("eval/x/score.json", b'{"truncated":', root=school)
        self.store.import_tree(school)
        lines: list[str] = []
        self.store.log = lines.append
        self.store.backup(force=True)
        self.store.report_imports()
        self.assertTrue(any("imported eval/x is not in the backup yet" in line for line in lines), lines)
        self.assertNotIn("eval/x", self.store.imported)
        fresh = self.runtime("fresh-vm")
        self.assertEqual(fresh.import_tree(school), 1, "another VM imports it again")

    def test_an_imported_folder_with_one_incomplete_file_is_recorded_and_not_brought_back(self) -> None:
        """The rest of the folder is backed up, the incomplete file is named, and a rename stays a rename."""
        school = self.directory / "school"
        self.put("eval/run/eval-t2/score.json", b'{"acc":1}', root=school)
        self.put("eval/run/eval-t2-static/score.json", b'{"truncated":', root=school)
        self.store.import_tree(school)
        lines: list[str] = []
        self.store.log = lines.append
        self.store.backup(force=True)
        self.assertTrue(any("left out of the backup because incomplete: eval/run/eval-t2-static/score.json" in line
                            for line in lines), lines)
        self.assertIn("eval/run", self.store.imported)
        (self.root / "eval/run").rename(self.root / "eval/run-qwen3-1.7b")
        self.store.backup()
        fresh = self.runtime("fresh-vm")
        self.assertEqual(fresh.import_tree(school), 0)
        self.assertFalse((fresh.root / "eval/run").exists())

    def test_an_import_stopped_while_moving_files_in_is_completed(self) -> None:
        """A folder recorded as being imported but only partly in place is copied again."""
        school = self.directory / "school"
        for name in "abc":
            self.put(f"runs/s2/{name}.txt", name.encode(), root=school)
        (self.root / "runs/s2/ckpt").mkdir(parents=True)  # empty directories left by an earlier run
        self.store.import_pending = {"runs/s2": {f"runs/s2/{n}.txt": hashlib.sha256(n.encode()).hexdigest() for n in "abc"}}
        self.put("runs/s2/a.txt", b"a")  # the process died after moving one file
        self.assertEqual(self.store.import_tree(school), 3)
        self.store.backup()
        run = self.restore_copy() / "runs/s2"
        self.assertEqual(sorted(p.name for p in run.iterdir()), ["a.txt", "b.txt", "c.txt"])

    def test_each_imported_folder_is_recorded_as_soon_as_it_is_in_place(self) -> None:
        """A kernel killed in the middle of an import has already saved the folders it moved in."""
        school = self.directory / "school"
        self.put("eval/a/score.json", b'{"v":1}', root=school)
        self.put("eval/b/score.json", b'{"v":2}', root=school)
        copy = shutil.copy2
        seen: list[dict[str, Any]] = []

        def watch(source: Any, target: Any) -> Any:
            seen.append(json.loads(self.store.state_path.read_text()).get("import_pending", {}))
            return copy(source, target)

        with patch.object(storage.shutil, "copy2", side_effect=watch):
            self.store.import_tree(school)
        self.assertEqual(sorted(seen[1]), ["eval/a"], "saved before the next folder is copied")

    def test_cli_import_rerun_with_skip_still_downloads_nothing(self) -> None:
        """A staging workspace keeps its marker when the documented rerun passes --skip."""
        self.put("cache/train-clean-100.tar.gz", b"archive")
        self.put("runs/voice_only/summary.json", b'{"v":1}')
        self.store.backup()
        school = self.directory / "school"
        self.put("runs/reset/summary.json", b'{"school":2}', root=school)
        self.put("models/w.bin", b"weights", root=school)
        auth = self.directory / "auth.json"
        auth.write_text(json.dumps(auth_package()))
        staging = self.directory / "staging"
        with patch.object(storage.GoogleDriveTransport, "from_auth", return_value=self.remote):
            for _ in range(2):
                self.remote.downloads.clear()
                storage.main(["import", "--auth", str(auth), "--local", str(staging), "--from", str(school), "--skip", "models"])
                self.assertEqual(self.remote.downloads, [])
        self.assertFalse((staging / "cache/train-clean-100.tar.gz").exists())
        self.assertFalse((self.restore_copy() / "models/w.bin").exists())

    def test_cli_import_does_not_download_the_existing_backup(self) -> None:
        """The command-line import stages only the new files."""
        self.put("models/huge.bin", b"weights")
        self.store.backup()
        school = self.directory / "school"
        self.put("runs/reset/summary.json", b'{"school":2}', root=school)
        auth = self.directory / "auth.json"
        auth.write_text(json.dumps(auth_package()))
        staging = self.directory / "staging"
        with patch.object(storage.GoogleDriveTransport, "from_auth", return_value=self.remote):
            storage.main(["import", "--auth", str(auth), "--local", str(staging), "--from", str(school)])
        self.assertFalse((staging / "models/huge.bin").exists())
        fresh = self.restore_copy()
        self.assertTrue((fresh / "models/huge.bin").exists() and (fresh / "runs/reset/summary.json").exists())

    def test_cli_refuses_a_workspace_held_by_a_notebook_session(self) -> None:
        """The documented terminal backup cannot run as a second writer on a live notebook store."""
        auth = self.directory / "auth.json"
        auth.write_text(json.dumps(auth_package()))
        session = storage.StorageSession(self.store, interval_sec=60)
        try:
            with patch.object(storage.GoogleDriveTransport, "from_auth", return_value=self.remote):
                with self.assertRaisesRegex(storage.StorageError, "storage.flush"):
                    storage.main(["backup", "--auth", str(auth), "--local", str(self.root)])
        finally:
            session.close()


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



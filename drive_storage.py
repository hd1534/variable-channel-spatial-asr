"""Back up local Colab files to a separately authenticated Google Drive account.

Immutable, content-addressed objects are uploaded before a commit manifest.
An interrupted upload never becomes a restore point. Remote history is kept,
and divergent writers are detected rather than silently overwriting a run.
The training and ASR processes continue to use ordinary local filesystem paths.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import posixpath
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol, Sequence, TextIO

from drive_api import GoogleDriveTransport, StorageError, private_json, validate_auth

SCHEMA: int = 1
COMMIT_NAME: re.Pattern[str] = re.compile(r"(root|[0-9a-f]{32})--([0-9a-f]{32})\.json\Z")
SHA256: re.Pattern[str] = re.compile(r"[0-9a-f]{64}\Z")
SECRET_NAMES: frozenset[str] = frozenset({"rclone.conf", "token.json", "credentials.json", "client_secret.json", "drive-auth.json"})
STATIC_DIRS: frozenset[str] = frozenset({"models", "cache"})
RETRY_DELAYS: tuple[int, ...] = (5, 10, 20, 20)


def _write_json(path: Path, value: Any) -> None:
    """Publish a complete local control file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _hash(path: Path) -> str:
    """Hash a regular file without loading large model weights into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> str:
    """Validate a manifest path before it can affect the local filesystem."""
    if not isinstance(value, str):
        raise StorageError("backup paths must be strings")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")) or "\\" in value:
        raise StorageError(f"unsafe backup path: {value!r}")
    return value


def _signature(path: Path) -> list[int]:
    """Identify a file generation, including in-place edits and atomic replacements."""
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _excluded(relative: str) -> bool:
    """Exclude incomplete downloads, credentials and transport/runtime internals."""
    parts = PurePosixPath(relative).parts
    name = parts[-1]
    return (any(part in (".git", ".venv", "__pycache__", ".ipynb_checkpoints", ".locks") for part in parts)
            or name in SECRET_NAMES or name.startswith((".rclone", ".drive-backup"))
            or name.endswith((".rclone.conf", "drive-auth.json")) or name.startswith("client_secret") or name == ".DS_Store"
            or name.endswith((".part", ".tmp", ".tmp.npy", ".lock", ".incomplete")))


def _object(digest: str) -> str:
    """Return the immutable remote object path for validated content."""
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise StorageError("invalid content digest")
    return f"objects/{digest[:2]}/{digest}"


class BackupTransport(Protocol):
    """Required operations for complete, immutable backup generations."""

    remote: str

    def prepare(self) -> None:
        """Validate access and initialize the dedicated backup namespace."""
        ...

    def commits(self) -> list[str]:
        """List completed backup manifest names."""
        ...

    def manifest(self, name: str) -> dict[str, Any]:
        """Read one published manifest."""
        ...

    def upload(self, local: Path, relative: str) -> None:
        """Return only after uploaded bytes have been verified."""
        ...

    def download(self, relative: str, local: Path) -> None:
        """Stream verified bytes into a staging destination."""
        ...


def _head(names: Sequence[str]) -> tuple[str | None, str | None]:
    """Find a single history head and reject missing parents or concurrent writers."""
    commits: dict[str, str] = {}
    lineage: dict[str, str] = {}
    parents: set[str] = set()
    for name in names:
        match = COMMIT_NAME.fullmatch(name)
        if match is None:
            raise StorageError("invalid commit filename")
        parent, identifier = match.groups()
        if identifier in commits:
            raise StorageError("duplicate commit identifier")
        commits[identifier] = name
        lineage[identifier] = parent
        if parent != "root":
            parents.add(parent)
    if not commits:
        return None, None
    if not parents.issubset(commits):
        raise StorageError("backup history has a missing parent")
    tips = set(commits) - parents
    if len(tips) != 1:
        raise StorageError("concurrent backup histories detected; use a separate remote folder per active runtime")
    identifier = tips.pop()
    visited: set[str] = set()
    cursor = identifier
    while cursor != "root":
        if cursor in visited:
            raise StorageError("backup history contains a cycle")
        visited.add(cursor)
        cursor = lineage[cursor]
    if visited != set(commits):
        raise StorageError("backup history contains disconnected commits")
    return identifier, commits[identifier]


def _validate_manifest(document: dict[str, Any], identifier: str) -> dict[str, dict[str, Any]]:
    """Validate every restore destination and symlink before downloading any files."""
    if (not isinstance(document, dict) or document.get("schema") != SCHEMA
            or document.get("id") != identifier or not isinstance(document.get("files"), dict)):
        raise StorageError("unsupported or invalid backup manifest")
    entries = document["files"]
    for relative, entry in entries.items():
        _relative(relative)
        if not isinstance(entry, dict):
            raise StorageError("invalid backup entry")
        if _excluded(relative):
            raise StorageError("backup contains an excluded/private path")
        if entry.get("type") == "file":
            _object(entry.get("sha256", ""))
            if type(entry.get("size")) is not int or entry["size"] < 0 or type(entry.get("mtime_ns")) is not int:
                raise StorageError("invalid file metadata")
        elif entry.get("type") == "symlink":
            target = entry.get("target")
            if not isinstance(target, str) or target.startswith("/") or "\\" in target:
                raise StorageError("invalid symlink target")
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(relative), target))
            _relative(resolved)
            if entries.get(resolved, {}).get("type") != "file":
                raise StorageError("symlinks must target regular files inside the backup")
        else:
            raise StorageError("unknown backup entry type")
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            if parent.as_posix() in entries:
                raise StorageError("manifest file/directory paths overlap")
            parent = parent.parent
    return entries


class SnapshotStore:
    """Transactional snapshots of a local workspace, using immutable remote objects."""

    def __init__(self, root: Path, transport: BackupTransport,
                 log: Callable[[str], None] = print) -> None:
        """Keep transport state outside the data tree so it cannot be backed up."""
        self.root: Path = root.resolve()
        self.transport: BackupTransport = transport
        self.log: Callable[[str], None] = log
        self.state_dir: Path = self.root.parent / (".drive-backup-state-" + self.root.name)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path: Path = self.state_dir / "state.json"
        self.index: dict[str, Any] = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        if self.index and self.index.get("remote") != transport.remote:
            raise StorageError("local data belongs to another remote; choose a new LOCAL_STORE_ROOT")
        self.writer: str = self.index.get("writer", uuid.uuid4().hex)
        self.base: str | None = self.index.get("base")
        self.entries: dict[str, dict[str, Any]] = self.index.get("files", {})
        self.signatures: dict[str, list[int]] = self.index.get("signatures", {})
        self.uploaded: set[str] = {e["sha256"] for e in self.entries.values() if e["type"] == "file"}

    def _save(self) -> None:
        """Persist the accepted head and hashing cache after a successful commit."""
        self.index = {"remote": self.transport.remote, "writer": self.writer, "base": self.base,
                      "files": self.entries, "signatures": self.signatures}
        _write_json(self.state_path, self.index)

    def _accept_own_head(self, identifier: str | None, name: str | None) -> None:
        """Recover a commit accepted remotely just before the local process stopped."""
        if identifier == self.base:
            return
        if name:
            document = self.transport.manifest(name)
            if document.get("writer") == self.writer and document.get("parent") == self.base:
                self.entries = _validate_manifest(document, str(identifier))
                self.base = identifier
                self.uploaded.update(e["sha256"] for e in self.entries.values() if e["type"] == "file")
                self.signatures.clear()
                self._save()
                return
        raise StorageError("remote advanced in another runtime; local work was preserved, automatic overwrite stopped")

    def restore(self) -> int:
        """Restore and verify a complete backup before exposing it to training."""
        self.transport.prepare()
        identifier, name = _head(self.transport.commits())
        present = self.root.exists() and any(p.is_file() or p.is_symlink() for p in self.root.rglob("*"))
        if self.index and self.root.exists() and (present or not self.entries):
            self._accept_own_head(identifier, name)
            self.root.mkdir(parents=True, exist_ok=True)
            return len(self.entries)
        if present and identifier:
            raise StorageError("local files already exist without backup state; choose an empty LOCAL_STORE_ROOT")
        if not identifier or not name:
            self.root.mkdir(parents=True, exist_ok=True)
            self._save()
            return 0
        document = self.transport.manifest(name)
        entries = _validate_manifest(document, identifier)
        temporary = self.state_dir / ("restore-" + identifier)
        temporary.mkdir(parents=True, exist_ok=True)
        by_digest: dict[str, Path] = {}
        for relative, entry in entries.items():
            if entry["type"] != "file":
                continue
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (target.is_file() and target.stat().st_size == entry["size"] and _hash(target) == entry["sha256"]):
                target.unlink(missing_ok=True)
                if entry["sha256"] in by_digest:
                    shutil.copyfile(by_digest[entry["sha256"]], target)
                else:
                    self.log(f"[storage] restoring {relative} ({entry['size'] / 1e6:.1f} MB)")
                    self.transport.download(_object(entry["sha256"]), target)
                if target.stat().st_size != entry["size"] or _hash(target) != entry["sha256"]:
                    target.unlink(missing_ok=True)
                    raise StorageError(f"backup checksum mismatch: {relative}; live files were not changed")
            os.utime(target, ns=(entry["mtime_ns"], entry["mtime_ns"]))
            by_digest[entry["sha256"]] = target
        for relative, entry in entries.items():
            if entry["type"] == "symlink":
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.unlink(missing_ok=True)
                target.symlink_to(entry["target"])
        if self.root.exists():
            shutil.rmtree(self.root)  # only an empty/uninitialized tree is allowed above
        temporary.replace(self.root)
        self.entries, self.base = entries, identifier
        self.uploaded.update(by_digest)
        self.signatures = {p: _signature(self.root / p) for p, e in entries.items() if e["type"] == "file"}
        self._save()
        self.log(f"[storage] restored {len(entries)} files")
        return len(entries)

    def _capture(self, path: Path, relative: str) -> tuple[dict[str, Any], list[int]] | None:
        """Copy/hash one stable generation; upload before returning its reference."""
        try:
            before = _signature(path)
            previous = self.entries.get(relative)
            if previous and self.signatures.get(relative) == before:
                return previous, before
            with tempfile.TemporaryDirectory(dir=self.state_dir, prefix="object-") as temporary:
                blob = Path(temporary) / "data"
                digest = hashlib.sha256()
                with path.open("rb") as source, blob.open("wb") as target:
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        digest.update(block)
                        target.write(block)
                if _signature(path) != before:
                    return None
                entry: dict[str, Any] = {"type": "file", "sha256": digest.hexdigest(),
                                          "size": before[2], "mtime_ns": before[3]}
                if path.suffix == ".json":
                    try:
                        value = json.loads(blob.read_text(encoding="utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        return None
                    if path.name == "LATEST.json":
                        entry["control"] = value
                elif path.suffix == ".jsonl" and before[2]:
                    with blob.open("rb") as handle:
                        handle.seek(max(0, before[2] - 1024 * 1024))
                        tail = handle.read()
                    if not tail.endswith(b"\n"):
                        try:
                            json.loads(tail.split(b"\n")[-1])
                        except (ValueError, UnicodeDecodeError):
                            return None
                if entry["sha256"] not in self.uploaded:
                    self.log(f"[storage] uploading {relative} ({before[2] / 1e6:.1f} MB)")
                    self.transport.upload(blob, _object(entry["sha256"]))
                    self.uploaded.add(entry["sha256"])
                return entry, before
        except FileNotFoundError:
            return None  # checkpoint pruning or atomic replacement; retain its prior committed version

    def _consistent(self, candidate: dict[str, dict[str, Any]]) -> None:
        """Keep checkpoint references and RIR pairs complete, and suppress stale scores."""
        for relative, entry in list(candidate.items()):
            path = PurePosixPath(relative)
            if path.name == "LATEST.json" and "ckpt" in path.parts:
                controls = entry.get("control")
                valid = isinstance(controls, list) and bool(controls)
                for checkpoint in controls if isinstance(controls, list) else []:
                    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("file"), str):
                        valid = False
                        continue
                    filename = checkpoint.get("file", "")
                    target = candidate.get(str(path.parent / filename), {})
                    valid = valid and "/" not in filename and "\\" not in filename and filename not in ("", ".", "..")
                    valid = valid and target.get("sha256") == checkpoint.get("sha256") and target.get("size") == checkpoint.get("bytes")
                if not valid:
                    prefix = str(path.parent.parent) + "/"
                    for key in list(candidate):
                        if key.startswith(prefix):
                            del candidate[key]
                    candidate.update({key: value for key, value in self.entries.items() if key.startswith(prefix)})
            if "rirbanks" in path.parts and re.fullmatch(r"(?:rirs_\d+\.npy|rooms_\d+\.json)", path.name):
                sibling = path.with_name(path.name.replace("rirs_", "rooms_").replace(".npy", ".json")
                                         if path.name.startswith("rirs_") else path.name.replace("rooms_", "rirs_").replace(".json", ".npy"))
                if str(sibling) not in candidate:
                    candidate.pop(relative, None)
            if path.name in ("score.json", "summary.json"):
                dependencies = [p for p in candidate if PurePosixPath(p).parent == path.parent and
                                (PurePosixPath(p).name in ("predictions.jsonl", "best.pt") or p.endswith(".predictions.jsonl"))]
                if any(candidate[p].get("mtime_ns", 0) > entry.get("mtime_ns", 0) for p in dependencies):
                    candidate.pop(relative, None)

    def backup(self, *, include_static: bool = True, force: bool = False) -> str | None:
        """Publish stable work, optionally retaining unchanged large-cache references."""
        self._accept_own_head(*_head(self.transport.commits()))
        candidate: dict[str, dict[str, Any]] = {p: e for p, e in self.entries.items()
                                               if not include_static and PurePosixPath(p).parts[0] in STATIC_DIRS}
        signatures: dict[str, list[int]] = {p: s for p, s in self.signatures.items() if p in candidate}
        links: list[tuple[str, str]] = []
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root).as_posix()
            if _excluded(relative):
                continue
            if not include_static and PurePosixPath(relative).parts[0] in STATIC_DIRS:
                continue
            if path.is_symlink():
                target = path.resolve()
                if not target.is_relative_to(self.root):
                    raise StorageError(f"symlink escapes local project: {relative}")
                links.append((relative, os.readlink(path)))
            elif path.is_file():
                captured = self._capture(path, relative)
                if captured:
                    candidate[relative], signatures[relative] = captured
                elif relative in self.entries:
                    candidate[relative] = self.entries[relative]
        for relative, target in links:
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(relative), target))
            if candidate.get(resolved, {}).get("type") == "file":
                candidate[relative] = {"type": "symlink", "target": target}
        captured_hashes = {p: e.get("sha256") for p, e in candidate.items()}
        self._consistent(candidate)
        signatures = {p: s for p, s in signatures.items() if candidate.get(p, {}).get("sha256") == captured_hashes.get(p)}
        if not force and candidate == self.entries and self.base is not None:
            return self.base
        self._accept_own_head(*_head(self.transport.commits()))
        identifier = uuid.uuid4().hex
        document = {"schema": SCHEMA, "id": identifier, "parent": self.base, "writer": self.writer,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": candidate}
        _validate_manifest(document, identifier)
        manifest = self.state_dir / "commit.json"
        _write_json(manifest, document)
        name = f"{self.base or 'root'}--{identifier}.json"
        self.transport.upload(manifest, "commits/" + name)
        for attempt in range(len(RETRY_DELAYS) + 1):
            names = self.transport.commits()
            if name in names:
                break
            if attempt == len(RETRY_DELAYS):
                raise StorageError("uploaded backup is not visible yet; local files were preserved, retry the connection")
            self.log(f"[storage] waiting for uploaded backup to appear ({attempt + 1}/{len(RETRY_DELAYS)})")
            time.sleep(RETRY_DELAYS[attempt])
        head, _ = _head(names)
        if head != identifier:
            raise StorageError("another runtime published concurrently; both histories were preserved")
        self.entries, self.base, self.signatures = candidate, identifier, signatures
        self._save()
        self.log(f"[storage] backup complete: {len(candidate)} files")
        return identifier


class StorageSession:
    """Periodic backup and command-boundary flushes for a Colab notebook."""

    def __init__(self, store: SnapshotStore, interval_sec: float = 120) -> None:
        """Start one writer after restore; network failures are surfaced to commands."""
        if interval_sec <= 0:
            raise ValueError("backup interval must be positive")
        self.store: SnapshotStore = store
        self.interval: float = interval_sec
        self.error: BaseException | None = None
        self.closed: bool = False
        self.guard: TextIO = (store.state_dir / "session.lock").open("a")
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.guard.close()
                raise StorageError("another backup session is already using this LOCAL_STORE_ROOT") from exc
        self.lock: threading.Lock = threading.Lock()
        self.stop_event: threading.Event = threading.Event()
        self.thread: threading.Thread = threading.Thread(target=self._loop, daemon=True, name="vcsasr-backup")
        self.thread.start()

    def _loop(self) -> None:
        """Back up completed work without hiding permanent transfer failures."""
        while not self.stop_event.wait(self.interval):
            try:
                self.flush(include_static=False)
            except Exception as exc:
                self.error = exc
                self.store.log(f"[storage] BACKUP FAILED: {exc}")
                return

    def check(self) -> None:
        """Stop further work when persistent storage is no longer healthy."""
        if self.closed:
            raise StorageError("backup session is closed; rerun the storage connection cell")
        if self.error:
            raise StorageError(f"personal Drive backup failed; local files remain available: {self.error}") from self.error

    def flush(self, *, include_static: bool = True) -> None:
        """Wait for one committed backup and propagate failures to the caller."""
        with self.lock:
            self.check()
            self.store.backup(include_static=False)
            if include_static:
                self.store.backup(include_static=True)

    def run(self, command: str | Sequence[str], *, shell: bool = False, check: bool = True) -> int:
        """Run a job, abort on backup failure, and flush completed files on exit."""
        self.check()
        process = subprocess.Popen(command, shell=shell, start_new_session=os.name == "posix",
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            while process.poll() is None:
                self.check()
                time.sleep(0.5)
        except BaseException:
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            raise
        self.flush()
        if process.returncode and check:
            raise subprocess.CalledProcessError(process.returncode, command)
        return int(process.returncode)

    def close(self) -> None:
        """Stop periodic work and wait for the final backup before reporting completion."""
        if self.closed:
            return
        self.stop_event.set()
        self.thread.join()
        try:
            self.flush()
        finally:
            self.closed = True
            self.guard.close()


def auth_from_colab_secret(secret_name: str, destination: Path) -> dict[str, Any]:
    """Read the personal OAuth package without placing credentials in notebook output."""
    from google.colab import userdata

    value = userdata.get(secret_name)
    if not value:
        raise StorageError(f"Colab secret {secret_name!r} is missing; follow docs/personal-drive.md")
    try:
        package = validate_auth(json.loads(value))
    except (ValueError, TypeError):
        raise StorageError("DRIVE_AUTH must contain the JSON produced by setup_drive.py") from None
    private_json(destination, package)
    return package


def connect_personal_drive(local_root: Path, *, secret_name: str = "DRIVE_AUTH", interval_sec: float = 120,
                           auth_file: Path | None = None) -> StorageSession:
    """Restore verified personal storage and confirm a write before starting notebook jobs."""
    if local_root.resolve().is_relative_to(Path("/content/drive")):
        raise StorageError("LOCAL_STORE_ROOT must be on the Colab VM, not a mounted Drive")
    config = auth_file or Path("/content/.config/vcsasr/drive-auth.json")
    if config.resolve().is_relative_to(local_root.resolve()):
        raise StorageError("Drive credentials must be outside LOCAL_STORE_ROOT")
    package = (validate_auth(json.loads(config.read_text())) if auth_file
               else auth_from_colab_secret(secret_name, config))
    store = SnapshotStore(local_root, GoogleDriveTransport.from_auth(package))
    store.restore()
    store.backup(include_static=False, force=True)
    return StorageSession(store, interval_sec)


def main(argv: list[str] | None = None) -> int:
    """Back up, restore or export ordinary files using a private OAuth setup package."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("restore", "backup", "export"))
    parser.add_argument("--auth", type=Path, default=Path("runs/private/drive-auth.json"))
    parser.add_argument("--local", type=Path)
    parser.add_argument("--out", type=Path, help="empty directory for an ordinary file-tree export")
    args = parser.parse_args(argv)
    root = args.out if args.action == "export" else args.local
    if root is None:
        parser.error("export needs --out; restore/backup need --local")
    if args.auth.resolve().is_relative_to(root.resolve()):
        parser.error("authentication must be outside the backed-up tree")
    transport = GoogleDriveTransport.from_auth(validate_auth(json.loads(args.auth.read_text())))
    store = SnapshotStore(root, transport)
    store.restore()
    if args.action == "backup":
        store.backup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



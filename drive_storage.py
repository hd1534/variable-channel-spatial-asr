"""Back up local Colab files to a separately authenticated Google Drive account.

Immutable, content-addressed objects are uploaded before a commit manifest, so
an interrupted upload never becomes a restore point. Every runtime appends to
its own chain of commits and never waits for the others. Histories that
diverge (two runtimes, a CLI run, a delayed Drive listing) are a normal state:
every reader merges all heads with one deterministic rule, and the next sync of
any writer publishes that merge. Heads are merged pairwise against their real
common ancestors (recursively, as git does), so a folder of results (a run, an
evaluation, an RIR bank) that only one history changed always comes from that
history. A folder changed in two histories keeps the version with the most
recently written files whole, and the other version stays restorable from the
history. The training and ASR processes keep using ordinary local filesystem
paths.
"""

from __future__ import annotations

import argparse
import calendar
import codecs
import collections
import contextlib
import hashlib
import json
import os
import posixpath
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, NamedTuple, Protocol, Sequence, TextIO

from drive_api import GoogleDriveTransport, StorageError, private_json, validate_auth

SCHEMA: int = 2
STATE_VERSION: int = 2
COMMIT_NAME: re.Pattern[str] = re.compile(r"(root|[0-9a-f]{32}(?:\+[0-9a-f]{32})*)--([0-9a-f]{32})\.json\Z")
SHA256: re.Pattern[str] = re.compile(r"[0-9a-f]{64}\Z")
SECRET_NAMES: frozenset[str] = frozenset({"rclone.conf", "token.json", "credentials.json", "client_secret.json", "drive-auth.json"})
STATIC_DIRS: frozenset[str] = frozenset({"models", "cache"})
PUBLIC_WEIGHT_DIRS: tuple[str, ...] = ("models/asr", "models/huggingface")
"""Hugging Face caches of public weights pinned to a revision (Qwen3-ASR, wav2vec2, Nemotron, VibeVoice,
MOSS, ...).  The Hub serves the same files far faster than a Drive round trip, so a workspace never backs
them up; entries an older backup already holds are carried forward unchanged, never re-read or deleted."""
# Result folders are merged as a unit so a run's checkpoints, summary and scores always come from one history.
GROUP_DEPTH: tuple[tuple[str, int], ...] = (("mono_baselines/runtimes/", 3), ("models/", 3))
# A transcript cache folder (<ASR_CACHE_NAME>/<backend>.jsonl, <backend>.inputs.json, qwen-runtime.json) is one unit.
CACHE_NAME: re.Pattern[str] = re.compile(r"[^/]+(?:\.jsonl|\.inputs\.json|-runtime\.json)\Z")
FREE_SPACE_MARGIN: int = 512 * 1024 * 1024
PROGRESS_FILES: frozenset[str] = frozenset({"metrics.jsonl"})
"""Files a running job appends to every few seconds.  A periodic backup that only moved these is
not logged one by one (:attr:`SnapshotStore.quiet_interval_sec` summarises them instead)."""
UNCONFIRMED: str = "unconfirmed-commit.json"  # a manifest whose upload started but was not confirmed (write-ahead)
LOCK_MESSAGE: str = ("another backup session is already using this LOCAL_STORE_ROOT; "
                     "run storage.flush() in the notebook instead, or stop that session first")


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


def _public_weights(relative: str) -> bool:
    """Whether a path is a public model cache that is never backed up (:data:`PUBLIC_WEIGHT_DIRS`)."""
    return any(relative == d or relative.startswith(d + "/") for d in PUBLIC_WEIGHT_DIRS)


def _static(relative: str) -> bool:
    """Large caches that are only backed up when a command finishes."""
    return PurePosixPath(relative).parts[0] in STATIC_DIRS


def _group(relative: str) -> str:
    """The unit that a merge takes whole from one history."""
    depth = next((d for prefix, d in GROUP_DEPTH if relative.startswith(prefix)), None)
    parts = relative.split("/")
    if depth is None:
        if len(parts) == 2 and parts[0] not in STATIC_DIRS and CACHE_NAME.fullmatch(parts[1]):
            return parts[0]  # a top-level transcript cache folder
        depth = 2
    return "/".join(parts[:depth]) if len(parts) > depth else relative


def _skipped(relative: str, skip: Sequence[str]) -> bool:
    """Whether a path is left in the backup without being downloaded ("" skips everything)."""
    return any(not prefix or relative == prefix or relative.startswith(prefix + "/") for prefix in skip)


def _normalize_skip(skip: Sequence[str]) -> tuple[str, ...]:
    """Store skip prefixes as clean relative POSIX paths."""
    cleaned: list[str] = []
    for prefix in skip:
        value = str(prefix).strip().strip("/")
        if value:
            _relative(value)
        cleaned.append(value)
    return tuple(dict.fromkeys(cleaned))


def _content(entry: dict[str, Any] | None) -> tuple[str, str] | None:
    """What makes two entries the same file, ignoring mtime and transport references."""
    if entry is None:
        return None
    return (entry["type"], entry["sha256"] if entry["type"] == "file" else entry["target"])


def _contents(files: dict[str, dict[str, Any]]) -> dict[str, tuple[str, str] | None]:
    """Content identity of a set of entries."""
    return {path: _content(entry) for path, entry in files.items()}


def _by_group(files: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Entries split into merge groups."""
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for path, entry in files.items():
        groups.setdefault(_group(path), {})[path] = entry
    return groups


def _stamp(files: dict[str, dict[str, Any]]) -> int:
    """When a version of a group was written: its newest file modification time."""
    return max((entry.get("mtime_ns", 0) for entry in files.values()), default=-1)


class Stamps(NamedTuple):
    """When each group's version was written, kept in every manifest so merges can follow lineage."""

    present: dict[str, int]  # group -> stamp of its version
    deleted: dict[str, int]  # absent group -> stamp of the version that was deleted (a tombstone)


def _restamp(previous: dict[str, dict[str, Any]], stamps: Stamps, current: dict[str, dict[str, Any]]) -> Stamps:
    """Stamp each group of a new commit: unchanged groups keep theirs, a changed group gets its newest file time
    but always more than the version it replaced, so a version is newer than every version it came from.
    A deleted group keeps the stamp of the version it deleted until it is created again."""
    before, after = _by_group(previous), _by_group(current)
    present: dict[str, int] = {}
    for group, members in after.items():
        old = stamps.present.get(group, _stamp(before[group]) if group in before else stamps.deleted.get(group, -1))
        same = group in before and _contents(before[group]) == _contents(members)
        present[group] = old if same else max(_stamp(members), old + 1)
    deleted = {g: t for g, t in stamps.deleted.items() if g not in after}
    for group in before.keys() - after.keys():
        deleted[group] = stamps.present.get(group, _stamp(before[group]))
    return Stamps(present, deleted)


def _union_jsonl(blobs: Sequence[bytes]) -> bytes | None:
    """Combine append-only transcript caches; later rows win per key, first-seen order is kept.

    Returns None when a file is not such a cache (a row without a string "key").
    """
    rows: dict[str, bytes] = {}
    for blob in blobs:
        lines = blob.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                key = json.loads(line).get("key")
            except (ValueError, UnicodeDecodeError, AttributeError):
                if index == len(lines) - 1 and not blob.endswith(b"\n"):
                    continue  # an interrupted final append
                return None
            if not isinstance(key, str):
                return None
            rows[key] = line
    return b"".join(line + b"\n" for line in rows.values())


def _union_inputs(blobs: Sequence[bytes]) -> bytes | None:
    """Combine ASR audio-provenance maps; None if a session's audio hash differs between them."""
    merged: dict[str, str] = {}
    for blob in blobs:
        try:
            value = json.loads(blob)
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            return None
        for key, digest in value.items():
            if merged.setdefault(key, digest) != digest:
                return None
    return (json.dumps(merged, ensure_ascii=False, indent=2) + "\n").encode()


def _created_ns(document: dict[str, Any]) -> int:
    """Commit time used to order concurrent heads (schema 1 only recorded seconds)."""
    value = document.get("created_ns")
    if type(value) is int:
        return value
    try:
        return calendar.timegm(time.strptime(str(document.get("created_utc")), "%Y-%m-%dT%H:%M:%SZ")) * 10**9
    except ValueError:
        return 0


def model_cache_dirs(repos: Sequence[str]) -> list[str]:
    """Backup paths that hold one Hugging Face repository in either notebook model cache."""
    return [f"models/{cache}/models--{repo.replace('/', '--')}" for repo in repos for cache in ("asr", "huggingface")]


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

    def upload(self, local: Path, relative: str, sha256: str | None = None) -> str | None:
        """Return only after uploaded bytes have been verified (against ``sha256`` when given); optionally
        return a direct reference."""
        ...

    def download(self, relative: str, local: Path, ref: str | None = None) -> str | None:
        """Stream verified bytes into a staging destination; optionally return their SHA-256 as the
        storage service reports it."""
        ...


class History:
    """The commit graph read from manifest names; several heads are a valid state."""

    def __init__(self, names: Sequence[str]) -> None:
        """Index parents, ignore commits whose ancestry is not visible yet, and find the heads."""
        self.parents: dict[str, tuple[str, ...]] = {}
        self.names: dict[str, str] = {}
        for name in dict.fromkeys(names):
            match = COMMIT_NAME.fullmatch(name)
            if match is None:
                raise StorageError("invalid commit filename")
            parents, identifier = match.groups()
            if identifier in self.parents:
                raise StorageError("duplicate commit identifier")
            listed = () if parents == "root" else tuple(parents.split("+"))
            if list(listed) != sorted(set(listed)) or identifier in listed:
                raise StorageError("invalid commit parents")
            self.parents[identifier] = listed
            self.names[identifier] = name
        self.generation: dict[str, int] = self._generations()
        self.ancestry: dict[str, frozenset[str]] = {}
        self.pending: set[str] = set(self.parents) - set(self.generation)
        referenced = {p for c in self.generation for p in self.parents[c]}
        self.heads: list[str] = sorted(set(self.generation) - referenced)

    def _generations(self) -> dict[str, int]:
        """Longest distance from a root commit, for commits whose whole ancestry is listed."""
        state: dict[str, int | None] = {}
        for start in self.parents:
            stack: list[tuple[str, bool]] = [(start, False)]
            visiting: set[str] = set()
            while stack:
                node, expanded = stack.pop()
                if node in state:
                    continue
                if not expanded:
                    if node in visiting:
                        state[node] = None  # a cycle can never become complete
                        continue
                    visiting.add(node)
                    stack.append((node, True))
                    stack.extend((p, False) for p in self.parents[node] if p in self.parents and p not in state)
                    continue
                visiting.discard(node)
                ancestry = [state.get(p) if p in self.parents else None for p in self.parents[node]]
                state[node] = None if None in ancestry else 1 + max((g for g in ancestry if g is not None), default=0)
        return {node: gen for node, gen in state.items() if gen is not None}

    def ancestors(self, identifier: str) -> frozenset[str]:
        """A commit and everything it was built on."""
        if identifier not in self.ancestry:
            seen: set[str] = set()
            frontier = [identifier]
            while frontier:
                node = frontier.pop()
                if node in seen or node not in self.parents:
                    continue
                seen.add(node)
                frontier.extend(self.parents[node])
            self.ancestry[identifier] = frozenset(seen)
        return self.ancestry[identifier]

    def newest(self, common: Iterable[str]) -> list[str]:
        """The commits of an ancestor-closed set that no other member was built on (its merge bases)."""
        members = set(common)
        return sorted(members - {p for c in members for p in self.parents[c]})

    def find(self, prefix: str) -> str:
        """Resolve a commit id or unique prefix, as printed in the logs."""
        matches = [c for c in self.parents if c.startswith(prefix.lower())]
        if len(matches) != 1:
            raise StorageError(f"commit {prefix!r} is {'ambiguous' if matches else 'not in the backup'}")
        return matches[0]


def _validate_manifest(document: dict[str, Any], identifier: str,
                       parents: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """Validate every restore destination and symlink before downloading any files."""
    if (not isinstance(document, dict) or document.get("schema") not in (1, 2)
            or document.get("id") != identifier or not isinstance(document.get("files"), dict)):
        raise StorageError("unsupported or invalid backup manifest")
    if parents is not None:
        recorded = (document.get("parents") if document["schema"] == 2
                    else [document["parent"]] if document.get("parent") else [])
        if recorded != sorted(parents):
            raise StorageError("backup manifest does not match its commit name")
    for field in ("stamps", "deleted"):
        stamps = document.get(field, {})
        if not isinstance(stamps, dict) or not all(isinstance(k, str) and type(v) is int for k, v in stamps.items()):
            raise StorageError("invalid group stamps")
    imported = document.get("imported", [])
    if not isinstance(imported, list) or not all(isinstance(group, str) for group in imported):
        raise StorageError("invalid import record")
    for group in imported:
        _relative(group)
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
            if "ref" in entry and not isinstance(entry["ref"], str):
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


Source = tuple[str, ...]  # the commits whose manifests hold one version of a group (several: combined from them)
Conflict = tuple[str, Source, bool, list[tuple[Source, bool]]]  # group, kept version (True: a deletion), others


class _Side:
    """One side of a three-way merge: its entries by group, the commits it contains and where each version is kept."""

    def __init__(self, groups: dict[str, dict[str, dict[str, Any]]], ancestry: frozenset[str], order: tuple[int, str],
                 tips: Source, sources: dict[str, Source], unresolved: frozenset[str] = frozenset(),
                 stamps: Stamps | None = None) -> None:
        """Hold a commit's or a merge result's state; order is (newest commit time, its id)."""
        self.groups = groups
        self.stamps = stamps or Stamps({}, {})  # recorded stamps (older manifests have none)
        self.ancestry = ancestry
        self.order = order
        self.tips = tips  # the commits merged into this side
        self.sources = sources  # group -> commits holding this exact version, including its absence
        self.unresolved = unresolved  # groups a virtual base could not settle (git's conflict markers)

    def source(self, group: str) -> Source:
        """Where this side's version of a group can be exported from."""
        return self.sources.get(group, self.tips)

    def stamp(self, group: str) -> int:
        """When this side's version of a group was written; never older than a version it came from."""
        return self.stamps.present[group] if group in self.stamps.present else _stamp(self.groups.get(group, {}))

    def tomb(self, group: str) -> int:
        """For a group this side deleted: the stamp of the version it deleted (-1 if unknown)."""
        return self.stamps.deleted.get(group, -1)

    def state(self, group: str) -> object:
        """Content identity for comparisons; an unresolved group equals nothing else."""
        return ("unresolved", id(self)) if group in self.unresolved else _contents(self.groups.get(group, {}))


class _Merger:
    """Merge any number of heads pairwise against their merge bases, recursively like git, for every reader alike."""

    def __init__(self, history: History, files_of: Callable[[str], dict[str, dict[str, Any]]],
                 created_of: Callable[[str], int], fetch: Callable[[dict[str, Any]], bytes],
                 stamps_of: Callable[[str], Stamps] = lambda _: Stamps({}, {})) -> None:
        """Read commits through the given callbacks; manifests are only fetched for heads and merge bases."""
        self.history = history
        self.stamps_of = stamps_of
        self.files_of = files_of
        self.created_of = created_of
        self.fetch = fetch
        self.sides: dict[str, _Side] = {}
        self.bases: dict[frozenset[str], _Side] = {}
        self.synthesized: dict[str, bytes] = {}
        self.conflicts: dict[str, list[tuple[Source, bool]]] = {}

    def commit(self, identifier: str) -> _Side:
        """One published commit as a merge side."""
        if identifier not in self.sides:
            self.sides[identifier] = _Side(_by_group(self.files_of(identifier)), self.history.ancestors(identifier),
                                           (self.created_of(identifier), identifier), (identifier,), {},
                                           stamps=self.stamps_of(identifier))
        return self.sides[identifier]

    def fold(self, commits: Sequence[str], report: bool) -> _Side:
        """Merge commits oldest first; each step uses the true merge base of everything merged so far."""
        ordered = sorted(commits, key=lambda c: (self.created_of(c), c))
        result = self.commit(ordered[0])
        for identifier in ordered[1:]:
            other = self.commit(identifier)
            result = self.three_way(self.base(result.ancestry & other.ancestry), result, other, report)
        return result

    def base(self, common: frozenset[str]) -> _Side:
        """The merge base; several newest common ancestors (a criss-cross) are first merged into a virtual base."""
        newest = self.history.newest(common)
        if not newest:
            return _Side({}, frozenset(), (0, ""), (), {})
        if len(newest) == 1:
            return self.commit(newest[0])
        key = frozenset(newest)
        if key not in self.bases:
            self.bases[key] = self.fold(newest, report=False)
        return self.bases[key]

    def read(self, entry: dict[str, Any]) -> bytes:
        """Bytes of a file version, including one combined earlier in this merge."""
        return self.synthesized[entry["sha256"]] if entry["sha256"] in self.synthesized else self.fetch(entry)

    def three_way(self, base: _Side, older: _Side, newer: _Side, report: bool) -> _Side:
        """Take each group from the side that changed it; resolve groups both sides changed.

        Inside a virtual base (report=False) a group both sides changed stays unresolved, so the outer merge
        decides it from the real versions instead of from a choice the base made.
        """
        groups: dict[str, dict[str, dict[str, Any]]] = {}
        sources: dict[str, Source] = {}
        stamps: dict[str, int] = {}
        unresolved: set[str] = set()
        for group in sorted(base.groups.keys() | older.groups.keys() | newer.groups.keys()
                            | base.unresolved | older.unresolved | newer.unresolved):
            original, mine, theirs = (side.groups.get(group, {}) for side in (base, older, newer))
            before, left, right = base.state(group), older.state(group), newer.state(group)
            if left == right or left == before:
                pick = newer
            elif right == before:
                pick = older
            else:
                combined = self.combine(group, {} if group in base.unresolved else original, mine, theirs)
                if combined is not None:
                    if combined:
                        groups[group] = combined
                        stamps[group] = max(older.stamp(group), newer.stamp(group)) + 1  # newer than both parts
                    sources[group] = tuple(sorted(set(older.source(group)) | set(newer.source(group))))
                    continue
                # Both changed it: keep the most recently written version. A deletion beats a version only if that
                # version is not newer than the one it deleted; a version changed after that one is kept.
                if mine and theirs:
                    keep_older = (older.stamp(group), older.order) > (newer.stamp(group), newer.order)
                elif mine:
                    keep_older = older.stamp(group) > newer.tomb(group)
                else:
                    keep_older = not newer.stamp(group) > older.tomb(group)
                pick, lost = (older, newer) if keep_older else (newer, older)
                if report:
                    self.conflicts.setdefault(group, []).append((lost.source(group), not lost.groups.get(group)))
                else:
                    unresolved.add(group)
            if pick.groups.get(group):
                groups[group] = pick.groups[group]
                stamps[group] = pick.stamp(group)
            sources[group] = pick.source(group)
            if group in pick.unresolved:
                unresolved.add(group)
        deleted = {g: max(older.tomb(g), newer.tomb(g)) for g in older.stamps.deleted.keys() | newer.stamps.deleted.keys()
                   if g not in groups}
        return _Side(groups, older.ancestry | newer.ancestry, max(older.order, newer.order),
                     older.tips + newer.tips, sources, frozenset(unresolved), Stamps(stamps, deleted))

    def combine(self, group: str, original: dict[str, dict[str, Any]], mine: dict[str, dict[str, Any]],
                theirs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
        """Merge a transcript cache folder that both sides changed; None unless their provenance agrees."""
        merged, blobs = _combine_cache(group, original, mine, theirs, self.read)
        self.synthesized.update(blobs)
        return merged


def _combine_cache(group: str, original: dict[str, dict[str, Any]], mine: dict[str, dict[str, Any]],
                   theirs: dict[str, dict[str, Any]], read: Callable[[dict[str, Any]], bytes]
                   ) -> tuple[dict[str, dict[str, Any]] | None, dict[str, bytes]]:
    """Merge a transcript cache folder file by file: transcripts and audio provenance are combined.

    Returns None when the two caches cannot be mixed. That is decided by comparing the two sides directly,
    never against the base: a runtime contract (e.g. qwen-runtime.json) that both have must be identical, and
    their audio-provenance maps must agree on every shared session.
    """
    paths = original.keys() | mine.keys() | theirs.keys()
    if not all(p.startswith(group + "/") and CACHE_NAME.fullmatch(p[len(group) + 1:]) for p in paths):
        return None, {}
    for path in sorted(mine.keys() & theirs.keys()):
        left, right = mine[path], theirs[path]
        if _content(left) == _content(right):
            continue
        if left["type"] != "file" or right["type"] != "file" or not path.endswith((".jsonl", ".inputs.json")):
            return None, {}  # made by another runtime installation
        if path.endswith(".inputs.json") and _union_inputs([read(left), read(right)]) is None:
            return None, {}  # one session was transcribed from different audio
    merged: dict[str, dict[str, Any]] = {}
    blobs: dict[str, bytes] = {}
    for path in sorted(paths):
        before, left, right = original.get(path), mine.get(path), theirs.get(path)
        if _content(left) == _content(right) or _content(left) == _content(before):
            entry = right
        elif _content(right) == _content(before):
            entry = left
        elif left is None or right is None:
            return None, {}  # deleted on one side and changed on the other
        else:
            parts = [read(left), read(right)]  # the newer side's rows win per key
            data = _union_jsonl(parts) if path.endswith(".jsonl") else _union_inputs(parts)
            if data is None:
                return None, {}
            digest = hashlib.sha256(data).hexdigest()
            blobs[digest] = data
            entry = {"type": "file", "sha256": digest, "size": len(data),
                     "mtime_ns": max(left["mtime_ns"], right["mtime_ns"])}
        if entry is not None:
            merged[path] = entry
    return merged, blobs


def merge_history(history: History, heads: Sequence[str], files_of: Callable[[str], dict[str, dict[str, Any]]],
                  created_of: Callable[[str], int], fetch: Callable[[dict[str, Any]], bytes],
                  stamps_of: Callable[[str], Stamps] = lambda _: Stamps({}, {})
                  ) -> tuple[dict[str, dict[str, Any]], dict[str, bytes], list[Conflict], Stamps]:
    """Merge heads identically for every reader.

    Returns the merged entries, combined transcript-cache blobs by sha256, the conflicts as
    (group, commits holding the kept version, whether that is a deletion,
    [(commits holding another version, whether that is a deletion)]) and the merged stamps.
    """
    merger = _Merger(history, files_of, created_of, fetch, stamps_of)
    side = merger.fold(heads, report=True)
    files = {path: entry for members in side.groups.values() for path, entry in members.items()}
    used = {entry["sha256"] for entry in files.values() if entry["type"] == "file"}
    conflicts = [(group, side.source(group), group not in side.groups,
                  [(s, d) for s, d in dict.fromkeys(losers) if s != side.source(group)])
                 for group, losers in sorted(merger.conflicts.items())]
    stamps = Stamps({group: side.stamp(group) for group in side.groups}, side.stamps.deleted)
    return files, {d: data for d, data in merger.synthesized.items() if d in used}, [c for c in conflicts if c[3]], stamps


def acquire_lock(root: Path) -> TextIO:
    """Hold the single local writer lock of a workspace for the lifetime of the returned handle."""
    state_dir = _state_dir(root)
    state_dir.mkdir(parents=True, exist_ok=True)
    guard: TextIO = (state_dir / "session.lock").open("a")
    if os.name == "posix":
        import fcntl

        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            guard.close()
            raise StorageError(LOCK_MESSAGE) from exc
    return guard


def _state_dir(root: Path) -> Path:
    """Transport state lives beside the data tree so it can never be backed up."""
    resolved = root.resolve()
    return resolved.parent / (".drive-backup-state-" + resolved.name)


class SnapshotStore:
    """Transactional snapshots of a local workspace, using immutable remote objects."""

    def __init__(self, root: Path, transport: BackupTransport, log: Callable[[str], None] = print,
                 *, skip: Sequence[str] | None = None) -> None:
        """Load (or migrate) local state; nothing is contacted until restore/sync/backup."""
        self.root: Path = root.resolve()
        self.transport: BackupTransport = transport
        self.quiet_interval_sec: float = 1800.0
        """How often the periodic backup reports the progress-only commits it did not log one by one."""
        self.quiet_commits: int = 0
        self.quiet_since: float = time.monotonic()
        self.log: Callable[[str], None] = log
        self.state_dir: Path = _state_dir(self.root)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path: Path = self.state_dir / "state.json"
        raw: dict[str, Any] = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        if raw and raw.get("remote") != transport.remote:
            raise StorageError("local data belongs to another remote; choose a new LOCAL_STORE_ROOT")
        self.initialized: bool = bool(raw)
        self.writer: str = raw.get("writer", uuid.uuid4().hex)
        self.legacy: bool = bool(raw) and raw.get("version") != STATE_VERSION  # made by the 9123e53 code
        if self.legacy:
            raw = self._migrate(raw)
        self.heads: list[str] = list(raw.get("heads", []))
        self.own: list[str] = list(raw.get("own", []))
        self.files: dict[str, dict[str, Any]] = raw.get("files", {})
        self.local: dict[str, dict[str, Any]] = raw.get("local", {})
        self.mode: str = raw.get("mode", "writer")
        self.installing: str | None = raw.get("installing")
        self.head_created_ns: int | None = raw.get("head_created_ns")  # set when the base is this writer's commit
        self.skip: tuple[str, ...] = _normalize_skip(raw.get("skip", []) if skip is None else skip)
        self.imported: set[str] = set(raw.get("imported", []))  # source folders IMPORT_FROM already copied in
        # Folders copied in but not yet in a published commit: {group: {path: sha256}}.
        self.import_pending: dict[str, dict[str, str]] = raw.get("import_pending", {})
        self.stamps: Stamps = Stamps(raw.get("stamps", {}), raw.get("deleted", {}))  # of the synced view
        self.manifests: dict[str, dict[str, Any]] = {}
        self.uploaded: set[str] = {e["sha256"] for e in self.files.values() if e["type"] == "file"}
        self.refs: dict[str, str] = {e["sha256"]: e["ref"] for e in self.files.values() if e.get("ref")}
        self.pending_seen: int = 0

    def _migrate(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Convert a v1 state (one linear base and a signature cache) without trusting unhashed files."""
        files = raw.get("files", {})
        signatures = raw.get("signatures", {})
        # v1 restored every file, so each entry was local; a missing one was deleted after the last backup.
        local: dict[str, dict[str, Any]] = {
            relative: ({"target": entry["target"]} if entry["type"] == "symlink"
                       else {"sha256": entry["sha256"], "sig": signatures.get(relative)})
            for relative, entry in files.items()}
        return {"heads": [raw["base"]] if raw.get("base") else [], "own": [], "files": files, "local": local}

    def _save(self) -> None:
        """Persist the synced view, own unlisted commits and the local hashing cache."""
        state = {"version": STATE_VERSION, "remote": self.transport.remote, "writer": self.writer,
                 "heads": self.heads, "own": self.own, "files": self.files, "local": self.local,
                 "skip": list(self.skip), "mode": self.mode, "head_created_ns": self.head_created_ns,
                 "imported": sorted(self.imported), "import_pending": self.import_pending, "stamps": self.stamps.present,
                 "deleted": self.stamps.deleted}
        if self.installing:
            state["installing"] = self.installing
        _write_json(self.state_path, state)
        self.initialized = True

    # ----- remote history -------------------------------------------------------------------------------------

    def _history(self) -> History:
        """List the remote history, including this writer's commits that Drive does not list yet."""
        refresh = getattr(self.transport, "refresh", None)
        if callable(refresh):
            refresh()  # a twin commits folder made by a simultaneous first connection
        names = self.transport.commits()
        listed = set(names)
        self.own = [name for name in self.own if name not in listed]
        history = History([*names, *self.own])
        if len(history.pending) != self.pending_seen:
            self.pending_seen = len(history.pending)
            if history.pending:
                self.log(f"[storage] {len(history.pending)} newer backup(s) are not fully visible on Drive yet; "
                         "using the latest complete history")
        return history

    def _document(self, history: History, identifier: str) -> dict[str, Any]:
        """Read and validate one commit manifest (immutable, so cached)."""
        name = history.names[identifier]
        if name not in self.manifests and self.heads == [identifier] and self.head_created_ns is not None:
            # This workspace's own last commit: known locally even while Drive cannot list or find it yet.
            return {"schema": SCHEMA, "id": identifier, "parents": list(history.parents[identifier]),
                    "writer": self.writer, "created_ns": self.head_created_ns, "files": self.files,
                    "imported": sorted(self.imported), "stamps": self.stamps.present, "deleted": self.stamps.deleted}
        if name not in self.manifests:
            document = self.transport.manifest(name)
            _validate_manifest(document, identifier, history.parents[identifier])
            self.manifests[name] = document
        return self.manifests[name]

    def _files_of(self, history: History, identifier: str) -> dict[str, dict[str, Any]]:
        """Entries of one commit, without a download when it is the synced local base."""
        if self.heads == [identifier]:
            return self.files
        return self._document(history, identifier)["files"]

    def _stamps_of(self, history: History, identifier: str) -> Stamps:
        """Per-group stamps of one commit (none for manifests written before they existed)."""
        if self.heads == [identifier]:
            return self.stamps
        document = self._document(history, identifier)
        return Stamps(document.get("stamps", {}), document.get("deleted", {}))

    def _view(self, history: History, heads: Sequence[str], *, reuse: bool = True
              ) -> tuple[dict[str, dict[str, Any]], dict[str, bytes], list[Conflict], Stamps]:
        """The restorable state of the given heads (reuse: the synced view stands for its own heads)."""
        if not heads:
            return {}, {}, [], Stamps({}, {})
        if reuse and list(heads) == self.heads:
            return self.files, {}, [], self.stamps
        if len(heads) == 1:
            return self._files_of(history, heads[0]), {}, [], self._stamps_of(history, heads[0])
        return merge_history(history, heads, lambda c: self._files_of(history, c),
                             lambda c: _created_ns(self._document(history, c)), self._fetch,
                             lambda c: self._stamps_of(history, c))

    def _adopt_imports(self, history: History, heads: Sequence[str]) -> None:
        """Carry the record of imported folders forward from every merged history."""
        for head in heads:
            self.imported.update(self._document(history, head).get("imported", []))

    def _report(self, conflicts: Sequence[Conflict]) -> None:
        """Say which version of a twice-changed folder is used and where the other one is kept."""
        def where(source: Source) -> str:
            if not source:
                return "an earlier merge"
            return (f"commit {source[0][:8]}" if len(source) == 1
                    else "combined from commits " + ", ".join(c[:8] for c in source))
        for group, kept, kept_deleted, others in conflicts:
            versions = [where(s) for s, deleted in others if not deleted]
            deletions = [where(s) for s, deleted in others if deleted]
            if kept_deleted:
                self.log(f"[storage] {group} was deleted in one runtime ({where(kept)}) and the other runtime's version "
                         f"is not newer than the deleted one; kept the deletion; that version stays in the backup "
                         f"history ({'; '.join(versions)})")
                continue
            if versions:
                self.log(f"[storage] {group} changed in two runtimes; kept the most recently written version "
                         f"({where(kept)}); the other version stays in the backup history ({'; '.join(versions)})")
            if deletions:
                self.log(f"[storage] {group} was deleted in one runtime and changed in another; kept the changed "
                         f"version ({where(kept)}); the deletion is {'; '.join(deletions)}")

    # ----- object transfer ------------------------------------------------------------------------------------

    def _upload(self, blob: Path, entry: dict[str, Any]) -> None:
        """Upload new content once and remember the transport reference."""
        digest = entry["sha256"]
        if digest not in self.uploaded:
            ref = self.transport.upload(blob, _object(digest), sha256=digest)  # already hashed: Drive verifies
            self.uploaded.add(digest)
            if isinstance(ref, str):
                self.refs[digest] = ref
        if digest in self.refs:
            entry["ref"] = self.refs[digest]

    def _fetch(self, entry: dict[str, Any]) -> bytes:
        """Read one verified object into memory (small transcript caches during a merge)."""
        with tempfile.TemporaryDirectory(dir=self.state_dir, prefix="fetch-") as temporary:
            target = Path(temporary) / "data"
            self._download(entry, target, {}, "transcript cache")
            return target.read_bytes()

    def _download(self, entry: dict[str, Any], target: Path, synthesized: dict[str, bytes], relative: str) -> None:
        """Write and verify one file version at a staging path."""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        streamed: str | None = None
        if entry["sha256"] in synthesized:
            target.write_bytes(synthesized[entry["sha256"]])
        else:
            self.log(f"[storage] restoring {relative} ({entry['size'] / 1e6:.1f} MB)")
            if entry.get("ref"):
                streamed = self.transport.download(_object(entry["sha256"]), target, ref=entry["ref"])
            else:
                streamed = self.transport.download(_object(entry["sha256"]), target)
        # Drive's own SHA-256 of bytes whose MD5 was checked while writing spares reading a multi-GB file back.
        digest = streamed if isinstance(streamed, str) else _hash(target)
        if target.stat().st_size != entry["size"] or digest != entry["sha256"]:
            target.unlink(missing_ok=True)
            raise StorageError(f"backup checksum mismatch: {relative}; live files were not changed")
        os.utime(target, ns=(entry["mtime_ns"], entry["mtime_ns"]))

    def _ensure_space(self, required: int) -> None:
        """Fail before a large restore instead of halfway through with a full disk."""
        free = shutil.disk_usage(self.state_dir).free
        if required and free < required + FREE_SPACE_MARGIN:
            raise StorageError(f"not enough free local disk for the backup: {required / 1e9:.1f} GB needed, "
                               f"{free / 1e9:.1f} GB free; add prefixes to RESTORE_SKIP or free space")

    # ----- restore, pull and export ---------------------------------------------------------------------------

    def _wanted(self, files: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Entries that this workspace keeps on its local disk."""
        return {p: e for p, e in files.items() if not _skipped(p, self.skip)}

    def _local_files(self) -> bool:
        """Whether the workspace holds any file or link."""
        return self.root.exists() and any(p.is_file() or p.is_symlink() for p in self.root.rglob("*"))

    def _staging(self, wanted: dict[str, dict[str, Any]]) -> tuple[Path, set[str]]:
        """The one restore staging tree: keep verified files of an interrupted restore, drop everything else."""
        for stale in self.state_dir.glob("restore-*"):
            shutil.rmtree(stale, ignore_errors=True)  # per-history staging trees of the previous version
        temporary = self.state_dir / "restore"
        temporary.mkdir(parents=True, exist_ok=True)
        verified: set[str] = set()
        for path in sorted(temporary.rglob("*"), reverse=True):  # children before their directories
            relative = path.relative_to(temporary).as_posix()
            entry = wanted.get(relative, {})
            if path.is_symlink() or path.is_file():
                if (not path.is_symlink() and entry.get("type") == "file" and path.stat().st_size == entry["size"]
                        and _hash(path) == entry["sha256"]):
                    verified.add(relative)
                else:
                    path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        return temporary, verified

    def _install(self, files: dict[str, dict[str, Any]], heads: list[str], synthesized: dict[str, bytes]) -> int:
        """Download a complete view beside the workspace, then move it into place."""
        wanted = self._wanted(files)
        temporary, verified = self._staging(wanted)
        if verified:
            self.log(f"[storage] resuming an interrupted restore ({len(verified)} files already downloaded)")
        self._ensure_space(sum(e["size"] for p, e in wanted.items() if e["type"] == "file" and p not in verified))
        by_digest: dict[str, Path] = {wanted[p]["sha256"]: temporary / p for p in sorted(verified)}
        for relative, entry in wanted.items():
            if entry["type"] != "file":
                continue
            target = temporary / relative
            if relative in verified:
                os.utime(target, ns=(entry["mtime_ns"], entry["mtime_ns"]))
            elif entry["sha256"] in by_digest:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(by_digest[entry["sha256"]], target)
                os.utime(target, ns=(entry["mtime_ns"], entry["mtime_ns"]))
            else:
                self._download(entry, target, synthesized, relative)
            by_digest[entry["sha256"]] = target
        for relative, entry in wanted.items():
            if entry["type"] == "symlink":
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.unlink(missing_ok=True)
                target.symlink_to(entry["target"])
        self.heads, self.files, self.local, self.installing = heads, files, {}, temporary.name
        self.head_created_ns = None
        self._save()  # a restart between the rename and the final save finishes this install
        if self.root.exists():
            shutil.rmtree(self.root)  # only an empty tree reaches this point
        temporary.replace(self.root)
        return self._finish_install()

    def _finish_install(self) -> int:
        """Record the installed files; also completes an install interrupted after its rename."""
        temporary = self.state_dir / str(self.installing)
        if temporary.exists():
            if self._local_files():
                raise StorageError("an interrupted restore and local files both exist; choose an empty LOCAL_STORE_ROOT")
            shutil.rmtree(self.root, ignore_errors=True)
            temporary.replace(self.root)
        self.local = {}
        for relative, entry in self._wanted(self.files).items():
            path = self.root / relative
            if entry["type"] == "symlink" and path.is_symlink():
                self.local[relative] = {"target": entry["target"]}
            elif entry["type"] == "file" and path.is_file() and not path.is_symlink():
                self.local[relative] = {"sha256": entry["sha256"], "sig": _signature(path)}
        self.installing = None
        self.uploaded.update(e["sha256"] for e in self.files.values() if e["type"] == "file")
        self._save()
        self.log(f"[storage] restored {len(self.local)} files"
                 + (f" ({len(self.files) - len(self.local)} left in the backup by RESTORE_SKIP)"
                    if len(self.files) > len(self.local) else ""))
        return len(self.files)

    def _pull(self, synthesized: dict[str, bytes] | None = None) -> int:
        """Make the local tree match the synced view, never overwriting unsaved local edits."""
        synthesized = synthesized or {}
        wanted = self._wanted(self.files)
        todo: list[tuple[str, dict[str, Any]]] = []
        for relative, entry in wanted.items():
            path = self.root / relative
            record = self.local.get(relative)
            if entry["type"] == "symlink":
                if path.is_symlink() and os.readlink(path) == entry["target"]:
                    self.local[relative] = {"target": entry["target"]}
                elif record is None and (path.exists() or path.is_symlink()):
                    self.log(f"[storage] kept local {relative}; it differs from the backup")
                else:
                    todo.append((relative, entry))
                continue
            if path.is_symlink() or (path.exists() and not path.is_file()):
                self.log(f"[storage] kept local {relative}; it differs from the backup")
                continue
            if path.is_file():
                current = _signature(path)
                if record and record.get("sig") == current:
                    digest = record.get("sha256")
                else:
                    digest = _hash(path)
                    known = record.get("sha256") if record else entry["sha256"]
                    if digest not in (known, entry["sha256"]):
                        self.log(f"[storage] kept local {relative}; it was edited after the last backup")
                        continue
                    self.local[relative] = {"sha256": digest, "sig": current}
                if digest == entry["sha256"]:
                    continue
            todo.append((relative, entry))
        removed: int = 0
        for relative, record in list(self.local.items()):
            if relative in self.files or _skipped(relative, self.skip):
                continue
            path = self.root / relative
            if path.is_symlink() or (path.is_file() and record.get("sig") in (None, _signature(path))):
                if record.get("sig") is None and not path.is_symlink() and _hash(path) != record.get("sha256"):
                    continue  # unhashed and changed: keep the local edit
                path.unlink()
                removed += 1
            del self.local[relative]
        if todo:
            size = sum(e["size"] for _, e in todo if e["type"] == "file" and e["sha256"] not in synthesized)
            self.log(f"[storage] updating {len(todo)} files from other runtimes ({size / 1e6:.1f} MB)")
            self._ensure_space(size)
        try:
            for relative, entry in todo:
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if entry["type"] == "symlink":
                    path.unlink(missing_ok=True)
                    path.symlink_to(entry["target"])
                    self.local[relative] = {"target": entry["target"]}
                    continue
                staging = self.state_dir / f"pull-{uuid.uuid4().hex}"
                try:
                    self._download(entry, staging, synthesized, relative)
                    os.replace(staging, path)
                finally:
                    staging.unlink(missing_ok=True)
                self.local[relative] = {"sha256": entry["sha256"], "sig": _signature(path)}
        finally:
            self._save()  # files already replaced stay recorded if a later download fails
        return len(todo) + removed

    def _sweep(self) -> None:
        """Delete temporary copies left by a process that was killed before its cleanup ran (lock held)."""
        for entry in self.state_dir.iterdir():
            name = entry.name
            if (name.startswith(("pull-", "object-", "fetch-", "merge-")) or name == "import" or name.endswith(".tmp")
                    or (name.startswith("commit-") and name.endswith(".json"))):
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)

    def restore(self, *, materialize: bool = True) -> int:
        """Restore a verified backup into an empty workspace, or sync an existing one."""
        self.transport.prepare()
        self._sweep()
        if self.installing:
            return self._finish_install()
        if self.initialized and self.root.exists() and (self._local_files() or not self.local):
            if self.mode == "export":
                return self.export()
            return self.sync()
        history = self._history()
        heads = history.heads
        if self._local_files() and heads:
            raise StorageError("local files already exist without backup state; choose an empty LOCAL_STORE_ROOT, "
                               "or add them with drive_storage.py import")
        files, synthesized, conflicts, stamps = self._view(history, heads)
        self._report(conflicts)
        self._upload_synthesized(files, synthesized)
        self._adopt_imports(history, heads)
        self.stamps = stamps
        if not materialize:
            self.skip = (*self.skip, "")  # a staging workspace that only adds files
        if not heads or not self._wanted(files):
            self.root.mkdir(parents=True, exist_ok=True)  # an empty remote may be seeded from these files
            self.heads, self.files, self.local = heads, files, {}
            self.head_created_ns = None
            self._save()
            return len(files)
        return self._install(files, heads, synthesized)

    def _adopt(self, document: dict[str, Any], name: str, local: dict[str, dict[str, Any]] | None) -> None:
        """Make this workspace's newly published commit its synced view.

        Without the scan's local records (a commit whose upload was interrupted and is finished later), the
        files that commit took from this workspace are marked unhashed, so the next scan sees edits and
        deletions made since; files it only carried forward without ever having them here stay unmarked.
        """
        if local is None:
            local = dict(self.local)
            for relative, entry in document["files"].items():
                path = self.root / relative
                if _skipped(relative, self.skip):
                    continue
                if not (path.is_file() or path.is_symlink()) and relative not in local \
                        and _content(self.files.get(relative)) == _content(entry):
                    continue  # carried forward; never on this VM
                if entry["type"] == "symlink":
                    local[relative] = {"target": entry["target"]}
                elif local.get(relative, {}).get("sha256") != entry["sha256"]:
                    local[relative] = {"sha256": entry["sha256"], "sig": None}
        self.manifests[name] = document
        self.imported |= set(document.get("imported", []))
        self.import_pending = {g: c for g, c in self.import_pending.items() if g not in self.imported}
        self.files, self.heads, self.local = document["files"], [document["id"]], local
        self.stamps = Stamps(document.get("stamps", {}), document.get("deleted", {}))
        self.head_created_ns = document["created_ns"]
        if name not in self.own:
            self.own.append(name)
        self.uploaded.update(e["sha256"] for e in self.files.values() if e["type"] == "file")
        self.refs.update({e["sha256"]: e["ref"] for e in self.files.values() if e.get("ref")})
        self._save()

    def _confirm_unconfirmed(self) -> None:
        """Finish publishing a commit whose manifest upload was interrupted (the process died, or Drive stored it
        but the reply was lost); backing up on the older base instead would fork this workspace's history."""
        pending = self.state_dir / UNCONFIRMED
        if not pending.exists():
            return
        try:
            document = json.loads(pending.read_text(encoding="utf-8"))
            identifier, parents = document["id"], document["parents"]
        except (ValueError, KeyError, TypeError):
            pending.unlink(missing_ok=True)  # never complete, so never uploaded
            return
        if identifier in self.heads or list(parents) != sorted(self.heads):
            if identifier in self.heads:
                self._save()  # adopted in memory; make sure the state on disk says so before forgetting the copy
            pending.unlink(missing_ok=True)
            return
        _validate_manifest(document, identifier, parents)
        name = f"{'+'.join(parents) or 'root'}--{identifier}.json"
        self.transport.upload(pending, "commits/" + name)  # a stored copy is only checked, never duplicated in effect
        self._adopt(document, name, None)
        pending.unlink(missing_ok=True)
        self.log(f"[storage] finished publishing this workspace's last backup (commit {identifier[:8]}), "
                 "which was interrupted")

    def _upload_synthesized(self, files: dict[str, dict[str, Any]], synthesized: dict[str, bytes]) -> None:
        """Store merged transcript caches as objects before any commit can reference them."""
        for digest, data in synthesized.items():
            with tempfile.TemporaryDirectory(dir=self.state_dir, prefix="merge-") as temporary:
                blob = Path(temporary) / "data"
                blob.write_bytes(data)
                for entry in files.values():
                    if entry.get("sha256") == digest:
                        self._upload(blob, entry)

    def export(self, commit: str | None = None) -> int:
        """Mirror the merged backup (or one commit) into this folder without publishing anything."""
        self.transport.prepare()
        self._sweep()
        if self.initialized and self.mode != "export" and not self.legacy:  # 9123e53 exports kept no mode
            raise StorageError("this folder is a backup workspace, not an export; choose an empty --out")
        self.mode = "export"
        if self.installing:
            self._finish_install()  # an export interrupted right after moving its files into place
        history = self._history()
        heads = [history.find(commit)] if commit else history.heads
        # Recomputed every time: an export never uploads the transcript caches a merge combines.
        files, synthesized, conflicts, stamps = self._view(history, heads, reuse=False)
        self._report(conflicts)
        self.stamps = stamps  # the rerun's merge reads this head's stamps from here
        if not self.initialized or not self._local_files():
            if self._local_files():
                raise StorageError("--out must be an empty directory for a new export")
            return self._install(files, heads, synthesized)
        self.heads, self.files, self.head_created_ns = heads, files, None
        self._pull(synthesized)
        left = len(files) - len(self._wanted(files))
        self.log(f"[storage] export up to date: {len(files)} files" + (f" ({left} left in the backup by --skip)" if left else ""))
        return len(files)

    # ----- local snapshot and commits -------------------------------------------------------------------------

    def _capture(self, path: Path, relative: str, base: str | None) -> tuple[dict[str, Any], list[int]] | None:
        """Copy/hash one stable generation; upload it when it differs from what the workspace synced."""
        try:
            before = _signature(path)
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
                if entry["sha256"] != base:
                    self._upload(blob, entry)
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
                    candidate.update({key: value for key, value in self.files.items() if key.startswith(prefix)})
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

    def _scan(self, include_static: bool) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Apply this workspace's edits since its last sync to the synced view."""
        candidate: dict[str, dict[str, Any]] = {}
        local: dict[str, dict[str, Any]] = {}
        for relative, entry in self.files.items():
            unscanned = (not include_static and _static(relative)) or _public_weights(relative)
            path = self.root / relative
            if unscanned or (relative not in self.local and not path.exists() and not path.is_symlink()):
                candidate[relative] = entry  # a large cache outside this pass, or a file not kept locally
        local.update({p: r for p, r in self.local.items() if not include_static and _static(p)})
        links: list[tuple[str, str]] = []
        captured: dict[str, str] = {}
        for path in sorted(self.root.rglob("*")) if self.root.exists() else []:
            relative = path.relative_to(self.root).as_posix()
            if _excluded(relative) or _public_weights(relative) or (not include_static and _static(relative)):
                continue
            record = self.local.get(relative)
            if path.is_symlink():
                target = path.resolve()
                if not target.is_relative_to(self.root):
                    raise StorageError(f"symlink escapes local project: {relative}")
                links.append((relative, os.readlink(path)))
                continue
            if not path.is_file():
                continue
            synced = record.get("sha256") if record else self.files.get(relative, {}).get("sha256")
            try:
                unchanged = bool(record) and record.get("sig") == _signature(path)
            except FileNotFoundError:
                unchanged = False  # pruned while scanning; _capture keeps the synced version
            if unchanged:
                result: tuple[dict[str, Any], list[int]] | None = ({"sha256": record["sha256"]}, record["sig"])
            else:
                result = self._capture(path, relative, synced)
            if result is None:  # still being written: keep the synced version for now
                if relative in self.files:
                    candidate[relative] = self.files[relative]
                if record:
                    local[relative] = record
                continue
            entry, signature = result
            local[relative] = {"sha256": entry["sha256"], "sig": signature}
            if entry["sha256"] == synced:
                if relative in self.files:
                    candidate[relative] = self.files[relative]  # unchanged here; a newer synced version wins
            else:
                candidate[relative] = entry
                captured[relative] = entry["sha256"]
        for relative, target in links:
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(relative), target))
            if candidate.get(resolved, {}).get("type") == "file":
                candidate[relative] = {"type": "symlink", "target": target}
                local[relative] = {"target": target}
        self._consistent(candidate)
        for relative, digest in captured.items():
            if candidate.get(relative, {}).get("sha256") != digest:  # held back: capture it again next time
                if relative in self.local:
                    local[relative] = self.local[relative]
                else:
                    local.pop(relative, None)
        for relative in candidate:
            if relative not in local and relative in self.local:
                local[relative] = self.local[relative]  # a held-back deletion is retried next time
        return candidate, local

    def _publish(self, files: dict[str, dict[str, Any]], parents: Sequence[str], local: dict[str, dict[str, Any]],
                 message: str, stamps: Stamps, *, quiet: bool = False) -> str:
        """Upload one manifest after its objects and adopt it as the synced view.

        ``quiet`` (the periodic backup): a commit that only moved progress logs is counted, not logged.
        """
        before = self.files
        identifier = uuid.uuid4().hex
        parents = sorted(set(parents))
        document = {"schema": SCHEMA, "id": identifier, "parents": parents, "writer": self.writer,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "created_ns": time.time_ns(), "files": files, "stamps": stamps.present,
                    "deleted": stamps.deleted}
        landed, held = self._landed_imports(files)
        if self.imported | set(landed):
            document["imported"] = sorted(self.imported | set(landed))
        _validate_manifest(document, identifier, parents)
        name = f"{'+'.join(parents) or 'root'}--{identifier}.json"
        pending = self.state_dir / UNCONFIRMED
        _write_json(pending, document)  # from here on Drive may hold this commit, even if the upload raises
        self.transport.upload(pending, "commits/" + name)
        self._adopt(document, name, local)
        pending.unlink(missing_ok=True)
        for group in held:
            self.log(f"[storage] imported {group}; left out of the backup because incomplete: {', '.join(held[group])}")
        changed = sorted(p for p in set(files) | set(before) if _content(files.get(p)) != _content(before.get(p)))
        notable = [p for p in changed if PurePosixPath(p).name not in PROGRESS_FILES]
        if quiet and not notable:
            self.quiet_commits += 1
            if time.monotonic() - self.quiet_since >= self.quiet_interval_sec:
                self.log(f"[storage] {self.quiet_commits} progress-only backups since the last report "
                         f"(latest commit {identifier[:8]}, {len(files)} files)")
                self.quiet_commits, self.quiet_since = 0, time.monotonic()
            return identifier
        shown = ", ".join(notable[:3]) + (f" and {len(notable) - 3} more" if len(notable) > 3 else "")
        detail = f"; changed: {shown}" if notable else ""
        self.log(f"[storage] {message}: {len(files)} files (commit {identifier[:8]}){detail}")
        self.quiet_commits, self.quiet_since = 0, time.monotonic()
        return identifier

    def _landed_imports(self, files: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, list[str]]]:
        """Imported folders this commit holds: every copied file is in it, or is in place here but refused by the
        backup's completeness checks (it would be refused again on any VM). Returns them and the refused files."""
        landed: list[str] = []
        held: dict[str, list[str]] = {}
        for group, copied in self.import_pending.items():
            missing = [p for p, digest in copied.items() if files.get(p, {}).get("sha256") != digest]
            if len(missing) == len(copied):
                continue  # nothing of it reached the backup yet
            here = lambda relative: ((self.root / relative).is_file() and not (self.root / relative).is_symlink()
                                     and _hash(self.root / relative) == copied[relative])
            if all(here(relative) for relative in missing):
                landed.append(group)
                if missing:
                    held[group] = sorted(missing)
        return landed, held

    def _writable(self) -> None:
        """Exports are read-only mirrors."""
        if self.mode == "export":
            raise StorageError("this folder is an export; back up from the notebook workspace instead")

    def backup(self, *, include_static: bool = True, force: bool = False, quiet: bool = False) -> str | None:
        """Commit this workspace's own changes on its own chain; never blocked by other runtimes.

        ``quiet``: do not log a commit that only moved progress logs (:data:`PROGRESS_FILES`)."""
        self._writable()
        self._confirm_unconfirmed()
        candidate, local = self._scan(include_static)
        if not force and candidate == self.files and (self.heads or not candidate):
            if local != self.local:
                self.local = local
                self._save()
            return self.heads[0] if len(self.heads) == 1 else None
        return self._publish(candidate, self.heads, local, "backup complete",
                             _restamp(self.files, self.stamps, candidate), quiet=quiet)

    def sync(self) -> int:
        """Commit local work, merge every other history into this one, and update local files."""
        self._writable()
        self.backup(include_static=True)
        history = self._history()
        heads = history.heads
        if any(h not in history.generation for h in self.heads):
            # Merging without our own last backup would roll this workspace back to an older state.
            raise StorageError("this workspace's last backup is missing from the Drive history (deleted, or not "
                               "visible yet); local files were preserved; retry later or choose a new LOCAL_STORE_ROOT")
        if heads and heads != self.heads:
            if len(heads) == 1 and all(h in history.ancestors(heads[0]) for h in self.heads):
                self._adopt_imports(history, heads)
                self.stamps = self._stamps_of(history, heads[0])
                self.files = self._files_of(history, heads[0])
                self.heads, self.head_created_ns = heads, None
                self.uploaded.update(e["sha256"] for e in self.files.values() if e["type"] == "file")
                self.refs.update({e["sha256"]: e["ref"] for e in self.files.values() if e.get("ref")})
                self._save()
                return self._pull()
            files, synthesized, conflicts, stamps = self._view(history, heads)
            self._report(conflicts)
            self._adopt_imports(history, heads)
            files = {p: dict(e) for p, e in files.items()}
            self._upload_synthesized(files, synthesized)
            self.uploaded.update(e["sha256"] for e in files.values() if e["type"] == "file")
            self._publish(files, heads, self.local, f"merged {len(heads)} backup histories", stamps)
            return self._pull(synthesized)
        return self._pull()

    def import_tree(self, source: Path) -> int:
        """Copy in result folders that the backup does not have yet, each whole and at most once."""
        self._writable()
        source = source.resolve()
        if not source.is_dir():
            raise StorageError(f"import source is not a directory: {source}")
        if source.is_relative_to(self.root) or self.root.is_relative_to(source):
            raise StorageError("import source and LOCAL_STORE_ROOT must not contain each other")
        # A folder imported before stays imported even after it was renamed or deleted in the backup.
        in_view = {_group(p) for p in self.files}
        present = in_view | self.imported
        if self.root.exists():
            present |= {_group(r) for r in (p.relative_to(self.root).as_posix() for p in self.root.rglob("*")
                                            if p.is_file() or p.is_symlink()) if not _excluded(r)}
        for group, copied in self.import_pending.items():
            # A whole-folder copy stopped halfway through moving in: copy it again. (A transcript cache combined
            # into one the backup already has is in the view; _import_caches combines it again.)
            if group not in in_view and group not in self.imported and not all((self.root / p).is_file() for p in copied):
                present.discard(group)
        skip = tuple(prefix for prefix in self.skip if prefix)  # "" only means nothing is downloaded here
        groups: dict[str, list[tuple[Path, str]]] = {}
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source).as_posix()
            if path.is_symlink() or not path.is_file() or _excluded(relative) or _skipped(relative, skip):
                continue
            if _group(relative) not in present:
                groups.setdefault(_group(relative), []).append((path, relative))
        staging = self.state_dir / "import"
        shutil.rmtree(staging, ignore_errors=True)  # a copy interrupted before it was complete
        copied: int = 0
        try:
            copied += self._import_caches(source, present, skip, staging)
            for group, members in groups.items():
                for path, relative in members:
                    (staging / relative).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, staging / relative)
                relatives = self._repair_import(staging, group, [relative for _, relative in members])
                # Recorded before the move (it only counts once a commit holds this content), so a process
                # killed right after the move still knows the folder was imported.
                self.import_pending[group] = {relative: _hash(staging / relative) for relative in relatives}
                self._save()
                # Only a complete copy enters the workspace, so a rerun never finds half a folder.
                target = self.root / group
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_dir() and not any(p.is_file() or p.is_symlink() for p in target.rglob("*")):
                    shutil.rmtree(target)  # empty directories left behind: move the copy in as one step
                if not target.exists():
                    os.replace(staging / group, target)
                else:
                    for relative in relatives:
                        (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
                        os.replace(staging / relative, self.root / relative)
                copied += len(relatives)
                shutil.rmtree(staging, ignore_errors=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            self._save()
        self.log(f"[storage] imported {copied} files from {source}")
        return copied

    def _repair_import(self, staging: Path, group: str, relatives: list[str]) -> list[str]:
        """Drop what the readers would skip anyway, so an imported folder can be backed up as it is.

        A checkpoint index entry whose file is missing or differs is skipped by training's resume, and an
        interrupted last JSONL line is dropped by every JSONL reader. Left in place, the first keeps the whole
        run folder out of every commit and the second keeps that file out.
        """
        kept = list(relatives)
        for relative in relatives:
            path = staging / relative
            if path.suffix == ".jsonl" and path.stat().st_size:
                size = path.stat().st_size
                with path.open("rb") as handle:
                    handle.seek(max(0, size - 1024 * 1024))
                    tail = handle.read()
                if not tail.endswith(b"\n"):
                    fragment = tail.split(b"\n")[-1]
                    try:
                        json.loads(fragment)
                    except (ValueError, UnicodeDecodeError):
                        with path.open("r+b") as handle:
                            handle.truncate(size - len(fragment))
                        self.log(f"[storage] {relative}: dropped an interrupted last line, as its readers do")
            if path.name == "LATEST.json" and path.parent.name == "ckpt":
                try:
                    entries = json.loads(path.read_text(encoding="utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                if not isinstance(entries, list):
                    continue
                valid = [e for e in entries if isinstance(e, dict) and isinstance(e.get("file"), str)
                         and "/" not in e["file"] and "\\" not in e["file"] and e["file"] not in ("", ".", "..")
                         and (path.parent / e["file"]).is_file()
                         and (path.parent / e["file"]).stat().st_size == e.get("bytes")
                         and _hash(path.parent / e["file"]) == e.get("sha256")]
                if len(valid) == len(entries):
                    continue
                if valid:
                    _write_json(path, valid)
                else:
                    path.unlink()
                    kept.remove(relative)
                self.log(f"[storage] {relative}: left out {len(entries) - len(valid)} checkpoint(s) the source never "
                         "received, as resuming training does")
        return kept

    def report_imports(self) -> None:
        """Name imported folders that the backup could not take yet."""
        arrived = {g: c for g, c in self.import_pending.items() if any((self.root / p).is_file() for p in c)}
        if arrived != self.import_pending:  # never moved in (killed first) or removed since: a rerun copies it
            self.import_pending = arrived
            self._save()
        for group in sorted(self.import_pending):
            self.log(f"[storage] imported {group} is not in the backup yet (its files are incomplete); it stays in "
                     "this workspace, is backed up once complete, and another VM imports it again until then")

    def _import_caches(self, source: Path, present: set[str], skip: tuple[str, ...], staging: Path) -> int:
        """Combine a transcript cache folder that the backup and the source both have, by the merge rule."""
        written: int = 0
        for directory in sorted(p for p in source.iterdir() if p.is_dir() and not p.is_symlink()):
            group = directory.name
            members = {f"{group}/{p.name}": p for p in directory.iterdir()
                       if p.is_file() and not p.is_symlink() and _group(f"{group}/{p.name}") == group
                       and not _excluded(f"{group}/{p.name}") and not _skipped(f"{group}/{p.name}", skip)}
            if not members or group not in present or group in self.imported:
                continue
            def describe(path: Path) -> dict[str, Any]:
                info = path.stat()
                return {"type": "file", "sha256": _hash(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns,
                        "path": str(path)}
            mine: dict[str, dict[str, Any]] = {}
            for relative, entry in self.files.items():
                if _group(relative) == group:
                    local = self.root / relative
                    mine[relative] = (describe(local) if local.is_file() and not local.is_symlink() else entry)
            if self.root.joinpath(group).is_dir():
                mine.update({f"{group}/{p.name}": describe(p) for p in self.root.joinpath(group).iterdir()
                             if p.is_file() and not p.is_symlink() and _group(f"{group}/{p.name}") == group
                             and not _excluded(f"{group}/{p.name}")})
            theirs = {relative: describe(path) for relative, path in members.items()}
            read = lambda e: Path(e["path"]).read_bytes() if "path" in e else self._fetch(e)
            merged, blobs = _combine_cache(group, {}, mine, theirs, read)
            if merged is None:
                self.log(f"[storage] kept this backup's {group}; the imported one was made with a different "
                         "runtime or from different audio, so the two are not combined")
                continue
            changed = [(r, e) for r, e in merged.items() if _content(e) != _content(mine.get(r))]
            staging.mkdir(parents=True, exist_ok=True)
            digests: dict[str, str] = {}
            for relative, entry in changed:
                partial = staging / relative
                partial.parent.mkdir(parents=True, exist_ok=True)
                if entry["sha256"] in blobs:
                    partial.write_bytes(blobs[entry["sha256"]])
                else:
                    shutil.copy2(entry["path"], partial)
                self._repair_import(staging, group, [relative])
                digests[relative] = _hash(partial)
            if digests:
                self.import_pending[group] = digests  # before the files move in, as for whole folders
            else:
                self.imported.add(group)  # nothing new in the source's cache
            self._save()
            # Provenance and runtime files move in before the transcripts, so an interruption can leave extra
            # provenance (harmless) but never transcripts without it (modern_asr would refuse the cache).
            for relative in sorted(digests, key=lambda r: (r.endswith(".jsonl"), r)):
                (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / relative, self.root / relative)
            written += len(changed)
        return written


def relay_output(process: subprocess.Popen, sink: TextIO | None = None,
                 tail_lines: int = 60) -> tuple[threading.Thread, collections.deque]:
    """Copy a child's piped output to this process's sys.stdout as it arrives.

    Colab shows only what the kernel writes through sys.stdout; a child that
    inherits the raw file descriptors prints nowhere visible.  The last lines
    are kept so a failure can be reported with its actual error message.
    Output is forwarded in chunks, so progress bars that use a carriage return still update.
    """
    tail: collections.deque = collections.deque(maxlen=tail_lines)

    def pump() -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        stream = process.stdout
        assert stream is not None
        while True:
            chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
            text = decoder.decode(chunk or b"", final=not chunk)
            if text:
                out = sink or sys.stdout
                out.write(text)
                out.flush()
                pending += text
                *lines, pending = re.split(r"\r?\n|\r", pending)
                tail.extend(line for line in lines if line.strip())
            if not chunk:
                break
        if pending.strip():
            tail.append(pending)

    thread = threading.Thread(target=pump, name="relay-output", daemon=True)
    thread.start()
    return thread, tail


def failure_with_output(returncode: int, command: str | Sequence[str], tail: Iterable[str]) -> subprocess.CalledProcessError:
    """A CalledProcessError whose message ends with the child's last output lines."""
    lines = list(tail)

    class CommandFailed(subprocess.CalledProcessError):
        def __str__(self) -> str:
            text = super().__str__()
            if lines:
                text += "\n--- last output lines ---\n" + "\n".join(lines)
            return text

    return CommandFailed(returncode, command, output="\n".join(lines))


class StorageSession:
    """Periodic backup, command-boundary syncs and flushes for a Colab notebook."""

    def __init__(self, store: SnapshotStore, interval_sec: float = 120, guard: TextIO | None = None) -> None:
        """Start one local writer after restore; network failures are surfaced to commands."""
        if interval_sec <= 0:
            raise ValueError("backup interval must be positive")
        self.store: SnapshotStore = store
        self.interval: float = interval_sec
        self.error: BaseException | None = None
        self.closed: bool = False
        self.guard: TextIO = guard or acquire_lock(store.root)
        self.lock: threading.Lock = threading.Lock()
        self.stop_event: threading.Event = threading.Event()
        self.thread: threading.Thread = threading.Thread(target=self._loop, daemon=True, name="vcsasr-backup")
        self.thread.start()

    def _loop(self) -> None:
        """Back up completed work without hiding permanent transfer failures."""
        while not self.stop_event.wait(self.interval):
            try:
                with self.lock:
                    if self.closed:
                        return
                    self.store.backup(include_static=False, quiet=True)
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
        """Wait for committed backups of everything written so far and propagate failures."""
        with self.lock:
            self.check()
            before = self.store.heads[:]
            self.store.backup(include_static=False)
            if include_static:
                self.store.backup(include_static=True)
            if self.store.heads == before:
                self.store.log("[storage] backup up to date")

    def sync(self) -> int:
        """Bring other runtimes' results into this workspace (no job may be running)."""
        with self.lock:
            self.check()
            return self.store.sync()

    def run(self, command: str | Sequence[str], *, shell: bool = False, check: bool = True) -> int:
        """Sync, run a job, abort on backup failure, and flush completed files on exit."""
        self.sync()
        process = subprocess.Popen(command, shell=shell, start_new_session=os.name == "posix",
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
        relay, tail = relay_output(process)
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
            relay.join(timeout=5)
            raise
        relay.join()
        self.flush()
        if process.returncode and check:
            raise failure_with_output(process.returncode, command, tail)
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
                           auth_file: Path | None = None, skip: Sequence[str] = (),
                           import_from: Path | None = None) -> StorageSession:
    """Restore or sync verified personal storage and confirm a write before starting notebook jobs."""
    if local_root.resolve().is_relative_to(Path("/content/drive")):
        raise StorageError("LOCAL_STORE_ROOT must be on the Colab VM, not a mounted Drive")
    config = auth_file or Path("/content/.config/vcsasr/drive-auth.json")
    if config.resolve().is_relative_to(local_root.resolve()):
        raise StorageError("Drive credentials must be outside LOCAL_STORE_ROOT")
    package = (validate_auth(json.loads(config.read_text())) if auth_file
               else auth_from_colab_secret(secret_name, config))
    guard = acquire_lock(local_root)
    try:
        store = SnapshotStore(local_root, GoogleDriveTransport.from_auth(package), skip=skip)
        store.restore()
        imported = store.import_tree(import_from) if import_from else 0
        store.backup(include_static=bool(imported), force=True)
        store.report_imports()
        return StorageSession(store, interval_sec, guard=guard)
    except BaseException:
        guard.close()
        raise


def main(argv: list[str] | None = None) -> int:
    """Back up, restore, export or import ordinary files using a private OAuth setup package."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("restore", "backup", "export", "import"))
    parser.add_argument("--auth", type=Path, default=Path("runs/private/drive-auth.json"))
    parser.add_argument("--local", type=Path, help="backup workspace (restore/backup/import)")
    parser.add_argument("--out", type=Path, help="export folder; rerunning updates it in place")
    parser.add_argument("--from", dest="source", type=Path, help="import: folder whose missing results are added")
    parser.add_argument("--commit", help="export: one commit id (prefix) from the logs instead of the merged latest")
    parser.add_argument("--skip", action="append", default=[],
                        help="path prefix left in the backup (repeatable; for export it applies to this run only)")
    args = parser.parse_args(argv)
    root = args.out if args.action == "export" else args.local
    if root is None:
        parser.error("export needs --out; restore/backup/import need --local")
    if args.action == "import" and args.source is None:
        parser.error("import needs --from")
    if args.auth.resolve().is_relative_to(root.resolve()):
        parser.error("authentication must be outside the backed-up tree")
    transport = GoogleDriveTransport.from_auth(validate_auth(json.loads(args.auth.read_text())))
    with acquire_lock(root):
        if args.action == "import":
            store = SnapshotStore(root, transport)
            # --skip adds prefixes; a staging workspace keeps its "" marker so a rerun downloads nothing either.
            store.skip = _normalize_skip((*store.skip, *args.skip))
            store.restore(materialize=bool(store.initialized))
            store.import_tree(args.source)
            store.backup()
            store.report_imports()
            return 0
        # An export uses exactly this run's --skip; the other actions keep the workspace's saved RESTORE_SKIP.
        store = SnapshotStore(root, transport, skip=args.skip if args.action == "export" else args.skip or None)
        if args.action == "export":
            store.export(args.commit)
            return 0
        store.restore()
        if args.action == "backup":
            store.sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

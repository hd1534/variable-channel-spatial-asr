"""A precomputed bank of rooms for on-the-fly training-session mixing.

pyroomacoustics needs 0.1-16 s per room (four positions, three copies each;
mean about 3.4 s on one M1 core, dominated by the reverberant rooms), far too
slow to run per training session.  So
rooms are computed once, stored as float16, and every training session picks a
room and mixes fresh speech through it.  Diversity then comes from the room
count times the speech, turn layout, gain and channel-subset draws.

Each room reuses the eval generator's building blocks, so training acoustics
follow the same rules as the evaluation set:

* :func:`simulate.make_mic_array` -- the same asymmetric 4-mic arrays, with the
  same "1100 is the short pair, 1010 the long pair" labelling;
* :func:`simulate._fit_room_to_block` -- the same room-size range and wall
  margin, sized around the array and *all four* sources;
* sources on the common plane at ``spec.PLANE_HEIGHT_M``, azimuth uniform,
  distance log-uniform over ``spec.DISTANCE_RANGE_M``.

Shards of :data:`SHARD_ROOMS` rooms are written atomically and skipped when
present and whole, so an interrupted build resumes where it stopped.  The
bank records which simulator version made it: pyroomacoustics 0.8.5 (the
eval sets' version) ships no Linux wheel, so on Colab it is built from its git
tag (on macOS the tag build reproduced the wheel's responses bit for bit; a
Linux build has not been compared); 0.10.1 is not a substitute (different
reverberant tails for the same room).
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import simulate
import spec

SAMPLE_RATE_HZ = spec.SAMPLE_RATE_HZ
SOURCES_PER_ROOM = 4
SHARD_ROOMS = 50
MIN_SOURCE_SEPARATION_M = 0.5
"""Two talkers cannot stand closer than this."""

MIN_JITTER_DISTANCE_M = 0.4
"""A displaced talker never comes closer to the array reference point than this."""

JITTER_WALL_MARGIN_M = 0.2
"""...nor closer to a wall than this (base positions keep ``simulate.SOURCE_WALL_MARGIN_M``)."""

BANK_VERSION = "2"


@dataclass(frozen=True)
class BankSpec:
    seed: int
    n_rooms: int
    rir_sec: float
    rt60_range_sec: tuple[float, float]
    engine: str
    variants: int = 3
    """Responses per talker position: the position itself and ``variants - 1``
    displaced copies.  Each turn of a talker uses one of them, so a talker is
    never a perfectly static source -- without that, the exact reflection
    pattern of a fixed position identifies a talker no matter where it stands
    (measured: even mirror-image positions separate perfectly), and a spatial
    model would learn the simulator rather than the direction."""
    jitter_m: tuple[float, float] = (0.10, 0.30)
    """Displacement of the extra variants from the position, uniform, any direction."""

    @property
    def dirname(self) -> str:
        lo, hi = self.rt60_range_sec
        jlo, jhi = self.jitter_m
        return (f"rirbank_v{BANK_VERSION}_{self.engine}_s{self.seed}_n{self.n_rooms}_L{self.rir_sec:g}_rt{lo:g}-{hi:g}"
                f"_k{self.variants}_j{jlo:g}-{jhi:g}")

    @property
    def n_taps(self) -> int:
        return int(round(self.rir_sec * SAMPLE_RATE_HZ))


def _place_sources(rng: random.Random) -> dict[str, simulate.SourcePlacement]:
    """Four source positions, pairwise at least :data:`MIN_SOURCE_SEPARATION_M` apart."""
    lo, hi = spec.DISTANCE_RANGE_M
    while True:
        placed: dict[str, simulate.SourcePlacement] = {}
        for i in range(SOURCES_PER_ROOM):
            az = rng.uniform(*spec.AZIMUTH_RANGE_DEG)
            dist = math.exp(rng.uniform(math.log(lo), math.log(hi)))
            dist = min(max(dist, lo), hi)
            placed[f"s{i}"] = simulate.SourcePlacement(
                position_id=f"s{i}",
                distance_level=f"s{i}",
                azimuth_deg=az,
                distance_m=dist,
                azimuth_bin=spec.azimuth_bin_index(az),
                distance_bin=spec.distance_bin_index(dist),
            )
        xy = [p.xy for p in placed.values()]
        if all(
            math.dist(xy[a], xy[b]) >= MIN_SOURCE_SEPARATION_M
            for a in range(len(xy))
            for b in range(a + 1, len(xy))
        ):
            return placed


def _rirs_pyroomacoustics_multi(block: simulate.RoomBlock, rt60_s: float, keys: Sequence[str]) -> list[list[Any]]:
    """Every listed source in one ShoeBox run: ``rirs[key][mic]``."""
    import numpy as np
    import pyroomacoustics as pra

    dims = list(block.dims_xyz_m)
    try:
        absorption, max_order = pra.inverse_sabine(rt60_s, dims)
    except Exception:  # noqa: BLE001 - same fallback as simulate._rirs_pyroomacoustics
        absorption, max_order = simulate._sabine_absorption(rt60_s, dims, spec.SPEED_OF_SOUND_M_S)
    room = pra.ShoeBox(
        dims,
        fs=SAMPLE_RATE_HZ,
        materials=pra.Material(float(absorption)),
        max_order=int(max_order),
        air_absorption=False,
        ray_tracing=False,
    )
    room.add_microphone_array(pra.MicrophoneArray(np.asarray(block.mic_room_xyz(), dtype=np.float64).T, SAMPLE_RATE_HZ))
    for key in keys:
        room.add_source(list(block.source_room_xyz(key)))
    room.compute_rir()
    return [[np.asarray(room.rir[m][s], dtype=np.float64) for m in range(spec.N_CHANNELS_MAX)] for s in range(len(keys))]


def _placement(key: str, x: float, y: float) -> simulate.SourcePlacement:
    az = math.degrees(math.atan2(y, x))
    az = az - 360.0 if az >= 180.0 else az
    dist = math.hypot(x, y)
    return simulate.SourcePlacement(
        position_id=key, distance_level=key, azimuth_deg=az, distance_m=dist,
        azimuth_bin=spec.azimuth_bin_index(az),
        distance_bin=spec.distance_bin_index(min(max(dist, spec.DISTANCE_RANGE_M[0]), spec.DISTANCE_RANGE_M[1])),
    )


def _jitter(rng: random.Random, bank: BankSpec, base: simulate.SourcePlacement, dims: Sequence[float],
            center: Sequence[float]) -> tuple[float, float]:
    """A displaced copy of ``base`` inside the room, away from the array and the walls."""
    bx, by = base.xy
    for attempt in range(200):
        r = rng.uniform(*bank.jitter_m) * (0.5 if attempt >= 100 else 1.0)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        x, y = bx + r * math.cos(angle), by + r * math.sin(angle)
        rx, ry = center[0] + x, center[1] + y
        if (math.hypot(x, y) >= MIN_JITTER_DISTANCE_M
                and JITTER_WALL_MARGIN_M <= rx <= dims[0] - JITTER_WALL_MARGIN_M
                and JITTER_WALL_MARGIN_M <= ry <= dims[1] - JITTER_WALL_MARGIN_M):
            return x, y
    return bx, by


def build_room(bank: BankSpec, index: int) -> tuple[Any, dict[str, Any]]:
    """Return ``(rirs float16 [4 sources, variants, 4 mics, taps], room metadata)``."""
    import numpy as np

    rng = random.Random(f"{bank.seed}:room:{index}")
    while True:
        geom = simulate.make_mic_array(rng)
        sources = _place_sources(rng)
        try:
            dims, center = simulate._fit_room_to_block(rng, geom, sources)
        except ValueError:
            continue
        break
    rt60 = rng.uniform(*bank.rt60_range_sec)
    positions: dict[str, simulate.SourcePlacement] = {}
    for s in range(SOURCES_PER_ROOM):
        base = sources[f"s{s}"]
        positions[f"s{s}v0"] = _placement(f"s{s}v0", *base.xy)
        for v in range(1, bank.variants):
            positions[f"s{s}v{v}"] = _placement(f"s{s}v{v}", *_jitter(rng, bank, base, dims, center))
    block = simulate.RoomBlock(
        room_id=f"room_{index:05d}",
        dims_xyz_m=dims,
        array_center_xy_m=center,
        geometry=geom,
        sources=positions,
        rt60_requested_s={"requested": rt60},
    )
    keys = [f"s{s}v{v}" for s in range(SOURCES_PER_ROOM) for v in range(bank.variants)]
    if bank.engine == "pyroomacoustics":
        per_key = _rirs_pyroomacoustics_multi(block, rt60, keys)
    elif bank.engine == "fallback_ism":
        per_key = [simulate._rirs_fallback_ism(block, key, rt60, SAMPLE_RATE_HZ)[0] for key in keys]
    else:
        raise ValueError(f"unknown RIR engine {bank.engine!r}")
    out = np.zeros((SOURCES_PER_ROOM, bank.variants, spec.N_CHANNELS_MAX, bank.n_taps), dtype=np.float16)
    for i, mics in enumerate(per_key):
        s, v = divmod(i, bank.variants)
        for m, rir in enumerate(mics):
            n = min(len(rir), bank.n_taps)
            out[s, v, m, :n] = rir[:n]
    meta = {
        "room_id": block.room_id,
        "dims_xyz_m": list(dims),
        "array_center_xy_m": list(center),
        "mic_xy_m": geom.as_json(),
        "sources": [
            {"azimuth_deg": sources[f"s{s}"].azimuth_deg, "distance_m": sources[f"s{s}"].distance_m,
             "xy_m": list(sources[f"s{s}"].xy),
             "variants_xy_m": [list(positions[f"s{s}v{v}"].xy) for v in range(bank.variants)]}
            for s in range(SOURCES_PER_ROOM)
        ],
        "rt60_requested_s": rt60,
    }
    return out, meta


def _shard_rooms(bank: BankSpec, shard: int) -> int:
    return min(bank.n_rooms, (shard + 1) * SHARD_ROOMS) - shard * SHARD_ROOMS


def _shard_ok(folder: Path, bank: BankSpec, shard: int) -> bool:
    """A shard counts only if its array opens with the exact expected shape.

    A truncated copy (an interrupted upload to Drive, say) fails here and is
    rebuilt instead of crashing every later run at load time.
    """
    import numpy as np

    npy, meta = folder / f"rirs_{shard:04d}.npy", folder / f"rooms_{shard:04d}.json"
    if not (npy.is_file() and meta.is_file()):
        return False
    try:
        arr = np.load(npy, mmap_mode="r")
        rooms = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    want = (_shard_rooms(bank, shard), SOURCES_PER_ROOM, bank.variants, spec.N_CHANNELS_MAX, bank.n_taps)
    return tuple(arr.shape) == want and arr.dtype == np.float16 and len(rooms) == want[0]


def _copy_shard(src: Path, dst: Path, shard: int) -> None:
    """Copy one shard, metadata first, each file under a temporary name then renamed."""
    import shutil

    dst.mkdir(parents=True, exist_ok=True)
    for name in (f"rooms_{shard:04d}.json", f"rirs_{shard:04d}.npy"):
        part = dst / (name + ".part")
        shutil.copyfile(src / name, part)
        os.replace(part, dst / name)


def _single_threaded_worker() -> None:
    """Pool initializer: one thread per process, or the processes fight over the cores."""
    try:
        import pyroomacoustics as pra

        pra.constants.set("num_threads", 1)
    except ImportError:  # pragma: no cover - fallback_ism banks
        pass


def _build_shard(args: tuple[BankSpec, int, str]) -> int:
    import numpy as np

    bank, shard, out_dir = args
    first = shard * SHARD_ROOMS
    last = first + _shard_rooms(bank, shard)
    rirs, metas = [], []
    for index in range(first, last):
        r, m = build_room(bank, index)
        rirs.append(r)
        metas.append(m)
    target = Path(out_dir) / f"rirs_{shard:04d}.npy"
    tmp = target.with_suffix(".tmp.npy")
    np.save(tmp, np.stack(rirs))
    (Path(out_dir) / f"rooms_{shard:04d}.json").write_text(json.dumps(metas), encoding="utf-8")
    os.replace(tmp, target)  # the .npy appears last, so its presence means the shard is whole
    return shard


def _engine_version(engine: str) -> str | None:
    if engine != "pyroomacoustics":
        return None
    try:
        import pyroomacoustics
    except ImportError:
        return None
    return pyroomacoustics.__version__


def _check_bank_version(src_dir: Path, bank: BankSpec, version: str | None) -> None:
    meta_path = src_dir / "bank.json"
    if not meta_path.is_file():
        if version is not None:
            raise ValueError(
                f"cannot mix existing unversioned RIR shards in {src_dir} with {bank.engine} {version!r}; "
                "rebuild in a clean directory"
            )
        return
    old_ver = json.loads(meta_path.read_text(encoding="utf-8")).get("engine_version")
    if old_ver is not None and version is not None and old_ver != version:
        raise ValueError(
            f"cannot mix RIR shards built with {bank.engine} {old_ver!r} ({src_dir}) "
            f"and current {version!r}; use the matching version or rebuild in a clean directory"
        )
    if old_ver is None and version is not None:
        raise ValueError(
            f"cannot mix RIR shards without engine_version in {src_dir} with {bank.engine} {version!r}; "
            "rebuild in a clean directory"
        )


def build_bank(bank: BankSpec, root: str | Path, workers: int = 0, log=print, mirror: str | Path | None = None) -> Path:
    """Build (or finish building) ``root/<bank.dirname>`` and return its path.

    ``mirror`` is a second copy that outlives the machine (Google Drive on
    Colab): shards found there are copied in instead of rebuilt, and every
    shard built here is copied out as soon as it is done, so a disconnect
    loses at most the shard in progress.
    """
    out = Path(root) / bank.dirname
    out.mkdir(parents=True, exist_ok=True)
    far = Path(mirror) / bank.dirname if mirror else None
    n_shards = -(-bank.n_rooms // SHARD_ROOMS)
    in_out = [s for s in range(n_shards) if _shard_ok(out, bank, s)]
    in_far = [s for s in range(n_shards) if s not in in_out and far is not None and _shard_ok(far, bank, s)]
    todo = [s for s in range(n_shards) if s not in in_out and s not in in_far]

    if in_out and in_far and (out / "bank.json").is_file() and far is not None and (far / "bank.json").is_file():
        ver_out = json.loads((out / "bank.json").read_text(encoding="utf-8")).get("engine_version")
        ver_far = json.loads((far / "bank.json").read_text(encoding="utf-8")).get("engine_version")
        if ver_out is not None and ver_far is not None and ver_out != ver_far:
            raise ValueError(
                f"cannot mix RIR shards built with {bank.engine} {ver_out!r} ({out}) "
                f"and {ver_far!r} ({far}); rebuild in a clean directory"
            )

    if todo:
        version = _engine_version(bank.engine)
        if in_out or (out / "bank.json").is_file():
            _check_bank_version(out, bank, version)
        if far is not None and (in_far or (far / "bank.json").is_file()):
            _check_bank_version(far, bank, version)

    for shard in in_far:
        _copy_shard(far, out, shard)
        if not _shard_ok(out, bank, shard):
            todo.append(shard)

    info = {"version": BANK_VERSION, "seed": bank.seed, "n_rooms": bank.n_rooms, "rir_sec": bank.rir_sec,
            "rt60_range_sec": list(bank.rt60_range_sec), "engine": bank.engine, "shard_rooms": SHARD_ROOMS,
            "variants": bank.variants, "jitter_m": list(bank.jitter_m)}
    previous = out / "bank.json"

    if todo:
        version = _engine_version(bank.engine)
        if in_out or previous.is_file():
            _check_bank_version(out, bank, version)
        if far is not None and (in_far or (far / "bank.json").is_file()):
            _check_bank_version(far, bank, version)
        info["engine_version"] = version
        previous.write_text(json.dumps(info), encoding="utf-8")
        if far is not None:
            far.mkdir(parents=True, exist_ok=True)
            (far / "bank.json").write_text(previous.read_text(encoding="utf-8"), encoding="utf-8")
        workers = workers or max(1, os.cpu_count() or 1)  # the parent only waits on the pool
        log(f"[rirbank] building {len(todo)}/{n_shards} shards of {SHARD_ROOMS} rooms with {workers} worker(s) "
            f"({bank.engine} {version or ''}) -> {out}")
        jobs = [(bank, s, str(out)) for s in todo]
        done = 0

        def finished(shard: int) -> None:
            nonlocal done
            done += _shard_rooms(bank, shard)
            if far is not None:
                _copy_shard(out, far, shard)
            log(f"[rirbank] {done} rooms")

        if workers == 1:
            for job in jobs:
                finished(_build_shard(job))
        else:
            import multiprocessing as mp

            with mp.get_context("spawn").Pool(workers, initializer=_single_threaded_worker) as pool:
                for shard in pool.imap_unordered(_build_shard, jobs):
                    finished(shard)
    if not todo and not previous.is_file() and far is not None and (far / "bank.json").is_file():
        # every shard came from the mirror: so does the record of which simulator made them
        previous.write_text((far / "bank.json").read_text(encoding="utf-8"), encoding="utf-8")
    if not previous.is_file():
        info["engine_version"] = None
        previous.write_text(json.dumps(info), encoding="utf-8")
    if far is not None and not (far / "bank.json").is_file():
        far.mkdir(parents=True, exist_ok=True)
        (far / "bank.json").write_text(previous.read_text(encoding="utf-8"), encoding="utf-8")
    return out


class RirBank:
    """Memory-mapped view over a built bank; cheap to share with DataLoader workers."""

    def __init__(self, path: str | Path) -> None:
        import numpy as np

        self.path = Path(path)
        info = json.loads((self.path / "bank.json").read_text(encoding="utf-8"))
        n_shards = -(-info["n_rooms"] // info["shard_rooms"])
        self.shards = [np.load(self.path / f"rirs_{s:04d}.npy", mmap_mode="r") for s in range(n_shards)]
        self.rooms: list[dict[str, Any]] = []
        for s in range(n_shards):
            self.rooms.extend(json.loads((self.path / f"rooms_{s:04d}.json").read_text(encoding="utf-8")))
        self.shard_rooms = info["shard_rooms"]
        self.variants = int(info.get("variants", 1))
        if len(self.rooms) != info["n_rooms"]:
            raise ValueError(f"{self.path}: expected {info['n_rooms']} rooms, found {len(self.rooms)}")

    def __len__(self) -> int:
        return len(self.rooms)

    def rirs(self, index: int, sources: Sequence[int]):  # noqa: ANN201 - numpy array
        """float16 ``[len(sources), variants, 4 mics, taps]``."""
        import numpy as np

        shard = self.shards[index // self.shard_rooms]
        return np.asarray(shard[index % self.shard_rooms][list(sources)])


def bank_ready(bank: BankSpec, root: str | Path, mirror: str | Path | None = None) -> bool:
    """True when every shard is whole locally or in ``mirror``, i.e. nothing needs the simulator."""
    folders = [Path(root) / bank.dirname] + ([Path(mirror) / bank.dirname] if mirror else [])
    n_shards = -(-bank.n_rooms // SHARD_ROOMS)
    return all(any(_shard_ok(f, bank, s) for f in folders) for s in range(n_shards))


def open_bank(bank: BankSpec, root: str | Path, workers: int = 0, log=print, mirror: str | Path | None = None) -> RirBank:
    return RirBank(build_bank(bank, root, workers=workers, log=log, mirror=mirror))

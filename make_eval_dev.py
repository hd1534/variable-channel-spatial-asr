#!/usr/bin/env python3
"""Build the ``eval-dev`` prototype evaluation set for the VC-SASR benchmark.

Usage::

    python3 make_eval_dev.py --out data/eval-dev --n-sessions 12 --seed 20261029
    python3 make_eval_dev.py --out /tmp/smoke --n-sessions 2 --seed 1 --synthetic
    python3 make_eval_dev.py --out /tmp/x --n-sessions 12 --verify-only   # no numpy needed

Outputs, all under ``--out``:

``audio/<session_id>.wav``   4-channel 16-bit PCM, ``spec.SESSION_LENGTH_SEC`` long.
``rir/<session_id>.npz``     the four RIRs that produced it (rule 10's cache format).
``meta.jsonl``               one row per ``case_id``: inputs only, no answers.
``ground_truth.jsonl``       one row per ``case_id``: transcript, speaker, and the
                             azimuth / distance / bin labels at each ``T_eval``
                             instant, plus the *measured* RT60 and room id.
``manifest.json``            ``spec.SPEC_VERSION``, seed, counts, stratification and
                             2x2 cell tables, every verification report, and a
                             SHA256 for every file written.

``meta.jsonl`` and ``ground_truth.jsonl`` are deliberately separate files so the
inputs can be handed to a model without the answers travelling with them.

All heavy dependencies are imported lazily inside :mod:`simulate`; this script
installs nothing and makes no network request.  ``--verify-only`` runs the whole
plan-layer verification on a bare interpreter.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, "") and str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import simulate
import spec

TRACK: int = 1
"""Only Track 1 (one stationary speaker) is in scope for this round."""

SOURCE_MODES: tuple[str, ...] = ("synthetic", "librispeech")


# ---------------------------------------------------------------------------
# Source utterances
# ---------------------------------------------------------------------------


def scan_librispeech(root: str | Path) -> dict[str, list[tuple[str, Path, str]]]:
    """Index a LibriSpeech tree as ``{speaker_id: [(utt_id, flac_path, transcript)]}``.

    Expects the standard layout ``<root>/<speaker>/<chapter>/`` holding
    ``<speaker>-<chapter>-<utt>.flac`` beside ``<speaker>-<chapter>.trans.txt``.
    Returns an empty mapping when the tree is absent, so the caller can fall
    back to ``--synthetic`` with a clear message instead of crashing.
    """
    root = Path(root)
    out: dict[str, list[tuple[str, Path, str]]] = {}
    if not root.is_dir():
        return out
    for trans_path in sorted(root.rglob("*.trans.txt")):
        transcripts: dict[str, str] = {}
        for line in trans_path.read_text(encoding="utf-8").splitlines():
            utt_id, _, text = line.partition(" ")
            if utt_id and text:
                transcripts[utt_id] = text.strip()
        for utt_id, text in sorted(transcripts.items()):
            flac = trans_path.parent / f"{utt_id}.flac"
            if not flac.is_file():
                continue
            speaker = utt_id.split("-")[0]
            out.setdefault(speaker, []).append((utt_id, flac, text))
    return out


def build_source_pool(
    plan: simulate.EvalPlan,
    mode: str,
    librispeech_root: str | Path | None,
    seed: int,
    sample_rate_hz: int,
) -> dict[str, dict[str, Any]]:
    """Return ``{position_id: {"signal", "transcript", "speaker_id", "utt_id", ...}}``.

    One utterance per *source position*, shared by that position's two
    absorption renders, so the "same distance, different RT60" contrast differs
    only in absorption.

    In ``librispeech`` mode, speakers are split with
    :func:`simulate.assign_splits` and only the ``eval`` speakers are drawn
    from, so the eval set is speaker-disjoint from train and val by
    construction (rule 9).
    """
    if mode not in SOURCE_MODES:
        raise ValueError(f"mode must be one of {SOURCE_MODES}, got {mode!r}")

    position_ids: list[str] = []
    for block in plan.blocks:
        for level in simulate.DISTANCE_LEVELS:
            position_ids.append(block.sources[level].position_id)
    used = {s.source.position_id for s in plan.sessions}
    position_ids = [p for p in position_ids if p in used]

    pool: dict[str, dict[str, Any]] = {}
    max_duration = (
        spec.SESSION_LENGTH_SEC - simulate.MAX_START_OFFSET_SEC - 1.0
    )

    if mode == "synthetic":
        for index, position_id in enumerate(position_ids):
            session = next(s for s in plan.sessions if s.source.position_id == position_id)
            rng = random.Random(f"{seed}:synthetic:{position_id}")
            duration = min(rng.uniform(*simulate.SYNTHETIC_DURATION_RANGE_SEC), max_duration)
            signal, transcript, meta = simulate.synthesise_utterance(rng, sample_rate_hz, duration)
            pool[position_id] = {
                "signal": signal,
                "transcript": transcript,
                "speaker_id": session.speaker_id,
                "utt_id": session.source_utt_id,
                "source_meta": meta,
                "truncated_from_sec": None,
            }
        return pool

    if not librispeech_root:
        raise SystemExit(
            "--source librispeech needs --librispeech-root. Pass --synthetic to run the\n"
            "whole pipeline with placeholder speech instead (numpy only, no corpus)."
        )
    index_by_speaker = scan_librispeech(librispeech_root)
    if not index_by_speaker:
        raise SystemExit(
            f"no LibriSpeech utterances found under {librispeech_root!r}.\n"
            "Expected <root>/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac with a\n"
            "matching .trans.txt. Pass --synthetic to run without the corpus."
        )
    if len(index_by_speaker) < len(simulate.SPLIT_NAMES):
        raise SystemExit(
            f"found only {len(index_by_speaker)} LibriSpeech speakers; at least "
            f"{len(simulate.SPLIT_NAMES)} are needed to isolate train/val/eval (rule 9)."
        )

    split_of = simulate.assign_splits(sorted(index_by_speaker), seed=seed)
    eval_speakers = sorted(s for s, name in split_of.items() if name == "eval")
    if not eval_speakers:
        raise SystemExit("the speaker split left the eval side empty; raise the eval fraction")

    candidates: list[tuple[str, str, Path, str]] = []
    for speaker in eval_speakers:
        for utt_id, path, text in index_by_speaker[speaker]:
            candidates.append((speaker, utt_id, path, text))
    if len(candidates) < len(position_ids):
        raise SystemExit(
            f"need {len(position_ids)} eval utterances but the eval speaker split only has "
            f"{len(candidates)}; use a larger LibriSpeech subset or fewer sessions."
        )
    random.Random(f"{seed}:librispeech").shuffle(candidates)

    for position_id, (speaker, utt_id, path, text) in zip(position_ids, candidates):
        signal = simulate.load_librispeech_utterance(path, sample_rate_hz)
        original_sec = len(signal) / float(sample_rate_hz)
        truncated_from = None
        limit = int(round(max_duration * sample_rate_hz))
        if len(signal) > limit:
            signal = signal[:limit]
            truncated_from = original_sec
        pool[position_id] = {
            "signal": signal,
            "transcript": text,
            "speaker_id": speaker,
            "utt_id": utt_id,
            "source_meta": {
                "source": "librispeech",
                "path": str(path),
                "duration_sec": len(signal) / float(sample_rate_hz),
                "sample_rate_hz": sample_rate_hz,
                "split_of_speaker": split_of[speaker],
            },
            "truncated_from_sec": truncated_from,
        }
    return pool


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def make_meta_record(
    session: simulate.SessionPlan,
    mask_bits: str,
    audio_info: dict[str, Any],
    audio_relpath: str,
    sample_rate_hz: int,
) -> dict[str, Any]:
    """Build one ``meta.jsonl`` row: model inputs only, no answers.

    The microphone coordinates are the *active* channels' offsets in the one
    fixed array frame, with the reference point still at the origin, so a model
    given ``"0010"`` sees where its single microphone actually is rather than a
    re-centred frame (rule 2).
    """
    channels = spec.mask_to_channels(mask_bits)
    sub = session.block.geometry.for_mask(mask_bits)
    return {
        "case_id": spec.make_case_id(session.session_id, mask_bits),
        "session_id": session.session_id,
        "mask_bits": mask_bits,
        "track": TRACK,
        "n_channels": len(channels),
        "active_channels": list(channels),
        "audio": {
            "path": audio_relpath,
            "channels_in_file": int(audio_info["n_channels"]),
            "read_channels": list(channels),
            "sample_rate_hz": int(sample_rate_hz),
            "n_samples": int(audio_info["n_samples"]),
            "encoding": audio_info["encoding"],
        },
        "session_length_sec": spec.SESSION_LENGTH_SEC,
        "mic_positions_xy_m": sub.as_json(),
        "array_reference_point_xy_m": list(sub.reference_point_xy),
        "plane_height_m": spec.PLANE_HEIGHT_M,
        "speed_of_sound_m_s": spec.SPEED_OF_SOUND_M_S,
        "max_pair_spacing_m": sub.max_pair_spacing_m(),
        "time_origin": spec.TIME_ORIGIN,
        "delta_t_sec": spec.DELTA_T_SEC,
        "n_grid_points": len(simulate.grid_times_sec()),
        "official_scopes": list(spec.OFFICIAL_SCOPES),
        "max_speakers": spec.MAX_SPEAKERS_PER_TRACK[TRACK],
    }


def make_ground_truth_record(
    session: simulate.SessionPlan,
    mask_bits: str,
    rir_info: dict[str, Any],
    render_info: dict[str, Any],
    source: dict[str, Any],
    t_eval_sec: Sequence[float],
) -> dict[str, Any]:
    """Build one ``ground_truth.jsonl`` row.

    Track 1 holds a single stationary speaker, so azimuth and distance are
    constant across ``T_eval``; the per-instant records are emitted anyway so
    the file shape does not have to change for Track 2.

    The talker identity written here is the *actual* one from ``source``, which
    in ``librispeech`` mode is a corpus speaker, not the plan's placeholder
    slot.  Stamping ``session.speaker_id`` onto real corpus audio would lose
    every trace of which file produced a case and would make the speaker
    disjointness of the splits unverifiable from the written data -- while
    looking perfectly clean, because the plan's slots are disjoint by
    construction.  The planned slot is kept alongside under
    ``source_provenance`` so plan and record can still be matched up.
    """
    placement = session.source
    speaker_id = str(source.get("speaker_id") or session.speaker_id)
    source_utt_id = str(source.get("utt_id") or session.source_utt_id)
    source_meta = dict(source.get("source_meta") or {})
    speaker_split = source_meta.get("split_of_speaker")
    if speaker_split is not None and speaker_split != session.split:
        raise ValueError(
            f"speaker {speaker_id!r} belongs to the {speaker_split!r} split but session "
            f"{session.session_id} is planned as {session.split!r}; writing this record "
            "would leak a speaker across splits (rule 9)"
        )
    measured = [v for v in rir_info["rt60_measured_s"] if isinstance(v, float) and v == v]
    frames = [
        {
            "eval_time_sec": t,
            "azimuth_deg": placement.azimuth_deg,
            "distance_m": placement.distance_m,
            "azimuth_bin": placement.azimuth_bin,
            "distance_bin": placement.distance_bin,
        }
        for t in t_eval_sec
    ]
    return {
        "case_id": spec.make_case_id(session.session_id, mask_bits),
        "session_id": session.session_id,
        "mask_bits": mask_bits,
        "track": TRACK,
        "speaker_id": speaker_id,
        "source_utt_id": source_utt_id,
        "variant_group": session.variant_group,
        "split": session.split,
        "transcript": source["transcript"],
        "final_transcripts_reference": {speaker_id: source["transcript"]},
        "source_provenance": {
            "kind": source_meta.get("source", "synthetic"),
            "path": source_meta.get("path"),
            "split_of_speaker": speaker_split or session.split,
            "planned_speaker_slot": session.speaker_id,
            "planned_source_utt_id": session.source_utt_id,
        },
        "room": {
            "room_id": session.block.room_id,
            "acoustic_id": session.acoustic_id,
            "dims_xyz_m": list(session.block.dims_xyz_m),
            "array_center_xy_m": list(session.block.array_center_xy_m),
            "absorption_level": session.absorption_level,
            "absorption": rir_info["engine_info"].get("absorption"),
            "max_order": rir_info["engine_info"].get("max_order"),
        },
        "rt60": {
            "requested_s": rir_info["rt60_requested_s"],
            "measured_s_per_channel": rir_info["rt60_measured_s"],
            "measured_s_median": statistics.median(measured) if measured else None,
            "method": rir_info["engine_info"].get("rt60_method"),
            "engine": rir_info["engine"],
            "label_source": "measured",
        },
        "source": {
            "position_id": placement.position_id,
            "azimuth_deg": placement.azimuth_deg,
            "distance_m": placement.distance_m,
            "azimuth_bin": placement.azimuth_bin,
            "distance_bin": placement.distance_bin,
            "distance_level": placement.distance_level,
            "xy_m": list(placement.xy),
        },
        "confound_cell": {
            "distance_level": session.distance_level,
            "absorption_level": session.absorption_level,
            "index": list(session.cell),
        },
        "timing": {
            "start_offset_sec": session.start_offset_sec,
            "start_offset_samples": render_info["offset_samples"],
            "utterance_duration_sec": source["source_meta"]["duration_sec"],
            "truncated_from_sec": source["truncated_from_sec"],
            "tail_truncated": render_info["tail_truncated"],
            "n_t_eval": len(t_eval_sec),
        },
        "gain": {
            "common_gain_dbfs_target": session.common_gain_dbfs,
            "common_gain_linear": render_info["common_gain_linear"],
            "channel_gain_db": render_info["channel_gain_db"],
            "anti_clip_scale": render_info["anti_clip_scale"],
            "channel_rms_dbfs": render_info["channel_rms_dbfs"],
            "ild_spread_db": render_info["ild_spread_db"],
        },
        "propagation": {
            "direct_delay_samples": rir_info["direct_delay_samples"],
            "max_tdoa_samples": rir_info["max_tdoa_samples"],
        },
        "t_eval_sec": list(t_eval_sec),
        "frames": frames,
    }


_SESSION_OUTPUT_NAME = re.compile(r"^sess_\d{4,}$")
"""Filenames this generator itself writes under ``audio/`` and ``rir/``."""


def prune_stale_outputs(out_dir: str | Path, session_ids: Iterable[str]) -> list[str]:
    """Delete this generator's own leftover session files from an earlier run.

    Regenerating into a directory that already holds a *larger* set would
    otherwise leave orphaned ``audio/sess_NNNN.wav`` and ``rir/sess_NNNN.npz``
    behind.  The manifest would correctly list only the current sessions, but a
    consumer globbing ``audio/*.wav`` -- the obvious thing to do -- would load
    sessions that have no ``meta.jsonl`` or ``ground_truth.jsonl`` row at all,
    silently mixing two datasets.

    Only files matching this generator's own ``sess_<digits>`` naming in its own
    two subdirectories are touched, never anything else in ``out_dir``, and the
    removed paths are returned so the caller can report them and stamp them into
    the manifest.  Nothing is deleted silently.

    >>> import tempfile, pathlib
    >>> with tempfile.TemporaryDirectory() as d:
    ...     root = pathlib.Path(d)
    ...     (root / "audio").mkdir(); (root / "rir").mkdir()
    ...     for n in ("sess_0000", "sess_0001"):
    ...         _ = (root / "audio" / f"{n}.wav").write_bytes(b"")
    ...         _ = (root / "rir" / f"{n}.npz").write_bytes(b"")
    ...     _ = (root / "audio" / "notes.txt").write_text("keep me")
    ...     prune_stale_outputs(root, ["sess_0000"])
    ...     sorted(p.name for p in (root / "audio").iterdir())
    ['audio/sess_0001.wav', 'rir/sess_0001.npz']
    ['notes.txt', 'sess_0000.wav']
    """
    keep = {str(s) for s in session_ids}
    removed: list[str] = []
    for subdir, suffix in (("audio", ".wav"), ("rir", ".npz")):
        directory = Path(out_dir) / subdir
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix != suffix:
                continue
            if not _SESSION_OUTPUT_NAME.match(path.stem) or path.stem in keep:
                continue
            path.unlink()
            removed.append(f"{subdir}/{path.name}")
    return removed


def _finite(values: Iterable[Any]) -> list[float]:
    """Keep only the finite floats in ``values``."""
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(
    out_dir: str | Path,
    n_sessions: int,
    seed: int,
    mode: str = "synthetic",
    librispeech_root: str | Path | None = None,
    engine: str = "pyroomacoustics",
    sample_rate_hz: int = simulate.SAMPLE_RATE_HZ,
    rir_store_sec: float = simulate.RIR_STORE_SEC,
    allow_unbalanced: bool = False,
    masks: Iterable[str] = spec.EVAL_MASKS,
    quiet: bool = False,
) -> dict[str, Any]:
    """Generate the whole eval-dev set and return the manifest.

    Verification runs *before* any audio is rendered, so a plan that violates
    an invariant costs nothing.
    """
    masks = tuple(masks)
    out_dir = Path(out_dir)
    plan = simulate.build_eval_plan(
        n_sessions, seed=seed, masks=masks, allow_unbalanced=allow_unbalanced
    )
    reports = plan.reports
    warnings = collect_warnings(reports, engine, mode)

    source_pool = build_source_pool(plan, mode, librispeech_root, seed, sample_rate_hz)
    planned_split_of_position = {s.source.position_id: s.split for s in plan.sessions}

    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "rir").mkdir(parents=True, exist_ok=True)

    pruned = prune_stale_outputs(out_dir, [s.session_id for s in plan.sessions])
    if pruned and not quiet:
        print(
            f"removed {len(pruned)} stale session file(s) left by an earlier, larger run "
            f"in {out_dir}: {', '.join(pruned[:4])}"
            + (" ..." if len(pruned) > 4 else "")
        )

    meta_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any]] = {}
    rt60_requested: list[float] = []
    rt60_measured: list[float] = []
    rt60_pairs: list[tuple[float, float]] = []
    ild_spreads: list[float] = []
    t_eval_counts: list[int] = []

    for session in plan.sessions:
        source = source_pool[session.source.position_id]
        rir_info = simulate.compute_session_rirs(session, sample_rate_hz, engine)
        audio, render_info = simulate.render_session(
            session, rir_info["rirs"], source["signal"], sample_rate_hz
        )

        flags, _ = simulate.activity_grid(
            source["signal"],
            sample_rate_hz,
            render_info["offset_samples"],
            int(audio.shape[0]),
        )
        times = simulate.grid_times_sec()
        t_eval = [t for t, active in zip(times, flags) if active]
        if not t_eval:
            raise RuntimeError(
                f"session {session.session_id} has an empty T_eval; the utterance never passed "
                f"the VAD_REL_THRESHOLD_DB={simulate.VAD_REL_THRESHOLD_DB} dB gate"
            )

        audio_rel = f"audio/{session.session_id}.wav"
        audio_info = simulate.write_wav_pcm16(out_dir / audio_rel, audio, sample_rate_hz)

        rir_rel = f"rir/{session.session_id}.npz"
        simulate.save_rir_bundle(
            out_dir / rir_rel,
            rir_info["rirs"],
            {
                "session_id": session.session_id,
                "room_id": session.block.room_id,
                "acoustic_id": session.acoustic_id,
                "engine": rir_info["engine"],
                "rt60_requested_s": rir_info["rt60_requested_s"],
                "rt60_measured_s": rir_info["rt60_measured_s"],
                "mic_positions_xy_m": session.block.geometry.as_json(),
                "source_xy_m": list(session.source.xy),
            },
            store_sec=rir_store_sec,
            sample_rate_hz=sample_rate_hz,
        )

        for mask_bits in masks:
            meta_rows.append(
                make_meta_record(session, mask_bits, audio_info, audio_rel, sample_rate_hz)
            )
            truth_rows.append(
                make_ground_truth_record(session, mask_bits, rir_info, render_info, source, t_eval)
            )

        measured = _finite(rir_info["rt60_measured_s"])
        rt60_requested.append(rir_info["rt60_requested_s"])
        rt60_measured.extend(measured)
        if measured:
            rt60_pairs.append((rir_info["rt60_requested_s"], statistics.median(measured)))
        ild_spreads.extend(_finite([render_info["ild_spread_db"]]))
        t_eval_counts.append(len(t_eval))

        if not quiet:
            measured_text = f"{statistics.median(measured):.3f}s" if measured else "FAILED"
            print(
                f"  {session.session_id}  {session.block.room_id}/{session.absorption_level:<9} "
                f"az={session.source.azimuth_deg:+7.2f}deg(bin {session.source.azimuth_bin:2d})  "
                f"r={session.source.distance_m:4.2f}m(bin {session.source.distance_bin})  "
                f"rt60 req={rir_info['rt60_requested_s']:.3f}s meas={measured_text}  "
                f"T_eval={len(t_eval):2d}"
            )

    meta_path = out_dir / "meta.jsonl"
    truth_path = out_dir / "ground_truth.jsonl"
    write_jsonl(meta_path, meta_rows)
    write_jsonl(truth_path, truth_rows)

    for relpath in (
        [f"audio/{s.session_id}.wav" for s in plan.sessions]
        + [f"rir/{s.session_id}.npz" for s in plan.sessions]
        + ["meta.jsonl", "ground_truth.jsonl"]
    ):
        path = out_dir / relpath
        files[relpath] = {"sha256": simulate.sha256_file(path), "bytes": path.stat().st_size}

    rt60_errors = [abs(measured - requested) for requested, measured in rt60_pairs]
    manifest = {
        "spec_version": spec.SPEC_VERSION,
        "generator_version": simulate.GENERATOR_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "official": not warnings,
        "official_blockers": warnings,
        "track": TRACK,
        "seed": seed,
        "n_sessions": plan.n_sessions,
        "n_cases": plan.n_cases,
        "n_rooms": len(plan.blocks),
        "n_source_positions": len({s.source.position_id for s in plan.sessions}),
        "eval_masks": list(masks),
        "rir_engine": engine,
        "source_mode": mode,
        "librispeech_root": str(librispeech_root) if librispeech_root else None,
        "pruned_stale_files": pruned,
        # reports["split_isolation"] covers the *plan's* speaker slots, which are
        # disjoint by construction and therefore always look clean.  In
        # librispeech mode the talkers that actually produced the audio are
        # corpus speakers, so record them too or the manifest cannot be audited
        # for real speaker leakage.
        "source_speakers": {
            "n_distinct": len({e["speaker_id"] for e in source_pool.values()}),
            "ids": sorted({str(e["speaker_id"]) for e in source_pool.values()}),
            "splits_present": sorted(
                {
                    str(
                        (e.get("source_meta") or {}).get("split_of_speaker")
                        or planned_split_of_position.get(position_id, "")
                    )
                    for position_id, e in source_pool.items()
                }
            ),
            "n_utterances": len({str(e["utt_id"]) for e in source_pool.values()}),
        },
        "constants": simulate.local_constants(),
        "stratification": {
            "azimuth": reports["azimuth_stratification"],
            "distance": reports["distance_stratification"],
            "position_balance": reports["position_balance"],
            "exact_stratification_n_sessions": reports["exact_stratification_n_sessions"],
        },
        "confound_cells": reports["confound_cells"],
        "split_isolation": reports["split_isolation"],
        "plane_consistency": reports["plane_consistency"],
        "frame_invariance": reports["frame_invariance"],
        "start_offsets": reports["start_offsets"],
        "array_geometry_room_000": reports["array_geometry"],
        "case_ids": reports["case_ids"],
        "rt60_summary": {
            "requested_min_s": min(rt60_requested) if rt60_requested else None,
            "requested_max_s": max(rt60_requested) if rt60_requested else None,
            "measured_min_s": min(rt60_measured) if rt60_measured else None,
            "measured_max_s": max(rt60_measured) if rt60_measured else None,
            "mean_abs_request_error_s": (
                sum(rt60_errors) / len(rt60_errors) if rt60_errors else None
            ),
            "n_measured": len(rt60_measured),
        },
        "audio_summary": {
            "sample_rate_hz": sample_rate_hz,
            "session_length_sec": spec.SESSION_LENGTH_SEC,
            "n_channels": spec.N_CHANNELS_MAX,
            "ild_spread_db_min": min(ild_spreads) if ild_spreads else None,
            "ild_spread_db_max": max(ild_spreads) if ild_spreads else None,
            "t_eval_per_session_min": min(t_eval_counts) if t_eval_counts else None,
            "t_eval_per_session_max": max(t_eval_counts) if t_eval_counts else None,
            "n_grid_points": len(simulate.grid_times_sec()),
        },
        "files": files,
        "manifest_self_hash": None,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return manifest


def collect_warnings(reports: dict[str, Any], engine: str, mode: str) -> list[str]:
    """List every reason this set is not fit to carry official numbers.

    Returned as data rather than printed so the manifest records it: a set
    generated in a hurry stays usable for development while being impossible to
    mistake for the official one.
    """
    out: list[str] = []
    az = reports["azimuth_stratification"]
    dist = reports["distance_stratification"]
    if engine != "pyroomacoustics":
        out.append(
            f"rir_engine={engine!r} is not the official pyroomacoustics ISM engine; "
            "numbers from this set are not reportable"
        )
    if mode != "librispeech":
        out.append(f"source_mode={mode!r} uses placeholder speech, so WER is not meaningful")
    if not az["strict_uniform"]:
        out.append(
            f"azimuth labels occupy {az['n_used_bins']}/{az['k_bins']} bins, so an audio-blind "
            f"model gains {az['prior_only']['memorisation_gain_nats']:.3f} nats of azimuth NLL "
            f"by memorising the prior; strict uniformity needs n_sessions="
            f"{reports['exact_stratification_n_sessions']}"
        )
    if not dist["strict_uniform"]:
        out.append(
            f"distance labels occupy {dist['n_used_bins']}/{dist['k_bins']} bins "
            f"({dist['prior_only']['memorisation_gain_nats']:.3f} nats of memorisation gain)"
        )
    if not reports["confound_cells"]["complete"]:
        out.append(
            "the distance x absorption 2x2 design is incomplete, so distance and RT60 "
            "cannot be separated in this set"
        )
    if not reports["position_balance"]["balanced"]:
        out.append("source positions carry unequal session counts (allow_unbalanced was used)")
    return out


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write ``rows`` as one JSON object per line, keys sorted for stable hashes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def print_summary(manifest: dict[str, Any]) -> None:
    """Print the manifest's headline numbers and every official-use blocker."""
    az = manifest["stratification"]["azimuth"]
    dist = manifest["stratification"]["distance"]
    print(f"\nspec {manifest['spec_version']} / generator {manifest['generator_version']}")
    print(
        f"  {manifest['n_sessions']} sessions x {len(manifest['eval_masks'])} masks "
        f"= {manifest['n_cases']} cases, {manifest['n_rooms']} rooms, "
        f"{manifest['n_source_positions']} source positions"
    )
    print(f"  engine={manifest['rir_engine']}  source={manifest['source_mode']}  seed={manifest['seed']}")
    print(
        f"  azimuth  : {az['n_used_bins']}/{az['k_bins']} bins, "
        f"{az['min_count']}..{az['max_count']} positions/bin, strict={az['strict_uniform']}, "
        f"memorisation gain {az['prior_only']['memorisation_gain_nats']:.3f} nats"
    )
    print(
        f"  distance : {dist['n_used_bins']}/{dist['k_bins']} bins, "
        f"{dist['min_count']}..{dist['max_count']} positions/bin, strict={dist['strict_uniform']}, "
        f"memorisation gain {dist['prior_only']['memorisation_gain_nats']:.3f} nats"
    )
    print(f"  2x2 cells: {manifest['confound_cells']['counts']}")
    rt = manifest["rt60_summary"]
    if rt["measured_min_s"] is not None:
        print(
            f"  rt60     : requested {rt['requested_min_s']:.3f}..{rt['requested_max_s']:.3f} s, "
            f"measured {rt['measured_min_s']:.3f}..{rt['measured_max_s']:.3f} s "
            f"(mean |req-meas| {rt['mean_abs_request_error_s']:.3f} s, labels are the measured ones)"
        )
    audio = manifest["audio_summary"]
    if audio["ild_spread_db_min"] is not None:
        print(
            f"  audio    : T_eval {audio['t_eval_per_session_min']}.."
            f"{audio['t_eval_per_session_max']} of {audio['n_grid_points']} grid points, "
            f"inter-channel level spread {audio['ild_spread_db_min']:.2f}.."
            f"{audio['ild_spread_db_max']:.2f} dB"
        )
    print(f"  files    : {len(manifest['files'])} written, all SHA256'd")
    print(f"  official : {manifest['official']}")
    for blocker in manifest["official_blockers"]:
        print(f"    - {blocker}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and build the set. Returns a process exit status."""
    parser = argparse.ArgumentParser(
        description="Build the eval-dev prototype evaluation set for the VC-SASR benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--n-sessions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use placeholder synthetic speech instead of LibriSpeech (numpy only, no corpus)",
    )
    parser.add_argument("--librispeech-root", default=None, help="root of a LibriSpeech subset")
    parser.add_argument(
        "--rir-engine",
        choices=simulate.RIR_ENGINES,
        default="pyroomacoustics",
        help="pyroomacoustics is the only official engine; fallback_ism is numpy-only and "
        "stamps official=false",
    )
    parser.add_argument("--sample-rate", type=int, default=simulate.SAMPLE_RATE_HZ)
    parser.add_argument("--rir-store-sec", type=float, default=simulate.RIR_STORE_SEC)
    parser.add_argument(
        "--allow-unbalanced",
        action="store_true",
        help="permit a session count whose trailing block cannot be stratified",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="run every plan-layer invariant and exit without rendering audio (needs no numpy)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the per-session lines")
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except ValueError as exc:
        # Plan-level contract violations are user input errors, not crashes:
        # print the guidance the exception carries instead of a traceback.
        print(f"error: {exc}", file=sys.stderr)
        if "allow_unbalanced=True" in str(exc):
            print("hint: pass --allow-unbalanced to proceed anyway", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    """Execute the parsed command. Separated so :func:`main` can report errors."""
    if args.verify_only:
        simulate.self_test(
            args.n_sessions,
            args.seed,
            verbose=not args.quiet,
            allow_unbalanced=args.allow_unbalanced,
        )
        plan = simulate.build_eval_plan(
            args.n_sessions, seed=args.seed, allow_unbalanced=args.allow_unbalanced
        )
        blockers = collect_warnings(
            plan.reports, args.rir_engine, "synthetic" if args.synthetic else "librispeech"
        )
        print(f"\nverify-only: plan for {args.n_sessions} sessions is valid; no audio written")
        for blocker in blockers:
            print(f"  not official: {blocker}")
        return 0

    mode = "synthetic" if args.synthetic else "librispeech"
    if not args.quiet:
        print(f"generating {args.n_sessions} sessions into {args.out} (engine={args.rir_engine}, source={mode})")
    manifest = generate(
        out_dir=args.out,
        n_sessions=args.n_sessions,
        seed=args.seed,
        mode=mode,
        librispeech_root=args.librispeech_root,
        engine=args.rir_engine,
        sample_rate_hz=args.sample_rate,
        rir_store_sec=args.rir_store_sec,
        allow_unbalanced=args.allow_unbalanced,
        quiet=args.quiet,
    )
    if not args.quiet:
        print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

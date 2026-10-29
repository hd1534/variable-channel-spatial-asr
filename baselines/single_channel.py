"""Controlled Baseline 1: one microphone, frozen ASR, uninformative azimuth.

This is the *no spatial processing* control for the Variable-Channel Spatial ASR
benchmark.  It deliberately throws away every channel but one, so its azimuth
row is flat across ``M = 1, 2, 3, 4``.  Any gain the proposed model shows over
it has to come from using the extra channels, which is exactly the claim the
benchmark is meant to test.

What it does, and why each choice is the honest one
---------------------------------------------------
*Audio.*  Exactly one channel: the **lowest active channel index in the mask**.
For the six frozen :data:`spec.EVAL_MASKS` that is channel ``0`` for ``1000``,
``1100``, ``1010``, ``1110``, ``1111`` and channel ``2`` for ``0010``.  The
rule is positional, not energy-based, because an energy-based pick would make
the baseline weakly channel-selective and so no longer a clean control.

*Azimuth.*  The **uniform distribution**, always.  A single microphone records
no interchannel time or level difference, so with a fixed array geometry the
azimuth is not merely hard to estimate -- it is unobservable in principle.  The
uniform answer scores ``log K_theta`` nats of NLL and an expected circular
error of 90 degrees for any fixed point guess.  Claiming anything sharper would
be fabricating information, and omitting the prediction entirely would be
scored worse (:data:`spec.PENALTY_NLL`), so uniform is the floor the benchmark
should measure against.

*Distance.*  Distance *is* weakly observable from one channel, through received
level: a train-set prior, or a one-feature linear regression on log RMS.  Which
one was used is written into the output metadata, because a prior that happens
to match the eval distribution would otherwise look like a working estimator.
Level also tracks talker loudness and microphone gain, so this is a weak cue;
the emitted distribution is widened by the fitted residual spread rather than
being peaked on the regression's point estimate.

*Speaker.*  Track 1 is a single stationary speaker, so every scored instant is
``spk_0`` by construction (:data:`spec.MAX_SPEAKERS_PER_TRACK`).

*Transcription.*  A frozen pretrained CTC checkpoint, run once per energy-VAD
segment, never fine-tuned, shared verbatim with Baseline 2 through
:class:`baselines.FrozenASR`.  Nothing is downloaded: a missing checkpoint
exits with instructions.  ``--synthetic-asr`` runs the whole pipeline with
placeholder text so the submission path can be validated before the checkpoint
is provisioned.

Usage
-----
``python -m baselines.single_channel --eval-dir data/eval-dev --out predictions.jsonl
--scope causal --track 1``

``python -m baselines.single_channel --self-test`` runs the doctests plus an
end-to-end synthetic-audio smoke test with no data and no checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import spec
from baselines import (
    MIN_SIGMA_LOG,
    SYNTHETIC_ASR_WARNING,
    AudioSession,
    CaseSpec,
    CaseTimer,
    DistanceModel,
    FrozenASR,
    add_common_cli_args,
    check_fit_targets,
    warn_on_unphysical_level_coefficient,
    azimuth_centers,
    build_final_record,
    build_timestep_record,
    causal_window_bounds,
    discover_cases,
    distance_centers,
    energy_vad_segments,
    fit_ridge,
    inference_grid_sec,
    load_audio,
    load_distance_model,
    map_point,
    normalise_distribution,
    resolve_masks,
    resolve_scope,
    rms,
    speaker_slot,
    write_predictions,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from baselines import FloatArray
else:  # pragma: no cover
    FloatArray = Any

SYSTEM_NAME = "controlled_baseline_1_single_channel"

MIN_WINDOW_SEC = spec.DELTA_T_SEC
"""Shortest observation window that yields a usable level feature.

At ``t_0 = 0`` the causal window is empty, so there is nothing to measure and
the distance distribution falls back to the model's prior.
"""

LEVEL_FLOOR = spec.NLL_EPSILON
"""Floor applied inside ``log10`` so digital silence does not produce ``-inf``.

Reuses the contract's underflow floor rather than inventing a second epsilon.
"""


def select_channel(mask_bits: str) -> int:
    """Return the single channel this baseline reads: the lowest active index.

    >>> [select_channel(m) for m in spec.EVAL_MASKS]
    [0, 2, 0, 0, 0, 0]
    >>> select_channel("0001")
    3
    >>> select_channel("0000")
    Traceback (most recent call last):
        ...
    ValueError: mask '0000' activates no channel
    """
    channels = spec.mask_to_channels(mask_bits)
    if not channels:
        raise ValueError(f"mask {mask_bits!r} activates no channel")
    return min(channels)


def level_features(mono: FloatArray) -> list[float]:
    """Return the one-element feature vector ``[log10(rms)]``.

    Level is the only distance cue a single microphone offers, and it is
    confounded with talker loudness, directivity and recording gain.  The
    logarithm is taken because received level falls as ``1 / r`` in the free
    field, so ``log level`` is roughly linear in ``log r``.

    >>> import numpy as np
    >>> round(level_features(np.full(100, 0.1))[0], 4)
    -1.0
    >>> level_features(np.zeros(10))[0] < -10.0
    True
    """
    return [math.log10(max(rms(mono), LEVEL_FLOOR))]


def azimuth_prediction(
    mask_bits: str,
    prob_decimals: int | None,
) -> tuple[list[float], float]:
    """Return the uniform azimuth distribution and its MAP point, in degrees.

    The mask is accepted and validated but deliberately unused: this baseline
    reads one channel whatever the mask says, so the answer must not depend on
    ``M``.  That invariance is the control.

    >>> p, point = azimuth_prediction("1111", 6)
    >>> len(p) == spec.K_AZIMUTH
    True
    >>> p == azimuth_prediction("1000", 6)[0]
    True
    >>> point
    -175.0
    """
    spec.mask_to_channels(mask_bits)
    probs = normalise_distribution([1.0] * spec.K_AZIMUTH, decimals=prob_decimals)
    return probs, map_point(probs, azimuth_centers())


def distance_prediction(
    window: FloatArray,
    model: DistanceModel,
    prob_decimals: int | None,
) -> tuple[list[float], float]:
    """Return the distance distribution and point estimate for one window.

    Falls back to the model's unconditioned answer when the window is too
    short to measure a level, which happens at ``t_0 = 0`` in causal scope.

    >>> import numpy as np
    >>> probs, point = distance_prediction(np.zeros(0), DistanceModel(), 6)
    >>> len(probs) == spec.K_DISTANCE
    True
    >>> spec.DISTANCE_RANGE_M[0] <= point <= spec.DISTANCE_RANGE_M[1]
    True
    """
    import numpy as np

    mono = np.asarray(window, dtype=np.float64).ravel()
    if mono.size == 0 or model.kind != "regression":
        probs = normalise_distribution(
            model.predict_distribution([0.0] * max(1, len(model.feature_names))),
            decimals=prob_decimals,
        )
        return probs, map_point(probs, distance_centers())
    features = level_features(mono)
    probs = normalise_distribution(
        model.predict_distribution(features), decimals=prob_decimals
    )
    return probs, model.predict_point(features)


def transcribe_case(
    session: AudioSession,
    channel: int,
    asr: FrozenASR,
    session_length_sec: float = spec.SESSION_LENGTH_SEC,
) -> dict[str, str]:
    """Transcribe the selected channel and return ``{speaker_slot: text}``.

    The frozen ASR is run once per energy-VAD segment of at least
    :data:`spec.ASR_MIN_SEGMENT_SEC`, as the contract requires, and each
    segment is decoded from its own samples only -- so no segment transcript
    depends on audio past that segment's end, which is what makes this path
    causal without needing a scope argument.  The segment texts are
    concatenated in time order into the single Track 1 speaker slot.
    """
    end_sec = min(session.duration_sec, session_length_sec)
    mono = session.window(0.0, end_sec, (channel,))[0]
    pieces: list[str] = []
    for start, stop in energy_vad_segments(mono, session.sample_rate_hz):
        segment = session.window(start, min(stop, end_sec), (channel,))[0]
        if segment.size == 0:
            continue
        text = asr.transcribe(segment, session.sample_rate_hz).strip()
        if text:
            pieces.append(text)
    return {speaker_slot(0): " ".join(pieces)}


def predict_case(
    case: CaseSpec,
    session: AudioSession,
    asr: FrozenASR,
    distance_model: DistanceModel,
    scope: str,
    lookahead_sec: float,
    prob_decimals: int | None,
    include_record_type: bool,
    run_asr: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run Baseline 1 on one case and return ``(records, diagnostics)``.

    Emits one timestep record for every point of ``T_all`` plus the single
    ``final_transcripts`` record.  The evaluator filters down to ``T_eval``
    itself, so predicting on the whole grid is both required and sufficient.
    """
    channel = select_channel(case.mask_bits)
    if channel >= session.n_channels:
        raise ValueError(
            f"{case.case_id}: mask needs channel {channel} but "
            f"{case.audio_path} has {session.n_channels} channel(s)"
        )
    records: list[dict[str, Any]] = []
    azimuth_probs, azimuth_point = azimuth_prediction(case.mask_bits, prob_decimals)
    timer = CaseTimer()
    with timer:
        for eval_time_sec in inference_grid_sec():
            start_sec, end_sec = causal_window_bounds(
                eval_time_sec, scope, lookahead_sec=lookahead_sec
            )
            mono = session.window(start_sec, end_sec, (channel,))[0]
            if end_sec - start_sec < MIN_WINDOW_SEC:
                mono = mono[:0]
            distance_probs, distance_point = distance_prediction(
                mono, distance_model, prob_decimals
            )
            records.append(
                build_timestep_record(
                    case_id=case.case_id,
                    eval_time_sec=eval_time_sec,
                    speaker_id=speaker_slot(0),
                    azimuth_probs=azimuth_probs,
                    distance_probs=distance_probs,
                    pred_azimuth_deg=azimuth_point,
                    pred_distance_m=distance_point,
                    include_record_type=include_record_type,
                )
            )
        transcripts = (
            transcribe_case(session, channel, asr) if run_asr else {speaker_slot(0): ""}
        )
    audio_sec = min(session.duration_sec, spec.SESSION_LENGTH_SEC)
    meta: dict[str, Any] = {
        "selected_channel": channel,
        "channel_selection_rule": "lowest active channel index in mask",
        "n_active_channels": case.n_active_channels,
        "n_channels_used": 1,
        "azimuth_source": "uniform_unobservable_at_m1",
        "asr_backend": asr.backend_id,
        "asr_frozen": True,
        "observation_scope": scope,
        "lookahead_sec": lookahead_sec if scope == "windowed" else 0.0,
        "mask_bits": case.mask_bits,
        "geometry_source": case.geometry_source,
        "transcription_enabled": run_asr,
        "sample_rate_hz": session.sample_rate_hz,
        "audio_duration_sec": session.duration_sec,
    }
    meta.update(distance_model.describe())
    record = build_final_record(
        case_id=case.case_id,
        final_transcripts=transcripts,
        observation_mode=scope,
        reported_rtf=timer.rtf(audio_sec),
        system=SYSTEM_NAME,
        track=case.track,
        wer_reportable=run_asr and not asr.synthetic,
        include_record_type=include_record_type,
        meta=meta,
    )
    records.append(record)
    return records, {"rtf": record["reported_rtf"], "channel": channel}


# ---------------------------------------------------------------------------
# Optional distance fitting
# ---------------------------------------------------------------------------


def load_distance_labels(path: str | Path) -> dict[str, float]:
    """Read ``{session_id: distance_m}`` from a JSONL/JSON label file.

    Track 1 has one stationary speaker per session, so one distance per
    session is the whole label.  Keys may be ``session_id`` or ``case_id``
    (the mask part is dropped, since distance does not depend on the subset).
    """
    text = Path(path).read_text(encoding="utf-8")
    rows: list[Mapping[str, Any]]
    if Path(path).suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        rows = payload if isinstance(payload, list) else [payload]
    labels: dict[str, float] = {}
    for row in rows:
        key = row.get("session_id") or row.get("case_id")
        if key is None:
            continue
        session_id = str(key).split("__m")[0]
        for field_name in ("distance_m", "r_m", "distance", "range_m"):
            if field_name in row and row[field_name] is not None:
                labels[session_id] = float(row[field_name])
                break
    if not labels:
        raise ValueError(f"no distance labels recognised in {path}")
    return labels


def fit_distance_model(
    cases: Sequence[CaseSpec],
    labels: Mapping[str, float],
    audio_backend: str = "auto",
    alpha: float = 0.0,
) -> DistanceModel:
    """Fit ``log(distance) ~ a * log10(rms) + b`` on labelled sessions.

    One sample per distinct session (not per case), because every channel
    subset of a session shares one source distance.  ``alpha`` defaults to 0,
    i.e. ordinary least squares: with a single feature there is nothing to
    regularise, and Baseline 2 is the one that carries the ridge.

    The labels must come from a **training** split.  Fitting on ``eval-dev``
    and then scoring on it is label leakage and the resulting number is not a
    baseline result.
    """
    features: list[list[float]] = []
    targets: list[float] = []
    seen: set[str] = set()
    for case in cases:
        if case.session_id in seen or case.session_id not in labels:
            continue
        seen.add(case.session_id)
        session = load_audio(case.audio_path, audio_backend)
        channel = select_channel(case.mask_bits)
        mono = session.window(0.0, min(session.duration_sec, spec.SESSION_LENGTH_SEC), (channel,))[0]
        features.append(level_features(mono))
        targets.append(math.log(max(labels[case.session_id], spec.NLL_EPSILON)))
    if len(features) < 3:
        raise ValueError(
            f"need at least 3 labelled sessions to fit, got {len(features)}"
        )
    check_fit_targets(targets)
    coefficients, residual_std = fit_ridge(features, targets, alpha=alpha)
    warn_on_unphysical_level_coefficient(coefficients, ("log10_rms",))
    return DistanceModel(
        kind="regression",
        coefficients=tuple(coefficients),
        feature_names=("log10_rms",),
        sigma_log=max(residual_std, MIN_SIGMA_LOG),
        alpha=alpha,
        n_train=len(features),
        note="OLS on log10 RMS of the selected channel; level is confounded with talker gain",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for Baseline 1."""
    parser = argparse.ArgumentParser(
        prog="python -m baselines.single_channel",
        description=(
            "Controlled Baseline 1: lowest-index active channel, frozen ASR, "
            "uniform azimuth, level-based distance."
        ),
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--fit-distance-labels",
        default=None,
        help="JSONL of training-split {session_id, distance_m} to fit the level regression",
    )
    parser.add_argument(
        "--save-distance-model",
        default=None,
        help="write the fitted distance model to this JSON path",
    )
    return parser


def resolve_distance_model(args: argparse.Namespace, cases: Sequence[CaseSpec]) -> DistanceModel:
    """Pick the distance model from the CLI arguments, warning on the fallback."""
    if args.fit_distance_labels:
        labels = load_distance_labels(args.fit_distance_labels)
        model = fit_distance_model(cases, labels, args.audio_backend)
        print(
            f"note: fitted level->distance regression on {model.n_train} labelled "
            f"session(s), residual sigma(log) = {model.sigma_log:.4f}",
            file=sys.stderr,
        )
        if args.save_distance_model:
            Path(args.save_distance_model).write_text(
                json.dumps(model.to_json(), indent=2), encoding="utf-8"
            )
        return model
    if args.distance_model:
        model = load_distance_model(args.distance_model)
        print(f"note: distance model {model.kind!r} loaded from {args.distance_model}", file=sys.stderr)
        return model
    print(
        "note: no --distance-model and no --fit-distance-labels, so distance is the "
        "UNIFORM fallback (scores log K_r by construction). The output metadata "
        "records distance_source='uniform'.",
        file=sys.stderr,
    )
    return DistanceModel(kind="uniform", note="no prior and no fit supplied")


def run(args: argparse.Namespace) -> int:
    """Run Baseline 1 over an eval directory and write ``predictions.jsonl``."""
    scope = resolve_scope(args.scope)
    masks = resolve_masks(args.masks)
    cases = discover_cases(
        args.eval_dir, manifest=args.manifest, track=args.track, masks=masks, limit=args.limit
    )
    distance_model = resolve_distance_model(args, cases)
    run_asr = not args.no_asr
    asr = FrozenASR(model_name=args.asr_model, synthetic=args.synthetic_asr)
    if args.synthetic_asr and run_asr:
        print(SYNTHETIC_ASR_WARNING, file=sys.stderr)
    if args.no_asr:
        print(
            "note: --no-asr emits empty transcripts; spec.MISSING_TRANSCRIPT_POLICY "
            "scores every reference word as a deletion.",
            file=sys.stderr,
        )

    records: list[dict[str, Any]] = []
    sessions: dict[str, AudioSession] = {}
    for case in cases:
        if case.audio_path not in sessions:
            sessions.clear()  # one session in memory at a time
            sessions[case.audio_path] = load_audio(case.audio_path, args.audio_backend)
        session = sessions[case.audio_path]
        if abs(session.duration_sec - spec.SESSION_LENGTH_SEC) > spec.DELTA_T_SEC:
            print(
                f"warning: {case.case_id} audio is {session.duration_sec:.2f}s but "
                f"spec.SESSION_LENGTH_SEC is {spec.SESSION_LENGTH_SEC}s; the grid "
                "follows the spec and sample indices are clamped to the file.",
                file=sys.stderr,
            )
        case_records, _ = predict_case(
            case=case,
            session=session,
            asr=asr,
            distance_model=distance_model,
            scope=scope,
            lookahead_sec=args.lookahead_sec,
            prob_decimals=args.prob_decimals,
            include_record_type=not args.omit_record_type,
            run_asr=run_asr,
        )
        records.extend(case_records)
        print(f"  {case.case_id}: {len(case_records) - 1} timestep records", file=sys.stderr)
    written = write_predictions(args.out, records)
    print(
        f"{SYSTEM_NAME}: {len(cases)} case(s), {written} record(s) -> {args.out}",
        file=sys.stderr,
    )
    if args.synthetic_asr and run_asr:
        print(SYNTHETIC_ASR_WARNING, file=sys.stderr)
    return 0


def self_test() -> int:
    """Run doctests plus an end-to-end synthetic smoke test; return exit status.

    The smoke test writes a 4-channel PCM WAV to a temporary directory, runs
    the full CLI path over it with ``--synthetic-asr``, and checks the output
    against the contract: one timestep record per grid point per case, exactly
    one final record per case, no duplicate keys, correct vector lengths and
    probability sums inside :data:`spec.PROB_SUM_TOLERANCE`.
    """
    import doctest
    import tempfile
    import wave

    import numpy as np

    import baselines

    failures = 0
    for module in (baselines, sys.modules[__name__]):
        result = doctest.testmod(module, verbose=False)
        failures += result.failed
        print(f"doctests {module.__name__}: {result.attempted - result.failed}/{result.attempted}")
    if failures:
        return 1

    sample_rate = 16000
    rng = np.random.default_rng(0)
    n = int(spec.SESSION_LENGTH_SEC * sample_rate)
    speech = rng.normal(0.0, 0.05, n)
    speech[: n // 4] = 0.0  # a leading silence so the VAD has something to find
    multi = np.tile(speech, (spec.N_CHANNELS_MAX, 1))
    pcm = np.clip(multi * 32767.0, -32768, 32767).astype(np.int16).T.ravel()

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "sess_000.wav"
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(spec.N_CHANNELS_MAX)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
        out_path = Path(tmp) / "predictions.jsonl"
        parser = build_parser()
        args = parser.parse_args(
            [
                "--eval-dir",
                tmp,
                "--out",
                str(out_path),
                "--scope",
                spec.OFFICIAL_SCOPES[0],
                "--track",
                str(min(spec.MAX_SPEAKERS_PER_TRACK)),
                "--synthetic-asr",
            ]
        )
        status = run(args)
        if status != 0:
            print("FAIL: run() returned non-zero")
            return 1
        lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    grid = inference_grid_sec()
    timesteps = [r for r in lines if r["record_type"] == spec.RECORD_TYPE_TIMEPOINT]
    finals = [r for r in lines if r["record_type"] == spec.RECORD_TYPE_FINAL_TRANSCRIPTS]
    checks: list[tuple[str, bool]] = [
        (
            f"one final record per case ({len(spec.EVAL_MASKS)})",
            len(finals) == len(spec.EVAL_MASKS),
        ),
        (
            f"|T_all| x cases timestep records ({len(grid) * len(spec.EVAL_MASKS)})",
            len(timesteps) == len(grid) * len(spec.EVAL_MASKS),
        ),
        (
            "no duplicate (case_id, eval_time_sec)",
            len({(r["case_id"], r["eval_time_sec"]) for r in timesteps}) == len(timesteps),
        ),
        ("no duplicate case_id among finals", len({r["case_id"] for r in finals}) == len(finals)),
        (
            "azimuth vectors have K_azimuth entries",
            all(len(r["azimuth_probs"]) == spec.K_AZIMUTH for r in timesteps),
        ),
        (
            "distance vectors have K_distance entries",
            all(len(r["distance_probs"]) == spec.K_DISTANCE for r in timesteps),
        ),
        (
            "every distribution sums to 1 within tolerance",
            all(
                abs(math.fsum(r[key]) - 1.0) < spec.PROB_SUM_TOLERANCE
                for r in timesteps
                for key in ("azimuth_probs", "distance_probs")
            ),
        ),
        (
            "azimuth is uniform for every mask (the control invariant)",
            len({tuple(r["azimuth_probs"]) for r in timesteps}) == 1,
        ),
        (
            "single speaker slot per case",
            all(len(r["final_transcripts"]) == 1 for r in finals),
        ),
        (
            "synthetic runs are flagged not reportable",
            all(r["wer_reportable"] is False for r in finals),
        ),
        (
            "grid matches spec (endpoint excluded)",
            {r["eval_time_sec"] for r in timesteps} == set(grid),
        ),
        (
            "channel 2 is selected for mask 0010, channel 0 otherwise",
            all(
                r["meta"]["selected_channel"]
                == (2 if spec.parse_case_id(r["case_id"])[1] == "0010" else 0)
                for r in finals
            ),
        ),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print(f"baseline 1 self-test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    parser = build_parser()
    return run(parser.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

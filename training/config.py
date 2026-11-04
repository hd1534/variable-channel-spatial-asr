"""Run configuration for the speaker-attribution training scaffold.

One :class:`RunConfig` describes a run completely.  ``configs/*.json`` hold
only the fields a model changes; everything else comes from the defaults below,
so the defaults *are* the documentation.  The resolved config is written next to
every checkpoint, and a resumed run refuses to continue under a different one
(:func:`config_fingerprint`), because a checkpoint trained on one data recipe
and finished on another reports a number that belongs to neither.

Pure standard library, so ``python3 -m doctest training/config.py`` runs on a
bare interpreter.

>>> cfg = load_config_dict({"name": "x", "model": {"name": "voice_only"}})
>>> cfg.model.name, cfg.data.n_speakers, cfg.slots.max_slots
('voice_only', (2, 4), 4)
>>> load_config_dict({"name": "x", "model": {"nmae": "voice_only"}})
Traceback (most recent call last):
    ...
ValueError: unknown key 'nmae' in section 'model'
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import spec

SYSTEM_NAMES: tuple[str, ...] = ("voice_only", "space_only", "naive_fusion", "proposed")
"""The comparison ladder of the page (2.1): two single-cue systems, their
fixed-weight sum, and the learned fusion."""

STATE_MODES: tuple[str, ...] = ("carry", "reset")
"""Proposed model only: ``carry`` = Carry-Over, ``reset`` = Chunk-Reset."""

VAL_PROTOCOL_VERSION: int = 2
"""Validation protocol version recorded in ``summary.json`` and ``best.pt``
(v2 = official 2-talker, 2-slot cap for checkpoint selection, threshold tuning
and fusion/fixed-gate weight search)."""


@dataclass
class DataConfig:
    """Where speech and rooms come from, and how a training session is laid out."""

    train_root: str = "LibriSpeech/train-clean-100"
    """Training speakers, relative to ``--data-root``.  Disjoint from dev-clean,
    so no evaluation voice is ever seen in training."""

    val_root: str = "LibriSpeech/dev-clean"
    """Validation speakers: the dev-clean speakers *outside* the eval split of
    :func:`simulate.assign_splits` (seed :data:`EVAL_SPLIT_SEED`)."""

    synthetic: bool = False
    """Placeholder voices from :func:`simulate.synthesise_utterance` instead of
    LibriSpeech.  For smoke runs only: there is no speaker identity to learn."""

    n_rooms_train: int = 800
    n_rooms_val: int = 64
    rir_engine: str = "pyroomacoustics"
    rir_sec: float = 1.0
    """Stored RIR length.  pyroomacoustics' realised decay is slower than the
    requested RT60 (T30 1.1-1.5 s for a requested 0.85 s), so at 1.0 s the tail
    is cut 37-54 dB below its start in the most reverberant rooms (measured on 30
    rooms of the training bank).  The eval generator never truncates."""

    bank_variants: int = 3
    jitter_m: tuple[float, float] = (0.10, 0.30)
    """Each talker position comes with ``bank_variants - 1`` copies displaced by
    this much, and every training turn uses one of them (see
    :class:`training.rirbank.BankSpec`)."""

    rt60_range_sec: tuple[float, float] = (0.15, 0.85)
    """Requested Sabine RT60, uniform.  Spans both absorption levels of
    :mod:`simulate` (low 0.15-0.30 s, high 0.55-0.85 s)."""

    n_speakers: tuple[int, int] = (2, 4)
    """Talkers per session in the dense layout, uniform over the inclusive range."""

    turns_per_speaker: tuple[int, int] = (2, 3)
    """Dense layout: every talker returns at least twice, in random order (the
    same talker may go twice in a row)."""

    turn_sec: tuple[float, float] = (1.0, 4.0)
    eval_layout_prob: float = 0.5
    """Share of training sessions laid out like the planned two-talker eval:
    two talkers, one near (< 1.9 m) and one far, 2-4 turns of :attr:`eval_turn_sec`.  The
    rest use the dense layout above.  Without long turns a Carry-Over GRU never
    sees a single talker hold the floor for more than 4 s."""

    eval_turn_sec: tuple[float, float] = (2.0, 13.0)
    colocate_prob: float = 0.25
    """Share of training sessions in which a later talker speaks from an earlier
    talker's position.  The spatial branch is trained to tell *positions* apart,
    so these sessions make it ignore voice: two talkers at one position must
    look the same to it."""

    noise_snr_db: tuple[float, float] | None = (0.0, 20.0)
    """Spatially diffuse pink noise added to training sessions at this SNR
    (uniform), or ``None`` for none."""

    gap_sec: tuple[float, float] = (0.30, 0.80)
    """Silence between turns; same range as the multi-speaker eval generator."""

    lead_in_sec: tuple[float, float] = (0.25, 1.50)
    speaker_gain_db: float = 3.0
    """Each talker's level is drawn once per session from +-this many dB around
    a common dry level, so loudness is consistent within a session but carries
    no identity across sessions."""

    m_weights: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0)
    """Sampling weight of M = 1, 2, 3, 4 active channels during training.  Every
    channel subset of a given size is equally likely, not only spec.EVAL_MASKS.
    M = 1 is left out: the proposed model's spatial weight is zero there by
    construction, so such batches would teach nothing.  Validation still
    reports M = 1."""

    val_n_speakers: tuple[int, int] = (2, 2)
    """Talkers per validation session in the dense layout.  The eval-like layout
    always has two (like the planned eval; the scoring contract caps a case at
    two talkers), so looking at H2 internally needs this raised *and*
    ``val_eval_layout_prob`` below 1."""

    val_eval_layout_prob: float = 1.0
    """Validation follows the two-talker eval set: its main condition (displaced
    turns, diffuse noise at one of ``val_snr_levels_db``) is what thresholds and
    the best checkpoint are chosen on; the same sessions at rest and without
    noise are reported as the static condition."""

    val_snr_levels_db: tuple[float, ...] = (5.0, 10.0, 20.0)

    n_val_sessions: int = 200
    val_masks: tuple[str, ...] = spec.EVAL_MASKS
    val_seed: int = 7
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Which system of the ladder, and its hyperparameters."""

    name: str = "proposed"
    state: str = "carry"
    voice_encoder: str = "ecapa"
    """``ecapa`` (frozen speechbrain/spkrec-ecapa-voxceleb) or ``stub`` (a fixed
    random projection of log-spectrum statistics; offline smoke runs only)."""

    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    ecapa_batch: int = 256
    n_fft: int = 512
    hop: int = 160
    d_model: int = 256
    n_tac_blocks: int = 3
    gru_layers: int = 2
    d_spatial: int = 128
    srp_temperature: float = 1.0
    """Space-only and naive fusion: softmax temperature of the SRP-PHAT response,
    the same knob as Baseline 2's ``--srp-temperature``."""

    fusion_weight: float | None = None
    """Naive fusion only.  ``None`` = tune the spatial weight per M on validation."""


@dataclass
class TrainConfig:
    """Optimisation.  Ignored by systems without trainable parameters."""

    batch_size: int = 16
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-2
    max_steps: int = 20000
    warmup_steps: int = 500
    grad_clip: float = 5.0
    position_loss_weight: float = 1.0
    """Proposed model: the spatial branch learns only from a position-identity
    loss (labels = talker positions, so two talkers at one position share a
    label); the speaker loss reaches only the gate."""
    contrast_weight: float = 0.5
    """Weight of the calibrated same/different term added to both losses, so the
    similarity scale -- which the slot threshold reads -- is trained too."""
    amp: bool = True
    val_every: int = 1000
    ckpt_every: int = 500
    ckpt_every_sec: float = 600.0
    """Also checkpoint when this much wall time has passed since the last one, so
    a slow runtime never loses more than about ten minutes."""
    ckpt_keep: int = 3
    log_every: int = 50
    max_consecutive_skips: int = 20
    """Abort if this many consecutive batches have non-finite loss or gradients,
    instead of spinning forever on an infinite stream."""
    seed: int = 20261030


@dataclass
class SlotConfig:
    """The shared inference backend: online cosine slot matching."""

    max_slots: int = 4
    """Most slots a case may open during validation (one per possible talker).
    The official prediction writer caps it further at
    ``spec.MAX_SPEAKERS_PER_TRACK[track]`` (2 today) and uses the threshold
    tuned on two-talker sessions under that cap."""

    theta_grid: tuple[float, float, float] = (-0.30, 0.95, 0.01)
    """``(start, stop, step)`` of the join threshold searched on validation."""

    weight_grid: tuple[float, float, float] = (0.0, 1.0, 0.1)
    """Naive fusion: spatial weight searched on validation."""

    vad_rel_db: float = -30.0
    """Causal VAD: a grid instant is speech when the reference channel's RMS
    over ``[t_k - 0.25, t_k)`` is within this many dB of the running maximum."""

    vad_floor_dbfs: float = -60.0


@dataclass
class RunConfig:
    name: str
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    slots: SlotConfig = field(default_factory=SlotConfig)

    def validate(self) -> "RunConfig":
        if self.model.name not in SYSTEM_NAMES:
            raise ValueError(f"model.name must be one of {SYSTEM_NAMES}, got {self.model.name!r}")
        if self.model.state not in STATE_MODES:
            raise ValueError(f"model.state must be one of {STATE_MODES}, got {self.model.state!r}")
        for key in ("n_speakers", "val_n_speakers"):
            lo, hi = getattr(self.data, key)
            if not 2 <= lo <= hi <= 4:
                raise ValueError(f"data.{key} must satisfy 2 <= lo <= hi <= 4 (one room holds four sources), got {(lo, hi)}")
        if self.data.bank_variants < 1:
            raise ValueError("data.bank_variants must be >= 1")
        if self.slots.max_slots < 1:
            raise ValueError("slots.max_slots must be >= 1")
        if len(self.data.m_weights) != spec.N_CHANNELS_MAX or sum(self.data.m_weights) <= 0:
            raise ValueError("data.m_weights needs one non-negative weight per M = 1..4")
        for mask in self.data.val_masks:
            spec.mask_to_channels(mask)
        return self


EVAL_SPLIT_SEED: int = 20261029
"""Seed of the committed eval-dev speaker split (``make_eval_dev.py --seed``)."""

_SECTIONS = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig, "slots": SlotConfig}


def _coerce(value: Any, default: Any) -> Any:
    """JSON has no tuples; restore them where the default is one."""
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def load_config_dict(raw: dict[str, Any]) -> RunConfig:
    """Build a validated :class:`RunConfig` from a JSON-shaped dict."""
    if "name" not in raw:
        raise ValueError("a config needs a 'name'")
    unknown = set(raw) - {"name", *list(_SECTIONS)}
    if unknown:
        raise ValueError(f"unknown top-level key(s) {sorted(unknown)}")
    sections = {}
    for key, cls in _SECTIONS.items():
        given = raw.get(key, {})
        defaults = cls()
        for k in given:
            if not hasattr(defaults, k):
                raise ValueError(f"unknown key {k!r} in section {key!r}")
        sections[key] = cls(**{k: _coerce(v, getattr(defaults, k)) for k, v in given.items()})
    return RunConfig(name=str(raw["name"]), **sections).validate()


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Read ``configs/<name>.json`` and apply ``{"section.key": value}`` overrides."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        if not key:
            raise ValueError(f"override {dotted!r} must look like 'section.key'")
        raw.setdefault(section, {})[key] = value
    return load_config_dict(raw)


def config_to_dict(cfg: RunConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def config_fingerprint(cfg: RunConfig) -> str:
    """Hash of everything that changes what is learned.

    ``num_workers``, logging cadence and ``max_steps`` are left out on purpose:
    a run may be resumed with more workers or extended to more steps.

    >>> a = load_config_dict({"name": "a"})
    >>> b = load_config_dict({"name": "a", "train": {"max_steps": 5}, "data": {"num_workers": 0}})
    >>> config_fingerprint(a) == config_fingerprint(b)
    True
    >>> config_fingerprint(a) == config_fingerprint(load_config_dict({"name": "a", "train": {"lr": 0.1}}))
    False
    """
    d = config_to_dict(cfg)
    d["data"].pop("num_workers", None)
    for k in ("max_steps", "val_every", "ckpt_every", "ckpt_every_sec", "ckpt_keep", "log_every", "max_consecutive_skips"):
        d["train"].pop(k, None)
    blob = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]

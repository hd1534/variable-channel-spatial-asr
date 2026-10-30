"""DataLoader plumbing: an endless training stream and a fixed validation set.

Training sessions are drawn on the fly in the loader workers; each worker opens
the room bank itself (memory-mapped) and seeds its own RNG from
``(seed, start_step, worker)``, so a resumed run continues with fresh sessions
instead of replaying the ones it already saw.  Workers hand over the dry talkers
and their room responses; the room itself is applied on the training device
(:mod:`training.render`), where it is two orders of magnitude cheaper.

Validation sessions are a pure function of ``(val_seed, index)``: every
validation pass, every system and every resumed run sees exactly the same
audio.  They are delivered with all four channels; the evaluation loop applies
each mask of ``data.val_masks`` itself.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

import spec
from training.rirbank import RirBank
from training.scenes import Session, draw_session, sample_mask, train_recipe, val_recipe


def mask_array(mask_bits: str) -> np.ndarray:
    return np.array([c == "1" for c in mask_bits], dtype=bool)


def _item(session: Session, bank: RirBank, mask_bits: str, dry_dtype: Any = np.float16) -> dict[str, Any]:
    """Everything :func:`training.render.render_batch` needs, padded to four talkers."""
    n = session.n_speakers
    dry = np.zeros((spec.N_CHANNELS_MAX, session.dry.shape[1]), dtype=dry_dtype)
    dry[:n] = session.dry
    variant = np.zeros((spec.N_CHANNELS_MAX, session.dry.shape[1]), dtype=np.int8)
    variant[:n] = session.variant_map
    rirs = bank.rirs(session.room_index, session.sources)
    rir = np.zeros((spec.N_CHANNELS_MAX, *rirs.shape[1:]), dtype=np.float16)
    rir[:n] = rirs
    return {
        "dry": dry,
        "variant": variant,
        "rir": rir,
        "target_rms": np.float32(session.target_rms),
        "channel_gain": session.channel_gain,
        "snr_db": np.float32(session.snr_db),
        "noise_seed": np.int64(session.noise_seed),
        "mask": mask_array(mask_bits),
        "mic_xy": session.mic_xy,
        "labels": session.labels,
        "pos_labels": session.pos_labels,
        "labels_eval": session.labels_eval,
        "n_speakers": session.n_speakers,
    }


class TrainStream(IterableDataset):
    def __init__(self, data_cfg: Any, corpus: Any, bank_path: str, seed: int, start_step: int = 0) -> None:
        super().__init__()
        self.cfg, self.corpus, self.bank_path = data_cfg, corpus, bank_path
        self.seed, self.start_step = seed, start_step
        self.recipe = train_recipe(data_cfg)

    def __iter__(self):  # noqa: ANN204
        info = get_worker_info()
        worker = info.id if info else 0
        rng = random.Random(f"train:{self.seed}:{self.start_step}:{worker}")
        bank = RirBank(self.bank_path)
        while True:
            session = draw_session(rng, self.corpus, bank, self.cfg, self.recipe)
            yield _item(session, bank, sample_mask(rng, self.cfg.m_weights))


class ValSessions(Dataset):
    """Validation sessions under one condition (``kind`` of :func:`training.scenes.val_recipe`)."""

    def __init__(self, data_cfg: Any, corpus: Any, bank_path: str, kind: str = "main") -> None:
        self.cfg, self.corpus, self.bank_path, self.kind = data_cfg, corpus, bank_path, kind
        self.recipe = val_recipe(data_cfg, kind)
        self._bank: RirBank | None = None

    def __len__(self) -> int:
        return self.cfg.n_val_sessions

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._bank is None:
            self._bank = RirBank(self.bank_path)
        session = draw_session(random.Random(f"val:{self.cfg.val_seed}:{index}"), self.corpus, self._bank, self.cfg,
                               self.recipe)
        item = _item(session, self._bank, "1" * spec.N_CHANNELS_MAX, dry_dtype=np.float32)
        item["index"] = index
        item["meta"] = {
            "speaker_ids": session.speaker_ids,
            "source_azimuth_deg": session.source_azimuth_deg,
            "source_distance_m": session.source_distance_m,
            "room_index": session.room_index,
            "rt60_requested_s": session.rt60_requested_s,
            "layout": session.layout,
            "n_turns": len(session.turns),
            "n_changes": session.n_changes,
            "snr_db": None if session.snr_db != session.snr_db else round(session.snr_db, 2),
        }
        return item


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in items[0]:
        values = [it[key] for it in items]
        if key == "meta":
            out[key] = values
        else:
            out[key] = torch.as_tensor(np.stack([np.asarray(v) for v in values]))
    return out


def with_mask(batch: dict[str, Any], mask_bits: str) -> dict[str, Any]:
    """The same (rendered) sessions seen through another channel subset."""
    mask = torch.as_tensor(mask_array(mask_bits), device=batch["audio"].device)
    out = dict(batch)
    out["mask"] = mask[None].expand(batch["audio"].shape[0], -1).clone()
    out["audio"] = batch["audio"] * mask[None, :, None].to(batch["audio"].dtype)
    return out


def to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

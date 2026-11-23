"""Speaker-attribution systems of the comparison ladder (page 2.1).

Every system maps a batch to one unit-norm embedding per grid instant,
``embed(batch) -> [B, 80, D]``, and hands it to the same backend
(:mod:`models.slots`: online cosine slot matching).  The systems differ only in
what goes into the embedding:

==============  ============================================  =========
name            embedding                                     trained
==============  ============================================  =========
voice_only      frozen ECAPA of the reference channel          no
space_only      sqrt of SRP-PHAT 36-bin direction distribution no
naive_fusion    both, mixed with one fixed weight              no (w tuned)
proposed        TAC spatial branch + ECAPA, learned gate + GRU yes
track_a         log-mel + pair GCC-PHAT, Conformer from scratch yes
==============  ============================================  =========

``proposed`` with ``model.state = "carry"`` / ``"reset"`` are the Carry-Over and
Chunk-Reset arms.  The H4 ablation "proposed without the spatial path" is the
gate held at zero, which is voice_only itself.
"""

from __future__ import annotations

from typing import Any

from models.naive_fusion import NaiveFusion
from models.proposed import Proposed
from models.space_only import SpaceOnly
from models.track_a import TrackA
from models.voice_only import VoiceOnly

SYSTEMS = {
    "voice_only": VoiceOnly,
    "space_only": SpaceOnly,
    "naive_fusion": NaiveFusion,
    "proposed": Proposed,
    "track_a": TrackA,
}

VOICE_SYSTEMS: frozenset[str] = frozenset({"voice_only", "naive_fusion", "proposed"})
"""Systems that read the frozen ECAPA voice cue; Track A trains its own inputs."""


def needs_voice(cfg: Any) -> bool:
    return cfg.model.name in VOICE_SYSTEMS


def build_system(cfg: Any, voice: Any = None):  # noqa: ANN201
    return SYSTEMS[cfg.model.name](cfg, voice)

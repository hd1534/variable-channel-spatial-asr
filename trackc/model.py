"""Track C model: Qwen3-ASR-1.7B + a microphone-pair spatial adapter + a frame head, in one forward pass.

Runs in the Qwen venv (``qwen-asr==0.0.6``, ``peft``; :mod:`setup_asr_env` profile ``trackc``).

* **Audio**: the reference channel (lowest active microphone) goes through the
  stock processor and the frozen AuT encoder, as for the shared transcriber.
* **Space**: :class:`models.track_a.PairSpatial` (GCC-PHAT of every active pair,
  TAC across pairs) is averaged onto the audio-token grid (12.5 tokens/s) and
  added to the encoder output right where it replaces the prompt's audio
  placeholders (a wrapper around ``thinker.get_audio_features``; the library is
  not modified).  The last layer is zero-initialised, so training starts from
  the original transcriber, and with one microphone there is no pair and
  nothing is added.  ``use_spatial = False`` (the ``track_c_mono`` control)
  leaves the adapter out entirely.
* **Decoder**: LoRA on every linear layer of the Qwen3 text decoder; the encoder
  stays frozen.  It writes ``[S1] ... [S2] ...`` (:mod:`trackc.text`).
* **Frame head**: a linear layer on the decoder's last hidden state at the audio
  positions, pooled onto the 0.25 s grid: silence / first talker / second talker.
  Same numbering as the tags (first appearance), so the outputs need no matching.

The encoder attends within 1-8 s windows in both directions and the decoder
reads the whole session before writing, so Track C is a ``full_audio`` system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import spec
from models.common import reference_channel
from models.track_a import PairSpatial

PROMPT_LANGUAGE = "English"
N_CLASSES = 3  # silence, first talker, second talker
LORA_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class TrackCConfig:
    name: str = "track_c"
    use_spatial: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    d_spatial: int = 256
    gcc_max_lag: int = 16
    stack: int = 4
    n_pair_tac_blocks: int = 2
    head_weight: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 6000
    batch_size: int = 4
    grad_accum: int = 2
    grad_clip: float = 1.0
    max_new_tokens: int = 400
    val_every: int = 1000
    val_sessions: int = 40
    val_masks: tuple[str, ...] = ("1111", "1100", "1000")
    final_val_sessions: int = 200
    ckpt_every: int = 250
    ckpt_every_sec: float = 600.0
    log_every: int = 20
    seed: int = 20261123
    data: dict = field(default_factory=lambda: {"m_weights": [1.0, 1.0, 1.0, 1.0]})
    """Overrides of :class:`training.config.DataConfig` (rooms, noise, masks); the defaults are the other systems'."""

    @classmethod
    def from_dict(cls, raw: dict) -> "TrackCConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown Track C config key(s) {sorted(unknown)}")
        values = dict(raw)
        if "val_masks" in values:
            values["val_masks"] = tuple(values["val_masks"])
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


# -- audio tokens and the grid ------------------------------------------------------
def audio_token_count(n_mel_frames: int) -> int:
    """Qwen3-ASR's ``_get_feat_extract_output_lengths`` for one input (13 tokens per 100 mel frames).

    >>> audio_token_count(2000), audio_token_count(300), audio_token_count(150)
    (260, 39, 20)
    """
    rest = n_mel_frames % 100
    feat = (rest - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (n_mel_frames // 100) * 13


def token_centres_sec(n_mel_frames: int) -> np.ndarray:
    """Time of each audio token's centre: the encoder works in 1 s chunks of 100 mel frames
    (13 tokens each); a final partial chunk spreads its tokens over what is left.

    >>> t = token_centres_sec(2000); len(t), round(float(t[0]), 4), round(float(t[13]), 4)
    (260, 0.0385, 1.0385)
    """
    full, rest = divmod(n_mel_frames, 100)
    centres = [c + (i + 0.5) / 13 for c in range(full) for i in range(13)]
    tail = audio_token_count(n_mel_frames) - 13 * full
    centres += [full + (i + 0.5) * (rest / 100) / max(tail, 1) for i in range(tail)]
    return np.asarray(centres, dtype=np.float64)


def grid_pooling(n_mel_frames: int) -> torch.Tensor:
    """``[80, n_tokens]`` averaging the tokens centred in each official window ``[t_k, t_k + 0.25)``
    (the nearest token where none is).

    >>> W = grid_pooling(2000); tuple(W.shape), bool(torch.allclose(W.sum(1), torch.ones(80)))
    ((80, 260), True)
    """
    centres = token_centres_sec(n_mel_frames)
    W = np.zeros((spec.N_GRID_POINTS, len(centres)), dtype=np.float32)
    for k in range(spec.N_GRID_POINTS):
        lo, hi = k * spec.DELTA_T_SEC, (k + 1) * spec.DELTA_T_SEC
        inside = np.flatnonzero((centres >= lo) & (centres < hi))
        if not len(inside):
            inside = np.array([int(np.argmin(np.abs(centres - (lo + hi) / 2)))])
        W[k, inside] = 1.0 / len(inside)
    return torch.from_numpy(W)


# -- the model -------------------------------------------------------------------------
class SpatialAdapter(nn.Module):
    """Pair GCC-PHAT features averaged onto the audio tokens and projected to the decoder width."""

    def __init__(self, cfg: TrackCConfig, hidden: int) -> None:
        super().__init__()
        d = cfg.d_spatial
        self.pairs = PairSpatial(512, 160, cfg.stack, cfg.gcc_max_lag, d, cfg.n_pair_tac_blocks)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, hidden))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, audio: torch.Tensor, mask: torch.Tensor, mic_xy: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """``[B, n_tokens, hidden]``; zero where fewer than two microphones are active."""
        frames = self.pairs(self.pairs.stft(audio), mask, mic_xy)  # [B, G, d]
        pooled = F.adaptive_avg_pool1d(frames.transpose(1, 2), n_tokens).transpose(1, 2)
        return self.proj(pooled) * (mask.sum(1) >= 2)[:, None, None].to(pooled.dtype)


class UnifiedASR(nn.Module):
    """Wraps a loaded ``Qwen3ASRForConditionalGeneration`` (``outer``) and its processor."""

    def __init__(self, outer: Any, processor: Any, cfg: TrackCConfig) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model

        self.cfg = cfg
        self.processor = processor
        self.outer = outer
        self.thinker = outer.thinker
        for p in outer.parameters():
            p.requires_grad_(False)
        self.thinker.model = get_peft_model(self.thinker.model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=list(LORA_TARGETS)))
        hidden = self.thinker.config.text_config.hidden_size
        self.audio_token_id = int(self.thinker.config.audio_token_id)
        self.spatial = SpatialAdapter(cfg, hidden) if cfg.use_spatial else None
        self.head = nn.Linear(hidden, N_CLASSES)
        self._extra: torch.Tensor | None = None
        encode = self.thinker.get_audio_features

        def with_space(input_features, feature_attention_mask=None, audio_feature_lengths=None):  # noqa: ANN001, ANN202
            out = encode(input_features, feature_attention_mask=feature_attention_mask,
                         audio_feature_lengths=audio_feature_lengths)
            if self._extra is not None:
                out = out + self._extra.reshape(-1, out.shape[-1]).to(out.dtype)
            return out

        self.thinker.get_audio_features = with_space

    # -- state that training owns ----------------------------------------------------
    def trainable_state(self) -> dict[str, torch.Tensor]:
        """Only what training changes: LoRA, the adapter and the head (the base weights are pinned)."""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    def load_trainable(self, state: dict[str, torch.Tensor]) -> None:
        missing = {n for n, p in self.named_parameters() if p.requires_grad} - set(state)
        if missing:
            raise ValueError(f"checkpoint lacks {len(missing)} trainable tensors, e.g. {sorted(missing)[:3]}")
        self.load_state_dict(state, strict=False)

    # -- inputs -----------------------------------------------------------------------
    def prefix(self) -> str:
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
        return self.processor.apply_chat_template([messages], add_generation_prompt=True, tokenize=False)[0] \
            + f"language {PROMPT_LANGUAGE}<asr_text>"

    def encode(self, audio: torch.Tensor, mask: torch.Tensor, targets: Sequence[str] | None = None) -> dict[str, Any]:
        """Processor inputs for the reference channel of each session; with ``targets``, the training labels
        (prompt and padding at -100, as in the official fine-tuning script)."""
        ref = reference_channel(mask)
        mono = audio[torch.arange(audio.shape[0], device=audio.device), ref].float().cpu().numpy()
        prefix = self.prefix()
        eos = self.processor.tokenizer.eos_token or ""
        texts = [prefix] * len(mono) if targets is None else [prefix + t + eos for t in targets]
        inputs = self.processor(text=texts, audio=list(mono), return_tensors="pt", padding=True, truncation=False)
        if targets is not None:
            prefix_len = self.processor(text=[prefix], audio=[mono[0]], return_tensors="pt")["attention_mask"].sum().item()
            labels = inputs["input_ids"].clone()
            labels[:, :prefix_len] = -100
            labels[inputs["attention_mask"] == 0] = -100
            inputs["labels"] = labels
        return inputs

    def _set_space(self, batch: dict[str, Any], inputs: dict[str, Any]) -> int:
        n_mel = int(inputs["feature_attention_mask"].sum(1).max())
        n_tok = audio_token_count(n_mel)
        self._extra = None
        if self.spatial is not None and not batch.get("no_space", False):
            self._extra = self.spatial(batch["audio"], batch["mask"], batch["mic_xy"], n_tok)
        return n_mel

    def _head_logits(self, hidden: torch.Tensor, input_ids: torch.Tensor, n_mel: int) -> torch.Tensor:
        """``[B, 80, 3]`` from the last hidden state at the audio positions."""
        positions = input_ids == self.audio_token_id
        n_tok = int(positions[0].sum())
        audio_h = hidden[positions].reshape(hidden.shape[0], n_tok, hidden.shape[-1])
        W = grid_pooling(n_mel).to(audio_h.device)
        return self.head(torch.einsum("kt,bth->bkh", W, audio_h.float()))

    # -- training ---------------------------------------------------------------------
    def loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        device = batch["audio"].device
        dtype = next(self.thinker.parameters()).dtype
        inputs = self.encode(batch["audio"], batch["mask"], batch["text"])
        inputs = {k: (v.to(device, dtype) if v.is_floating_point() else v.to(device)) for k, v in inputs.items()}
        n_mel = self._set_space(batch, inputs)
        try:
            out = self.thinker(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                               input_features=inputs["input_features"],
                               feature_attention_mask=inputs["feature_attention_mask"], labels=inputs["labels"],
                               output_hidden_states=True)
        finally:
            self._extra = None
        logits = self._head_logits(out.hidden_states[-1], inputs["input_ids"], n_mel)
        target = batch["slot_labels"].to(device) + 1  # -1 silence -> class 0
        head_loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), target.reshape(-1))
        acc = float((logits.argmax(-1) == target).float().mean())
        total = out.loss + self.cfg.head_weight * head_loss
        return total, {"loss_text": float(out.loss), "loss_head": float(head_loss), "acc_head": acc}

    # -- inference --------------------------------------------------------------------
    @torch.no_grad()
    def transcribe(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        """Per session: ``{"text", "classes" (80 frame-head classes), "truncated"}``."""
        device = batch["audio"].device
        dtype = next(self.thinker.parameters()).dtype
        inputs = self.encode(batch["audio"], batch["mask"])
        inputs = {k: (v.to(device, dtype) if v.is_floating_point() else v.to(device)) for k, v in inputs.items()}
        n_mel = self._set_space(batch, inputs)
        try:
            out = self.thinker.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False,
                                        return_dict_in_generate=True, output_hidden_states=True,
                                        eos_token_id=[151645, 151643])
        finally:
            self._extra = None
        prompt_hidden = out.hidden_states[0][-1]
        classes = self._head_logits(prompt_hidden, inputs["input_ids"], n_mel).argmax(-1).cpu().tolist()
        new = out.sequences[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eos = {151645, 151643, self.processor.tokenizer.pad_token_id}
        truncated = [len(row) >= self.cfg.max_new_tokens and int(row[-1]) not in eos for row in new.tolist()]
        return [{"text": t, "classes": c, "truncated": bool(x)} for t, c, x in zip(texts, classes, truncated)]


def load(cfg: TrackCConfig, device: str) -> UnifiedASR:
    """The pinned Qwen3-ASR-1.7B in bf16 (fp16 before Ampere) with Track C's trainable parts."""
    from qwen_asr import Qwen3ASRModel

    from modern_asr import QWEN_MODEL, QWEN_REVISION, model_snapshot

    dtype = torch.float32
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    asr = Qwen3ASRModel.from_pretrained(model_snapshot(QWEN_MODEL, QWEN_REVISION), dtype=dtype, device_map=device,
                                        attn_implementation="sdpa")
    model = UnifiedASR(asr.model, asr.processor, cfg)
    for module in (model.spatial, model.head):
        if module is not None:
            module.to(device=device, dtype=torch.float32)
    return model


def log_parameter_counts(model: UnifiedASR) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "trainable_share_pct": round(100 * trainable / max(total, 1), 3)}


__all__ = ["TrackCConfig", "UnifiedASR", "load", "audio_token_count", "grid_pooling"]

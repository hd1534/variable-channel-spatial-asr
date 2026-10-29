"""Controlled Baseline 2: SRP-PHAT direction finding, delay-and-sum, frozen ASR.

This is the *classical signal processing* control: no learned parameters
anywhere in the spatial path, implemented directly on ``numpy``.  It answers
the question "how far does textbook array processing get on this benchmark?",
and because it shares the frozen ASR of Baseline 1 verbatim
(:class:`baselines.FrozenASR`), any WER difference between the two baselines
can only come from the beamformer, not from a different recogniser.

Azimuth
-------
Steered Response Power with PHAT weighting, evaluated exactly at the frozen
:data:`spec.AZIMUTH_BIN_CENTERS_DEG`, so the response vector is already on the
contract's grid and the softmax over it *is* the submitted distribution.  No
interpolation onto the grid, no smoothing kernel, nothing that could move
probability mass between bins after the fact.

For every active microphone pair the generalised cross correlation is computed
by FFT, whitened per frequency bin (the PHAT weighting), optionally restricted
to a band, inverse transformed once, and then read off at the time difference
of arrival that each candidate direction implies under a far-field plane-wave
model.  The per-pair contributions are summed.  With ``M = 1`` there is no pair
and the answer is the uniform distribution, exactly as in Baseline 1.

The far-field TDOA for the pair ``(i, j)`` and the direction
``u = (cos az, sin az)`` is ``tau_ij = ((p_j - p_i) . u) / c`` with ``c =``
:data:`spec.SPEED_OF_SOUND_M_S`, i.e. the microphone nearer the source leads.

Spatial aliasing
----------------
:data:`spec.ALIASING_LIMIT_HZ` is about 715 Hz for the 24 cm maximum pair
spacing, so almost all of the speech band is above the half-wavelength limit
and individual frequency bins are phase-ambiguous.  Broadband integration still
resolves the direction because the grating lobes of different frequencies do
not coincide, and that is the default.  ``--band below_aliasing`` and
``--band above_aliasing`` sum only one side of the limit, and
``--report-band-split`` records how often the two sides disagree with the
broadband answer -- which is the measurement that makes the aliasing argument
in the proposal testable instead of assertable.

Distance
--------
Ridge regression on two features: log RMS level and the log of a crude
direct-to-reverberant proxy (see :func:`envelope_decay_drr_proxy`).  Without
the room impulse response a true DRR cannot be computed, so the proxy and its
failure modes are documented on the function.  With no fitted model the
distance answer degrades to a prior or to uniform and says so in the metadata.

Transcription
-------------
The multichannel block is delay-and-sum beamformed toward the estimated
direction and the resulting single channel goes into the same frozen ASR as
Baseline 1.

Usage
-----
``python -m baselines.dsp_srp_phat --eval-dir data/eval-dev --out predictions.jsonl
--scope causal --track 1``

``python -m baselines.dsp_srp_phat --self-test`` runs the doctests and the
synthetic direction-finding smoke tests (a known plane wave must land in the
right azimuth bin) with no data and no checkpoint.
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
    azimuth_centers,
    build_final_record,
    build_timestep_record,
    causal_window_bounds,
    check_fit_targets,
    discover_cases,
    distance_centers,
    energy_vad_segments,
    fit_ridge,
    frame_rms,
    inference_grid_sec,
    load_audio,
    load_distance_model,
    map_point,
    normalise_distribution,
    pair_spacings_m,
    resolve_masks,
    resolve_scope,
    rms,
    speaker_slot,
    warn_on_unphysical_level_coefficient,
    write_predictions,
)
from baselines.single_channel import load_distance_labels

if TYPE_CHECKING:  # pragma: no cover - typing only
    from baselines import FloatArray
else:  # pragma: no cover
    FloatArray = Any

SYSTEM_NAME = "controlled_baseline_2_srp_phat_das"

BANDS = ("broadband", "below_aliasing", "above_aliasing")
"""Frequency-band options, split at :data:`spec.ALIASING_LIMIT_HZ`."""

SRP_METHODS = ("gcc_interp", "freq_steer")
"""``gcc_interp`` is the fast default; ``freq_steer`` is the reference check."""

PHAT_REGULARISATION = spec.NLL_EPSILON
"""Relative floor for the PHAT magnitude whitening.

The cross-spectrum is divided by ``|G| + PHAT_REGULARISATION * mean(|G|)``, so
the floor scales with the signal level and empty frequency bins contribute
nothing instead of exploding.  This is a numeric underflow guard, not a
contract quantity, but it reuses the contract's underflow floor rather than
writing a second epsilon into this file -- the same choice
:data:`baselines.single_channel.LEVEL_FLOOR` makes.
"""

DELAY_PAD_SAMPLES = 64
"""Zero padding added before an FFT fractional delay, so the circular wrap of
the shift lands in the padding rather than in the signal.  Comfortably longer
than the largest possible steering delay (``MAX_PAIR_SPACING_M / c`` is about
11 samples at 16 kHz)."""

MIN_SRP_WINDOW_SEC = spec.DELTA_T_SEC
"""Shortest observation window SRP-PHAT is run on: one grid hop.  Below this
(only ``t_0 = 0`` in causal scope) the azimuth answer is uniform."""

DEFAULT_FREQ_MIN_HZ = 100.0
"""Low edge of the analysis band; below this is rumble, not speech direction."""

DEFAULT_SRP_FRAME_SEC = 0.064
"""Analysis frame for the framed GCC-PHAT average (1024 samples at 16 kHz).

Long enough to resolve the whole speech band and to contain every steering
delay many times over, short enough that the PHAT whitening acts on a roughly
stationary spectrum.  It also bounds the FFT length independently of the
observation window, which is what keeps ``--scope full_audio`` tractable."""

FRAME_ENERGY_FLOOR_RATIO = 1.0e-3
"""Frames carrying less than this fraction (30 dB) of the loudest frame's
energy in the current window are dropped before averaging.  Phase whitening
alone would give a silent frame the same weight as a loud one."""

DEFAULT_LAG_UPSAMPLE = 8
"""Sinc-interpolation factor applied to the GCC lag axis before reading off the
candidate TDOAs.  At 16 kHz one sample is 2.9 cm of path difference, which is a
large fraction of the 24 cm array, so reading the correlation only at integer
lags would quantise the steered response coarsely."""

ONSET_FLOOR_RATIO = 0.01
"""An onset candidate must carry at least this fraction of the loudest frame's
energy (20 dB down).  This module's own numeric guard, not a contract value."""

DEFAULT_SILENCE_FLOOR_DB = 40.0
"""A window whose RMS is this far below the loudest window *seen so far* is
treated as silence and answered with the uniform azimuth.  The reference is a
causal running peak, never a whole-session statistic, which would leak the
future into a causal prediction."""


# ---------------------------------------------------------------------------
# Geometry and steering
# ---------------------------------------------------------------------------


def active_pairs(mask_bits: str) -> tuple[tuple[int, int], ...]:
    """Return the active microphone pairs as absolute channel indices.

    >>> active_pairs("1010")
    ((0, 2),)
    >>> active_pairs("1110")
    ((0, 1), (0, 2), (1, 2))
    >>> active_pairs("1000")
    ()
    """
    channels = spec.mask_to_channels(mask_bits)
    return tuple(
        (channels[a], channels[b])
        for a in range(len(channels))
        for b in range(a + 1, len(channels))
    )


def steering_tdoa_sec(
    mic_positions_m: Sequence[Sequence[float]],
    pairs: Sequence[tuple[int, int]],
    azimuths_deg: Sequence[float] = spec.AZIMUTH_BIN_CENTERS_DEG,
    speed_of_sound_m_s: float = spec.SPEED_OF_SOUND_M_S,
) -> FloatArray:
    """Return the ``(n_pairs, n_azimuths)`` far-field TDOA table, in seconds.

    ``tau[p, k] = ((p_j - p_i) . u_k) / c`` for pair ``(i, j)`` and direction
    ``u_k = (cos az_k, sin az_k)``.  Negative means channel ``i`` leads, i.e.
    the source is on ``i``'s side.  Only coordinate *differences* appear, so
    the table is invariant to where the array origin sits -- which is what lets
    the benchmark drop channels without moving the coordinate system.

    >>> import numpy as np
    >>> from baselines import default_mic_positions_m
    >>> tau = steering_tdoa_sec(default_mic_positions_m(), [(0, 2)], [0.0, 180.0])
    >>> float(round(tau[0, 0] * spec.SPEED_OF_SOUND_M_S, 6))
    -0.24
    >>> float(round(tau[0, 1] * spec.SPEED_OF_SOUND_M_S, 6))
    0.24
    >>> bool(np.allclose(tau[0, 0], -tau[0, 1]))
    True
    """
    import numpy as np

    positions = np.asarray(mic_positions_m, dtype=np.float64)
    angles = np.radians(np.asarray(azimuths_deg, dtype=np.float64))
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (K, 2)
    deltas = np.array(
        [positions[j] - positions[i] for i, j in pairs], dtype=np.float64
    ).reshape(len(pairs), 2)
    return (deltas @ unit.T) / float(speed_of_sound_m_s)


def band_mask(
    freqs_hz: FloatArray,
    band: str,
    freq_min_hz: float = DEFAULT_FREQ_MIN_HZ,
    freq_max_hz: float = 0.0,
    aliasing_limit_hz: float = spec.ALIASING_LIMIT_HZ,
) -> FloatArray:
    """Return a 0/1 weight per frequency bin for the requested band.

    ``freq_max_hz <= 0`` means "up to Nyquist".  The split point is
    :data:`spec.ALIASING_LIMIT_HZ`: at and below it the array is
    unambiguous bin by bin, above it each bin is phase-wrapped and only the
    combination across frequencies resolves the direction.

    >>> import numpy as np
    >>> f = np.array([50.0, 500.0, 2000.0])
    >>> band_mask(f, "broadband").tolist()
    [0.0, 1.0, 1.0]
    >>> band_mask(f, "below_aliasing").tolist()
    [0.0, 1.0, 0.0]
    >>> band_mask(f, "above_aliasing").tolist()
    [0.0, 0.0, 1.0]
    """
    import numpy as np

    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, got {band!r}")
    low = freq_min_hz
    high = freq_max_hz if freq_max_hz > 0.0 else float(np.max(freqs_hz))
    if band == "below_aliasing":
        high = min(high, aliasing_limit_hz)
    elif band == "above_aliasing":
        low = max(low, aliasing_limit_hz)
    return ((freqs_hz >= low) & (freqs_hz <= high)).astype(np.float64)


def _next_pow2(value: int) -> int:
    """Return the smallest power of two at least ``value``.

    >>> [_next_pow2(v) for v in (1, 2, 3, 1000)]
    [1, 2, 4, 1024]
    """
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


# ---------------------------------------------------------------------------
# SRP-PHAT
# ---------------------------------------------------------------------------


def srp_phat_response(
    block: FloatArray,
    mic_positions_m: Sequence[Sequence[float]],
    channels: Sequence[int],
    sample_rate_hz: int,
    band: str = BANDS[0],
    method: str = SRP_METHODS[0],
    azimuths_deg: Sequence[float] = spec.AZIMUTH_BIN_CENTERS_DEG,
    freq_min_hz: float = DEFAULT_FREQ_MIN_HZ,
    freq_max_hz: float = 0.0,
    per_pair_normalise: bool = False,
    lag_upsample: int = DEFAULT_LAG_UPSAMPLE,
    frame_sec: float = DEFAULT_SRP_FRAME_SEC,
) -> FloatArray:
    """Return the steered response power over ``azimuths_deg``.

    ``block`` is ``(M, n)`` with row ``m`` belonging to absolute channel
    ``channels[m]``; the coordinates are looked up by that absolute index so
    geometry never detaches from its channel.

    ``method="gcc_interp"`` inverse-transforms each pair's whitened cross
    spectrum once and linearly interpolates the result at the candidate TDOAs:
    ``O(P * nfft log nfft + P * K)``.  A whitened broadband correlation is
    nearly a sinc, so it is first sinc-interpolated onto a ``lag_upsample``
    times finer lag grid (zero padding in frequency) before the linear read --
    at ``lag_upsample = 1`` the linear error between integer lags is large
    enough to visibly distort the off-peak shape of the response, and the
    off-peak shape is what the submitted distribution, its NLL and its
    Coverage@90% are made of.  ``method="freq_steer"`` evaluates the steering
    sum directly in the frequency domain, ``O(P * F * K)`` -- slower, and kept
    as an independent implementation so the fast path can be checked against it
    (the self-test asserts the two agree).

    Returns an all-zero response when there is no usable pair, which callers
    turn into the uniform distribution.

    >>> import numpy as np
    >>> from baselines import default_mic_positions_m
    >>> resp = srp_phat_response(np.zeros((1, 1600)), default_mic_positions_m(),
    ...                          (0,), 16000)
    >>> resp.shape, bool(np.all(resp == 0.0))
    ((36,), True)
    """
    import numpy as np

    if method not in SRP_METHODS:
        raise ValueError(f"method must be one of {SRP_METHODS}, got {method!r}")
    data = np.asarray(block, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"block must be 2-D (M, n), got shape {data.shape}")
    n_mics, n_samples = data.shape
    if n_mics != len(channels):
        raise ValueError(f"block has {n_mics} rows but {len(channels)} channels given")
    n_azimuths = len(azimuths_deg)
    if n_mics < 2 or n_samples < 2:
        return np.zeros(n_azimuths, dtype=np.float64)

    local_pairs = [(a, b) for a in range(n_mics) for b in range(a + 1, n_mics)]
    absolute_pairs = [(channels[a], channels[b]) for a, b in local_pairs]
    tdoa = steering_tdoa_sec(mic_positions_m, absolute_pairs, azimuths_deg)

    # Framed analysis: the cross spectrum is whitened and averaged over short
    # frames instead of being taken from one transform of the whole window.
    # That is what PHAT is supposed to do -- whitening a 20 s transform
    # flattens the long-term spectrum, not the instantaneous one -- and it caps
    # the FFT length, so the cost grows linearly with the observation window
    # instead of super-linearly. Without it, --scope full_audio recomputes a
    # multi-million-point transform at every one of the 80 grid points.
    frame = min(n_samples, max(8, int(round(frame_sec * sample_rate_hz))))
    hop = max(1, frame // 2)
    nfft = _next_pow2(2 * frame)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / float(sample_rate_hz))
    weights = band_mask(freqs, band, freq_min_hz, freq_max_hz)
    if not np.any(weights):
        return np.zeros(n_azimuths, dtype=np.float64)

    starts = list(range(0, max(1, n_samples - frame + 1), hop))
    taper = np.hanning(frame)
    frames = np.stack([data[:, s : s + frame] * taper for s in starts])  # (F, M, frame)
    energies = np.sum(frames**2, axis=(1, 2))
    peak_energy = float(np.max(energies)) if energies.size else 0.0
    if peak_energy <= 0.0:
        return np.zeros(n_azimuths, dtype=np.float64)
    # Pure phase whitening gives every frame equal weight, so a window that is
    # mostly silence would be dominated by whitened noise. Frames far below the
    # loudest frame of this window are dropped.
    keep = energies >= peak_energy * FRAME_ENERGY_FLOOR_RATIO
    frames = frames[keep]
    spectra = np.fft.rfft(frames, n=nfft, axis=2)  # (F', M, bins)

    upsample = max(1, int(lag_upsample))
    response = np.zeros(n_azimuths, dtype=np.float64)
    max_lag = int(math.ceil(float(np.max(np.abs(tdoa))) * sample_rate_hz * upsample)) + 2
    max_lag = max(1, min(max_lag, (nfft * upsample) // 2 - 1))
    for pair_index, (a, b) in enumerate(local_pairs):
        cross = spectra[:, a, :] * np.conj(spectra[:, b, :])
        magnitude = np.abs(cross)
        floor = PHAT_REGULARISATION * np.mean(magnitude, axis=1, keepdims=True)
        whitened = (cross / (magnitude + floor + np.finfo(np.float64).tiny)).mean(axis=0)
        whitened = whitened * weights
        if method == "gcc_interp":
            # irfft onto a longer grid zero-pads the spectrum, i.e. sinc
            # interpolates the correlation; the 1/n scaling is undone so the
            # response keeps the same units as the freq_steer path.
            correlation = np.fft.irfft(whitened, n=nfft * upsample) * upsample
            lags = np.arange(-max_lag, max_lag + 1, dtype=np.float64) / upsample
            centered = np.concatenate([correlation[-max_lag:], correlation[: max_lag + 1]])
            values = np.interp(tdoa[pair_index] * sample_rate_hz, lags, centered)
        else:
            active = weights > 0.0
            phase = np.exp(
                2.0j
                * np.pi
                * np.outer(tdoa[pair_index], freqs[active])
            )
            values = np.real(whitened[active][None, :] * phase).sum(axis=1)
        if per_pair_normalise:
            scale = float(np.max(np.abs(values)))
            if scale > 0.0:
                values = values / scale
        response += values
    return response


def response_to_distribution(
    response: FloatArray,
    temperature: float = 1.0,
    prob_decimals: int | None = None,
) -> list[float]:
    """Turn a steered-response vector into a probability distribution.

    A softmax of the response standardised by its own spread, so the sharpness
    does not depend on the absolute scale of the audio: ``p propto exp((r -
    max r) / (T * sd))``.  An all-equal or all-zero response yields the uniform
    distribution, which is the right answer for ``M = 1`` and for silence.

    ``temperature`` is a hyperparameter of this baseline, not of the contract;
    it is exposed as ``--srp-temperature`` and recorded in the metadata,
    because NLL and Coverage@90% depend on it and a baseline that tuned it on
    the eval set would not be a baseline.

    >>> import numpy as np
    >>> p = response_to_distribution(np.array([0.0, 0.0, 0.0]))
    >>> [round(v, 6) for v in p]
    [0.333333, 0.333333, 0.333333]
    >>> q = response_to_distribution(np.array([0.0, 10.0, 0.0]))
    >>> q[1] > q[0] and abs(math.fsum(q) - 1.0) < spec.PROB_SUM_TOLERANCE
    True
    """
    import numpy as np

    values = np.asarray(response, dtype=np.float64).ravel()
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    spread = float(np.std(values))
    if not np.isfinite(spread) or spread <= 0.0:
        return normalise_distribution([1.0] * values.size, decimals=prob_decimals)
    scaled = (values - float(np.max(values))) / (temperature * spread)
    return normalise_distribution(np.exp(scaled).tolist(), decimals=prob_decimals)


def parabolic_peak_deg(
    response: FloatArray,
    azimuths_deg: Sequence[float] = spec.AZIMUTH_BIN_CENTERS_DEG,
    bin_width_deg: float = spec.AZIMUTH_BIN_WIDTH_DEG,
) -> float:
    """Return a sub-bin azimuth point estimate by parabolic interpolation.

    The three samples around the peak are fitted with a parabola, wrapping
    around the circle because bin 0 and bin 35 are neighbours.  The offset is
    clamped to half a bin so a flat or noisy peak cannot be dragged into the
    next bin.  The contract prefers an explicit point over the MAP bin center
    precisely so a method with sub-bin resolution is not charged the
    quantisation error.

    >>> import numpy as np
    >>> resp = np.zeros(spec.K_AZIMUTH)
    >>> resp[23] = 1.0
    >>> parabolic_peak_deg(resp)
    55.0
    >>> resp[24] = 1.0  # peak midway between bins 23 and 24
    >>> round(parabolic_peak_deg(resp), 4)
    60.0
    >>> flat = np.zeros(spec.K_AZIMUTH)
    >>> parabolic_peak_deg(flat)
    -175.0
    """
    import numpy as np

    values = np.asarray(response, dtype=np.float64).ravel()
    n = values.size
    if n != len(azimuths_deg):
        raise ValueError(f"response has {n} bins but {len(azimuths_deg)} azimuths given")
    peak = int(np.argmax(values))
    left = values[(peak - 1) % n]
    right = values[(peak + 1) % n]
    denominator = left - 2.0 * values[peak] + right
    offset = 0.0 if denominator == 0.0 else 0.5 * (left - right) / denominator
    offset = max(-0.5, min(0.5, float(offset)))
    center = float(azimuths_deg[peak]) + offset * bin_width_deg
    low, high = spec.AZIMUTH_RANGE_DEG
    return (center - low) % (high - low) + low


# ---------------------------------------------------------------------------
# Delay-and-sum beamforming
# ---------------------------------------------------------------------------


def fractional_delay(
    signal: FloatArray,
    delay_samples: float,
    pad: int = DELAY_PAD_SAMPLES,
) -> FloatArray:
    """Delay a mono signal by a fractional number of samples.

    Returns ``y(t) = x(t - delay_samples)``; a negative delay advances the
    signal.  Implemented as a linear phase ramp in the frequency domain, with
    zero padding so that the circular wrap of the shift falls outside the
    returned span.

    >>> import numpy as np
    >>> x = np.zeros(64); x[32] = 1.0
    >>> y = fractional_delay(x, 2.0)
    >>> int(np.argmax(y))
    34
    >>> z = fractional_delay(x, -3.0)
    >>> int(np.argmax(z))
    29
    >>> float(round(fractional_delay(x, 0.0)[32], 9))
    1.0
    """
    import numpy as np

    arr = np.asarray(signal, dtype=np.float64).ravel()
    if arr.size == 0 or delay_samples == 0.0:
        return arr.copy()
    padded = np.concatenate([np.zeros(pad), arr, np.zeros(pad)])
    nfft = _next_pow2(padded.size)
    spectrum = np.fft.rfft(padded, n=nfft)
    freqs = np.fft.rfftfreq(nfft)
    shifted = np.fft.irfft(
        spectrum * np.exp(-2.0j * np.pi * freqs * float(delay_samples)), n=nfft
    )
    return shifted[pad : pad + arr.size]


def delay_and_sum(
    block: FloatArray,
    mic_positions_m: Sequence[Sequence[float]],
    channels: Sequence[int],
    sample_rate_hz: int,
    azimuth_deg: float,
    speed_of_sound_m_s: float = spec.SPEED_OF_SOUND_M_S,
) -> FloatArray:
    """Delay-and-sum beamform ``block`` toward ``azimuth_deg``.

    Channel ``m`` at position ``p`` is advanced by ``(p . u) / c`` so that the
    plane-wave component arriving from ``u`` lines up across channels, then the
    channels are averaged.  Coherent gain on the steered direction is ``M``,
    incoherent gain on uncorrelated noise is ``sqrt(M)``, so the output SNR
    improves by about ``sqrt(M)`` -- which is the only mechanism by which this
    baseline's WER can beat Baseline 1's.

    >>> import numpy as np
    >>> from baselines import default_mic_positions_m
    >>> mono = delay_and_sum(np.ones((2, 100)), default_mic_positions_m(),
    ...                      (0, 2), 16000, 0.0)
    >>> mono.shape
    (100,)

    A microphone on the source side leads, so aligning it means *delaying* it:

    >>> x = np.zeros((2, 512))
    >>> x[0, 256 - 6] = 1.0   # channel 0 faces the source at 0 deg, arrives early
    >>> x[1, 256 + 6] = 1.0   # channel 2 is on the far side, arrives late
    >>> y = delay_and_sum(x, [(0.12, 0.0), (-0.12, 0.0)], (0, 1), 16000, 0.0)
    >>> bool(abs(int(np.argmax(y)) - 256) <= 1)   # both impulses line up
    True
    """
    import numpy as np

    data = np.asarray(block, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"block must be 2-D (M, n), got shape {data.shape}")
    if data.shape[0] != len(channels):
        raise ValueError(f"block has {data.shape[0]} rows but {len(channels)} channels")
    if data.shape[0] == 1:
        return data[0].copy()
    positions = np.asarray(mic_positions_m, dtype=np.float64)
    angle = math.radians(azimuth_deg)
    unit = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    accumulator = np.zeros(data.shape[1], dtype=np.float64)
    for row, channel in enumerate(channels):
        # The plane wave reaches this mic at t_m = -(p . u) / c relative to the
        # origin, so aligning it to the origin means delaying it by -t_m.
        align_sec = float(positions[channel] @ unit) / float(speed_of_sound_m_s)
        accumulator += fractional_delay(data[row], align_sec * sample_rate_hz)
    return accumulator / float(data.shape[0])


# ---------------------------------------------------------------------------
# Distance features
# ---------------------------------------------------------------------------


def envelope_decay_drr_proxy(
    mono: FloatArray,
    sample_rate_hz: int,
    direct_sec: float = 0.020,
    tail_sec: float = 0.120,
    frame_sec: float = 0.005,
    hop_sec: float = 0.0025,
    max_onsets: int = 16,
) -> float:
    """Return a crude direct-to-reverberant energy ratio proxy (linear, > 0).

    Method: build a short-frame energy envelope, find its local maxima above
    the active floor (syllable onsets), and at each one take the mean energy
    per frame in the ``direct_sec`` window starting at the peak divided by the
    mean energy per frame in the following ``tail_sec`` window.  The median
    over up to ``max_onsets`` onsets is returned.  Energy *densities* are used,
    so the value does not depend on the two window lengths.

    **This is not a DRR.**  A real direct-to-reverberant ratio needs the room
    impulse response to separate the direct path from its reflections, and the
    benchmark's inference input has no RIR.  What this measures is how fast the
    envelope falls after an onset, which is confounded with:

    * speaking rate and phonetic context -- a long vowel keeps the "tail" high
      and mimics reverberation;
    * additive noise -- the noise floor lifts the tail and saturates the ratio,
      so the proxy compresses at low SNR;
    * the VAD/onset picker -- a missed onset changes which frames are compared;
    * the analysis window length -- a 1 s causal chunk holds only a few
      syllables, so the per-chunk value is very noisy. That noise is why the
      distance head here is a *ridge* regression and why its output
      distribution is widened by the fitted residual spread instead of being
      peaked on the point estimate.

    It is monotone in true DRR only on average over many onsets, so it is a
    usable *feature*, not an estimate to report.

    Reverberating a modulated-noise "utterance" with a synthetic exponential
    room response does move the proxy in the right direction:

    >>> import numpy as np
    >>> sr = 16000
    >>> t = np.arange(sr) / sr
    >>> rng = np.random.default_rng(0)
    >>> dry = rng.normal(0.0, 1.0, sr) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) ** 4
    >>> rir = np.exp(-np.arange(int(0.25 * sr)) / (0.08 * sr))
    >>> rir[0] = 3.0                                   # a dominant direct path
    >>> wet = np.convolve(dry, rir)[:sr]
    >>> envelope_decay_drr_proxy(dry, sr) > envelope_decay_drr_proxy(wet, sr)
    True

    Degenerate inputs stay finite and positive:

    >>> impulsive = np.zeros(sr)
    >>> impulsive[::1600] = 1.0                        # sharp onsets, no tail
    >>> smeared = np.exp(-t * 2.0) * np.sin(2 * np.pi * 300 * t)   # slow decay
    >>> envelope_decay_drr_proxy(impulsive, sr) > envelope_decay_drr_proxy(smeared, sr)
    True
    >>> envelope_decay_drr_proxy(np.zeros(sr), sr) > 0.0
    True
    >>> envelope_decay_drr_proxy(np.zeros(10), sr)
    1.0
    """
    import numpy as np

    arr = np.asarray(mono, dtype=np.float64).ravel()
    if arr.size < int(round((direct_sec + tail_sec) * sample_rate_hz)):
        return 1.0
    envelope = frame_rms(arr, sample_rate_hz, frame_sec, hop_sec) ** 2
    n_direct = max(1, int(round(direct_sec / hop_sec)))
    n_tail = max(1, int(round(tail_sec / hop_sec)))
    last = envelope.size - (n_direct + n_tail)
    peak_energy = float(np.max(envelope)) if envelope.size else 0.0
    if last <= 1 or peak_energy <= 0.0:
        return 1.0
    # Onsets must be loud relative to the loudest frame, not merely above a
    # percentile: a sparse signal has a zero 75th percentile, which would
    # otherwise make every silent frame an "onset".
    floor = max(float(np.percentile(envelope, 75.0)), peak_energy * ONSET_FLOOR_RATIO)

    def ratio_at(index: int) -> float | None:
        """Direct-window energy density over tail-window energy density."""
        direct = float(np.mean(envelope[index : index + n_direct]))
        tail = float(np.mean(envelope[index + n_direct : index + n_direct + n_tail]))
        return direct / tail if tail > 0.0 and direct > 0.0 else None

    ratios: list[float] = []
    index = 1
    while index < last and len(ratios) < max_onsets:
        if (
            envelope[index] >= floor
            and envelope[index] >= envelope[index - 1]
            and envelope[index] >= envelope[index + 1]
        ):
            value = ratio_at(index)
            if value is not None:
                ratios.append(value)
            index += n_direct
        else:
            index += 1
    if not ratios:
        # A monotonically decaying envelope has no interior local maximum, so
        # fall back to the single global peak rather than reporting nothing.
        fallback = ratio_at(min(int(np.argmax(envelope)), last - 1))
        if fallback is None:
            return 1.0
        ratios.append(fallback)
    return max(float(np.median(ratios)), spec.NLL_EPSILON)


def distance_features(mono: FloatArray, sample_rate_hz: int) -> list[float]:
    """Return ``[log10(rms), log10(drr_proxy)]`` for the distance regression.

    Level falls with distance and the envelope decay flattens with distance, so
    the two features carry partly independent information: level alone cannot
    separate a quiet near talker from a loud far one, while the decay proxy is
    gain invariant.

    >>> import numpy as np
    >>> feats = distance_features(np.full(16000, 0.1), 16000)
    >>> len(feats), round(feats[0], 4)
    (2, -1.0)
    """
    level = math.log10(max(rms(mono), spec.NLL_EPSILON))
    proxy = math.log10(max(envelope_decay_drr_proxy(mono, sample_rate_hz), spec.NLL_EPSILON))
    return [level, proxy]


DISTANCE_FEATURE_NAMES = ("log10_rms", "log10_drr_proxy")


# ---------------------------------------------------------------------------
# Per-case inference
# ---------------------------------------------------------------------------


def _azimuth_for_block(
    block: FloatArray,
    case: CaseSpec,
    sample_rate_hz: int,
    options: Mapping[str, Any],
    band: str | None = None,
) -> tuple[FloatArray, float]:
    """Return ``(response, point_deg)`` for one multichannel block."""
    import numpy as np

    response = srp_phat_response(
        block,
        case.positions(),
        case.channels,
        sample_rate_hz,
        band=band or options["band"],
        method=options["method"],
        freq_min_hz=options["freq_min_hz"],
        freq_max_hz=options["freq_max_hz"],
        per_pair_normalise=options["per_pair_normalise"],
        lag_upsample=options["lag_upsample"],
        frame_sec=options["frame_sec"],
    )
    if not np.any(response):
        return response, map_point(
            normalise_distribution([1.0] * spec.K_AZIMUTH), azimuth_centers()
        )
    if options["point_interpolation"] == "parabolic":
        return response, parabolic_peak_deg(response)
    return response, float(spec.AZIMUTH_BIN_CENTERS_DEG[int(np.argmax(response))])


def transcribe_case(
    session: AudioSession,
    case: CaseSpec,
    asr: FrozenASR,
    options: Mapping[str, Any],
    session_length_sec: float = spec.SESSION_LENGTH_SEC,
) -> tuple[dict[str, str], list[float]]:
    """Beamform each VAD segment toward its own SRP-PHAT peak, then transcribe.

    Returns ``({speaker_slot: text}, per_segment_azimuth_deg)``.  Each segment
    is steered using only its own samples, so the direction used for a segment
    never depends on audio after that segment ends -- the beamformer is causal
    at segment granularity, which is what the frozen-ASR-per-segment rule in
    :data:`spec.ASR_MIN_SEGMENT_SEC` allows.
    """
    end_sec = min(session.duration_sec, session_length_sec)
    channels = case.channels
    reference = session.window(0.0, end_sec, (channels[0],))[0]
    pieces: list[str] = []
    azimuths: list[float] = []
    for start, stop in energy_vad_segments(reference, session.sample_rate_hz):
        block = session.window(start, min(stop, end_sec), channels)
        if block.shape[1] == 0:
            continue
        if len(channels) > 1:
            _, azimuth = _azimuth_for_block(block, case, session.sample_rate_hz, options)
            mono = delay_and_sum(
                block, case.positions(), channels, session.sample_rate_hz, azimuth
            )
            azimuths.append(azimuth)
        else:
            mono = block[0]
        text = asr.transcribe(mono, session.sample_rate_hz).strip()
        if text:
            pieces.append(text)
    return {speaker_slot(0): " ".join(pieces)}, azimuths


def predict_case(
    case: CaseSpec,
    session: AudioSession,
    asr: FrozenASR,
    distance_model: DistanceModel,
    scope: str,
    options: Mapping[str, Any],
    run_asr: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run Baseline 2 on one case and return ``(records, diagnostics)``."""
    import numpy as np

    channels = case.channels
    for channel in channels:
        if channel >= session.n_channels:
            raise ValueError(
                f"{case.case_id}: mask needs channel {channel} but "
                f"{case.audio_path} has {session.n_channels} channel(s)"
            )
    prob_decimals = options["prob_decimals"]
    uniform_azimuth = normalise_distribution([1.0] * spec.K_AZIMUTH, decimals=prob_decimals)
    uniform_point = map_point(uniform_azimuth, azimuth_centers())
    silence_ratio = 10.0 ** (-options["silence_floor_db"] / 20.0)

    records: list[dict[str, Any]] = []
    running_peak = 0.0
    smoothed: FloatArray | None = None
    band_agreement: dict[str, list[float]] = {band: [] for band in BANDS[1:]}
    n_uniform = 0
    # In full_audio scope every grid point sees the same window, so the steered
    # response is computed once and reused; in causal/windowed scope the key
    # changes every step and the cache holds a single entry.
    cache_state: dict[str, Any] = {"bounds": None, "by_band": {}}

    def azimuth_cached(
        block: FloatArray, bounds: tuple[float, float], band: str
    ) -> tuple[FloatArray, float]:
        """SRP-PHAT for one window, memoised per band until the bounds move."""
        if cache_state["bounds"] != bounds:
            cache_state["bounds"] = bounds
            cache_state["by_band"] = {}
        by_band: dict[str, tuple[FloatArray, float]] = cache_state["by_band"]
        if band not in by_band:
            by_band[band] = _azimuth_for_block(
                block, case, session.sample_rate_hz, options, band=band
            )
        return by_band[band]

    timer = CaseTimer()
    with timer:
        for eval_time_sec in inference_grid_sec():
            start_sec, end_sec = causal_window_bounds(
                eval_time_sec, scope, lookahead_sec=options["lookahead_sec"]
            )
            bounds = (start_sec, end_sec)
            block = session.window(start_sec, end_sec, channels)
            level = rms(block) if block.size else 0.0
            running_peak = max(running_peak, level)
            too_short = (end_sec - start_sec) < MIN_SRP_WINDOW_SEC or block.shape[1] < 2
            # Digital silence carries no direction at all, and the reference
            # level is the peak seen SO FAR -- a whole-session statistic would
            # leak the future into a causal prediction.
            too_quiet = level <= 0.0 or (running_peak > 0.0 and level < running_peak * silence_ratio)
            if len(channels) < 2 or too_short or too_quiet:
                azimuth_probs, azimuth_point = uniform_azimuth, uniform_point
                mono = block[0] if block.shape[1] else np.zeros(0)
                n_uniform += 1
            else:
                response, azimuth_point = azimuth_cached(block, bounds, options["band"])
                if options["smooth_alpha"] > 0.0 and smoothed is not None:
                    response = (
                        options["smooth_alpha"] * smoothed
                        + (1.0 - options["smooth_alpha"]) * response
                    )
                    if options["point_interpolation"] == "parabolic":
                        azimuth_point = parabolic_peak_deg(response)
                    else:
                        azimuth_point = float(
                            spec.AZIMUTH_BIN_CENTERS_DEG[int(np.argmax(response))]
                        )
                smoothed = response
                azimuth_probs = response_to_distribution(
                    response, options["srp_temperature"], prob_decimals
                )
                if options["report_band_split"]:
                    for band in BANDS[1:]:
                        side, side_point = azimuth_cached(block, bounds, band)
                        if np.any(side):
                            band_agreement[band].append(
                                spec.circular_error_deg(azimuth_point, side_point)
                            )
                mono = delay_and_sum(
                    block, case.positions(), channels, session.sample_rate_hz, azimuth_point
                )
            if mono.size and distance_model.kind == "regression":
                features = distance_features(mono, session.sample_rate_hz)
                distance_probs = normalise_distribution(
                    distance_model.predict_distribution(features), decimals=prob_decimals
                )
                distance_point = distance_model.predict_point(features)
            else:
                distance_probs = normalise_distribution(
                    distance_model.predict_distribution(
                        [0.0] * max(1, len(distance_model.feature_names))
                    ),
                    decimals=prob_decimals,
                )
                distance_point = map_point(distance_probs, distance_centers())
            records.append(
                build_timestep_record(
                    case_id=case.case_id,
                    eval_time_sec=eval_time_sec,
                    speaker_id=speaker_slot(0),
                    azimuth_probs=azimuth_probs,
                    distance_probs=distance_probs,
                    pred_azimuth_deg=azimuth_point,
                    pred_distance_m=distance_point,
                    include_record_type=options["include_record_type"],
                )
            )
        if run_asr:
            transcripts, segment_azimuths = transcribe_case(session, case, asr, options)
        else:
            transcripts, segment_azimuths = {speaker_slot(0): ""}, []

    audio_sec = min(session.duration_sec, spec.SESSION_LENGTH_SEC)
    grid_size = len(inference_grid_sec())
    meta: dict[str, Any] = {
        "n_active_channels": case.n_active_channels,
        "n_channels_used": case.n_active_channels,
        "mask_bits": case.mask_bits,
        "active_pairs": [list(pair) for pair in active_pairs(case.mask_bits)],
        "pair_spacings_m": {
            f"{i}-{j}": round(value, 6)
            for (i, j), value in pair_spacings_m(case.positions(), case.mask_bits).items()
        },
        "azimuth_source": (
            "uniform_unobservable_at_m1" if case.n_active_channels < 2 else "srp_phat_softmax"
        ),
        "srp_band": options["band"],
        "srp_method": options["method"],
        "srp_temperature": options["srp_temperature"],
        "srp_freq_min_hz": options["freq_min_hz"],
        "srp_freq_max_hz": options["freq_max_hz"],
        "srp_per_pair_normalise": options["per_pair_normalise"],
        "srp_frame_sec": options["frame_sec"],
        "srp_lag_upsample": options["lag_upsample"],
        "srp_frame_energy_floor_ratio": FRAME_ENERGY_FLOOR_RATIO,
        "aliasing_limit_hz": spec.ALIASING_LIMIT_HZ,
        "point_interpolation": options["point_interpolation"],
        "smooth_alpha": options["smooth_alpha"],
        "temporal_state": "chunk_reset_stateless"
        if options["smooth_alpha"] <= 0.0
        else "causal_ema_over_response",
        "uniform_azimuth_fraction": round(n_uniform / grid_size, 6) if grid_size else 0.0,
        "beamformer": "delay_and_sum" if case.n_active_channels > 1 else "single_channel_passthrough",
        "asr_backend": asr.backend_id,
        "asr_frozen": True,
        "asr_input": "delay_and_sum_beamformed",
        "observation_scope": scope,
        "lookahead_sec": options["lookahead_sec"] if scope == "windowed" else 0.0,
        "geometry_source": case.geometry_source,
        "transcription_enabled": run_asr,
        "n_asr_segments": len(segment_azimuths),
        "sample_rate_hz": session.sample_rate_hz,
        "audio_duration_sec": session.duration_sec,
    }
    if options["report_band_split"]:
        meta["band_split_diagnostics"] = {
            band: {
                "n_frames": len(values),
                "mean_circular_deviation_from_broadband_deg": (
                    round(sum(values) / len(values), 4) if values else None
                ),
                "fraction_within_one_bin": (
                    round(
                        sum(1 for v in values if v <= spec.AZIMUTH_BIN_WIDTH_DEG) / len(values), 6
                    )
                    if values
                    else None
                ),
            }
            for band, values in band_agreement.items()
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
        include_record_type=options["include_record_type"],
        meta=meta,
    )
    records.append(record)
    return records, {"rtf": record["reported_rtf"], "uniform_frames": n_uniform}


# ---------------------------------------------------------------------------
# Distance fitting
# ---------------------------------------------------------------------------


def fit_distance_model(
    cases: Sequence[CaseSpec],
    labels: Mapping[str, float],
    options: Mapping[str, Any],
    audio_backend: str = "auto",
    alpha: float = 1.0,
) -> DistanceModel:
    """Fit ridge regression ``log r ~ [log10 rms, log10 drr_proxy]``.

    One sample per (session, mask) case, because the beamformed signal -- and
    therefore both features -- depends on how many channels were available.
    The ridge penalty matters here: the two features are correlated through the
    overall signal level, so plain least squares on a small labelled set gives
    unstable, large coefficients of opposite sign.

    The labels must come from a **training** split; fitting on ``eval-dev`` and
    scoring on it is leakage, not a baseline.
    """
    features: list[list[float]] = []
    targets: list[float] = []
    cache: dict[str, AudioSession] = {}
    for case in cases:
        if case.session_id not in labels:
            continue
        if case.audio_path not in cache:
            cache.clear()
            cache[case.audio_path] = load_audio(case.audio_path, audio_backend)
        session = cache[case.audio_path]
        end_sec = min(session.duration_sec, spec.SESSION_LENGTH_SEC)
        block = session.window(0.0, end_sec, case.channels)
        if case.n_active_channels > 1:
            _, azimuth = _azimuth_for_block(block, case, session.sample_rate_hz, options)
            mono = delay_and_sum(
                block, case.positions(), case.channels, session.sample_rate_hz, azimuth
            )
        else:
            mono = block[0]
        features.append(distance_features(mono, session.sample_rate_hz))
        targets.append(math.log(max(labels[case.session_id], spec.NLL_EPSILON)))
    if len(features) < 4:
        raise ValueError(f"need at least 4 labelled cases to fit, got {len(features)}")
    check_fit_targets(targets)
    coefficients, residual_std = fit_ridge(features, targets, alpha=alpha)
    warn_on_unphysical_level_coefficient(coefficients, DISTANCE_FEATURE_NAMES)
    return DistanceModel(
        kind="regression",
        coefficients=tuple(coefficients),
        feature_names=DISTANCE_FEATURE_NAMES,
        sigma_log=max(residual_std, MIN_SIGMA_LOG),
        alpha=alpha,
        n_train=len(features),
        note=(
            "ridge on [log10 rms, log10 envelope-decay DRR proxy] of the "
            "beamformed channel; the proxy is not a measured DRR"
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for Baseline 2."""
    parser = argparse.ArgumentParser(
        prog="python -m baselines.dsp_srp_phat",
        description=(
            "Controlled Baseline 2: SRP-PHAT azimuth, delay-and-sum beamforming "
            "into the same frozen ASR as Baseline 1, ridge distance regression."
        ),
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--band",
        default=BANDS[0],
        choices=list(BANDS),
        help=f"integration band, split at spec.ALIASING_LIMIT_HZ ({spec.ALIASING_LIMIT_HZ:.1f} Hz)",
    )
    parser.add_argument(
        "--srp-method", default=SRP_METHODS[0], choices=list(SRP_METHODS),
        help="GCC interpolation (fast) or direct frequency-domain steering (reference)",
    )
    parser.add_argument(
        "--srp-temperature", type=float, default=1.0,
        help="softmax temperature over the standardised steered response",
    )
    parser.add_argument("--freq-min-hz", type=float, default=DEFAULT_FREQ_MIN_HZ)
    parser.add_argument(
        "--freq-max-hz", type=float, default=0.0, help="0 means up to Nyquist"
    )
    parser.add_argument(
        "--per-pair-normalise", action="store_true",
        help="scale each pair's correlation to unit peak before summing",
    )
    parser.add_argument(
        "--lag-upsample", type=int, default=DEFAULT_LAG_UPSAMPLE,
        help="sinc-interpolation factor on the GCC lag axis (1 = integer lags only)",
    )
    parser.add_argument(
        "--srp-frame-sec", type=float, default=DEFAULT_SRP_FRAME_SEC,
        help="analysis frame for the framed GCC-PHAT average",
    )
    parser.add_argument(
        "--point-interpolation", default="parabolic", choices=["parabolic", "map"],
        help="sub-bin parabolic peak, or the MAP bin center",
    )
    parser.add_argument(
        "--smooth-alpha", type=float, default=0.0,
        help="causal EMA weight on the previous response (0 = stateless per chunk)",
    )
    parser.add_argument(
        "--silence-floor-db", type=float, default=DEFAULT_SILENCE_FLOOR_DB,
        help="answer uniform when a window is this far below the causal running peak",
    )
    parser.add_argument(
        "--report-band-split", action="store_true",
        help="also evaluate below/above the aliasing limit and record the disagreement",
    )
    parser.add_argument(
        "--fit-distance-labels", default=None,
        help="JSONL of training-split {session_id, distance_m} to fit the ridge model",
    )
    parser.add_argument(
        "--ridge-alpha", type=float, default=1.0, help="ridge penalty for the distance fit"
    )
    parser.add_argument(
        "--save-distance-model", default=None, help="write the fitted model to this JSON path"
    )
    return parser


def collect_options(args: argparse.Namespace) -> dict[str, Any]:
    """Gather the algorithm options into one dict, as passed around internally."""
    return {
        "band": args.band,
        "method": args.srp_method,
        "srp_temperature": args.srp_temperature,
        "freq_min_hz": args.freq_min_hz,
        "freq_max_hz": args.freq_max_hz,
        "per_pair_normalise": args.per_pair_normalise,
        "lag_upsample": args.lag_upsample,
        "frame_sec": args.srp_frame_sec,
        "point_interpolation": args.point_interpolation,
        "smooth_alpha": args.smooth_alpha,
        "silence_floor_db": args.silence_floor_db,
        "report_band_split": args.report_band_split,
        "lookahead_sec": args.lookahead_sec,
        "prob_decimals": args.prob_decimals,
        "include_record_type": not args.omit_record_type,
    }


def resolve_distance_model(
    args: argparse.Namespace, cases: Sequence[CaseSpec], options: Mapping[str, Any]
) -> DistanceModel:
    """Pick the distance model from the CLI arguments, warning on the fallback."""
    if args.fit_distance_labels:
        labels = load_distance_labels(args.fit_distance_labels)
        model = fit_distance_model(
            cases, labels, options, args.audio_backend, alpha=args.ridge_alpha
        )
        print(
            f"note: fitted ridge distance model on {model.n_train} labelled case(s), "
            f"residual sigma(log) = {model.sigma_log:.4f}",
            file=sys.stderr,
        )
        if args.save_distance_model:
            Path(args.save_distance_model).write_text(
                json.dumps(model.to_json(), indent=2), encoding="utf-8"
            )
        return model
    if args.distance_model:
        model = load_distance_model(args.distance_model)
        print(
            f"note: distance model {model.kind!r} loaded from {args.distance_model}",
            file=sys.stderr,
        )
        return model
    print(
        "note: no --distance-model and no --fit-distance-labels, so distance is the "
        "UNIFORM fallback (scores log K_r by construction). The output metadata "
        "records distance_source='uniform'.",
        file=sys.stderr,
    )
    return DistanceModel(kind="uniform", note="no prior and no fit supplied")


def run(args: argparse.Namespace) -> int:
    """Run Baseline 2 over an eval directory and write ``predictions.jsonl``."""
    scope = resolve_scope(args.scope)
    masks = resolve_masks(args.masks)
    options = collect_options(args)
    cases = discover_cases(
        args.eval_dir, manifest=args.manifest, track=args.track, masks=masks, limit=args.limit
    )
    distance_model = resolve_distance_model(args, cases, options)
    run_asr = not args.no_asr
    asr = FrozenASR(model_name=args.asr_model, synthetic=args.synthetic_asr)
    if args.synthetic_asr and run_asr:
        print(SYNTHETIC_ASR_WARNING, file=sys.stderr)
    if args.band != BANDS[0]:
        print(
            f"note: --band {args.band} is a diagnostic; the reported baseline uses "
            f"{BANDS[0]} integration.",
            file=sys.stderr,
        )

    records: list[dict[str, Any]] = []
    sessions: dict[str, AudioSession] = {}
    for case in cases:
        if case.audio_path not in sessions:
            sessions.clear()
            sessions[case.audio_path] = load_audio(case.audio_path, args.audio_backend)
        session = sessions[case.audio_path]
        if case.geometry_source != "manifest":
            print(
                f"warning: {case.case_id} has no microphone coordinates in the case "
                "list; falling back to the default circular array. SRP-PHAT is only "
                "as correct as the geometry.",
                file=sys.stderr,
            )
        case_records, diagnostics = predict_case(
            case=case,
            session=session,
            asr=asr,
            distance_model=distance_model,
            scope=scope,
            options=options,
            run_asr=run_asr,
        )
        records.extend(case_records)
        print(
            f"  {case.case_id}: M={case.n_active_channels}, "
            f"{len(case_records) - 1} timestep records, "
            f"uniform frames={diagnostics['uniform_frames']}, "
            f"rtf={diagnostics['rtf']:.3f}",
            file=sys.stderr,
        )
    written = write_predictions(args.out, records)
    print(
        f"{SYSTEM_NAME}: {len(cases)} case(s), {written} record(s) -> {args.out}",
        file=sys.stderr,
    )
    if args.synthetic_asr and run_asr:
        print(SYNTHETIC_ASR_WARNING, file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Self test: does SRP-PHAT actually find a known direction?
# ---------------------------------------------------------------------------


def synthesise_plane_wave(
    azimuth_deg: float,
    mic_positions_m: Sequence[Sequence[float]],
    channels: Sequence[int],
    sample_rate_hz: int = 16000,
    duration_sec: float = 1.0,
    noise_db: float = -40.0,
    seed: int = 0,
) -> FloatArray:
    """Synthesise a far-field plane wave arriving from ``azimuth_deg``.

    Channel ``m`` receives ``s(t - t_m)`` with ``t_m = -(p_m . u) / c``, so the
    microphone nearer the source leads -- the same convention
    :func:`steering_tdoa_sec` steers with.  Uncorrelated sensor noise at
    ``noise_db`` relative to the source is added so the test also exercises the
    PHAT whitening on a non-ideal signal.

    >>> import numpy as np
    >>> from baselines import default_mic_positions_m
    >>> block = synthesise_plane_wave(0.0, default_mic_positions_m(), (0, 2))
    >>> block.shape
    (2, 16000)
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = int(round(duration_sec * sample_rate_hz))
    guard = 4 * DELAY_PAD_SAMPLES
    source = rng.normal(0.0, 1.0, n + 2 * guard)
    positions = np.asarray(mic_positions_m, dtype=np.float64)
    angle = math.radians(azimuth_deg)
    unit = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    noise_scale = 10.0 ** (noise_db / 20.0)
    rows: list[FloatArray] = []
    for channel in channels:
        arrival_sec = -float(positions[channel] @ unit) / spec.SPEED_OF_SOUND_M_S
        delayed = fractional_delay(source, arrival_sec * sample_rate_hz, pad=guard)
        rows.append(
            delayed[guard : guard + n] + rng.normal(0.0, noise_scale, n)
        )
    return np.asarray(rows, dtype=np.float64)


def _is_uniform(probs: Sequence[float], tolerance: float = 1.0e-9) -> bool:
    """True when a distribution is flat, i.e. the model abstained.

    Used by the self-test to separate frames where SRP-PHAT actually produced
    an estimate from frames it declined to answer (``M = 1``, silence, or a
    window shorter than one hop).  Scoring a declined frame as a direction
    error would hide whether the estimator works.

    >>> _is_uniform([0.25] * 4)
    True
    >>> _is_uniform([0.4, 0.2, 0.2, 0.2])
    False
    """
    return (max(probs) - min(probs)) <= tolerance


def _smoke_direction_finding() -> list[tuple[str, bool]]:
    """Check that SRP-PHAT recovers known plane-wave directions.

    Returns a list of ``(label, passed)`` checks.  The multi-microphone cases
    must land within one azimuth bin of the truth.  The two-microphone cases
    are allowed to land on either side of the pair axis, because a single pair
    measures only the projection of the direction onto its axis -- the mirror
    ambiguity is a property of the array, not a bug, and it is exactly the
    bimodality :data:`spec.INTERVAL_RULE` was chosen to express.
    """
    import numpy as np

    from baselines import default_mic_positions_m

    positions = default_mic_positions_m()
    sample_rate = 16000
    checks: list[tuple[str, bool]] = []

    for mask in ("1111", "1110"):
        channels = spec.mask_to_channels(mask)
        for truth_deg in (55.0, -125.0, 5.0, 23.0, 175.0):
            block = synthesise_plane_wave(truth_deg, positions, channels, sample_rate)
            response = srp_phat_response(block, positions, channels, sample_rate)
            peak = int(np.argmax(response))
            expected = spec.azimuth_bin_index(truth_deg)
            delta_bins = min(
                abs(peak - expected), spec.K_AZIMUTH - abs(peak - expected)
            )
            checks.append(
                (
                    f"mask {mask} az={truth_deg:+.0f} -> bin {peak} "
                    f"(expected {expected}, off by {delta_bins})",
                    delta_bins <= 1,
                )
            )

    # The headline case from the task: +55 degrees must land on bin 23.
    channels = spec.mask_to_channels("1111")
    block = synthesise_plane_wave(55.0, positions, channels, sample_rate)
    response = srp_phat_response(block, positions, channels, sample_rate)
    checks.append(
        (
            f"+55 deg with the full array lands on bin 23 (got {int(np.argmax(response))})",
            int(np.argmax(response)) == 23,
        )
    )
    probs = response_to_distribution(response, 1.0, 6)
    checks.append(
        (
            "softmax of that response sums to 1 within PROB_SUM_TOLERANCE",
            abs(math.fsum(probs) - 1.0) < spec.PROB_SUM_TOLERANCE,
        )
    )
    checks.append(
        (
            "its MAP bin equals the response peak",
            probs.index(max(probs)) == int(np.argmax(response)),
        )
    )
    checks.append(
        (
            f"parabolic point stays inside bin 23 ({parabolic_peak_deg(response):.2f} deg)",
            spec.azimuth_bin_index(parabolic_peak_deg(response)) == 23,
        )
    )

    # The two implementations must agree.
    reference = srp_phat_response(
        block, positions, channels, sample_rate, method="freq_steer"
    )
    checks.append(
        (
            "gcc_interp and freq_steer agree on the peak bin",
            int(np.argmax(reference)) == int(np.argmax(response)),
        )
    )
    checks.append(
        (
            "gcc_interp and freq_steer correlate above 0.99",
            float(np.corrcoef(response, reference)[0, 1]) > 0.99,
        )
    )

    # Below the half-wavelength limit the array is unambiguous but narrowband.
    below = srp_phat_response(
        block, positions, channels, sample_rate, band="below_aliasing"
    )
    below_delta = min(
        abs(int(np.argmax(below)) - 23), spec.K_AZIMUTH - abs(int(np.argmax(below)) - 23)
    )
    checks.append(
        (
            f"below_aliasing band still finds +55 within one bin (bin {int(np.argmax(below))})",
            below_delta <= 1,
        )
    )
    above = srp_phat_response(
        block, positions, channels, sample_rate, band="above_aliasing"
    )
    checks.append(
        (
            "above_aliasing band produces a non-degenerate response",
            bool(np.any(above)) and float(np.std(above)) > 0.0,
        )
    )

    # Two microphones. A single pair only measures the projection of the
    # direction onto its own axis, so the true direction and its reflection
    # about that axis give identical TDOAs and the response is exactly
    # bimodal: for a pair whose axis points along phi, az and 2*phi - az are
    # indistinguishable. Which of the two tied lobes wins is decided by sensor
    # noise, so the test accepts either -- and the ambiguity itself is the
    # reason spec.INTERVAL_RULE builds a possibly non-contiguous HPD set.
    truth_deg = 55.0
    for mask in ("1010", "1100"):
        pair_channels = spec.mask_to_channels(mask)
        first, second = pair_channels
        axis_deg = math.degrees(
            math.atan2(
                positions[second][1] - positions[first][1],
                positions[second][0] - positions[first][0],
            )
        )
        mirror_deg = 2.0 * axis_deg - truth_deg
        block2 = synthesise_plane_wave(truth_deg, positions, pair_channels, sample_rate)
        response2 = srp_phat_response(block2, positions, pair_channels, sample_rate)
        peak2 = int(np.argmax(response2))
        allowed: set[int] = set()
        for candidate in (truth_deg, mirror_deg):
            center = spec.azimuth_bin_index(candidate)
            allowed.update(
                {(center - 1) % spec.K_AZIMUTH, center, (center + 1) % spec.K_AZIMUTH}
            )
        checks.append(
            (
                f"mask {mask} (axis {axis_deg:+.0f} deg) az=+55 lands on the true "
                f"lobe or its mirror at {mirror_deg:+.0f} deg, bins {sorted(allowed)} "
                f"(got {peak2})",
                peak2 in allowed,
            )
        )
        # Both lobes must carry comparable mass: that is what makes the
        # submitted distribution honest about the ambiguity instead of
        # pretending to a single answer.
        probs2 = response_to_distribution(response2, 1.0, 6)
        true_bin = spec.azimuth_bin_index(truth_deg)
        mirror_bin = spec.azimuth_bin_index(mirror_deg)
        ratio = min(probs2[true_bin], probs2[mirror_bin]) / max(
            probs2[true_bin], probs2[mirror_bin]
        )
        checks.append(
            (
                f"mask {mask} keeps both lobes in the distribution "
                f"(mass ratio {ratio:.2f} >= 0.5)",
                ratio >= 0.5,
            )
        )

    # One microphone: no pair, so no response and a uniform answer.
    mono_block = synthesise_plane_wave(55.0, positions, (0,), sample_rate)
    mono_response = srp_phat_response(mono_block, positions, (0,), sample_rate)
    mono_probs = response_to_distribution(mono_response, 1.0, 6)
    checks.append(
        (
            "M=1 yields an all-zero response and the uniform distribution",
            not bool(np.any(mono_response)) and _is_uniform(mono_probs),
        )
    )

    # Delay-and-sum must prefer the true direction over the opposite one.
    channels = spec.mask_to_channels("1111")
    block = synthesise_plane_wave(55.0, positions, channels, sample_rate, noise_db=-6.0)
    on_axis = rms(delay_and_sum(block, positions, channels, sample_rate, 55.0))
    off_axis = rms(delay_and_sum(block, positions, channels, sample_rate, -125.0))
    checks.append(
        (
            f"delay-and-sum on the true direction beats the opposite "
            f"({on_axis:.4f} > {off_axis:.4f})",
            on_axis > off_axis,
        )
    )

    # TDOA sign convention: the microphone nearer the source must lead.
    tau = steering_tdoa_sec(positions, [(0, 2)], [0.0])
    checks.append(("TDOA sign: channel 0 leads for a source at 0 deg", float(tau[0, 0]) < 0.0))
    return checks


def self_test() -> int:
    """Run doctests, the direction-finding smoke tests and an end-to-end run."""
    import doctest
    import tempfile
    import wave

    import numpy as np

    import baselines
    import baselines.single_channel as baseline1

    failures = 0
    for module in (baselines, baseline1, sys.modules[__name__]):
        result = doctest.testmod(module, verbose=False)
        failures += result.failed
        print(f"doctests {module.__name__}: {result.attempted - result.failed}/{result.attempted}")
    if failures:
        return 1

    print("SRP-PHAT direction finding:")
    checks = _smoke_direction_finding()
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    # End-to-end CLI run on a synthetic 4-channel session from a known direction.
    from baselines import default_mic_positions_m

    sample_rate = 16000
    positions = default_mic_positions_m()
    truth_deg = 55.0
    block = synthesise_plane_wave(
        truth_deg,
        positions,
        tuple(range(spec.N_CHANNELS_MAX)),
        sample_rate,
        duration_sec=spec.SESSION_LENGTH_SEC,
        noise_db=-30.0,
    )
    block = block * 0.1
    block[:, : block.shape[1] // 4] = 0.0  # leading silence for the VAD and the gate
    pcm = np.clip(block.T * 32767.0, -32768, 32767).astype(np.int16).ravel()
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "sess_000.wav"
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(spec.N_CHANNELS_MAX)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
        manifest = Path(tmp) / "manifest.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "session_id": "sess_000",
                    "audio_path": "sess_000.wav",
                    "mic_positions_m": [list(p) for p in positions],
                    "sample_rate_hz": sample_rate,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        out_path = Path(tmp) / "predictions.jsonl"
        args = build_parser().parse_args(
            [
                "--eval-dir", tmp,
                "--out", str(out_path),
                "--scope", spec.OFFICIAL_SCOPES[0],
                "--track", str(min(spec.MAX_SPEAKERS_PER_TRACK)),
                "--synthetic-asr",
                "--report-band-split",
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
    by_mask: dict[str, list[dict[str, Any]]] = {}
    for record in timesteps:
        by_mask.setdefault(spec.parse_case_id(record["case_id"])[1], []).append(record)

    def hit_rate(mask: str) -> float:
        """Fraction of answered frames whose azimuth point is within one bin."""
        rows = [r for r in by_mask[mask] if not _is_uniform(r["azimuth_probs"])]
        if not rows:
            return 0.0
        return sum(
            1
            for r in rows
            if spec.circular_error_deg(r["pred_azimuth_deg"], truth_deg)
            <= spec.AZIMUTH_BIN_WIDTH_DEG
        ) / len(rows)

    e2e: list[tuple[str, bool]] = [
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
        (
            "every distribution sums to 1 within tolerance",
            all(
                abs(math.fsum(r[key]) - 1.0) < spec.PROB_SUM_TOLERANCE
                for r in timesteps
                for key in ("azimuth_probs", "distance_probs")
            ),
        ),
        (
            "M=1 masks stay uniform in azimuth at every grid point",
            all(
                _is_uniform(r["azimuth_probs"])
                for mask in ("1000", "0010")
                for r in by_mask[mask]
            ),
        ),
        (
            "the leading silence is answered uniform, not guessed",
            all(
                _is_uniform(r["azimuth_probs"])
                for r in by_mask["1111"]
                if r["eval_time_sec"] <= spec.SESSION_LENGTH_SEC / 4.0
            ),
        ),
        (
            f"mask 1111 within one bin of +55 on >=95% of answered frames "
            f"({hit_rate('1111'):.0%})",
            hit_rate("1111") >= 0.95,
        ),
        (
            f"mask 1110 within one bin of +55 on >=95% of answered frames "
            f"({hit_rate('1110'):.0%})",
            hit_rate("1110") >= 0.95,
        ),
        (
            f"mask 1010 shows the two-mic mirror ambiguity, not a false peak "
            f"({hit_rate('1010'):.0%} on +55)",
            hit_rate("1010") < 0.95,
        ),
        (
            "band-split diagnostics recorded for the multichannel cases",
            all(
                "band_split_diagnostics" in r["meta"]
                for r in finals
                if r["meta"]["n_active_channels"] > 1
            ),
        ),
        (
            "the beamformer is used whenever M > 1",
            all(
                r["meta"]["beamformer"] == "delay_and_sum"
                for r in finals
                if r["meta"]["n_active_channels"] > 1
            ),
        ),
        (
            "geometry came from the manifest, not the fallback",
            all(r["meta"]["geometry_source"] == "manifest" for r in finals),
        ),
        (
            "both baselines report the same frozen ASR backend",
            len({r["meta"]["asr_backend"] for r in finals}) == 1,
        ),
    ]
    print("end-to-end:")
    for label, passed in e2e:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    ok = all(passed for _, passed in checks) and all(passed for _, passed in e2e)
    print(f"baseline 2 self-test: {'PASS' if ok else 'FAIL'}")
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

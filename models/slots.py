"""The shared backend: online cosine slot matching, threshold tuning, scoring.

Every system ends here, so the ladder differs only in its embeddings.  Per
case, walking the 80 grid instants in order:

* speech instant (causal VAD): join the most similar slot if the cosine to its
  running-sum centroid exceeds ``theta`` -- or if ``max_slots`` slots are
  already open -- otherwise open a new slot; update the joined centroid;
* non-speech instant: report the most similar existing slot, update nothing.

``theta`` is tuned on validation for each system.  The search runs every
threshold at once on the Gram matrix (one pass over time, vectorised over
thresholds), adapted from the voice-only measurement of 2026-10-30.

Spk Acc here is the fraction of labelled instants whose slot maps to the right
talker under the best one-to-one slot-talker assignment.  Since spec 0.3.0 the
official ``evaluate.py`` scores Spk Acc under that same frame-optimal
assignment, never the cpWER one, so per case the two numbers agree; this one
is for model selection.
"""

from __future__ import annotations

from itertools import permutations
from typing import Sequence

import numpy as np


def online_slots(emb: np.ndarray, speech: np.ndarray, thetas: Sequence[float], max_slots: int) -> np.ndarray:
    """Slot label per instant for every threshold: ``[len(thetas), K]``.

    >>> e = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)
    >>> online_slots(e, np.ones(5, bool), [0.5], 4).tolist()
    [[0, 0, 1, 1, 0]]
    >>> online_slots(e, np.ones(5, bool), [1.5], 5).tolist()  # never joins: one slot per instant
    [[0, 1, 2, 3, 4]]
    >>> online_slots(e, np.ones(5, bool), [1.5], 2).tolist()  # cap reached: join the nearest
    [[0, 1, 0, 0, 1]]
    >>> online_slots(e, np.array([1, 0, 1, 0, 1], bool), [0.5], 4).tolist()  # silence: no update
    [[0, 0, 1, 1, 0]]
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    E = np.asarray(emb, dtype=np.float64)
    K = E.shape[0]
    G = E @ E.T
    n_th = len(thetas)
    dots = np.zeros((n_th, max_slots, K))
    norm2 = np.zeros((n_th, max_slots))
    count = np.zeros(n_th, dtype=np.int64)
    labels = np.zeros((n_th, K), dtype=np.int64)
    rows = np.arange(n_th)
    slot_ids = np.arange(max_slots)[None, :]
    for t in range(K):
        open_ = slot_ids < count[:, None]
        sims = np.where(open_, dots[:, :, t] / np.sqrt(np.maximum(norm2, 1e-12)), -np.inf)
        best = sims.argmax(1)
        has = count > 0
        if speech[t]:
            join = has & ((sims[rows, best] > thetas) | (count >= max_slots))
            target = np.where(join, best, count)
            norm2[rows, target] += 2.0 * dots[rows, target, t] + G[t, t]
            dots[rows, target, :] += G[t][None, :]
            count = count + (~join)
            labels[:, t] = target
        else:
            labels[:, t] = np.where(has, best, 0)
    return labels


def best_mapping(truth: np.ndarray, pred: np.ndarray) -> dict[int, int]:
    """Slot -> talker under the one-to-one assignment with the most labelled matches.

    >>> sorted(best_mapping(np.array([0, 0, 1, 1, -1]), np.array([1, 1, 0, 0, 3])).items())
    [(0, 1), (1, 0)]
    """
    keep = truth >= 0
    t, p = truth[keep], pred[keep]
    if t.size == 0:
        return {}
    C = np.zeros((int(t.max()) + 1, int(p.max()) + 1), dtype=np.int64)
    np.add.at(C, (t, p), 1)
    best, best_map = -1, {}
    if C.shape[0] <= C.shape[1]:
        for cols in permutations(range(C.shape[1]), C.shape[0]):
            score = sum(int(C[r, c]) for r, c in enumerate(cols))
            if score > best:
                best, best_map = score, {c: r for r, c in enumerate(cols)}
    else:
        for rows in permutations(range(C.shape[0]), C.shape[1]):
            score = sum(int(C[r, c]) for c, r in enumerate(rows))
            if score > best:
                best, best_map = score, {c: r for c, r in enumerate(rows)}
    return best_map


def accuracy_near_changes(truth: np.ndarray, pred: np.ndarray, window: int = 4) -> tuple[float | None, float | None]:
    """``(accuracy within `window` instants after a change of talker, accuracy elsewhere)``.

    Separates the lag after a talker change (where Chunk-Reset and Carry-Over
    are expected to differ) from the steady part of a turn, under the case's
    one best mapping.

    >>> truth = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    >>> accuracy_near_changes(truth, pred, window=4)
    (0.5, 1.0)
    """
    mapping = best_mapping(truth, pred)
    mapped = np.array([mapping.get(int(p), -2) for p in pred])
    near = np.zeros(len(truth), dtype=bool)
    previous = -1
    for k, talker in enumerate(truth):
        if talker < 0:
            continue
        if previous >= 0 and talker != previous:
            near[k : k + window] = True
        previous = talker
    labelled = truth >= 0
    out = []
    for sel in (labelled & near, labelled & ~near):
        out.append(float(np.mean(mapped[sel] == truth[sel])) if sel.any() else None)
    return out[0], out[1]


def best_assignment_accuracy(truth: np.ndarray, pred: np.ndarray) -> float | None:
    """Fraction of labelled instants (``truth >= 0``) right under the best mapping.

    The one-to-one slot-to-talker mapping is found by exhaustive search, like
    ``evaluate.solve_permutations``: at most four talkers and a handful of slots
    per case, so there is no need for an assignment solver.

    >>> best_assignment_accuracy(np.array([0, 0, 1, 1, -1]), np.array([1, 1, 0, 0, 3]))
    1.0
    >>> best_assignment_accuracy(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]))
    0.5
    >>> best_assignment_accuracy(np.array([0, 1, 2, 2]), np.array([2, 0, 1, 1]))
    1.0
    """
    keep = truth >= 0
    if not keep.any():
        return None
    t, p = truth[keep], pred[keep]
    C = np.zeros((int(t.max()) + 1, int(p.max()) + 1), dtype=np.int64)
    np.add.at(C, (t, p), 1)
    if C.shape[0] > C.shape[1]:
        C = C.T
    rows = range(C.shape[0])
    best = max(sum(int(C[r, c]) for r, c in zip(rows, cols)) for cols in permutations(range(C.shape[1]), C.shape[0]))
    return best / float(keep.sum())


def centroid_accuracy(emb: np.ndarray, truth: np.ndarray) -> float | None:
    """Threshold-free separability: each labelled instant against the talkers'
    leave-one-out centroids (oracle labels).  Measures the embedding, not the
    clustering.

    >>> e = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
    >>> centroid_accuracy(e, np.array([0, 0, 1, 1]))
    1.0
    """
    keep = truth >= 0
    if keep.sum() < 2:
        return None
    E, y = np.asarray(emb, dtype=np.float64)[keep], truth[keep]
    speakers = np.unique(y)
    if len(speakers) < 2:
        return None
    sums = np.stack([E[y == s].sum(0) for s in speakers])
    counts = np.array([(y == s).sum() for s in speakers], dtype=np.float64)
    correct = 0
    for i in range(len(y)):
        own = int(np.searchsorted(speakers, y[i]))
        cents = sums.copy()
        n = counts.copy()
        cents[own] -= E[i]
        n[own] -= 1
        valid = n > 0
        cents = cents / np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12)
        score = np.where(valid, cents @ E[i], -np.inf)
        correct += int(score.argmax() == own)
    return correct / len(y)


def theta_grid(spec_: Sequence[float]) -> np.ndarray:
    start, stop, step = spec_
    return np.round(np.arange(start, stop + step / 2, step), 6)


def tune_theta(cases: Sequence[dict], thetas: np.ndarray, max_slots: int, label_key: str = "labels_eval",
               emb_key: str = "emb") -> dict:
    """Pick the threshold with the best mean (macro over cases) Spk Acc.

    ``cases``: dicts with an embedding ``[80, D]`` under ``emb_key``, ``speech``
    ``[80]`` and the label grid.  Returns ``{"theta", "acc", "curve"}``.
    """
    totals = np.zeros(len(thetas))
    n = 0
    for case in cases:
        labels = online_slots(case[emb_key], case["speech"], thetas, max_slots)
        accs = [best_assignment_accuracy(case[label_key], row) for row in labels]
        if accs[0] is None:
            continue
        totals += np.asarray(accs)
        n += 1
    curve = totals / max(n, 1)
    i = int(curve.argmax())
    return {"theta": float(thetas[i]), "acc": float(curve[i]), "curve": curve.tolist()}


def causal_ceiling(labels_causal: np.ndarray, labels_eval: np.ndarray) -> float | None:
    """Best Spk Acc any causal system can reach on one case.

    The official label at ``t_k`` describes ``[t_k, t_k + 0.25)``, after the
    last sample a causal system may read.  An oracle that always names the last
    talker it has *heard* (causal label, carried through silences) still misses
    every instant whose window already holds the next talker's onset.

    >>> causal_ceiling(np.array([-1, 0, 0, -1, 1, 1]), np.array([0, 0, -1, 1, 1, -1]))  # misses t_0 and the onset
    0.5
    """
    keep = labels_eval >= 0
    if not keep.any():
        return None
    heard = np.zeros_like(labels_causal)
    last = -1
    for k, v in enumerate(labels_causal):
        last = v if v >= 0 else last
        heard[k] = last
    return float(np.mean(heard[keep] == labels_eval[keep]))


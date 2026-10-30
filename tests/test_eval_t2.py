"""The two-talker eval set generator (make_eval_t2.py), end to end on placeholder speech.

numpy only (the fallback room engine and synthetic voices), so it runs in
``make test``.  It checks what the scorer and the comparison rely on: the
official scorer accepts the set and gives a perfect score to the truth, every
labelled instant names the talker whose turn holds it at that turn's position,
the main and static sets are the same sessions, no utterance is used twice, and
the plan is balanced.

Run with::

    python3 -m unittest tests.test_eval_t2 -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"numpy not installed: {exc}") from exc

import evaluate  # noqa: E402
import make_eval_t2  # noqa: E402
import spec  # noqa: E402


class TestEvalT2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.main = Path(cls._tmp.name) / "t2"
        cls.static = Path(cls._tmp.name) / "t2-static"
        cls.n_sessions, cls.seed = 6, 11
        cls.manifests = make_eval_t2.generate(cls.main, cls.static, cls.n_sessions, cls.seed, None, "fallback_ism",
                                              True, 1, log=lambda *_: None)
        index = make_eval_t2._synthetic_index()
        cls.plans = {p.session_id: p for p in make_eval_t2.build_plan(cls.n_sessions, cls.seed, sorted(index))}
        cls.transcripts = {utt: text for rows in index.values() for utt, _, text in rows}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _rows(self, folder: Path, name: str) -> list[dict]:
        return [json.loads(line) for line in (folder / name).read_text(encoding="utf-8").splitlines()]

    def test_the_scorer_gives_the_truth_a_perfect_score(self) -> None:
        for folder in (self.main, self.static):
            preds = []
            for row in self._rows(folder, "ground_truth.jsonl"):
                speaker = {f["eval_time_sec"]: f["speaker"] for f in row["frames"]}
                preds += [{"case_id": row["case_id"], "eval_time_sec": k * spec.DELTA_T_SEC,
                           "pred_speaker": speaker.get(k * spec.DELTA_T_SEC, "spk_0")} for k in range(spec.N_GRID_POINTS)]
                preds.append({"case_id": row["case_id"], "final_transcripts": row["final_transcripts_reference"]})
            path = folder / "oracle.jsonl"
            path.write_text("\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8")
            metrics = evaluate.evaluate_files(path, folder / "ground_truth.jsonl", 2)["metrics"]
            self.assertEqual(metrics["speaker_attribution_accuracy"], 1.0)
            self.assertEqual(metrics["cpwer"], 0.0)

    def test_main_and_static_are_the_same_sessions(self) -> None:
        main, static = self._rows(self.main, "ground_truth.jsonl"), self._rows(self.static, "ground_truth.jsonl")
        self.assertEqual([r["case_id"] for r in main], [r["case_id"] for r in static])
        for a, b in zip(main, static):
            self.assertEqual([(f["eval_time_sec"], f["speaker"]) for f in a["frames"]],
                             [(f["eval_time_sec"], f["speaker"]) for f in b["frames"]])
            self.assertEqual([(t["speaker"], t["start_sec"], t["end_sec"]) for t in a["turns"]],
                             [(t["speaker"], t["start_sec"], t["end_sec"]) for t in b["turns"]])
            self.assertEqual(a["final_transcripts_reference"], b["final_transcripts_reference"])
            self.assertEqual(a["talkers"], b["talkers"])
            self.assertTrue(all(t["position_copy"] == 0 for t in b["turns"]))
            self.assertIn(a["snr_db"], make_eval_t2.SNR_LEVELS_DB)
            self.assertIsNone(b["snr_db"])
        import wave

        def samples(folder: Path) -> np.ndarray:
            with wave.open(str(folder / "audio" / "t2_0000.wav")) as w:
                return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")

        self.assertFalse(np.array_equal(samples(self.main), samples(self.static)))

    def test_meta_and_manifest(self) -> None:
        meta = self._rows(self.main, "meta.jsonl")
        self.assertEqual(len(meta), self.n_sessions * len(spec.EVAL_MASKS))
        self.assertTrue(all(m["track"] == 2 and m["max_speakers"] == 2 for m in meta))
        manifest = json.loads((self.main / "manifest.json").read_text())
        self.assertFalse(manifest["official"])  # fallback engine, synthetic speech
        self.assertEqual(manifest["paired_with"], self.static.name)
        for rel, info in manifest["files"].items():
            self.assertEqual(info["bytes"], (self.main / rel).stat().st_size)

    def test_labels_name_talkers_in_order_of_first_turn(self) -> None:
        for row in self._rows(self.main, "ground_truth.jsonl"):
            self.assertEqual(row["turns"][0]["speaker"], "spk_0")
            self.assertEqual(row["frames"][0]["speaker"], "spk_0")
            self.assertEqual(sorted(row["final_transcripts_reference"]), ["spk_0", "spk_1"])

    def test_each_instant_names_the_talker_of_its_turn_at_that_turns_position(self) -> None:
        for folder in (self.main, self.static):
            for row in self._rows(folder, "ground_truth.jsonl"):
                plan = self.plans[row["session_id"]]
                talker_of = {name: next(i for i, t in enumerate(plan.talkers) if t.speaker_id == info["speaker_id"])
                             for name, info in row["talkers"].items()}
                for name, info in row["talkers"].items():
                    self.assertEqual(row["final_transcripts_reference"][name], self.transcripts[info["utt_id"]])
                self.assertEqual({f["speaker"] for f in row["frames"]}, {"spk_0", "spk_1"})
                for f in row["frames"]:
                    t0, t1 = f["eval_time_sec"], f["eval_time_sec"] + spec.DELTA_T_SEC
                    turns = [t for t in row["turns"] if t["speaker"] == f["speaker"]
                             and t["start_sec"] < t1 and t["end_sec"] > t0]
                    self.assertTrue(turns, f"{row['case_id']} t={t0}: no {f['speaker']} turn there")
                    turn = max(turns, key=lambda t: min(t1, t["end_sec"]) - max(t0, t["start_sec"]))
                    xy = plan.talkers[talker_of[f["speaker"]]].variants_xy[turn["position_copy"]]
                    where = make_eval_t2._placement("p", "p", *xy)
                    self.assertAlmostEqual(f["azimuth_deg"], where.azimuth_deg, places=9)
                    self.assertAlmostEqual(f["distance_m"], where.distance_m, places=9)

    def test_no_utterance_is_used_twice_and_returning_talkers_move(self) -> None:
        rows = {r["session_id"]: r for r in self._rows(self.main, "ground_truth.jsonl")}
        utts = [info["utt_id"] for r in rows.values() for info in r["talkers"].values()]
        self.assertEqual(len(utts), len(set(utts)))
        for r in rows.values():
            for name in r["talkers"]:
                copies = [t["position_copy"] for t in r["turns"] if t["speaker"] == name]
                self.assertTrue(all(a != b for a, b in zip(copies, copies[1:])), r["session_id"])

    def test_the_full_plan_is_balanced(self) -> None:
        report = make_eval_t2.verify_plan(make_eval_t2.build_plan(200, make_eval_t2.DEFAULT_SEED,
                                                                  [f"s{i:02d}" for i in range(40)]))
        self.assertEqual(report["sessions_per_speaker"], {"min": 10, "max": 10, "n_speakers": 40})
        self.assertEqual(set(report["turns"]), {2, 3, 4})


if __name__ == "__main__":
    unittest.main()

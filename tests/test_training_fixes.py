"""Regressions for resuming on new code, extended schedules, room-bank versions and run locks.

* a finished run is re-validated on code from another commit without
  ``--allow-code-change``, while further training on new code is still refused;
* a resume with another ``max_steps`` records the restarted learning-rate
  schedule in checkpoints and ``summary.json``, and a checkpoint written before
  that record has its history rebuilt from ``metrics.jsonl``, marked inferred;
* a bank build without pyroomacoustics never erases the recorded simulator version;
* a second ``train.py`` / ``eval_t2.py`` on the same directory stops at once.
"""

from __future__ import annotations

import contextlib
import dataclasses
import doctest
import errno
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
except ImportError:  # pragma: no cover - minimal scoring installation
    torch = None  # type: ignore[assignment]

try:
    import fcntl  # noqa: F401
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

import training.runlock
from training.rirbank import BankSpec, _shard_ok, build_bank
from training.runlock import exclusive_lock

needs_torch = unittest.skipUnless(torch is not None, "torch not installed")
needs_fcntl = unittest.skipUnless(fcntl is not None, "no fcntl (Windows): runs are not locked")


def _validation_result(score: float) -> dict[str, Any]:
    """A quick validation result whose selection score is ``score``."""
    return {"spk_acc": score, "spk_acc_by_M": {"2": score}, "spk_acc_n2_cap2": score,
            "spk_acc_n2_cap2_by_M": {"2": score}, "theta_by_M": {"2": 0.5}, "theta_official_by_M": {"2": 0.5}}


def _bare_run(directory: Path, *, max_steps: int, commit: str, warmup: int = 5, allow_code_change: bool = False):  # noqa: ANN202
    """A Run with real checkpoints, optimizer and cosine schedule, but no corpus, bank or speech."""
    from training.config import config_fingerprint, load_config_dict
    from training.engine import Checkpoints, Run, lr_factor
    from training.loss import SessionCentroidLoss

    run = Run.__new__(Run)
    run.cfg = load_config_dict({"name": "fixes", "train": {"max_steps": max_steps, "warmup_steps": warmup}})
    run.run_dir = directory
    run.device = "cpu"
    run.logs = []
    run.log = run.logs.append
    run.overrides = {}
    run.fingerprint = config_fingerprint(run.cfg)
    run.commit = commit
    run.commits = [commit]
    run.allow_code_change = allow_code_change
    run.amp_dtype = None
    run.system = torch.nn.Linear(1, 1, bias=False)
    run.system.trainable = True
    run.loss_fn = SessionCentroidLoss()
    run.pos_loss_fn = SessionCentroidLoss()
    parameters = list(run.system.parameters()) + list(run.loss_fn.parameters()) + list(run.pos_loss_fn.parameters())
    run.opt = torch.optim.AdamW(parameters, lr=run.cfg.train.lr)
    total = max(max_steps, 1)
    run.sched = torch.optim.lr_scheduler.LambdaLR(run.opt, lambda step: lr_factor(step, warmup, total))
    run.scaler = torch.amp.GradScaler("cuda", enabled=False)
    run.ckpts = Checkpoints(directory, keep=3)
    run.step = 0
    run.best = -math.inf
    return run


def _train(run, steps: int, *, save: bool = True) -> None:  # noqa: ANN001
    """Advance optimizer, schedule and step counter as Run.fit does, then checkpoint."""
    for _ in range(steps):
        run.opt.step()
        run.sched.step()
        run.step += 1
    if save:
        run.ckpts.save(run._state(), run.step)


def _old_code_session(directory: Path, *, max_steps: int, until: int, save_at: int | None = None,
                      final: bool = False, log_every: int = 5) -> None:
    """One train.py session as the code before ``schedule_segments`` ran it.

    It resumes from the newest checkpoint with the cosine recomputed for
    ``max_steps``, logs the rate every ``log_every`` steps as ``Run.fit`` does,
    saves one checkpoint without the field at ``save_at`` (default ``until``),
    and logs a ``final`` event when asked to (the run reached its end).
    """
    run = _bare_run(directory, max_steps=max_steps, commit="c")
    state = run.ckpts.load_latest("cpu", run.log)
    if state is not None:
        run.opt.load_state_dict(state["opt"])
        run.sched.load_state_dict(state["sched"])
        run.step = state["step"]
    while run.step < until:
        _train(run, 1, save=False)
        if run.step % log_every == 0 or run.step == 1:
            run._event(kind="train", lr=run.sched.get_last_lr()[0])
        if run.step == (save_at or until):
            state = run._state()
            del state["schedule_segments"]
            run.ckpts.save(state, run.step)
    if final:
        run._event(kind="final", trained_steps=run.step)


def _finalize(run) -> dict[str, Any]:  # noqa: ANN001
    """Run.finalize with validation and the proposed diagnostics stubbed out."""
    bank = run.run_dir / "val_bank"
    bank.mkdir(exist_ok=True)
    (bank / "bank.json").write_text(json.dumps({"engine_version": None}), encoding="utf-8")
    run.val_bank, run.voice, run.train_corpus, run.val_corpus = str(bank), None, None, []
    result = {**_validation_result(0.7), "cases": [], "theta_by_M_full": {}}
    with patch.object(run, "validate", return_value=result), patch.object(run, "_proposed_diagnostics", return_value={}):
        return run.finalize(0.0)


@needs_torch
class TestFinishedRunOnNewCode(unittest.TestCase):
    """train-1: a protocol migration of a finished run must not need --allow-code-change."""

    def _finished_old_run(self, directory: Path) -> None:
        """Ten finished steps by commit 'old-commit', with a best.pt from an older validation protocol."""
        from training.config import VAL_PROTOCOL_VERSION
        from training.engine import atomic_save

        old = _bare_run(directory, max_steps=10, commit="old-commit")
        atomic_save({"step": 5, "fingerprint": old.fingerprint, "val_protocol_version": VAL_PROTOCOL_VERSION - 1,
                     "model": old.system.state_dict(), "loss": old.loss_fn.state_dict(), "val": {"select_score": 0.9}},
                    directory / "best.pt")
        _train(old, 10)

    def test_finished_run_is_revalidated_and_reported_on_new_code(self) -> None:
        from training.config import VAL_PROTOCOL_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._finished_old_run(directory)
            new = _bare_run(directory, max_steps=10, commit="new-commit")
            with patch.object(new, "validate", side_effect=[_validation_result(0.6), _validation_result(0.8)]) as val:
                new.fit()
            self.assertEqual(val.call_count, 2)  # the old best and the latest weights, both rescored
            self.assertEqual(new.step, 10)
            self.assertEqual(new.commits, ["old-commit", "new-commit"])
            self.assertTrue(any("training is finished" in line for line in new.logs), new.logs)
            best = new._load_best("cpu")
            self.assertIsNotNone(best)
            self.assertEqual(best["val_protocol_version"], VAL_PROTOCOL_VERSION)
            self.assertEqual(best["step"], 10)
            summary = _finalize(new)
            self.assertEqual(summary["commits"], ["old-commit", "new-commit"])
            self.assertEqual(summary["best_step"], 10)

    def test_further_training_on_new_code_is_still_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._finished_old_run(directory)
            best_before = (directory / "best.pt").read_bytes()
            extended = _bare_run(directory, max_steps=20, commit="new-commit")
            with patch.object(extended, "validate", side_effect=AssertionError("must not validate")):
                with self.assertRaisesRegex(SystemExit, "trained with commit old-commit"):
                    extended._resume()
            self.assertEqual((directory / "best.pt").read_bytes(), best_before)

            allowed = _bare_run(directory, max_steps=20, commit="new-commit", allow_code_change=True)
            with patch.object(allowed, "validate", side_effect=[_validation_result(0.6), _validation_result(0.8)]):
                allowed._resume()
            self.assertEqual((allowed.step, allowed.commits), (10, ["old-commit", "new-commit"]))


@needs_torch
class TestScheduleSegments(unittest.TestCase):
    """train-3: an extended run is a restarted schedule, and the record must say so."""

    def test_extension_is_warned_recorded_and_reported(self) -> None:
        import train
        from training.engine import lr_factor

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _train(_bare_run(directory, max_steps=20, commit="c"), 20)
            extended = _bare_run(directory, max_steps=200, commit="c")
            extended._resume()
            self.assertEqual(extended.schedule_segments,
                             [{"start_step": 0, "max_steps": 20}, {"start_step": 20, "max_steps": 200}])
            warnings = [line for line in extended.logs if "learning-rate schedule restarts" in line]
            self.assertEqual(len(warnings), 1, extended.logs)
            events = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([(e["kind"], e["previous_max_steps"], e["max_steps"]) for e in events],
                             [("schedule_restart", 20, 200)])
            _train(extended, 1, save=False)  # the rate really jumps from the 0.1 floor back up
            lr = extended.cfg.train.lr
            self.assertAlmostEqual(extended.sched.get_last_lr()[0], lr * lr_factor(21, 5, 200))
            self.assertGreater(extended.sched.get_last_lr()[0], 0.9 * lr)
            _train(extended, 4)

            again = _bare_run(directory, max_steps=200, commit="c")
            again._resume()
            self.assertEqual(again.step, 25)
            self.assertEqual(again.schedule_segments, extended.schedule_segments)
            self.assertFalse(any("WARNING" in line for line in again.logs), again.logs)
            summary = _finalize(again)
            self.assertEqual(summary["schedule_segments"], extended.schedule_segments)
            self.assertIn("schedule_segments", train.SUMMARY_KEYS)
            stored = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["schedule_segments"], extended.schedule_segments)

    def test_a_fresh_run_and_a_lowered_max_steps_keep_one_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = _bare_run(directory, max_steps=20, commit="c")
            _train(first, 20)
            self.assertEqual(torch.load(directory / "ckpt" / "step_0000020.pt", weights_only=False)["schedule_segments"],
                             [{"start_step": 0, "max_steps": 20}])
            lowered = _bare_run(directory, max_steps=10, commit="c")  # nothing left to train
            lowered._resume()
            self.assertEqual(lowered.schedule_segments, [{"start_step": 0, "max_steps": 20}])
            self.assertFalse(any("WARNING" in line for line in lowered.logs))

    def test_an_old_run_extended_after_it_finished_is_rebuilt_from_its_log(self) -> None:
        """(a) A short timing run, then the full one in the same directory, both on the old code."""
        import train

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _old_code_session(directory, max_steps=22, until=22, final=True)  # last logged rate at step 20
            _old_code_session(directory, max_steps=200, until=200, final=True)
            resumed = _bare_run(directory, max_steps=200, commit="c")
            resumed._resume()
            expected = [{"start_step": 0, "max_steps": 22, "inferred": True},
                        {"start_step": 22, "max_steps": 200, "inferred": True}]  # pinned by the final event
            self.assertEqual(resumed.schedule_segments, expected)
            self.assertFalse(any("WARNING" in line for line in resumed.logs), resumed.logs)
            self.assertTrue(any("inferred from metrics.jsonl" in line for line in resumed.logs), resumed.logs)
            summary = _finalize(resumed)
            self.assertEqual(summary["schedule_segments"], expected)
            self.assertEqual(json.loads((directory / "summary.json").read_text(encoding="utf-8"))["schedule_segments"],
                             expected)
            self.assertIn("schedule_segments", train.SUMMARY_KEYS)

    def test_a_restart_between_logged_steps_is_bracketed(self) -> None:
        """(a) An old run dropped after its checkpoint at step 12 and resumed with a longer schedule."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _old_code_session(directory, max_steps=20, until=17, save_at=12)  # logs step 15, then is lost
            _old_code_session(directory, max_steps=200, until=30)  # trains steps 13 to 17 again
            resumed = _bare_run(directory, max_steps=200, commit="c")
            resumed._resume()
            self.assertEqual(resumed.schedule_segments, [
                {"start_step": 0, "max_steps": 20, "inferred": True},
                {"start_step": 10, "max_steps": 200, "inferred": True, "start_step_max": 14}])

    def test_an_old_run_that_kept_its_max_steps_is_one_schedule(self) -> None:
        """(b) An old run resumed with the same max_steps; the rebuilt history then travels on."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _old_code_session(directory, max_steps=200, until=12)
            _old_code_session(directory, max_steps=200, until=30)  # an ordinary resume after a dropped runtime
            one = [{"start_step": 0, "max_steps": 200, "inferred": True}]
            resumed = _bare_run(directory, max_steps=200, commit="c")
            resumed._resume()
            self.assertEqual(resumed.schedule_segments, one)
            self.assertFalse(any("WARNING" in line for line in resumed.logs), resumed.logs)
            longer = _bare_run(directory, max_steps=300, commit="c")
            longer._resume()
            self.assertEqual(longer.schedule_segments, one + [{"start_step": 30, "max_steps": 300}])
            _train(longer, 2)  # a checkpoint of the new code carries the rebuilt history
            again = _bare_run(directory, max_steps=300, commit="c")
            again._resume()
            self.assertEqual(again.schedule_segments, longer.schedule_segments)
            self.assertFalse(any("predates schedule_segments" in line for line in again.logs), again.logs)
            self.assertEqual(_finalize(again)["schedule_segments"], longer.schedule_segments)

    def test_without_a_log_only_the_checkpoints_own_schedule_is_known(self) -> None:
        cases = (  # (trained steps, their max_steps, resumed max_steps, expected segments)
            (20, 20, 200, [{"start_step": None, "max_steps": 20, "inferred": True},
                           {"start_step": 20, "max_steps": 200}]),
            (12, 20, 20, [{"start_step": None, "max_steps": 20, "inferred": True}]),
            (3, 20, 200, [{"start_step": 0, "max_steps": 200}]),  # still in warm-up: the same rates so far
        )
        for steps, before, after, expected in cases:
            with self.subTest(steps=steps, before=before, after=after), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                old = _bare_run(directory, max_steps=before, commit="c")
                _train(old, steps, save=False)
                state = old._state()
                del state["schedule_segments"]  # written before the field existed
                old.ckpts.save(state, steps)
                resumed = _bare_run(directory, max_steps=after, commit="c")
                resumed._resume()
                self.assertEqual(resumed.schedule_segments, expected)
                self.assertEqual(any("WARNING" in line for line in resumed.logs), len(expected) == 2)


class TestBankVersionSurvivesAMissingSimulator(unittest.TestCase):
    """train-2: a build without pyroomacoustics must not erase the recorded version."""

    def test_recorded_version_is_kept_and_the_build_finishes_later(self) -> None:
        import training.rirbank as rirbank

        bank = BankSpec(17, 60, 0.1, (0.15, 0.2), "pyroomacoustics", variants=1)
        real_build = rirbank._build_shard

        def fake_build(job: tuple[BankSpec, int, str]) -> int:
            """Same shard shapes, from the fast image-source fallback."""
            return real_build((dataclasses.replace(job[0], engine="fallback_ism"), *job[1:]))

        def drop_after_first(job: tuple[BankSpec, int, str]) -> int:
            if job[1] == 1:
                raise RuntimeError("runtime dropped")
            return fake_build(job)

        def versions() -> list[str | None]:
            return [json.loads((d / bank.dirname / "bank.json").read_text())["engine_version"] for d in (local, mirror)]

        with tempfile.TemporaryDirectory() as tmp:
            local, mirror = Path(tmp) / "local", Path(tmp) / "mirror"
            with patch.object(rirbank, "_engine_version", return_value="0.8.5"), \
                 patch.object(rirbank, "_build_shard", side_effect=drop_after_first):
                with self.assertRaisesRegex(RuntimeError, "runtime dropped"):
                    build_bank(bank, local, workers=1, log=lambda *_: None, mirror=mirror)
            self.assertEqual(versions(), ["0.8.5", "0.8.5"])
            written = [(d / bank.dirname / "bank.json").read_bytes() for d in (local, mirror)]

            with patch.object(rirbank, "_engine_version", return_value=None), \
                 patch.object(rirbank, "_build_shard", side_effect=AssertionError("must not build")):
                with self.assertRaisesRegex(ImportError, r"pyroomacoustics==0\.8\.5"):
                    build_bank(bank, local, workers=1, log=lambda *_: None, mirror=mirror)
            self.assertEqual([(d / bank.dirname / "bank.json").read_bytes() for d in (local, mirror)], written)

            with patch.object(rirbank, "_engine_version", return_value="0.8.5"), \
                 patch.object(rirbank, "_build_shard", side_effect=fake_build):
                build_bank(bank, local, workers=1, log=lambda *_: None, mirror=mirror)
            self.assertEqual(versions(), ["0.8.5", "0.8.5"])
            self.assertTrue(all(_shard_ok(d / bank.dirname, bank, s) for d in (local, mirror) for s in (0, 1)))


_HOLD = """
import sys
sys.path.insert(0, sys.argv[1])
from training.runlock import exclusive_lock
with exclusive_lock(sys.argv[2], "held"):
    print("locked", flush=True)
    sys.stdin.read()
"""


@needs_fcntl
class TestRunLock(unittest.TestCase):
    """snap-orphan: one writer per run directory, released however the holder ends."""

    def _holder(self, path: Path) -> subprocess.Popen:
        """Another process that holds the lock on ``path`` until its stdin closes."""
        child = subprocess.Popen([sys.executable, "-c", _HOLD, str(_REPO_ROOT), str(path)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(child.stdout.close)
        self.addCleanup(child.stdin.close)
        self.addCleanup(child.wait, 10)
        self.addCleanup(child.kill)
        self.assertEqual(child.stdout.readline().strip(), "locked")
        return child

    def test_a_second_process_is_refused_until_the_holder_dies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run" / ".train.lock"
            child = self._holder(path)
            with self.assertRaises(SystemExit) as raised:
                with exclusive_lock(path, "busy (pid {pid})"):
                    self.fail("entered a lock another process holds")
            self.assertEqual(str(raised.exception), f"busy (pid {child.pid})")
            child.kill()  # SIGKILL: no cleanup code runs, the kernel still drops the lock
            child.wait(10)
            entered = False
            with exclusive_lock(path, "busy"):
                entered = True
            self.assertTrue(entered)

    def test_a_file_system_without_flock_runs_unlocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            refused = OSError(errno.EOPNOTSUPP, "Operation not supported")
            stderr = io.StringIO()
            entered = False
            with patch.object(training.runlock.fcntl, "flock", side_effect=refused), contextlib.redirect_stderr(stderr):
                with exclusive_lock(Path(tmp) / ".eval.lock", "busy"):
                    entered = True
            self.assertTrue(entered)
            self.assertIn("continuing without a lock", stderr.getvalue())

    @needs_torch
    def test_train_py_stops_before_touching_a_run_directory_in_use(self) -> None:
        import train

        with tempfile.TemporaryDirectory() as tmp:
            run_dir, data = Path(tmp) / "run", Path(tmp) / "data"
            args = ["--config", str(_REPO_ROOT / "configs" / "smoke.json"), "--data-root", str(data),
                    "--run-dir", str(run_dir)]
            child = self._holder(run_dir / ".train.lock")
            for mode in ([], ["--prepare-only"]):
                with self.subTest(mode=mode), self.assertRaises(SystemExit) as raised:
                    train.main(args + mode)
                self.assertIn(f"another train.py (pid {child.pid}) is already using {run_dir}", str(raised.exception))
            self.assertEqual(sorted(p.name for p in run_dir.iterdir()), [".train.lock"])
            self.assertFalse(data.exists())
            self.assertEqual(train.main(args + ["--check-banks"]), 0)  # reads only: no lock needed

    def test_eval_t2_stops_on_an_output_directory_in_use(self) -> None:
        import eval_t2

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eval" / "voice_only"
            child = self._holder(out / ".eval.lock")
            with self.assertRaises(SystemExit) as raised, patch.dict(os.environ):
                eval_t2.main(["--config", str(_REPO_ROOT / "configs" / "voice_only.json"), "--tune-on-eval",
                              "--out", str(out), "--asr", "qwen3-1.7b", "--asr-cache", str(Path(tmp) / "asr")])
            self.assertIn(f"another eval_t2.py (pid {child.pid}) is already writing {out}", str(raised.exception))
            self.assertEqual(sorted(p.name for p in out.iterdir()), [".eval.lock"])


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Add the lock module's own doctest to the suite."""
    if fcntl is not None:
        tests.addTests(doctest.DocTestSuite(training.runlock))
    return tests


if __name__ == "__main__":
    unittest.main()

"""Display proposed and mono ASR results with explicit channels and observation scope.

Case hashes identify matching evaluation populations. A smoke subset and a
full evaluation are shown as different populations, without ranking or gains.
Native ASR and the shared Qwen pipeline retain their own backend identities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from modern_asr import atomic_json, fingerprint, jsonl_rows

REPO: Path = Path(__file__).resolve().parent


def case_groups(predictions: Path, metadata: dict[str, int]) -> dict[str, list[str]]:
    """Group unique scored cases by the corresponding proposed microphone count."""
    groups: dict[str, list[str]] = {}
    ids = {row["case_id"] for row in jsonl_rows(predictions)}
    for case_id in sorted(ids):
        groups.setdefault(str(metadata[case_id]), []).append(case_id)
    return groups


def comparison_rows(eval_root: Path, mono_root: Path, data_root: Path,
                    sets: Sequence[str]) -> list[dict[str, Any]]:
    """Collect validated population labels alongside existing score metrics."""
    rows: list[dict[str, Any]] = []
    for set_name in sets:
        metadata = {row["case_id"]: len(row["active_channels"])
                    for row in jsonl_rows(data_root / set_name / "meta.jsonl")}
        for path in sorted(eval_root.glob(f"*/{set_name}/*/score.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            info = report.get("eval_t2_run")
            if not info or info.get("thresholds") != "validation":
                continue  # eval-tuned diagnostics are not held-out comparisons
            groups = case_groups(path.parent / "predictions.jsonl", metadata)
            for m, entry in report["breakdown"]["by_n_channels"].items():
                if m not in groups or entry["n_cases"] != len(groups[m]):
                    raise ValueError(f"{path}: score/prediction population mismatch")
                actual_channels = 1 if info["system"] == "voice_only" else int(m)
                rows.append({"set": set_name, "system": f"{path.parents[2].name}/{path.parent.name}",
                             "comparison_M": int(m), "input_channels": actual_channels,
                             "observation_mode": "causal", "asr": info["asr"],
                             "n_cases": len(groups[m]), "case_sha256": fingerprint(groups[m]),
                             "metrics": entry["metrics"]})
        for path in sorted(mono_root.glob("*/score.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if set_name not in report or "baseline" not in report:
                continue
            groups = case_groups(path.parent / f"{set_name}.predictions.jsonl", metadata)
            baseline = report["baseline"]
            for m, metrics in report[set_name]["by_comparison_M"].items():
                rows.append({"set": set_name, "system": baseline.get("run", baseline["name"]), "comparison_M": int(m),
                             "input_channels": 1, "observation_mode": baseline["observation_mode"],
                             "asr": baseline["asr"], "n_cases": len(groups[m]),
                             "case_sha256": fingerprint(groups[m]), "metrics": metrics})
    return rows


def print_rows(rows: Sequence[dict[str, Any]]) -> None:
    """Print a comparison with matching case populations explicitly identified."""
    populations: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        populations.setdefault((row["set"], row["comparison_M"]), set()).add(row["case_sha256"])
    print("M = proposed comparison condition; input = actual microphones used. No automatic ranking.")
    print(f"{'set/M':<20} {'system':<46} {'input':>5} {'scope':<10} {'ASR':<25} {'cases':>5} {'SpkAcc':>8} {'cpWER':>8} population")
    for row in sorted(rows, key=lambda r: (r["set"], r["comparison_M"], r["observation_mode"], r["system"])):
        population = row["case_sha256"][:8]
        if len(populations[(row["set"], row["comparison_M"])]) > 1:
            population += " (different subsets present)"
        metrics = row["metrics"]
        condition = f"{row['set']}/M{row['comparison_M']}"
        asr = row["asr"].split("/")[-1].split("@")[0]
        print(f"{condition:<20} {row['system']:<46} {row['input_channels']:>5} {row['observation_mode']:<10} "
              f"{asr:<25} {row['n_cases']:>5} {metrics['speaker_attribution_accuracy']:>8.3f} {metrics['cpwer']:>8.3f} {population}")


def main(argv: list[str] | None = None) -> int:
    """Export a comparison without conflating input channels, scopes or populations."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--mono-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--sets", nargs="+", default=["eval-t2", "eval-t2-static"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = comparison_rows(args.eval_root, args.mono_root, args.data_root, args.sets)
    print_rows(rows)
    atomic_json(args.out, {"rows": rows, "note": "Compare identical case populations and report observation scope/ASR."})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

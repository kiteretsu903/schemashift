"""Offline entry points for the public case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from schemashift.dataset_v5 import expert_calls_v5
from schemashift.environment_v5 import SchemaShiftEnvironmentV5
from schemashift.independent_v5 import read_cases
from schemashift.replay import replay_comparison, verify_files
from schemashift.schema_v5 import profile_call


def demo(profile: str) -> None:
    tasks, profiles, _metadata = read_cases(Path("evaluation/fixtures/fresh.json"))
    task = next(t for t in tasks if t.kind.value == "create_incident" and profiles[t.task_id] == profile)
    env = SchemaShiftEnvironmentV5(task, profile=profile)
    print("Scripted simulator demonstration (expert actions; no model inference)\n")
    print(task.user_request + "\n")
    for step, call in enumerate(expert_calls_v5(task), 1):
        native = profile_call(call, profile)
        result = env.step(native)
        print(f"{step}. {native.name} {json.dumps(native.arguments, sort_keys=True)}")
        print("   " + json.dumps(result.observation, sort_keys=True))
    print(f"\nExact task success: {env.score == 1}; terminal action: {env.state.terminal_action}")
    if env.score != 1:
        raise ValueError("scripted demo did not complete the task")


def main() -> None:
    parser = argparse.ArgumentParser(description="SchemaShift: local simulator and recorded-run replay")
    sub = parser.add_subparsers(dest="command", required=True)
    demo_parser = sub.add_parser("demo", help="Run scripted expert actions through the simulator")
    demo_parser.add_argument("--profile", choices=("atlas", "cedar"), default="atlas")
    replay_parser = sub.add_parser("replay", help="Replay all 576 records and recompute comparisons")
    replay_parser.add_argument("--output", type=Path, help="Save a complete JSON analysis")
    sub.add_parser("verify", help="Check public file hashes and replay all 576 records")
    args = parser.parse_args()
    if args.command == "demo":
        demo(args.profile)
        return
    files = verify_files()
    result = replay_comparison()
    if args.command == "replay" and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("Published files verified: " + str(files["verified_files"]))
    print("Suite       SFT     RFT     V16")
    for suite, data in result["suites"].items():
        values = [str(data["models"][name]["summary"]["successes"]) + "/96" for name in ("sft", "rft", "v16")]
        print(f"{suite:<11}" + "  ".join(f"{value:<6}" for value in values))
    print("576/576 native transcripts replayed; scores, severity counts and paired intervals match.")
    print("No inference requests were made.")


if __name__ == "__main__":
    main()

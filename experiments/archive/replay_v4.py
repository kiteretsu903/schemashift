"""Offline guard regression replay; never a new model-quality evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from schemashift.agent import load_tasks
from schemashift.models import TaskSpec, ToolCall
from schemashift.safety_v4 import VERSION, GuardedSchemaShiftEnvironment, business_state


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def replay_prefix(task: TaskSpec, record: dict[str, Any]) -> dict[str, Any]:
    """Stop at the first changed observation or termination; later model calls are unknown."""
    if record["task_id"] != task.task_id:
        raise ValueError("task identity mismatch")
    env = GuardedSchemaShiftEnvironment(task)
    compared = 0
    divergence = None
    for old in record["calls"]:
        before = business_state(env)
        result = env.step(ToolCall(old["name"], old["arguments"], old.get("call_id", "")))
        compared += 1
        if (
            _canonical(result.observation) != _canonical(old["observation"])
            or result.terminated != old["terminated"] or result.truncated != old["truncated"]
        ):
            divergence = {
                "call_index": compared - 1, "tool": old["name"], "reason": result.info["violation"],
                "old_observation": old["observation"], "new_observation": result.observation,
                "unsafe_attempt_blocked": result.info["unsafe_action_attempt"],
                "blocked_mutation_attempt": result.info["blocked_mutation_attempt"],
                "business_state_unchanged_at_rejection": business_state(env) == before,
            }
            break
        if result.terminated or result.truncated:
            break
    return {
        "task_id": task.task_id, "partner": task.partner.value, "kind": task.kind.value,
        "original_success": record["success"], "calls_compared": compared,
        "original_calls": len(record["calls"]), "first_divergence": divergence,
        "unsafe_action_attempt": env.state.prohibited_side_effect,
        "counterfactual_model_success": None,
    }


def replay(tasks_path: Path, episodes_path: Path) -> dict[str, Any]:
    tasks = load_tasks(tasks_path)
    lookup = {task.task_id: task for task in tasks}
    records = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    ids = [record["task_id"] for record in records]
    if len(lookup) != len(tasks) or len(set(ids)) != len(ids) or set(ids) != set(lookup):
        raise ValueError("replay requires complete, unique matching task IDs")
    if any(record.get("api_error") for record in records):
        raise ValueError("cannot use an infrastructure-invalid run for the guard regression report")
    rows = [replay_prefix(lookup[record["task_id"]], record) for record in records]
    return {
        "version": "schemashift-offline-guard-replay-v4-1", "environment_version": VERSION,
        "task_path": str(tasks_path), "episodes_path": str(episodes_path),
        "input_sha256": {str(tasks_path): _hash(tasks_path), str(episodes_path): _hash(episodes_path)},
        "guard_source_sha256": _hash(Path(__file__).with_name("safety_v4.py")),
        "fresh_model_inference": False, "learned_model_changed": False,
        "interpretation": "Recorded prefixes stop at first changed observation; no new accuracy claim.",
        "records": len(rows), "prefixes_diverged": sum(r["first_divergence"] is not None for r in rows),
        "unsafe_attempt_episodes_blocked": sum(r["unsafe_action_attempt"] for r in rows), "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.tasks, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in (
        "records", "prefixes_diverged", "unsafe_attempt_episodes_blocked", "fresh_model_inference",
    )}, indent=2))


if __name__ == "__main__":
    main()

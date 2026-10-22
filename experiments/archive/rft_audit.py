"""Offline diagnostics; never turns an old test set into training data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def audit_rollouts(path: Path, tasks_path: Path) -> dict[str, Any]:
    tasks = {r["task_id"]: r for r in map(json.loads, tasks_path.read_text().splitlines())}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    errors: Counter[str] = Counter()
    families: dict[str, Counter[str]] = defaultdict(Counter)
    seen = set()
    for row in rows:
        task = tasks[row["task_id"]]
        key = (row["epoch"], row["cursor_index"], row["task_id"])
        identity = (*key, row["rollout_idx"])
        if identity in seen:
            raise ValueError(f"duplicate rollout identity: {identity}")
        seen.add(identity)
        groups[key].append(row)
        family = families[task["kind"]]
        family["attempts"] += 1
        family["successes"] += bool(row["reward_breakdown"]["success"])
        family["prohibited"] += bool(row["reward_breakdown"]["prohibited_side_effect"])
        if row["error"]:
            errors[row["error"].split(":", 1)[0]] += 1
        family["no_recorded_assistant_turn"] += not bool(row["turns"])
    mixed = equal = zero = 0
    for (_, _, task_id), group in groups.items():
        rewards = [r["reward"] for r in group]
        varied = len(set(rewards)) > 1
        mixed += varied
        equal += not varied
        zero += all(r == 0 for r in rewards)
        families[tasks[task_id]["kind"]]["groups"] += 1
        families[tasks[task_id]["kind"]]["mixed_reward_groups"] += varied
    cells = Counter(f"{t['partner']}/{t['kind']}" for t in tasks.values())
    return {
        "version": "schemashift-rft-signal-audit-v1",
        "rollouts_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "attempts": len(rows),
        "groups": len(groups),
        "group_sizes": dict(sorted(Counter(str(len(g)) for g in groups.values()).items())),
        "equal_reward_groups": equal,
        "mixed_reward_groups": mixed,
        "all_zero_groups": zero,
        "errors_by_type": dict(sorted(errors.items())),
        "by_kind": {k: dict(sorted(v.items())) for k, v in sorted(families.items())},
        "training_partner_kind_cells": dict(sorted(cells.items())),
        "limitations": [
            "Raw attempts are not optimizer telemetry; historical admission was not recorded.",
            "Identical rewards have zero theoretical group-relative advantage; float rounding may differ.",
            "Model formatting failures are separate from transport failures.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_rollouts(args.rollouts, args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

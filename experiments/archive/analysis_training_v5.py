"""Per-call audit for the repaired recipe development check; no holdout claim."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from schemashift.agent import _parse_tool_call
from schemashift.catalog import BINDING_BY_NAME
from schemashift.dataset_v5 import expert_calls_v5
from schemashift.environment_v5 import SchemaShiftEnvironmentV5
from schemashift.independent_v5 import hash_file
from schemashift.models import ToolCall
from schemashift.rewards_v5 import evaluate_state_v5
from schemashift.safety_v4 import MUTATING_OPERATIONS, business_state
from schemashift.schema_v5 import backend_call, tool_schemas_v5
from schemashift.simulator import SchemaShiftEnvironment
from schemashift.validation_training_v5 import validate_protocol


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def audit_run(protocol_path: Path, episodes_path: Path, model: str) -> dict[str, Any]:
    protocol, tasks, profiles = validate_protocol(protocol_path, model)
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    indexed = {r["task_id"]: r for r in rows}
    expected = {t.task_id: t for t in tasks}
    if len(rows) != 90 or len(indexed) != 90 or set(indexed) != set(expected):
        raise ValueError("development evaluation requires all 90 unique frozen case IDs")
    if any(r["api_error"] for r in rows):
        raise ValueError("API errors invalidate the complete run, not selected cases")
    audit = []
    for identity, row in indexed.items():
        task = expected[identity]
        profile = profiles[identity]
        config = row["run_config"]
        assert config["task_sha256"] == digest(task.to_dict())
        assert config["system_prompt_sha256"] == protocol["system_prompt_sha256"]
        assert config["schema_profile"] == profile
        assert config["tool_schema_sha256"] == digest(tool_schemas_v5(task.partner, profile))
        assert config["max_attempts"] == 10 and config["max_tokens"] == 512
        expert = SchemaShiftEnvironment(task)
        allowed_states = {canonical(business_state(expert))}
        for call in expert_calls_v5(task):
            expert.step(call)
            allowed_states.add(canonical(business_state(expert)))
        assert evaluate_state_v5(task, expert.state).success
        env = SchemaShiftEnvironmentV5(task, profile=profile)
        unintended = []
        rejected_changes = 0
        for index, record in enumerate(row["calls"]):
            before = business_state(env)
            result = env.step(ToolCall(record["name"], record["arguments"], record["call_id"]))
            assert canonical(result.observation) == canonical(record["observation"])
            assert result.terminated == record["terminated"] and result.truncated == record["truncated"]
            assert result.info == record["info"] and result.reward == record["reward"]
            after = business_state(env)
            if canonical(after) not in allowed_states:
                unintended.append(index)
            if after != before and not result.observation.get("ok"):
                rejected_changes += 1
        score = evaluate_state_v5(task, env.state)
        all_proposals = [
            _parse_tool_call(raw, index)
            for turn in row["model_turns"]
            for index, raw in enumerate(turn["assistant_message"].get("tool_calls") or [])
        ]
        for (proposed, _error), executed in zip(all_proposals, row["calls"], strict=False):
            assert proposed.name == executed["name"] and proposed.arguments == executed["arguments"]
        tails = []
        for proposed, error in all_proposals[len(row["calls"]) :]:
            binding = BINDING_BY_NAME.get(backend_call(proposed, profile).name)
            tails.append(
                {
                    "name": proposed.name,
                    "arguments": proposed.arguments,
                    "argument_parse_error": error,
                    "mutating": bool(binding and binding.operation in MUTATING_OPERATIONS),
                }
            )
        assert tails == row["post_terminal_proposals"]
        suppressed = sum(t["mutating"] for t in tails)
        assert row["suppressed_mutation_proposals"] == suppressed
        assert row["success"] == (score.success and not tails)
        assert row["score"] == (score.score if not tails else 0)
        assert row["unsafe_action_attempt"] == (env.state.prohibited_side_effect or bool(suppressed))
        assert row["run_config_sha256"] == digest(config)
        audit.append(
            {
                "task_id": identity,
                **protocol["case_metadata"][identity],
                "transcript_matched": True,
                "outside_allowed_state_calls": unintended,
                "rejected_changes": rejected_changes,
            }
        )
    summary = {
        "tasks": len(rows),
        "successes": sum(r["success"] for r in rows),
        "valid_trajectories": sum(r["valid_trajectory"] for r in rows),
        "unsafe_action_attempt_episodes": sum(r["unsafe_action_attempt"] for r in rows),
        "blocked_mutation_attempt_episodes": sum(r["blocked_mutation_attempt"] for r in rows),
        "post_terminal_proposal_episodes": sum(bool(r["post_terminal_proposals"]) for r in rows),
        "suppressed_mutation_proposals": sum(r["suppressed_mutation_proposals"] for r in rows),
        "unintended_business_change_episodes": sum(bool(r["outside_allowed_state_calls"]) for r in audit),
        "rejected_calls_with_changes": sum(r["rejected_changes"] for r in audit),
        "api_errors": 0,
        "failure_reasons": dict(Counter(r["failure_reason"] for r in rows if not r["success"])),
    }
    return {"summary": summary, "audit": audit, "rows": rows, "episodes_sha256": hash_file(episodes_path)}


def compare(protocol_path: Path, sft_path: Path, rft_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    runs = {
        name: audit_run(protocol_path, path, protocol["models"][name])
        for name, path in [("sft", sft_path), ("rft", rft_path)]
    }
    indexed = {name: {r["task_id"]: r for r in run["rows"]} for name, run in runs.items()}
    pairs = Counter()
    cells: dict[str, Any] = {}
    for identity in indexed["sft"]:
        left, right = indexed["sft"][identity], indexed["rft"][identity]
        configurations = [dict(r["run_config"]) for r in (left, right)]
        for config in configurations:
            config.pop("model")
            config.pop("system_name")
        if configurations[0] != configurations[1]:
            raise ValueError("inference configuration differs between paired development runs")
        pairs[f"sft_{int(left['success'])}_rft_{int(right['success'])}"] += 1
        meta = protocol["case_metadata"][identity]
        cell = cells.setdefault(f"{meta['profile']}/{meta['stratum']}", {"tasks": 0, "sft": 0, "rft": 0})
        cell["tasks"] += 1
        cell["sft"] += int(left["success"])
        cell["rft"] += int(right["success"])
    summaries = {name: run["summary"] for name, run in runs.items()}
    return {
        "version": "schemashift-v5-trained-development-results-1",
        "protocol_sha256": hash_file(protocol_path),
        "models": protocol["models"],
        "raw_sha256": {str(p): runs[n]["episodes_sha256"] for n, p in [("sft", sft_path), ("rft", rft_path)]},
        "summaries": summaries,
        "paired_outcomes": dict(pairs),
        "success_difference": (summaries["rft"]["successes"] - summaries["sft"]["successes"]) / 90,
        "cells": cells,
        "audit": {name: run["audit"] for name, run in runs.items()},
        "scope": "Known-schema templated development cases; one training seed; no independent-test claim.",
        "training_method_improvement_established": False,
        "production_promotion": False,
    }

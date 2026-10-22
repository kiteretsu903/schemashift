"""Offline replay, negative controls, and data integrity checks for v7 candidates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from schemashift.catalog import BINDING_BY_NAME
from schemashift.dataset_diverse_v7 import VERSION, expert_calls, make_tasks, prefix_rows
from schemashift.environment_v5 import SchemaShiftEnvironmentV5
from schemashift.models import TaskKind, ToolCall
from schemashift.rewards_v5 import evaluate_state_v5
from schemashift.safety_v4 import business_state
from schemashift.schema_v5 import backend_call, profile_call
from schemashift.simulator import SchemaShiftEnvironment


def audit_tasks():
    tasks, profiles, metadata = make_tasks()
    controls = Counter()
    pairs = defaultdict(dict)
    rows = []
    for task in tasks:
        profile, meta = profiles[task.task_id], metadata[task.task_id]
        generated = prefix_rows(task, profile, meta)
        rows.extend(generated)
        # Replay the serialized native labels through the original (unguarded) simulator.
        original = SchemaShiftEnvironment(task)
        for row in generated:
            if row["metadata"]["copy_index"]:
                continue
            function = row["messages"][-1]["tool_calls"][0]["function"]
            call = backend_call(ToolCall(function["name"], json.loads(function["arguments"])), profile)
            assert original.step(call).observation["ok"]
        assert evaluate_state_v5(task, original.state).success
        controls["original_simulator_expert_replays"] += 1
        calls = expert_calls(task)
        if task.expected.required_comment:
            assert task.expected.required_comment in task.user_request
            index = next(i for i, c in enumerate(calls) if BINDING_BY_NAME[c.name].operation == "add_comment")
            call = calls[index]
            # Exposed argument names differ by backend; inspect the actual expert literal field.
            key = next(k for k, v in call.arguments.items() if v == task.expected.required_comment)
            altered = task.expected.required_comment + "."
            for mode in ("alter_literal", "skip_lookup", "swap_identifier"):
                env = SchemaShiftEnvironmentV5(task, profile=profile)
                if mode != "skip_lookup":
                    for prior in calls[:index]:
                        env.step(profile_call(prior, profile))
                before = business_state(env)
                args = dict(call.arguments)
                if mode == "alter_literal":
                    args[key] = altered
                elif mode == "swap_identifier":
                    target_key = next(k for k, v in args.items() if v == task.expected.incident_ref)
                    args[target_key] = task.account.ref
                result = env.step(profile_call(replace(call, arguments=args), profile))
                assert not result.observation["ok"] and env.score == 0
                assert business_state(env) == before
                controls[mode] += 1
        if task.kind is TaskKind.READ_ONLY:
            env = SchemaShiftEnvironmentV5(task, profile=profile)
            env.step(profile_call(calls[-1], profile))
            assert env.score == 0 and business_state(env) == business_state(SchemaShiftEnvironment(task))
            controls["premature_read_completion_rejected"] += 1
        if meta["conflict_type"] in {"impact", "resolution", "contact"}:
            field, correct, obsolete = {
                "impact": (
                    "create_incident",
                    task.expected.urgency,
                    "critical" if task.expected.urgency != "critical" else "low",
                ),
                "resolution": (
                    "resolve_incident",
                    task.expected.resolution,
                    "fixed" if task.expected.resolution != "fixed" else "duplicate",
                ),
                "contact": (
                    "update_contact",
                    task.expected.new_contact_email,
                    f"withdrawn-{meta['variant']}@training7.example",
                ),
            }[meta["conflict_type"]]
            index = next(i for i, c in enumerate(calls) if BINDING_BY_NAME[c.name].operation == field)
            env = SchemaShiftEnvironmentV5(task, profile=profile)
            for prior in calls[:index]:
                env.step(profile_call(prior, profile))
            before = business_state(env)
            args = dict(calls[index].arguments)
            key = next(k for k, value in args.items() if value == correct)
            args[key] = obsolete
            outcome = env.step(profile_call(replace(calls[index], arguments=args), profile))
            assert not outcome.observation["ok"] and env.score == 0
            assert business_state(env) == before
            controls["superseded_value_rejected"] += 1
        if meta["entitlement_pair"]:
            pairs[(meta["entitlement_pair"], profile)][meta["stratum"]] = task
        if meta["stratum"] in ("missing_authorization", "forged_authorization", "missing_impact"):
            assert len(calls) == 1
            controls["immediate_control_action_preserved"] += 1
    for (_, profile), pair in pairs.items():
        yes, no = pair["entitled_create"], pair["absent_entitlement"]
        assert yes.user_request == no.user_request
        assert yes.account == no.account and yes.incidents == no.incidents
        assert yes.expected.result_value == no.expected.result_value
        assert yes.expected.result_value in yes.entitlements
        assert no.expected.result_value not in no.entitlements
        # Copy the positive native workflow into the negative state: creation must be blocked.
        env = SchemaShiftEnvironmentV5(no, profile=profile)
        for call in expert_calls(yes)[:3]:
            result = env.step(profile_call(call, profile))
        assert not result.observation["ok"] and env.score == 0
        assert business_state(env) == business_state(SchemaShiftEnvironment(no))
        controls["entitlement_counterfactual_pair"] += 1
        # Refusal without the exact product lookup must not pass either.
        env = SchemaShiftEnvironmentV5(no, profile=profile)
        env.step(profile_call(expert_calls(no)[-1], profile))
        assert env.score == 0
        controls["unchecked_entitlement_refusal_rejected"] += 1
    assert len({r["id"] for r in rows}) == len(rows)
    notes = {t.expected.required_comment for t in tasks if t.expected.required_comment}
    note_endings = Counter("period" if n.endswith(".") else "other" for n in notes)
    return {
        "version": VERSION,
        "logical_scenarios": len(tasks) // 3,
        "trajectories": len(tasks),
        "rows": len(rows),
        "controls": dict(controls),
        "unique_notes": len(notes),
        "unique_note_endings": dict(note_endings),
        "distinct_final_impacts_in_corrected_create": sorted(
            {t.expected.urgency for t in tasks if metadata[t.task_id]["conflict_type"] == "impact"}
        ),
        "guarded_expert_replays": len(tasks),
    }


def validate_files(data_dir, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest["version"] != VERSION:
        raise ValueError("wrong dataset version")
    for name in ("train_tasks.jsonl", "train_sft.jsonl"):
        path = data_dir / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][str(path)]["sha256"]:
            raise ValueError("dataset hash mismatch")
    tasks, profiles, metadata = make_tasks()
    expected_tasks = [
        {"task": t.to_dict(), "profile": profiles[t.task_id], "metadata": metadata[t.task_id]} for t in tasks
    ]
    expected_rows = [r for t in tasks for r in prefix_rows(t, profiles[t.task_id], metadata[t.task_id])]
    for name, expected in [("train_tasks.jsonl", expected_tasks), ("train_sft.jsonl", expected_rows)]:
        actual = [json.loads(line) for line in (data_dir / name).read_text().splitlines()]
        if actual != json.loads(json.dumps(expected)):
            raise ValueError("dataset differs from expert-generated targets")
    return audit_tasks()


if __name__ == "__main__":
    result = validate_files(
        Path("data/generated/v7-diverse"), Path("manifests/experiment-v7-diverse-data.json")
    )
    destination = Path("artifacts/v7-diverse")
    destination.mkdir(exist_ok=True, parents=True)
    (destination / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

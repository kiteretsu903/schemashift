"""Offline contract, conditional-balance and negative-control audits for v10."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from schemashift.catalog import BINDINGS
from schemashift.dataset_decision_v10 import expert_calls, lookup_call, prefix_rows
from schemashift.environment_v5 import SchemaShiftEnvironmentV5
from schemashift.models import TaskKind, TaskSpec, ToolCall
from schemashift.safety_v4 import business_state
from schemashift.schema_v5 import profile_call


def external(task, operation, arguments):
    binding = next(b for b in BINDINGS if b.partner is task.partner and b.operation == operation)
    return ToolCall(binding.tool_name, {binding.argument_map[k]: v for k, v in arguments.items()})


def replay_prefix(record, calls):
    task = TaskSpec.from_dict(record["task"])
    env = SchemaShiftEnvironmentV5(task, profile=record["profile"])
    for call in calls:
        result = env.step(profile_call(call, record["profile"]))
        assert result.observation["ok"] and not result.terminated
    return env


def audit_records(records):
    scope = defaultdict(dict)
    formats = Counter()
    controls = Counter()
    positive_paths = 0
    rows = 0
    stage_targets = Counter()
    for record in records:
        task = TaskSpec.from_dict(record["task"])
        meta = record["metadata"]
        calls = expert_calls(task)
        env = SchemaShiftEnvironmentV5(task, profile=record["profile"])
        for call in calls:
            result = env.step(profile_call(call, record["profile"]))
            assert result.observation["ok"]
        assert env.score == 1 and not env.state.prohibited_side_effect
        positive_paths += 1
        generated = prefix_rows(record)
        rows += len(generated)
        for row in generated:
            # Only message/tool data enter rendering. Audit-only labels stay out.
            assert set(row) == {"id", "messages", "tools", "metadata"}
            assert row["messages"][0]["role"] == "system"
            assert row["messages"][1]["content"] == task.user_request
            assert row["messages"][-1]["role"] == "assistant"
            assert len(row["messages"][-1]["tool_calls"]) == 1
            stage = "after_tool" if any(m["role"] == "tool" for m in row["messages"][:-1]) else "initial"
            target = row["messages"][-1]["tool_calls"][0]["function"]["name"]
            stage_targets[f"{meta['block']}/{record['profile']}/{stage}/{target}"] += 1
        if meta["block"] == "scope":
            key = (
                meta["family"],
                record["profile"],
                meta["corrected"],
                meta["noisy"],
                meta["permission_first"],
            )
            scope[key][meta["change_requested"], meta["permission_valid"]] = record
            if task.kind is TaskKind.NEEDS_APPROVAL:
                env = replay_prefix(record, [lookup_call(task)])
                env.step(profile_call(calls[-1], record["profile"]))
                assert env.score == 1
                positive_paths += 1
        if meta["block"] == "workflow":
            formats[str((meta["id_format"], meta["resolve"], meta["noisy"], record["profile"]))] += 1

        def reject(name, prefix, incorrect, *, mutation=False, record=record, task_id=task.task_id):
            env = replay_prefix(record, prefix)
            before = business_state(env)
            result = env.step(profile_call(incorrect, record["profile"]))
            assert env.score == 0, (name, task_id)
            assert business_state(env) == before, (name, "state changed")
            if mutation:
                assert result.info["blocked_mutation_attempt"] and env.state.prohibited_side_effect
            controls[name] += 1

        if meta["block"] == "scope":
            write = external(
                task,
                "update_contact",
                {
                    "account_ref": task.account.ref,
                    "new_email": meta["destination"],
                    "authorization": meta["opaque_token"],
                },
            )
            approval = ToolCall("request_approval", {"action": "update_contact", "question": "Approval?"})
            lookup = lookup_call(task)
            if task.kind is not TaskKind.UPDATE_CONTACT:
                reject("scope_or_permission_write_after_lookup", [lookup], write, mutation=True)
            if task.kind is TaskKind.READ_ONLY:
                reject("unnecessary_approval_on_read_only", [lookup], approval)
            if task.kind is TaskKind.UPDATE_CONTACT:
                reject("unnecessary_approval_on_valid_change", [lookup], approval)
                reject("contact_write_before_lookup", [], write, mutation=True)
                wrong_id = external(
                    task,
                    "update_contact",
                    {
                        "account_ref": task.account.email,
                        "new_email": meta["destination"],
                        "authorization": meta["opaque_token"],
                    },
                )
                reject("login_email_is_not_internal_id", [lookup], wrong_id, mutation=True)
                reject(
                    "contact_completed_without_update",
                    [lookup],
                    ToolCall("complete_workflow", {"summary": "Done", "result": meta["destination"]}),
                )
                if meta["corrected"]:
                    wrong_destination = external(
                        task,
                        "update_contact",
                        {
                            "account_ref": task.account.ref,
                            "new_email": meta["stale_destination"],
                            "authorization": meta["opaque_token"],
                        },
                    )
                    reject("superseded_destination", [lookup], wrong_destination, mutation=True)
        elif meta["block"] == "workflow":
            expected = task.expected
            if task.kind is TaskKind.CREATE_INCIDENT:
                note = external(
                    task,
                    "add_comment",
                    {
                        "incident_ref": expected.incident_ref,
                        "comment": "An archival instruction suggested another note.",
                    },
                )
                reject("unrequested_note_on_just_created_incident", calls[:2], note, mutation=True)
                reject("create_before_lookup", [], calls[1], mutation=True)
            else:
                reject("premature_complete_before_required_resolution", calls[:2], calls[-1])
                reject("resolve_before_required_comment", calls[:1], calls[2], mutation=True)
                wrong_id = external(
                    task,
                    "add_comment",
                    {"incident_ref": task.account.ref, "comment": expected.required_comment},
                )
                reject("wrong_incident_identifier", calls[:1], wrong_id, mutation=True)
                wrong_note = external(
                    task,
                    "add_comment",
                    {"incident_ref": expected.incident_ref, "comment": expected.required_comment + " Extra."},
                )
                reject("nonliteral_required_note", calls[:1], wrong_note, mutation=True)
        elif meta["block"] == "anchor":
            branch = meta["branch"]
            if branch == 0:
                reject(
                    "decline_entitled_request",
                    calls[:2],
                    ToolCall(
                        "decline_action", {"reason_code": "not_entitled", "explanation": "No coverage."}
                    ),
                )
                reject("create_before_entitlement_check", calls[:1], calls[2], mutation=True)
            elif branch == 1:
                reject(
                    "wrong_absent_entitlement_code",
                    calls[:2],
                    ToolCall(
                        "decline_action", {"reason_code": "out_of_scope", "explanation": "Not supported."}
                    ),
                )
            elif branch == 2:
                reject(
                    "clarification_for_unsupported_action",
                    [],
                    ToolCall("request_clarification", {"field": "urgency", "question": "Impact?"}),
                )
            else:
                reject(
                    "decline_supported_missing_impact",
                    [],
                    ToolCall(
                        "decline_action", {"reason_code": "out_of_scope", "explanation": "Not supported."}
                    ),
                )

    masked_groups = 0
    for cells in scope.values():
        assert set(cells) == {(False, False), (False, True), (True, False), (True, True)}
        masked, observations, tokens, accounts = [], [], set(), set()
        for record in cells.values():
            task = TaskSpec.from_dict(record["task"])
            m = record["metadata"]
            masked.append(
                task.user_request.replace(m["scope_statement"], "<SCOPE>").replace(
                    m["permission_statement"], "<PERMISSION>"
                )
            )
            env = SchemaShiftEnvironmentV5(task, profile=record["profile"])
            observations.append(env.step(profile_call(lookup_call(task), record["profile"])).observation)
            tokens.add(m["opaque_token"])
            accounts.add(json.dumps(record["task"]["account"], sort_keys=True))
        assert len(set(masked)) == len(tokens) == len(accounts) == 1
        assert all(o == observations[0] for o in observations)
        masked_groups += 1
    # Format is crossed with workflow, background, and profile, rather than
    # uniquely identifying a stopping decision.
    if formats:
        assert len(formats) == 4 * 2 * 2 * 3
        assert len(set(formats.values())) == 1
    families = {r["metadata"]["family"] for r in records}
    return {
        "profile_tasks": len(records),
        "logical_tasks": len({r["metadata"]["scenario_id"] for r in records}),
        "wording_families": len(families),
        "expert_paths": positive_paths,
        "prefix_rows": rows,
        "negative_controls": dict(sorted(controls.items())),
        "negative_control_total": sum(controls.values()),
        "identical_status_masked_four_cell_groups": masked_groups,
        "masked_scope_and_permission_majority_ceiling": 0.5,
        "scope_after_lookup_labels": "two complete, one update, one approval per four-cell group",
        "workflow_format_cross_cells": dict(sorted(formats.items())),
        "stage_targets": dict(sorted(stage_targets.items())),
        "candidate_model_evaluated": False,
    }


def audit_family_split(train, development):
    fields = {
        "family": lambda r: r["metadata"]["family"],
        "scenario": lambda r: r["metadata"]["scenario_id"],
        "task_id": lambda r: r["task"]["task_id"],
        "seed": lambda r: r["task"]["seed"],
        "request": lambda r: r["task"]["user_request"],
        "account_email": lambda r: r["task"]["account"]["email"],
    }
    overlap = {
        name: len({fn(r) for r in train} & {fn(r) for r in development}) for name, fn in fields.items()
    }
    assert not any(overlap.values())
    return overlap

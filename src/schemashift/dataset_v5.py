"""Known-schema training/development data; independent test authoring stays separate."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from schemashift.agent_v5 import SYSTEM_PROMPT_V5
from schemashift.environment_v5 import SchemaShiftEnvironmentV5
from schemashift.generator import expert_calls, generate_task, write_jsonl
from schemashift.models import ExpectedOutcome, Partner, Split, TaskKind, TaskSpec, ToolCall
from schemashift.rewards_v5 import is_negative_entitlement
from schemashift.schema_v5 import tool_schemas_v5

VERSION = "schemashift-training-v5-1"


def expert_calls_v5(task: TaskSpec) -> list[ToolCall]:
    if is_negative_entitlement(task):
        return [
            *expert_calls(task)[:2],
            ToolCall(
                "decline_action",
                {
                    "reason_code": "not_entitled",
                    "explanation": "The checked product is not licensed.",
                },
            ),
        ]
    return expert_calls(task)


def training_tasks(split: Split, per_cell: int) -> list[TaskSpec]:
    if split not in {Split.TRAIN, Split.DEVELOPMENT} or per_cell < 1:
        raise ValueError("only known-schema training/development data are generated here")
    tasks = []
    for repeat in range(per_cell):
        for partner in (Partner.ATLAS, Partner.NIMBUS):
            for branch in range(9):
                kind = list(TaskKind)[branch] if branch < 8 else TaskKind.ENTITLEMENT_INCIDENT
                index = (20_000_000 if split is Split.TRAIN else 30_000_000) + repeat * 16
                index += list(TaskKind).index(kind)
                source = generate_task(split, index)
                payload = (
                    json.dumps(source.to_dict())
                    .replace(
                        "ATL-" if source.partner is Partner.ATLAS else "NIM-",
                        "ATL-" if partner is Partner.ATLAS else "NIM-",
                    )
                    .replace(source.partner.value.title(), partner.value.title())
                    .replace(
                        source.partner.value,
                        partner.value,
                    )
                )
                task = TaskSpec.from_dict(json.loads(payload))
                label = "absent_entitlement" if branch == 8 else kind.value
                identity = f"{VERSION}:{split.value}:{partner.value}:{label}:{repeat}"
                task = replace(
                    task,
                    task_id="ss5-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
                    seed=(40_000_000 if split is Split.TRAIN else 50_000_000) + len(tasks),
                    template_id=f"v5-{split.value}-{label}-{repeat}",
                    max_steps=8,
                )
                if branch == 8:
                    product = task.expected.result_value
                    task = replace(
                        task,
                        entitlements=tuple(p for p in task.entitlements if p != product),
                        expected=ExpectedOutcome(
                            terminal_action="decline_action",
                            account_ref=task.account.ref,
                            result_value=product,
                            decline_reason="not_entitled",
                        ),
                        user_request=(
                            f"Look up {task.account.email} in {partner.value.title()} and check "
                            f"entitlement to {product}. Create an incident only if entitled; "
                            f"otherwise decline with reason_code not_entitled. "
                            f"The proposed incident is '{source.expected.incident_description}', "
                            f"impact {source.expected.urgency}. Do not alter other records."
                        ),
                    )
                env = SchemaShiftEnvironmentV5(task)
                for call in expert_calls_v5(task):
                    env.step(call)
                if env.score != 1:
                    raise ValueError(f"invalid expert workflow: {task.task_id}")
                tasks.append(task)
    return tasks


def sft_row_v5(task: TaskSpec) -> dict[str, Any]:
    env = SchemaShiftEnvironmentV5(task)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V5},
        {"role": "user", "content": task.user_request},
    ]
    for index, call in enumerate(expert_calls_v5(task)):
        call_id = f"{task.task_id}-{index}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, sort_keys=True),
                        },
                    }
                ],
            }
        )
        result = env.step(call)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": call.name,
                "content": json.dumps(result.observation, sort_keys=True),
            }
        )
    if env.score != 1:
        raise ValueError("SFT target must complete safely")
    return {
        "id": task.task_id,
        "messages": messages,
        "tools": tool_schemas_v5(task.partner),
        "metadata": {
            "task_id": task.task_id,
            "split": task.split.value,
            "reward_version": VERSION,
            "negative_entitlement": is_negative_entitlement(task),
        },
    }


def prepare(output: Path, manifest: Path) -> dict[str, Any]:
    if output.exists() or manifest.exists():
        raise FileExistsError("v5 data preparation never overwrites prior artifacts")
    splits = {
        Split.TRAIN: training_tasks(Split.TRAIN, 16),
        Split.DEVELOPMENT: training_tasks(Split.DEVELOPMENT, 5),
    }
    output.mkdir(parents=True)
    files = {}
    for split, tasks in splits.items():
        path = output / f"{split.value}_tasks.jsonl"
        files[str(path)] = {
            "sha256": write_jsonl(path, ({**t.to_dict(), "id": t.task_id} for t in tasks)),
            "rows": len(tasks),
        }
    path = output / "train_sft.jsonl"
    files[str(path)] = {
        "sha256": write_jsonl(path, (sft_row_v5(t) for t in splits[Split.TRAIN])),
        "rows": len(splits[Split.TRAIN]),
    }
    result = {
        "version": VERSION,
        "files": files,
        "expert_replays": sum(map(len, splits.values())),
        "cells": {
            s.value: dict(
                Counter(
                    f"{t.partner.value}/"
                    + ("absent_entitlement" if is_negative_entitlement(t) else t.kind.value)
                    for t in ts
                )
            )
            for s, ts in splits.items()
        },
        "independent_test_included": False,
        "held_out_profiles_included": False,
        "scope": "Repair data using legacy scenario families; not an independent evaluation set.",
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(
        json.dumps(
            prepare(Path("data/generated/v5"), Path("manifests/experiment-v5-training-data.json")),
            indent=2,
            sort_keys=True,
        )
    )

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from schemashift.catalog import tool_schemas
from schemashift.generator import expert_calls
from schemashift.models import TaskSpec
from schemashift.prompts import OPTIMIZED_SYSTEM_PROMPT
from schemashift.simulator import SchemaShiftEnvironment


def rft_row(task: TaskSpec) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
        ],
        "input_metadata": {
            "row_id": task.task_id,
            "dataset_info": {
                "user_prompt_template": task.user_request,
                "environment_context": {
                    "seed": task.seed,
                    "task": task.to_dict(),
                },
            },
        },
        "ground_truth": {
            "task": task.to_dict(),
        },
        "metadata": {
            "task_id": task.task_id,
            "split": task.split.value,
            "partner": task.partner.value,
            "kind": task.kind.value,
            "seed": task.seed,
            "template_id": task.template_id,
        },
    }


def sft_row(task: TaskSpec) -> dict[str, Any]:
    env = SchemaShiftEnvironment(task)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
        {"role": "user", "content": task.user_request},
    ]
    for index, call in enumerate(expert_calls(task), start=1):
        call_id = f"call_{task.task_id}_{index}"
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
    if env.score != 1.0:
        raise ValueError(f"expert trajectory for {task.task_id} scored {env.score}")
    return {
        "messages": messages,
        "tools": tool_schemas(),
        "metadata": {
            "task_id": task.task_id,
            "partner": task.partner.value,
            "kind": task.kind.value,
            "generator_version": "schemashift-v1",
            "expert_calls": [asdict(call) for call in expert_calls(task)],
        },
    }

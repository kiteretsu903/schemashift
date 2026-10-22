"""Frozen-model normal versus genuine lookup-prefix evaluation.

The model loop, native-call handling, guard and scoring retain v5 semantics.
Injected calls are separated from model-selected calls and consume the same horizon.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from schemashift.agent import ToolModelClient, _parse_tool_call, _tool_message
from schemashift.agent_v5 import SYSTEM_PROMPT_V5, _digest
from schemashift.catalog import BINDING_BY_NAME
from schemashift.environment_v5 import VERSION, SchemaShiftEnvironmentV5
from schemashift.models import Partner, TaskKind, TaskSpec, ToolCall
from schemashift.rewards_v5 import evaluate_state_v5 as evaluate_state
from schemashift.safety_v4 import MUTATING_OPERATIONS, business_state
from schemashift.schema_v5 import backend_call, profile_call, tool_schemas_v5


def initial_state(task, profile, start_condition):
    if task.partner is not Partner.ATLAS or task.kind not in {
        TaskKind.UPDATE_CONTACT,
        TaskKind.NEEDS_APPROVAL,
    }:
        raise ValueError("v14 only probes the sealed Atlas-logical contact cases")
    if start_condition not in {"normal", "lookup_prefix"}:
        raise ValueError("unknown diagnostic start")
    env = SchemaShiftEnvironmentV5(task, profile=profile)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V5},
        {"role": "user", "content": task.user_request},
    ]
    injected = []
    if start_condition == "lookup_prefix":
        call = profile_call(
            ToolCall("atlas_find_account", {"email_address": task.account.email}, "v14-lookup-prefix"),
            profile,
        )
        before = business_state(env)
        result = env.step(call)
        if (
            not result.observation.get("ok")
            or result.terminated
            or result.truncated
            or result.info["unsafe_action_attempt"]
        ):
            raise ValueError("prefix must be a legal nonterminal read")
        if business_state(env) != before or env.state.steps != 1:
            raise ValueError("lookup changed business state or consumed wrong budget")
        if result.observation["account_ref"] not in env.state.resolved_accounts:
            raise ValueError("genuine observation not resolved in simulator")
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, sort_keys=True),
                        },
                    }
                ],
            }
        )
        messages.append(_tool_message(call.call_id, call.name, result.observation))
        injected.append(
            {
                "name": call.name,
                "arguments": call.arguments,
                "call_id": call.call_id,
                "argument_parse_error": None,
                "observation": result.observation,
                "reward": result.reward,
                "terminated": result.terminated,
                "truncated": result.truncated,
                "info": result.info,
            }
        )
    return env, messages, injected


async def run_episode_v14(
    task: TaskSpec,
    client: ToolModelClient,
    *,
    system_name: str,
    start_condition: str,
    system_prompt: str = SYSTEM_PROMPT_V5,
    max_attempts: int = 10,
    max_tokens: int = 512,
    profile: str | None = None,
) -> dict[str, Any]:
    env, messages, injected = initial_state(task, profile or task.partner.value, start_condition)
    schemas = tool_schemas_v5(task.partner, profile)
    config = {
        "start_condition": start_condition,
        "injected_prefix_sha256": _digest(injected),
        "schema_profile": profile or task.partner.value,
        "environment_version": VERSION,
        "model": client.model,
        "system_name": system_name,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "tool_schema_sha256": _digest(schemas),
        "max_attempts": max_attempts,
        "max_tokens": max_tokens,
        "max_steps": task.max_steps,
        "trusted_scope_source": "benchmark TaskSpec; not model-extracted intent",
        "task_sha256": _digest(task.to_dict()),
    }
    if system_prompt != SYSTEM_PROMPT_V5:
        raise ValueError("diagnostic prompt must remain frozen")
    calls: list[dict[str, Any]] = []
    post_terminal_proposals: list[dict[str, Any]] = []
    model_turns: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    started = time.perf_counter()
    api_error = None
    stop_reason = None
    while env.state.steps < task.max_steps and env.state.terminal_action is None and not env.state.violation:
        turn = await client.complete(messages, schemas, max_attempts=max_attempts, max_tokens=max_tokens)
        for key in usage:
            usage[key] += turn.usage[key]
        model_turns.append(asdict(turn))
        if turn.error or not turn.assistant_message:
            api_error = turn.error or "empty assistant message"
            stop_reason = "api_error"
            break
        messages.append(turn.assistant_message)
        raw_calls = turn.assistant_message.get("tool_calls") or []
        if not raw_calls:
            stop_reason = "model_stopped_without_terminal_tool"
            break
        for index, raw in enumerate(raw_calls):
            call, parse_error = _parse_tool_call(raw, index)
            result = env.step(call)
            calls.append(
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "call_id": call.call_id,
                    "argument_parse_error": parse_error,
                    "observation": result.observation,
                    "reward": result.reward,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "info": result.info,
                }
            )
            messages.append(_tool_message(call.call_id, call.name, result.observation))
            if result.terminated or result.truncated:
                for tail_index, tail in enumerate(raw_calls[index + 1 :], start=index + 1):
                    proposed, error = _parse_tool_call(tail, tail_index)
                    binding = BINDING_BY_NAME.get(backend_call(proposed, profile or task.partner.value).name)
                    post_terminal_proposals.append(
                        {
                            "name": proposed.name,
                            "arguments": proposed.arguments,
                            "argument_parse_error": error,
                            "mutating": bool(binding and binding.operation in MUTATING_OPERATIONS),
                        }
                    )
                break
    score = evaluate_state(task, env.state)
    suppressed_mutations = sum(p["mutating"] for p in post_terminal_proposals)
    if post_terminal_proposals:
        score = replace(
            score,
            score=0.0,
            terminal=0.0,
            success=False,
            valid_trajectory=False,
            reason="tool_calls_after_terminal_boundary",
        )
        stop_reason = "tool_calls_after_terminal_boundary"
    return {
        "record_version": "schemashift-prefix-diagnostic-v14-1",
        "start_condition": start_condition,
        "injected_prefix": injected,
        "episode_id": task.task_id + "/" + start_condition,
        "run_config": config,
        "run_config_sha256": _digest(config),
        "task_id": task.task_id,
        "split": task.split.value,
        "partner": task.partner.value,
        "kind": task.kind.value,
        "template_id": task.template_id,
        "seed": task.seed,
        "success": score.success,
        "score": score.score,
        "valid_trajectory": (
            score.valid_trajectory
            and env.state.terminal_action is not None
            and api_error is None
            and not any(c["argument_parse_error"] for c in calls)
        ),
        "unsafe_action_attempt": env.state.prohibited_side_effect or bool(suppressed_mutations),
        "post_terminal_proposals": post_terminal_proposals,
        "suppressed_mutation_proposals": suppressed_mutations,
        "blocked_mutation_attempt": any(c["info"]["blocked_mutation_attempt"] for c in calls),
        "business_mutation_calls": sum(c["info"]["business_state_changed"] for c in calls),
        "tool_error_calls": sum(not c["observation"].get("ok", False) for c in calls),
        "failure_reason": None if score.success else stop_reason or env.state.violation or score.reason,
        "api_error": api_error,
        "calls": calls,
        "model_turns": model_turns,
        "messages": messages,
        "usage": usage,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "reward_breakdown": score.to_dict(),
        "final_state": {
            "terminal_action": env.state.terminal_action,
            "terminal_payload": env.state.terminal_payload,
            "violation": env.state.violation,
            "steps": env.state.steps,
            "transition_state_digests": [t.state_digest for t in env.state.transitions],
        },
    }


async def run_diagnostic(cases, order, client, output_path, *, system_name):
    indexed = {c["task"]["task_id"]: c for c in cases}
    expected = {i + "/" + s for i in indexed for s in ["normal", "lookup_prefix"]}
    if len(order) != 48 or set(order) != expected or len(indexed) != 24:
        raise ValueError("all 48 predeclared task/start cells are required")
    semaphore = asyncio.Semaphore(8)
    output_path = Path(output_path)
    with output_path.open("x") as handle:

        async def run(identity):
            task_id, start = identity.split("/")
            case = indexed[task_id]
            async with semaphore:
                row = await run_episode_v14(
                    TaskSpec.from_dict(case["task"]),
                    client,
                    system_name=system_name,
                    start_condition=start,
                    profile=case["metadata"]["profile"],
                )
            row["case_metadata"] = case["metadata"]
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            return row

        async with asyncio.TaskGroup() as group:
            pending = [group.create_task(run(identity)) for identity in order]
    return [p.result() for p in pending]

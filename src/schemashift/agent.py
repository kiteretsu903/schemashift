from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from schemashift.catalog import tool_schemas
from schemashift.fireworks import ModelTurn
from schemashift.metrics import summarize
from schemashift.models import TaskSpec, ToolCall
from schemashift.rewards import evaluate_state
from schemashift.simulator import SchemaShiftEnvironment


class ToolModelClient(Protocol):
    model: str

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_attempts: int = 10,
        max_tokens: int = 512,
    ) -> ModelTurn: ...


@dataclass(frozen=True)
class RunConfig:
    model: str
    system_name: str
    system_prompt: str
    max_attempts: int
    max_tokens: int

    def to_dict(self) -> dict[str, Any]:
        tools_json = json.dumps(tool_schemas(), sort_keys=True, separators=(",", ":"))
        return {
            "model": self.model,
            "system_name": self.system_name,
            "system_prompt_sha256": hashlib.sha256(self.system_prompt.encode()).hexdigest(),
            "tool_schema_sha256": hashlib.sha256(tools_json.encode()).hexdigest(),
            "max_attempts": self.max_attempts,
            "max_tokens": self.max_tokens,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def load_tasks(path: Path) -> list[TaskSpec]:
    with path.open(encoding="utf-8") as handle:
        return [TaskSpec.from_dict(json.loads(line)) for line in handle if line.strip()]


def _tool_message(call_id: str, name: str, observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(observation, sort_keys=True),
    }


def _parse_tool_call(raw: dict[str, Any], index: int) -> tuple[ToolCall, str | None]:
    function = raw.get("function") or {}
    raw_arguments = function.get("arguments", "{}")
    parse_error: str | None = None
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        if not isinstance(arguments, dict):
            raise ValueError("arguments were not a JSON object")
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        arguments = {}
        parse_error = f"{type(exc).__name__}: {exc}"
    call_id = str(raw.get("id") or f"call_{index}")
    return ToolCall(name=str(function.get("name") or ""), arguments=arguments, call_id=call_id), parse_error


async def run_episode(
    task: TaskSpec,
    client: ToolModelClient,
    *,
    system_name: str,
    system_prompt: str,
    max_attempts: int = 10,
    max_tokens: int = 512,
) -> dict[str, Any]:
    env = SchemaShiftEnvironment(task)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.user_request},
    ]
    calls: list[dict[str, Any]] = []
    model_turns: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    started = time.perf_counter()
    api_error: str | None = None
    stop_reason: str | None = None

    while env.state.steps < task.max_steps and env.state.terminal_action is None and not env.state.violation:
        turn = await client.complete(
            messages,
            tool_schemas(),
            max_attempts=max_attempts,
            max_tokens=max_tokens,
        )
        usage["prompt_tokens"] += turn.usage["prompt_tokens"]
        usage["completion_tokens"] += turn.usage["completion_tokens"]
        model_turns.append(asdict(turn))
        if turn.error or not turn.assistant_message:
            api_error = turn.error or "empty assistant message"
            stop_reason = "api_error"
            break

        assistant = turn.assistant_message
        messages.append(assistant)
        raw_calls = assistant.get("tool_calls") or []
        if not raw_calls:
            stop_reason = "model_stopped_without_terminal_tool"
            break

        for index, raw_call in enumerate(raw_calls):
            call, parse_error = _parse_tool_call(raw_call, index)
            result = env.step(call)
            call_record = {
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
            calls.append(call_record)
            messages.append(_tool_message(call.call_id, call.name, result.observation))
            if result.terminated or result.truncated:
                break

    breakdown = evaluate_state(task, env.state)
    if stop_reason is None and not breakdown.success:
        stop_reason = env.state.violation or breakdown.reason
    failure_reason = None if breakdown.success else stop_reason or breakdown.reason
    config = RunConfig(client.model, system_name, system_prompt, max_attempts, max_tokens)
    return {
        "record_version": "schemashift-eval-v1",
        "run_config": config.to_dict(),
        "run_config_sha256": config.digest,
        "task_id": task.task_id,
        "split": task.split.value,
        "partner": task.partner.value,
        "kind": task.kind.value,
        "template_id": task.template_id,
        "seed": task.seed,
        "success": breakdown.success,
        "valid_trajectory": breakdown.valid_trajectory,
        "prohibited_side_effect": breakdown.prohibited_side_effect,
        "score": breakdown.score,
        "reward_breakdown": breakdown.to_dict(),
        "failure_reason": failure_reason,
        "api_error": api_error,
        "tool_call_count": len(calls),
        "model_turn_count": len(model_turns),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "usage": usage,
        "calls": calls,
        "model_turns": model_turns,
        "messages": messages,
        "final_state": {
            "terminal_action": env.state.terminal_action,
            "terminal_payload": env.state.terminal_payload,
            "violation": env.state.violation,
            "steps": env.state.steps,
            "transition_state_digests": [row.state_digest for row in env.state.transitions],
        },
    }


def _read_existing(path: Path, config_digest: str) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_config_sha256") != config_digest:
                raise ValueError(f"refusing to resume {path}: run configuration changed")
            rows.append(row)
    ids = [str(row["task_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"refusing to resume {path}: duplicate task records")
    return rows, set(ids)


async def run_evaluation(
    tasks: list[TaskSpec],
    client: ToolModelClient,
    *,
    system_name: str,
    system_prompt: str,
    output_path: Path,
    concurrency: int = 1,
    max_attempts: int = 10,
    max_tokens: int = 512,
    resume: bool = False,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least one")
    config = RunConfig(client.model, system_name, system_prompt, max_attempts, max_tokens)
    if output_path.exists() and not resume:
        raise FileExistsError(f"output exists; pass resume=True to continue: {output_path}")
    records, completed = _read_existing(output_path, config.digest) if resume else ([], set())
    pending = [task for task in tasks if task.task_id not in completed]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    async def run_and_persist(task: TaskSpec) -> dict[str, Any]:
        async with semaphore:
            record = await run_episode(
                task,
                client,
                system_name=system_name,
                system_prompt=system_prompt,
                max_attempts=max_attempts,
                max_tokens=max_tokens,
            )
        async with write_lock:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
        return record

    if pending:
        records.extend(await asyncio.gather(*(run_and_persist(task) for task in pending)))
    summary = summarize(records, system_name).to_dict()
    summary.update(
        {
            "model": client.model,
            "run_config_sha256": config.digest,
            "output_path": str(output_path),
        }
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

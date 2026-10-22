"""Auditable token-native rollouts with bounded whole-group resampling."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from schemashift.agent import _tool_message
from schemashift.catalog import tool_schemas
from schemashift.models import TaskSpec
from schemashift.prompts import OPTIMIZED_SYSTEM_PROMPT
from schemashift.rewards import evaluate_state
from schemashift.simulator import SchemaShiftEnvironment
from schemashift.training_v2 import completion_logprobs, parse_qwen_tool_calls, qwen_tool_response_suffix


def append_json(path: Path, record: dict[str, Any]) -> None:
    # All calls run on one event-loop thread; no await inside the append.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def has_signal(rewards: list[float]) -> bool:
    if len(rewards) < 2 or not all(math.isfinite(r) for r in rewards):
        return False
    return max(rewards) - min(rewards) > 1e-8


def bounded_groups(single: Any, *, size: int, rounds: int, concurrency: int, log_path: Path) -> Any:
    """Sample entire independent groups; never cherry-pick good individual runs.

    Rejected all-failure safety groups remain in telemetry and the replay queue.
    Only a full group may reach the trainer, so transport losses cannot change
    the comparison's membership silently.
    """
    if size < 2 or rounds < 1 or concurrency < size:
        raise ValueError("invalid group size, round bound, or concurrency")
    semaphore = asyncio.Semaphore(concurrency)
    pending: dict[tuple[int, str], tuple[asyncio.Task, set[int]]] = {}

    async def draw(row: dict[str, Any], context: dict[str, Any], round_index: int, index: int) -> Any:
        async with semaphore:
            return await single(row, **{**context, "rollout_idx": index, "resample_round": round_index})

    async def sample_group(row: dict[str, Any], context: dict[str, Any]) -> list[Any]:
        last = [None] * size
        for attempt in range(rounds):
            last = await asyncio.gather(*(draw(row, context, attempt, i) for i in range(size)))
            complete = all(run is not None for run in last)
            rewards = [float(run.segments[0].reward) for run in last if run is not None]
            signal = complete and has_signal(rewards)
            append_json(log_path, {
                "task_id": row["task_id"], "kind": row["kind"], "partner": row["partner"],
                "cursor_index": context["cursor_index"], "round": attempt,
                "expected_runs": size, "returned_runs": len(rewards), "rewards": rewards,
                "has_signal": signal, "selected": complete and (signal or attempt == rounds - 1),
                "needs_expert_replay": complete and all(r == 0 for r in rewards),
            })
            if complete and signal:
                return last
        # Incomplete groups are dropped together, not renormalized on survivors.
        return last if all(run is not None for run in last) else [None] * size

    async def rollout(row: dict[str, Any], *, cursor_index: int, rollout_idx: int, **context: Any) -> Any:
        if not 0 <= rollout_idx < size:
            raise ValueError("rollout index outside group")
        key = (cursor_index, row["task_id"])
        if key not in pending:
            ctx = {**context, "cursor_index": cursor_index}
            pending[key] = (asyncio.create_task(sample_group(row, ctx)), set())
        task, returned = pending[key]
        if rollout_idx in returned:
            raise ValueError("duplicate group member")
        returned.add(rollout_idx)
        try:
            runs = await task
            return runs[rollout_idx]
        finally:
            if len(returned) == size and task.done():
                pending.pop(key, None)

    return rollout


def make_rollout_fn(setup: Any) -> Any:
    from fireworks.training.sdk.deployment import DeploymentSampler
    from training.utils.rl.rollout import MessageTrajectoryAssembler, RolloutRun, RolloutSample, TITOTokenizer

    class Tokenizer(TITOTokenizer):
        def _tokenize_tool_segment(self, appended_messages: list[dict], *, tools: Any = None) -> list[int]:
            return self._encode_text(qwen_tool_response_suffix(appended_messages))

    sampler = DeploymentSampler(
        inference_url=setup.inference_base_url, model=setup.model, api_key=setup.api_key,
        tokenizer=setup.tokenizer,
    )
    sample_kwargs = {**setup.sample_kwargs, "top_p": 1.0, "top_k": 40}
    tools = tool_schemas()
    root = Path(setup.extras["run_root"])

    async def single(row: dict[str, Any], **context: Any) -> Any:
        task = TaskSpec.from_dict(row)
        env = SchemaShiftEnvironment(task)
        messages = [
            {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
            {"role": "user", "content": task.user_request},
        ]
        assembler = MessageTrajectoryAssembler(
            Tokenizer(setup.tokenizer, chat_template_kwargs={"enable_thinking": False})
        )
        turns = []
        failure = None
        infrastructure_error = None
        while env.state.steps < task.max_steps and not env.state.terminal_action and not env.state.violation:
            prompt_tokens = assembler.prepare_next_input(messages, tools=tools)
            try:
                completions = await sampler.sample_with_prompt_tokens(prompt_tokens, n=1, **sample_kwargs)
            except Exception as exc:
                # A transport failure is not a negative demonstration of model behavior.
                infrastructure_error = type(exc).__name__
                break
            if not completions:
                infrastructure_error = "empty_completion_response"
                break
            completion = completions[0]
            tokens = list(completion.full_tokens[int(completion.prompt_len):])
            logprobs = completion_logprobs(completion)
            if not tokens or logprobs is None or not all(math.isfinite(x) for x in logprobs):
                infrastructure_error = "missing_or_invalid_token_logprobs"
                break
            text = str(completion.text or setup.tokenizer.decode(tokens))
            try:
                content, raw_calls, calls = parse_qwen_tool_calls(text)
            except (ValueError, TypeError):
                # Keep sampled tokens so malformed JSON can receive a learning signal.
                content, raw_calls, calls = text, [], []
                failure = "malformed_tool_call"
            assistant = {"role": "assistant", "content": content}
            if raw_calls:
                assistant["tool_calls"] = raw_calls
            assembler.add_assistant_response(
                request_messages=messages, assistant_message=assistant,
                prompt_token_ids=prompt_tokens, completion_token_ids=tokens,
                completion_logprobs=logprobs, finish_reason=str(completion.finish_reason),
            )
            messages.append(assistant)
            record = {
                "text": text, "completion_token_ids": tokens, "completion_logprobs": logprobs,
                "finish_reason": str(completion.finish_reason), "calls": [],
            }
            turns.append(record)
            if not calls:
                failure = failure or "no_parseable_tool_call"
                break
            for call in calls:
                result = env.step(call)
                record["calls"].append({"call": asdict(call), "observation": result.observation})
                messages.append(_tool_message(call.call_id, call.name, result.observation))
                if result.terminated or result.truncated:
                    break
        score = evaluate_state(task, env.state)
        # Only the training objective changes; the frozen simulator/evaluator does not.
        training_reward = 0.0 if failure else score.score
        admitted = bool(turns) and infrastructure_error is None
        append_json(root / "rollouts.jsonl", {
            "version": "schemashift-v3-rollout-1", "task_id": task.task_id,
            **context, "training_reward": training_reward, "evaluation": score.to_dict(),
            "model_failure": failure, "infrastructure_error": infrastructure_error,
            "returned_to_group": admitted, "turns": turns,
        })
        if not admitted:
            return None
        tokens, logprobs, mask = assembler.trajectory.to_flat()
        return RolloutRun(segments=[RolloutSample(
            tokens=tokens, logprobs=logprobs, loss_mask=mask, reward=training_reward,
            finish_reason=turns[-1]["finish_reason"], text=turns[-1]["text"],
        )], metadata={"task_id": task.task_id, "success": score.success})

    return bounded_groups(
        single, size=setup.completions_per_prompt,
        rounds=int(setup.extras["sampling_rounds"]),
        concurrency=int(setup.extras.get("sampling_concurrency", 8)), log_path=root / "groups.jsonl",
    )

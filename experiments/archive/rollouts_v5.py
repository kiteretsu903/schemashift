"""V5 token-native, task-aware safe rollouts with whole-group signal admission."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from schemashift.agent import _tool_message
from schemashift.agent_v5 import SYSTEM_PROMPT_V5
from schemashift.catalog import BINDING_BY_NAME
from schemashift.environment_v5 import SchemaShiftEnvironmentV5
from schemashift.models import TaskSpec
from schemashift.rewards_v5 import evaluate_state_v5
from schemashift.safety_v4 import MUTATING_OPERATIONS
from schemashift.schema_v5 import backend_call, tool_schemas_v5
from schemashift.training_v2 import completion_logprobs, parse_qwen_tool_calls, qwen_tool_response_suffix


def parse_tool_calls_v5(text: str) -> Any:
    """Reject malformed leftover calls that the frozen permissive parser omits."""
    content, raw_calls, calls = parse_qwen_tool_calls(text)
    if "<tool_call>" in content or "</tool_call>" in content:
        raise ValueError("incomplete tool-call markup")
    if "<tool_call>" not in text and text.strip().startswith("{"):
        payload = json.loads(text.strip())
        proposals = payload.get("tool_calls") if isinstance(payload, dict) else None
        if isinstance(proposals, list) and any(not isinstance(call, dict) for call in proposals):
            raise ValueError("every proposed tool call must be an object")
    return content, raw_calls, calls


def append_json(path: Path, record: dict[str, Any]) -> None:
    # All calls run on one event-loop thread; no await inside the append.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def has_signal(rewards: list[float]) -> bool:
    if len(rewards) < 2 or not all(math.isfinite(r) for r in rewards):
        return False
    return max(rewards) - min(rewards) > 1e-8


def bounded_groups(
    single: Any,
    *,
    size: int,
    rounds: int,
    concurrency: int,
    log_path: Path,
    policy_version_getter: Any = None,
) -> Any:
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
            return await single(
                row,
                **{
                    **context,
                    "rollout_idx": index,
                    "resample_round": round_index,
                    "draw_id": f"{context['cursor_index']}:{round_index}:{index}",
                },
            )

    async def sample_group(row: dict[str, Any], context: dict[str, Any]) -> list[Any]:
        last = [None] * size
        for attempt in range(rounds):
            last = await asyncio.gather(*(draw(row, context, attempt, i) for i in range(size)))
            complete = all(run is not None for run in last)
            rewards = [float(run.segments[0].reward) for run in last if run is not None]
            signal = complete and has_signal(rewards)
            append_json(
                log_path,
                {
                    "task_id": row["task_id"],
                    "kind": row["kind"],
                    "partner": row["partner"],
                    "cursor_index": context["cursor_index"],
                    "round": attempt,
                    "draw_ids": [f"{context['cursor_index']}:{attempt}:{i}" for i in range(size)],
                    "policy_version": context.get("policy_version"),
                    "expected_runs": size,
                    "returned_runs": len(rewards),
                    "rewards": rewards,
                    "has_signal": signal,
                    "selected": signal,
                    "admitted_to_optimizer": False,
                    "needs_expert_replay": complete and all(r == 0 for r in rewards),
                },
            )
            if complete and signal:
                return last
        # Incomplete and constant-reward groups are dropped whole. No zero-signal Adam call.
        return [None] * size

    async def rollout(row: dict[str, Any], *, cursor_index: int, rollout_idx: int, **context: Any) -> Any:
        if not 0 <= rollout_idx < size:
            raise ValueError("rollout index outside group")
        key = (cursor_index, row["task_id"])
        if key not in pending:
            ctx = {
                **context,
                "cursor_index": cursor_index,
                "policy_version": policy_version_getter() if policy_version_getter else None,
            }
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
        inference_url=setup.inference_base_url,
        model=setup.model,
        api_key=setup.api_key,
        tokenizer=setup.tokenizer,
    )
    sample_kwargs = {**setup.sample_kwargs, "top_p": 1.0, "top_k": 0}
    root = Path(setup.extras["run_root"])

    async def single(row: dict[str, Any], **context: Any) -> Any:
        if row.get("id") != row["task_id"]:
            raise ValueError("rollout requires id=task_id")
        task = TaskSpec.from_dict(row)
        env = SchemaShiftEnvironmentV5(task)
        tools = tool_schemas_v5(task.partner)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V5},
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
            tokens = list(completion.full_tokens[int(completion.prompt_len) :])
            logprobs = completion_logprobs(completion)
            if not tokens or logprobs is None or not all(math.isfinite(x) for x in logprobs):
                infrastructure_error = "missing_or_invalid_token_logprobs"
                break
            text = str(completion.text or setup.tokenizer.decode(tokens))
            try:
                content, raw_calls, calls = parse_tool_calls_v5(text)
            except (ValueError, TypeError):
                # Keep sampled tokens so malformed JSON can receive a learning signal.
                content, raw_calls, calls = text, [], []
                failure = "malformed_tool_call"
            assistant = {"role": "assistant", "content": content}
            if raw_calls:
                assistant["tool_calls"] = raw_calls
            assembler.add_assistant_response(
                request_messages=messages,
                assistant_message=assistant,
                prompt_token_ids=prompt_tokens,
                completion_token_ids=tokens,
                completion_logprobs=logprobs,
                finish_reason=str(completion.finish_reason),
            )
            messages.append(assistant)
            record = {
                "text": text,
                "completion_token_ids": tokens,
                "completion_logprobs": logprobs,
                "finish_reason": str(completion.finish_reason),
                "proposed_calls": [asdict(call) for call in calls],
                "calls": [],
                "suppressed_calls": [],
            }
            turns.append(record)
            if not calls:
                failure = failure or "no_parseable_tool_call"
                break
            for call_index, call in enumerate(calls):
                result = env.step(call)
                record["calls"].append(
                    {"call": asdict(call), "observation": result.observation, "info": result.info}
                )
                messages.append(_tool_message(call.call_id, call.name, result.observation))
                if result.terminated or result.truncated:
                    for suppressed in calls[call_index + 1 :]:
                        binding = BINDING_BY_NAME.get(backend_call(suppressed, env.profile).name)
                        classification = (
                            "unknown"
                            if binding is None
                            else "mutation"
                            if binding.operation in MUTATING_OPERATIONS
                            else "nonmutating"
                        )
                        record["suppressed_calls"].append(
                            {
                                "call": asdict(suppressed),
                                "classification": classification,
                                "reason": "after_termination_or_truncation",
                                "executed": False,
                            }
                        )
                    if record["suppressed_calls"]:
                        failure = "post_termination_tool_calls"
                    break
        score = evaluate_state_v5(task, env.state)
        environment_evaluation = score.to_dict()
        suppressed_calls = [call for turn in turns for call in turn["suppressed_calls"]]
        suppressed_mutations = sum(call["classification"] == "mutation" for call in suppressed_calls)
        executed_calls = [call for turn in turns for call in turn["calls"]]
        if suppressed_calls:
            score = replace(
                score, score=0.0, success=False, valid_trajectory=False, reason="post_termination_tool_calls"
            )
        # The versioned scorer gives every unsafe or unterminated run zero credit.
        training_reward = 0.0 if failure else score.score
        admitted = bool(turns) and infrastructure_error is None
        append_json(
            root / "rollouts.jsonl",
            {
                "version": "schemashift-v5-rollout-1",
                "task_id": task.task_id,
                **context,
                "training_reward": training_reward,
                "evaluation": score.to_dict(),
                "environment_evaluation": environment_evaluation,
                "suppressed_tool_proposals": len(suppressed_calls),
                "suppressed_mutation_proposals": suppressed_mutations,
                "unsafe_action_attempt": bool(suppressed_mutations)
                or any(call["info"].get("unsafe_action_attempt", False) for call in executed_calls),
                "blocked_mutation_attempt": any(
                    call["info"].get("blocked_mutation_attempt", False) for call in executed_calls
                ),
                "model_failure": failure,
                "infrastructure_error": infrastructure_error,
                "returned_to_group": admitted,
                "turns": turns,
            },
        )
        if not admitted:
            return None
        tokens, logprobs, mask = assembler.trajectory.to_flat()
        return RolloutRun(
            segments=[
                RolloutSample(
                    tokens=tokens,
                    logprobs=logprobs,
                    loss_mask=mask,
                    reward=training_reward,
                    finish_reason=turns[-1]["finish_reason"],
                    text=turns[-1]["text"],
                )
            ],
            metadata={
                "task_id": task.task_id,
                "success": score.success,
                "cursor_index": context["cursor_index"],
                "draw_id": context["draw_id"],
                "policy_version": context.get("policy_version"),
            },
        )

    return bounded_groups(
        single,
        size=setup.completions_per_prompt,
        rounds=int(setup.extras["sampling_rounds"]),
        concurrency=int(setup.extras.get("sampling_concurrency", 8)),
        log_path=root / "groups.jsonl",
        policy_version_getter=setup.extras.get("policy_version_getter"),
    )

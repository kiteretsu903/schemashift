from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from schemashift.agent import _tool_message
from schemashift.catalog import tool_schemas
from schemashift.models import Partner, Split, TaskSpec, ToolCall
from schemashift.prompts import OPTIMIZED_SYSTEM_PROMPT
from schemashift.rewards import evaluate_state
from schemashift.simulator import SchemaShiftEnvironment

LOGGER = logging.getLogger(__name__)

BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
TOKENIZER_MODEL = "Qwen/Qwen3-4B"
TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora"
OUTPUT_MODEL = "schemashift-rft-qwen3-4b-v2"
COOKBOOK_COMMIT = "152faf05ae6f039b74158d2a84e133e65ad96071"

_TOOL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class CompletionLike(Protocol):
    full_tokens: list[int]
    prompt_len: int
    text: str
    finish_reason: str
    sampling_logprobs: list[float] | None
    logprobs_echoed: bool


def _json_object(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("tool-call payload must be a JSON object")
    return value


def _normalize_tool_payload(payload: dict[str, Any], index: int) -> tuple[dict[str, Any], ToolCall]:
    function = payload.get("function") if isinstance(payload.get("function"), dict) else payload
    name = str(function.get("name") or "").strip()
    if not name:
        raise ValueError("tool-call payload is missing name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = _json_object(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("tool-call arguments must be a JSON object")
    call_id = str(payload.get("id") or f"call_{index}")
    raw = {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, sort_keys=True, separators=(",", ":")),
        },
    }
    return raw, ToolCall(name=name, arguments=arguments, call_id=call_id)


def parse_qwen_tool_calls(text: str) -> tuple[str, list[dict[str, Any]], list[ToolCall]]:
    """Parse Qwen XML tool calls without accepting prose as an action."""
    matches = list(_TOOL_BLOCK.finditer(text))
    payloads: list[dict[str, Any]]
    if matches:
        payloads = [_json_object(match.group(1)) for match in matches]
        content = _TOOL_BLOCK.sub("", text).strip()
    else:
        stripped = text.strip()
        if not stripped.startswith("{"):
            return stripped, [], []
        payload = _json_object(stripped)
        raw_calls = payload.get("tool_calls")
        if not isinstance(raw_calls, list):
            return stripped, [], []
        payloads = [item for item in raw_calls if isinstance(item, dict)]
        content = ""

    raw_calls: list[dict[str, Any]] = []
    calls: list[ToolCall] = []
    for index, payload in enumerate(payloads):
        raw, call = _normalize_tool_payload(payload, index)
        raw_calls.append(raw)
        calls.append(call)
    return content, raw_calls, calls


def completion_logprobs(completion: CompletionLike) -> list[float] | None:
    values = completion.sampling_logprobs
    if values is None:
        return None
    normalized = list(values)
    prompt_len = int(completion.prompt_len)
    output_len = len(completion.full_tokens) - prompt_len
    if getattr(completion, "logprobs_echoed", False):
        full_len = len(completion.full_tokens)
        if len(normalized) == full_len:
            normalized = normalized[prompt_len:]
        elif len(normalized) == max(0, full_len - 1):
            normalized = normalized[max(0, prompt_len - 1) :]
    if len(normalized) != output_len or any(value is None for value in normalized):
        return None
    return [float(value) for value in normalized]


def qwen_tool_response_suffix(messages: list[dict[str, Any]]) -> str:
    if not messages or any(message.get("role") != "tool" for message in messages):
        raise ValueError("Qwen tool-response suffix requires one or more tool messages")
    chunks = ["<|im_start|>user"]
    for message in messages:
        chunks.extend(
            [
                "\n<tool_response>\n",
                str(message.get("content") or ""),
                "\n</tool_response>",
            ]
        )
    chunks.append("<|im_end|>\n")
    return "".join(chunks)


def validate_training_rows(rows: list[dict[str, Any]]) -> list[TaskSpec]:
    tasks = [TaskSpec.from_dict(row) for row in rows]
    if not tasks:
        raise ValueError("training rows are empty")
    wrong_split = [task.task_id for task in tasks if task.split is not Split.TRAIN]
    if wrong_split:
        raise ValueError(f"non-training rows are forbidden: {wrong_split[:3]}")
    orbit = [task.task_id for task in tasks if task.partner is Partner.ORBIT]
    if orbit:
        raise ValueError(f"held-out Orbit rows are forbidden: {orbit[:3]}")
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate training task IDs")
    return tasks


async def _append_record(path: Path, lock: asyncio.Lock, record: dict[str, Any]) -> None:
    async with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()


def make_rollout_fn(setup: Any) -> Any:
    """Build the token-native multi-turn rollout used by Fireworks async GRPO."""
    from fireworks.training.sdk.deployment import DeploymentSampler
    from training.utils.rl.rollout import (
        MessageTrajectoryAssembler,
        RolloutRun,
        RolloutSample,
        TITOTokenizer,
    )

    class QwenTITOTokenizer(TITOTokenizer):
        """Use Qwen's literal tool suffix when its full template is not prefix-stable."""

        def _tokenize_tool_segment(
            self,
            appended_messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
        ) -> list[int]:
            del tools
            return self._encode_text(qwen_tool_response_suffix(appended_messages))

    sampler = DeploymentSampler(
        inference_url=setup.inference_base_url,
        model=setup.model,
        api_key=setup.api_key,
        tokenizer=setup.tokenizer,
    )
    sample_kwargs = dict(setup.sample_kwargs)
    # The recipe deliberately defaults to full-distribution top_k=0. The frozen
    # managed-RFT contract used top_k=40, so override only that generated default.
    sample_kwargs.update({"top_p": 1.0, "top_k": 40})
    raw_path = Path(str(setup.extras["raw_rollout_path"]))
    raw_lock = asyncio.Lock()
    tools = tool_schemas()

    async def rollout_fn(
        sample_prompt: dict[str, Any],
        *,
        cursor_index: int = -1,
        row_index: int = -1,
        epoch: int = 0,
        rollout_idx: int = -1,
        **_: Any,
    ) -> Any:
        started = time.perf_counter()
        task = TaskSpec.from_dict(sample_prompt)
        validate_training_rows([sample_prompt])
        env = SchemaShiftEnvironment(task)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
            {"role": "user", "content": task.user_request},
        ]
        assembler = MessageTrajectoryAssembler(
            QwenTITOTokenizer(
                setup.tokenizer,
                chat_template_kwargs={"enable_thinking": False},
            )
        )
        turns: list[dict[str, Any]] = []
        error: str | None = None

        try:
            while (
                env.state.steps < task.max_steps
                and env.state.terminal_action is None
                and env.state.violation is None
            ):
                prompt_tokens = assembler.prepare_next_input(messages, tools=tools)
                completions = await sampler.sample_with_prompt_tokens(
                    prompt_tokens,
                    n=1,
                    **sample_kwargs,
                )
                if not completions:
                    error = "sampler returned no completion"
                    break
                completion = completions[0]
                prompt_len = int(completion.prompt_len)
                output_tokens = [int(value) for value in completion.full_tokens[prompt_len:]]
                output_logprobs = completion_logprobs(completion)
                if not output_tokens or output_logprobs is None:
                    error = "missing or misaligned completion tokens/logprobs"
                    break

                text = str(getattr(completion, "text", "") or setup.tokenizer.decode(output_tokens))
                content, raw_calls, calls = parse_qwen_tool_calls(text)
                assistant: dict[str, Any] = {"role": "assistant", "content": content}
                if raw_calls:
                    assistant["tool_calls"] = raw_calls
                assembler.add_assistant_response(
                    request_messages=messages,
                    assistant_message=assistant,
                    prompt_token_ids=prompt_tokens,
                    completion_token_ids=output_tokens,
                    completion_logprobs=output_logprobs,
                    finish_reason=str(getattr(completion, "finish_reason", "stop")),
                )
                messages.append(assistant)
                turn_record: dict[str, Any] = {
                    "text": text,
                    "prompt_token_count": len(prompt_tokens),
                    "completion_token_ids": output_tokens,
                    "completion_logprobs": output_logprobs,
                    "finish_reason": str(getattr(completion, "finish_reason", "stop")),
                    "calls": [],
                }
                turns.append(turn_record)
                if not calls:
                    error = "model stopped without a parseable tool call"
                    break

                for call in calls:
                    result = env.step(call)
                    turn_record["calls"].append(
                        {
                            "call": asdict(call),
                            "observation": result.observation,
                            "step_reward": result.reward,
                            "terminated": result.terminated,
                            "truncated": result.truncated,
                            "info": result.info,
                        }
                    )
                    messages.append(_tool_message(call.call_id, call.name, result.observation))
                    if result.terminated or result.truncated:
                        break
        except Exception as exc:  # preserve the trajectory before dropping invalid trainer data
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Training V2 rollout failed for %s", task.task_id)

        breakdown = evaluate_state(task, env.state)
        record = {
            "record_version": "schemashift-training-v2-rollout-v1",
            "task_id": task.task_id,
            "cursor_index": cursor_index,
            "row_index": row_index,
            "epoch": epoch,
            "rollout_idx": rollout_idx,
            "reward": breakdown.score,
            "reward_breakdown": breakdown.to_dict(),
            "error": error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "turns": turns,
            "final_state": {
                "terminal_action": env.state.terminal_action,
                "terminal_payload": env.state.terminal_payload,
                "violation": env.state.violation,
                "steps": env.state.steps,
            },
        }
        await _append_record(raw_path, raw_lock, record)
        if not assembler.checkpoints:
            return None
        tokens, logprobs, loss_mask = assembler.trajectory.to_flat()
        sample = RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            loss_mask=loss_mask,
            reward=breakdown.score,
            finish_reason=str(turns[-1]["finish_reason"]),
            text=str(turns[-1]["text"]),
        )
        return RolloutRun(
            segments=[sample],
            metadata={
                "task_id": task.task_id,
                "success": breakdown.success,
                "valid_trajectory": breakdown.valid_trajectory,
            },
        )

    return rollout_fn


def _load_rows(path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    validate_training_rows(rows)
    return rows if max_rows is None else rows[:max_rows]


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from training.recipes.async_rl_loop import Config, main
    from training.utils import DeployConfig, TrainerConfig, WandBConfig

    rows = _load_rows(args.tasks_path, args.max_rows)
    run_root = args.results_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    log_handler = logging.FileHandler(run_root / "training.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(log_handler)
    raw_rollout_path = run_root / "rollouts.jsonl"
    if raw_rollout_path.exists() and not args.resume:
        raise FileExistsError(f"raw rollout output exists: {raw_rollout_path}")
    config_record = {
        "run_id": args.run_id,
        "cookbook_commit": COOKBOOK_COMMIT,
        "base_model": BASE_MODEL,
        "tokenizer_model": TOKENIZER_MODEL,
        "training_shape": TRAINING_SHAPE,
        "output_model": args.output_model,
        "task_count": len(rows),
        "task_ids": [row["task_id"] for row in rows],
        "learning_rate": 0.0001,
        "lora_rank": 8,
        "epochs": 1,
        "completions_per_prompt": 4,
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": 40,
        "max_completion_tokens_per_turn": 512,
        "max_seq_len": 8192,
        "prompt_groups_per_step": args.prompt_groups_per_step,
        "max_concurrency_rollout_sample": 8,
        "max_head_offpolicy_versions": 0,
        "pipeline_chunks_per_step": 1,
        "kl_beta": 0.0,
        "anchor_logp": "old_policy",
        "seed": 0,
    }
    (run_root / "config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    cfg = Config(
        log_path=str(run_root / "recipe"),
        base_model=BASE_MODEL,
        learning_rate=0.0001,
        kl_beta=0.0,
        completions_per_prompt=4,
        max_completion_tokens=512,
        temperature=0.8,
        epochs=1,
        shuffle=True,
        seed=0,
        max_rows=len(rows),
        max_seq_len=8192,
        lora_rank=8,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_head_offpolicy_versions=0,
        pipeline_chunks_per_step=1,
        max_concurrency_rollout_sample=8,
        output_model_id=args.output_model,
        trainer=TrainerConfig(
            training_shape_id=TRAINING_SHAPE,
            inactivity_timeout="1800s",
        ),
        deployment=DeployConfig(
            tokenizer_model=TOKENIZER_MODEL,
            replica_count=1,
        ),
        wandb=WandBConfig(project=None),
        cleanup_on_exit=True,
        save_final_checkpoint=True,
    )
    result = main(
        cfg,
        rollout_fn_factory=make_rollout_fn,
        rows=rows,
        rollout_extras={"raw_rollout_path": str(raw_rollout_path)},
    )
    (run_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SchemaShift Fireworks Training V2 async GRPO")
    parser.add_argument("--tasks-path", type=Path, default=Path("data/generated/train_tasks.jsonl"))
    parser.add_argument("--results-root", type=Path, default=Path("results/training-v2"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--prompt-groups-per-step", type=int, default=8)
    parser.add_argument("--output-model", default=OUTPUT_MODEL)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main_cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("max_rows must be positive")
    if args.prompt_groups_per_step < 1:
        raise ValueError("prompt_groups_per_step must be positive")
    _run(args)


if __name__ == "__main__":
    main_cli()

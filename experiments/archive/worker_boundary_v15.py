"""Externally supervised v15 worker; matched variable-dose token-mean updates."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

from schemashift.execution_boundary_v15 import read, write_new
from schemashift.training_serialization_v11 import forward_backward_record
from schemashift.worker_decision_v10 import warmup

PLAN_PATH = Path("manifests/experiment-v15-boundary-plan.json")
EXPECTED_SUITES = {
    "regression",
    "previous",
    "observed",
    "holdout",
    "recent",
    "v10_fresh",
    "v11_fresh",
    "v13_fresh",
    "fresh",
}
SOURCE_JOB = "schemashift-v13-trainer-a-001"


def validate_dose(plan):
    dose = plan["dose"]
    updates = dose["per_update"]
    if dose["updates"] != 36 or dose["total_positive_loss_tokens"] != 51840 or len(updates) != 36:
        raise ValueError("v15 must contain 36 updates and 51840 loss tokens")
    totals = Counter()
    for step, item in enumerate(updates, 1):
        if item["update"] != step or set(item["profiles"]) != {"atlas", "nimbus", "cedar"}:
            raise ValueError("dose update/profile identities differ")
        if any(type(v) is not int or v <= 0 for v in item["profiles"].values()):
            raise ValueError("profile token counts must be positive integers")
        if item["tokens"] != sum(item["profiles"].values()):
            raise ValueError("dose total/profile sum differs")
        totals.update(item["profiles"])
    if sum(item["tokens"] for item in updates) != 51840 or totals != {p: 17280 for p in totals}:
        raise ValueError("whole-run token/profile dose differs")
    return updates


def render_batches(phase, plan=None):
    from training.recipes import sft_loop
    from training.utils.supervised import render_messages_to_datums

    plan = read(PLAN_PATH) if plan is None else plan
    expected_updates = validate_dose(plan)
    sft_loop._init_render_worker(
        "Qwen/Qwen3-4B",
        "qwen3_disable_thinking",
        "last_assistant_message",
        8192,
        tokenizer_revision="1cfa9a7208912126459214e8b04321603b3df60c",
    )
    state = sft_loop._worker_state
    serialized = list(map(json.loads, Path(phase["data_file"]).read_text().splitlines()))
    rows = {row["id"]: row for row in serialized}
    if len(rows) != len(serialized):
        raise ValueError("duplicate serialized row IDs")
    schedule = read(phase["schedule_file"])
    if len(schedule["updates"]) != len(expected_updates):
        raise ValueError("schedule update count differs")
    scheduled_ids = {assignment["id"] for update in schedule["updates"] for assignment in update}
    if scheduled_ids != set(rows):
        raise ValueError("scheduled and serialized IDs differ")
    batches, receipts = [], []
    cache = {}
    for step, assignments in enumerate(schedule["updates"], 1):
        batch, profile_tokens = [], {}
        for a in assignments:
            row = rows[a["id"]]
            if row["metadata"]["profile"] != a["profile"] or row["metadata"]["split"] != "train":
                raise ValueError("row profile/split differs")
            if a["id"] not in cache:
                rendered = render_messages_to_datums(
                    row["messages"],
                    renderer=state["renderer"],
                    train_on_what=state["train_on_what"],
                    tools=row["tools"],
                )
                if not isinstance(rendered, list):
                    rendered = [rendered]
                if len(rendered) != 1:
                    raise ValueError("prefix split")
                cache[a["id"]] = rendered[0].datum
            datum = cache[a["id"]]
            weights = list(datum.loss_fn_inputs["weights"].data)
            if not all(w in (0, 1) for w in weights):
                raise ValueError("fractional mask")
            count = int(sum(weights))
            if len(weights) > 8192 or count <= 0:
                raise ValueError("empty target or context overflow")
            if count != a["tokens"]:
                raise ValueError("token count changed")
            profile_tokens[a["profile"]] = profile_tokens.get(a["profile"], 0) + count
            batch.append(datum)
        expected = expected_updates[step - 1]
        if profile_tokens != expected["profiles"] or sum(profile_tokens.values()) != expected["tokens"]:
            raise ValueError("rendered per-update dose differs")
        batches.append(batch)
        receipts.append(
            {
                "step": step,
                "loss_tokens": expected["tokens"],
                "profiles": profile_tokens,
                "rows": len(batch),
                "input_tokens": sum(len(d.loss_fn_inputs["weights"].data) for d in batch),
                "max_sequence": max(len(d.loss_fn_inputs["weights"].data) for d in batch),
            }
        )
    if len(batches) != 36:
        raise ValueError("update count differs")
    return batches, receipts


def make_token_mean_loss(expected_tokens, expected_rows=None):
    """One denominator for every supervised token in the actual full update."""
    if type(expected_tokens) is not int or expected_tokens <= 0:
        raise ValueError("expected token count must be positive")

    def token_mean_loss(data, logprobs):
        from training.utils.losses import make_batch_weighted_sft_loss_fn

        if len(data) != len(logprobs) or (expected_rows is not None and len(data) != expected_rows):
            raise ValueError("update row/logprob count differs")
        masks = [list(d.loss_fn_inputs["weights"].data) for d in data]
        if not masks or any(w not in (0, 1) for mask in masks for w in mask):
            raise ValueError("whole-target binary masks required")
        total = sum(sum(mask) for mask in masks)
        if total != expected_tokens:
            raise ValueError("actual update normalization denominator differs")
        # No per-row or microbatch means; SDK computes one weighted token mean.
        return make_batch_weighted_sft_loss_fn()(data, logprobs)

    return token_mean_loss


def training_worker(seal, phase):
    import tinker
    from training.utils import ReconnectableClient, TrainerConfig, build_service_client
    from training.utils.checkpoints import TrainingCheckpoints

    from schemashift.training_v5 import verify_sdk

    root = Path(phase["result_root"])
    verify_sdk()
    if seal["source_job_id"] != SOURCE_JOB:
        raise ValueError("v15 source must be the frozen v13 control trainer")
    plan = read(seal.get("plan_path", PLAN_PATH))
    batches, receipts = render_batches(phase, plan)  # Before any trainer allocation.
    write_new(root / "rendered-batches.json", receipts)
    base = seal["base_model"]
    service = build_service_client(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url="https://api.fireworks.ai",
        additional_headers=None,
        base_model=base,
        tokenizer_model="Qwen/Qwen3-4B",
        lora_rank=8,
        max_context_length=8192,
        learning_rate=5e-6,
        trainer=TrainerConfig(
            job_id=phase["resource_id"],
            training_shape_id=seal["training_shape"],
            inactivity_timeout="600s",
            timeout_s=300,
            pending_timeout_s=300,
        ),
        cleanup_trainer_on_close=True,
    )
    try:
        raw = service.create_training_client(base, lora_rank=8, seed=0, lora_alpha=32)
        if service.trainer_job_id != phase["resource_id"]:
            raise ValueError("trainer identity changed")
        client = ReconnectableClient.from_training_client(
            raw,
            base_model=base,
            lora_rank=8,
            job_id=phase["resource_id"],
            service=service,
            default_timeout=120,
        )
        if seal["source_job_id"] != SOURCE_JOB:
            raise ValueError("v15 source must be the frozen v13 control trainer")
        source = raw.resolve_checkpoint_path(
            seal["source_checkpoint_name"], source_job_id=seal["source_job_id"]
        )
        if source != seal["source_checkpoint"]:
            raise ValueError("source differs")
        client.load_state(source)
        write_new(
            root / "initialization.json",
            {
                "source": source,
                "source_job_id": seal["source_job_id"],
                "source_checkpoint_name": seal["source_checkpoint_name"],
                "load_method": "load_state",
                "optimizer_reset": True,
                "local_step": 0,
                "cursor_index": 0,
                "seed": 0,
                "lora_rank": 8,
                "lora_alpha": 32,
            },
        )
        adam = tinker.AdamParams(
            learning_rate=5e-6, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, grad_clip_norm=0.0
        )
        write_new(root / "optimizer.json", adam.model_dump())
        for step, batch in enumerate(batches, 1):
            started = time.monotonic()
            expected_tokens = receipts[step - 1]["loss_tokens"]
            fb = client.forward_backward_custom(batch, make_token_mean_loss(expected_tokens, len(batch)))
            opt = client.optim_step(adam, grad_accumulation_normalization="none")
            record = {
                "step": step,
                "loss_tokens": expected_tokens,
                "profile_loss_tokens": receipts[step - 1]["profiles"],
                "seconds": time.monotonic() - started,
                "forward_backward": forward_backward_record(fb),
                "optimizer": opt.model_dump(mode="json"),
            }
            with (root / "steps.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps({"step": step, "seconds": record["seconds"], "metrics": fb.metrics}), flush=True)
        ckpt = TrainingCheckpoints(
            client, service, trainer_id=phase["resource_id"], log_path=str(root / "checkpoints"), lora_rank=8
        )
        ckpt.save("step-36", resumable=True, promotable=True, data_consumed=sum(map(len, batches)))
        promotion = ckpt.promote_latest(phase["output_model_id"], base)
        write_new(
            root / "result.json",
            {
                "steps": 36,
                "loss_tokens": 51840,
                "job_id": phase["resource_id"],
                "model": phase["model"],
                "registration": promotion,
                "adopted": False,
                "requested_final_checkpoint_name": "step-36",
                "checkpoint_counter_note": (
                    "Server counter may inherit the parent; selection is final local update 36 only."
                ),
            },
        )
    finally:
        service.close()


def validate_prefix_order(cases, order):
    indexed = {case["task"]["task_id"]: case for case in cases}
    expected = {task_id + "/lookup_prefix" for task_id in indexed}
    if len(cases) != 24 or len(indexed) != 24 or len(order) != 24 or set(order) != expected:
        raise ValueError("exactly the 24 frozen lookup-prefix cases are required")
    return indexed


async def run_lookup_prefix(cases, order, client, output_path, *, system_name):
    from schemashift.analysis_prefix_v14 import audit_episode, summarize
    from schemashift.diagnostic_prefix_v14 import run_episode_v14
    from schemashift.models import TaskSpec

    indexed = validate_prefix_order(cases, order)
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
                    max_attempts=10,
                    max_tokens=512,
                )
            row["case_metadata"] = case["metadata"]
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            return row

        async with asyncio.TaskGroup() as group:
            pending = [group.create_task(run(identity)) for identity in order]
    rows = [task.result() for task in pending]
    if len(rows) != 24 or {row["episode_id"] for row in rows} != set(order):
        raise ValueError("lookup-prefix output is incomplete or duplicated")
    # Model failures and unsafe attempts remain outcomes. Replay drift/API errors
    # abort the phase. The old audit_run requires 48 normal+prefix rows, so audit
    # each genuine prefix episode directly without fabricating normal episodes.
    audited = [audit_episode(row, indexed[row["task_id"]], client.model) for row in rows]
    if not all(row["replay_passed"] for row in audited):
        raise ValueError("lookup-prefix replay failed")
    audit = {
        "version": "v15-lookup-prefix-replay-1",
        "passed": True,
        "episodes": 24,
        "raw_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rows": audited,
    }
    write_new(output_path.with_name("lookup_prefix-audit.json"), audit)
    return {"records": 24, "replay_passed": True, **summarize(audited)}


def evaluation_worker(seal, phase):
    from schemashift.agent_v5 import run_evaluation_v5
    from schemashift.execution_boundary_v15 import checked
    from schemashift.fireworks import FireworksToolClient
    from schemashift.independent_v5 import validate_protocol

    root = Path(phase["result_root"])
    model = phase["model"]
    suites = {}
    expected_suites = {"fresh"} if phase["label"] == "v9" else EXPECTED_SUITES
    if len(phase["suites"]) != len(expected_suites) or set(phase["suites"]) != expected_suites:
        raise ValueError("evaluation suite coverage differs")
    prefix_enabled = phase.get("include_lookup_prefix", phase["label"] != "v9")
    if prefix_enabled != (phase["label"] != "v9"):
        raise ValueError("lookup-prefix coverage differs")
    cases = read(seal["diagnostic_cases_path"])["cases"] if prefix_enabled else None
    if cases is not None:
        validate_prefix_order(cases, seal["prefix_episode_order"])
    for name in phase["suites"]:
        path = Path(seal["suite_protocols"][name])
        protocol, tasks, profiles = validate_protocol(path, model)
        if protocol["model_endpoints"][model] != f"accounts/example/deployments/{phase['resource_id']}":
            raise ValueError("model/deployment binding changed")
        by_id = {task.task_id: task for task in tasks}
        order = seal["task_orders"][phase["label"]][name]
        if len(tasks) != 96 or len(by_id) != 96 or len(order) != 96 or set(order) != set(by_id):
            raise ValueError("suite task IDs/order must contain all 96 frozen episodes")
        tasks = [by_id[identity] for identity in order]
        suites[name] = (tasks, profiles)
    write_new(
        root / "config.json",
        {
            **phase,
            "suite_protocols": seal["suite_protocols"],
            "temperature": 0,
            "max_tokens": 512,
            "max_attempts": 10,
            "concurrency": 8,
            "optional_suites": [],
        },
    )
    payload = checked(
        "deployment",
        "create",
        model,
        "--deployment-id",
        phase["resource_id"],
        "--deployment-shape",
        phase["deployment_shape"],
        "--min-replica-count",
        "0",
        "--max-replica-count",
        "1",
        "--scale-to-zero-window",
        "300s",
        "-o",
        "json",
        timeout=60,
    )
    write_new(root / "deployment-create.json", json.loads(payload))
    client = FireworksToolClient(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url="https://api.fireworks.ai/inference/v1",
        model=f"accounts/example/deployments/{phase['resource_id']}",
    )

    async def evaluate():
        try:
            await warmup(client, root)
            results = {}
            for suite, (tasks, profiles) in suites.items():
                results[suite] = await run_evaluation_v5(
                    tasks,
                    client,
                    system_name=phase["label"],
                    output_path=root / f"{suite}.jsonl",
                    concurrency=8,
                    max_attempts=10,
                    max_tokens=512,
                    profiles=profiles,
                )
                if results[suite]["records"] != 96 or results[suite]["api_error_episodes"]:
                    raise ValueError("incomplete or API-error evaluation suite")
                (root / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps({"suite_completed": suite, "result": results[suite]}), flush=True)
            if prefix_enabled:
                results["lookup_prefix"] = await run_lookup_prefix(
                    cases,
                    seal["prefix_episode_order"],
                    client,
                    root / "lookup_prefix.jsonl",
                    system_name=phase["label"],
                )
            write_new(root / "result.json", results)
        finally:
            await client.client.close()

    asyncio.run(evaluate())


if __name__ == "__main__":
    from schemashift.execution_boundary_v15 import verify_seal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seal = verify_seal()
    phase = seal["phases"][args.phase]
    if not os.environ.get("FIREWORKS_API_KEY") or not (Path(phase["result_root"]) / "started.json").exists():
        raise RuntimeError("external supervisor required")
    (training_worker if args.phase.startswith("training_") else evaluation_worker)(seal, phase)

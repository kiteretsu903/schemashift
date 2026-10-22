"""Single supervised v13 phase; exact whole-target token-mean SFT."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from schemashift.execution_decision_v10 import checked, read, write_new
from schemashift.training_serialization_v11 import forward_backward_record
from schemashift.worker_decision_v10 import warmup


def render_batches(phase):
    from training.recipes import sft_loop
    from training.utils.supervised import render_messages_to_datums

    sft_loop._init_render_worker(
        "Qwen/Qwen3-4B",
        "qwen3_disable_thinking",
        "last_assistant_message",
        8192,
        tokenizer_revision="1cfa9a7208912126459214e8b04321603b3df60c",
    )
    state = sft_loop._worker_state
    rows = {r["id"]: r for r in map(json.loads, Path(phase["data_file"]).read_text().splitlines())}
    schedule = read(phase["schedule_file"])
    batches, receipts = [], []
    cache = {}
    for step, assignments in enumerate(schedule["updates"], 1):
        batch, profile_tokens = [], {}
        for a in assignments:
            row = rows[a["id"]]
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
            if count != a["tokens"]:
                raise ValueError("token count changed")
            profile_tokens[a["profile"]] = profile_tokens.get(a["profile"], 0) + count
            batch.append(datum)
        if profile_tokens != {"atlas": 480, "nimbus": 480, "cedar": 480}:
            raise ValueError("dose differs")
        batches.append(batch)
        receipts.append(
            {
                "step": step,
                "loss_tokens": 1440,
                "profiles": profile_tokens,
                "rows": len(batch),
                "input_tokens": sum(len(d.loss_fn_inputs["weights"].data) for d in batch),
                "max_sequence": max(len(d.loss_fn_inputs["weights"].data) for d in batch),
            }
        )
    if len(batches) != 36:
        raise ValueError("update count differs")
    return batches, receipts


def token_mean_loss(data, logprobs):
    from training.utils.losses import make_batch_weighted_sft_loss_fn

    if len(data) != len(logprobs):
        raise ValueError("logprob rows differ")
    total = sum(sum(d.loss_fn_inputs["weights"].data) for d in data)
    if total != 1440:
        raise ValueError("normalization denominator differs")
    return make_batch_weighted_sft_loss_fn()(data, logprobs)


def training_worker(seal, phase):
    import tinker
    from training.utils import ReconnectableClient, TrainerConfig, build_service_client
    from training.utils.checkpoints import TrainingCheckpoints

    from schemashift.training_v5 import verify_sdk

    root = Path(phase["result_root"])
    verify_sdk()
    batches, receipts = render_batches(phase)  # Before any trainer allocation.
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
        source = raw.resolve_checkpoint_path("step-117", source_job_id="training-api-service-b77db661")
        if source != seal["source_checkpoint"]:
            raise ValueError("source differs")
        client.load_state(source)
        write_new(
            root / "initialization.json",
            {
                "source": source,
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
            fb = client.forward_backward_custom(batch, token_mean_loss)
            opt = client.optim_step(adam, grad_accumulation_normalization="none")
            record = {
                "step": step,
                "loss_tokens": 1440,
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
            },
        )
    finally:
        service.close()


def evaluation_worker(seal, phase):
    from schemashift.agent_v5 import run_evaluation_v5
    from schemashift.fireworks import FireworksToolClient
    from schemashift.independent_v5 import validate_protocol

    root = Path(phase["result_root"])
    model = phase["model"]
    suites = {}
    for name in phase["suites"]:
        path = Path(seal["suite_protocols"][name])
        protocol, tasks, profiles = validate_protocol(path, model)
        if protocol["model_endpoints"][model] != f"accounts/example/deployments/{phase['resource_id']}":
            raise ValueError("model/deployment binding changed")
        if name == "fresh":
            by_id = {t.task_id: t for t in tasks}
            tasks = [by_id[i] for i in seal["fresh_task_orders"][phase["label"]]]
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
                (root / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps({"suite_completed": suite, "result": results[suite]}), flush=True)
            write_new(root / "result.json", results)
        finally:
            await client.client.close()

    asyncio.run(evaluate())


if __name__ == "__main__":
    from schemashift.execution_contact_v13 import verify_seal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seal = verify_seal()
    phase = seal["phases"][args.phase]
    if not os.environ.get("FIREWORKS_API_KEY") or not (Path(phase["result_root"]) / "started.json").exists():
        raise RuntimeError("external supervisor required")
    (training_worker if args.phase.startswith("training_") else evaluation_worker)(seal, phase)

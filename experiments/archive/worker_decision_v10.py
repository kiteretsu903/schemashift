"""Worker for one externally supervised v10 paid phase. Not a standalone launcher."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from schemashift.execution_decision_v10 import checked, read, verify_seal, write_new


def training_worker(seal, phase):
    from training.recipes import sft_loop

    from schemashift.training_counterbalanced_v9 import isolated_sft_entry
    from schemashift.training_v5 import archive_training_inputs, verify_sdk

    root = Path(phase["result_root"])
    protocol = read("manifests/experiment-v10-training-protocol.json")
    write_new(root / "config.json", protocol)
    write_new(
        root / "source-receipt.json",
        {
            "checkpoint": protocol["source_checkpoint"],
            "weights_only": True,
            "local_cursor": 0,
            "local_step": 0,
        },
    )
    provenance = verify_sdk()
    args = argparse.Namespace(
        data_dir=Path(protocol["data_dir"]),
        manifest=Path(protocol["data_manifest"]),
        source_receipt=root / "source-receipt.json",
    )
    archive_training_inputs(root, args, provenance)
    config = build_training_config(protocol, phase)
    # The pinned SDK creates this stable ID only when it is absent. The supervisor
    # establishes absence before starting and independently owns deletion afterward.
    result = isolated_sft_entry(sft_loop, root)(config)
    if result["steps"] != 113 or result["job_id"] != phase["resource_id"]:
        raise RuntimeError("training steps/resource identity differ from frozen recipe")
    initialization = [json.loads(line) for line in (root / "initialization.jsonl").read_text().splitlines()]
    if len(initialization) != 1 or initialization[0]["load_method"] != "load_state":
        raise RuntimeError("weights-only initialization was not verified")
    if initialization[0]["recipe_step"] or initialization[0]["cursor_index"]:
        raise RuntimeError("local optimizer step or data cursor was resumed")
    write_new(root / "result.json", result)


def build_training_config(protocol, phase):
    from training.recipes import sft_loop
    from training.utils import TrainerConfig, WandBConfig

    root = Path(phase["result_root"])
    settings = protocol["sft"]
    return sft_loop.Config(
        log_path=str(root / "sft"),
        base_model=protocol["base_model"],
        dataset=str(Path(protocol["data_dir"]) / "train_sft.jsonl"),
        tokenizer_model=protocol["tokenizer_model"],
        tokenizer_revision=protocol["tokenizer_revision"],
        renderer_name=settings["renderer_name"],
        train_on_what=settings["train_on_what"],
        learning_rate=settings["learning_rate"],
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        max_seq_len=settings["max_seq_len"],
        max_examples=settings["rows"],
        lora_rank=settings["lora_rank"],
        seed=settings["seed"],
        render_workers=settings["render_workers"],
        pipeline_depth=settings["pipeline_depth"],
        group_by_length=False,
        evaluation_dataset="",
        eval_auto_carveout=False,
        save_final_checkpoint=True,
        dcp_save_interval=0,
        adam_beta2=settings["beta2"],
        weight_decay=settings["weight_decay"],
        grad_clip_norm=settings["grad_clip_norm"],
        warmup_steps=0,
        init_from_checkpoint=protocol["source_checkpoint"],
        output_model_id=protocol["output_model_id"],
        trainer=TrainerConfig(
            job_id=phase["resource_id"],
            training_shape_id=protocol["training_shape"],
            inactivity_timeout="600s",
            timeout_s=300,
            pending_timeout_s=300,
        ),
        wandb=WandBConfig(project=None),
    )


async def warmup(client, root):
    started = time.monotonic()
    errors = []
    probe = client.client.with_options(timeout=20, max_retries=0)
    for attempt in range(120):
        try:
            await probe.chat.completions.create(
                model=client.model,
                messages=[{"role": "user", "content": "Reply ready."}],
                temperature=0,
                reasoning_effort="none",
                max_tokens=8,
            )
            break
        except Exception as exc:
            errors.append(type(exc).__name__)
            if attempt == 119:
                raise RuntimeError("evaluation deployment never became ready") from exc
            await asyncio.sleep(5)
    await asyncio.sleep(60)
    stable = 0
    for _ in range(24):
        replies = await asyncio.gather(
            *[
                probe.chat.completions.create(
                    model=client.model,
                    messages=[{"role": "user", "content": "Reply ready."}],
                    temperature=0,
                    reasoning_effort="none",
                    max_tokens=8,
                )
                for _ in range(8)
            ],
            return_exceptions=True,
        )
        failures = [type(reply).__name__ for reply in replies if isinstance(reply, BaseException)]
        errors.extend(failures)
        stable = 0 if failures else stable + 1
        if stable == 3:
            break
        await asyncio.sleep(5)
    if stable != 3:
        raise RuntimeError("evaluation routing did not become consistently ready")
    write_new(
        root / "warmup.json",
        {
            "seconds": time.monotonic() - started,
            "transient_errors": errors,
            "successful_probe_batches": stable,
            "excluded_from_outcome_metrics": True,
            "included_in_phase_budget": True,
        },
    )


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=["training", "candidate_evaluation", "control_evaluation"]
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    execution = verify_seal()
    config = execution["phases"][args.phase]
    if not os.environ.get("FIREWORKS_API_KEY") or not (Path(config["result_root"]) / "started.json").exists():
        raise RuntimeError("requires the authorized external supervisor")
    if args.phase == "training":
        training_worker(execution, config)
    else:
        evaluation_worker(execution, config)

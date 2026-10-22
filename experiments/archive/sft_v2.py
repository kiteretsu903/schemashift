from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
TOKENIZER_MODEL = "Qwen/Qwen3-4B"
RENDERER_NAME = "qwen3_disable_thinking"
TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora"
OUTPUT_MODEL = "schemashift-sft-qwen3-4b-v2"
COOKBOOK_COMMIT = "152faf05ae6f039b74158d2a84e133e65ad96071"


def load_and_validate_rows(path: Path, *, expected_rows: int = 240) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} SFT rows, found {len(rows)}")
    task_ids: list[str] = []
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        messages = row.get("messages")
        tools = row.get("tools")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("task_id"), str):
            raise ValueError(f"row {index} lacks a task_id")
        if metadata.get("partner") not in {"atlas", "nimbus"}:
            raise ValueError(f"row {index} contains a held-out or unknown partner")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"row {index} lacks messages")
        if not isinstance(tools, list) or not tools:
            raise ValueError(f"row {index} lacks tool schemas")
        task_ids.append(metadata["task_id"])
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate SFT task IDs")
    return rows


def audit_rendering(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from training.utils.supervised import build_renderer, render_messages_to_datums
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    renderer = build_renderer(
        tokenizer,
        TOKENIZER_MODEL,
        renderer_name=RENDERER_NAME,
        load_image_processor=False,
    )
    examples = 0
    trained_tokens = 0.0
    maximum_length = 0
    for row in rows:
        datums = render_messages_to_datums(
            row["messages"],
            renderer=renderer,
            tools=row["tools"],
            train_on_what="all_assistant_messages",
            max_seq_len=8192,
        )
        if not datums:
            raise ValueError(f"no rendered datum for {row['metadata']['task_id']}")
        for rendered in datums:
            if len(rendered.token_ids) > 8192:
                raise ValueError(f"rendered sequence exceeds 8192 for {row['metadata']['task_id']}")
            weight = float(sum(rendered.token_weights))
            if weight <= 0:
                raise ValueError(f"no supervised tokens for {row['metadata']['task_id']}")
            examples += 1
            trained_tokens += weight
            maximum_length = max(maximum_length, len(rendered.token_ids))
    return {
        "rows": len(rows),
        "rendered_examples": examples,
        "trained_tokens": int(trained_tokens),
        "max_sequence_tokens": maximum_length,
        "renderer": RENDERER_NAME,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from training.recipes import sft_loop
    from training.utils import TrainerConfig, WandBConfig

    rows = load_and_validate_rows(args.dataset)
    run_root = args.results_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler = logging.FileHandler(run_root / "training.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)

    config_record = {
        "base_model": BASE_MODEL,
        "batch_size": 32,
        "cookbook_commit": COOKBOOK_COMMIT,
        "dataset": str(args.dataset),
        "epochs": 2,
        "learning_rate": 0.0001,
        "lora_rank": 8,
        "max_seq_len": 8192,
        "output_model": args.output_model,
        "renderer": RENDERER_NAME,
        "rows": len(rows),
        "seed": 0,
        "train_on_what": "all_assistant_messages",
        "training_shape": TRAINING_SHAPE,
    }
    (run_root / "config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = sft_loop.Config(
        log_path=str(run_root / "recipe"),
        render_samples_file=str(run_root / "render-samples.jsonl"),
        render_samples_limit=20,
        base_model=BASE_MODEL,
        dataset=str(args.dataset),
        tokenizer_model=TOKENIZER_MODEL,
        renderer_name=RENDERER_NAME,
        train_on_what="all_assistant_messages",
        learning_rate=0.0001,
        epochs=2,
        batch_size=32,
        max_seq_len=8192,
        max_examples=len(rows),
        lora_rank=8,
        output_model_id=args.output_model,
        seed=0,
        render_workers=1,
        pipeline_depth=1,
        trainer=TrainerConfig(
            training_shape_id=TRAINING_SHAPE,
            inactivity_timeout="1800s",
        ),
        wandb=WandBConfig(project=None),
    )
    result = sft_loop.main(config)
    (run_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SchemaShift Fireworks Training V2 SFT")
    parser.add_argument("--dataset", type=Path, default=Path("data/generated/train_sft.jsonl"))
    parser.add_argument("--results-root", type=Path, default=Path("results/training-v2-sft"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-model", default=OUTPUT_MODEL)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        rows = load_and_validate_rows(args.dataset)
        print(json.dumps(audit_rendering(rows), indent=2, sort_keys=True))
        return
    run(args)


if __name__ == "__main__":
    main()

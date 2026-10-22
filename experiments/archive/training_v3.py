"""Plan and run a bounded SFT-to-GRPO experiment without modifying v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from schemashift.experiment_v3 import VERSION, generate_balanced
from schemashift.models import Split
from schemashift.rollouts_v3 import append_json, has_signal, make_rollout_fn
from schemashift.training_v2 import BASE_MODEL, COOKBOOK_COMMIT, TOKENIZER_MODEL, TRAINING_SHAPE

ARMS = {
    "warmstart": {"sampling_rounds": 1, "learning_rate": 1e-4, "filter_constant": False},
    "resample": {"sampling_rounds": 3, "learning_rate": 1e-4, "filter_constant": True},
    "conservative": {"sampling_rounds": 3, "learning_rate": 1e-5, "filter_constant": True},
}


def load_rows(data_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != VERSION or manifest.get("private_test_generated") is not False:
        raise ValueError("requires the v3 development-only manifest")
    for name in ("train_tasks.jsonl", "train_sft.jsonl", "development_tasks.jsonl"):
        path = data_dir / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest["files"].get(str(path), {}).get("sha256") != digest:
            raise ValueError(f"dataset hash mismatch: {path}")
    rows = [json.loads(line) for line in (data_dir / "train_tasks.jsonl").read_text().splitlines()]
    expected = [json.loads(json.dumps(t.to_dict())) for t in generate_balanced(Split.TRAIN)]
    if rows != expected:
        raise ValueError("training rows differ from the balanced generator; no evaluation rows allowed")
    return rows


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,50}", args.run_id):
        raise ValueError("run_id must be a short lowercase name")
    if args.max_rows < 16 or args.max_rows > 240 or args.max_rows % 16:
        raise ValueError("max_rows must preserve complete 16-cell blocks (16..240)")
    if not 60 <= args.timeout_seconds <= 1800:
        raise ValueError("timeout_seconds must be between 60 and 1800")
    concurrency = getattr(args, "rollout_concurrency", 8)
    if concurrency not in (8, 16, 32):
        raise ValueError("rollout_concurrency must be 8, 16, or 32")
    rows = load_rows(args.data_dir, args.manifest)[:args.max_rows]
    arm = ARMS[args.arm]
    plan = {
        "version": "schemashift-v3-run-1", "run_id": args.run_id, "arm": args.arm,
        "base_model": BASE_MODEL, "cookbook_commit": COOKBOOK_COMMIT,
        "data_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "source": "fresh balanced SFT control, then reload its checkpoint before RFT",
        "sft": {
            "rows": 16 if args.max_rows != 240 else 240,
            "epochs": 1 if args.max_rows != 240 else 2,
            "learning_rate": 1e-4, "batch_size": 32,
        },
        "rft": {
            **arm, "rows": len(rows), "epochs": 1, "group_size": 4,
            "max_rollout_attempts": len(rows) * 4 * arm["sampling_rounds"],
            "kl_beta": 0.0, "lora_rank": 8, "temperature": 0.8,
            "top_k": 40, "max_tokens_per_turn": 512, "max_context": 8192,
            "rollout_concurrency": concurrency,
        },
        "timeout_seconds": args.timeout_seconds,
        "is_smoke": len(rows) != 240,
        "checkpoint_selection": "development safety gates; never training reward or old private test",
        "private_test_generated": False,
        "limitations": [
            "No claim of improvement until matched development and fresh final evaluation.",
            "KL remains zero: the pinned recipe references base weights, not the SFT adapter.",
            "Cross-phase DCP loading restores SFT optimizer state as well as weights.",
            "Safety replay examples are prepared; interleaved supervised replay is not implemented.",
        ],
    }

    if getattr(args, "source_receipt", None):
        source = json.loads(args.source_receipt.read_text())
        if source["data_manifest_sha256"] != plan["data_manifest_sha256"]:
            raise ValueError("source checkpoint used different training data")
        if source["is_smoke"] != plan["is_smoke"]:
            raise ValueError("cannot mix smoke and full-quality source checkpoints")
        if not re.fullmatch(r"cross_job://[a-zA-Z0-9-]+/[a-zA-Z0-9-]+", source["source_checkpoint"]):
            raise ValueError("requires a verified cross-job checkpoint reference")
        plan["source_checkpoint"] = source["source_checkpoint"]
        plan["source"] = "verified retained SFT checkpoint; skip SFT retraining"
    return plan


def budget_check(receipt: dict[str, Any], plan: dict[str, Any]) -> None:
    """A short-lived receipt binds the launch to a reviewed conservative estimate."""
    checked = datetime.fromisoformat(receipt["checked_at_utc"])
    age = datetime.now(UTC) - checked
    if checked.tzinfo is None or not timedelta(0) <= age <= timedelta(minutes=15):
        raise ValueError("budget preflight is stale or has an invalid timestamp")
    if receipt.get("run_id") != plan["run_id"] or receipt.get("account") != "example":
        raise ValueError("budget receipt belongs to a different run or account")
    if receipt.get("active_project_resources") != 0:
        raise ValueError("account for active project resource cost before launching")
    fields = ("balance_usd", "posted_month_usage_usd", "estimated_max_cost_usd", "hourly_cost_bound_usd")
    if any(not math.isfinite(float(receipt[k])) or float(receipt[k]) < 0 for k in fields):
        raise ValueError("invalid cost estimate")
    cost = float(receipt["estimated_max_cost_usd"])
    if float(receipt["hourly_cost_bound_usd"]) <= 0:
        raise ValueError("hourly cost bound must be positive")
    minimum_estimate = float(receipt["hourly_cost_bound_usd"]) * (plan["timeout_seconds"] + 300) / 3600
    if cost <= 0 or cost < minimum_estimate:
        raise ValueError("estimate must cover timeout plus five minutes of cleanup")
    if cost > float(receipt["balance_usd"]) - 5:
        raise ValueError("preserve a $5 credit reserve; no automatic top-up")
    if float(receipt["posted_month_usage_usd"]) + cost > 45:
        raise ValueError("project's $45 monthly operational cutoff would be exceeded")


def group_filter(root: Path, enabled: bool) -> Any:
    def keep(group: Any) -> bool:
        signal_present = has_signal(list(group.rewards))
        accepted = signal_present or not enabled
        append_json(root / "optimizer-groups.jsonl", {
            "row": group.row_meta, "rewards": group.rewards,
            "advantages": group.advantages, "has_signal": signal_present, "accepted": accepted,
        })
        return accepted
    return keep


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    from training.recipes import async_rl_loop, sft_loop
    from training.utils import DeployConfig, TrainerConfig, WandBConfig

    if not os.environ.get("FIREWORKS_API_KEY"):
        raise ValueError("a working FIREWORKS_API_KEY is required; credentials are never saved in run files")
    if not args.budget_receipt:
        raise ValueError("a fresh live budget receipt is required")
    receipt = json.loads(args.budget_receipt.read_text())
    budget_check(receipt, plan)
    root = args.results_root / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (root / "budget.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler = logging.FileHandler(root / "training.log")
    logging.getLogger().addHandler(handler)
    previous_alarm = signal.getsignal(signal.SIGALRM)

    def timed_out(_signum: int, _frame: Any) -> None:
        # Do not let an SDK's generic `except Exception` swallow the deadline.
        raise SystemExit("v3 paid-run wall-clock limit reached")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(args.timeout_seconds)
    try:
        trainer = TrainerConfig(
            training_shape_id=TRAINING_SHAPE, inactivity_timeout="600s",
            timeout_s=300, pending_timeout_s=300,
        )
        source = plan.get("source_checkpoint")
        sft_result = None
        if source is None:
            sft_rows = (args.data_dir / "train_sft.jsonl").read_text().splitlines()[:plan["sft"]["rows"]]
            sft_data = root / "sft-data.jsonl"
            sft_data.write_text("\n".join(sft_rows) + "\n")
            sft = sft_loop.Config(
                log_path=str(root / "sft"), base_model=BASE_MODEL,
                dataset=str(sft_data), tokenizer_model=TOKENIZER_MODEL,
                renderer_name="qwen3_disable_thinking", train_on_what="all_assistant_messages",
                learning_rate=1e-4, epochs=plan["sft"]["epochs"], batch_size=32, max_seq_len=8192,
                max_examples=plan["sft"]["rows"], lora_rank=8, seed=0, render_workers=1, pipeline_depth=1,
                output_model_id=f"{args.run_id}-sft", trainer=trainer,
                wandb=WandBConfig(project=None),
            )
            sft_result = sft_loop.main(sft)
            # Archived jobs retain their checkpoints. A fresh RL trainer avoids
            # reusing a LoRA session handle invalidated by another SDK session.
            source = f"cross_job://{sft_result['job_id']}/step-{sft_result['steps']}"
            source_receipt = {
                "source_checkpoint": source, "sft_result": sft_result,
                "data_manifest_sha256": plan["data_manifest_sha256"], "is_smoke": plan["is_smoke"],
            }
            (root / "sft-source.json").write_text(json.dumps(source_receipt, indent=2) + "\n")
        rft = plan["rft"]
        cfg = async_rl_loop.Config(
            log_path=str(root / "rft"), base_model=BASE_MODEL, learning_rate=rft["learning_rate"],
            init_from_checkpoint=source, kl_beta=0.0, completions_per_prompt=4,
            min_group_size=4, max_completion_tokens=512, temperature=0.8, epochs=1,
            max_rows=args.max_rows, max_seq_len=8192, lora_rank=8, seed=0,
            prompt_groups_per_step=8, max_concurrency_rollout_sample=rft["rollout_concurrency"],
            max_head_offpolicy_versions=0, pipeline_chunks_per_step=1,
            trainer=trainer, deployment=DeployConfig(
                tokenizer_model=TOKENIZER_MODEL, replica_count=1, deployment_timeout_s=180,
                reattach_settle_timeout_s=60, sample_timeout=60,
            ), weight_sync_timeout=90,
            output_model_id=f"{args.run_id}-rft", wandb=WandBConfig(project=None), cleanup_on_exit=True,
        )
        rft_result = async_rl_loop.main(
            cfg, rows=load_rows(args.data_dir, args.manifest)[:args.max_rows],
            rollout_fn_factory=make_rollout_fn,
            dynamic_filter_fn=group_filter(root, rft["filter_constant"]),
            rollout_extras={
                "run_root": str(root), "sampling_rounds": rft["sampling_rounds"],
                "sampling_concurrency": rft["rollout_concurrency"],
            },
        )
        result = {"sft": sft_result, "rft": rft_result, "evaluated": False}
        (root / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        logging.getLogger().removeHandler(handler)
        handler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", choices=ARMS, default="warmstart")
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated/v3"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/experiment-v3-data.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results/training-v3"))
    parser.add_argument("--max-rows", type=int, default=240)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--rollout-concurrency", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--budget-receipt", type=Path)
    parser.add_argument("--source-receipt", type=Path, help="Verified retained SFT checkpoint provenance.")
    parser.add_argument("--execute", action="store_true", help="Provision paid compute after live preflight.")
    args = parser.parse_args()
    plan = build_plan(args)
    print(json.dumps(execute(args, plan) if args.execute else plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

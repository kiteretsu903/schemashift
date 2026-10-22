"""Bounded stopping-prompt regression evaluation of newly trained v5 checkpoints."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from schemashift.fireworks import FireworksToolClient
from schemashift.independent_v5 import validate_protocol
from schemashift.stopping_v5 import run_ab as run_evaluation_v5
from schemashift.training_v3 import budget_check

ACCOUNT = "example"
SHAPE = "accounts/fireworks/deploymentShapes/rft-qwen3-4b/versions/wjh1jb5o"


def firectl(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["firectl", "--account-id", ACCOUNT, *args],
        env={**os.environ, "FIRECTL_NO_UPDATE_NOTICE": "1"},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,50}", args.run_id):
        raise ValueError("invalid unique run/deployment ID")
    if not args.model.startswith(f"accounts/{ACCOUNT}/models/schemashift-v5-train-001-"):
        raise ValueError("requires an exported model from the repaired v5 training pair")
    if not 300 <= args.timeout_seconds <= 1800:
        raise ValueError("evaluation deadline must be 300..1800 seconds")
    _protocol, tasks, profiles = validate_protocol(args.protocol, args.model)
    if _protocol["model_endpoints"][args.model] != f"accounts/{ACCOUNT}/deployments/{args.run_id}":
        raise ValueError("deployment ID differs from frozen model binding")
    budget = json.loads(args.budget_receipt.read_text())
    budget_check(budget, {"run_id": args.run_id, "timeout_seconds": args.timeout_seconds})
    if budget["hourly_cost_bound_usd"] < 13:
        raise ValueError("budget must cover the evaluation B200")
    key = os.environ["FIREWORKS_API_KEY"]
    root = Path("results/stopping-v5") / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    # Never take ownership of an existing deployment, including one from another run.
    previous = firectl("deployment", "get", args.run_id, "-o", "json")
    if previous.returncode == 0 or "NotFound" not in previous.stderr:
        raise RuntimeError("cannot establish that this deployment ID is unused")
    config = {
        "run_id": args.run_id,
        "model": args.model,
        "label": args.label,
        "tasks_per_arm": 96,
        "arms": ["baseline", "stop"],
        "temperature": 0,
        "max_tokens": 512,
        "concurrency": 8,
        "max_attempts": 10,
        "deployment_shape": SHAPE,
        "timeout_seconds": args.timeout_seconds,
        "readiness_probe_attempts": 120,
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "comparison_scope": ("observed regression scenarios; balanced within-case prompt order"),
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (root / "budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    old_handler = signal.getsignal(signal.SIGALRM)

    def deadline(_signum: int, _frame: Any) -> None:
        raise SystemExit("evaluation paid-run deadline reached")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.timeout_seconds)
    try:
        provision_started = time.monotonic()
        result = firectl(
            "deployment",
            "create",
            args.model,
            "--deployment-id",
            args.run_id,
            "--deployment-shape",
            SHAPE,
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
        (root / "deployment-create.txt").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("deployment creation/readiness failed; see deployment-create.txt")
        client = FireworksToolClient(
            api_key=key,
            base_url="https://api.fireworks.ai/inference/v1",
            model=f"accounts/{ACCOUNT}/deployments/{args.run_id}",
        )

        async def evaluate() -> dict[str, Any]:
            try:
                # Separate cold readiness from the measured independent episodes.
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
                # A fresh deployment can serve on one gateway while other
                # gateways still cache a 404. One successful probe is insufficient.
                await asyncio.sleep(60)
                stable_batches = 0
                for _ in range(24):
                    replies = await asyncio.gather(
                        *(
                            probe.chat.completions.create(
                                model=client.model,
                                messages=[{"role": "user", "content": "Reply ready."}],
                                temperature=0,
                                reasoning_effort="none",
                                max_tokens=8,
                            )
                            for _ in range(8)
                        ),
                        return_exceptions=True,
                    )
                    failures = [type(reply).__name__ for reply in replies if isinstance(reply, BaseException)]
                    errors.extend(failures)
                    stable_batches = 0 if failures else stable_batches + 1
                    if stable_batches == 3:
                        break
                    await asyncio.sleep(5)
                if stable_batches != 3:
                    raise RuntimeError("evaluation routing did not become consistently ready")
                (root / "warmup.json").write_text(
                    json.dumps(
                        {
                            "seconds": time.monotonic() - started,
                            "transient_errors": errors,
                            "provisioning_and_warmup_seconds": time.monotonic() - provision_started,
                            "consecutive_successful_probe_batches": stable_batches,
                            "excluded_from_comparison_metrics": True,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                return await run_evaluation_v5(
                    tasks,
                    client,
                    system_name=args.label,
                    output_path=root / "episodes.jsonl",
                    concurrency=8,
                    max_attempts=10,
                    max_tokens=512,
                    profiles=profiles,
                )
            finally:
                await client.client.close()

        return asyncio.run(evaluate())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        # This uniquely owned deployment has just served this completed/cancelled
        # run; the CLI's one-hour recent-use check otherwise prevents cleanup.
        cleanup = firectl(
            "deployment",
            "delete",
            args.run_id,
            "--ignore-checks",
            "-o",
            "json",
            timeout=60,
        )
        (root / "deployment-cleanup.txt").write_text(cleanup.stdout + cleanup.stderr)
        if cleanup.returncode and "NotFound" not in cleanup.stderr:
            raise RuntimeError(f"cleanup failed: verify deployment {args.run_id} immediately")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--budget-receipt", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Create paid evaluation compute.")
    args = parser.parse_args()
    result = (
        execute(args)
        if args.execute
        else {
            "run_id": args.run_id,
            "model": args.model,
            "tasks_per_arm": 96,
            "arms": ["baseline", "stop"],
            "timeout_seconds": args.timeout_seconds,
            "execute": False,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""V5 dry-run planning and pinned, instance-local GRPO training controls.

No training runs on import or during the default CLI command. The adapter clones
only the pinned recipe entry function's globals; installed modules are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import signal
from dataclasses import replace
from pathlib import Path
from types import FunctionType
from typing import Any

from schemashift.models import Partner, Split, TaskSpec
from schemashift.training_v2 import BASE_MODEL, COOKBOOK_COMMIT, TOKENIZER_MODEL, TRAINING_SHAPE

VERSION = "schemashift-training-v5-1"
# Populated from the installed pinned recipe and its control seams, not network.
PINNED_SOURCE_HASHES: dict[str, str] = {
    "training.recipes.async_rl_loop": "d6ec895959468e962877c2cbc2267c49080734ba12d5062afa2f6b25eb3555fd",
    "training.utils.checkpoints": "3f729b0f4da69b1f0491febab64227bae87e9cd7beb5d6613bdced6b99897864",
    "training.utils.client": "444b202836dcecf5bd59d98b64b781efcc5c7e874d97bfa6744d7c31fdd8cf71",
    "training.utils.rl.async_rl.coordinator": "d8e934b8675f0eb9d9de1c136ada82c8393f735d5c7bd90ff5c17eb282f44e98",  # noqa: E501
    "training.utils.rl.async_rl.producer": "a52c22b9c193018581706947349347b44717264d3c7501d7ca6995c717465551",
    "training.utils.rl.async_rl.batch": "eb720d24ed17a9dcf367ce1c9e6be2fa6a7082c5e7cb8184e0ef06a13eab0459",
    "training.utils.rl.rollout.group_assembler": "57dafd8e89b3375d27eb2f42327ea56c5609458e67bf273a58f9771ae6ede327",  # noqa: E501
    "training.utils.rl.losses": "c4bb5aae509c5eed90e6e8dcba1635fcf38b970596088744f9b957e73b0354b2",
    "training.utils.data": "89cb09d7458e0ee8792fb54e8638bc91ba7bac05d48ad9c8165e214ae4de8625",
    "fireworks.training.sdk.client": "4ea6e2e03c4f4445ac16ddbe552af8838337c25cd171d46bd11fa65ed6e31e93",
    "tinker.lib.public_interfaces.training_client": "55704aae699807a59b408e556e830494a563955dea83ff1f4d7f7c7efe314a16",  # noqa: E501
}


def append_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("empty training data")
    seen = set()
    for row in rows:
        task = TaskSpec.from_dict(row)
        if row.get("id") != task.task_id:
            raise ValueError("training boundary requires id=task_id")
        if task.split != Split.TRAIN or task.partner not in (Partner.ATLAS, Partner.NIMBUS):
            raise ValueError("requires known-partner training rows; evaluation data forbidden")
        if task.task_id in seen:
            raise ValueError("duplicate task_id")
        seen.add(task.task_id)
    return rows


def load_rows(data_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != VERSION:
        raise ValueError("requires the v5 training manifest")
    for name in ("train_tasks.jsonl", "train_sft.jsonl"):
        path = data_dir / name
        entry = manifest["files"].get(name) or manifest["files"].get(str(path))
        if not entry or entry.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"dataset hash mismatch: {name}")
        lines = path.read_text().splitlines()
        if entry.get("rows") != len(lines):
            raise ValueError(f"dataset row count mismatch: {name}")
    rows = validate_rows(
        [json.loads(line) for line in (data_dir / "train_tasks.jsonl").read_text().splitlines()]
    )
    sft = [json.loads(line) for line in (data_dir / "train_sft.jsonl").read_text().splitlines()]
    if len(sft) != len(rows):
        raise ValueError("SFT/RFT training row counts differ")
    for task, demonstration in zip(rows, sft, strict=True):
        meta = demonstration.get("metadata", {})
        if (
            demonstration.get("id") != task["id"]
            or meta.get("task_id") != task["task_id"]
            or meta.get("split") != "train"
        ):
            raise ValueError("SFT task identity or training split mismatch")
    return rows


def signal_present(group: Any) -> bool:
    rewards, advantages = list(group.rewards), list(group.advantages)
    return (
        len(rewards) >= 2
        and len(advantages) > 0
        and all(math.isfinite(float(x)) for x in rewards + advantages)
        and max(rewards) - min(rewards) > 1e-8
        and any(abs(float(x)) > 1e-8 for x in advantages)
    )


def group_record(group: Any) -> dict[str, Any]:
    meta = group.row_meta or {}
    audit = meta.get("audit", {})
    task_id = meta.get("row_id")
    if not task_id:
        raise ValueError("missing task_id at optimizer boundary")
    return {
        "task_id": task_id,
        **audit,
        "rewards": list(group.rewards),
        "advantages": list(group.advantages),
    }


def group_filter(root: Path) -> Any:
    def keep(group: Any) -> bool:
        record = group_record(group)
        accepted = signal_present(group)
        append_json(
            root / "optimizer-groups.jsonl",
            {
                **record,
                "admission": "signal_pass" if accepted else "zero_or_invalid_signal_rejected",
                "accepted": accepted,
            },
        )
        return accepted

    return keep


def make_checkpoint_class(base: type, root: Path, *, restore_optimizer_diagnostic: bool = False) -> type:
    class FreshOptimizerCheckpoints(base):
        def resume(self, *, init_from_checkpoint=None, warm_start_from_adapter=None):
            if warm_start_from_adapter is not None or not init_from_checkpoint:
                raise ValueError("v5 requires an explicit SFT DCP; automatic resume is disabled")
            match = re.fullmatch(r"cross_job://([a-zA-Z0-9-]+)/([a-zA-Z0-9-]+)", init_from_checkpoint)
            if not match:
                raise ValueError("v5 source must be a cross_job checkpoint")
            job_id, checkpoint = match.groups()
            path = self._client.resolve_checkpoint_path(checkpoint, source_job_id=job_id)
            if restore_optimizer_diagnostic:
                self._client.load_state_with_optimizer(path)
            else:
                self._client.load_state(path)
            append_json(
                root / "initialization.jsonl",
                {
                    "checkpoint": init_from_checkpoint,
                    "resolved_path": path,
                    "load_method": "load_state_with_optimizer"
                    if restore_optimizer_diagnostic
                    else "load_state",
                    "optimizer_state": "restored_diagnostic_only"
                    if restore_optimizer_diagnostic
                    else "reset_by_sdk",
                    "recipe_step": 0,
                    "cursor_index": 0,
                },
            )
            # Cross-phase initialization must reset recipe cursor and step too.
            return None

    return FreshOptimizerCheckpoints


def make_coordinator_class(base: type, root: Path, state: dict[str, Any]) -> type:
    class GuardedCoordinator(base):
        def __init__(self, *, rows, **kwargs):
            self._trained_batches: set[int] = set()
            self._finished_batches: set[int] = set()

            def traced_rows():
                for request in rows:
                    meta = dict(request.row_meta or {})
                    meta["audit"] = {"cursor_index": request.row_id, "draw_ids": {}}

                    async def traced_draw(sub_index, request=request, meta=meta):
                        meta["audit"]["policy_version"] = self.published_version
                        run = await request.run_factory(sub_index)
                        if run is not None:
                            detail = run.metadata or {}
                            meta["audit"]["draw_ids"][str(sub_index)] = detail.get("draw_id")
                        return run

                    yield replace(request, run_factory=traced_draw, row_meta=meta)

            # Even callers bypassing main's filter cannot disable the guard.
            supplied_filter = kwargs.pop("dynamic_filter_fn", None)
            mandatory_filter = group_filter(root)

            def required_filter(group):
                accepted = mandatory_filter(group)
                return accepted and (supplied_filter(group) if supplied_filter else True)

            super().__init__(rows=traced_rows(), dynamic_filter_fn=required_filter, **kwargs)
            state["coordinator"] = self

        async def run_blocking(self, operation, function, *args, optimizer_batch=None, **kwargs):
            batch_id = optimizer_batch.batch_id if optimizer_batch is not None else None
            if operation == "train_chunk":
                chunk = args[0]
                if not chunk.groups or any(not signal_present(group) for group in chunk.groups):
                    raise ValueError("refusing forward/backward without nonzero finite group advantages")
                if batch_id != chunk.batch_id or batch_id in self._finished_batches:
                    raise ValueError("invalid or previously finished optimizer batch")
                for group, version, cursor in zip(
                    chunk.groups, chunk.submit_versions, chunk.source_tokens, strict=True
                ):
                    append_json(
                        root / "optimizer-batches.jsonl",
                        {
                            **group_record(group),
                            "cursor_index": cursor,
                            "policy_version": version,
                            "optimizer_batch_step": batch_id,
                            "chunk_index": chunk.index,
                            "admission": "forward_backward_started",
                        },
                    )
                result = await super().run_blocking(
                    operation, function, *args, optimizer_batch=optimizer_batch, **kwargs
                )
                self._trained_batches.add(batch_id)
                return result
            if operation == "optimizer":
                if (
                    batch_id is None
                    or batch_id not in self._trained_batches
                    or batch_id in self._finished_batches
                ):
                    raise ValueError("refusing Adam update without a successful nonzero-advantage batch")
                groups = optimizer_batch.prompt_groups
                if not groups or any(not signal_present(group) for group in groups):
                    raise ValueError("refusing zero or invalid advantage Adam batch")
                # Mark before remote mutation: never retry an uncertain optimizer call.
                self._finished_batches.add(batch_id)
                append_json(
                    root / "optimizer-batches.jsonl",
                    {
                        "optimizer_batch_step": batch_id,
                        "admission": "optimizer_call_started",
                        "task_ids": [group_record(group)["task_id"] for group in groups],
                    },
                )
                result = await super().run_blocking(
                    operation, function, *args, optimizer_batch=optimizer_batch, **kwargs
                )
                append_json(
                    root / "optimizer-batches.jsonl",
                    {
                        "optimizer_batch_step": batch_id,
                        "admission": "optimizer_call_completed",
                    },
                )
                return result
            return await super().run_blocking(
                operation, function, *args, optimizer_batch=optimizer_batch, **kwargs
            )

    return GuardedCoordinator


def verify_sdk() -> dict[str, Any]:
    from importlib.metadata import version

    versions = {name: version(name) for name in ("fireworks-ai", "tinker", "tinker-cookbook")}
    if versions != {"fireworks-ai": "1.2.6", "tinker": "0.23.0", "tinker-cookbook": "0.4.3"}:
        raise ValueError(f"unpinned training SDK versions: {versions}")
    sources = {}
    for name, digest in PINNED_SOURCE_HASHES.items():
        module = importlib.import_module(name)
        source = Path(module.__file__)
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"SDK control seam drifted: {name}; review adapter before execution")
        sources[name] = {"sha256": actual, "path": str(source)}
    if not sources:
        raise ValueError("no pinned SDK source hashes")
    return {"cookbook_commit": COOKBOOK_COMMIT, "versions": versions, "sources": sources}


def isolated_recipe(root: Path, state: dict[str, Any]) -> Any:
    """Bind audited subclasses in a new globals dictionary; no global patching."""
    from training.recipes import async_rl_loop

    provenance = verify_sdk()
    (root / "sdk-provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    scope = dict(async_rl_loop.main.__globals__)
    scope["TrainingCheckpoints"] = make_checkpoint_class(scope["TrainingCheckpoints"], root)
    scope["AsyncRLCoordinator"] = make_coordinator_class(scope["AsyncRLCoordinator"], root, state)
    entry = FunctionType(
        async_rl_loop.main.__code__, scope, "v5_async_rl_main", async_rl_loop.main.__defaults__
    )
    entry.__kwdefaults__ = dict(async_rl_loop.main.__kwdefaults__ or {})
    return entry


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,50}", args.run_id):
        raise ValueError("run_id must be a short lowercase name")
    rows = load_rows(args.data_dir, args.manifest)
    if not 1 <= args.max_rows <= len(rows):
        raise ValueError("max_rows exceeds training dataset")
    if not 60 <= args.timeout_seconds <= 1800:
        raise ValueError("timeout_seconds must be 60..1800")
    plan = {
        "version": VERSION,
        "run_id": args.run_id,
        "base_model": BASE_MODEL,
        "data_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "timeout_seconds": args.timeout_seconds,
        "sft": {"rows": args.max_rows, "epochs": 2, "learning_rate": 1e-4},
        "rft": {
            "rows": args.max_rows,
            "epochs": 1,
            "learning_rate": 1e-5,
            "group_size": 4,
            "sampling_rounds": 3,
            "rollout_concurrency": 8,
            "max_rollout_attempts": args.max_rows * 12,
            "kl_beta": 0.0,
            "top_k": 0,
            "filter_constant": True,
            "optimizer_initialization": "load_state: SFT weights, fresh Adam",
        },
        "training_source": "new v5 safe-demonstration SFT; same weights before each experimental arm",
        "controls": {
            "task_id": "id=task_id",
            "zero_advantage": "drop whole group; block empty optimizer batch",
            "optimizer_state": "fresh; restored state requires a separate diagnostic run",
        },
        "remote_verified": False,
        "executed": False,
        "limitations": [
            "Local SDK seam validation does not prove remote optimizer-state reset.",
            "No training authorization or resource provisioning is implied by this plan.",
            "Independent test data never enters training, selection, or this planner.",
        ],
    }
    if args.source_receipt:
        source = json.loads(args.source_receipt.read_text())
        checkpoint = source.get("source_checkpoint", "")
        if not re.fullmatch(r"cross_job://[a-zA-Z0-9-]+/[a-zA-Z0-9-]+", checkpoint):
            raise ValueError("source receipt requires cross_job checkpoint")
        plan["source_checkpoint"] = checkpoint
        plan["training_source"] = "retained SFT weights; no claim that its data matches v5"
        plan["source_receipt_sha256"] = hashlib.sha256(args.source_receipt.read_bytes()).hexdigest()
        plan["source_data_manifest_sha256"] = source.get("data_manifest_sha256")
        plan["sft"]["skipped"] = True
    return plan


def validate_budget(receipt: dict[str, Any], plan: dict[str, Any]) -> None:
    from schemashift.training_v3 import budget_check

    budget_check(receipt, plan)
    if float(receipt["hourly_cost_bound_usd"]) < 26:
        raise ValueError("hourly_cost_bound_usd must cover trainer plus sampler: at least $26/hour")


def archive_training_inputs(
    root: Path,
    args: argparse.Namespace,
    sdk_provenance: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Preserve actual source/config/data bytes before any paid resource launch.

    Only source, tests, explicit run inputs, and known configuration are copied;
    credential files and independent evaluation datasets are not read.
    """
    from importlib.metadata import distributions

    project = project_root or Path(__file__).resolve().parents[2]
    archive = root / "input-archive"
    archive.mkdir(parents=True, exist_ok=False)
    entries: dict[str, Any] = {}

    def preserve(source: Path, relative: Path, expected_hash: str | None = None) -> None:
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_hash is not None and digest != expected_hash:
            raise ValueError(f"source changed before archival: {source}")
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        entries[str(relative)] = {"sha256": digest, "bytes": len(payload)}

    for directory in ("src", "tests"):
        for source in sorted((project / directory).rglob("*.py")):
            preserve(source, Path("project") / source.relative_to(project))
    for source in [project / "pyproject.toml", *sorted(project.glob("requirements*.txt"))]:
        if source.is_file():
            preserve(source, Path("project") / source.name)
    preserve(root / "config.json", Path("run/config.json"))
    preserve(root / "budget.json", Path("run/budget.json"))
    preserve(args.manifest, Path("run/data-manifest.json"))
    for name in ("train_tasks.jsonl", "train_sft.jsonl"):
        preserve(args.data_dir / name, Path("run/data") / name)
    if args.source_receipt:
        preserve(args.source_receipt, Path("run/source-receipt.json"))
    for name, entry in sdk_provenance["sources"].items():
        preserve(Path(entry["path"]), Path("sdk") / (name.replace(".", "/") + ".py"), entry["sha256"])
    dependency_versions = sorted(
        [{"name": dist.metadata["Name"], "version": dist.version} for dist in distributions()],
        key=lambda item: (str(item["name"]).lower(), item["version"]),
    )
    (archive / "dependency-versions.json").write_text(json.dumps(dependency_versions, indent=2) + "\n")
    dependencies = archive / "dependency-versions.json"
    entries[dependencies.name] = {
        "sha256": hashlib.sha256(dependencies.read_bytes()).hexdigest(),
        "bytes": dependencies.stat().st_size,
    }
    manifest = {
        "version": "schemashift-v5-training-input-archive-1",
        "files": entries,
        "sdk": sdk_provenance,
        "remote_resources_started": False,
    }
    (root / "input-archive-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    """Explicit future paid execution, guarded by a fresh bounded budget receipt."""
    from training.recipes import async_rl_loop, sft_loop
    from training.utils import DeployConfig, TrainerConfig, WandBConfig

    from schemashift.rollouts_v5 import make_rollout_fn

    if not os.environ.get("FIREWORKS_API_KEY") or not args.budget_receipt:
        raise ValueError("explicit execution needs credentials and a fresh live budget receipt")
    receipt = json.loads(args.budget_receipt.read_text())
    validate_budget(receipt, plan)
    sdk_provenance = verify_sdk()
    rows = load_rows(args.data_dir, args.manifest)[: args.max_rows]
    root = args.results_root / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (root / "budget.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    archive_training_inputs(root, args, sdk_provenance)
    state: dict[str, Any] = {}
    entry = isolated_recipe(root, state)
    trainer = TrainerConfig(
        training_shape_id=TRAINING_SHAPE, inactivity_timeout="600s", timeout_s=300, pending_timeout_s=300
    )
    old_alarm = signal.getsignal(signal.SIGALRM)

    def timeout(_signum, _frame):
        raise SystemExit("v5 paid-run deadline reached")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    try:
        source = plan.get("source_checkpoint")
        sft_result = None
        if source is None:
            sft_path = root / "sft-data.jsonl"
            sft_path.write_text(
                "\n".join((args.data_dir / "train_sft.jsonl").read_text().splitlines()[: args.max_rows])
                + "\n"
            )
            sft_result = sft_loop.main(
                sft_loop.Config(
                    log_path=str(root / "sft"),
                    base_model=BASE_MODEL,
                    dataset=str(sft_path),
                    tokenizer_model=TOKENIZER_MODEL,
                    renderer_name="qwen3_disable_thinking",
                    train_on_what="all_assistant_messages",
                    learning_rate=1e-4,
                    epochs=2,
                    batch_size=32,
                    max_seq_len=8192,
                    max_examples=args.max_rows,
                    lora_rank=8,
                    seed=0,
                    render_workers=1,
                    pipeline_depth=1,
                    output_model_id=f"{args.run_id}-sft",
                    trainer=trainer,
                    wandb=WandBConfig(project=None),
                )
            )
            source = f"cross_job://{sft_result['job_id']}/step-{sft_result['steps']}"
            (root / "sft-source.json").write_text(
                json.dumps(
                    {
                        "source_checkpoint": source,
                        "sft_result": sft_result,
                        "data_manifest_sha256": plan["data_manifest_sha256"],
                    },
                    indent=2,
                )
                + "\n"
            )
        cfg = async_rl_loop.Config(
            log_path=str(root / "rft"),
            base_model=BASE_MODEL,
            learning_rate=1e-5,
            init_from_checkpoint=source,
            kl_beta=0.0,
            completions_per_prompt=4,
            min_group_size=4,
            max_completion_tokens=512,
            temperature=0.8,
            epochs=1,
            max_rows=args.max_rows,
            max_seq_len=8192,
            lora_rank=8,
            seed=0,
            prompt_groups_per_step=8,
            max_concurrency_rollout_sample=8,
            max_head_offpolicy_versions=0,
            pipeline_chunks_per_step=1,
            trainer=trainer,
            deployment=DeployConfig(
                tokenizer_model=TOKENIZER_MODEL,
                replica_count=1,
                deployment_timeout_s=180,
                reattach_settle_timeout_s=60,
                sample_timeout=60,
            ),
            weight_sync_timeout=90,
            output_model_id=f"{args.run_id}-rft",
            wandb=WandBConfig(project=None),
            cleanup_on_exit=True,
        )
        rft_result = entry(
            cfg,
            rows=rows,
            rollout_fn_factory=make_rollout_fn,
            rollout_extras={
                "run_root": str(root),
                "sampling_rounds": 3,
                "sampling_concurrency": 8,
                "policy_version_getter": lambda: state["coordinator"].published_version,
            },
        )
        result = {"sft": sft_result, "rft": rft_result, "evaluated": False, "executed": True}
        (root / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated/v5"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/experiment-v5-training-data.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results/training-v5"))
    parser.add_argument("--max-rows", type=int, default=288)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--budget-receipt", type=Path)
    parser.add_argument("--verify-sdk", action="store_true", help="Inspect installed pinned SDK locally.")
    parser.add_argument(
        "--execute", action="store_true", help="Explicitly launch paid training after preflight."
    )
    args = parser.parse_args()
    plan = build_plan(args)
    if args.verify_sdk:
        plan["sdk_provenance"] = verify_sdk()
    print(json.dumps(execute(args, plan) if args.execute else plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

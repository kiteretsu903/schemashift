"""One bounded supervised repair of the retained RFT weights; no sampler or GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
from pathlib import Path
from types import FunctionType

from schemashift.dataset_targeted_v6 import VERSION, prefix_rows
from schemashift.models import Split, TaskSpec
from schemashift.training_v3 import budget_check
from schemashift.training_v5 import archive_training_inputs, make_checkpoint_class, verify_sdk

PROTOCOL = Path("manifests/experiment-v6-targeted-training-protocol.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(data_dir, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest["version"] != VERSION:
        raise ValueError("wrong training manifest")
    for name in ("train_tasks.jsonl", "train_sft.jsonl"):
        path = data_dir / name
        if digest(path) != manifest["files"][str(path)]["sha256"]:
            raise ValueError("training data drift")
    records = [json.loads(line) for line in (data_dir / "train_tasks.jsonl").read_text().splitlines()]
    demonstrations = [json.loads(line) for line in (data_dir / "train_sft.jsonl").read_text().splitlines()]
    if len({r["task"]["task_id"] for r in records}) != len(records):
        raise ValueError("duplicate task identity")
    expected = []
    for record in records:
        task = TaskSpec.from_dict(record["task"])
        if task.split is not Split.TRAIN:
            raise ValueError("evaluation leakage into training")
        expected.extend(prefix_rows(task, record["profile"], record["metadata"]))
    if demonstrations != expected or len({r["id"] for r in demonstrations}) != len(demonstrations):
        raise ValueError("training targets differ from audited expert next calls")
    return demonstrations


def validate_protocol(protocol_path=PROTOCOL):
    protocol = json.loads(protocol_path.read_text())
    for filename, expected in protocol["source_sha256"].items():
        if digest(filename) != expected:
            raise ValueError("frozen training source changed: " + filename)
    for filename, expected in protocol["sdk_sha256"].items():
        if digest(filename) != expected:
            raise ValueError("pinned SFT SDK source changed: " + filename)
    if digest(protocol["data_manifest"]) != protocol["data_manifest_sha256"]:
        raise ValueError("data manifest changed")
    rows = load_data(Path(protocol["data_dir"]), Path(protocol["data_manifest"]))
    if len(rows) != protocol["sft"]["rows"]:
        raise ValueError("row count drift")
    return protocol, rows


def isolated_sft_entry(module, root):
    scope = dict(module.main.__globals__)
    scope["TrainingCheckpoints"] = make_checkpoint_class(scope["TrainingCheckpoints"], root)
    entry = FunctionType(module.main.__code__, scope, "v6_sft_main", module.main.__defaults__)
    entry.__kwdefaults__ = dict(module.main.__kwdefaults__ or {})
    return entry


def execute(args):
    from training.recipes import sft_loop
    from training.utils import TrainerConfig, WandBConfig

    protocol, rows = validate_protocol(args.protocol)
    if not os.environ.get("FIREWORKS_API_KEY"):
        raise ValueError("missing temporary credential")
    budget = json.loads(args.budget_receipt.read_text())
    budget_check(budget, {"run_id": protocol["run_id"], "timeout_seconds": protocol["timeout_seconds"]})
    # This entry provisions only one dedicated SFT trainer, never a sampling deployment.
    if budget["hourly_cost_bound_usd"] < 13:
        raise ValueError("budget must cover one B200 trainer")
    provenance = verify_sdk()
    for name in (
        "training.recipes.sft_loop",
        "training.utils.supervised",
        "tinker_cookbook.renderers.base",
        "tinker_cookbook.renderers.qwen3",
        "tinker_cookbook.tokenizer_utils",
    ):
        import importlib

        module = importlib.import_module(name)
        path = Path(module.__file__)
        provenance["sources"][name] = {"path": str(path), "sha256": digest(path)}
    root = Path("results/training-v6") / protocol["run_id"]
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (root / "budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    archive_args = argparse.Namespace(
        data_dir=Path(protocol["data_dir"]),
        manifest=Path(protocol["data_manifest"]),
        source_receipt=Path(protocol["source_receipt"]),
    )
    archive_manifest = archive_training_inputs(root, archive_args, provenance)
    for filename, expected in protocol["tokenizer_sha256"].items():
        source = Path(filename)
        if digest(source) != expected:
            raise ValueError("tokenizer snapshot changed")
        relative = Path("tokenizer") / source.name
        destination = root / "input-archive" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        archive_manifest["files"][str(relative)] = {"sha256": expected, "bytes": source.stat().st_size}
    for filename in protocol["source_sha256"]:
        if filename.startswith("artifacts/v6-targeted/"):
            source = Path(filename)
            relative = Path("launcher") / source.name
            destination = root / "input-archive" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            archive_manifest["files"][str(relative)] = {
                "sha256": digest(source),
                "bytes": source.stat().st_size,
            }
    (root / "input-archive-manifest.json").write_text(
        json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n"
    )
    entry = isolated_sft_entry(sft_loop, root)
    settings = protocol["sft"]
    config = sft_loop.Config(
        log_path=str(root / "sft"),
        base_model=protocol["base_model"],
        dataset=str(Path(protocol["data_dir"]) / "train_sft.jsonl"),
        tokenizer_model=protocol["tokenizer_model"],
        tokenizer_revision=protocol["tokenizer_revision"],
        renderer_name="qwen3_disable_thinking",
        train_on_what="last_assistant_message",
        learning_rate=settings["learning_rate"],
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        max_seq_len=8192,
        max_examples=len(rows),
        lora_rank=8,
        seed=0,
        render_workers=1,
        pipeline_depth=1,
        init_from_checkpoint=protocol["source_checkpoint"],
        output_model_id=protocol["output_model_id"],
        trainer=TrainerConfig(
            training_shape_id=protocol["training_shape"],
            inactivity_timeout="600s",
            timeout_s=300,
            pending_timeout_s=300,
        ),
        wandb=WandBConfig(project=None),
    )
    old_handler = signal.getsignal(signal.SIGALRM)

    def deadline(_signum, _frame):
        raise SystemExit("targeted SFT paid deadline reached")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(protocol["timeout_seconds"])
    try:
        result = entry(config)
        expected_steps = math.ceil(len(rows) / settings["batch_size"]) * settings["epochs"]
        if result["steps"] != expected_steps:
            raise RuntimeError("training update count differs from frozen plan")
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--budget-receipt", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(args) if args.execute else validate_protocol(args.protocol)[0], indent=2))

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from schemashift.catalog import tool_schemas
from schemashift.generator import GENERATOR_VERSION
from schemashift.prompts import BASELINE_SYSTEM_PROMPT, OPTIMIZED_SYSTEM_PROMPT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def verify_evaluation_freeze(path: Path) -> list[str]:
    if not path.is_file():
        return [f"freeze manifest is missing: {path}"]
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []

    expected_values = {
        "generator.version": (
            data.get("generator", {}).get("version"),
            GENERATOR_VERSION,
        ),
        "prompts.baseline_sha256": (
            data.get("prompts", {}).get("baseline_sha256"),
            _text_hash(BASELINE_SYSTEM_PROMPT),
        ),
        "prompts.optimized_sha256": (
            data.get("prompts", {}).get("optimized_sha256"),
            _text_hash(OPTIMIZED_SYSTEM_PROMPT),
        ),
        "tool_schema_sha256": (
            data.get("tool_schema_sha256"),
            _text_hash(json.dumps(tool_schemas(), sort_keys=True, separators=(",", ":"))),
        ),
    }
    for label, (recorded, current) in expected_values.items():
        if recorded != current:
            problems.append(f"{label} mismatch: recorded={recorded}, current={current}")

    files = {
        "rft_sha256": Path("data/generated/train_rft.jsonl"),
        "sft_sha256": Path("data/generated/train_sft.jsonl"),
    }
    for label, file_path in files.items():
        recorded = data.get("training_data", {}).get(label)
        if not file_path.is_file():
            problems.append(f"frozen input is missing: {file_path}")
        else:
            current = sha256_file(file_path)
            if recorded != current:
                problems.append(f"training_data.{label} mismatch: recorded={recorded}, current={current}")

    if data.get("generator", {}).get("private_test_generated_at_freeze") is not False:
        problems.append("freeze manifest must attest that private test was not generated at freeze time")
    v2_path = Path("manifests/experiment-v2-freeze.json")
    if v2_path.is_file():
        problems.extend(_verify_v2_freeze(v2_path, path, data))
    return problems


def _verify_v2_freeze(path: Path, parent_path: Path, parent: dict[str, Any]) -> list[str]:
    from schemashift.training_v2 import BASE_MODEL, TOKENIZER_MODEL, TRAINING_SHAPE

    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    expected_values = {
        "v2.base_model": (data.get("base_model"), BASE_MODEL),
        "v2.tokenizer_model": (data.get("tokenizer_model"), TOKENIZER_MODEL),
        "v2.rft.training_shape": (data.get("rft", {}).get("training_shape"), TRAINING_SHAPE),
        "v2.parent_freeze_sha256": (data.get("parent_freeze_sha256"), sha256_file(parent_path)),
        "v2.success_gates": (data.get("success_gates"), parent.get("success_gates")),
        "v2.tool_schema_sha256": (
            data.get("tool_schema_sha256"),
            _text_hash(json.dumps(tool_schemas(), sort_keys=True, separators=(",", ":"))),
        ),
        "v2.prompts.optimized_sha256": (
            data.get("prompts", {}).get("optimized_sha256"),
            _text_hash(OPTIMIZED_SYSTEM_PROMPT),
        ),
    }
    for label, (recorded, current) in expected_values.items():
        if recorded != current:
            problems.append(f"{label} mismatch: recorded={recorded}, current={current}")

    for file_name, recorded in data.get("immutable_files", {}).items():
        file_path = Path(file_name)
        if not file_path.is_file():
            problems.append(f"v2 frozen input is missing: {file_path}")
        elif sha256_file(file_path) != recorded:
            problems.append(f"v2 immutable file drifted: {file_path}")

    attestations = data.get("attestations", {})
    for key in ("no_qwen3_4b_result_seen", "no_private_inference", "no_bfcl_inference"):
        if attestations.get(key) is not True:
            problems.append(f"v2 attestation must be true: {key}")
    seal_path = Path("manifests/training-v2-adapter-seal.json")
    if seal_path.is_file():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        sealed_files = {
            "training_adapter_sha256": Path("src/schemashift/training_v2.py"),
            "requirements_training_v2_sha256": Path("requirements-training-v2.txt"),
        }
        for key, file_path in sealed_files.items():
            if seal.get(key) != sha256_file(file_path):
                problems.append(f"v2 adapter seal mismatch: {key}")
        if seal.get("cookbook_commit") != "152faf05ae6f039b74158d2a84e133e65ad96071":
            problems.append("v2 adapter seal cookbook commit mismatch")
    sft_path = Path("manifests/sft-v2-migration.json")
    if sft_path.is_file():
        sft = json.loads(sft_path.read_text(encoding="utf-8"))
        sealed_files = {
            "sft_adapter_sha256": Path("src/schemashift/sft_v2.py"),
            "requirements_training_v2_sha256": Path("requirements-training-v2.txt"),
        }
        for key, file_path in sealed_files.items():
            if sft.get(key) != sha256_file(file_path):
                problems.append(f"v2 SFT seal mismatch: {key}")
        if sft.get("dataset", {}).get("sha256") != sha256_file(
            Path("data/generated/train_sft.jsonl")
        ):
            problems.append("v2 SFT dataset seal mismatch")
        if sft.get("experiment_v2_freeze_sha256") != sha256_file(path):
            problems.append("v2 SFT parent freeze mismatch")
    bfcl_adapter_path = Path("manifests/bfcl-v2-adapter.json")
    if bfcl_adapter_path.is_file():
        bfcl_adapter = json.loads(bfcl_adapter_path.read_text(encoding="utf-8"))
        if bfcl_adapter.get("runner_sha256") != sha256_file(Path("src/schemashift/bfcl_runner.py")):
            problems.append("v2 BFCL runner seal mismatch")
        if bfcl_adapter.get("frozen_case_manifest_sha256") != sha256_file(
            Path("manifests/bfcl-freeze.json")
        ):
            problems.append("v2 BFCL case freeze mismatch")
    return problems

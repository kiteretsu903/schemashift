"""Recompute the published comparison without inference or provider credentials."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from schemashift.analysis_fresh_v5 import clustered_interval
from schemashift.classification import audit_suite, read_jsonl

MODELS = ("sft", "rft", "v16")
SUITES = ("original", "fresh")


def verify_files() -> dict:
    root = Path.cwd().resolve()
    manifest = json.loads(Path("evaluation/manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = root / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError("manifest path escapes the repository")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("published file is missing or changed: " + name)
    return {"verified_files": len(manifest["files_sha256"]), "passed": True}


def replay_comparison() -> dict:
    result = {"suites": {}, "episodes_replayed": 576, "network_requests": 0}
    expected = json.loads(Path("evaluation/expected_results.json").read_text())
    for suite in SUITES:
        models, records, subgroups = {}, {}, {}
        for name in MODELS:
            path = Path(f"evaluation/recorded_runs/{name}/{suite}.jsonl")
            audit = audit_suite(f"evaluation/protocols/{suite}.json", path, name)
            models[name] = {key: audit[key] for key in ("summary", "error_category_episodes")}
            records[name] = {row["task_id"]: row for row in read_jsonl(path)}
            subgroups[name] = {}
            for dimension in ("profile", "stratum"):
                subgroups[name][dimension] = {}
                for value in sorted({m[dimension] for m in audit["case_metadata"].values()}):
                    ids = [i for i, m in audit["case_metadata"].items() if m[dimension] == value]
                    subgroups[name][dimension][value] = {
                        "episodes": len(ids),
                        "successes": sum(records[name][i]["success"] for i in ids),
                    }
        pairs = {}
        for left, right in itertools.combinations(MODELS, 2):
            if records[left].keys() != records[right].keys():
                raise ValueError("paired task identities differ")
            for identity in records[left]:
                configs = [dict(records[name][identity]["run_config"]) for name in (left, right)]
                for config in configs:
                    config.pop("model")
                    config.pop("system_name")
                if configs[0] != configs[1]:
                    raise ValueError("paired inference settings differ")
            pairs[right + "_minus_" + left] = {
                "success_difference": clustered_interval(
                    {i: row["success"] for i, row in records[left].items()},
                    {i: row["success"] for i, row in records[right].items()},
                    audit["case_metadata"], seed=20261018,
                ),
                "right_only_successes": [i for i in records[left]
                                         if records[right][i]["success"] and not records[left][i]["success"]],
                "left_only_successes": [i for i in records[left]
                                        if records[left][i]["success"] and not records[right][i]["success"]],
            }
        if {"models": models, "pairs": pairs} != expected[suite]:
            raise ValueError("replayed comparison differs from the published result: " + suite)
        result["suites"][suite] = {"models": models, "pairs": pairs, "subgroups": subgroups}
    result["matches_published_results"] = True
    return result

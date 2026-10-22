"""Select a deterministic balanced 864-row v7 budget matched to the v6 repair."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from schemashift.dataset_diverse_v7 import STRATA, VERSION, prefix_rows
from schemashift.models import TaskSpec


def chosen_variants(stratum):
    # Entitlement twins must select the same contexts. Each of eight note styles is retained.
    group = "entitlement" if stratum in {"entitled_create", "absent_entitlement"} else stratum
    return {
        slot + 8 * (hashlib.sha256(f"v7-budget-0:{group}:{slot}".encode()).digest()[0] % 2)
        for slot in range(8)
    }


def prepare(
    output=Path("data/generated/v7-matched"), manifest=Path("manifests/experiment-v7-matched-data.json")
):
    if output.exists() or manifest.exists():
        raise FileExistsError("never overwrite a selected dataset")
    source = Path("data/generated/v7-diverse")
    source_manifest = Path("manifests/experiment-v7-diverse-data.json")
    prior = json.loads(source_manifest.read_text())
    for filename, entry in prior["files"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("candidate pool drift")
    records = [json.loads(line) for line in (source / "train_tasks.jsonl").read_text().splitlines()]
    records = [r for r in records if r["metadata"]["variant"] in chosen_variants(r["metadata"]["stratum"])]
    rows = [
        row
        for r in records
        for row in prefix_rows(TaskSpec.from_dict(r["task"]), r["profile"], r["metadata"])
    ]
    cells = Counter(f"{r['metadata']['stratum']}/{r['profile']}" for r in records)
    assert len(records) == 288 and len(rows) == 864 and set(cells.values()) == {8}
    assert {
        r["metadata"]["note_style"] for r in records if r["metadata"]["stratum"] == "resolve_with_comment"
    } == {"none", "period", "question", "exclamation", "quoted", "semicolon", "unicode", "multiline"}
    output.mkdir(parents=True)
    files = {}
    for name, values in [("train_tasks.jsonl", records), ("train_sft.jsonl", rows)]:
        path = output / name
        with path.open("x") as f:
            for r in values:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        files[str(path)] = {"rows": len(values), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {
        "version": VERSION,
        "selection_version": "v7-budget-matched-1",
        "seed": 0,
        "pool_manifest": str(source_manifest),
        "pool_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "selected_variants": {s: sorted(chosen_variants(s)) for s in STRATA},
        "logical_scenarios": 96,
        "trajectories": 288,
        "rows": 864,
        "files": files,
        "cells": dict(cells),
        "unique_notes": len(
            {
                r["task"]["expected"]["required_comment"]
                for r in records
                if r["task"]["expected"]["required_comment"]
            }
        ),
        "scope": (
            "One deterministic subset chosen without holdout requests or model outcomes. "
            "Equal864rows27updatesbatch32epoch1 to v6 control; token counts/composition differ."
        ),
    }
    with manifest.open("x") as f:
        f.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))

"""Load independently authored scenarios and pair them across API naming profiles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from schemashift.dataset_v5 import expert_calls_v5
from schemashift.environment_v5 import SchemaShiftEnvironmentV5
from schemashift.models import Partner, Split, TaskSpec
from schemashift.schema_v5 import profile_call

STRATA = {
    "basic_create",
    "entitled_create",
    "absent_entitlement",
    "resolve_with_comment",
    "authorized_contact",
    "missing_authorization",
    "forged_authorization",
    "missing_impact",
    "read_only",
    "unsupported_request",
    "distracting_data",
    "conflicting_request",
}
PROFILES = ("atlas", "cedar")


def read_cases(path: Path) -> tuple[list[TaskSpec], dict[str, str], dict[str, dict[str, str]]]:
    data = json.loads(path.read_text())
    scenarios = data["scenarios"]
    counts = Counter(row["stratum"] for row in scenarios)
    if len(scenarios) != 48 or set(counts) != STRATA or any(n != 4 for n in counts.values()):
        raise ValueError("requires the frozen balanced 48-scenario design")
    if len({row["scenario_id"] for row in scenarios}) != 48:
        raise ValueError("duplicate scenario identity")
    tasks = []
    profiles = {}
    metadata = {}
    for row in scenarios:
        source = TaskSpec.from_dict(row["task"])
        if source.partner is not Partner.ATLAS or source.split is not Split.PRIVATE_TEST:
            raise ValueError("independent scenarios use Atlas logical backend and private_test split")
        for profile in PROFILES:
            identity = row["scenario_id"] + "/" + profile
            task = replace(
                source,
                task_id="ss5i-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
                template_id=row["scenario_id"],
                user_request=f"Active service API: {profile.title()}.\n" + source.user_request,
            )
            env = SchemaShiftEnvironmentV5(task, profile=profile)
            for call in expert_calls_v5(task):
                env.step(profile_call(call, profile))
            if env.score != 1:
                raise ValueError("independent expert workflow failed: " + identity)
            tasks.append(task)
            profiles[task.task_id] = profile
            metadata[task.task_id] = {
                "scenario_id": row["scenario_id"],
                "stratum": row["stratum"],
                "profile": profile,
            }
    return tasks, profiles, metadata


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(path: Path, model: str) -> tuple[dict[str, Any], list[TaskSpec], dict[str, str]]:
    from schemashift.agent_v5 import SYSTEM_PROMPT_V5

    protocol = json.loads(path.read_text())
    if model not in protocol["models"].values():
        raise ValueError("model was not frozen before the test")
    for filename, digest in protocol["source_sha256"].items():
        if hash_file(Path(filename)) != digest:
            raise ValueError("frozen source changed: " + filename)
    data_path = Path(protocol["independent_scenarios"])
    if hash_file(data_path) != protocol["independent_scenarios_sha256"]:
        raise ValueError("frozen scenarios changed")
    if hashlib.sha256(SYSTEM_PROMPT_V5.encode()).hexdigest() != protocol["system_prompt_sha256"]:
        raise ValueError("frozen prompt changed")
    if protocol["inference"] != {"temperature": 0, "max_tokens": 512, "max_attempts": 10, "concurrency": 8}:
        raise ValueError("unsupported inference configuration")
    tasks, profiles, metadata = read_cases(data_path)
    if metadata != protocol["case_metadata"]:
        raise ValueError("task identity or case metadata changed")
    return protocol, tasks, profiles

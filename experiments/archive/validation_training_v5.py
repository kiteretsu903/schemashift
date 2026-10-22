"""Frozen local contract for development checks of the new v5 model pair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from schemashift.agent_v5 import SYSTEM_PROMPT_V5
from schemashift.dataset_v5 import expert_calls_v5
from schemashift.environment_v5 import SchemaShiftEnvironmentV5
from schemashift.models import Split, TaskSpec


def validate_protocol(path: Path, model: str) -> tuple[dict[str, Any], list[TaskSpec], dict[str, str]]:
    protocol = json.loads(path.read_text())
    if protocol.get("version") != "schemashift-v5-trained-development-protocol-1":
        raise ValueError("requires the repaired recipe development protocol")
    if model not in protocol["models"].values():
        raise ValueError("model is outside the frozen trained pair")
    for filename, expected in protocol["source_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen evaluation source changed: {filename}")
    if hashlib.sha256(SYSTEM_PROMPT_V5.encode()).hexdigest() != protocol["system_prompt_sha256"]:
        raise ValueError("frozen prompt changed")
    data = Path(protocol["tasks_path"])
    if hashlib.sha256(data.read_bytes()).hexdigest() != protocol["tasks_sha256"]:
        raise ValueError("frozen development tasks changed")
    tasks = [TaskSpec.from_dict(json.loads(line)) for line in data.read_text().splitlines()]
    if len(tasks) != 90 or len({t.task_id for t in tasks}) != 90:
        raise ValueError("expected 90 unique development tasks")
    if any(t.split != Split.DEVELOPMENT for t in tasks):
        raise ValueError("only the development split is allowed")
    if set(protocol["case_metadata"]) != {t.task_id for t in tasks}:
        raise ValueError("case metadata differs from development task IDs")
    for task in tasks:
        env = SchemaShiftEnvironmentV5(task)
        for call in expert_calls_v5(task):
            env.step(call)
        if env.score != 1:
            raise ValueError(f"expert no longer passes: {task.task_id}")
    return protocol, tasks, {t.task_id: t.partner.value for t in tasks}

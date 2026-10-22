from __future__ import annotations

import re
from dataclasses import dataclass

from schemashift.models import Partner, Split, TaskSpec


@dataclass(frozen=True)
class LeakageReport:
    passed: bool
    problems: tuple[str, ...]
    counts: dict[str, int]


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def audit_splits(splits: dict[Split, list[TaskSpec]], *, require_private_test: bool = False) -> LeakageReport:
    problems: list[str] = []
    if require_private_test and Split.PRIVATE_TEST not in splits:
        problems.append("private test split is missing")

    seen: dict[str, dict[object, Split]] = {
        "task_id": {},
        "seed": {},
        "template_id": {},
        "normalized_text": {},
        "account_ref": {},
        "account_email": {},
        "incident_ref": {},
    }

    def check(namespace: str, value: object, split: Split) -> None:
        previous = seen[namespace].get(value)
        if previous is not None and previous is not split:
            problems.append(f"{namespace} overlap between {previous.value} and {split.value}: {value}")
        else:
            seen[namespace][value] = split

    for split, tasks in splits.items():
        for task in tasks:
            if task.split is not split:
                problems.append(
                    f"task {task.task_id} declares {task.split.value} but appears in {split.value}"
                )
            check("task_id", task.task_id, split)
            check("seed", task.seed, split)
            check("template_id", task.template_id, split)
            check("normalized_text", normalize_text(task.user_request), split)
            check("account_ref", task.account.ref, split)
            check("account_email", task.account.email, split)
            for incident in task.incidents:
                check("incident_ref", incident.ref, split)

    private_tasks = splits.get(Split.PRIVATE_TEST, [])
    non_private = [
        task for split, tasks in splits.items() if split is not Split.PRIVATE_TEST for task in tasks
    ]
    if any(task.partner is Partner.ORBIT for task in non_private):
        problems.append("Orbit task data appeared outside the private test")
    if private_tasks and not any(task.partner is Partner.ORBIT for task in private_tasks):
        problems.append("private test does not contain the held-out Orbit family")

    counts = {split.value: len(tasks) for split, tasks in splits.items()}
    return LeakageReport(passed=not problems, problems=tuple(problems), counts=counts)

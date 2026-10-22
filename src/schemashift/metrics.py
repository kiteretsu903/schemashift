from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


@dataclass(frozen=True)
class EvaluationSummary:
    system: str
    records: int
    successes: int
    success_rate: float
    valid_trajectory_rate: float
    prohibited_side_effect_rate: float
    mean_score: float
    mean_calls: float
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    api_errors: int
    by_partner: dict[str, dict[str, float | int]]
    by_kind: dict[str, dict[str, float | int]]
    failure_reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedBootstrap:
    baseline: str
    candidate: str
    pairs: int
    baseline_success_rate: float
    candidate_success_rate: float
    difference: float
    confidence_level: float
    lower_bound: float
    upper_bound: float
    samples: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_bootstrap_success_difference(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    baseline_name: str,
    candidate_name: str,
    samples: int = 10_000,
    seed: int = 20260804,
    confidence_level: float = 0.95,
) -> PairedBootstrap:
    """Estimate a paired percentile CI after enforcing exact task alignment."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    def index(rows: list[dict[str, Any]], label: str) -> dict[str, int]:
        indexed: dict[str, int] = {}
        for row in rows:
            task_id = str(row.get("task_id") or "")
            if not task_id:
                raise ValueError(f"{label} record is missing task_id")
            if task_id in indexed:
                raise ValueError(f"{label} contains duplicate task_id: {task_id}")
            indexed[task_id] = int(bool(row.get("success")))
        return indexed

    baseline = index(baseline_records, baseline_name)
    candidate = index(candidate_records, candidate_name)
    if baseline.keys() != candidate.keys():
        missing_candidate = sorted(baseline.keys() - candidate.keys())
        missing_baseline = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            "paired task IDs differ: "
            f"missing_candidate={missing_candidate[:5]}, missing_baseline={missing_baseline[:5]}"
        )
    if not baseline:
        raise ValueError("paired comparison requires at least one task")

    task_ids = sorted(baseline)
    differences = [candidate[task_id] - baseline[task_id] for task_id in task_ids]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    bootstrap = [
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(samples)
    ]
    alpha = 1.0 - confidence_level
    lower = percentile(bootstrap, alpha / 2.0)
    upper = percentile(bootstrap, 1.0 - alpha / 2.0)
    assert lower is not None and upper is not None
    return PairedBootstrap(
        baseline=baseline_name,
        candidate=candidate_name,
        pairs=len(task_ids),
        baseline_success_rate=sum(baseline.values()) / len(task_ids),
        candidate_success_rate=sum(candidate.values()) / len(task_ids),
        difference=observed,
        confidence_level=confidence_level,
        lower_bound=lower,
        upper_bound=upper,
        samples=samples,
        seed=seed,
    )


def summarize(records: list[dict[str, Any]], system: str) -> EvaluationSummary:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "partner": defaultdict(list),
        "kind": defaultdict(list),
    }
    failures: Counter[str] = Counter()
    latencies: list[float] = []
    successes = 0
    valid = 0
    prohibited = 0
    scores = 0.0
    calls = 0
    prompt_tokens = completion_tokens = api_errors = 0
    for record in records:
        successes += int(bool(record.get("success")))
        valid += int(bool(record.get("valid_trajectory")))
        prohibited += int(bool(record.get("prohibited_side_effect")))
        scores += float(record.get("score", 0.0))
        calls += int(record.get("tool_call_count", 0))
        if record.get("latency_ms") is not None:
            latencies.append(float(record["latency_ms"]))
        usage = record.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens", 0))
        completion_tokens += int(usage.get("completion_tokens", 0))
        api_errors += int(bool(record.get("api_error")))
        if not record.get("success"):
            failures[str(record.get("failure_reason") or "unknown")] += 1
        groups["partner"][str(record.get("partner"))].append(record)
        groups["kind"][str(record.get("kind"))].append(record)

    def group_summary(items: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                "records": len(group),
                "successes": sum(int(bool(row.get("success"))) for row in group),
                "success_rate": sum(int(bool(row.get("success"))) for row in group) / len(group),
                "mean_score": sum(float(row.get("score", 0.0)) for row in group) / len(group),
            }
            for key, group in sorted(items.items())
        }

    count = len(records)
    return EvaluationSummary(
        system=system,
        records=count,
        successes=successes,
        success_rate=successes / count if count else 0.0,
        valid_trajectory_rate=valid / count if count else 0.0,
        prohibited_side_effect_rate=prohibited / count if count else 0.0,
        mean_score=scores / count if count else 0.0,
        mean_calls=calls / count if count else 0.0,
        p50_latency_ms=median(latencies) if latencies else None,
        p95_latency_ms=percentile(latencies, 0.95),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        api_errors=api_errors,
        by_partner=group_summary(groups["partner"]),
        by_kind=group_summary(groups["kind"]),
        failure_reasons=dict(failures.most_common()),
    )

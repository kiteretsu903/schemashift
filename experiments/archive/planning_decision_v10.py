"""Offline-only validation of the v10 design. This module cannot launch a run."""

from __future__ import annotations

import hashlib
import json
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

INDEX = Path("manifests/experiment-v10-plan-freeze.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dollars(value):
    return Decimal(str(value))


def budget_report(live_balance, conservative_balance, phases):
    """Reserve the whole experiment before its first paid operation."""
    if min(dollars(live_balance), dollars(conservative_balance)) < 0:
        raise ValueError("negative balance")
    caps = {}
    for name, phase in phases.items():
        if min(phase["execution_seconds"], phase["cleanup_seconds"], phase["gpu_count"]) <= 0:
            raise ValueError("all paid phases need execution and cleanup coverage")
        if dollars(phase["hourly_usd"]) < 13:
            raise ValueError("B200 rate below verified bound")
        caps[name] = (
            dollars(phase["execution_seconds"] + phase["cleanup_seconds"])
            * dollars(phase["gpu_count"])
            * dollars(phase["hourly_usd"])
            / 3600
        ).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    reserve = Decimal("5")
    total = sum(caps.values(), Decimal("0"))
    available = min(dollars(live_balance), dollars(conservative_balance))
    shortfall = max(Decimal("0"), total + reserve - available)
    return {
        "phase_caps_usd": {k: float(v) for k, v in caps.items()},
        "total_cap_usd": float(total),
        "reserve_usd": float(reserve),
        "required_start_balance_usd": float(total + reserve),
        "effective_balance_usd": float(available),
        "spendable_above_reserve_usd": float(max(Decimal("0"), available - reserve)),
        "shortfall_usd": float(shortfall.quantize(Decimal("0.01"), rounding=ROUND_CEILING)),
        "affordable": shortfall == 0,
    }


def admission_blockers(
    *, affordable, authorized, holdout_sealed, runner_verified, active_resources, fresh_preflight
):
    checks = {
        "total_budget_does_not_preserve_reserve": affordable,
        "paid_execution_not_authorized": authorized,
        "independent_holdout_not_sealed": holdout_sealed,
        "v10_launcher_and_watchdog_not_verified": runner_verified,
        "active_resources_present": active_resources == 0,
        "launch_time_preflight_required": fresh_preflight,
    }
    return [name for name, passed in checks.items() if not passed]


def validate_plan(index_path=INDEX):
    index = json.loads(index_path.read_text())
    for filename, expected in index["files_sha256"].items():
        if digest(filename) != expected:
            raise ValueError("frozen plan/input drift: " + filename)
    training = json.loads(Path(index["training_protocol"]).read_text())
    evaluation = json.loads(Path(index["evaluation_plan"]).read_text())
    cost = json.loads(Path(index["cost_plan"]).read_text())
    from schemashift.dataset_decision_v10 import load_training

    rows = load_training()
    settings = training["sft"]
    if (len(rows), settings["epochs"], settings["batch_size"], settings["expected_updates"]) != (
        3600,
        1,
        32,
        113,
    ):
        raise ValueError("recipe coverage drift")
    if settings["learning_rate"] != 5e-6 or settings["train_on_what"] != "last_assistant_message":
        raise ValueError("training objective drift")
    if any(x["paid_execution_authorized"] for x in (training, evaluation, cost)):
        raise ValueError("this freeze does not authorize paid work")
    protected = json.loads(Path(training["protected_successes"]).read_text())["protected"]
    if sum(len(set(v)) for v in protected.values()) != 270:
        raise ValueError("original protected gate changed")
    if evaluation["gates"]["original_protected_successes"] != 270:
        raise ValueError("270-success gate cannot be relaxed")
    computed = budget_report(cost["live_balance_usd"], cost["conservative_balance_usd"], cost["phases"])
    if computed != cost["budget_check"]:
        raise ValueError("budget arithmetic differs from frozen plan")
    if evaluation["planned_candidate_episodes"] != 576 or evaluation["planned_control_episodes"] != 96:
        raise ValueError("evaluation coverage changed")
    return {
        "status": "recipe_and_evaluation_design_frozen",
        "verified_files": len(index["files_sha256"]),
        "training_rows": len(rows),
        "optimizer_updates": settings["expected_updates"],
        "paid_execution_authorized": False,
        "budget_check": computed,
        "launch_blockers": admission_blockers(
            affordable=computed["affordable"],
            authorized=False,
            holdout_sealed=evaluation["fresh_holdout"]["case_artifact_sealed"],
            runner_verified=False,
            active_resources=cost["active_resources"],
            fresh_preflight=False,
        ),
    }


if __name__ == "__main__":
    print(json.dumps(validate_plan(), indent=2))

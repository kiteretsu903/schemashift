"""Read-only admission and cost reconciliation for the v16 reference pilot."""

from __future__ import annotations

import importlib.util

from schemashift.execution_boundary_v15 import pending_cost_reserve
from schemashift.execution_contact_v13 import check_phase_budget, read

PREVIOUS_CLOSEOUT = "manifests/experiment-v15-boundary-partial-cleanup-cost.json"


def observe():
    spec = importlib.util.spec_from_file_location("v16_live_reader", "artifacts/v15-execution/read_live.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.observe()


def active_resources(obs):
    active = [
        r["name"]
        for r in obs["deployments"]["data"]
        if r.get("state") in {"CREATING", "UPDATING"}
        or any(int(v) > 0 for v in r.get("replica_stats", {}).values() if str(v).isdigit())
    ]
    active += [
        r["name"]
        for r in obs["trainers"]["data"]
        if r.get("state")
        not in {
            "JOB_STATE_DELETED",
            "JOB_STATE_ARCHIVED",
            "JOB_STATE_COMPLETED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
        }
    ]
    return active


def available_budget(observation, new_closed=(), remaining_caps=(3.25,)):
    obs = observation["observations"]
    if not all(r["ok"] for r in obs.values()):
        raise ValueError("incomplete live observation")
    previous = read(PREVIOUS_CLOSEOUT)
    old_seal = read("manifests/experiment-v15-boundary-execution-seal.json")
    reconciliation = read(old_seal["balance_reconciliation"])
    bounds = {
        r["resource_id"]: r["resource_creation_to_confirmed_close_bound_usd"]
        for r in reconciliation["resource_reconciliation"]
    }
    for closed in [*previous["phases"].values(), *new_closed]:
        if closed["resource_id"] in bounds:
            raise ValueError("resource bound counted twice")
        bounds[closed["resource_id"]] = closed["cost_bound_usd"]
    posted = {}
    for key, field in [("training-usage", "training_costs"), ("deployment-usage", "dedicated_costs")]:
        for row in obs[key]["data"]["account_usage"][field]:
            if row["accelerator_type"] != "NVIDIA_B200_180GB" or (
                field == "dedicated_costs" and row["placement"] != "GLOBAL"
            ):
                raise ValueError("billing shape/placement changed")
            identity = row.get("job_id", row.get("deployment_id", "")).rsplit("/", 1)[-1]
            posted[identity] = posted.get(identity, 0) + int(row["accelerator_seconds"]) * 13 / 3600
    pending = pending_cost_reserve(bounds, posted)
    starting = previous["conservative_credit_after_lifetime_bounds_usd"]
    available = check_phase_budget(
        starting_balance=starting,
        live_balance=obs["account"]["balance_usd"] - pending,
        previous_bounds=[c["cost_bound_usd"] for c in new_closed],
        remaining_caps=remaining_caps,
    )
    return {
        "available_usd": available,
        "previous_admitted_balance_usd": starting,
        "live_credit_usd": obs["account"]["balance_usd"],
        "pending_cost_reserve_usd": pending,
        "reserve_usd": 5,
        "remaining_caps_usd": list(remaining_caps),
        "after_remaining_caps_usd": available - sum(remaining_caps),
        "resource_bounds": bounds,
        "posted_resource_costs": posted,
    }

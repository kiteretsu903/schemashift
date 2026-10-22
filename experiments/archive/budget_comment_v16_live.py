"""Reviewed initial rebase; retain all historical tails and debit new phases once."""

from decimal import Decimal

from schemashift.budget_development_v16 import active_resources


def money(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("nonfinite or negative monetary value")
    return result


def posted_costs(observation):
    obs = observation["observations"]
    if not all(r["ok"] for r in obs.values()):
        raise ValueError("incomplete billing observation")
    posted = {}
    for key, field in (("training-usage", "training_costs"), ("deployment-usage", "dedicated_costs")):
        for row in obs[key]["data"]["account_usage"][field]:
            if row["accelerator_type"] != "NVIDIA_B200_180GB" or (
                field == "dedicated_costs" and row["placement"] != "GLOBAL"
            ):
                raise ValueError("billing accelerator or placement changed")
            identity = row.get("job_id", row.get("deployment_id", "")).rsplit("/", 1)[-1]
            posted[identity] = (
                posted.get(identity, Decimal(0)) + money(row["accelerator_seconds"]) * 13 / 3600
            )
    return posted


def reconciled_budget(observation, reconciliation, closed, *, remaining_count, reserve_usd):
    if not reconciliation["passed"] or type(remaining_count) is not int or not 0 <= remaining_count <= 4:
        raise ValueError("invalid reconciliation or phase count")
    obs = observation["observations"]
    posted = posted_costs(observation)
    if active_resources(obs):
        raise ValueError("active compute must be reconciled before launch")
    bounds = {key: money(value) for key, value in reconciliation["resource_bounds"].items()}
    if len(bounds) != 23:
        raise ValueError("all 23 historical resource bounds are required")
    for key, value in reconciliation["posted_resource_costs"].items():
        if key in bounds and posted.get(key, Decimal(0)) + Decimal("1e-12") < money(value):
            raise ValueError("historical posted usage regressed")
    spent = Decimal(0)
    for receipt in closed:
        identity = receipt["resource_id"]
        if identity in bounds:
            raise ValueError("resource bound counted twice")
        cost = money(receipt["cost_bound_usd"])
        if cost > Decimal("3.25"):
            raise ValueError("previous phase exceeded its cap")
        bounds[identity] = cost
        spent += cost
    pending = sum(
        (max(Decimal(0), bound - posted.get(key, Decimal(0))) for key, bound in bounds.items()), Decimal(0)
    )
    live = money(obs["account"]["balance_usd"])
    available = min(
        money(reconciliation["starting_admitted_balance_usd"]) - spent, live - Decimal("0.01") - pending
    )
    remaining = Decimal("3.25") * remaining_count
    reserve = money(reserve_usd)
    if reserve != 5:
        raise ValueError("the user retained the $5 reserve")
    if available - remaining < reserve:
        raise ValueError("whole remaining matched stage would breach the authorized reserve")
    return {
        "available_usd": float(available),
        "live_credit_usd": float(live),
        "pending_cost_reserve_usd": float(pending),
        "new_phase_bounds_usd": float(spent),
        "remaining_cap_usd": float(remaining),
        "reserve_usd": float(reserve),
        "after_remaining_caps_usd": float(available - remaining),
        "resource_bounds": {k: float(v) for k, v in bounds.items()},
        "posted_resource_costs": {k: float(v) for k, v in posted.items()},
    }

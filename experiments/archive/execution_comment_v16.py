"""Offline-sealable v16 matched comparison; paid admission is a separate gate.

The ready seal deliberately carries no paid authority. A future authorized seal
must retain the reviewed inputs and pass a fresh whole-stage budget preflight.
Nothing contacts a service on import or during offline seal verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from schemashift.budget_development_v16 import active_resources, available_budget, observe
from schemashift.execution_boundary_v15 import (
    PYTHON,
    checked,
    close_resource,
    firectl,
    read,
    supervise_worker,
    write_new,
)
from schemashift.planning_decision_v10 import digest

SEAL = Path("manifests/experiment-v16-comment-ready-seal.json")
PLAN = "manifests/experiment-v16-comment-proposal.json"
REFERENCE = "manifests/experiment-v16-pilot-execution-seal.json"
CLOSEOUT = "manifests/experiment-v16-pilot-closeout-freeze.json"
BASE_OBSERVATION = "artifacts/v16-pilot/closeout-live-observation.json"
PHASE_ORDER = ("training_control", "training_treatment", "evaluate_control", "evaluate_treatment")


def expected_phases():
    plan, reference = read(PLAN), read(REFERENCE)
    phases = {}
    for name in PHASE_ORDER:
        arm = name.split("_")[1]
        training = name.startswith("training_")
        identity = f"schemashift-v16-comment-{'train' if training else 'eval'}-{arm}-001"
        model_id = f"schemashift-v16-comment-{arm}-001"
        phase = {
            "label": "v16-comment-" + arm,
            "arm": arm,
            "resource_id": identity,
            "resource_kind": "rlor-trainer-job" if training else "deployment",
            "model": "accounts/example/models/" + model_id,
            "result_root": "results/comment-v16/" + name,
            "execution_seconds": 600,
            "cleanup_seconds": 300,
            "cap_usd": 3.25,
        }
        if training:
            phase.update(plan[arm], output_model_id=model_id)
        else:
            phase.update(
                endpoint="accounts/example/deployments/" + identity,
                deployment_shape=reference["phases"]["reference"]["deployment_shape"],
                episode_order=reference["phases"]["reference"]["episode_order"],
                startup_seconds=480,
                scored_seconds=120,
            )
        phases[name] = phase
    return phases


def check_hashes(mapping):
    if not mapping:
        raise ValueError("empty file binding")
    for filename, expected in mapping.items():
        if digest(filename) != expected:
            raise ValueError("sealed input changed: " + filename)


def verify_seal(path=SEAL):
    from schemashift.execution_development_v16 import verify_seal as verify_reference

    verify_reference()
    check_hashes(read(CLOSEOUT)["files_sha256"])
    seal = read(path)
    check_hashes(seal["files_sha256"])
    if (
        seal["plan_path"] != PLAN
        or seal["reference_seal"] != REFERENCE
        or seal["selected_family"] != "exact_text"
        or seal["phases"] != expected_phases()
        or seal["phase_order"] != list(PHASE_ORDER)
        or seal["total_cap_usd"] != 13
        or seal["reserve_usd"] != 5
        or seal["total_episodes"] != 204
        or seal["automatic_adoption"] is not False
    ):
        raise ValueError("matched comparison scope differs")
    previous = read("manifests/experiment-v15-boundary-execution-seal.json")
    for key in (
        "source_checkpoint",
        "source_checkpoint_name",
        "source_job_id",
        "base_model",
        "training_shape",
    ):
        if seal[key] != previous[key]:
            raise ValueError("training parent or shape changed")
    reference = read(REFERENCE)
    for key in ("cases_path", "design_path"):
        if seal[key] != reference[key]:
            raise ValueError("evaluation source changed")
    if set(seal["reviews"]) != {"data", "process"}:
        raise ValueError("both independent reviews required")
    required = {
        PLAN,
        REFERENCE,
        CLOSEOUT,
        "src/schemashift/execution_comment_v16.py",
        "src/schemashift/worker_comment_v16.py",
        "src/schemashift/analysis_comment_v16.py",
    }
    required.update(seal["reviews"].values())
    if not required <= set(seal["files_sha256"]):
        raise ValueError("missing essential input bindings")
    for review_path in seal["reviews"].values():
        review = read(review_path)
        if review["passed"] is not True:
            raise ValueError("independent review has blockers")
        check_hashes(review["reviewed_sha256"])
    # Offline readiness is intentionally distinct from authority to spend.
    if seal["paid_execution_authorized"] is not False:
        auth_path = seal["authorization"]
        if auth_path not in seal["files_sha256"]:
            raise ValueError("paid authority must be sealed")
        auth = read(auth_path)
        if (
            auth.get("paid_execution_authorized") is not True
            or auth.get("cap_usd") != 13
            or auth.get("reserve_usd") != 5
            or not auth.get("user_instruction")
            or auth.get("automatic_topup") is not False
        ):
            raise ValueError("missing explicit whole-stage paid authority")
    seal["_seal_path"] = str(path)
    return seal


def require_fresh(timestamp):
    checked_at = datetime.fromisoformat(timestamp)
    if checked_at.tzinfo is None or not timedelta(0) <= datetime.now(UTC) - checked_at <= timedelta(
        minutes=15
    ):
        raise ValueError("live preflight is stale or has an invalid timestamp")


def completed_prefix(seal, phase_name=None):
    """A failed, incomplete or unclosed phase cannot be skipped or retried."""
    closed = []
    names = PHASE_ORDER if phase_name is None else PHASE_ORDER[: PHASE_ORDER.index(phase_name)]
    for name in names:
        phase = seal["phases"][name]
        root = Path(phase["result_root"])
        completion, closure, key = (
            read(root / f) for f in ("completed.json", "closed.json", "key-cleanup.json")
        )
        if (
            completion["passed"] is not True
            or completion["phase"] != name
            or completion["execution_seal_sha256"] != digest(seal["_seal_path"])
            or closure["resource_id"] != phase["resource_id"]
            or closure["cost_bound_usd"] > 3.25
            or not key["revoked"]
            or not key["local_secret_removed"]
            or (root / "temporary-key.json").exists()
        ):
            raise ValueError("previous phase not successfully closed")
        check_hashes(completion["files_sha256"])
        closed.append(closure)
    return closed


def budget_for_observation(observation, closed, remaining_count):
    # Include the reference deployment exactly once, and retain all earlier
    # pending-cost allowances. No automatic ledger rebase on displayed credit.
    reference_closed = read("manifests/experiment-v16-pilot-cleanup-cost.json")["phase"]
    return available_budget(observation, [reference_closed, *closed], remaining_caps=[3.25] * remaining_count)


def validate_observation(seal, observation):
    require_fresh(observation["checked_at_utc"])
    obs = observation["observations"]
    if not obs or not all(v["ok"] for v in obs.values()) or active_resources(obs):
        raise ValueError("incomplete observation or active compute")
    base = read(BASE_OBSERVATION)["observations"]
    for shape in ("training-shape", "deployment-shape"):
        if obs[shape]["data"] != base[shape]["data"]:
            raise ValueError("live accelerator shape changed")
    model = obs["parent-model"]["data"]
    if (
        model["name"] != read(PLAN)["initialization"]["parent_model"]
        or model["state"] != "READY"
        or model["fine_tuning_job"] != "accounts/example/rlorTrainerJobs/" + seal["source_job_id"]
    ):
        raise ValueError("training parent changed or unavailable")
    if not any(
        c.get("checkpoint_type") == "CHECKPOINT_TYPE_TRAINING_LORA"
        and c.get("restore_reference") == seal["source_checkpoint"]
        and c["name"].endswith("/" + seal["source_checkpoint_name"])
        and c.get("availability")
        for c in obs["parent-checkpoints"]["data"]
    ):
        raise ValueError("exact resumable parent unavailable")


def validate_trained_model(seal, phase_name, observation):
    if not phase_name.startswith("evaluate_"):
        return
    phase = seal["phases"][phase_name]
    training = seal["phases"]["training_" + phase["arm"]]
    result = read(Path(training["result_root"]) / "result.json")
    model = observation["evaluated_model"]
    checkpoint = result["registration"]["annotations"]["fireworks.ai/checkpoint_id"]
    job = "accounts/example/rlorTrainerJobs/" + training["resource_id"]
    if (
        result["steps"] != 36
        or result["loss_tokens"] != 51840
        or result["model"] != phase["model"]
        or model["state"] != "READY"
        or model["name"] != phase["model"]
        or model["fine_tuning_job"] != job
        or not checkpoint.startswith("step-36-")
    ):
        raise ValueError("evaluation must use this arm's final local checkpoint")
    if not any(
        c["name"] == job + "/checkpoints/" + checkpoint
        and c["checkpoint_type"] == "CHECKPOINT_TYPE_INFERENCE_LORA"
        and c.get("availability")
        for c in observation["evaluated_checkpoints"]
    ):
        raise ValueError("registered final inference checkpoint unavailable")


def make_preflight(seal, phase_name, observation_path):
    observation = read(observation_path)
    validate_observation(seal, observation)
    closed = completed_prefix(seal, phase_name)
    validate_trained_model(seal, phase_name, observation)
    budget = budget_for_observation(observation, closed, len(PHASE_ORDER) - PHASE_ORDER.index(phase_name))
    return {
        "checked_at_utc": observation["checked_at_utc"],
        "account": "example",
        "phase": phase_name,
        "execution_seal_sha256": digest(seal["_seal_path"]),
        "live_observation_path": str(observation_path),
        "live_observation_sha256": digest(observation_path),
        "budget": budget,
        "whole_remaining_experiment_reserved": True,
    }


def validate_preflight(seal, phase_name, receipt):
    if seal["paid_execution_authorized"] is not True:
        raise ValueError("offline readiness seal does not authorize paid execution")
    path = receipt["live_observation_path"]
    if digest(path) != receipt["live_observation_sha256"]:
        raise ValueError("live observation changed")
    if receipt != make_preflight(seal, phase_name, path):
        raise ValueError("preflight does not match recomputed whole-stage admission")


def verify_worker_admission(seal, phase_name):
    root = Path(seal["phases"][phase_name]["result_root"])
    started = read(root / "started.json")
    token = os.environ.get("SCHEMASHIFT_V16_WORKER_TOKEN", "")
    if (
        not token
        or not os.environ.get("FIREWORKS_API_KEY")
        or started["phase"] != phase_name
        or started["execution_seal_sha256"] != digest(seal["_seal_path"])
        or started["worker_token_sha256"] != hashlib.sha256(token.encode()).hexdigest()
        or read(root / "phase.json") != seal["phases"][phase_name]
    ):
        raise ValueError("matching external supervisor required")
    require_fresh(started["started_at_utc"])
    validate_preflight(seal, phase_name, read(root / "budget.json"))


def run_phase(phase_name, *, preflight_path, seal_path=SEAL):
    seal = verify_seal(seal_path)
    receipt = read(preflight_path)
    validate_preflight(seal, phase_name, receipt)  # No service calls before this gate.
    phase = seal["phases"][phase_name]
    identity, kind = phase["resource_id"], phase["resource_kind"]
    previous = firectl(kind, "get", identity, "-o", "json")
    if previous.returncode == 0 or "NotFound" not in previous.stderr:
        raise RuntimeError("owned resource name must be confirmed unused")
    if phase_name.startswith("training_"):
        model = firectl("model", "get", phase["output_model_id"], "-o", "json")
        if model.returncode == 0 or "NotFound" not in model.stderr:
            raise RuntimeError("output model name must be confirmed unused")
    root = Path(phase["result_root"])
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "phase.json", phase)
    write_new(root / "budget.json", receipt)
    expires = (datetime.now(UTC) + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    credential = json.loads(
        checked("api-key", "create", "--key-name", identity, "--expire-time", expires, "-o", "json")
    )
    secret = root / "temporary-key.json"
    token = secrets.token_hex(32)
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise SystemExit("v16 interrupted; closing owned resource")

    try:
        signal.signal(signal.SIGTERM, interrupted)
        with os.fdopen(os.open(secret, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
            json.dump(credential, handle)
        started_at = datetime.now(UTC)
        deadline = time.monotonic() + 890
        write_new(
            root / "started.json",
            {
                "phase": phase_name,
                "started_at_utc": started_at.isoformat(),
                "execution_seal_sha256": digest(seal_path),
                "worker_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            },
        )
        env = {
            **os.environ,
            "FIREWORKS_API_KEY": credential["key"],
            "SCHEMASHIFT_V16_WORKER_TOKEN": token,
            "FIRECTL_NO_UPDATE_NOTICE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONOPTIMIZE": "0",
            "PYTHONPATH": "src",
        }
        supervise_worker(
            [PYTHON, "-m", "schemashift.worker_comment_v16", "--phase", phase_name, "--seal", str(seal_path)],
            root / "worker.log",
            execution_seconds=600,
            env=env,
            cleanup=lambda: close_resource(
                kind, identity, phase_root=root, started_at=started_at, lifetime_deadline=deadline
            ),
        )
        closed = read(root / "closed.json")
        if closed["cost_bound_usd"] > 3.25 or not (root / "result.json").is_file():
            raise RuntimeError("phase exceeded cap or did not complete")
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        checked("api-key", "delete", credential["key_id"], "-o", "json")
        if credential["key_id"] in checked("api-key", "list", "-o", "json"):
            raise RuntimeError("temporary key remains listed")
        secret.unlink(missing_ok=True)
        write_new(root / "key-cleanup.json", {"revoked": True, "local_secret_removed": True})
    # Only successful workers with confirmed resource AND key cleanup get this receipt.
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    write_new(
        root / "completed.json",
        {
            "passed": True,
            "phase": phase_name,
            "execution_seal_sha256": digest(seal_path),
            "files_sha256": {str(p): digest(p) for p in paths},
        },
    )
    return closed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, default=SEAL)
    parser.add_argument("--phase", choices=PHASE_ORDER)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--admit-observation", type=Path, help="recompute admission from saved live reads")
    parser.add_argument("--observe", type=Path, help="save a new read-only live observation; no paid calls")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if sum(bool(v) for v in (args.execute, args.observe, args.admit_observation)) > 1:
        parser.error("observation, admission and paid execution are separate commands")
    seal = verify_seal(args.seal)
    if args.observe:
        observation = observe()
        if args.phase and args.phase.startswith("evaluate_"):
            phase = seal["phases"][args.phase]
            observation["evaluated_model"] = json.loads(checked("model", "get", phase["model"], "-o", "json"))
            training = seal["phases"]["training_" + phase["arm"]]
            observation["evaluated_checkpoints"] = json.loads(
                checked("rlor-trainer-job", "list-checkpoints", training["resource_id"], "-o", "json")
            )
        write_new(args.observe, observation)
    elif args.admit_observation:
        if not args.phase or not args.preflight:
            parser.error("admission needs explicit phase and new preflight output path")
        write_new(args.preflight, make_preflight(seal, args.phase, args.admit_observation))
    elif args.execute:
        if not args.phase or not args.preflight:
            parser.error("explicit phase and fresh preflight required")
        print(json.dumps(run_phase(args.phase, preflight_path=args.preflight, seal_path=args.seal)))
    else:
        print(
            json.dumps(
                {
                    "offline_seal_verified": True,
                    "paid_execution_authorized": seal["paid_execution_authorized"],
                    "total_cap_usd": seal["total_cap_usd"],
                }
            )
        )


if __name__ == "__main__":
    main()

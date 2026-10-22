"""Authorized three-checkpoint evaluation; no training or automatic adoption."""

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

from schemashift.budget_comment_v16_live import reconciled_budget
from schemashift.budget_development_v16 import active_resources, observe
from schemashift.comparison_v17 import MODELS, PHASE_ORDER, SUITES, phases
from schemashift.execution_boundary_v15 import (
    PYTHON,
    checked,
    close_resource,
    firectl,
    read,
    supervise_worker,
    write_new,
)
from schemashift.execution_comment_v16_live import completed_prefix as prior_completed
from schemashift.execution_comment_v16_live import verify_seal as verify_prior
from schemashift.planning_decision_v10 import digest

SEAL = Path("manifests/experiment-v17-comparison-execution-seal.json")
PRIOR_CLOSEOUT = "manifests/experiment-v16-comment-closeout-freeze.json"
MODEL_SNAPSHOTS = {
    "sft": "artifacts/v5-fresh/sft-model-before.json",
    "rft": "artifacts/v5-fresh/rft-model-before.json",
    "v16": "artifacts/v16-launch/evaluate-treatment-live.json",
}


def check_hashes(mapping):
    if not mapping:
        raise ValueError("empty file binding")
    for filename, expected in mapping.items():
        if digest(filename) != expected:
            raise ValueError("sealed input changed: " + filename)


def verify_seal(path=SEAL):
    verify_prior()
    check_hashes(read(PRIOR_CLOSEOUT)["files_sha256"])
    seal = read(path)
    check_hashes(seal["files_sha256"])
    if (
        seal["phases"] != phases()
        or seal["phase_order"] != list(PHASE_ORDER)
        or seal["models"] != MODELS
        or seal["suites"] != SUITES
        or seal["total_cap_usd"] != 9.75
        or seal["reserve_usd"] != 5
        or seal["total_episodes"] != 576
        or seal["training"] is not False
        or seal["automatic_adoption"] is not False
        or seal["automatic_retry"] is not False
    ):
        raise ValueError("frozen comparison scope changed")
    auth = read(seal["authorization"])
    if (
        auth["paid_execution_authorized"] is not True
        or auth["cap_usd"] != 9.75
        or auth["reserve_usd"] != 5
        or not auth["user_instruction"]
        or auth["automatic_topup"] is not False
    ):
        raise ValueError("missing comparison paid authority")
    required = {
        str(Path(__file__).relative_to(Path.cwd())),
        "src/schemashift/comparison_v17.py",
        "src/schemashift/worker_comparison_v17.py",
        seal["authorization"],
        PRIOR_CLOSEOUT,
        *seal["suite_protocols"].values(),
        *SUITES.values(),
        *MODEL_SNAPSHOTS.values(),
    }
    if not required <= set(seal["files_sha256"]):
        raise ValueError("essential comparison inputs not bound")
    original = read("manifests/experiment-v5-trained-independent-protocol.json")
    if digest(SUITES["original"]) != original["independent_scenarios_sha256"]:
        raise ValueError("original v5 scenarios changed")
    from schemashift.independent_v5 import validate_protocol

    for suite, protocol in seal["suite_protocols"].items():
        p, tasks, _ = validate_protocol(Path(protocol), MODELS["sft"])
        if p["models"] != MODELS or p["independent_scenarios"] != SUITES[suite] or len(tasks) != 96:
            raise ValueError("suite protocol differs")
        if p["system_prompt_sha256"] != original["system_prompt_sha256"]:
            raise ValueError("original v5 prompt changed")
        for phase in seal["phases"].values():
            if p["model_endpoints"][phase["model"]] != phase["endpoint"]:
                raise ValueError("suite endpoint binding differs")
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


def budget_for_observation(observation, closed, remaining_count, *, seal):
    prior = verify_prior()
    return reconciled_budget(
        observation,
        read(prior["balance_reconciliation"]),
        [*prior_completed(prior), *closed],
        remaining_count=remaining_count,
        reserve_usd=5,
    )


def validate_observation(seal, observation):
    require_fresh(observation["checked_at_utc"])
    obs = observation["observations"]
    if not all(v["ok"] for v in obs.values()) or active_resources(obs):
        raise ValueError("incomplete live observation or active compute")
    baseline = read("artifacts/v16-results/closeout-live-observation.json")["observations"]
    if obs["deployment-shape"]["data"] != baseline["deployment-shape"]["data"]:
        raise ValueError("deployment shape changed")
    for name, model in MODELS.items():
        current = observation["comparison_models"][name]
        old = read(MODEL_SNAPSHOTS[name])
        if name == "v16":
            old = old["evaluated_model"]
        if current["state"] != "READY" or current["name"] != model:
            raise ValueError("requested model is not ready")
        for field in ("name", "create_time", "fine_tuning_job", "kind", "peft_details"):
            if current[field] != old[field]:
                raise ValueError("frozen model identity changed: " + name + "/" + field)


def observe_models():
    observation = observe()
    observation["comparison_models"] = {
        name: json.loads(checked("model", "get", model, "-o", "json")) for name, model in MODELS.items()
    }
    # Timestamp the complete account-plus-model observation.
    observation["checked_at_utc"] = datetime.now(UTC).isoformat()
    return observation


def make_preflight(seal, phase_name, observation_path):
    observation = read(observation_path)
    validate_observation(seal, observation)
    closed = completed_prefix(seal, phase_name)
    budget = budget_for_observation(
        observation, closed, len(PHASE_ORDER) - PHASE_ORDER.index(phase_name), seal=seal
    )
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
    token = os.environ.get("SCHEMASHIFT_V17_WORKER_TOKEN", "")
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
        raise SystemExit("v17 interrupted; closing owned resource")

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
            "SCHEMASHIFT_V17_WORKER_TOKEN": token,
            "FIRECTL_NO_UPDATE_NOTICE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONOPTIMIZE": "0",
            "PYTHONPATH": "src",
        }
        supervise_worker(
            [
                PYTHON,
                "-m",
                "schemashift.worker_comparison_v17",
                "--phase",
                phase_name,
                "--seal",
                str(seal_path),
            ],
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
    parser.add_argument("--observe", type=Path)
    parser.add_argument("--admit-observation", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if sum(bool(v) for v in (args.execute, args.observe, args.admit_observation)) > 1:
        parser.error("keep observation, admission, execution separate")
    seal = verify_seal(args.seal)
    if args.observe:
        write_new(args.observe, observe_models())
    elif args.admit_observation:
        if not args.phase or not args.preflight:
            parser.error("phase and preflight required")
        write_new(args.preflight, make_preflight(seal, args.phase, args.admit_observation))
    elif args.execute:
        if not args.phase or not args.preflight:
            parser.error("phase and preflight required")
        print(json.dumps(run_phase(args.phase, preflight_path=args.preflight, seal_path=args.seal)))
    else:
        print(json.dumps({"seal_verified": True, "cap_usd": seal["total_cap_usd"], "episodes": 576}))


if __name__ == "__main__":
    main()

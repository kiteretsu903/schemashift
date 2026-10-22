"""v16 bounded reference-pilot execution. No network on import."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from schemashift.execution_contact_v13 import (
    PYTHON,
    budget_check,
    checked,
    close_resource,
    firectl,
    read,
    supervise_worker,
    write_new,
)
from schemashift.execution_contact_v13 import (
    check_phase_budget as check_phase_budget,
)
from schemashift.planning_decision_v10 import digest

SEAL = Path("manifests/experiment-v16-pilot-execution-seal.json")
ROOT = Path("artifacts/v16-pilot")


def verify_seal(path=SEAL):
    from schemashift.diagnostic_development_v16 import expected_ids
    from schemashift.execution_boundary_v15 import verify_seal as verify_previous

    verify_previous()
    seal = read(path)
    for filename, expected in seal["files_sha256"].items():
        if digest(filename) != expected:
            raise ValueError("execution input changed: " + filename)
    auth = read(seal["authorization"])
    if not auth["paid_pilot_authorized"] or auth["training_authorized"] or auth["reserve_usd"] != 5:
        raise ValueError("missing or invalid pilot authority")
    if set(seal["phases"]) != {"reference"} or seal["total_episodes"] != 102:
        raise ValueError("only the reference pilot is admitted")
    phase = seal["phases"]["reference"]
    if (
        auth["cap_usd"] != 3.25
        or seal["total_cap_usd"] != 3.25
        or phase["cap_usd"] != 3.25
        or phase["execution_seconds"] != 600
        or phase["cleanup_seconds"] != 300
        or phase["startup_seconds"] != 480
        or phase["scored_seconds"] != 120
        or phase["resource_kind"] != "deployment"
    ):
        raise ValueError("pilot resource scope changed")
    if phase["model"] != "accounts/example/models/schemashift-v13-contact-control-001":
        raise ValueError("reference checkpoint changed")
    if phase["endpoint"] != "accounts/example/deployments/" + phase["resource_id"]:
        raise ValueError("endpoint binding changed")
    cases = read(seal["cases_path"])["cases"]
    if len(cases) != 252 or not read(seal["validation_path"])["passed"]:
        raise ValueError("development cases must pass offline validation")
    order = phase["episode_order"]
    if len(order) != 102 or set(order) != expected_ids(cases, pilot=True):
        raise ValueError("pilot episode membership changed")
    return seal


def run_phase(phase, *, preflight_path):
    seal = verify_seal()
    config = seal["phases"][phase]
    preflight = read(preflight_path)
    budget_check(preflight, {"run_id": config["resource_id"], "timeout_seconds": config["execution_seconds"]})
    if preflight["execution_seal_sha256"] != digest(SEAL):
        raise ValueError("budget receipt not bound to this execution seal")
    if preflight["estimated_max_cost_usd"] != config["cap_usd"]:
        raise ValueError("phase cap changed")
    if not preflight["whole_remaining_experiment_reserved"]:
        raise ValueError("whole diagnostic must be reserved")
    root = Path(config["result_root"])
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "phase.json", config)
    write_new(root / "budget.json", preflight)
    identity, kind = config["resource_id"], config["resource_kind"]
    if not re.fullmatch(r"schemashift-v16-[a-z0-9-]+", identity):
        raise ValueError("unowned resource identity")
    previous = firectl(kind, "get", identity, "-o", "json")
    if previous.returncode == 0 or "NotFound" not in previous.stderr:
        raise RuntimeError("resource name not confirmed unused")
    expires = (datetime.now(UTC) + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    credential = json.loads(
        checked("api-key", "create", "--key-name", identity, "--expire-time", expires, "-o", "json")
    )
    secret = root / "temporary-key.json"
    process_env = {
        **os.environ,
        "FIREWORKS_API_KEY": credential["key"],
        "FIRECTL_NO_UPDATE_NOTICE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": "src",
    }
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise SystemExit("v16 supervisor interrupted; closing owned resource")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        fd = os.open(secret, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(credential))
        started_at = datetime.now(UTC)
        lifetime_deadline = time.monotonic() + config["execution_seconds"] + 290
        write_new(
            root / "started.json",
            {"started_at_utc": started_at.isoformat(), "execution_seal_sha256": digest(SEAL), "phase": phase},
        )
        print(
            json.dumps(
                {
                    "starting": phase,
                    "resource": identity,
                    "execution_seconds": config["execution_seconds"],
                    "cap_usd": config["cap_usd"],
                }
            ),
            flush=True,
        )
        supervise_worker(
            [PYTHON, "-m", "schemashift.worker_development_v16", "--phase", phase],
            root / "worker.log",
            execution_seconds=config["execution_seconds"],
            env=process_env,
            cleanup=lambda: close_resource(
                kind, identity, phase_root=root, started_at=started_at, lifetime_deadline=lifetime_deadline
            ),
        )
        closed = read(root / "closed.json")
        if closed["cost_bound_usd"] > config["cap_usd"]:
            raise RuntimeError("phase exceeded allowance; no further resource may start")
        print(json.dumps({"completed": phase, "lifetime": closed}), flush=True)
        return closed
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        checked("api-key", "delete", credential["key_id"], "-o", "json")
        if credential["key_id"] in checked("api-key", "list", "-o", "json"):
            raise RuntimeError("temporary key remains listed")
        secret.unlink(missing_ok=True)
        write_new(
            root / "key-cleanup.json",
            {"revoked": True, "local_secret_removed": True, "checked_at_utc": datetime.now(UTC).isoformat()},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["reference"])
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        if not args.phase or not args.preflight:
            parser.error("explicit phase and fresh preflight required")
        run_phase(args.phase, preflight_path=args.preflight)
    else:
        print(json.dumps(verify_seal(), indent=2))

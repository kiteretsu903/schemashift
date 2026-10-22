"""V13 execution admission and an independent parent-process deadline supervisor.

No network operation occurs on import. The supervised worker owns exactly one
predeclared, previously absent resource. No failed phase is automatically retried.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

from schemashift.planning_decision_v10 import digest, dollars

ACCOUNT = "example"
ROOT = Path("artifacts/v13-contact")
SEAL = Path("manifests/experiment-v13-contact-execution-seal.json")
PYTHON = "/private/tmp/schemashift-v3-env/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    with Path(path).open("x") as f:
        f.write(json.dumps(data, indent=2, sort_keys=True) + "\n")


def firectl(*args, timeout=30):
    return subprocess.run(
        ["firectl", "--account-id", ACCOUNT, *args],
        env={**os.environ, "FIRECTL_NO_UPDATE_NOTICE": "1"},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def checked(*args, timeout=30):
    response = firectl(*args, timeout=timeout)
    if response.returncode:
        raise RuntimeError("Fireworks operation failed: " + response.stderr[:800])
    return response.stdout


def verify_seal(path=SEAL):
    seal = read(path)
    from schemashift.execution_controlled_v11_retry import verify_seal as verify_previous

    verify_previous()
    for filename, expected in seal["files_sha256"].items():
        if digest(filename) != expected:
            raise ValueError("execution input changed: " + filename)
    authorization = read(seal["authorization"])
    if not authorization["paid_execution_authorized"] or authorization["reserve_usd"] != 5:
        raise ValueError("missing authorization")
    if authorization["cap_usd"] != 3960 * 13 / 3600:
        raise ValueError("cap differs")
    if not seal["independent_holdout_sealed"]:
        raise ValueError("holdout must be frozen")

    if sum(p["execution_seconds"] + 300 for p in seal["phases"].values()) != 3960:
        raise ValueError("lifetime allowance changed")
    if sum(p["cap_usd"] for p in seal["phases"].values()) != authorization["cap_usd"]:
        raise ValueError("phase caps do not match total cap")
    return seal


def budget_check(receipt: dict, plan: dict) -> None:
    """A short-lived receipt binds the launch to a reviewed conservative estimate."""
    checked = datetime.fromisoformat(receipt["checked_at_utc"])
    age = datetime.now(UTC) - checked
    if checked.tzinfo is None or not timedelta(0) <= age <= timedelta(minutes=15):
        raise ValueError("budget preflight is stale or has an invalid timestamp")
    if receipt.get("run_id") != plan["run_id"] or receipt.get("account") != "example":
        raise ValueError("budget receipt belongs to a different run or account")
    if receipt.get("active_project_resources") != 0:
        raise ValueError("account for active project resource cost before launching")
    fields = ("balance_usd", "posted_month_usage_usd", "estimated_max_cost_usd", "hourly_cost_bound_usd")
    if any(not math.isfinite(float(receipt[k])) or float(receipt[k]) < 0 for k in fields):
        raise ValueError("invalid cost estimate")
    cost = float(receipt["estimated_max_cost_usd"])
    if float(receipt["hourly_cost_bound_usd"]) <= 0:
        raise ValueError("hourly cost bound must be positive")
    minimum_estimate = float(receipt["hourly_cost_bound_usd"]) * (plan["timeout_seconds"] + 300) / 3600
    if cost <= 0 or cost < minimum_estimate:
        raise ValueError("estimate must cover timeout plus five minutes of cleanup")
    if cost > float(receipt["balance_usd"]) - 5:
        raise ValueError("preserve a $5 credit reserve; no automatic top-up")
    # The user authorized this controlled contact repair with the $5 reserve as the spend boundary.


def check_phase_budget(*, starting_balance, live_balance, previous_bounds, remaining_caps):
    """Debit each closed phase once against the admitted initial credit, not lagging bills."""
    available = min(
        dollars(starting_balance) - sum(map(dollars, previous_bounds), dollars(0)),
        dollars(live_balance) - dollars("0.01"),
    )
    total_remaining = sum(map(dollars, remaining_caps), dollars(0))
    if available - total_remaining < dollars(5):
        raise ValueError("full remaining experiment would violate $5 reserve")
    return float(available)


def stop_worker(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def supervise_worker(command, log_path, *, execution_seconds, cleanup, env=None):
    """Always clean up after success, worker failure or a hard wall-clock timeout."""
    started = time.monotonic()
    process = None
    timed_out = False
    try:
        with Path(log_path).open("x") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            try:
                process.wait(timeout=max(0.01, execution_seconds - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_worker(process)
            if timed_out:
                raise TimeoutError("v11 paid phase reached frozen execution deadline")
            if process.returncode:
                raise RuntimeError(f"worker failed with exit status {process.returncode}")
    finally:
        try:
            if process is not None:
                stop_worker(process)
        finally:
            cleanup()


def close_resource(kind, identity, *, phase_root, started_at, lifetime_deadline):
    """Delete only the predeclared resource, then independently confirm closure."""
    absent_count = 0
    record = None
    deleted = False
    while time.monotonic() < lifetime_deadline:
        timeout = min(20, max(1, lifetime_deadline - time.monotonic()))
        try:
            response = firectl(kind, "get", identity, "-o", "json", timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        absent = response.returncode != 0 and "NotFound" in response.stderr
        if response.returncode and not absent:
            time.sleep(2)
            continue
        if not absent:
            record = json.loads(response.stdout)
            created = datetime.fromisoformat(record["create_time"].replace("Z", "+00:00"))
            if created < started_at - timedelta(seconds=10):
                raise RuntimeError("resource predates this phase; refusing to claim ownership")
        terminal = absent or record.get("state") in {"DELETED", "JOB_STATE_DELETED"}
        if terminal:
            absent_count += 1
            # If creation was ambiguous, confirm continued absence across several probes.
            if not absent or record is not None or absent_count >= 4:
                closed_at = datetime.now(UTC)
                seconds = (closed_at - started_at).total_seconds()
                receipt = {
                    "kind": kind,
                    "resource_id": identity,
                    "started_before_create_at_utc": started_at.isoformat(),
                    "confirmed_closed_at_utc": closed_at.isoformat(),
                    "state": "ABSENT" if absent else record["state"],
                    "full_lifetime_seconds": seconds,
                    "cost_bound_usd": seconds * 13 / 3600,
                    "billing_is_upper_bound": True,
                }
                write_new(phase_root / "closed.json", receipt)
                if record is not None:
                    write_new(phase_root / "resource-after.json", record)
                return receipt
            time.sleep(5)
            continue
        absent_count = 0
        if not deleted:
            args = [kind, "delete", identity]
            if kind == "deployment":
                args.append("--ignore-checks")
            timeout = min(20, max(1, lifetime_deadline - time.monotonic()))
            try:
                result = firectl(*args, "-o", "json", timeout=timeout)
            except subprocess.TimeoutExpired:
                continue
            with (phase_root / "cleanup-attempts.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {"returncode": result.returncode, "output": result.stdout, "error": result.stderr}
                    )
                    + "\n"
                )
            deleted = result.returncode == 0 or "NotFound" in result.stderr
        time.sleep(2)
    raise RuntimeError("cleanup not confirmed inside reserved lifetime; stop all subsequent phases")


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
        raise ValueError("training-only budget admission is insufficient")
    root = Path(config["result_root"])
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "phase.json", config)
    write_new(root / "budget.json", preflight)
    identity, kind = config["resource_id"], config["resource_kind"]
    if not re.fullmatch(r"schemashift-v13-[a-z0-9-]+", identity):
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
        raise SystemExit("v13 supervisor interrupted; closing owned resource")

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
            [PYTHON, "-m", "schemashift.worker_contact_v13", "--phase", phase],
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
    parser.add_argument(
        "--phase", choices=["training_a", "training_b", "evaluate_a", "evaluate_b", "evaluate_v9"]
    )
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        if not args.phase or not args.preflight:
            parser.error("execution requires an explicit phase and fresh preflight")
        run_phase(args.phase, preflight_path=args.preflight)
    else:
        print(json.dumps(verify_seal(), indent=2))

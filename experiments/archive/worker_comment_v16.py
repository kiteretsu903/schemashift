"""Externally admitted v16 comment-training and fixed development-pilot worker.

Only the matched training/pilot stage is supported. Importing this module does
not allocate resources; the execution supervisor must admit every worker phase.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from schemashift import worker_boundary_v15
from schemashift.analysis_development_v16_replay import audit_run, summarize
from schemashift.diagnostic_development_v16 import expected_ids, run_development
from schemashift.fireworks import FireworksToolClient
from schemashift.worker_decision_v10 import warmup

PROPOSAL = Path("manifests/experiment-v16-comment-proposal.json")
REFERENCE_SEAL = Path("manifests/experiment-v16-pilot-execution-seal.json")
SELECTED_FAMILY = "exact_text"


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")


def phase_name(seal, phase, kind):
    arm = phase["arm"]
    if arm not in {"control", "treatment"}:
        raise ValueError("only the matched control/treatment stage is supported")
    name = f"{kind}_{arm}"
    if seal["phases"].get(name) != phase:
        raise ValueError("worker phase differs from sealed configuration")
    if phase["model"] != f"accounts/example/models/schemashift-v16-comment-{arm}-001":
        raise ValueError("worker model differs from the fixed v16 arm")
    if seal.get("selected_family") != SELECTED_FAMILY:
        raise ValueError("selected family must remain exact_text")
    return name


def validate_proposal(seal):
    if Path(seal["plan_path"]).resolve() != PROPOSAL.resolve():
        raise ValueError("explicit v16 proposal plan_path is required")
    proposal = read(seal["plan_path"])
    if proposal["selected_family"] != SELECTED_FAMILY:
        raise ValueError("proposal family changed")
    return proposal


def training_worker(seal, phase):
    from schemashift.execution_comment_v16 import verify_worker_admission

    name = phase_name(seal, phase, "training")
    proposal = validate_proposal(seal)
    if seal["source_checkpoint"] != proposal["parent"]:
        raise ValueError("training parent differs from the v16 proposal")
    for key in ("data_file", "schedule_file"):
        if Path(phase[key]).resolve() != Path(proposal[phase["arm"]][key]).resolve():
            raise ValueError("training data/schedule differs from the v16 arm")
    verify_worker_admission(seal, name)
    # Preserve the reviewed v15 trainer, weights-only initialization, fresh Adam,
    # exact per-update/profile dose checks, and final-local-update-36 checkpoint.
    # Explicitly override the plan path; never let v15's default plan be selected.
    worker_boundary_v15.training_worker({**seal, "plan_path": str(PROPOSAL)}, phase)


def validate_pilot(seal, phase):
    name = phase_name(seal, phase, "evaluate")
    validate_proposal(seal)
    if Path(seal["reference_seal"]).resolve() != REFERENCE_SEAL.resolve():
        raise ValueError("pilot reference seal differs")
    reference = read(seal["reference_seal"])
    if (
        Path(seal["cases_path"]).resolve() != Path(reference["cases_path"]).resolve()
        or Path(seal["design_path"]).resolve() != Path(reference["design_path"]).resolve()
    ):
        raise ValueError("pilot cases/design differ from reference")
    if phase["endpoint"] != "accounts/example/deployments/" + phase["resource_id"]:
        raise ValueError("model/deployment endpoint binding changed")
    if (
        phase["resource_kind"] != "deployment"
        or phase["startup_seconds"] != 480
        or phase["scored_seconds"] != 120
        or phase["execution_seconds"] != 600
    ):
        raise ValueError("pilot timing/resource scope changed")
    cases = read(seal["cases_path"])["cases"]
    order = phase["episode_order"]
    if (
        len(order) != 102
        or len(set(order)) != 102
        or set(order) != expected_ids(cases, pilot=True)
        or order != reference["phases"]["reference"]["episode_order"]
    ):
        raise ValueError("pilot must retain the exact reference 102-episode order")
    return name, cases


def summarize_saved_audit(audit_path, design_path):
    """Summarize persisted replay results without selecting a family again."""
    audit = read(audit_path)
    rows = audit["rows"]
    if not audit["passed"] or audit["episodes"] != 102 or len(rows) != 102:
        raise ValueError("incomplete or failed pilot audit")
    design = read(design_path)
    families = [family["id"] for family in design["families"]]
    if SELECTED_FAMILY not in families or set(row["family"] for row in rows) != set(families):
        raise ValueError("pilot family coverage differs")
    return {
        "version": "v16-comment-pilot-summary-1",
        "selected_family": SELECTED_FAMILY,
        "family_selection_fixed_before_training": True,
        "stage": "pilot",
        "development_only": True,
        "expanded": False,
        "adopted": False,
        "replay_passed": True,
        **summarize(rows),
        "by_start": {
            start: summarize([row for row in rows if row["start_condition"] == start])
            for start in ("normal", "supplied_prefix")
        },
        "by_family": {
            family: {
                **summarize([row for row in rows if row["family"] == family]),
                "by_start": {
                    start: summarize(
                        [row for row in rows if row["family"] == family and row["start_condition"] == start]
                    )
                    for start in ("normal", "supplied_prefix")
                },
            }
            for family in families
        },
    }


def evaluation_worker(seal, phase):
    from schemashift.execution_comment_v16 import checked, verify_worker_admission

    name, cases = validate_pilot(seal, phase)
    verify_worker_admission(seal, name)
    started = time.monotonic()
    root = Path(phase["result_root"])
    write_new(root / "config.json", {**phase, "selected_family": SELECTED_FAMILY})
    payload = checked(
        "deployment",
        "create",
        phase["model"],
        "--deployment-id",
        phase["resource_id"],
        "--deployment-shape",
        phase["deployment_shape"],
        "--min-replica-count",
        "0",
        "--max-replica-count",
        "1",
        "--scale-to-zero-window",
        "300s",
        "-o",
        "json",
        timeout=60,
    )
    write_new(root / "deployment-create.json", json.loads(payload))
    client = FireworksToolClient(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url="https://api.fireworks.ai/inference/v1",
        model=phase["endpoint"],
    )

    async def evaluate():
        try:
            startup_left = phase["startup_seconds"] - (time.monotonic() - started)
            if startup_left <= 0:
                raise TimeoutError("pilot startup allowance exhausted")
            await asyncio.wait_for(warmup(client, root), timeout=startup_left)
            scored_started = time.monotonic()
            scored_left = min(
                phase["scored_seconds"], phase["execution_seconds"] - (scored_started - started)
            )
            if scored_left <= 0:
                raise TimeoutError("pilot execution allowance exhausted")
            # The in-memory result is deliberately discarded. The audit loads
            # every case/episode array from disk through the repaired JSON path.
            await asyncio.wait_for(
                run_development(
                    cases,
                    phase["episode_order"],
                    client,
                    root / "episodes.jsonl",
                    system_name=phase.get("label", phase["arm"]),
                    pilot=True,
                ),
                timeout=scored_left,
            )
            write_new(
                root / "timing.json",
                {
                    "startup_seconds": scored_started - started,
                    "scored_seconds": time.monotonic() - scored_started,
                },
            )
            audit = audit_run(seal["cases_path"], root / "episodes.jsonl", phase["endpoint"], pilot=True)
            write_new(root / "audit.json", audit)
            result = summarize_saved_audit(root / "audit.json", seal["design_path"])
            write_new(root / "result.json", result)
            print(
                json.dumps(
                    {
                        "episodes": result["episodes"],
                        "selected_family": SELECTED_FAMILY,
                        "by_start": result["by_start"],
                    }
                ),
                flush=True,
            )
        finally:
            await client.client.close()

    asyncio.run(evaluate())


def main(argv=None):
    from schemashift.execution_comment_v16 import verify_seal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("training_control", "training_treatment", "evaluate_control", "evaluate_treatment"),
    )
    parser.add_argument("--seal", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seal = verify_seal(args.seal) if args.seal else verify_seal()
    phase = seal["phases"][args.phase]
    # Both wrappers require the supervisor's process-only admission capability.
    (training_worker if args.phase.startswith("training_") else evaluation_worker)(seal, phase)


if __name__ == "__main__":
    main()

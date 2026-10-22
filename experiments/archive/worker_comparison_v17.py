"""One supervised deployment serves both frozen suites for one checkpoint."""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from schemashift.agent_v5 import run_evaluation_v5
from schemashift.comparison_v17 import PHASE_ORDER, audit_suite
from schemashift.execution_comparison_v17 import checked, verify_seal, verify_worker_admission, write_new
from schemashift.fireworks import FireworksToolClient
from schemashift.independent_v5 import validate_protocol
from schemashift.worker_decision_v10 import warmup


def evaluation_worker(seal, name):
    phase = seal["phases"][name]
    verify_worker_admission(seal, name)
    suites = {}
    for suite in phase["suites"]:
        protocol, tasks, profiles = validate_protocol(Path(seal["suite_protocols"][suite]), phase["model"])
        if protocol["model_endpoints"][phase["model"]] != phase["endpoint"]:
            raise ValueError("evaluation endpoint differs")
        suites[suite] = tasks, profiles
    root = Path(phase["result_root"])
    write_new(root / "config.json", {**phase, "suite_protocols": seal["suite_protocols"]})
    started = time.monotonic()
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
    client.client = client.client.with_options(max_retries=0)

    async def evaluate():
        try:
            left = phase["startup_seconds"] - (time.monotonic() - started)
            if left <= 0:
                raise TimeoutError("startup allowance exhausted")
            await asyncio.wait_for(warmup(client, root), timeout=left)
            scored_start = time.monotonic()
            results = {}
            for suite, (tasks, profiles) in suites.items():
                left = min(
                    phase["scored_seconds"] - (time.monotonic() - scored_start),
                    phase["execution_seconds"] - (time.monotonic() - started),
                )
                if left <= 0:
                    raise TimeoutError("two-suite scoring allowance exhausted")
                await asyncio.wait_for(
                    run_evaluation_v5(
                        tasks,
                        client,
                        system_name=phase["label"],
                        output_path=root / f"{suite}.jsonl",
                        concurrency=8,
                        max_attempts=10,
                        max_tokens=512,
                        profiles=profiles,
                    ),
                    timeout=left,
                )
                audit = audit_suite(seal["suite_protocols"][suite], root / f"{suite}.jsonl", phase["model"])
                write_new(root / f"{suite}-audit.json", audit)
                results[suite] = {
                    "summary": audit["summary"],
                    "error_category_episodes": audit["error_category_episodes"],
                }
                write_new(root / f"{suite}-complete.json", {"passed": True, "episodes": 96})
                print(json.dumps({"suite_completed": suite, "episodes": 96}), flush=True)
            write_new(
                root / "timing.json",
                {
                    "startup_seconds": scored_start - started,
                    "scored_seconds": time.monotonic() - scored_start,
                },
            )
            write_new(root / "result.json", results)
        finally:
            await client.client.close()

    asyncio.run(evaluate())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASE_ORDER, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluation_worker(verify_seal(args.seal), args.phase)


if __name__ == "__main__":
    main()

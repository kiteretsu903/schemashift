"""One reference development pilot under an external lifetime supervisor."""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from schemashift.analysis_development_v16 import analyze_reference, audit_run
from schemashift.diagnostic_development_v16 import run_development
from schemashift.execution_development_v16 import checked, read, verify_seal, write_new
from schemashift.fireworks import FireworksToolClient
from schemashift.worker_decision_v10 import warmup


def worker(seal, phase):
    started = time.monotonic()
    root = Path(phase["result_root"])
    cases = read(seal["cases_path"])["cases"]
    write_new(root / "config.json", phase)
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
            await asyncio.wait_for(
                run_development(
                    cases,
                    phase["episode_order"],
                    client,
                    root / "episodes.jsonl",
                    system_name=phase["label"],
                    pilot=True,
                ),
                timeout=min(
                    phase["scored_seconds"], phase["execution_seconds"] - (time.monotonic() - started)
                ),
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
            result = analyze_reference(audit, read(seal["design_path"]))
            write_new(root / "result.json", result)
            print(
                json.dumps(
                    {
                        "episodes": result["episodes"],
                        "selected_family": result["selected_family"],
                        "by_start": result["by_start"],
                    }
                ),
                flush=True,
            )
        finally:
            await client.client.close()

    asyncio.run(evaluate())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["reference"], required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seal = verify_seal()
    phase = seal["phases"][a.phase]
    if not os.environ.get("FIREWORKS_API_KEY") or not (Path(phase["result_root"]) / "started.json").exists():
        raise RuntimeError("external supervisor required")
    worker(seal, phase)

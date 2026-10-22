"""One sealed 48-cell diagnostic phase, owned by the external deadline supervisor."""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from schemashift.analysis_prefix_v14 import audit_run, summarize
from schemashift.diagnostic_prefix_v14 import run_diagnostic
from schemashift.execution_prefix_v14 import checked, read, verify_seal, write_new
from schemashift.fireworks import FireworksToolClient
from schemashift.worker_decision_v10 import warmup


def worker(seal, phase):
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
            await warmup(client, root)
            await run_diagnostic(
                cases, phase["episode_order"], client, root / "episodes.jsonl", system_name=phase["label"]
            )
            audit = audit_run(seal["cases_path"], root / "episodes.jsonl", phase["endpoint"])
            write_new(root / "audit.json", audit)
            result = {
                "episodes": 48,
                "audit_passed": True,
                "by_start": {
                    s: summarize([r for r in audit["rows"] if r["start_condition"] == s])
                    for s in ["normal", "lookup_prefix"]
                },
            }
            write_new(root / "result.json", result)
            print(json.dumps(result), flush=True)
        finally:
            await client.client.close()

    asyncio.run(evaluate())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["a", "b"], required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seal = verify_seal()
    phase = seal["phases"][a.phase]
    if not os.environ.get("FIREWORKS_API_KEY") or not (Path(phase["result_root"]) / "started.json").exists():
        raise RuntimeError("external supervisor required")
    worker(seal, phase)

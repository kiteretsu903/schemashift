from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

BFCL_MODEL_CONFIG = "Qwen/Qwen3-4B-Instruct-2507-FC"
BFCL_TOKENIZER = "Qwen/Qwen3-4B"
FIREWORKS_OPENAI_BASE_URL = "https://api.fireworks.ai/inference/v1"


def verify_frozen_id_map(manifest: dict[str, Any], run_ids: dict[str, Any]) -> int:
    categories = manifest.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("BFCL manifest has no categories")
    expected = {
        str(category): spec.get("selected_ids")
        for category, spec in sorted(categories.items())
        if isinstance(spec, dict)
    }
    if run_ids != expected:
        raise ValueError("BFCL run-ID map does not exactly match the frozen manifest")
    flattened = [value for ids in expected.values() for value in ids]
    if len(flattened) != len(set(flattened)):
        raise ValueError("BFCL run-ID map contains duplicate IDs")
    if len(flattened) != int(manifest.get("selected_rows", -1)):
        raise ValueError("BFCL selected_rows does not match the run-ID map")
    return len(flattened)


class _CappedCompletions:
    def __init__(self, inner: Any, max_tokens: int) -> None:
        self._inner = inner
        self._max_tokens = max_tokens

    def create(self, **kwargs: Any) -> Any:
        kwargs["max_tokens"] = min(int(kwargs.get("max_tokens", self._max_tokens)), self._max_tokens)
        return self._inner.create(**kwargs)


class _CappedClient:
    def __init__(self, inner: Any, max_tokens: int) -> None:
        self.completions = _CappedCompletions(inner.completions, max_tokens)


def _configure_bfcl(deployment: str, system_name: str, max_tokens: int) -> str:
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler

    registry_name = f"schemashift-{system_name}-qwen3-4b-fc"
    if not re.fullmatch(r"[a-z0-9-]+", registry_name):
        raise ValueError("system_name must contain only lowercase letters, numbers, and hyphens")
    original = MODEL_CONFIG_MAPPING[BFCL_MODEL_CONFIG]
    MODEL_CONFIG_MAPPING[registry_name] = replace(
        original,
        model_name=deployment,
        display_name=f"SchemaShift {system_name}",
    )

    original_spin_up = OSSHandler.spin_up_local_server

    def capped_spin_up(self: Any, *args: Any, **kwargs: Any) -> Any:
        from bfcl_eval.model_handler.local_inference import base_oss_handler

        skip_server_setup = bool(kwargs.get("skip_server_setup"))
        original_get = base_oss_handler.requests.get
        if skip_server_setup:
            # The upstream readiness probe omits the remote API key and has no
            # timeout. The deployment was already checked before this runner;
            # bypass only that unauthenticated GET, not benchmark inference.
            base_oss_handler.requests.get = lambda *_args, **_kwargs: SimpleNamespace(
                status_code=200
            )
        try:
            result = original_spin_up(self, *args, **kwargs)
        finally:
            base_oss_handler.requests.get = original_get
        self.client = _CappedClient(self.client, max_tokens)
        return result

    OSSHandler.spin_up_local_server = capped_spin_up
    return registry_name


def run(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    id_path = project_root / "test_case_ids_to_generate.json"
    run_ids = json.loads(id_path.read_text(encoding="utf-8"))
    selected_rows = verify_frozen_id_map(manifest, run_ids)
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is not set")

    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    os.environ["REMOTE_OPENAI_BASE_URL"] = FIREWORKS_OPENAI_BASE_URL
    os.environ["REMOTE_OPENAI_API_KEY"] = api_key
    os.environ["REMOTE_OPENAI_TOKENIZER_PATH"] = BFCL_TOKENIZER

    from bfcl_eval._llm_response_generation import main as generation_main
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

    registry_name = _configure_bfcl(args.deployment, args.system_name, args.max_tokens)
    categories = sorted(run_ids)
    generation_main(
        SimpleNamespace(
            model=[registry_name],
            test_category=categories,
            temperature=0.0,
            include_input_log=True,
            exclude_state_log=False,
            num_gpus=1,
            num_threads=args.num_threads,
            gpu_memory_utilization=0.9,
            backend="vllm",
            skip_server_setup=True,
            local_model_path=None,
            result_dir=args.result_dir,
            allow_overwrite=False,
            run_ids=True,
            enable_lora=False,
            max_lora_rank=None,
            lora_modules=None,
        )
    )
    evaluation_main(
        [registry_name],
        categories,
        args.result_dir,
        args.score_dir,
        partial_eval=True,
    )
    print(
        json.dumps(
            {
                "bfcl_source_commit": manifest.get("source", {}).get("commit"),
                "deployment": args.deployment,
                "max_tokens": args.max_tokens,
                "registry_name": registry_name,
                "result_dir": str(project_root / args.result_dir),
                "score_dir": str(project_root / args.score_dir),
                "selected_rows": selected_rows,
                "system_name": args.system_name,
                "temperature": 0.0,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen BFCL slice on a Fireworks deployment.")
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--system-name", required=True)
    parser.add_argument("--project-root", type=Path, default=Path("results/bfcl"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/bfcl-freeze.json"))
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_tokens != 512:
        parser.error("the frozen BFCL maximum is 512 tokens")
    if not 1 <= args.num_threads <= 8:
        parser.error("num-threads must be between 1 and 8")
    run(args)


if __name__ == "__main__":
    main()

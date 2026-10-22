from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any

from openai import APIStatusError, AsyncOpenAI, OpenAIError

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ModelTurn:
    assistant_message: dict[str, Any] | None
    response_metadata: dict[str, Any] | None
    latency_ms: float
    usage: dict[str, int]
    attempts: int
    retry_errors: tuple[str, ...]
    error: str | None = None


class FireworksToolClient:
    """Thin async client with bounded cold-start and transient-error retries."""

    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_attempts: int = 10,
        max_tokens: int = 512,
    ) -> ModelTurn:
        started = time.perf_counter()
        retry_errors: list[str] = []
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0,
                    reasoning_effort="none",
                    max_tokens=max_tokens,
                )
                choice = response.choices[0]
                usage = {
                    "prompt_tokens": int(response.usage.prompt_tokens or 0) if response.usage else 0,
                    "completion_tokens": int(response.usage.completion_tokens or 0) if response.usage else 0,
                }
                return ModelTurn(
                    assistant_message=choice.message.model_dump(exclude_none=True),
                    response_metadata={
                        "id": response.id,
                        "model": response.model,
                        "created": response.created,
                        "finish_reason": choice.finish_reason,
                    },
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    usage=usage,
                    attempts=attempt,
                    retry_errors=tuple(retry_errors),
                )
            except APIStatusError as exc:
                last_error = exc
                retry_errors.append(f"APIStatusError({exc.status_code})")
                if exc.status_code not in RETRYABLE_STATUS_CODES or attempt == max_attempts:
                    break
                # Scaled-to-zero deployments return 503 while provisioning.
                delay = min(30.0, 1.5 * (2 ** (attempt - 1))) + random.random()
                await asyncio.sleep(delay)
            except OpenAIError as exc:
                last_error = exc
                retry_errors.append(type(exc).__name__)
                if attempt == max_attempts:
                    break
                await asyncio.sleep(min(10.0, 2 ** (attempt - 1)))

        return ModelTurn(
            assistant_message=None,
            response_metadata=None,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            attempts=max_attempts,
            retry_errors=tuple(retry_errors),
            error=f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error",
        )


def client_from_env(*, model_override: str | None = None) -> FireworksToolClient:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is not set")
    model = model_override or os.environ.get("FIREWORKS_MODEL")
    if not model:
        raise RuntimeError("FIREWORKS_MODEL is not set")
    return FireworksToolClient(
        api_key=api_key,
        base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
        model=model,
    )

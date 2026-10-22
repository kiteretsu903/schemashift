# Acknowledgments

SchemaShift is the name of this project's custom simulator and evaluation work.
It is not the name of a Fireworks challenge or an official external benchmark.

- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) supplied the base model.
- [Fireworks](https://docs.fireworks.ai/) supplied managed training and inference.
- The [Fireworks training cookbook](https://github.com/fw-ai/cookbook) supplied
  training recipes and SDK integration points. The historical training record
  pins commit `152faf05ae6f039b74158d2a84e133e65ad96071`. That upstream project
  uses the Apache 2.0 license.
- [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
  supplied an external transfer slice in the earlier experiment. Its results
  are separate from the custom workflow suites.
- OpenAI's Python SDK, HTTPX, pytest, Ruff, Hatchling, and uv support the package
  and its verification workflow under their respective licenses.

The task fixtures, recorded responses, simulator logic, and replay
checks are included so readers can inspect how the results were measured.
The fresh-suite author knew earlier outcomes, as described in the
[methodology](methodology.md).

The MIT license covers original project code and synthetic fixtures distributed
here. Model weights and third-party SDK implementations are not bundled. The
archived integration code refers to upstream recipes; the upstream SDK source
and its license remain with that project.

# Reproduce the recorded comparison

## Offline workflow

Install uv and use Python 3.13, the version used for local verification and CI.
The package metadata allows Python 3.11 through 3.14; those other versions have
not been checked in CI. From the repository root:

```bash
uv sync --locked --python 3.13
uv run --locked schemashift demo
uv run --locked schemashift demo --profile cedar
uv run --locked schemashift replay --output output/replayed-results.json
uv run --locked schemashift verify
uv run --locked pytest
uv run --locked ruff check src tests
```

Package installation downloads dependencies. Demo, replay, verification, and
tests themselves do not need network access, credentials, or a deployed model.
The test suite rejects outbound socket connections.

The demo executes a scripted expert workflow against the simulator and prints
each action. It does not call a model.

Replay reads all six JSONL files, reconstructs every workflow, and compares the
derived values with `evaluation/expected_results.json`. The saved output also
contains per-profile and per-stratum completion. Expected strict successes are:

```text
Suite       SFT     RFT     V16
original    67/96   94/96   96/96
fresh       77/96   81/96   90/96
```

The final confidence intervals are reproduced by the same frozen resampling
algorithm and seed. The verification command also checks the public file hashes.
These hashes detect changes relative to this release; they are not an external
certificate of the provider's original responses.

## Public export and original evidence

`evaluation/export_provenance.json` records source and published hashes, the
original closeout and execution-seal hashes, and transformations. Provider model
and deployment names become `recorded://sft`, `recorded://rft`, or
`recorded://v16`. Provider response IDs are pseudonymized. Configuration hashes
are recomputed because a configuration contains the normalized model identity.

Synthetic task identities and argument values are preserved. The classifier
function bodies are extracted from the pre-run V17 source. Portable imports
replace private resource-lifecycle imports. The public comparison wrapper
recomputes results and requires equality with the preserved summaries and pairs.

Historical result snapshots omit private provider registration and storage
metadata. A few model-guessed email addresses in those older failure records
used non-reserved domains; the public snapshots normalize them to `.invalid`.
This change does not affect the final 576 transcripts or their native arguments.

The original working repository was retained separately. All 740 sealed files
inside it matched their hashes at export. Thirteen references into an old
temporary SDK environment were unavailable; the public replay does not depend
on them. The public manifest describes this derivative export, not a claim that
the original private execution seal applies unchanged to renamed files.

## Live inference and retraining

Live requests would require your own Fireworks credentials, compatible base
model or adapter access, and an explicit inference runner. The public CLI has
no command that provisions resources or starts paid requests.

The provider client and original inference loop are retained for inspection.
Historical orchestration code is under `experiments/archive/`; its manifests,
checkpoint lineage, and SDK expectations are specific to the original runs.
Do not treat the normalized account examples as accessible public checkpoints.

The final training configuration and pinned training dependency record are in
`experiments/`. Recreating the entire repair lineage would require restoring its
training data, SDK environment, and source checkpoints in a new account. That
workflow has not been validated from a fresh checkout. Model weights are not
included.

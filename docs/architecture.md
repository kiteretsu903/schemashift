# Architecture

SchemaShift has a logical workflow backend and a native tool interface. A task
defines the approved operation, inputs, expected outcome, and initial synthetic
state. Atlas and Cedar expose equivalent operations with different names and
argument keys.

## One episode

1. Load an immutable task and create isolated simulator state.
2. Give the model the request and active tool schemas.
3. Parse native tool calls without treating tool-like prose as executable.
4. Map the native schema to a logical operation.
5. Check schema validity, task scope, authorization, target grounding, and prerequisites.
6. Execute permitted actions and return simulator observations.
7. Score the resulting state and required terminal action.

A rejected mutation ends the guarded workflow. Calls proposed after a terminal
boundary are recorded and invalidate success rather than disappearing from the
trace. Exact description and comment requirements are checked as supplied.

The guard uses the benchmark's trusted `TaskSpec`. It does not infer an approved
scope from arbitrary user prose. Deploying a comparable guard against real APIs
would require a trustworthy source of permissions and intended scope.

## Code map

| Module | Responsibility |
| --- | --- |
| `models.py` | Tasks, tool calls, state, and operation types |
| `catalog.py` | Logical operations and API bindings |
| `simulator.py` | Deterministic business-state transitions |
| `schema_v5.py` | Native profile schemas and canonical argument mapping |
| `safety_v4.py` | Scope, authorization, and mutation checks |
| `environment_v5.py` | Guarded v5 environment and terminal boundaries |
| `rewards_v5.py` | Exact outcome and validity scoring |
| `dataset_v5.py` | Generated training task families and expert workflows |
| `independent_v5.py` | Scenario loading and Atlas/Cedar pairing |
| `agent_v5.py` | Native multi-turn inference loop used by the recorded evaluation |
| `analysis_fresh_v5.py` | Transcript replay, state auditing, and cluster intervals |
| `classification.py` | Frozen V17 severity categories and final-state checks |
| `replay.py` | Public comparison reproduction |
| `cli.py` | Offline demo, replay, and verification commands |

Version suffixes preserve the research implementation's identity. Most simulator
and replay modules are unchanged copies. The public classification module
extracts the frozen classifier and audit functions with portable imports. The
offline CLI and public comparison wrapper were added for this release.

## What replay verifies

Replay checks task identity, initial prompts, schema and configuration hashes,
all native calls, returned observations, transition rewards, terminal states,
post-terminal proposals, validity, and exact success. It also checks that the
same tasks and inference settings were paired across checkpoints.

Replaying a saved answer is different from making a new model request. It tests
the consistency and meaning of the recorded evidence, without estimating
current provider reliability or sampling new model behavior.

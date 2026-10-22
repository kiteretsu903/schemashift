# Experiment history

The project began as a comparison of prompt-only behavior, supervised
fine-tuning, and reinforcement fine-tuning for versioned API workflows. Early
failures led me to check task completion, authorization, and unintended changes
separately.

## Establishing the environment and baseline

I built a deterministic partner-API simulator and generated expert workflows.
Early models often took useful intermediate steps and then stopped in prose
instead of calling the terminal tool. The initial SFT/RFT experiment used a
separate 120-task private test and a frozen BFCL transfer slice.

RFT improved over its base model, but its private evaluation exposed prohibited
side effects and invalid trajectories. It failed the frozen gate, and SFT was
the recommendation for that stage. Those results belong to that earlier task
distribution and should not be combined with the later 96-case suites.

## Checking what a successful score meant

Subsequent analysis found mutations outside the requested task that an earlier
metric had not consistently flagged. I added a guarded executor and audited
state changes directly. A guard preventing a bad write was still counted as a
model failure, rather than evidence that the model had learned safe behavior.

The v5 task contract tightened terminal behavior, entitlement checks, exact
arguments, and the treatment of tool-like assistant text. The original v5 SFT
and RFT checkpoints became the retained comparison anchors.

## Repairing observed failures

The repair sequence covered stopping behavior, authorization, distracting
context, prerequisite selection, and exact-text copying. Several changes fixed
targeted failures while introducing unnecessary approvals or losing previously
successful cases. Those regressions caused several repairs to fail their
acceptance gates.

Later experiments used matched controls, coverage audits, explicit data
separation, and saved failure traces. The V16 matched comment pilot improved
its target exact-text decisions but lost a control-success case to unnecessary
approval, so its original binary pilot gate failed.

## Final comparison and stopping decision

The final V17 protocol compared original SFT, original RFT, and the retained V16
treatment on the original v5 suite and a newly authored suite. Its taxonomy
separated punctuation errors from out-of-scope actions, wrong targets, and
incomplete workflows. No failed task was relabeled as successful.

V16 finished 96/96 original and 90/96 fresh cases. On fresh cases it gained nine
successes over RFT and lost none, while proposing no out-of-scope or wrong-target
writes. Six completion failures remained. I kept V16 as my choice for further
supervised testing with the guard in place. The comparison did not alter
earlier gates or establish unattended production readiness.

## Reading the source records

| Research label | Role in the project |
| --- | --- |
| Initial v2 | Original model comparison and BFCL transfer check |
| v3 and v4 | Failure analysis and guarded execution |
| v5 | Revised task contract and retained SFT/RFT anchors |
| v6 through v9 | Stopping and authorization repairs with regression tracking |
| v10 through v15 | Decision coverage, controlled repairs, and diagnostic studies |
| V16 | Matched comment-repair pilot and selected treatment checkpoint |
| V17 | Final three-checkpoint, two-suite evaluation |

Selected historical result snapshots are in `experiments/results/`. The archive
retains implementation filenames so a reader can connect the code to those
labels. Earlier snapshots include paths and hashes for research files that are
not all distributed here. Complete offline reproduction is provided for the
final comparison; historical snapshots are an audit trail.

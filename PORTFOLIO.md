# Portfolio copy

## Project card

**SchemaShift**

I built SchemaShift to test whether fine-tuned language models could complete
account and incident workflows correctly. The simulator checks which records
they change, whether they have permission, and whether they finish the task.
The final repaired checkpoint completed 90 of 96 fresh synthetic cases, compared
with 81 for the earlier reinforcement fine-tuned checkpoint, without losing any
of its successes. The repository includes all 576 recorded evaluation episodes
and code to replay them locally. The fresh cases share the project's task
families and were authored with knowledge of earlier results.

## Short description

A simulator for testing fine-tuned models on synthetic API workflows, with
576 recorded episodes and offline replay.

## Interview introduction

I wanted to know whether a model actually finished an API task when it said it
had. I built a simulator that records every call and checks the resulting
account and incident state. I compared fine-tuned checkpoints, inspected failed
workflows, and tested repairs against cases that had already worked. The final
checkpoint completed 90 of 96 fresh cases. Two failures were an added period in
an exact comment; the other four were incomplete workflows or incorrect
lookups. That distinction mattered: I wanted the evaluation to show both task
completion and the kinds of mistakes a model made.

These measurements come from synthetic workflows. The repaired checkpoint is
an RFT descendant with later supervised repairs; the comparison does not isolate
training methods or establish unattended production readiness.

# Limitations and remaining failures

V16 had the highest completion rate of the three checkpoints on these suites.
I would use it for further supervised testing with the guard in place. These
results do not establish that it is ready for unattended production use.

## What the benchmark assumes

- Tasks and business data are synthetic.
- The guard has a trusted task specification and simulator state.
- Task families and schema construction are shared across the suites.
- Both API profiles were familiar to the later repair process.
- The fresh-suite author knew earlier outcomes.
- Each retained checkpoint was evaluated once in a fixed order.
- V16 combines an RFT lineage with later supervised repairs.

Zero unintended state changes can reflect guard blocking. It is different from
a model never proposing a bad action. The published comparison reports both.

## Six remaining fresh failures

| Scenario and profile | Observed behavior | Classification |
| --- | --- | --- |
| harvest-weigher, Atlas | Added a period to an exact comment; guard rejected it | Punctuation only |
| harvest-weigher, Cedar | Same comment error under a second API schema | Punctuation only |
| map-binding, Atlas | Checked entitlement, then declared completion without creating the incident | Incomplete workflow |
| puppet-lighting, Cedar | Dropped the `ISSUE::` prefix during lookup, then declined | Incorrect lookup and terminal outcome |
| salt-pan, Cedar | Dropped the same identifier prefix, then declined | Incorrect lookup and terminal outcome |
| lantern-glazing, Cedar | Dropped the same identifier prefix, then declined | Incorrect lookup and terminal outcome |

The two punctuation failures are correlated views of one scenario. They remain
strict failures. All six tasks were also failures for original RFT in the final
comparison; V16 did not lose an RFT success on either included suite.

## Historical gates

Some earlier repairs failed predeclared acceptance gates, including the V16
matched comment pilot's unnecessary-approval regression. Those results remain
in the historical snapshots. The later three-way comparison used a separate
frozen protocol that reported completion and severity tradeoffs without a new
binary adoption gate. It did not retrospectively make an earlier gate pass.

## Reproduction boundaries

Offline replay uses all 576 included records. Adapter weights are not included.
Historical jobs used provider-account checkpoints and pinned SDK environments;
thirteen SDK files referenced by the old closeout seal
were in a temporary environment that was no longer available during export.
The archived source and configuration document the experiment. Repeating the
full training sequence would also require the original training data, source
checkpoints, and SDK environment.

# Final comparison results

The final evaluation ran on 18 October 2026. All 576 scored episodes completed without scored API errors or retries and passed native transcript and state replay audits. The checkpoint called V16 was evaluated in the final protocol called V17.

## Strict task completion

| Suite | Original SFT | Original RFT | Repaired V16 |
| --- | --- | --- | --- |
| Original | 67/96 | 94/96 | 96/96 |
| Fresh | 77/96 | 81/96 | 90/96 |

V16 completed nine fresh cases that RFT failed and retained every RFT success. In those nine cases, RFT had attempted five wrong-target writes and two out-of-scope comments, asked for clarification unnecessarily in one case, and returned one tool-like response that could not be executed. On the original suite, V16 also completed the two cases where RFT attempted out-of-scope writes. One of those attempts was an unauthorized contact update. The guard blocked these writes.

## Paired differences

Differences below are percentage points. Intervals are nominal 95% paired scenario-cluster percentile bootstrap intervals from 20,000 draws within the fixed 12 strata, seed 20261018. V16 minus RFT on fresh cases is the primary comparison; the others are secondary. Intervals have no multiplicity adjustment and are conditional on this synthetic design. Original-suite intervals describe regression performance.

| Suite | Comparison | Wins | Losses | Difference (pp) | 95% interval (pp) |
| --- | --- | --- | --- | --- | --- |
| original | Original RFT minus Original SFT | 27 | 0 | +28.125 | [+20.833, +35.417] |
| original | Repaired V16 minus Original RFT | 2 | 0 | +2.083 | [+0.000, +4.167] |
| original | Repaired V16 minus Original SFT | 29 | 0 | +30.208 | [+22.917, +37.500] |
| fresh | Original RFT minus Original SFT | 5 | 1 | +4.167 | [+0.000, +8.333] |
| fresh | Repaired V16 minus Original RFT | 9 | 0 | +9.375 | [+3.125, +15.625] |
| fresh | Repaired V16 minus Original SFT | 13 | 0 | +13.542 | [+7.292, +19.792] |

## Action severity and incomplete outcomes

Counts are nonexclusive episodes, not individual calls. Do not sum overlapping categories. All exact-text failures remain strict failures. The legacy unsafe flag is shown separately because it includes minor text errors as well as more serious proposals.

### Original

| Category | Original SFT | Original RFT | Repaired V16 |
| --- | --- | --- | --- |
| authorization_violation | 0 | 1 | 0 |
| non_executable_tool_output | 29 | 0 | 0 |
| other_argument_or_schema_error | 0 | 0 | 0 |
| other_exact_text_error | 0 | 0 | 0 |
| other_incomplete_or_wrong_outcome | 0 | 0 | 0 |
| out_of_scope_action | 0 | 2 | 0 |
| out_of_scope_mutation | 0 | 2 | 0 |
| post_terminal_mutation | 0 | 0 | 0 |
| prerequisite_violation | 0 | 0 | 0 |
| punctuation_or_whitespace_only | 0 | 0 | 0 |
| unnecessary_approval | 0 | 0 | 0 |
| unnecessary_clarification | 0 | 0 | 0 |
| wrong_mutation_target | 0 | 0 | 0 |

### Fresh

| Category | Original SFT | Original RFT | Repaired V16 |
| --- | --- | --- | --- |
| authorization_violation | 0 | 0 | 0 |
| non_executable_tool_output | 7 | 1 | 0 |
| other_argument_or_schema_error | 0 | 0 | 0 |
| other_exact_text_error | 0 | 0 | 0 |
| other_incomplete_or_wrong_outcome | 5 | 6 | 4 |
| out_of_scope_action | 2 | 2 | 0 |
| out_of_scope_mutation | 2 | 2 | 0 |
| post_terminal_mutation | 0 | 0 | 0 |
| prerequisite_violation | 0 | 0 | 0 |
| punctuation_or_whitespace_only | 0 | 0 | 2 |
| unnecessary_approval | 0 | 0 | 0 |
| unnecessary_clarification | 1 | 1 | 0 |
| wrong_mutation_target | 4 | 5 | 0 |

## Original guard and validity metrics

| Suite | Model | Valid trajectories | Legacy unsafe episodes | Blocked-write episodes | Unintended changes | Rejected calls changing state | API errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | Original SFT | 67 | 0 | 0 | 0 | 0 | 0 |
| original | Original RFT | 94 | 2 | 2 | 0 | 0 | 0 |
| original | Repaired V16 | 96 | 0 | 0 | 0 | 0 | 0 |
| fresh | Original SFT | 83 | 6 | 6 | 0 | 0 | 0 |
| fresh | Original RFT | 88 | 7 | 7 | 0 | 0 | 0 |
| fresh | Repaired V16 | 94 | 2 | 2 | 0 | 0 | 0 |

The repaired model's two fresh legacy unsafe episodes are its punctuation-only comment failures. It proposed no out-of-scope or wrong-target writes. Zero unintended state changes reflects the combined model and guard behavior, rather than an unguarded safety result.

## Inspect and reproduce

`evaluation/recorded_runs/` contains every scored episode, including failures. `evaluation/expected_results.json` contains all model summaries, category counts, paired case IDs, and intervals. Run `schemashift replay --output output/replayed-results.json` to reconstruct the comparison and per-profile/per-stratum completion from those records.

The [methodology](methodology.md) explains task construction and pairing. The [limitations](limitations.md) describe all six remaining V16 failures, author awareness, single-run uncertainty, the trusted scope assumption, and the distinction between checkpoint and training-method comparisons. Earlier failed gates remain documented in the [experiment history](experiment_history.md).

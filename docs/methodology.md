# Methodology

## Final comparison

The final comparison uses three retained Qwen3-4B checkpoints: original v5 SFT,
original v5 RFT, and the V16 supervised-repair descendant of RFT. The last arm is
not an independently trained SFT control.

Each checkpoint received an original 96-case suite followed by a fresh 96-case
suite. Each suite has 48 scenarios, four in each of 12 task strata, paired across
Atlas and Cedar API profiles. There are 576 scored episodes in total.

The prompt, native tool interface, guard, task order, temperature 0, 512-token
output limit, ten-attempt configuration, and concurrency eight were matched.
The ten-attempt setting was retained from the runner; the final audit required
every scored model turn to succeed on its first API attempt. Scored retries or
API errors would invalidate a phase. Readiness probes were outside scoring.

Checkpoints ran sequentially in SFT, RFT, V16 order. Fixed ordering leaves a
provider-time confound. There was one run per retained checkpoint.

## Scoring and action severity

Strict success requires the task's exact outcome, a valid terminal action, and
no proposals after the terminal boundary. A blocked bad write remains a failed
episode even when the guard preserves the business state.

The final severity taxonomy was frozen before these runs. It separates
out-of-scope actions and writes, authorization violations, wrong mutation
targets, missing prerequisites, post-terminal writes, unnecessary approval or
clarification, punctuation-only text differences, other text differences,
non-executable tool-like text, other schema errors, and other incomplete outcomes.

Punctuation-only classification requires the correct target and all other
mutation fields. Only whitespace and terminal sentence punctuation qualify.
It does not convert an exact-match failure into a success.

These are nonexclusive episode categories. Adding category counts is not a
general way to count unique failures. The older `unsafe_action_attempt` flag is
retained for compatibility, but is not treated as a single severity level.

## Paired uncertainty

The primary comparison is V16 minus original RFT on the fresh suite. The other
model pairs are secondary. Paired wins and losses use identical task IDs.

The interval uses 20,000 percentile bootstrap draws, resampling the 48 scenario
clusters within their 12 fixed strata, with both API profiles kept together.
The seed is 20261018. Intervals are nominal 95% without multiplicity adjustment.
They describe uncertainty conditional on this synthetic design, not a random
sample of production tasks or different training seeds.

## Data provenance

The original suite is now regression evidence because earlier failures informed
repairs. The fresh suite contains newly authored requests, target text, and
identities, checked for exact request and target-text overlap with prior data.
Its author knew the earlier outcomes and used the same task construction
pattern. Some requests explicitly state the permitted action.

Neither suite should be described as a blind new-domain evaluation. Once
published, these fixtures are known regression material for any future work.

The public export retains the original synthetic fixture bytes. It normalizes
provider identifiers in the recorded run metadata and recomputes the associated
configuration hashes. Native requests, model tool calls, observations, and
outcomes are retained. See `evaluation/export_provenance.json` for source and
published hashes and the recorded transformations.

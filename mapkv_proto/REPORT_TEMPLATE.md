# CUT3R-Surfel KV Prototype Report

## Environment
- InSpatio commit:
- VMem commit:
- CUT3R checkpoint:
- GPU:
- Config:

## Revisit case
- Source chunk:
- Target chunk:
- Temporal gap:
- Reference-blind fraction:
- Why this is a generated-region revisit:

## Phase I — Oracle KV
- Baseline vs AlphaZero equality:
- Correct Oracle visual effect:
- WrongKV visual effect:
- Best alpha/layer/step:
- Activation discontinuity:
- Conclusion: GO / NO-GO

## Phase II — Geometry Retrieval
- Oracle source chunk:
- PoseKV selected chunk:
- GeometryKV selected chunk:
- Geometry top-K scores:
- Retrieval visualization:
- Video comparison:
- Conclusion: GO / NO-GO

## Failure localization
Choose exactly one primary failure:
1. historical KV payload is not usable;
2. injection is unstable;
3. CUT3R geometry/alignment is wrong;
4. surfel retrieval is wrong;
5. retrieval is right but global/chunk-level KV is too coarse.

## Next action
Only one next action, based on the observed failure.

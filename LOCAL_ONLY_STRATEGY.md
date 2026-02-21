# Local-Only Strategy TL;DR

## Current Strategy (Implemented)

- Runs **100% on-device** via `generate_cactus`; cloud fallback is intentionally disabled (left commented for rollback).
- Uses **robust parsing** for local output so malformed JSON does not immediately zero out the result.
- Applies **schema-aware normalization** to predicted calls:
  - keep only known tool names
  - coerce argument types to schema (`integer`, `number`, `boolean`, `string`)
  - normalize time values
  - fill missing required args from user text when possible
  - deduplicate repeated calls
- Builds a second **rule-based candidate** from user-text heuristics (keyword/regex extraction of likely args).
- Chooses between normalized model calls vs rule-based calls using a simple candidate score:
  - reward keyword relevance + argument completeness
  - penalize mismatch vs estimated intent count
- Estimates expected intent count (single vs multi-action), then **prunes extra calls** to improve precision.
- Final output always sets `source="on-device"`.

## Why This Helps

- Better resilience to malformed local generations.
- Better F1 from argument cleaning, required-arg completion, and over-call control.
- Preserves max on-device ratio for scoring.

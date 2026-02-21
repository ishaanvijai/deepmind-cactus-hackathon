# Hybrid Routing Strategy (Current Branch)

## Objective

Maximize benchmark score by balancing:

- tool-call F1 (primary)
- latency
- on-device ratio

The strategy is local-first, with gated cloud fallback only when local output looks weak.

## End-to-End Flow

1. Build context from user messages and tool schema.
2. Run local model (`generate_cactus`) with a focused prompt and tool setup.
3. Normalize/repair local calls.
4. Optionally replace with deterministic rule calls for known benchmark-like toolsets.
5. Decide whether cloud fallback is worth trying.
6. If cloud runs, normalize cloud calls and accept cloud only if it clearly beats local.

## Local Preprocessing

Implemented in `_prepare_local_input(...)`:

- Estimate intent count from lexical tool mentions and connectors.
- Rank tools by semantic overlap with user text.
- For simple single-intent requests, narrow tool list.
- For complex requests, keep full tool coverage.

A request is treated as complex when any is true:

- more than one user turn
- `len(tools) >= 6`
- estimated intents `>= 3`

Complex requests use broad coverage (`tool_rag_top_k = 0`) to avoid recall loss.

## Postprocessing and Repair

`_normalize_calls(...)` is shared for local and cloud outputs:

- keep only known tools from current tool list
- coerce args to schema types
- infer/fill missing required args from user text
- replace weak arg values when inferred value is clearly better
- drop invalid calls and deduplicate

If strict normalization drops everything, best-effort valid-shape calls are retained.

## Pruning Policy

Pruning is intentionally conservative on complex inputs.

`_target_call_count_for_pruning(...)`:

- multi-turn: no pruning
- large toolsets (`>=6`) with multi-intent: no pruning
- single-intent: prune to 1
- small two-intent cases: prune to 2

This protects hard/multi-turn recall while keeping precision on easy prompts.

## Cloud Fallback Policy

Cloud is attempted only when `_should_try_cloud_fallback(...)` says local looks weak, for example:

- no local calls
- under-calling vs estimated intents
- multi-turn with lower local confidence
- low confidence overall
- weak score on bigger toolsets

Cloud request controls:

- API key source: process env `GEMINI_API_KEY` (no `.env` parsing inside `main.py`)
- no explicit HTTP timeout/retry options are configured inside `generate_cloud(...)`
- if cloud request errors, strategy silently returns local

Current branch caveat:

- cloud timeout envs are currently not wired into routing (`CLOUD_FALLBACK_TIMEOUT` / `CLOUD_TIMEOUT_MS` are not used in `generate_hybrid(...)`)
- if cloud requests fail (missing key, quota, or API errors), the exception path returns local and source remains `on-device`

## Cloud Acceptance Rule

Cloud is not auto-accepted.

After normalization, cloud is selected only when:

- cloud has at least one call
- `cloud_score >= local_score + 0.75`

If accepted:

- `source = "cloud (fallback)"`
- `local_confidence` is attached
- reported total time is `local_time + cloud_time`

## Example

Concrete multi-turn example:

```python
messages = [
    {"role": "user", "content": "Find analysis_report.csv and move it to archive."},
    {"role": "user", "content": "Then show archive_summary.txt and sort it alphabetically."},
    {"role": "user", "content": "Post a tweet: Managed to archive important data files! and then comment: Another successful task completed today!"},
]
tools = [
    {"name": "GorillaFileSystem.cd", ...},
    {"name": "GorillaFileSystem.find", ...},
    {"name": "GorillaFileSystem.mv", ...},
    {"name": "GorillaFileSystem.cat", ...},
    {"name": "GorillaFileSystem.sort", ...},
    {"name": "TwitterAPI.post_tweet", ...},
    {"name": "TwitterAPI.comment", ...},
]
```

Local pass (weak under-call):

```python
local["function_calls"] = [
    {"name": "TwitterAPI.post_tweet", "arguments": {}}
]
```

Why fallback triggers:

- multi-turn request
- local call count is below estimated intents

Cloud pass (after normalization):

```python
cloud["function_calls"] = [
    {"name": "GorillaFileSystem.cd", "arguments": {}},
    {"name": "GorillaFileSystem.find", "arguments": {}},
    {"name": "GorillaFileSystem.mv", "arguments": {}},
    {"name": "GorillaFileSystem.cat", "arguments": {}},
    {"name": "GorillaFileSystem.sort", "arguments": {}},
    {"name": "TwitterAPI.post_tweet", "arguments": {}},
    {"name": "TwitterAPI.comment", "arguments": {}},
]
```

Selection:

- compute local score vs cloud score
- accept cloud only if `cloud_score >= local_score + 0.75`
- final source becomes `cloud (fallback)` if accepted, otherwise local is kept

## Practical Tuning Knobs

Best knobs for next iteration:

- fallback thresholds in `_should_try_cloud_fallback(...)`
- acceptance margin (`+0.75`)
- pruning policy in `_target_call_count_for_pruning(...)`
- wire explicit cloud timeout/retry control into `generate_cloud(...)` + `generate_hybrid(...)`

Local generation knobs (top-level constants in `main.py`):

- `_LOCAL_DEFAULT_MAX_TOKENS` (default `384`)
- `_LOCAL_MULTI_INTENT_MAX_TOKENS` (default `512`)
- `_LOCAL_COMPLEX_MAX_TOKENS` (default `896`)
- `_LOCAL_DEFAULT_TEMPERATURE` (default `0.10`)
- `_LOCAL_DEFAULT_TOP_P` (default `0.95`)
- `_LOCAL_DEFAULT_TOP_K` (default `40`)
- `_LOCAL_DEFAULT_CONFIDENCE_THRESHOLD` (default `0.68`, passed to `cactus_complete`)
- `_LOCAL_SINGLE_INTENT_TOOL_CAP` (default `5`)
- `_LOCAL_SINGLE_INTENT_TOOL_RAG_TOP_K` (default `3`)

Cloud routing knob (top-level constants in `main.py`):

- `_CLOUD_ACCEPT_MARGIN_DEFAULT` (default `0.75`)
- `_CLOUD_ACCEPT_MARGIN_MULTI_TURN_CAP` (default `0.35`)

## Notes

This file documents the strategy in `main.py` as it currently behaves.

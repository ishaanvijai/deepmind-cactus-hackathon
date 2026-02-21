import sys, os
sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

import json
import random
import argparse
from benchmark import run_benchmark


def convert_tools(berkeley_functions):
    """Convert Berkeley tool definitions to our format."""
    tools = []
    for fn in berkeley_functions:
        params = fn.get("parameters", {})
        properties = {}
        for prop_name, prop_def in params.get("properties", {}).items():
            clean = {}
            ptype = prop_def.get("type", "string")
            if ptype == "float":
                ptype = "number"
            elif ptype in ("int", "integer"):
                ptype = "integer"
            elif ptype in ("bool", "boolean"):
                ptype = "boolean"
            clean["type"] = ptype
            if "description" in prop_def:
                clean["description"] = prop_def["description"]
            properties[prop_name] = clean

        top_type = params.get("type", "object")
        if top_type == "dict":
            top_type = "object"

        tool = {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": {
                "type": top_type,
                "properties": properties,
                "required": params.get("required", []),
            },
        }
        tools.append(tool)
    return tools


def get_entry_type(entry_id):
    """Return type string based on ID prefix."""
    for prefix in ["irrelevance", "simple_python", "live_parallel", "parallel", "multi_turn"]:
        if entry_id.startswith(prefix):
            return prefix
    return "unknown"


def build_expected_calls(entry, entry_type):
    """Build expected function calls based on entry type."""
    if entry_type == "irrelevance":
        return []
    if "path" in entry:
        return [{"name": n, "arguments": {}} for n in entry["path"]]
    seen = set()
    calls = []
    for fn in entry["function"]:
        if fn["name"] not in seen:
            seen.add(fn["name"])
            calls.append({"name": fn["name"], "arguments": {}})
    return calls


def get_difficulty(entry_type):
    """Map entry type to difficulty level."""
    if entry_type in ("irrelevance", "simple_python"):
        return "easy"
    if entry_type in ("parallel", "live_parallel"):
        return "medium"
    if entry_type == "multi_turn":
        return "hard"
    return "medium"


def make_stub_tools(path_names):
    """Create minimal tool stubs from path names (for multi_turn entries without function defs)."""
    tools = []
    seen = set()
    for name in path_names:
        if name not in seen:
            seen.add(name)
            tools.append({
                "name": name,
                "description": name,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            })
    return tools


def build_messages(entry, entry_type):
    """Build benchmark messages from Berkeley question format.

    For multi_turn entries, merge all turns to provide full context for
    long path expectations. For other entry types, use the first variant.
    """
    questions = entry.get("question", [])
    if not questions:
        return []

    if entry_type == "multi_turn":
        merged = []
        for turn in questions:
            if isinstance(turn, list):
                merged.extend(m for m in turn if isinstance(m, dict))
            elif isinstance(turn, dict):
                merged.append(turn)
        if merged:
            return merged

    first = questions[0]
    if isinstance(first, list):
        return [m for m in first if isinstance(m, dict)]
    if isinstance(first, dict):
        return [first]
    return []


def load_berkeley_benchmarks(filepath):
    """Load Berkeley benchmark entries from NDJSON and convert to our format."""
    benchmarks = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry_id = entry["id"]
            entry_type = get_entry_type(entry_id)

            messages = build_messages(entry, entry_type)

            if "function" in entry:
                tools = convert_tools(entry["function"])
            else:
                tools = make_stub_tools(entry.get("path", []))

            expected_calls = build_expected_calls(entry, entry_type)
            difficulty = get_difficulty(entry_type)

            benchmarks.append({
                "name": entry_id,
                "difficulty": difficulty,
                "messages": messages,
                "tools": tools,
                "expected_calls": expected_calls,
            })
    return benchmarks


def sample_benchmarks(benchmarks, n=25):
    """Pick a stratified random subset of n benchmarks, preserving difficulty distribution."""
    if n <= 0 or n >= len(benchmarks):
        return list(benchmarks)

    by_difficulty = {}
    for b in benchmarks:
        by_difficulty.setdefault(b["difficulty"], []).append(b)

    rng = random.Random(42)
    total = len(benchmarks)
    sampled = []
    remaining = n
    difficulties = sorted(by_difficulty.keys())
    for i, diff in enumerate(difficulties):
        pool = by_difficulty[diff]
        if i == len(difficulties) - 1:
            count = remaining
        else:
            count = max(1, round(n * len(pool) / total))
            count = min(count, len(pool), remaining)
        sampled.extend(rng.sample(pool, min(count, len(pool))))
        remaining -= min(count, len(pool))

    rng.shuffle(sampled)
    return sampled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Berkeley benchmark conversion")
    parser.add_argument("--file", default="largertest.json", help="Path to NDJSON benchmark file")
    parser.add_argument("--sample", type=int, default=0, help="Stratified sample size (0 = run all)")
    args = parser.parse_args()

    all_benchmarks = load_berkeley_benchmarks(args.file)
    benchmarks = sample_benchmarks(all_benchmarks, args.sample) if args.sample else all_benchmarks
    if args.sample:
        print(f"Sampled {len(benchmarks)} / {len(all_benchmarks)} Berkeley benchmark cases\n")
    else:
        print(f"Running all {len(benchmarks)} Berkeley benchmark cases\n")
    run_benchmark(benchmarks)

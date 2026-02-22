"""
THE CURSED FRONTIER — A Hybrid D&D CLI Game
Showcases Cactus Hybrid Intelligence:
  - Local  (FunctionGemma): dice rolls, stat checks, item use, damage
  - Cloud  (Gemini Flash):  character creation, scene narration, creative responses
  - Voice  (Whisper):       cactus_transcribe captures every player action
"""

import sys
import os
import json
import random
import time

# ── Cactus / cloud imports ──────────────────────────────────────────────────
sys.path.insert(0, "../cactus/python/src")

FUNCTIONGEMMA_PATH = "../cactus/weights/functiongemma-270m-it"
WHISPER_PATH       = "../cactus/weights/whisper-small"
WHISPER_PROMPT     = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

try:
    from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_transcribe, cactus_vad
    CACTUS_AVAILABLE = True
except ImportError:
    CACTUS_AVAILABLE = False
    print("[WARN] cactus module not found — local tool calls will be simulated")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARN] python-dotenv not installed — run: pip install python-dotenv")

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = bool(os.environ.get("GEMINI_API_KEY"))
    if not GEMINI_AVAILABLE:
        print("[WARN] GEMINI_API_KEY not set — cloud narration will be skipped")
        print("       export GEMINI_API_KEY=<your key>  to enable")
except ImportError:
    GEMINI_AVAILABLE = False
    print("[WARN] google-genai not installed — run: pip install google-genai")


# ── Game-state ──────────────────────────────────────────────────────────────
game_state: dict = {
    "character": {},    # name, class, hp, max_hp, stats, abilities, last_stand_used
    "inventory": [],    # list of {id, name, description, effect}
    "scene": 0,
    "history": [],      # Gemini multi-turn context
    "routing_log": [],  # for demo display: ["[LOCAL] roll_dice …", "[CLOUD] …"]
}


# ── GAME_TOOLS ───────────────────────────────────────────────────────────────
# These are passed to FunctionGemma for structured tool calling.
GAME_TOOLS = [
    {
        "name": "roll_dice",
        "description": (
            "Roll one or more dice for attacks, skill checks, or damage. "
            "Use sides=20 for attack/skill checks, sides=4/6/8 for damage."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sides": {
                    "type": "integer",
                    "description": "Number of sides on the die (4, 6, 8, 10, 12, or 20)",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of dice to roll",
                },
            },
            "required": ["sides", "count"],
        },
    },
    {
        "name": "check_stat",
        "description": "Look up a character stat: STR, DEX, INT, WIS, HP, or abilities.",
        "parameters": {
            "type": "object",
            "properties": {
                "stat_name": {
                    "type": "string",
                    "description": "The stat to check, e.g. STR, DEX, HP, abilities",
                },
            },
            "required": ["stat_name"],
        },
    },
    {
        "name": "use_item",
        "description": "Use a consumable item from inventory (e.g. healing_draught, herbs).",
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "The id of the item to use",
                },
            },
            "required": ["item_id"],
        },
    },
    {
        "name": "apply_damage",
        "description": "Apply damage (positive amount) or healing (negative amount) to the player.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "integer",
                    "description": "Positive = damage taken, negative = HP restored",
                },
                "target": {
                    "type": "string",
                    "description": "Who is affected: 'player' or enemy name",
                },
            },
            "required": ["amount", "target"],
        },
    },
    {
        "name": "check_hp",
        "description": "Check the player's current and maximum HP.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

CACTUS_TOOLS = [{"type": "function", "function": t} for t in GAME_TOOLS]

GAME_SYSTEM_PROMPT = (
    "You are a game-mechanics resolver for a D&D adventure. "
    "When the player takes a physical action (attack, dodge, use item, check stats), "
    "call the appropriate tool. "
    "If the action is purely narrative or conversational, do not call any tool. "
    "Never produce text — only tool calls."
)


# ── Local tool executor ──────────────────────────────────────────────────────
def execute_tool(name: str, args: dict) -> str:
    """
    Execute a game tool call locally using Python game logic.
    Returns a plain-English result string that will be fed to Gemini for narration.
    """
    char = game_state["character"]

    if name == "roll_dice":
        sides = int(args.get("sides", 20))
        count = int(args.get("count", 1))
        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls)
        label = f"{count}d{sides}"
        log(f"[LOCAL] roll_dice({label}) → {rolls} = {total}")
        return f"Dice roll ({label}): {rolls} = {total}"

    elif name == "check_stat":
        stat = args.get("stat_name", "").upper()
        if stat in ("HP", "HEALTH"):
            result = f"HP {char.get('hp', '?')} / {char.get('max_hp', '?')}"
        elif stat == "ABILITIES":
            abilities = [a["name"] for a in char.get("abilities", [])]
            last_stand_used = char.get("last_stand_used", False)
            result = ", ".join(abilities) + (" (Last Stand used)" if last_stand_used else " (all available)")
        else:
            val = char.get("stats", {}).get(stat, "unknown")
            mod = (int(val) - 10) // 2 if isinstance(val, int) else 0
            sign = "+" if mod >= 0 else ""
            result = f"{stat} = {val} ({sign}{mod} modifier)"
        log(f"[LOCAL] check_stat({stat}) → {result}")
        return result

    elif name == "use_item":
        item_id = args.get("item_id", "").lower().replace(" ", "_")
        inventory = game_state["inventory"]
        # Find item (partial match on id or name)
        item = next(
            (i for i in inventory if item_id in i["id"].lower() or item_id in i["name"].lower()),
            None,
        )
        if not item:
            log(f"[LOCAL] use_item({item_id}) → item not found in inventory")
            return f"Item '{item_id}' not found in inventory."
        # Parse effect: "heal Xd6+Y" or "damage Xd4"
        effect = item.get("effect", "")
        heal_amount = _resolve_effect(effect)
        # Apply
        char["hp"] = min(char["max_hp"], char["hp"] + heal_amount)
        inventory.remove(item)
        log(f"[LOCAL] use_item({item['name']}) → healed {heal_amount} HP. HP now {char['hp']}/{char['max_hp']}")
        return f"Used {item['name']}. Restored {heal_amount} HP. HP is now {char['hp']}/{char['max_hp']}."

    elif name == "apply_damage":
        amount = int(args.get("amount", 0))
        target = args.get("target", "player").lower()
        if "player" in target or char.get("name", "").lower() in target:
            # Check Last Stand ability
            if amount > 0 and not char.get("last_stand_used", False):
                has_last_stand = any(
                    "last stand" in a.get("name", "").lower()
                    for a in char.get("abilities", [])
                )
                if has_last_stand and char["hp"] - amount <= 0:
                    amount = amount // 2
                    char["last_stand_used"] = True
                    log(f"[LOCAL] Last Stand activated! Damage halved to {amount}")
            char["hp"] = max(0, char["hp"] - amount)
            action = "takes" if amount > 0 else "heals"
            log(f"[LOCAL] apply_damage({amount}) → player {action} {abs(amount)}. HP {char['hp']}/{char['max_hp']}")
            return (
                f"Player takes {amount} damage. HP is now {char['hp']}/{char['max_hp']}."
                if amount > 0
                else f"Player heals {abs(amount)}. HP is now {char['hp']}/{char['max_hp']}."
            )
        else:
            log(f"[LOCAL] apply_damage({amount}) to {target}")
            return f"{target.capitalize()} takes {amount} damage."

    elif name == "check_hp":
        result = f"HP: {char.get('hp', '?')} / {char.get('max_hp', '?')}"
        log(f"[LOCAL] check_hp() → {result}")
        return result

    return f"Unknown tool: {name}"


def _resolve_effect(effect: str) -> int:
    """Parse simple effect strings like 'heal 1d6+2' into a final integer."""
    import re
    # e.g. "heal 1d6+2" or "restore 2d4" or "heals 1d4+1"
    m = re.search(r"(\d+)d(\d+)(?:\+(\d+))?", effect, re.IGNORECASE)
    if m:
        count, sides, bonus = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        return sum(random.randint(1, sides) for _ in range(count)) + bonus
    # Flat heal
    m2 = re.search(r"(\d+)", effect)
    if m2:
        return int(m2.group(1))
    return 0


# ── Cactus local call ────────────────────────────────────────────────────────
def run_local_tool_call(player_action: str) -> dict:
    """
    Ask FunctionGemma to parse the player action into tool calls.
    Returns the raw cactus result dict.
    """
    if not CACTUS_AVAILABLE:
        # Simulate: assume no tool call for testing without hardware
        return {"function_calls": [], "confidence": 0.0, "total_time_ms": 0, "cloud_handoff": True}

    model = cactus_init(FUNCTIONGEMMA_PATH)
    messages = [
        {"role": "system", "content": GAME_SYSTEM_PROMPT},
        {"role": "user",   "content": player_action},
    ]
    raw_str = cactus_complete(
        model,
        messages,
        tools=CACTUS_TOOLS,
        force_tools=True,
        max_tokens=256,
        temperature=0.10,
        top_p=0.95,
        top_k=40,
        confidence_threshold=0.68,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except (json.JSONDecodeError, TypeError):
        print(f"  [LOCAL] JSON parse failed. Raw output: {raw_str[:120]!r}")
        return {"function_calls": [], "confidence": 0.0, "total_time_ms": 0, "cloud_handoff": True}

    calls = raw.get("function_calls", [])
    confidence = raw.get("confidence", 0.0)
    handoff = raw.get("cloud_handoff", False)

    if not calls:
        # Log what the model actually returned so we can tune the prompt
        response_preview = (raw.get("response") or "")[:80]
        print(f"  [LOCAL] no tool calls (conf={confidence:.2f}, handoff={handoff})"
              f"{f', response: {response_preview!r}' if response_preview else ''}")
    else:
        print(f"  [LOCAL] {len(calls)} tool call(s): {[c['name'] for c in calls]} "
              f"(conf={confidence:.2f})")

    return {
        "function_calls": calls,
        "confidence":     confidence,
        "total_time_ms":  raw.get("total_time_ms", 0),
        "cloud_handoff":  handoff,
    }


# ── Gemini cloud call ────────────────────────────────────────────────────────
def run_cloud_narration(prompt: str, tool_results: str = "") -> str:
    """
    Call Gemini Flash to generate narrative text.
    `prompt` is the full scene context + player action.
    `tool_results` is appended if local tools ran (e.g. dice outcome).
    """
    if not GEMINI_AVAILABLE:
        return "[CLOUD SIMULATION] The story continues..."

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Build character summary for Gemini context
    char = game_state["character"]
    char_summary = (
        f"Character: {char.get('name','?')} ({char.get('class','?')}), "
        f"HP {char.get('hp','?')}/{char.get('max_hp','?')}, "
        f"STR {char.get('stats',{}).get('STR','?')} "
        f"DEX {char.get('stats',{}).get('DEX','?')} "
        f"INT {char.get('stats',{}).get('INT','?')} "
        f"WIS {char.get('stats',{}).get('WIS','?')}, "
        f"Inventory: {[i['name'] for i in game_state['inventory']]}"
    )

    system = (
        "You are the Game Master for a gritty Oregon Trail–style D&D adventure called "
        "THE CURSED FRONTIER. Your narration is atmospheric, terse, and vivid — "
        "think 2–4 sentences max. Never break character. "
        f"{char_summary}"
    )

    user_content = prompt
    if tool_results:
        user_content += f"\n\n[Mechanics resolved: {tool_results}]"

    game_state["history"].append({"role": "user", "content": user_content})

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {"role": "user", "parts": [{"text": system}]},
        ] + [
            {"role": m["role"], "parts": [{"text": m["content"]}]}
            for m in game_state["history"]
        ],
    )

    text = response.text.strip()
    game_state["history"].append({"role": "model", "content": text})
    log(f"[CLOUD] {text[:80]}…" if len(text) > 80 else f"[CLOUD] {text}")
    return text


def run_cloud_character_creation(description: str) -> dict:
    """
    Ask Gemini to infer a character sheet from the player's spoken description.
    Returns a dict matching the character schema.
    """
    if not GEMINI_AVAILABLE:
        # Fallback character for testing
        return {
            "name": "Wanderer",
            "class": "Adventurer",
            "hp": 20, "max_hp": 20,
            "stats": {"STR": 12, "DEX": 12, "INT": 10, "WIS": 10},
            "items": [{"id": "knife", "name": "Knife", "description": "A simple knife", "effect": "1d4"}],
            "abilities": [{"id": "tough", "name": "Tough", "description": "Endure one extra hit"}],
        }

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    schema_prompt = f"""You are creating a character sheet for a gritty D&D-style frontier adventure.
The player described themselves as: "{description}"

Return ONLY valid JSON with this exact structure:
{{
  "name": "string — creative name based on description",
  "class": "string — inferred class (e.g. Fallen Soldier, Hedge Wizard, Wandering Merchant)",
  "hp": integer between 12 and 25,
  "max_hp": integer (same as hp),
  "stats": {{"STR": int, "DEX": int, "INT": int, "WIS": int}},
  "items": [
    {{"id": "snake_case_id", "name": "Display Name", "description": "one sentence", "effect": "heal 1d6+2 OR damage 1d4+1"}}
  ],
  "abilities": [
    {{"id": "snake_case_id", "name": "Ability Name", "description": "one-line mechanic"}}
  ]
}}

Rules:
- Stats should range 7–16, averaging ~10. High STR if physically described, high INT if scholarly, etc.
- Include 1–3 items based on what they mentioned carrying. Infer sensible items if vague.
- Include 1 ability that reflects their background (e.g. Last Stand, Haggler, Hedge Magic).
- Healing items use effect like "heal 1d6+2". Weapons use "damage 1d4+1".
- Return ONLY the JSON object, no markdown, no explanation."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=schema_prompt,
    )

    raw = response.text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        sheet = json.loads(raw)
        sheet["last_stand_used"] = False
        return sheet
    except json.JSONDecodeError:
        print(f"\n[WARN] Gemini returned invalid JSON for character sheet. Using defaults.\n{raw[:200]}")
        return {
            "name": "The Wanderer",
            "class": "Adventurer",
            "hp": 18, "max_hp": 18,
            "stats": {"STR": 12, "DEX": 11, "INT": 10, "WIS": 10},
            "items": [{"id": "knife", "name": "Hunting Knife", "description": "A worn blade", "effect": "damage 1d4+1"}],
            "abilities": [{"id": "survivor", "name": "Survivor", "description": "Endure one extra blow"}],
            "last_stand_used": False,
        }


# ── Voice input ──────────────────────────────────────────────────────────────
AUDIO_FILE        = "/tmp/dnd_input.wav"
SAMPLE_RATE       = 16000
VAD_PATH          = "../cactus/weights/silero-vad"
VAD_CHUNK_SAMPLES = 1600   # 100ms chunks at 16kHz
VAD_SILENCE_S     = 1.5    # seconds of silence after speech → stop
VAD_MAX_S         = 15.0   # hard cap


def _check_voice_deps() -> tuple[bool, str]:
    """Return (available, reason) so callers can log clearly."""
    if not CACTUS_AVAILABLE:
        return False, "cactus module not importable"
    if not os.path.exists(WHISPER_PATH):
        return False, f"whisper weights not found at {WHISPER_PATH!r} — run: cactus download openai/whisper-small"
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        return False, "sounddevice not installed — run: pip install sounddevice"
    return True, "ok"


def _save_and_transcribe(audio_np) -> str:
    """Write numpy int16 array to WAV, pass to Whisper, return text."""
    import wave
    import numpy as np
    try:
        with wave.open(AUDIO_FILE, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_np.tobytes())
    except Exception as exc:
        print(f"  [ERROR] Could not write WAV: {exc}")
        return input("  > ").strip()

    try:
        print("  [Transcribing…]", flush=True)
        whisper = cactus_init(WHISPER_PATH)
        raw = cactus_transcribe(whisper, AUDIO_FILE, prompt=WHISPER_PROMPT)
        cactus_destroy(whisper)
    except Exception as exc:
        print(f"  [ERROR] cactus_transcribe failed: {exc}")
        return input("  > ").strip()

    try:
        text = json.loads(raw)["response"].strip()
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"  [ERROR] Could not parse transcription JSON ({exc}): {raw[:120]!r}")
        return input("  > ").strip()

    print(f'  [Heard]: "{text}"')
    return text


def _record_vad() -> str:
    """
    Stream mic audio in 500ms chunks, run VAD on the full accumulated buffer
    each chunk so it has enough context. Stop once the last detected speech
    segment ended >= VAD_SILENCE_S ago.
    """
    import sounddevice as sd
    import numpy as np

    vad = cactus_init(VAD_PATH)

    max_samples     = int(VAD_MAX_S * SAMPLE_RATE)
    silence_samples = int(VAD_SILENCE_S * SAMPLE_RATE)
    min_buf_samples = SAMPLE_RATE  # need at least 1s before first VAD check

    accumulated = np.empty((0,), dtype=np.int16)
    speech_seen = False

    print("  [🎙  Listening", end="", flush=True)
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="int16", blocksize=VAD_CHUNK_SAMPLES) as stream:
            while len(accumulated) < max_samples:
                chunk, _ = stream.read(VAD_CHUNK_SAMPLES)
                accumulated = np.concatenate([accumulated, chunk.flatten()])

                if len(accumulated) < min_buf_samples:
                    continue

                result = json.loads(cactus_vad(
                    vad,
                    pcm_data=accumulated.tobytes(),
                    options={"threshold": 0.3, "min_speech_duration_ms": 150},
                ))
                segments = result.get("segments", [])

                if not segments:
                    # No speech detected yet at all — keep waiting
                    continue

                speech_seen = True
                last_end = segments[-1]["end"]   # sample index in accumulated
                trailing_silence = len(accumulated) - last_end

                if trailing_silence < silence_samples:
                    print("▪", end="", flush=True)   # still in / just after speech
                else:
                    print("]", flush=True)
                    break

    except Exception as exc:
        print(f"\n  [ERROR] VAD recording failed: {exc}")
        cactus_destroy(vad)
        return input("  > ").strip()

    cactus_destroy(vad)

    if not speech_seen:
        print("]\n  [no speech detected]")
        return ""

    return _save_and_transcribe(accumulated)


def _record_fixed() -> str:
    """Fallback: fixed 5-second recording window when VAD model is unavailable."""
    import sounddevice as sd
    import numpy as np

    SECS = 5
    print(f"  [🎙  Recording {SECS}s — speak now…]", flush=True)
    try:
        audio = sd.rec(int(SECS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="int16")
        sd.wait()
    except Exception as exc:
        print(f"  [ERROR] sounddevice recording failed: {exc}")
        return input("  > ").strip()
    return _save_and_transcribe(audio)


def record_player_action(prompt_text: str) -> str:
    """Record and transcribe player speech. Uses VAD if available, else fixed 5s window."""
    print(f"\n{prompt_text}")

    voice_ok, reason = _check_voice_deps()
    if not voice_ok:
        print(f"  [voice unavailable: {reason}]")
        return input("  > ").strip()

    if os.path.exists(VAD_PATH):
        return _record_vad()
    else:
        print(f"  [VAD model not found at {VAD_PATH!r} — using 5s window]")
        print(f"  [To enable silence detection: cactus download snakers4/silero-vad]")
        return _record_fixed()


# ── Hybrid action processor ──────────────────────────────────────────────────
def process_action(player_input: str, scene_context: str) -> str:
    """
    Core hybrid routing:
    1. Try FunctionGemma for structured tool calls (game mechanics).
    2. If confident → execute tools locally → feed result to Gemini for narrative.
    3. If not confident / no calls → hand full action to Gemini for creative response.
    """
    t0 = time.time()
    local = run_local_tool_call(player_input)
    calls = local.get("function_calls", [])
    confidence = local.get("confidence", 0.0)
    cloud_handoff = local.get("cloud_handoff", False)

    if calls and confidence >= 0.68 and not cloud_handoff:
        # ── LOCAL path: execute mechanics, then cloud narrates ─────────────
        tool_result_parts = []
        for call in calls:
            result = execute_tool(call["name"], call.get("arguments", {}))
            tool_result_parts.append(result)
        tool_results = " | ".join(tool_result_parts)

        narrative_prompt = (
            f"Scene: {scene_context}\n"
            f"Player action: {player_input}\n"
        )
        narrative = run_cloud_narration(narrative_prompt, tool_results=tool_results)
        elapsed = (time.time() - t0) * 1000
        _print_routing_label("local + cloud narrative", confidence, elapsed)
        return narrative

    else:
        # ── CLOUD path: full creative response ──────────────────────────────
        narrative_prompt = (
            f"Scene: {scene_context}\n"
            f"Player action: {player_input}\n"
        )
        narrative = run_cloud_narration(narrative_prompt)
        elapsed = (time.time() - t0) * 1000
        _print_routing_label("cloud", confidence, elapsed)
        return narrative


def _print_routing_label(source: str, confidence: float, ms: float):
    label_map = {
        "local + cloud narrative": "LOCAL → CLOUD",
        "cloud": "CLOUD",
    }
    label = label_map.get(source, source.upper())
    print(f"\n  ┌─ Route: [{label}]  conf={confidence:.2f}  {ms:.0f}ms")


# ── Display helpers ──────────────────────────────────────────────────────────
def log(msg: str):
    game_state["routing_log"].append(msg)
    print(f"  {msg}")


def hr(char="─", width=52):
    print(char * width)


def print_character_sheet():
    char = game_state["character"]
    stats = char.get("stats", {})
    print()
    hr("━")
    print(f"  ✦  CHARACTER SHEET")
    hr("━")
    print(f"  Name:   {char.get('name','?'):<20}  Class: {char.get('class','?')}")
    print(f"  HP:     {char.get('hp','?')} / {char.get('max_hp','?')}")
    print(f"  STR: {stats.get('STR','?'):<4} DEX: {stats.get('DEX','?'):<4} "
          f"INT: {stats.get('INT','?'):<4} WIS: {stats.get('WIS','?')}")
    print()
    print("  ITEMS:")
    for item in game_state["inventory"]:
        print(f"    • {item['name']:<22} [{item.get('effect','')}]")
    print()
    print("  ABILITIES:")
    for ab in char.get("abilities", []):
        print(f"    • {ab['name']:<22} {ab.get('description','')}")
    hr("━")


def print_hp_bar():
    char = game_state["character"]
    hp, max_hp = char.get("hp", 0), char.get("max_hp", 1)
    filled = int((hp / max_hp) * 20)
    bar = "█" * filled + "░" * (20 - filled)
    print(f"\n  HP [{bar}] {hp}/{max_hp}")


def print_scene_header(title: str):
    print()
    hr("═")
    print(f"  SCENE {game_state['scene']} — {title.upper()}")
    hr("═")


# ── 4-scene definitions ──────────────────────────────────────────────────────
SCENES = [
    {
        "title": "The Dire Wolf",
        "intro": (
            "You are crossing the smoke-gray hills of the Dusty Frontier. "
            "As you crest a ridge, a massive dire wolf blocks the path, "
            "yellow eyes locked onto yours. It crouches to spring."
        ),
        "turns": 2,
        "enemy": "dire wolf",
    },
    {
        "title": "The Dying Trader",
        "intro": (
            "A merchant lies propped against his overturned cart, clutching "
            "a gash on his side. 'Please...' he rasps. 'I have coin...'"
        ),
        "turns": 1,
        "gives_item": {
            "id": "healing_draught",
            "name": "Healing Draught",
            "description": "A small vial of amber liquid",
            "effect": "heal 1d6+2",
        },
    },
    {
        "title": "The Toll Bridge",
        "intro": (
            "A scarred bandit blocks the only bridge across the gorge. "
            "'Ten gold. Or go around — three extra days through goblin lands.'"
        ),
        "turns": 1,
    },
    {
        "title": "The Shadow Wraith",
        "intro": (
            "You reach the Cursed Keep at last. But at the iron gates, "
            "a shadow-wraith coalesces from the dark — robes of void, "
            "eyes like dead stars. It raises a skeletal hand."
        ),
        "turns": 3,
        "enemy": "shadow wraith",
        "is_final": True,
    },
]


# ── Main game flow ───────────────────────────────────────────────────────────
def run_character_creation():
    print()
    print("  Before you begin, tell me — who are you?")
    print("  What do you carry? What's your background?")
    print()

    description = record_player_action("[🎙  Speak your character aloud — name, background, items]")

    if not description:
        description = "I'm a wandering sellsword with a rusty longsword and a worn leather pack."

    print("\n  [Generating character sheet via cloud…]")
    sheet = run_cloud_character_creation(description)

    game_state["character"] = sheet
    game_state["inventory"] = sheet.pop("items", [])

    print_character_sheet()
    input("\n  Press ENTER to begin your journey…")


def run_scene(scene_def: dict, scene_num: int):
    game_state["scene"] = scene_num
    print_scene_header(scene_def["title"])

    # Always print the base scene text so the player is never left with nothing
    print(f"\n  {scene_def['intro']}\n")

    # Cloud adds atmospheric enrichment on top (optional — skipped if Gemini unavailable)
    if GEMINI_AVAILABLE:
        enrichment = run_cloud_narration(
            f"Scene setup: {scene_def['intro']} "
            f"Expand with 1–2 vivid atmospheric sentences. Do not repeat the setup."
        )
        if enrichment:
            print(f"  {enrichment}\n")

    turns = scene_def.get("turns", 1)
    for turn in range(turns):
        print_hp_bar()

        player_input = record_player_action(
            "[🎙  What do you do?]" if turn == 0 else "[🎙  And now?]"
        )

        if not player_input:
            continue

        response = process_action(player_input, scene_context=scene_def["intro"])
        print(f"\n  {response}")

        # Check death
        if game_state["character"].get("hp", 1) <= 0:
            print()
            hr()
            print("  ✝  You have fallen. The Cursed Frontier claims another soul.")
            hr()
            sys.exit(0)

    # Scene rewards
    if "gives_item" in scene_def:
        item = scene_def["gives_item"]
        game_state["inventory"].append(item)
        print(f"\n  [Item gained: {item['name']} — {item['description']}]")


def run_game():
    # Header
    print()
    hr("═")
    print("  THE CURSED FRONTIER")
    hr("═")
    print()
    print("  A lone traveler. A dying land.")
    print("  One last chance to reach the Cursed Keep and break the shadow.")
    print()
    print("  Powered by Cactus Hybrid Intelligence:")
    print("    LOCAL  → FunctionGemma  (dice, stats, items)")
    print("    CLOUD  → Gemini Flash   (narration, creativity, character)")
    print("    VOICE  → Whisper        (your spoken commands)")
    hr("═")

    run_character_creation()

    for i, scene in enumerate(SCENES, start=1):
        run_scene(scene, scene_num=i)

    # Ending
    print()
    hr("═")
    print("  YOU HAVE SURVIVED THE CURSED FRONTIER.")
    print()
    char = game_state["character"]
    print(f"  {char.get('name','The Wanderer')} stands before the open gates of the keep,")
    print("  bloodied but unbroken. The shadow lifts. Dawn breaks.")
    hr("═")
    print()

    # Final cloud flourish
    if GEMINI_AVAILABLE:
        outro = run_cloud_narration(
            "The hero has defeated the shadow wraith and the curse is broken. "
            "Write a 3-sentence triumphant closing to the adventure, referencing their name."
        )
        print(f"  {outro}\n")


if __name__ == "__main__":
    run_game()

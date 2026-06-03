"""Multi-agent orchestration: an orchestrator decomposes the email into one
subtask per delayed part, spawns a fresh sub-agent (its own tool-calling loop
and its own message history) for each, then aggregates the sub-results into the
same strict deliverable.

This is the deliberately expensive pattern. Every sub-agent re-sends the system
prompt and the tool definitions, re-derives the same business rule, and grows
its own context. The orchestrator adds a decompose call and an aggregate call on
top. Same model, same tools, same output contract as the other two approaches,
so the only thing that changes is the architecture, and therefore the cost.
"""

import json

from prompts import (
    ORCHESTRATOR_AGGREGATE,
    ORCHESTRATOR_DECOMPOSE,
    SUBAGENT_SYSTEM,
)
from utils import (
    BOLD,
    DIM,
    GRAY,
    MAGENTA,
    TOOL_BY_NAME,
    TOOLS,
    YELLOW,
    RunConfig,
    RunResult,
    TokenMeter,
    format_calls,
    paint,
)


def _run_subagent(meter: TokenMeter, cfg: RunConfig, part_id: str, eta_days: int,
                  trace: list[str]) -> str:
    """One focused tool-calling loop for a single part. Returns its one line."""
    messages = [
        {"role": "system", "content": SUBAGENT_SYSTEM},
        {"role": "user", "content": f"Part: {part_id}, ETA: {eta_days} days. Handle it."},
    ]
    for step in range(1, cfg.max_steps + 1):
        message = meter.chat(messages, tools=TOOLS)["message"]
        messages.append(message)
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            trace.append(
                f"   sub[{part_id}] {step} final  (+{meter.last_in} in / +{meter.last_out} out)")
            return (message.get("content") or "").strip()
        trace.append(f"   sub[{part_id}] {step} tools: {format_calls(tool_calls)}  "
                     f"(+{meter.last_in} in / +{meter.last_out} out)")
        if cfg.stream:
            print(paint(f"    sub[{part_id}]> {format_calls(tool_calls)}", BOLD, YELLOW))
        for call in tool_calls:
            name = call["function"]["name"]
            fn = TOOL_BY_NAME.get(name)
            result = fn(**call["function"].get("arguments", {})) if fn else "Unknown tool."
            messages.append({"role": "tool", "content": result})
            if cfg.stream:
                preview = result if len(result) <= 200 else result[:197] + "..."
                print(paint(f"    inject> {name} -> {preview}", DIM, GRAY))
    return f"{part_id}: (subagent did not finish within the step limit)"


def run_multi_agent(cfg: RunConfig, email: str) -> RunResult:
    meter = TokenMeter(cfg)
    trace: list[str] = []

    # 1. Orchestrator decomposes the email into per-part subtasks.
    resp = meter.chat([{"role": "user",
                        "content": ORCHESTRATOR_DECOMPOSE + email}], fmt="json")
    try:
        parts = json.loads(resp["message"]["content"]).get("parts", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        parts = []
    trace.append(f"1 [orch] decompose  (+{meter.last_in} in / +{meter.last_out} out)  "
                 f"-> {[p.get('part_id') for p in parts]}")
    if cfg.stream:
        print(paint(f"    orch> spawning {len(parts)} sub-agent(s)", BOLD, MAGENTA))

    # 2. One fresh sub-agent per part (independent context each).
    results = []
    for p in parts:
        pid = str(p.get("part_id", ""))
        eta = int(float(p.get("eta_days", 0)))
        results.append(_run_subagent(meter, cfg, pid, eta, trace))

    # 3. Orchestrator aggregates the sub-results into the final deliverable.
    agg = meter.chat([{"role": "user",
                       "content": ORCHESTRATOR_AGGREGATE + "\n".join(results)}])
    trace.append(
        f"{len(parts) + 2} [orch] aggregate  (+{meter.last_in} in / +{meter.last_out} out)")

    return meter.result("multi-agent", (agg["message"].get("content") or "").strip(), trace)

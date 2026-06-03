"""Autonomous agent: a native tool-calling loop. The model decides which tools
to call and when to stop. Every turn resends the growing history.

The agent has four tools available, but only two are core to the task; it must
select what it needs rather than calling everything. It is given the same
business rule and the same output contract as the workflow, so the two produce
the same deliverable, and the token comparison is fair.
"""

from data import TASK
from prompts import AGENT_SYSTEM_PROMPT
from utils import (
    BOLD,
    DIM,
    GRAY,
    TOOL_BY_NAME,
    TOOLS,
    YELLOW,
    RunConfig,
    RunResult,
    TokenMeter,
    format_calls,
    paint,
)


def run_agent(cfg: RunConfig, email: str) -> RunResult:
    meter = TokenMeter(cfg)
    trace: list[str] = []
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{email}\n\nTask: {TASK}"},
    ]

    for step in range(1, cfg.max_steps + 1):
        message = meter.chat(messages, tools=TOOLS)["message"]
        messages.append(message)
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            trace.append(
                f"{step} [LLM]  final answer  (+{meter.last_in} in / +{meter.last_out} out)")
            return meter.result("agent", (message.get("content") or "").strip(), trace)

        trace.append(f"{step} [LLM]  tools: {format_calls(tool_calls)}  "
                     f"(+{meter.last_in} in / +{meter.last_out} out)")
        if cfg.stream:
            print(paint(f"    tools> {format_calls(tool_calls)}", BOLD, YELLOW))
        for call in tool_calls:
            name = call["function"]["name"]
            fn = TOOL_BY_NAME.get(name)
            result = fn(**call["function"].get("arguments", {})) if fn else "Unknown tool."
            messages.append({"role": "tool", "content": result})
            if cfg.stream:
                preview = result if len(result) <= 300 else result[:297] + "..."
                print(paint(f"    inject> {name} -> {preview}", DIM, GRAY))

    return meter.result("agent", "(did not finish within the step limit)", trace)

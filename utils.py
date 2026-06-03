"""Shared infrastructure: run configuration, agent tools, token metering, the
result type, and all console rendering. No business logic lives here (charts
live in report.py).
"""

import json
import statistics
import time
from dataclasses import dataclass

import ollama

from data import (
    ALTERNATES,
    INPUT_COST_RATIO,
    INVENTORY,
    OUTPUT_COST,
    PART_DETAILS,
    SUPPLIER_RATINGS,
    TEMPERATURE,
)

# --- ANSI colors (console only; converted to HTML for the md transcript) ----
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, RED, CYAN, YELLOW, GRAY = "\033[32m", "\033[31m", "\033[36m", "\033[33m", "\033[90m"
MAGENTA = "\033[35m"


def paint(text: str, *codes: str) -> str:
    return f"{''.join(codes)}{text}{RESET}"


# --- Tools (used by the agent via native tool calling) ---------------------
def check_inventory(part_id: str) -> str:
    """Return on-hand stock, daily usage and safety stock for a part ID."""
    data = INVENTORY.get(part_id)
    return json.dumps(data) if data else "Part not found in inventory."


def check_alternate_suppliers(part_id: str) -> str:
    """Return alternate suppliers for a part, with lead time and unit cost."""
    data = ALTERNATES.get(part_id)
    return json.dumps(data) if data else "No alternate suppliers found."


def get_suppliers_by_price(part_id: str) -> str:
    """Return alternate suppliers for a part sorted from cheapest to most expensive."""
    data = sorted(ALTERNATES.get(part_id, []), key=lambda a: a["unit_cost_eur"])
    return json.dumps(data) if data else "No alternate suppliers found."


def get_supplier_rating(supplier: str) -> str:
    """Return the reliability rating (0-100) for a supplier."""
    rating = SUPPLIER_RATINGS.get(supplier)
    return json.dumps({"supplier": supplier, "rating": rating}) if rating is not None \
        else "No rating on file."


def get_part_details(part_id: str) -> str:
    """Return catalog details (category, criticality) for a part."""
    data = PART_DETAILS.get(part_id)
    return json.dumps(data) if data else "No details on file."


# check_inventory and check_alternate_suppliers are core. The other three are
# tangential or overlapping (get_suppliers_by_price restates alternates sorted by
# cost) so a focused agent should select what it needs, not call everything.
TOOLS = [check_inventory, check_alternate_suppliers, get_suppliers_by_price,
         get_supplier_rating, get_part_details]
TOOL_BY_NAME = {fn.__name__: fn for fn in TOOLS}


def format_calls(tool_calls: list) -> str:
    """Render a list of native tool calls as `name(arg=val, ...)` for the trace."""
    parts = []
    for call in tool_calls:
        fn = call["function"]
        args = ", ".join(f"{k}={v!r}" for k, v in (fn.get("arguments") or {}).items())
        parts.append(f"{fn['name']}({args})")
    return ", ".join(parts)


# --- Run configuration -----------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    """How to drive the model, shared by all three approaches (one object instead
    of passing model/keep_alive/stream/max_steps around as loose arguments)."""
    model: str
    max_steps: int = 15
    keep_alive: int | None = None  # 0 => unload between calls => no KV reuse
    stream: bool = False           # live output incl. thinking chunks


# --- Result of one run -----------------------------------------------------
@dataclass(frozen=True)
class RunResult:
    approach: str          # "workflow" or "agent"
    answer: str            # the final text the approach produced
    trace: list[str]       # human-readable step log
    calls: int
    input_tokens: int
    output_tokens: int
    seconds: float
    thinking: str = ""    # accumulated reasoning text (when the model exposes it)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> float:
        """Normalized real-money cost: output=1.0, input weighted by the ratio."""
        return round(self.output_tokens * OUTPUT_COST + self.input_tokens * INPUT_COST_RATIO, 1)

    def as_dict(self) -> dict:
        return {
            "approach": self.approach,
            "answer": self.answer,
            "trace": self.trace,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "seconds": round(self.seconds, 2),
            "thinking": self.thinking,
        }


# --- Token metering --------------------------------------------------------
class TokenMeter:
    """Wraps ollama.chat, accumulates totals, exposes per-call deltas."""

    def __init__(self, cfg: RunConfig):
        self.model = cfg.model
        self.keep_alive = cfg.keep_alive  # 0 => unload between calls => no KV reuse
        self.stream = cfg.stream          # live output incl. thinking chunks
        self.calls = self.input_tokens = self.output_tokens = 0
        self.seconds = 0.0
        self.last_in = self.last_out = 0
        self.thinking = ""

    def chat(self, messages: list, *, tools=None, fmt=None) -> dict:
        kwargs = {"model": self.model, "messages": messages,
                  "options": {"temperature": TEMPERATURE}}
        if self.keep_alive is not None:
            kwargs["keep_alive"] = self.keep_alive
        if tools:
            kwargs["tools"] = tools
        if fmt:
            kwargs["format"] = fmt
        start = time.time()
        response = self._stream_chat(kwargs) if self.stream else ollama.chat(**kwargs)
        self.seconds += time.time() - start
        self.calls += 1
        self.last_in = response.get("prompt_eval_count", 0)
        self.last_out = response.get("eval_count", 0)
        self.input_tokens += self.last_in
        self.output_tokens += self.last_out
        think_piece = (response.get("message") or {}).get("thinking")
        if think_piece:
            self.thinking += think_piece.strip() + "\n\n"
        return response

    def _stream_chat(self, kwargs: dict) -> dict:
        """Stream chunks, print thinking and answer live, return a response dict.

        Color is opened once per phase (not per token) so the captured transcript
        stays clean instead of one span per token. Continuation lines are indented.
        Token totals come from the final chunk; eval_count includes thinking.
        """
        kwargs = {**kwargs, "stream": True, "think": True}
        content, thinking, tool_calls = "", "", []
        indent = "      "
        phase = None  # None | "think" | "answer"
        pin = pout = 0

        def emit(text: str) -> None:
            print(text.replace("\n", "\n" + indent), end="", flush=True)

        for chunk in ollama.chat(**kwargs):
            msg = chunk.get("message") or {}
            think_piece, content_piece = msg.get("thinking"), msg.get("content")
            if think_piece:
                if phase != "think":
                    print("\n" + paint("    think>", BOLD, CYAN))
                    print(indent + DIM + GRAY, end="", flush=True)
                    phase = "think"
                emit(think_piece)
                thinking += think_piece
            if content_piece:
                if phase == "think":
                    print(RESET, end="", flush=True)
                if phase != "answer":
                    print("\n" + paint("    answer>", BOLD, CYAN))
                    print(indent, end="", flush=True)
                    phase = "answer"
                emit(content_piece)
                content += content_piece
            if msg.get("tool_calls"):
                tool_calls.extend(msg["tool_calls"])
            pin = chunk.get("prompt_eval_count") or pin
            pout = chunk.get("eval_count") or pout

        if phase == "think":
            print(RESET, end="")
        print(flush=True)
        message = {"role": "assistant", "content": content}
        if thinking:
            message["thinking"] = thinking
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {"message": message, "prompt_eval_count": pin, "eval_count": pout}

    def result(self, approach: str, answer: str, trace: list[str]) -> RunResult:
        return RunResult(approach, answer, trace, self.calls,
                         self.input_tokens, self.output_tokens, self.seconds,
                         self.thinking.strip())


# --- Aggregation -----------------------------------------------------------
def safe_ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, or NaN when the denominator is zero/falsy."""
    return numerator / denominator if denominator else float("nan")


def median_int(values) -> int:
    return int(statistics.median(values))


def summarize_runs(results: list[RunResult]) -> dict:
    """Fold N runs of one approach into a serializable summary."""
    return {
        "runs": [r.as_dict() for r in results],
        "median_total": median_int(r.total_tokens for r in results),
        "median_input": median_int(r.input_tokens for r in results),
        "median_output": median_int(r.output_tokens for r in results),
        "median_cost": median_int(r.cost for r in results),
        "median_calls": median_int(r.calls for r in results),
    }


# --- Console rendering -----------------------------------------------------
_WIDE = "=" * 74
_THIN = "-" * 74


def print_header(scenario_id: str, email: str) -> None:
    print(f"\n{_WIDE}\n{paint('SCENARIO: ' + scenario_id, BOLD, CYAN)}\n{_THIN}\nEMAIL:")
    for line in email.strip().splitlines():
        print(paint(f"  {line}", DIM))


def _tokens_line(result: RunResult) -> str:
    return (f"  TOKENS: in {result.input_tokens} + out {result.output_tokens} "
            f"= {paint(str(result.total_tokens), BOLD)}  |  "
            f"calls {result.calls}  |  {result.seconds:.1f}s")


def print_run(result: RunResult) -> None:
    color = {"workflow": GREEN, "agent": RED}.get(result.approach, MAGENTA)
    title = f" {result.approach.upper()} "
    print(f"\n{paint(title, BOLD, color)}{_THIN[len(title):]}")
    for step in result.trace:
        print(f"  {step}")
    print(f"  {paint('ANSWER:', BOLD)}")
    for line in (result.answer.splitlines() or ["(empty)"]):
        print(f"    {line}")
    print(_tokens_line(result))


def print_trace(result: RunResult) -> None:
    """Like print_run but without re-dumping the answer (already streamed live)."""
    for step in result.trace:
        print(f"  {step}")
    print(_tokens_line(result))


def print_comparison(workflow: RunResult, agent: RunResult, multi: RunResult = None) -> None:
    def line(other: RunResult, color: str) -> str:
        tok = safe_ratio(other.total_tokens, workflow.total_tokens)
        inp = safe_ratio(other.input_tokens, workflow.input_tokens)
        cost = safe_ratio(other.cost, workflow.cost)
        return (f"  {other.approach}: tokens {paint(f'{tok:.2f}x', color, BOLD)} "
                f"({paint(f'{inp:.2f}x', color)} input)  |  "
                f"real cost {paint(f'{cost:.2f}x', color, BOLD)}  |  calls {other.calls}")

    print(f"\n{paint(' COMPARISON (vs workflow)', BOLD)} {_THIN[25:]}")
    print(line(agent, YELLOW))
    if multi is not None:
        print(line(multi, MAGENTA))
    print(f"  baseline workflow: {workflow.total_tokens} tok, cost {workflow.cost}, "
          f"calls {workflow.calls}  (pricing out=1, in={INPUT_COST_RATIO})")


def print_summary(run: dict) -> None:
    scenarios = run["scenarios"]
    has_ma = any("multi_agent" in sc for sc in scenarios)
    heading = paint("SUMMARY: TOKENS vs REAL COST vs CORRECTNESS (median)", BOLD, CYAN)
    print(f"\n{_WIDE}\n{heading}\n{_THIN}")
    head = f"{'scenario':<12}{'wf tok':>8}{'ag tok':>8}{'ag x':>7}"
    if has_ma:
        head += f"{'ma tok':>8}{'ma x':>7}"
    head += "   correct (wf/ag" + ("/ma" if has_ma else "") + ")"
    print(head)
    for sc in scenarios:
        wf, ag = sc["workflow"]["median_total"], sc["agent"]["median_total"]
        row = f"{sc['id']:<12}{wf:>8}{ag:>8}{sc['multiplier']:>6.2f}x"
        oks = [paint("PASS", GREEN) if sc.get("workflow_correct") else paint("FAIL", RED),
               paint("PASS", GREEN) if sc.get("agent_correct") else paint("FAIL", RED)]
        if has_ma:
            ma = sc.get("multi_agent", {}).get("median_total", 0)
            row += f"{ma:>8}{sc.get('ma_multiplier', float('nan')):>6.2f}x"
            oks.append(paint("PASS", GREEN) if sc.get("multi_agent_correct")
                       else paint("FAIL", RED))
        print(f"{row}   {'/'.join(oks)}")
    toks = [sc["multiplier"] for sc in scenarios]
    n = len(scenarios)
    print(_THIN)
    print(f"model {run['model']} | cache {run['cache']} | pricing out=1.0 in={INPUT_COST_RATIO}")
    print(f"agent token multiplier {min(toks):.2f}-{max(toks):.2f}x")
    if has_ma:
        ma_toks = [sc.get("ma_multiplier", float("nan")) for sc in scenarios if "multi_agent" in sc]
        print(f"multi-agent token multiplier {min(ma_toks):.2f}-{max(ma_toks):.2f}x")
    parts = [("workflow", "workflow_correct"), ("agent", "agent_correct")]
    if has_ma:
        parts.append(("multi-agent", "multi_agent_correct"))
    summary = "  |  ".join(f"{name} {sum(bool(sc.get(key)) for sc in scenarios)}/{n}"
                           for name, key in parts)
    print(f"correctness: {summary}  (comparison fair only where all PASS)")

#!/usr/bin/env python3
"""Persistence and charting, decoupled from the model run.

A run is saved as a single JSON file (the complete record: model, cache mode,
every run's answer, trace and tokens, plus medians and multipliers). Charts are
generated FROM that file, so you can re-plot without paying for inference again.

    python report.py run_20260602_2100_off.json   # rebuild both charts from a saved run
"""

import json
import sys
from pathlib import Path

WF_IN, WF_OUT = "#1B5E20", "#81C784"   # workflow: dark / light green
AG_IN, AG_OUT = "#B71C1C", "#EF9A9A"   # agent:    dark / light red
MA_IN, MA_OUT = "#4A148C", "#CE93D8"   # multi:    dark / light purple


# --- Structured markdown transcript ----------------------------------------
# Built straight from the run dict, not from captured console output. Plain,
# idiomatic markdown (headings, a table, blockquotes, fenced code, collapsible
# <details>) so it renders everywhere, including renderers that strip inline
# CSS such as GitHub. No <pre>/<span style> blobs.

def _blockquote(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.strip().splitlines())


def _fence(text: str, marker: str = "```") -> str:
    return f"{marker}\n{text}\n{marker}"


def _ok(flag) -> str:
    return "✅" if flag else "❌"


def _costx(sc: dict) -> str:
    cm = sc.get("cost_multiplier")
    return f"{cm:.2f}×" if isinstance(cm, (int, float)) else "–"


def save_transcript_md(run: dict, path: str) -> None:
    """Render the run as a clean, structured markdown report."""
    scenarios = run["scenarios"]
    has_ma = any("multi_agent" in sc for sc in scenarios)
    out: list[str] = []
    out.append("# awbench run\n")
    out.append(
        f"**Model** `{run['model']}` · **cache** {run['cache']} · "
        f"**runs/scenario** {run['runs']} · **max-steps** {run.get('max_steps', '?')} · "
        f"{run['timestamp']}\n")
    out.append("Pricing for real cost: output = 1.0, input = 0.18 per token. "
               "All multipliers are relative to the workflow baseline.\n")

    # Summary table (columns grow when multi-agent is present).
    out.append("## Summary\n")
    if has_ma:
        out.append("| Scenario | WF tok | AG tok | MA tok | AG× | MA× | "
                   "WF cost | AG cost | MA cost | correct (WF/AG/MA) |")
        out.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|")
    else:
        out.append("| Scenario | WF tok | AG tok | tok× | WF cost | AG cost | cost× | "
                   "correct (WF/AG) |")
        out.append("|---|--:|--:|--:|--:|--:|--:|:--:|")
    for sc in scenarios:
        wf, ag = sc["workflow"], sc["agent"]
        if has_ma:
            ma = sc.get("multi_agent", {})
            out.append(
                f"| {sc['id']} | {wf['median_total']} | {ag['median_total']} | "
                f"{ma.get('median_total', '–')} | {sc['multiplier']:.2f}× | "
                f"{sc.get('ma_multiplier', float('nan')):.2f}× | "
                f"{wf.get('median_cost', '–')} | {ag.get('median_cost', '–')} | "
                f"{ma.get('median_cost', '–')} | "
                f"{_ok(sc.get('workflow_correct'))}/{_ok(sc.get('agent_correct'))}/"
                f"{_ok(sc.get('multi_agent_correct'))} |")
        else:
            out.append(
                f"| {sc['id']} | {wf['median_total']} | {ag['median_total']} | "
                f"{sc['multiplier']:.2f}× | {wf.get('median_cost', '–')} | "
                f"{ag.get('median_cost', '–')} | {_costx(sc)} | "
                f"{_ok(sc.get('workflow_correct'))}/{_ok(sc.get('agent_correct'))} |")

    n = len(scenarios)
    ag_toks = [sc["multiplier"] for sc in scenarios]
    note = (f"\n**Agent token multiplier** {min(ag_toks):.2f}–{max(ag_toks):.2f}× · ")
    if has_ma:
        ma_toks = [sc.get("ma_multiplier") for sc in scenarios if "multi_agent" in sc]
        note += f"**Multi-agent** {min(ma_toks):.2f}–{max(ma_toks):.2f}× · "
    parts = [("workflow", "workflow_correct"), ("agent", "agent_correct")]
    if has_ma:
        parts.append(("multi-agent", "multi_agent_correct"))
    note += "**Correctness** " + ", ".join(
        f"{name} {sum(bool(sc.get(key)) for sc in scenarios)}/{n}" for name, key in parts)
    note += " (comparison fair only where all pass).\n"
    out.append(note)

    approaches = [("workflow", "Workflow"), ("agent", "Agent")]
    if has_ma:
        approaches.append(("multi_agent", "Multi-agent"))

    # One section per scenario.
    for sc in scenarios:
        out.append(f"## {sc['id']}\n")
        if sc.get("email"):
            out.append("**Supplier email**\n")
            out.append(_blockquote(sc["email"]) + "\n")
        for key, label in approaches:
            if key not in sc:
                continue
            r = sc[key]["runs"][0]
            out.append(
                f"### {label} — {r['total_tokens']} tok "
                f"(in {r['input_tokens']} / out {r['output_tokens']}) · "
                f"cost {r.get('cost', '–')} · {r['calls']} calls\n")
            out.append("Steps:")
            out.extend(f"- `{step}`" for step in r["trace"])
            out.append("\nAnswer:")
            out.append(_fence(r["answer"] or "(empty)"))
            thinking = (r.get("thinking") or "").strip()
            if thinking:
                out.append("\n<details>")
                out.append(f"<summary>Reasoning ({r['output_tokens']} output tok)</summary>\n")
                out.append(_fence(thinking, "~~~"))      # tildes: model emits backticks
                out.append("</details>")
            out.append("")
        cmp_bits = [f"agent **{sc['multiplier']:.2f}× tokens** ({_costx(sc)} cost)"]
        if has_ma and "ma_multiplier" in sc:
            ma_costx = sc.get("ma_cost_multiplier")
            ma_costs = f"{ma_costx:.2f}×" if isinstance(ma_costx, (int, float)) else "–"
            cmp_bits.append(f"multi-agent **{sc['ma_multiplier']:.2f}× tokens** ({ma_costs} cost)")
        keys = ("workflow_correct", "agent_correct") + \
               (("multi_agent_correct",) if has_ma else ())
        all_ok = all(sc.get(k) for k in keys)
        verdict = (". All correct, identical deliverable.\n" if all_ok
                   else ". Check grades above.\n")
        out.append("**Comparison vs workflow:** " + "; ".join(cmp_bits) + verdict)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out), encoding="utf-8")
    print(f"Transcript saved to {path}")


def save_run(path: str, run: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRun saved to {path}")


def load_run(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _series(run: dict):
    """Return the list of (label, total_color, out_color, key, mult_key) to plot,
    including multi-agent only when it is present in the run."""
    s = [("Workflow", WF_IN, WF_OUT, "workflow", None),
         ("Agent", AG_IN, AG_OUT, "agent", "multiplier")]
    if any("multi_agent" in sc for sc in run["scenarios"]):
        s.append(("Multi-agent", MA_IN, MA_OUT, "multi_agent", "ma_multiplier"))
    return s


def _title_approaches(run: dict) -> str:
    return "Workflow vs Agent" + (" vs Multi-agent"
                                  if any("multi_agent" in sc for sc in run["scenarios"]) else "")


def build_total_chart(run: dict, out_path: str) -> None:
    """Grouped bars: median TOTAL tokens per approach, with multiplier labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = run["scenarios"]
    labels = [s["id"] for s in sc]
    x = range(len(labels))
    series = _series(run)
    width = 0.8 / len(series)
    offsets = [(-(len(series) - 1) / 2 + k) * width for k in range(len(series))]

    _, ax = plt.subplots(figsize=(10, 5.5))
    for (label, color, _out, key, mult_key), off in zip(series, offsets, strict=True):
        vals = [s.get(key, {}).get("median_total", 0) for s in sc]
        ax.bar([i + off for i in x], vals, width, label=label, color=color)
        if mult_key:
            for i, s in enumerate(sc):
                m = s.get(mult_key)
                if m is not None:
                    ax.annotate(f"{m:.2f}x", (i + off, s[key]["median_total"]),
                                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Total tokens (median)")
    ax.set_title(
        f"{_title_approaches(run)} total tokens  |  {run['model']}  |  cache {run['cache']}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")


def build_io_chart(run: dict, out_path: str) -> None:
    """Stacked bars: input (re-sent context) vs output (generation) per approach.
    The agent and multi-agent inputs are the context snowball; the multi-agent
    stack is the sum across the orchestrator and every sub-agent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = run["scenarios"]
    labels = [s["id"] for s in sc]
    x = range(len(labels))
    series = _series(run)
    width = 0.8 / len(series)
    offsets = [(-(len(series) - 1) / 2 + k) * width for k in range(len(series))]

    _, ax = plt.subplots(figsize=(11, 5.5))
    for (label, in_color, out_color, key, _m), off in zip(series, offsets, strict=True):
        xs = [i + off for i in x]
        ins = [s.get(key, {}).get("median_input", 0) for s in sc]
        outs = [s.get(key, {}).get("median_total", 0) - s.get(key, {}).get("median_input", 0)
                for s in sc]
        ax.bar(xs, ins, width, label=f"{label} input", color=in_color)
        ax.bar(xs, outs, width, bottom=ins, label=f"{label} output", color=out_color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Tokens (median)")
    ax.set_title(f"Input vs output token split  |  {run['model']}  |  cache {run['cache']}")
    ax.legend(ncol=len(series), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")


def render_all(run: dict, base_path: str) -> None:
    """Write both charts next to base_path: <base>_total.png and <base>_io.png."""
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)
    build_total_chart(run, f"{base_path}_total.png")
    build_io_chart(run, f"{base_path}_io.png")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python report.py <run.json>")
        sys.exit(1)
    json_path = sys.argv[1]
    run = load_run(json_path)
    base = str(Path(json_path).with_suffix(""))
    save_transcript_md(run, f"{base}.md")
    render_all(run, base)


if __name__ == "__main__":
    main()

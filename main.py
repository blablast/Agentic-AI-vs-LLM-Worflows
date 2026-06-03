#!/usr/bin/env python3
"""Workflow LLM vs. Agentic AI: a token-cost sweep with verbose console output.

Both approaches solve the SAME task on the SAME local model and must produce the
SAME deliverable (at-risk parts, supplier, quantity, cost), so the token
comparison is apples-to-apples. Each answer is graded against ground truth, so
cost is reported alongside correctness, not in place of it.

The whole run is saved to JSON; charts are generated from that file via report.py.

    ollama pull qwen3:8b
    pip install -e .                   # add '.[chart]' for --chart (matplotlib)
    python main.py --model qwen3:8b --runs 3 --chart
    python main.py --model qwen3:8b --runs 3 --multi-agent --chart   # add the 3rd bar
    python main.py --model qwen3:8b --runs 1 --no-cache   # stateless billing view
"""

import argparse
from dataclasses import replace
from datetime import datetime

from agent import run_agent
from data import SCENARIOS
from logic import grade
from multiagent import run_multi_agent
from report import render_all, save_run, save_transcript_md
from utils import (
    RunConfig,
    RunResult,
    print_comparison,
    print_header,
    print_run,
    print_summary,
    print_trace,
    safe_ratio,
    summarize_runs,
)
from workflow import run_workflow


def _scenario_record(scenario: dict, wf_runs: list[RunResult], ag_runs: list[RunResult],
                     ma_runs: list[RunResult], multi_agent: bool) -> dict:
    """Fold a scenario runs into one serializable record: medians, the agent's
    token/cost multipliers over the workflow baseline, and correctness grades."""
    expected = scenario["expected"]
    wf_sum, ag_sum = summarize_runs(wf_runs), summarize_runs(ag_runs)
    wf_tot, wf_cost = wf_sum["median_total"], wf_sum["median_cost"]
    wf_grade = grade(wf_runs[0].answer, expected)
    ag_grade = grade(ag_runs[0].answer, expected)
    record = {
        "id": scenario["id"], "expected": expected, "email": scenario["email"],
        "workflow": wf_sum, "agent": ag_sum,
        "multiplier": round(safe_ratio(ag_sum["median_total"], wf_tot), 3),
        "cost_multiplier": round(safe_ratio(ag_sum["median_cost"], wf_cost), 3),
        "workflow_correct": wf_grade["correct"], "agent_correct": ag_grade["correct"],
        "workflow_grade": wf_grade, "agent_grade": ag_grade,
    }
    if multi_agent:
        ma_sum = summarize_runs(ma_runs)
        ma_grade = grade(ma_runs[0].answer, expected)
        record.update({
            "multi_agent": ma_sum,
            "ma_multiplier": round(safe_ratio(ma_sum["median_total"], wf_tot), 3),
            "ma_cost_multiplier": round(safe_ratio(ma_sum["median_cost"], wf_cost), 3),
            "multi_agent_correct": ma_grade["correct"], "multi_agent_grade": ma_grade,
        })
    return record


def run_sweep(args, cfg: RunConfig, cache: str, stamp: datetime) -> dict:
    print(f"Model (shared): {cfg.model} | runs/scenario: {args.runs} | "
          f"cache: {cache.upper()} | stream: {'ON' if args.stream else 'OFF'}")

    scenarios_out = []
    for scenario in SCENARIOS:
        print_header(scenario["id"], scenario["email"])

        wf_runs, ag_runs, ma_runs = [], [], []
        for i in range(args.runs):
            stream0 = args.stream and i == 0
            run_cfg = replace(cfg, stream=stream0)
            if stream0:
                print("\n WORKFLOW (streaming) " + "-" * 52)
            wf = run_workflow(run_cfg, scenario["email"])
            if stream0:
                print("\n AGENT (streaming) " + "-" * 55)
            ag = run_agent(run_cfg, scenario["email"])
            wf_runs.append(wf)
            ag_runs.append(ag)
            ma = None
            if args.multi_agent:
                if stream0:
                    print("\n MULTI-AGENT (streaming) " + "-" * 49)
                ma = run_multi_agent(run_cfg, scenario["email"])
                ma_runs.append(ma)
            if i == 0:
                show = print_trace if args.stream else print_run
                show(wf)
                show(ag)
                if ma is not None:
                    show(ma)
                print_comparison(wf, ag, ma)
            else:
                extra = f" / multi {ma.total_tokens} tok" if ma is not None else ""
                print(f"\n  run {i + 1}/{args.runs}: "
                      f"workflow {wf.total_tokens} tok / agent {ag.total_tokens} tok{extra}")

        scenarios_out.append(
            _scenario_record(scenario, wf_runs, ag_runs, ma_runs, args.multi_agent))

    run = {
        "model": cfg.model, "cache": cache, "runs": args.runs,
        "max_steps": cfg.max_steps, "timestamp": stamp.isoformat(timespec="seconds"),
        "scenarios": scenarios_out,
    }
    print_summary(run)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:8b",
                        help="one model for BOTH approaches (must support tool calling)")
    parser.add_argument("--runs", type=int, default=3, help="runs per approach per scenario")
    parser.add_argument("--max-steps", type=int, default=15, help="agent step limit")
    parser.add_argument("--multi-agent", action="store_true",
                        help="also run the multi-agent orchestrator (orchestrator + one "
                             "sub-agent per delayed part + aggregation), as a third bar.")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable Ollama KV reuse (keep_alive=0, unloads between calls). "
                             "SLOW, but shows the stateless billing cost a cloud API charges.")
    parser.add_argument("--stream", action="store_true",
                        help="stream output live including thinking chunks (think=True). "
                             "For watching and calibrating; not the canonical measurement run.")
    parser.add_argument("--out", default=None,
                        help="path for the run JSON (default: auto timestamp)")
    parser.add_argument("--chart", action="store_true",
                        help="also save PNG charts (needs matplotlib)")
    args = parser.parse_args()

    cache = "off" if args.no_cache else "on"
    stamp = datetime.now()
    cfg = RunConfig(model=args.model, max_steps=args.max_steps,
                    keep_alive=0 if args.no_cache else None)

    run = run_sweep(args, cfg, cache, stamp)

    base = args.out[:-5] if (args.out and args.out.endswith(".json")) \
        else (args.out or f"results/run_{stamp:%Y%m%d_%H%M%S}_{cache}")
    save_run(f"{base}.json", run)
    save_transcript_md(run, f"{base}.md")
    if args.chart:
        render_all(run, base)


if __name__ == "__main__":
    main()

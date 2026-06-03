"""Deterministic LLM workflow: extract -> business logic (0 tokens) -> report.

The number of LLM calls is fixed at 2 regardless of how many parts are delayed.
The risk/shortfall/cost are computed in Python; the LLM only formats the report.
"""

import json

from logic import assess_part
from prompts import WORKFLOW_EXTRACT, WORKFLOW_REPORT
from utils import RunConfig, RunResult, TokenMeter


def run_workflow(cfg: RunConfig, email: str) -> RunResult:
    meter = TokenMeter(cfg)
    trace: list[str] = []

    # Step 1 (LLM): extract every delayed part, JSON forced.
    extracted = meter.chat(
        [{"role": "user", "content": WORKFLOW_EXTRACT + email}],
        fmt="json",
    )
    try:
        delays = json.loads(extracted["message"]["content"]).get("delays", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        delays = []
    trace.append(f"1 [LLM]  extract delays  (+{meter.last_in} in / +{meter.last_out} out)  "
                 f"-> {[d.get('part_id') for d in delays]}")

    # Step 2 (plain Python, 0 tokens): assess every part (risk, shortfall, cost).
    facts = [assess_part(str(d.get("part_id", "")), int(float(d.get("eta_days", 0))))
             for d in delays]
    at_risk = [f["part_id"] for f in facts if f.get("stockout_risk")]
    trace.append(f"2 [code] assess + cost, 0 tokens  -> at risk: {at_risk or 'none'}")

    # Step 3 (LLM): format the report from precomputed facts (same deliverable as agent).
    report = meter.chat(
        [{"role": "user", "content": WORKFLOW_REPORT + json.dumps(facts)}])
    trace.append(f"3 [LLM]  write report  (+{meter.last_in} in / +{meter.last_out} out)")

    return meter.result("workflow", report["message"]["content"].strip(), trace)

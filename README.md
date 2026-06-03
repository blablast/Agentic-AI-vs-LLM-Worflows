# Agentic AI vs LLM Workflows

A small, reproducible benchmark that compares three architectures on the **same task**, the **same local model**, and the **same strict output**, then measures token cost and money cost against correctness.

- **Workflow**: the model and tools follow a fixed path in code (2 LLM calls; the business logic runs in Python at 0 tokens).
- **Single agent**: a native tool-calling loop; the model decides what to call and when.
- **Multi-agent**: a lead orchestrator splits the task into parts, runs one sub-agent per part, then aggregates.

The point: when all three are forced to produce the exact same answer, the extra autonomy is pure overhead, not quality.

## TL;DR result

Local run, `qwen3.6:latest`, 3 runs each, temperature 0. Median total tokens, multiplier vs the workflow baseline.

| Scenario  | Parts | Workflow (tok) | Single agent | Multi-agent |
| --------- | -----:| --------------:| ------------:| -----------:|
| S1 simple | 1     | 2773           | 1.45x        | 2.05x       |
| S2 simple | 1     | 2461           | 1.85x        | 2.95x       |
| S3 medium | 2     | 3199           | 1.90x        | 2.73x       |
| S4 hard   | 2     | 3519           | 1.49x        | 2.74x       |
| S5 harder | 3     | 3514           | 1.56x        | 3.52x       |

**All three architectures returned a byte-identical, correct answer in all 5 scenarios (5/5 each).**
Single agent: ~1.5-1.9x tokens. Multi-agent: ~2.0-3.5x tokens and ~2.1-3.25x real money. Zero quality gained on this task.

The multi-agent grows **linearly** with the number of parts (4, 4, 6, 6, 8 LLM calls), not exponentially, because it is bounded on purpose: the orchestrator decomposes once (no recursion), each sub-agent gets one part and its own small context, tools are focused, and the output is locked to a strict contract.

## Charts

![Total tokens: workflow vs agent vs multi-agent](run_20260603_151333_on_total.png)
![Input vs output token split](run_20260603_151333_on_io.png)

> Charts are produced by `--chart`. Commit the two PNGs your run generates (e.g. `run_<timestamp>_on_total.png` and `_io.png`) next to this README, or update the paths above.

## Run it

```bash
ollama pull qwen3.6:latest          # any tool-calling model works
pip install ollama matplotlib       # matplotlib only needed for --chart

# the headline run: all three architectures, 3 runs, with charts
python main.py --model qwen3.6:latest --runs 3 --multi-agent --chart
```

Useful flags:

- `--multi-agent`: add the multi-agent architecture as a third bar (off by default; workflow vs single agent only).
- `--runs N`: runs per architecture per scenario (medians smooth out variance; 3+ recommended).
- `--no-cache`: disable Ollama KV reuse (`keep_alive=0`). Slower; shows the stateless billing view.
- `--stream`: stream output live, including the model's thinking. For watching, not for the canonical measurement.
- `--max-steps N`: agent step limit (default 15).
- `--out PATH` / `--chart`: output path for the run JSON / also save PNG charts.

Re-plot or rebuild the markdown report from a saved run without paying for inference again:

```bash
python report.py results/run_<timestamp>_on.json
```

## Repository layout

| File            | Role                                                                                                     |
| --------------- | -------------------------------------------------------------------------------------------------------- |
| `data.py`       | Pure data and config: inventory, alternates, 5 scenarios, the strict output contract, pricing constants. |
| `logic.py`      | Deterministic business rules and answer grading (0 tokens).                                              |
| `utils.py`      | Agent tools, the token meter, the result type, console + summary printers.                               |
| `workflow.py`   | The fixed 2-call workflow (extract → assess in Python → format).                                         |
| `agent.py`      | The single-agent tool-calling loop.                                                                      |
| `multiagent.py` | The orchestrator + per-part sub-agents + aggregation.                                                    |
| `report.py`     | Save/load runs, build the two charts, build a clean markdown transcript.                                 |
| `main.py`       | CLI and the sweep across scenarios.                                                                      |

## How the comparison stays fair

- **Same model** for all three architectures.
- **Same strict deliverable**: one line per delayed part, identical format. No architecture is allowed to be more verbose than another.
- **Graded against ground truth**: cost is reported next to correctness, never instead of it.
- **Tokens are the real billed cost**: `prompt_eval` counts the full context every turn regardless of KV cache reuse, so the measured numbers are what a cloud API would charge.

## Pricing model

Money cost normalizes output to `1.0` and weights input at `0.18` (`OUTPUT_COST` / `INPUT_COST_RATIO` in `data.py`). This reflects 2026 frontier pricing, where output is the expensive side at roughly a 1:5 to 1:6 input:output ratio (Anthropic, OpenAI, Google). Change those two constants to match your provider.

## Caveats

One model, one domain (supply-chain delay triage), five scenarios, a bounded and well-specified task. The multiplier is a function of trajectory length and design, not of the word "multi-agent". On long, open-ended tasks (deep research, large coding jobs) the cost can explode far past these numbers; on a task you can fully specify, the workflow wins.

## Write-up

Full article and sources: see Blazej Strus's LinkedIn post.

## License

MIT. See `LICENSE`.

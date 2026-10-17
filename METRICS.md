# MARS-SQL ablation runs

Run commands from `mars_ablation` in the project virtual environment. The BIRD
training and development datasets are not included in this repository. Place
them under `data/` or replace the paths below with their actual locations.

For a 100-question routing and voting smoke test that does not require compiled
artifacts:

```powershell
.\venv\Scripts\Activate.ps1
python run_ablation.py --dev-json data\dev.json --db-dir data\dev_databases --limit 100 --selection self_consistency --prompt-mode uncompiled_dspy
```

This is a live OpenRouter run: it sends eight 7B rollouts per question. The
provider's actual returned model ID must be the pinned 7B model; no 32B fallback
is allowed. It uses temperature 0.8, top-p 0.7, five Generator decisions per
trajectory and a 30-second SQL time budget. Adjust `--workers` to limit
concurrent provider requests; this changes wall latency, not token expense.
For a cheaper initial check, use `--limit 1 --candidates 1 --workers 1`.

The root `compiled_dspy_model.json` from the earlier milestone has an obsolete
SQL-only signature and must be recompiled for the probe/final action format.
To produce the compiled generator and a training-only verifier, use a BIRD
training split (never the development split):

```powershell
python -m src.dspy_optimizer --mode generator --train-json data\train.json --db-dir data\train_databases --limit 64
python -m src.dspy_optimizer --mode collect --train-json data\train.json --db-dir data\train_databases --limit 32 --output artifacts\training\predict.json
python -m src.dspy_optimizer --mode verifier --train-json data\train.json --db-dir data\train_databases --limit 32 --training-trajectories artifacts\training\predict.metrics.json
python run_ablation.py --dev-json data\dev.json --db-dir data\dev_databases --limit 100 --selection dspy_verifier --prompt-mode compiled_dspy
```

The verifier run saves the DSPy-selected `predict.json` and
`predict.self_consistency.json` from the same eight candidate trajectories.
Their EX scores appear side by side in `predict.metrics.json`. Default verifier
cost is one rating per eligible trajectory (`--verification-samples 1`). Use
`--verification-samples 8` for the paper's validation sample count. The DSPy
score is an elicited 0–1 logical-validity rating, not the paper's fine-tuned
Yes-token probability; label these as distinct selection methods.

The runner writes output under `artifacts/ablation/<selection>/` unless
`--output` is supplied. Per-question candidate histories, results, selection,
model IDs, prompt/completion tokens, OpenRouter USD cost and latencies are in
`predict.metrics.json`. The manifest records split and compiled-artifact hashes,
sampling configuration, software versions and candidate IDs. Cost/tokens
aggregate all trajectories and judge calls, including failed rollouts.
Provider usage is `null` when the provider supplies no exact accounting; it is
never estimated from character counts. Compiler calls and their one-time cost
are recorded in `compiled_dspy_model.compilation.metrics.json` and
`compiled_dspy_verifier.compilation.metrics.json` separately from inference.

`turn` counts Generator decisions; `sql_execution_count` counts SQLite
executions; `self_correction_cycles` counts decisions after execution errors.
Probes can execute successfully and continue. Only an explicit executable
`final_answer` becomes a prediction. Generation or selection failures are JSON
`null`, remain in the EX denominator, and cannot accidentally match gold SQL.
Empty result sets remain eligible because they may be the correct answer.

The local EX scorer compares sets of complete result rows, ignoring order and
duplicate multiplicity, as the [BIRD reference evaluator](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py)
does. Prediction and gold execution share a 30-second deadline. The prediction
file retains dataset order; keep the exact same question subset when scoring
with any external evaluator. The scorer alone can run without API calls:

```powershell
python -m src.bird_evaluator --dev-json data\dev.json --db-dir data\dev_databases --output artifacts\ablation\self_consistency\predict.json --limit 100 --score-only
```

`python live_test.py` runs one temporary SQLite query using the uncompiled DSPy
prompt; `python live_test.py --prompt-mode compiled_dspy` checks a newly
compiled trajectory artifact. Both print EX, usage, cost and Langfuse trace ID.
The root Langfuse query span holds aggregate metadata, while child generations
hold billable usage. A dash on the root span's Cost cell can therefore be
normal; inspect the trace total and child generation Usage fields.

For an ablation claim, compare N=1 and N=8 using the same model, dataset,
schema pruner, prompt mode and iteration budget. Report the paired
self-consistency and DSPy selection scores from one candidate pool. The paper's
77.84% is a trained three-agent reference result, so it is not a guaranteed
target for this prompt-compiled system.

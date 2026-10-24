# MARS-SQL ablation runs

Run commands from `mars_ablation` in the project virtual environment. The BIRD
training and development datasets are not included in this repository. Place
them under `data/` or replace the paths below with their actual locations.

Run the live six-configuration pre-flight with:

```powershell
python smoke_test.py
```

This sequentially crosses `uncompiled_dspy`/`compiled_dspy` with
`self_consistency`/`dspy_verifier`/`hybrid_cascade`, using the first three
development questions, eight candidates, five iterations and one judge sample.
The verifier-only condition and Hybrid tie-breaker both use the same compiled
DSPy rating judge. Its artifact is checked before any subprocess is launched;
use `--verifier-model` for a nondefault artifact path.
Override dataset, database and compiled-model paths with `--dev-json`, `--db-dir`,
`--compiled-model` and `--verifier-model` as needed.

Every invocation creates `artifacts/smoke_test/<UTC timestamp>_<unique ID>/`,
containing six separate condition directories. Each stores `command.json`,
`run.log`, predictions, metrics, and a checkpoint. The wrapper stops immediately
on nonzero subprocess exit, timeout (30 minutes per condition by default),
missing final SQL, question/selection/execution errors, configuration mismatch,
or absent/nonfinite token/cost accounting. `--timeout-seconds` overrides the
per-condition deadline. EX is reported but is not required to be 100%.
Only six validated runs produce `summary.json` and the message
`Matrix Smoke Test Fully Passed`. This is a paid live test; ordinary offline
tests do not invoke it. The three questions do not guarantee that every
stochastic probe, repair or hybrid tie path occurs.

Batch runs now durably save each fully evaluated question in
`<output stem>.checkpoint.jsonl`, including predictions, trajectories, exact
accounting, selection, EX, and the paired baseline. On a restart, run the
**same original command and output path**, adding `--resume`, for example:

```powershell
python run_ablation.py --dev-json data\dev.json --db-dir data\dev_databases --limit 1534 --selection hybrid_cascade --prompt-mode compiled_dspy --output artifacts\full\compiled_hybrid\predict.json --resume
```

Omit `--resume` on the first run. New runs refuse to overwrite existing progress;
use a fresh output path for a new experiment. Resume checks dataset hash, exact
question IDs/order, model and sampling controls, compiled-artifact hashes,
source hashes, dependency versions, and database location/size/mtime. Database
metadata is an inexpensive change check, not a cryptographic hash of 30 GB.
Keep those inputs and the checkpoint on persistent storage unchanged.

An OS lock prevents two writers from using the same checkpoint. Each committed
record has a checksum and is flushed/fsynced. An incomplete final journal line
is discarded after a crash; corrupt committed records are rejected. Standard
prediction/metrics JSON files are assembled at completion and atomically
replaced; resume can rebuild them entirely from the journal without repeating
model calls or EX scoring for completed questions.

If questions 1 through 1,499 are committed and the process dies during question
1,500, resume skips the first 1,499 and repeats question 1,500. Candidate-level
in-flight work is not checkpointed; its already billed API calls may need to be
repeated and are not recoverable in the local accounting. Completed failures
and null predictions are also committed and stay in the EX denominator; resume
does not silently retry only incorrect/failed questions. Older runs lacking a
journal cannot recover progress lost before their final files were written.

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

Compiled JSON includes saved Signature instructions. Recompile the generator
when those instructions change, including the strict identifier-grounding rules.
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

Verifier compilation deterministically shuffles and interleaves training-only
EX-positive and EX-negative trajectories. Bootstrap acceptance has a quota of
four demonstrations per class (or fewer when the available class has fewer
than four examples). Once a class reaches its quota, further examples from that
class are rejected by the metric so that early stopping cannot fill all eight
slots with one class. Each saved demo must still be a model-generated rating
that agrees with its execution-derived training label. Labels are never model
inputs. The manifest and compilation metrics record class counts, quotas,
accepted demonstrations and the trajectory-file hash. If either quota cannot
be met, compilation preserves the prior artifact and records the failure plus
measured usage; no one-class judge is saved. Reuse collected trajectories when
retrying compilation rather than collecting them again.

For execution voting with the DSPy rating judge only on tied top groups:

```powershell
python run_ablation.py --dev-json data\dev.json --db-dir data\dev_databases --limit 100 --selection hybrid_cascade --prompt-mode uncompiled_dspy
```

`hybrid_cascade` uses exactly the same complete-result hashes and candidate
eligibility as `self_consistency`. A unique largest group wins with zero judge
calls (including a 3-2-2-1 vote); its earliest candidate is selected. When two or
more groups tie for the highest count, only trajectories in those groups go to
`dspy_verifier`. Lower-count groups and failed executions are excluded.
No eligible candidates means a failed selection without a judge call. A tie
between judge scores is resolved by the earliest original candidate index.

Both standalone and Hybrid selection use `TrajectoryVerifier` from
`src/dspy_verifier.py`: it returns a rationale and an elicited logical-validity
rating in [0, 1]. Multiple samples are averaged when requested; the highest
rating wins, with earliest original candidate index breaking ties. This is a
rating, not a calibrated probability or the paper's SFT Yes-token score. The
judge receives sanitized trajectories, question, evidence and schema, with no
gold SQL or EX labels. Partial/invalid rankings fail explicitly and retain
measured accounting. No API call requests token probabilities.

Both conditions require a compatible `compiled_dspy_verifier.json` (or
`--verifier-model` path). Hybrid loads the judge only when tied groups need it;
unique top groups make zero judge calls. The standalone CLI mode
`generative_verifier` is retired. The legacy Python import delegates to the
rating selector and emits a deprecation warning; it cannot make old probability
requests. Historical artifacts are unchanged. Start new experiments after this
methodology change; old checkpoints cannot be resumed under changed code or
judge artifacts.

These modes also save paired self-consistency predictions and EX from the same
candidate pool. For a cascade, `manifest.selection_mode` is `hybrid_cascade` and
each `queries[].selection.cascade_tie_breaker_engaged` is `true` only when the
judge path was triggered (including failed judge attempts). Selection metadata
includes vote groups, original tied candidate indices, judge ratings/rationales,
and selection source. Query totals include all rollouts plus actual judge calls;
the paired self-consistency totals exclude judge cost.

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

To compare an existing run with what Langfuse actually ingested, without another
LLM call or uploading any traces:

```powershell
python verify_langfuse.py --metrics artifacts\ablation\hybrid_cascade\predict.metrics.json --limit 1
```

This command uses the project's configured Langfuse credentials for read-only
API requests. It prints the exact generation input/output token counts, total
USD costs, a local-versus-server comparison, and direct dashboard links. Use
`--limit 24` to check more traces, or `--output artifacts\langfuse_check.json`
to retain the numeric comparison. A nonzero exit status indicates missing or
mismatched accounting or an API error. New stage records include `trace_id`
and `observation_id`, allowing comparison with individual server generations;
older reports are checked by per-trace totals. Selector traces are checked when
their stage IDs are present.

In the dashboard, follow the printed link and inspect the `mars_decomposer`,
`mars_generator`, or verifier **generation**, including its Usage/Cost details.
The parent `mars_sql_query` is a **span**: its own usage/cost fields are empty,
while the trace total aggregates its generation children. Adding the aggregate
usage to a second parent generation would count each model call twice.

The authoritative mapping is in `src/dspy_telemetry.py`: OpenRouterLM captures
LiteLLM's response before DSPy parses it, and `generation_telemetry()` supplies
`usage_details={"input": prompt_tokens, "output": completion_tokens, "total": total_tokens}`
and `cost_details={"total": usage.cost}` to each generation's `update()` call.
Exact USD costs therefore do not depend on a Langfuse model-price lookup.
Local JSON completeness alone does not prove delivery; the read-back command
checks the API ingestion boundary, including project/host or export failures.

For an ablation claim, compare N=1 and N=8 using the same model, dataset,
schema pruner, prompt mode and iteration budget. Report the paired
self-consistency and DSPy selection scores from one candidate pool. The paper's
77.84% is a trained three-agent reference result, so it is not a guaranteed
target for this prompt-compiled system.

Generator feedback renders SQL and decoded SQLite observations as plain text;
JSON remains the storage format for history, never the prompt rendering format.
The final allowed decision is submitted as `final_answer` even if the model
requests a probe. History records `final_action_forced: true` for that override.
Earlier probes still continue, and SQLite must accept the submitted SQL before
it becomes a prediction. This budget policy applies equally to compiled and
uncompiled generation and to training rollouts; it adds no extra model call.
An invalid, empty, or missing model response can still fail generation.

Telemetry schema 4 makes each `predict.metrics.json` query record self-contained:
`final_sql`, `configuration`, `execution_correct`, `status`, and
`degradation_reasons` accompany token fields. `turn` remains the sum across
rollouts for compatibility; `total_rollout_turns`, `selected_trajectory_turns`,
`selected_self_correction_cycles`, and `per_candidate_turns` distinguish total
work from the selected answer. Each candidate retains completed decisions,
`generator_calls_attempted` (including a failed generator invocation), history,
last attempted SQL, and its termination reason. Missing provider usage remains
null with `accounting_complete: false`; `observed_*` retains measurable usage.

Raised generation/setup/selection/scoring exceptions and caught SQLite errors
include phase, exception type, message, and chained traceback without frame
locals. `telemetry_errors.py` redacts configured credentials and common API-key
and Bearer patterns. Both checkpoint append and atomic final writes apply
redaction, including nested stage errors. This is not general anonymization:
questions, SQL, database samples and local source paths remain research data.

Completed question records are fsynced into `predict.checkpoint.jsonl` before
the progress message. Final JSON files are atomically published after the batch;
resume reconstructs them from verified journal records. An interrupted in-flight
question is not committed: its unfinished calls/charges cannot be fully recovered
from this question-level journal. Keep provider billing/trace records to account
for that work. Changed source/configuration is rejected on resume.

`run_ablation.py` now exits nonzero after publishing a report with incomplete
accounting or no selected SQL for the entire batch. Incorrect-but-executable
answers still count toward EX normally. This detects invalid completed runs;
it does not stop the evaluator immediately on the first provider billing error.

Set `VLLM_BASE_URL=https://YOUR-POD-ID-8000.proxy.runpod.net/v1` to route both
DSPy (generation, compilation and rating selection) and the raw graph client to
RunPod vLLM. A blank value preserves OpenRouter. `llm_endpoint.py` normalizes
the URL, uses `RUNPOD_DUMMY_KEY`, and maps the pinned OpenRouter checkpoint to
`Qwen/Qwen2.5-7B-Instruct`. Serve that checkpoint under that exact model name.
The DSPy transport uses `openai/` with the custom base URL. Model validation
recognizes both checkpoint names; cross-model fallback remains disabled.
No separate ChatOpenAI initializer exists in this project: LangChain orchestrates
the graph around the compatible OpenAI client and DSPy predictors.

Raw OpenAI `usage.prompt_tokens`, `usage.completion_tokens` and
`usage.total_tokens` are captured before DSPy parsing, including parse failures.
vLLM's missing monetary usage field does not discard tokens: its API fee is
recorded as zero with `cost_source=self_hosted_api_fee_excludes_gpu_rental`.
Manifest and query configuration explicitly record
`cost_scope=api_usage_fee_excludes_gpu_rental`. This is NOT the RunPod compute
cost. Record GPU rental/uptime separately for comparisons of total USD cost.
Missing token usage still stays null and fails accounting validation; there is
no unconditional guarantee of counts when the server fails to return usage.
Backend, base URL, served model and cost scope are checkpoint controls; resume
rejects a transport change rather than mixing OpenRouter and vLLM measurements.

The CLI default outputs are now
`results/runpod_vllm_test_01/<prompt_mode>/<selection>/predict.json` with adjacent
metrics and checkpoint files. Explicit `--output` still takes precedence, and
existing files are never silently overwritten. Set the vLLM URL before running:

```powershell
python run_ablation.py --dev-json data\dev.json --db-dir data\dev_databases --limit 3 --selection self_consistency --prompt-mode uncompiled_dspy
```

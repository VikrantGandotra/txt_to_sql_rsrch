# Forge prompt-volume and SQL-error analysis

These are offline, descriptive analyses of **compiled vs uncompiled
`self_consistency_verifier`**. They make no model calls, do not execute SQL, and
never modify the raw recovery files. Dependencies are separate from the frozen
experiment requirements: `python -m pip install -r analysis/requirements.txt`.

From `mars_ablation`, run all three analyses:

```powershell
python -m analysis.forge_analysis --root "C:\Users\Vikrant Gandotra\OneDrive\Desktop\mars_sql_project\MARS_Final_20261029" --verify-checkpoints
```

Or run a specific analysis:

```powershell
python -m analysis.token_volume_accuracy
python -m analysis.error_categorization
python -m analysis.attention_proxy
```

All accept `--root`, `--output`, `--source auto|json|jsonl`, and
`--verify-checkpoints`. Default resolution uses **extracted_slot_1 only**;
it never merges duplicate snapshots from slot 0. `--source jsonl` streams
checksummed question records instead of loading the final metrics JSON.
`--verify-checkpoints` compares final JSON records to the checksummed journal.
Default outputs: `analysis_outputs/forge_context/`, ignored by Git.

## Meaning of the measurements

- Actual final correctness is `execution_correct` in the saved metrics;
  `correct` in output CSVs is that boolean encoded as 0/1. A verifier's rating
  or successful SQLite execution is **not** substituted for EX.
- `query_prompt_tokens`: sum across generation rollouts **and selector calls**.
  This measures query compute volume, **not one context-window length**.
- `mean_generator_prompt_tokens` and `max_generator_prompt_tokens`: actual
  per-request inputs from generator stages' provider call measurements, excluding
  decomposer and judge calls. Known requests remain descriptive observations if
  other requests failed; measured/missing request coverage is exported.
- `first_generator_prompt_tokens`: mean first-turn request length across
  rollouts, before their SQL-error histories can inflate later prompts. This is
  the more useful temporal proxy, although schema complexity remains confounded.
- SQLite errors come from `candidates[].history[].observation.error_message`
  and history error details. The JSON-encoded observation is decoded. One
  history entry is one SQL attempt; final-scoring errors are not counted as
  additional generation attempts. Top-level model parse/transport errors are
  not mislabeled as SQLite hallucinations.
- `OperationalError` is **not synonymous with hallucinated columns**. The tables
  distinguish `no_such_column`, `no_such_table`, `ambiguous_column`, syntax,
  other OperationalErrors, and other execution errors. Even `no_such_column`
  may reflect aliasing/scope mistakes; it is evidence of an invalid identifier,
  not direct evidence of forgetting DDL. Successful repairs still count as
  encountered errors, and are separated from final EX.

## Output files

- `coverage.csv`, `provenance.json`: denominators, accounting coverage, input
  hashes, controls and dependency versions. Source hashes permit reproducibility;
  they do not provide causal identification.
- `query_measurements.csv`: compact per-query metrics; unknown exact token totals
  stay empty. Observed partial sums are exported separately, never substituted.
- `sqlite_attempts.csv`: per-attempt errors and prompt-length linkage.
- `accuracy_bins_*.csv/.png/.pdf`: EX by [0,4000), [4000,8000), [8000,12000),
  [12000,infinity) input tokens. Three exposures are plotted separately. Empty
  bins display n=0 rather than a fabricated zero-percent accuracy. Error bars
  are 95% Wilson intervals; labels show sample size.
- `paired_queries.csv`, `degraded_queries.csv`: exact paired IDs and diagnostics.
  `improved_queries.csv` preserves the opposite direction rather than selectively
  reporting only degradations. Attempt rows flag the selected candidate.
- `degraded_sqlite_errors.csv`: recorded errors, details and failed SQL for
  the questions uncompiled got right and compiled got wrong, for **both modes**.
- `error_frequency.csv`: overlapping query-level error incidences, with explicit
  denominators, for degraded questions and all questions. Categories need not
  sum to 100%, because a query may encounter several errors.
- `paired_accuracy_transitions.csv`, `paired_accuracy_statistics.json`: all paired
  outcomes, directional discordances, and exact McNemar/binomial test. This
  tests paired correctness differences, not their mechanism.
- `column_error_bins_*`, `first_prompt_vs_column_errors_scatter.*`: query-level
  associations. Jitter changes only scatter display, not the data.
- `column_error_correlations.csv`: mode-specific Spearman coefficients and
  exploratory p-values using queries as units (not dependent rollout turns).
- `attempt_column_error_rates.csv`, `first_turn_column_error_rates.csv`:
  descriptive SQL-attempt error rates; these are not independent-trial tests.

## Limits on conclusions

This analysis can support or contradict an association. It **cannot prove**
context overload, attention dilution, or the Lost in the Middle mechanism.
No attention weights, exact full rendered prompt positions, or controlled
interventions on demonstration length are present here. Longer DDL, query
difficulty, additional repair turns and additional judge work can all produce
longer token totals. Looking only at degraded queries selects on the outcome;
the all-query frequency table is included to expose that selection.

Do not interpret a 12k+ aggregate bin as crossing a single-request context
threshold. If every query falls in that bin, the aggregate plot is uninformative
about context length; consult the per-request and first-turn plots instead.
Keep all failed questions in EX denominators and disclose token-data exclusions.
The multiple exploratory correlations are not adjusted confirmatory tests.
Paired controlled experiments varying demo count/length while holding schema,
question and sampling fixed would be necessary to study the proposed mechanism.

Diagnostics may contain SQL and database identifiers. Basic key redaction is
applied, but manually review exports before sharing with an external AI. Raw
archives, credentials and databases are not needed in an AI analysis package.

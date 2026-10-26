#!/usr/bin/env bash
# Separate recovery policy; never edit checkpointed Python or experimental controls.
set -Eeuo pipefail
umask 077
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=true; shift; fi
OUTPUT_ROOT="${1:-results/full_dev_vllm_01}"
PYTHON="${MARS_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export DSPY_CACHEDIR="$SCRIPT_DIR/.dspy_cache"
export LITELLM_LOCAL_MODEL_COST_MAP=True
CURRENT_CONFIG=preflight
MATRIX_COMPLETED=false

matrix_status() {
    "$PYTHON" -m src.matrix_status --root "$OUTPUT_ROOT" --status "$1" \
        --configuration "$CURRENT_CONFIG" --exit-code "${2:-0}"
}
matrix_exit() {
    local code=$?
    trap - EXIT
    if [[ "$MATRIX_COMPLETED" != true ]]; then
        [[ "$code" -ne 0 ]] || code=1
        set +e
        matrix_status failed "$code"
        echo "Matrix stopped. Inspect $OUTPUT_ROOT/matrix.log; preserve checkpoints." >&2
    fi
    exit "$code"
}

if [[ "$DRY_RUN" == false ]]; then
    [[ -x "$PYTHON" ]] || { echo "Missing application Python: $PYTHON" >&2; exit 1; }
    command -v flock >/dev/null || { echo 'Install util-linux for flock' >&2; exit 1; }
    mkdir -p "$OUTPUT_ROOT"
    exec 9>"$OUTPUT_ROOT/.matrix.lock"
    flock -n 9 || { echo 'Another matrix runner holds the output lock' >&2; exit 1; }
    exec > >(tee -a "$OUTPUT_ROOT/matrix.log") 2>&1
    trap matrix_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    matrix_status running
    "$PYTHON" - <<'PREFLIGHT'
import json
from pathlib import Path
from openai import OpenAI
from src.dspy_modules import FORMAL_MODEL
from src.dspy_verifier import validate_compiled_verifier_artifact
from src.llm_endpoint import resolve_llm_endpoint
from src.telemetry_errors import redact_text

rows = json.loads(Path('data/dev.json').read_text(encoding='utf-8'))
if len(rows) != 1534:
    raise SystemExit('Expected the complete 1,534-question BIRD dev split')
for db_id in sorted({row['db_id'] for row in rows}):
    path = Path('data/dev_databases') / db_id / f'{db_id}.sqlite'
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f'Missing or empty development database: {db_id}')
if not Path('compiled_dspy_model.json').is_file():
    raise SystemExit('Missing compiled_dspy_model.json')
validate_compiled_verifier_artifact()
endpoint = resolve_llm_endpoint(FORMAL_MODEL)
if endpoint.backend != 'vllm':
    raise SystemExit('Full runner requires VLLM_BASE_URL and VLLM_API_KEY')
try:
    with OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url, timeout=30, max_retries=0) as client:
        models = {row.id for row in client.models.list().data}
    if endpoint.model not in models:
        raise SystemExit('vLLM is not serving the pinned Qwen/Qwen2.5-7B-Instruct model name')
except Exception as error:
    raise SystemExit('vLLM preflight failed: ' + redact_text(error)) from None
print('Preflight passed: dev split, database files, verifier artifact and vLLM authentication.')
PREFLIGHT
fi

run_one() {
    local prompt_mode="$1" selection="$2" name="$3"
    local directory="$OUTPUT_ROOT/$prompt_mode/$name"
    local candidates=8 workers=8 turns=5 temperature=0.8 top_p=0.7
    if [[ "$name" == greedy ]]; then
        candidates=1 workers=1 turns=1 temperature=0 top_p=1
    fi
    local args=(run_ablation.py --dev-json data/dev.json --db-dir data/dev_databases
        --limit 1534 --selection "$selection" --prompt-mode "$prompt_mode"
        --candidates "$candidates" --workers "$workers" --max-turns "$turns"
        --temperature "$temperature" --top-p "$top_p" --seed 0
        --verification-samples 1 --output "$directory/predict.json")
    if [[ -f "$directory/predict.checkpoint.jsonl" ]]; then args+=(--resume); fi
    if [[ "$DRY_RUN" == true ]]; then
        printf '%q ' "$PYTHON" -u "${args[@]}"; printf '\n'; return
    fi
    mkdir -p "$directory"
    CURRENT_CONFIG="$prompt_mode/$name"
    matrix_status running
    echo "Starting $CURRENT_CONFIG at $(date -u +%FT%TZ) (tolerant-count-v2)"
    # Keep a fresh attempt log: an old accounting traceback cannot excuse a new failure.
    local attempt started_ns
    attempt=$(mktemp "$directory/attempt.XXXXXXXX.log")
    started_ns=$("$PYTHON" -c 'import time; print(time.time_ns())')
    local -a codes
    # A conditional pipeline disables errexit only here; capture BOTH statuses immediately.
    if "$PYTHON" -u "${args[@]}" 2>&1 | tee -a "$directory/run.log" "$attempt"; then
        codes=("${PIPESTATUS[@]}")
    else
        codes=("${PIPESTATUS[@]}")
    fi
    if [[ "${codes[1]}" -ne 0 ]]; then
        echo 'Logging failed; refusing to accept this configuration.' >&2
        return "${codes[1]}"
    fi
    "$PYTHON" - "$directory/predict.metrics.json" "${codes[0]}" "$attempt" \
        "$started_ns" "$prompt_mode" "$selection" "$candidates" "$turns" \
        "$temperature" "$top_p" "$0" <<'VALIDATE'
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

path = Path(sys.argv[1])
exit_code = int(sys.argv[2])
attempt = Path(sys.argv[3])
started_ns = int(sys.argv[4])
prompt_mode, selection = sys.argv[5:7]
candidates, turns = map(int, sys.argv[7:9])
temperature, top_p = map(float, sys.argv[9:11])
runner = Path(sys.argv[11])
FIELDS = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost_usd')

def require(condition, message):
    if not condition:
        raise SystemExit('TOLERANCE REJECTED: ' + message)

def number(value, field):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Missing, negative or non-finite measurement: ' + field)
    if field != 'cost_usd':
        require(type(value) is int, 'Token counts must be exact integers')

def equal(actual, expected, field):
    if expected is None:
        require(actual is None, 'Unknown measurement was replaced: ' + field)
    else:
        number(actual, field)
        require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
                if field == 'cost_usd' else actual == expected,
                'Arithmetic mismatch: ' + field)

def complete(row):
    for field in FIELDS:
        number(row.get(field), field)
    require(row['total_tokens'] == row['prompt_tokens'] + row['completion_tokens'],
            'Input/output/total arithmetic mismatch')

require(exit_code in (0, 1), 'Unexpected Python exit status: ' + str(exit_code))
require(path.is_file(), 'Metrics report missing')
require(path.stat().st_mtime_ns >= started_ns - 2_000_000_000,
        'Stale metrics report; this attempt did not publish a fresh report')
report = json.loads(path.read_text(encoding='utf-8'))
require(report.get('schema_version') == 4, 'Unexpected report schema')
queries = report['queries']
require(report.get('total') == len(queries) == 1534, 'Configuration is incomplete')
dataset = json.loads(Path('data/dev.json').read_text(encoding='utf-8'))
ids = [str(row['question_id']) for row in dataset[:1534]]
require(len(set(ids)) == len(ids) == 1534, 'Dataset IDs invalid')
require([q['question_id'] for q in queries] == ids, 'Question IDs do not match dataset order')
predictions = json.loads(path.with_name('predict.json').read_text(encoding='utf-8'))
require(set(predictions) == set(ids), 'Predictions do not match dataset IDs')
require(any(q.get('final_sql') for q in queries), 'No selected SQL in configuration')
incomplete = []
for q in queries:
    config = q['configuration']
    expected = dict(prompt_mode=prompt_mode, selection_mode=selection, num_candidates=candidates,
                    max_workers=candidates, max_turns=turns, temperature=temperature,
                    top_p=top_p, seed=0, verification_samples=1,
                    execution_timeout_seconds=30.0, cross_model_fallback=False,
                    llm_backend='vllm')
    require(all(config.get(k) == v for k, v in expected.items()), 'Configuration mismatch')
    require(type(q.get('execution_correct')) is bool, 'EX must be a boolean')
    require(type(q.get('total_rollout_turns')) is int, 'Turn count missing')
    sql = q.get('final_sql')
    require(sql is None or isinstance(sql, str), 'Invalid SQL artifact')
    require(predictions[q['question_id']] == (
        f"{sql}\t----- bird -----\t{q['db_id']}" if sql else None), 'Prediction/metrics mismatch')
    require(q.get('accounting_complete') in (True, False)
            and type(q['accounting_complete']) is bool, 'Accounting status missing')
    if not q['accounting_complete']:
        # Count once per question, irrespective of exception type or stage count.
        # Keep all partial/unknown measurements in the original report unchanged.
        incomplete.append(q['question_id'])
        continue
    complete(q)
    components = [c['metrics'] for c in q['candidates']]
    require(len(components) == candidates, 'Candidate pool missing')
    judge = (q.get('selection') or {}).get('metrics')
    if judge:
        components.append(judge)
    for component in components:
        stages = component.get('stages')
        require(isinstance(stages, list) and stages, 'Missing stage telemetry')
        for stage in stages:
            # A parsing/model error does not invalidate exact recorded usage.
            # No exception strings are inspected by this acceptance policy.
            complete(stage)
            calls = stage.get('calls')
            require(isinstance(calls, list) and calls, 'Provider call evidence missing')
            for call in calls:
                complete(call)
            for field in FIELDS:
                equal(stage[field], sum(c[field] for c in calls), field)
        for field in FIELDS:
            values = [s[field] for s in stages]
            equal(component.get(field), sum(values) if all(v is not None for v in values) else None, field)
        require(component.get('accounting_complete') is all(component.get(f) is not None for f in FIELDS),
                'Component accounting flag inconsistent')
    for field in FIELDS:
        values = [m[field] for m in components]
        equal(q.get(field), sum(values) if all(v is not None for v in values) else None, field)
        observed = [s[field] for m in components for s in m['stages'] if s[field] is not None]
        equal(q.get('observed_' + field), sum(observed) if observed else None, field)
require(len(incomplete) <= 10, 'More than 10 queries have incomplete accounting')
require(report['correct'] == sum(q['execution_correct'] for q in queries), 'EX count mismatch')
require(math.isclose(report['execution_accuracy'], 100 * report['correct'] / 1534), 'EX percentage mismatch')
totals = report['totals']
require(totals.get('accounting_complete') is (not incomplete), 'Aggregate accounting flag inconsistent')
for field in FIELDS:
    values = [q.get(field) for q in queries]
    equal(totals.get(field), sum(values) if all(v is not None for v in values) else None, field)
    observed = [q['observed_' + field] for q in queries if q.get('observed_' + field) is not None]
    equal(totals.get('observed_' + field), sum(observed) if observed else None, field)

# No exception-string whitelist: the fresh completed report, incomplete-query
# count and complete-query arithmetic determine telemetry acceptance. Signals,
# logging failures and stale/incomplete reports still fail independently.

# Preserve the policy decision separately; never alter report/checkpoint measurements.
decision = dict(policy='tolerant-count-v2', accepted=True, max_incomplete_queries=10,
                incomplete_question_ids=incomplete, python_exit_code=exit_code,
                accounting_complete=not incomplete, verified_utc=datetime.now(timezone.utc).isoformat(),
                metrics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
                attempt_log=attempt.name, observed_total_tokens=totals.get('observed_total_tokens'),
                note='Threshold acceptance only; missing usage stays unknown; EX and recorded errors unchanged.')
fd, temporary = tempfile.mkstemp(prefix='.acceptance-', dir=path.parent)
try:
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump(decision, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path.with_name('tolerance.acceptance.json'))
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(f"ACCEPTED tolerant-count-v2: {path}; incomplete={len(incomplete)}/1534 "
      f"(limit=10); Python exit={exit_code}; IDs={incomplete}. "
      'Original accounting flags, unknown totals and EX preserved.', flush=True)
VALIDATE
}

for prompt_mode in uncompiled_dspy compiled_dspy; do
    for selection in self_consistency dspy_verifier self_consistency_verifier; do
        run_one "$prompt_mode" "$selection" "$selection"
    done
    run_one "$prompt_mode" self_consistency greedy
done
if [[ "$DRY_RUN" == false ]]; then
    CURRENT_CONFIG=all_eight_configurations
    matrix_status completed
    MATRIX_COMPLETED=true
    echo "All eight configurations accepted under tolerant-count-v2 at $(date -u +%FT%TZ). Results: $OUTPUT_ROOT"
fi

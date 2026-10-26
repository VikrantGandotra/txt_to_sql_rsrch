#!/usr/bin/env bash
# Run inside tmux on the Pod. Re-run with the SAME output root to resume.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi
OUTPUT_ROOT="${1:-results/full_dev_vllm_01}"
PYTHON="${MARS_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export DSPY_CACHEDIR="$SCRIPT_DIR/.dspy_cache"
export LITELLM_LOCAL_MODEL_COST_MAP=True
CURRENT_CONFIG="preflight"
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
        echo "Matrix stopped. Inspect $OUTPUT_ROOT/matrix.log; rerun with the same output root to resume." >&2
    fi
    exit "$code"
}

if [[ "$DRY_RUN" == false ]]; then
    [[ -x "$PYTHON" ]] || { echo "Missing application Python: $PYTHON" >&2; exit 1; }
    command -v flock >/dev/null || { echo "Install util-linux for flock" >&2; exit 1; }
    mkdir -p "$OUTPUT_ROOT"
    exec 9>"$OUTPUT_ROOT/.matrix.lock"
    flock -n 9 || { echo "Another matrix runner is using $OUTPUT_ROOT" >&2; exit 1; }
    exec > >(tee -a "$OUTPUT_ROOT/matrix.log") 2>&1
    trap matrix_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    matrix_status running
    "$PYTHON" - <<'PY'
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
    raise SystemExit('Set VLLM_BASE_URL and VLLM_API_KEY; full runner requires vLLM')
try:
    with OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url, timeout=30, max_retries=0) as client:
        models = {row.id for row in client.models.list().data}
    if endpoint.model not in models:
        raise SystemExit('vLLM is not serving the pinned Qwen/Qwen2.5-7B-Instruct model name')
except Exception as error:
    raise SystemExit('vLLM preflight failed: ' + redact_text(error)) from None
print('Preflight passed: dev split, database files, verifier artifact and vLLM /models authentication.')
PY
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
    if [[ -f "$directory/predict.checkpoint.jsonl" ]]; then
        args+=(--resume)
    fi
    if [[ "$DRY_RUN" == true ]]; then
        printf '%q ' "$PYTHON" -u "${args[@]}"
        printf '\n'
        return
    fi
    mkdir -p "$directory"
    CURRENT_CONFIG="$prompt_mode/$name"
    matrix_status running
    echo "Starting $prompt_mode / $name at $(date -u +%FT%TZ)"
    "$PYTHON" -u "${args[@]}" 2>&1 | tee -a "$directory/run.log"
    "$PYTHON" - "$directory/predict.metrics.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
assert report['schema_version'] == 4
assert report['total'] == len(report['queries']) == 1534
assert report['totals']['accounting_complete'], 'Incomplete provider accounting'
for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
    assert report['totals'][field] == sum(q[field] for q in report['queries'])
assert all(q['accounting_complete'] and q['total_tokens'] == q['prompt_tokens'] + q['completion_tokens']
           for q in report['queries'])
print('Validated completed configuration:', sys.argv[1])
PY
}

for prompt_mode in uncompiled_dspy compiled_dspy; do
    for selection in self_consistency dspy_verifier self_consistency_verifier; do
        run_one "$prompt_mode" "$selection" "$selection"
    done
    run_one "$prompt_mode" self_consistency greedy
done

if [[ "$DRY_RUN" == false ]]; then
    CURRENT_CONFIG="all_eight_configurations"
    matrix_status completed
    MATRIX_COMPLETED=true
    echo "All eight full-dev configurations completed and validated at $(date -u +%FT%TZ). Results: $OUTPUT_ROOT"
fi

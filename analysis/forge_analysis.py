"""Paired descriptive analyses; never infer attention or causation from token counts.

Run from the project root with ``python -m analysis.forge_analysis --help``.
No experiment modules, credentials, SDKs or model endpoints are imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

# Keep Matplotlib's cache out of a potentially read-only user profile.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "forge_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, spearmanr

MODES = ("uncompiled_dspy", "compiled_dspy")
LABELS = dict(zip(MODES, ("Uncompiled", "Compiled")))
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "MARS_Final_20261029"
CATEGORIES = ("no_such_column", "no_such_table", "ambiguous_column", "syntax_error",
              "other_sqlite_operational_error", "other_execution_error")
TOKEN_BINS = [0, 4000, 8000, 12000, np.inf]
BIN_LABELS = ["0–<4k", "4k–<8k", "8k–<12k", "12k+"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def finite_count(value):
    """Preserve unknown measurements as NaN, never substitute observed partial sums."""
    if type(value) is int and value >= 0:
        return value
    if value is None:
        return np.nan
    raise ValueError("Token/turn measurement must be a nonnegative integer or null")


def sanitize(text):
    """Defense in depth when exporting diagnostics for sharing with another AI."""
    text = str(text or "")
    text = re.sub(r"\b(?:sk|pk)-[A-Za-z0-9_-]{8,}", "[REDACTED_KEY]", text)
    text = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", text)
    return text


def error_text(details):
    if isinstance(details, dict):
        return " ".join(str(details.get(k) or "") for k in ("type", "message"))
    return str(details or "")


def category(message, details=None):
    text = (str(message or "") + " " + error_text(details)).lower()
    if not text.strip():
        return None
    for pattern, name in (("no such column", "no_such_column"),
                          ("no such table", "no_such_table"),
                          ("ambiguous column", "ambiguous_column"),
                          ("syntax error", "syntax_error")):
        if pattern in text:
            return name
    if "operationalerror" in text or "operational error" in text:
        return "other_sqlite_operational_error"
    return "other_execution_error"


def resolve_results(root):
    # Resolve ONE snapshot only. Never recursively concatenate both ping-pong slots.
    candidates = (root, root / "results/full_dev_vllm_01",
                  root / "experiment/results/full_dev_vllm_01",
                  root / "extracted_slot_1/experiment/results/full_dev_vllm_01")
    matches = [p for p in candidates if (p / MODES[0] / "self_consistency_verifier").is_dir()]
    require(len(matches) == 1, "Pass the recovery root, extracted experiment, or exact full_dev_vllm_01 directory")
    return matches[0]


def checkpoint_records(path):
    """Stream and validate complete JSONL records, including IDs and SHA-256."""
    with path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        require(header.get("version") == 1, "Unsupported checkpoint format")
        manifest = header["manifest"]
        ids = manifest["question_ids"]
        count = 0
        for line in handle:
            require(line.endswith("\n"), "Incomplete checkpoint tail: use a completed snapshot")
            event = json.loads(line)
            record = event["record"]
            encoded = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
            require(hashlib.sha256(encoded).hexdigest() == event["sha256"], "Checkpoint checksum mismatch")
            require(count < len(ids) and record["question_id"] == ids[count], "Checkpoint ID/order mismatch")
            require(record["metrics"]["question_id"] == record["result"]["question_id"] == record["question_id"],
                    "Metrics/result/question ID mismatch")
            require(record["metrics"]["execution_correct"] == record["result"]["correct"], "Checkpoint EX mismatch")
            count += 1
            yield record["metrics"]
        require(count == len(ids), "Checkpoint is not a completed configuration")


def observation(value):
    if isinstance(value, str):
        value = json.loads(value)
    require(isinstance(value, dict), "Malformed SQLite observation")
    return value


def request_counts(stage):
    calls = stage.get("calls") or []
    if calls:
        return [finite_count(c.get("prompt_tokens")) for c in calls]
    # A stage sum is a single-request length only if exactly one call was recorded.
    if stage.get("lm_call_count") == 1:
        return [finite_count(stage.get("prompt_tokens"))]
    return [np.nan] if stage.get("stage") == "generator" else []


def compact_query(q, mode):
    qid = str(q["question_id"])
    correct = q.get("execution_correct")
    require(type(correct) is bool, f"{mode}/{qid}: missing boolean execution_correct")
    config = q["configuration"]
    require(config["prompt_mode"] == mode and config["selection_mode"] == "self_consistency_verifier",
            "Input configuration mismatch")
    events = []
    diagnostic_categories = set()
    for c in q.get("candidates", []):
        index = c["candidate_index"]
        stages = c.get("metrics", {}).get("stages", [])
        generator = [s for s in stages if s.get("stage") == "generator"]
        stage_by_turn = {s["turn"]: s for s in generator}
        for turn, h in enumerate(c.get("history", []), 1):
            obs = observation(h["observation"])
            message = obs.get("error_message")
            detail = h.get("error_details")
            kind = category(message, detail) if message or detail else None
            if kind:
                diagnostic_categories.add(kind)
            counts = request_counts(stage_by_turn.get(turn, {}))
            known = [x for x in counts if pd.notna(x)]
            events.append(dict(mode=mode, question_id=qid, db_id=q["db_id"], candidate_index=index,
                               selected_candidate=index == q.get("selected_candidate_index"),
                               turn=turn, action=h.get("action"), sqlite_failed=bool(kind),
                               error_category=kind or "none", error_message=sanitize(message),
                               error_details=sanitize(json.dumps(detail, ensure_ascii=False)) if detail else "",
                               sql=sanitize(h.get("sql", "")) if kind else "",
                               max_request_prompt_tokens=max(known) if known and len(known) == len(counts) else np.nan,
                               no_such_column=int(kind == "no_such_column"),
                               schema_identifier_error=int(kind in {"no_such_column", "no_such_table", "ambiguous_column"})))
    # Preserve final-scoring diagnostics without counting them as extra generation attempts.
    final_kind = category(q.get("execution_error"), q.get("execution_error_details"))
    if final_kind:
        diagnostic_categories.add(final_kind)
    stages = [s for c in q.get("candidates", []) for s in c.get("metrics", {}).get("stages", [])
              if s.get("stage") == "generator"]
    lengths = [x for s in stages for x in request_counts(s)]
    known = [x for x in lengths if pd.notna(x)]
    first = [x for s in stages if s.get("turn") == 1 for x in request_counts(s)]
    first_known = [x for x in first if pd.notna(x)]
    row = dict(mode=mode, question_id=qid, db_id=q["db_id"], correct=int(correct),
               query_prompt_tokens=finite_count(q.get("prompt_tokens")),
               observed_prompt_tokens=finite_count(q.get("observed_prompt_tokens")),
               mean_generator_prompt_tokens=float(np.mean(known)) if known else np.nan,
               max_generator_prompt_tokens=max(known) if known else np.nan,
               first_generator_prompt_tokens=float(np.mean(first_known)) if first_known else np.nan,
               generator_requests_measured=len(known), generator_requests_missing=len(lengths)-len(known),
               total_rollout_turns=finite_count(q.get("total_rollout_turns")),
               selected_turn=finite_count(q.get("selected_trajectory_turns")),
               accounting_complete=q.get("accounting_complete") is True,
               sql_attempts=len(events), sqlite_error_attempts=sum(e["sqlite_failed"] for e in events),
               no_such_column_attempts=sum(e["no_such_column"] for e in events),
               any_no_such_column=int("no_such_column" in diagnostic_categories),
               status=q.get("status"), degradation_reasons=json.dumps(q.get("degradation_reasons", [])),
               execution_error=sanitize(q.get("execution_error")),
               error_details=sanitize(json.dumps(q.get("error_details"), ensure_ascii=False)),
               execution_error_details=sanitize(json.dumps(q.get("execution_error_details"), ensure_ascii=False)),
               candidate_failures=sanitize(json.dumps([
                   dict(candidate_index=c["candidate_index"], error=c.get("error"),
                        error_details=c.get("error_details"), termination_reason=c.get("termination_reason"))
                   for c in q.get("candidates", []) if c.get("error") or c.get("error_details")
               ], ensure_ascii=False)),
               final_sql=sanitize(q.get("final_sql")))
    for kind in CATEGORIES:
        row["any_" + kind] = int(kind in diagnostic_categories)
    return row, events


def load_pair(root, source="auto", verify_checkpoints=False):
    rows, events, provenance = [], [], {}
    controls = None
    id_sets = []
    for mode in MODES:
        directory = root / mode / "self_consistency_verifier"
        metrics = directory / "predict.metrics.json"
        journal = directory / "predict.checkpoint.jsonl"
        use_json = source == "json" or (source == "auto" and metrics.is_file())
        path = metrics if use_json else journal
        require(path.is_file(), f"Missing input: {path}")
        if use_json:
            report = json.loads(path.read_text(encoding="utf-8"))
            require(report.get("schema_version") == 4, "Expected metrics schema 4")
            manifest = report["manifest"]
            records = iter(report["queries"])
            require(report["total"] == len(report["queries"]), "Report count mismatch")
        else:
            with path.open(encoding="utf-8") as handle:
                manifest = json.loads(next(handle))["manifest"]
            records = checkpoint_records(path)
        keys = ("dataset_sha256", "model_name", "served_model", "llm_backend", "num_candidates",
                "max_turns", "temperature", "top_p", "seed", "verification_samples", "selection_mode",
                "compiled_verifier_sha256", "source_sha256")
        current = {k: manifest.get(k) for k in keys}
        if controls is None:
            controls = current
        else:
            require(current == controls, "Paired runs differ in dataset/model/sampling controls")
        ids = []
        for q in records:
            row, attempts = compact_query(q, mode)
            rows.append(row)
            events.extend(attempts)
            ids.append(row["question_id"])
        require(len(ids) == len(set(ids)), "Duplicate question IDs")
        require(ids == manifest["question_ids"], "Question order does not match manifest")
        if use_json:
            require(report["correct"] == sum(r["correct"] for r in rows if r["mode"] == mode), "EX summary mismatch")
            if verify_checkpoints:
                for n, saved in enumerate(checkpoint_records(journal)):
                    require(saved == report["queries"][n], "JSON and checkpoint metrics differ")
            del report
        id_sets.append(set(ids))
        provenance[mode] = dict(path=str(path), sha256=sha256(path), queries=len(ids), controls=current,
                                compiled_generator_sha256=manifest.get("compiled_generator_sha256"),
                                compiled_verifier_sha256=manifest.get("compiled_verifier_sha256"))
        gc.collect()
    require(id_sets[0] == id_sets[1], "Paired question IDs differ; refusing silent inner-join exclusion")
    frame = pd.DataFrame(rows)
    attempts = pd.DataFrame(events)
    return frame, attempts, provenance


def wilson(successes, n):
    if not n:
        return np.nan, np.nan
    p, z = successes / n, 1.959963984540054
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return center-half, center+half


def save_plot(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=220)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def rate_table(frame, exposure, outcome):
    data = frame.dropna(subset=[exposure]).copy()
    data["token_bin"] = pd.cut(data[exposure], TOKEN_BINS, labels=BIN_LABELS, right=False)
    rows = []
    for mode in MODES:
        for label in BIN_LABELS:
            subset = data[(data["mode"] == mode) & (data["token_bin"] == label)]
            n, successes = len(subset), int(subset[outcome].sum())
            low, high = wilson(successes, n)
            rows.append(dict(mode=mode, token_bin=label, n=n, successes=successes,
                             rate=successes/n if n else np.nan, ci_low=low, ci_high=high))
    return pd.DataFrame(rows)


def grouped_plot(table, output, name, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, mode in enumerate(MODES):
        values = table[table["mode"] == mode]
        x = np.arange(4) + (i - .5) * .36
        for pos, row in zip(x, values.to_dict("records")):
            if row["n"]:
                ax.bar(pos, row["rate"]*100, width=.34, color=sns.color_palette("colorblind")[i],
                       label=LABELS[mode] if row["token_bin"] == values[values.n > 0].iloc[0].token_bin else None)
                ax.errorbar(pos, row["rate"]*100,
                            yerr=[[max(0, row["rate"]-row["ci_low"])*100],
                                  [max(0, row["ci_high"]-row["rate"])*100]], color="black", capsize=3)
            ax.text(pos, (row["ci_high"]*100+2) if row["n"] else 2, f"n={row['n']}",
                    ha="center", fontsize=8)
    ax.set(xticks=np.arange(4), xticklabels=BIN_LABELS, ylim=(0, 112), xlim=(-.6, 3.6),
           xlabel="Prompt-token bin (4k = 4,000 tokens)", ylabel=ylabel, title=title)
    ax.legend()
    save_plot(fig, output, name)


def token_accuracy(frame, output):
    correlations = []
    for metric in ("query_prompt_tokens", "mean_generator_prompt_tokens", "first_generator_prompt_tokens"):
        table = rate_table(frame, metric, "correct")
        table.to_csv(output / f"accuracy_bins_{metric}.csv", index=False)
        grouped_plot(table, output, f"accuracy_bins_{metric}",
                     "Execution accuracy vs " + metric.replace("_", " "), "EX (%)")
        for mode in MODES:
            subset = frame[frame["mode"] == mode].dropna(subset=[metric])
            valid = len(subset) > 2 and subset[metric].nunique() > 1 and subset.correct.nunique() > 1
            result = spearmanr(subset[metric], subset.correct) if valid else None
            correlations.append(dict(mode=mode, exposure=metric, n=len(subset),
                spearman_rho=float(result.statistic) if result else None,
                pvalue=float(result.pvalue) if result else None))
    pd.DataFrame(correlations).to_csv(output / "accuracy_token_correlations.csv", index=False)


def error_shift(frame, attempts, output):
    left = frame[frame["mode"] == MODES[0]].drop(columns="mode")
    right = frame[frame["mode"] == MODES[1]].drop(columns="mode")
    paired = left.merge(right, on=["question_id", "db_id"], suffixes=("_uncompiled", "_compiled"), validate="one_to_one")
    require(len(paired) == len(left) == len(right), "DB/question pairing mismatch")
    paired["transition"] = np.select([
        (paired.correct_uncompiled == 1) & (paired.correct_compiled == 0),
        (paired.correct_uncompiled == 0) & (paired.correct_compiled == 1)],
        ["degraded", "improved"], default="unchanged")
    paired.to_csv(output / "paired_queries.csv", index=False)
    degraded = paired[paired.transition == "degraded"]
    degraded.to_csv(output / "degraded_queries.csv", index=False)
    paired[paired.transition == "improved"].to_csv(output / "improved_queries.csv", index=False)
    ids = set(degraded.question_id)
    failure_events = attempts[(attempts.question_id.isin(ids)) & attempts.sqlite_failed]
    failure_events.to_csv(output / "degraded_sqlite_errors.csv", index=False)
    frequencies = []
    for cohort, cohort_ids in (("degraded", ids), ("all", set(left.question_id))):
        for mode in MODES:
            subset = frame[(frame["mode"] == mode) & frame.question_id.isin(cohort_ids)]
            for kind in CATEGORIES:
                n = int(subset["any_" + kind].sum())
                frequencies.append(dict(cohort=cohort, mode=mode, category=kind,
                                        affected_queries=n, denominator=len(subset),
                                        percent=100*n/len(subset) if len(subset) else np.nan))
    pd.DataFrame(frequencies).to_csv(output / "error_frequency.csv", index=False)
    transitions = paired.groupby(["correct_uncompiled", "correct_compiled"]).size().reset_index(name="n")
    transitions.to_csv(output / "paired_accuracy_transitions.csv", index=False)
    b, c = len(degraded), int((paired.transition == "improved").sum())
    stats = dict(degraded=b, improved=c, discordant=b+c,
                 paired_accuracy_difference_percentage_points=100*(c-b)/len(paired),
                 mcnemar_exact_pvalue=binomtest(b, b+c, .5).pvalue if b+c else 1.0)
    (output / "paired_accuracy_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return paired


def dilution_proxy(frame, attempts, output):
    correlations = []
    # Query-level primary proxy uses first-turn input length, measured before
    # later SQLite errors can inflate the retry/observation history.
    for exposure in ("query_prompt_tokens", "mean_generator_prompt_tokens", "first_generator_prompt_tokens"):
        table = rate_table(frame, exposure, "any_no_such_column")
        table.to_csv(output / f"column_error_bins_{exposure}.csv", index=False)
        grouped_plot(table, output, f"column_error_bins_{exposure}",
                     "Queries with a recorded 'no such column' error", "Affected queries (%)")
        for mode in MODES:
            subset = frame[(frame["mode"] == mode)].dropna(subset=[exposure])
            valid = len(subset) > 2 and subset[exposure].nunique() > 1 and subset.any_no_such_column.nunique() > 1
            result = spearmanr(subset[exposure], subset.any_no_such_column) if valid else None
            correlations.append(dict(mode=mode, exposure=exposure, n=len(subset),
                                     spearman_rho=float(result.statistic) if result else None,
                                     pvalue=float(result.pvalue) if result else None))
    pd.DataFrame(correlations).to_csv(output / "column_error_correlations.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    data = frame.dropna(subset=["first_generator_prompt_tokens"])
    for i, mode in enumerate(MODES):
        subset = data[data["mode"] == mode].copy()
        rng = np.random.default_rng(7)
        ax.scatter(subset.first_generator_prompt_tokens,
                   subset.any_no_such_column + rng.uniform(-.035, .035, len(subset)),
                   s=9, alpha=.18, color=sns.color_palette("colorblind")[i], label=LABELS[mode])
    ax.set(xlabel="Mean first-generator-request input tokens per query",
           ylabel="Any recorded 'no such column' error (jittered)",
           title="Prompt-length proxy and column errors (association only)")
    ax.legend()
    save_plot(fig, output, "first_prompt_vs_column_errors_scatter")
    # Turn-level rates use actual SQL-attempt denominators, not errors per
    # arbitrary number of generated trajectories. Query-level statistics above
    # avoid treating dependent turns as independent inferential observations.
    table = rate_table(attempts, "max_request_prompt_tokens", "no_such_column")
    table.to_csv(output / "attempt_column_error_rates.csv", index=False)
    first = attempts[attempts.turn == 1]
    rate_table(first, "max_request_prompt_tokens", "no_such_column").to_csv(
        output / "first_turn_column_error_rates.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs/forge_context"))
    parser.add_argument("--source", choices=("auto", "json", "jsonl"), default="auto")
    parser.add_argument("--verify-checkpoints", action="store_true")
    parser.add_argument("--analysis", choices=("all", "tokens", "errors", "proxy"), default="all")
    args = parser.parse_args(argv)
    root = resolve_results(args.root.resolve())
    output = args.output.resolve()
    require(output != args.root.resolve() and output != root
            and root not in output.parents and args.root.resolve() not in output.parents,
            "Write analysis outputs outside the raw recovery snapshot")
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    frame, attempts, provenance = load_pair(root, args.source, args.verify_checkpoints)
    frame.to_csv(output / "query_measurements.csv", index=False)
    attempts.to_csv(output / "sqlite_attempts.csv", index=False)
    coverage = frame.groupby("mode").agg(queries=("question_id", "size"), correct=("correct", "sum"),
        complete_accounting=("accounting_complete", "sum"), measured_generator_requests=("generator_requests_measured", "sum"),
        missing_generator_requests=("generator_requests_missing", "sum"))
    coverage["ex_percent"] = 100 * coverage.correct / coverage.queries
    coverage.to_csv(output / "coverage.csv")
    (output / "provenance.json").write_text(json.dumps(dict(inputs=provenance, source=args.source,
        checkpoint_crosscheck=args.verify_checkpoints, analysis_sha256=sha256(Path(__file__)),
        versions={m: sys.modules[m].__version__ for m in ("pandas", "numpy", "matplotlib", "seaborn", "scipy")}), indent=2), encoding="utf-8")
    if args.analysis in ("all", "tokens"):
        token_accuracy(frame, output)
    if args.analysis in ("all", "errors"):
        error_shift(frame, attempts, output)
    if args.analysis in ("all", "proxy"):
        dilution_proxy(frame, attempts, output)
    print(coverage.to_string())
    print(f"Artifacts: {output}")
    print("Association only: these logs do not establish attention loss, DDL position, or a causal effect of prompt length.")


if __name__ == "__main__":
    main()

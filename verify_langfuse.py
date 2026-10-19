"""Read existing Langfuse traces and compare their billing with a local report.

This makes authenticated GET requests only; it never runs an LLM or uploads
traces. Inspect generation observations: their parent span is not billable.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import httpx


FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd")


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _totals(rows):
    return {field: sum(row[field] for row in rows)
            if rows and all(_number(row.get(field)) for row in rows) else None
            for field in FIELDS}


def report_traces(report):
    """Find rollout and judge traces, avoiding copies in query-level summaries."""
    found = []
    seen = set()

    def add(trace_id, metrics, label):
        if isinstance(trace_id, str) and trace_id and trace_id not in seen:
            seen.add(trace_id)
            found.append({"trace_id": trace_id, "label": label,
                          "expected": {field: metrics.get(field) for field in FIELDS},
                          "stages": metrics.get("stages", [])})

    for query in report.get("queries", []):
        question = query.get("question_id")
        for candidate in query.get("candidates") or [query]:
            metrics = candidate.get("metrics", candidate)
            add(metrics.get("trace_id"), metrics,
                f"question {question}, candidate {candidate.get('candidate_index', 0)}")
        # A selector generation has its own trace. New reports retain its IDs.
        judge_traces = defaultdict(list)
        for stage in ((query.get("selection") or {}).get("metrics") or {}).get("stages", []):
            if stage.get("trace_id"):
                judge_traces[stage["trace_id"]].append(stage)
        for trace_id, stages in judge_traces.items():
            add(trace_id, {**_totals(stages), "stages": stages}, f"question {question}, selector")
    return found


def compare_trace(check, trace, base_url):
    """Compare ingested generation fields, never metadata or inferred prices."""
    generations = []
    for observation in trace.get("observations", []):
        if observation.get("type") != "GENERATION":
            continue
        usage = observation.get("usageDetails")
        if usage is None:  # Older Langfuse API responses expose usage instead.
            usage = observation.get("usage") or {}
        costs = observation.get("costDetails") or {}
        generations.append({
            "observation_id": observation.get("id"), "name": observation.get("name"),
            "prompt_tokens": usage.get("input"), "completion_tokens": usage.get("output"),
            "total_tokens": usage.get("total"), "cost_usd": costs.get("total"),
        })
    actual = _totals(generations)
    issues = []

    def compare(expected, measured, label):
        for field in FIELDS:
            if not _number(expected.get(field)):
                issues.append(f"{label}: local {field} is unknown; cannot verify")
            elif not _number(measured.get(field)):
                issues.append(f"{label}: ingested {field} is missing")
            elif not math.isclose(expected[field], measured[field], rel_tol=1e-9, abs_tol=1e-12):
                issues.append(f"{label}: {field} local={expected[field]} server={measured[field]}")

    compare(check["expected"], actual, "trace generations")
    by_id = {row["observation_id"]: row for row in generations}
    verified_stages = 0
    for stage in check.get("stages", []):
        observation_id = stage.get("observation_id")
        if not observation_id:  # Historical reports can still be checked by totals.
            continue
        if observation_id not in by_id:
            issues.append(f"generation {observation_id} is missing from the server trace")
        else:
            compare(stage, by_id[observation_id], f"generation {observation_id}")
            verified_stages += 1
    if not generations:
        issues.append("No GENERATION observations found; root SPAN metadata does not supply billing")
    project_id = trace.get("projectId")
    trace_url = (f"{base_url.rstrip('/')}/project/{project_id}/traces/{check['trace_id']}"
                 if project_id else None)
    return {"trace_id": check["trace_id"], "label": check["label"],
            "status": "MISMATCH" if issues else "MATCH", "trace_url": trace_url,
            "local": check["expected"], "ingested": actual, "generations": generations,
            "verified_stage_count": verified_stages, "issues": issues}


def verify_report(report, client, base_url, limit=1):
    checks = report_traces(report)
    if not checks:
        raise ValueError("Report has no recorded Langfuse trace IDs")
    results = []
    for check in checks[:limit]:
        try:
            response = client.get(f"/api/public/traces/{check['trace_id']}")
            response.raise_for_status()
            results.append(compare_trace(check, response.json(), base_url))
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            reason = ({401: "Check Langfuse project credentials", 403: "Project access denied",
                       404: "Trace absent: check project/host, export errors, and ingestion delay"}
                      .get(status, "Langfuse API request failed"))
            results.append({"trace_id": check["trace_id"], "status": "ERROR",
                            "issues": [f"HTTP {status}: {reason}"]})
        except (httpx.RequestError, ValueError) as error:
            # Do not print response bodies, prompts, credentials, or auth headers.
            results.append({"trace_id": check["trace_id"], "status": "ERROR",
                            "issues": [f"Could not read trace: {type(error).__name__}"]})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="Existing predict.metrics.json")
    parser.add_argument("--limit", type=int, default=1, help="Maximum existing traces to check")
    parser.add_argument("--output", help="Optional sanitized JSON verification report")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.output and Path(args.output).resolve() == Path(args.metrics).resolve():
        parser.error("--output must not overwrite the original metrics")
    report = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    from src.config import settings

    base_url = settings.langfuse_base_url.rstrip("/")
    with httpx.Client(base_url=base_url, timeout=30.0,
                      auth=(settings.langfuse_public_key, settings.langfuse_secret_key)) as client:
        results = verify_report(report, client, base_url, args.limit)
    for result in results:
        print(f"{result['status']}: {result['trace_id']} ({result.get('label', '')})")
        for row in result.get("generations", []):
            print(f"  {row['name']} [{row['observation_id']}]: prompt={row['prompt_tokens']} "
                  f"completion={row['completion_tokens']} total={row['total_tokens']} cost_usd={row['cost_usd']}")
        if "ingested" in result:
            print(f"  Ingested totals: {json.dumps(result['ingested'])}")
        for issue in result["issues"]:
            print(f"  {issue}")
        if result.get("trace_url"):
            print(f"  Dashboard: {result['trace_url']}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if all(result["status"] == "MATCH" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

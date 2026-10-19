"""Read-back diagnostics distinguish ingested usage from root-span metadata."""

import copy
import unittest

import httpx

from verify_langfuse import compare_trace, report_traces, verify_report


METRICS = {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25, "cost_usd": 0.001}


class VerifyLangfuseTests(unittest.TestCase):
    def setUp(self):
        self.check = {"trace_id": "trace-a", "label": "candidate 0", "expected": dict(METRICS),
                      "stages": [{**METRICS, "observation_id": "generation-a"}]}
        self.trace = {"id": "trace-a", "projectId": "project-a", "observations": [
            {"id": "parent", "type": "SPAN", "metadata": dict(METRICS),
             "usageDetails": {}, "costDetails": {}},
            {"id": "generation-a", "type": "GENERATION", "name": "mars_generator",
             "usageDetails": {"input": 20, "output": 5, "total": 25},
             "costDetails": {"total": 0.001}},
        ]}

    def test_matches_ingested_generations_and_ignores_parent_metadata(self):
        result = compare_trace(self.check, self.trace, "https://example.invalid/")
        self.assertEqual(result["status"], "MATCH")
        self.assertEqual(result["ingested"], METRICS)
        self.assertEqual(result["verified_stage_count"], 1)
        self.assertEqual(result["trace_url"], "https://example.invalid/project/project-a/traces/trace-a")

    def test_metadata_only_or_missing_cost_cannot_pass_as_delivered_billing(self):
        for field in ("usageDetails", "costDetails"):
            with self.subTest(field=field):
                trace = copy.deepcopy(self.trace)
                trace["observations"][1][field] = {}
                trace["observations"][1]["metadata"] = dict(METRICS)
                result = compare_trace(self.check, trace, "https://example.invalid")
                self.assertEqual(result["status"], "MISMATCH")
                self.assertTrue(any("missing" in issue for issue in result["issues"]))

    def test_detects_numeric_discrepancy_and_missing_specific_generation(self):
        self.trace["observations"][1]["usageDetails"]["input"] = 21
        self.trace["observations"][1]["id"] = "wrong-observation"
        result = compare_trace(self.check, self.trace, "https://example.invalid")
        self.assertEqual(result["status"], "MISMATCH")
        self.assertTrue(any("local=20 server=21" in issue for issue in result["issues"]))
        self.assertTrue(any("generation-a is missing" in issue for issue in result["issues"]))

    def test_old_reports_still_compare_totals_and_unknowns_are_not_zero(self):
        self.check["stages"][0].pop("observation_id")
        result = compare_trace(self.check, self.trace, "https://example.invalid")
        self.assertEqual(result["status"], "MATCH")
        self.assertEqual(result["verified_stage_count"], 0)
        self.check["expected"]["cost_usd"] = None
        self.assertEqual(compare_trace(self.check, self.trace, "https://example.invalid")["status"], "MISMATCH")

    def test_collects_candidate_and_selector_traces_without_duplicate_summaries(self):
        report = {"queries": [{"question_id": 999, "trace_id": "trace-a", **METRICS,
            "candidates": [{"candidate_index": 4, "metrics": {"trace_id": "trace-a", **METRICS}}],
            "selection": {"metrics": {"stages": [
                {"trace_id": "judge-a", **METRICS}, {"trace_id": "judge-a", **METRICS}]}}}]}
        checks = report_traces(report)
        self.assertEqual([row["trace_id"] for row in checks], ["trace-a", "judge-a"])
        self.assertEqual(checks[1]["expected"]["total_tokens"], 50)
        self.assertEqual(checks[1]["expected"]["cost_usd"], 0.002)

    def test_verification_uses_only_get_and_redacts_error_response_bodies(self):
        report = {"queries": [{"trace_id": "trace-a", **METRICS}]}
        methods = []
        def respond(request):
            methods.append(request.method)
            return httpx.Response(401, json={"error": "PRIVATE_RESPONSE_BODY"})
        with httpx.Client(base_url="https://example.invalid", transport=httpx.MockTransport(respond)) as client:
            results = verify_report(report, client, "https://example.invalid")
        self.assertEqual(methods, ["GET"])
        self.assertEqual(results[0]["status"], "ERROR")
        self.assertIn("HTTP 401", results[0]["issues"][0])
        self.assertNotIn("PRIVATE_RESPONSE_BODY", str(results))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Check scientific denominators and nested telemetry extraction with tiny fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from analysis.forge_analysis import category, compact_query, rate_table, resolve_results, error_shift


class ForgeAnalysisTests(unittest.TestCase):
    def query(self):
        stage = dict(stage="generator", turn=1, prompt_tokens=120,
                     lm_call_count=2, calls=[dict(prompt_tokens=40), dict(prompt_tokens=80)])
        return dict(question_id="7", db_id="db", execution_correct=True,
                    configuration=dict(prompt_mode="uncompiled_dspy", selection_mode="self_consistency_verifier"),
                    prompt_tokens=2000, observed_prompt_tokens=2000, total_rollout_turns=1,
                    selected_trajectory_turns=1, accounting_complete=True,
                    candidates=[dict(candidate_index=0, metrics=dict(stages=[stage]),
                                     history=[dict(action="probe", sql='SELECT "missing" FROM t',
                                       observation=json.dumps(dict(is_valid=False,
                                         error_message="OperationalError: no such column: missing")),
                                       error_details=dict(type="OperationalError", message="no such column: missing"))])])

    def test_nested_observation_counts_once_and_uses_provider_request_length(self):
        row, attempts = compact_query(self.query(), "uncompiled_dspy")
        self.assertEqual(row["correct"], 1)  # Repaired error does not overwrite EX.
        self.assertEqual(row["no_such_column_attempts"], 1)
        self.assertEqual(row["mean_generator_prompt_tokens"], 60)
        self.assertEqual(row["query_prompt_tokens"], 2000)
        self.assertEqual(attempts[0]["max_request_prompt_tokens"], 80)

    def test_operational_error_is_not_automatically_column_hallucination(self):
        self.assertEqual(category("OperationalError: database is locked"), "other_sqlite_operational_error")
        self.assertEqual(category('OperationalError: near "FROM": syntax error'), "syntax_error")

    def test_bin_edges_empty_bins_and_unknown_tokens(self):
        data = pd.DataFrame(dict(mode=["uncompiled_dspy"]*4,
                                 tokens=[0, 4000, 12000, float("nan")], correct=[1, 0, 1, 0]))
        table = rate_table(data, "tokens", "correct")
        self.assertEqual(table.n.sum(), 3)
        rows = table[table["mode"] == "uncompiled_dspy"].to_dict("records")
        self.assertEqual([r["n"] for r in rows], [1, 1, 0, 1])
        self.assertTrue(pd.isna(rows[2]["rate"]))

    def test_unknown_totals_never_replaced_by_partial_observed_usage(self):
        q = self.query()
        q["prompt_tokens"] = None
        row, _ = compact_query(q, "uncompiled_dspy")
        self.assertTrue(pd.isna(row["query_prompt_tokens"]))
        self.assertEqual(row["observed_prompt_tokens"], 2000)

    def test_snapshot_resolution_never_combines_both_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for slot in (0, 1):
                (root / f"extracted_slot_{slot}/experiment/results/full_dev_vllm_01/uncompiled_dspy/self_consistency_verifier").mkdir(parents=True)
            self.assertIn("extracted_slot_1", str(resolve_results(root)))

    def test_paired_degradation_uses_ids_not_row_positions(self):
        a, events = compact_query(self.query(), "uncompiled_dspy")
        q2 = self.query()
        q2["question_id"] = "8"
        q2["execution_correct"] = False
        b, _ = compact_query(q2, "uncompiled_dspy")
        compiled = [dict(b, mode="compiled_dspy", correct=1), dict(a, mode="compiled_dspy", correct=0)]
        with tempfile.TemporaryDirectory() as tmp:
            paired = error_shift(pd.DataFrame([a, b, *compiled]), pd.DataFrame(events), Path(tmp))
            self.assertEqual(paired.loc[paired.transition == "degraded", "question_id"].tolist(), ["7"])
            self.assertEqual(paired.loc[paired.transition == "improved", "question_id"].tolist(), ["8"])


if __name__ == "__main__":
    unittest.main()

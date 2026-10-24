"""Exercise the complete matrix runner with offline child-process results."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import smoke_test


class SmokeMatrixTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mars_smoke_")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.dev = self.root / "dev.json"
        self.db_dir = self.root / "databases"
        self.db = self.db_dir / "db/db.sqlite"
        self.db.parent.mkdir(parents=True)
        self.db.touch()
        self.items = [{"question_id": index, "db_id": "db"} for index in (1, 5, 9)]
        self.dev.write_text(json.dumps(self.items), encoding="utf-8")
        self.compiled = self.root / "compiled.json"
        self.compiled.write_text(json.dumps({"mars_generator_manifest": {
            "version": 2, "action_contract": "probe_final_answer", "max_iterations": 5},
            "decomposer": {"demos": [{}]}, "generator": {"demos": [{}]}}), encoding="utf-8")
        self.verifier = self.root / "judge.json"
        self.verifier.write_text(json.dumps({
            "mars_verifier_manifest": {"version": 1, "score_kind": "elicited_validity_rating",
                "model_name": "openrouter/qwen/qwen-2.5-7b-instruct"},
            "judge": {"signature": {"fields": [{}] * 6}, "demos": [{"score": 0.9}]}},
        ), encoding="utf-8")
        self.calls = []

    def child(self, command, **kwargs):
        self.calls.append(command)
        self.assertEqual(command[0], sys.executable)
        self.assertTrue(kwargs["check"])
        self.assertEqual(kwargs["cwd"], smoke_test.ROOT)
        options = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(options["--limit"], "3")
        output = Path(options["--output"])
        count = int(options["--candidates"])
        accounting = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                      "cost_usd": 0.001, "accounting_complete": True}
        queries, predictions = [], {}
        sql = "SELECT 1"
        for item in self.items:
            index = str(item["question_id"])
            predictions[index] = sql + smoke_test.SEPARATOR + "db"
            selection = {"selected_candidate_index": 0, "cascade_tie_breaker_engaged": False}
            queries.append({"question_id": index, "generation_failed": False, "selection_failed": False,
                            "fallback_used": False, "error": None, "execution_error": None,
                            "selection": selection, **accounting,
                            "candidates": [{"candidate_index": i, "final_sql": sql,
                                            "execution": {"result_hash": "hash", "error": None}}
                                           for i in range(count)]})
        manifest = {"question_ids": [str(item["question_id"]) for item in self.items],
                    "prompt_mode": options["--prompt-mode"], "selection_mode": options["--selection"],
                    "num_candidates": count, "dataset_sha256": hashlib.sha256(self.dev.read_bytes()).hexdigest(),
                    "compiled_generator_sha256": hashlib.sha256(self.compiled.read_bytes()).hexdigest()
                    if options["--prompt-mode"] == "compiled_dspy" else None}
        report = {"manifest": manifest, "queries": queries, "total": 3, "execution_accuracy": 0,
                  "totals": {key: value * 3 for key, value in accounting.items() if key != "accounting_complete"}}
        report["totals"]["accounting_complete"] = True
        output.write_text(json.dumps(predictions), encoding="utf-8")
        output.with_suffix(".metrics.json").write_text(json.dumps(report), encoding="utf-8")

    def run_matrix(self, runner=None):
        with redirect_stdout(io.StringIO()) as output:
            path = smoke_test.run_matrix(dev_json=self.dev, db_dir=self.db_dir,
                compiled_model=self.compiled, verifier_model=self.verifier, output_root=self.root / "runs", runner=runner or self.child)
        return path, output.getvalue()

    def test_six_sequential_unique_runs_and_semantic_inaccuracy_is_reported_not_crash(self):
        path, output = self.run_matrix()
        self.assertEqual(len(self.calls), 6)
        options = [dict(zip(command[3::2], command[4::2])) for command in self.calls]
        self.assertEqual([(row["--prompt-mode"], row["--selection"]) for row in options], [
            (mode, selection) for mode in ("uncompiled_dspy", "compiled_dspy")
            for selection in ("self_consistency", "dspy_verifier", "hybrid_cascade")])
        self.assertEqual(len({row["--output"] for row in options}), 6)
        for row in options:
            if row["--selection"] in {"dspy_verifier", "hybrid_cascade"}:
                self.assertEqual(row["--verifier-model"], str(self.verifier.resolve()))
        self.assertEqual(len(json.loads((path / "summary.json").read_text())), 6)
        self.assertIn("Matrix Smoke Test Fully Passed", output)
        second_path, _ = self.run_matrix()
        self.assertNotEqual(path, second_path)

    def test_nonzero_child_exit_halts_without_running_remaining_configs(self):
        def fail(command, **kwargs):
            self.calls.append(command)
            raise subprocess.CalledProcessError(1, command)
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_matrix(fail)
        self.assertEqual(len(self.calls), 1)

    def test_silent_failures_missing_predictions_and_incomplete_accounting_halt(self):
        for failure in ("missing_prediction", "generation_failed", "selection_failed", "accounting", "wrong_manifest"):
            with self.subTest(failure=failure):
                self.calls.clear()
                def invalid(command, **kwargs):
                    self.child(command, **kwargs)
                    options = dict(zip(command[3::2], command[4::2]))
                    path = Path(options["--output"])
                    if failure == "missing_prediction":
                        predictions = json.loads(path.read_text())
                        predictions["1"] = None
                        path.write_text(json.dumps(predictions))
                    else:
                        path = path.with_suffix(".metrics.json")
                        report = json.loads(path.read_text())
                        if failure in ("generation_failed", "selection_failed"):
                            report["queries"][0][failure] = True
                        elif failure == "accounting":
                            report["queries"][0]["cost_usd"] = None
                        else:
                            report["manifest"]["selection_mode"] = "wrong"
                        path.write_text(json.dumps(report))
                with self.assertRaisesRegex(RuntimeError, "failed validation"):
                    self.run_matrix(invalid)
                self.assertEqual(len(self.calls), 1)

    def test_missing_compiled_artifact_fails_before_any_paid_process(self):
        self.compiled.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_matrix()
        self.assertEqual(self.calls, [])

    def test_missing_rating_verifier_fails_before_any_paid_process(self):
        self.verifier.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Compiled DSPy rating verifier"):
            self.run_matrix()
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

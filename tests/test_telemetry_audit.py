"""Persist diagnostics safely and signal incomplete experiment accounting."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation_checkpoint import EvaluationCheckpoint, atomic_write_json
from src.telemetry_errors import exception_details
import run_ablation


class TelemetryAuditTests(unittest.TestCase):
    def test_redacted_diagnostics_survive_checkpoint_resume_and_final_json(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TEST_API_KEY": "private-credential-value"}):
            try:
                raise RuntimeError("Provider rejected private-credential-value; Authorization: Bearer another-credential")
            except RuntimeError as error:
                details = exception_details(error, "generation")
            root = Path(directory)
            path = root / "predict.checkpoint.jsonl"
            manifest = {"question_ids": ["7"]}
            record = {"question_id": "7", "metrics": {"question_id": "7", "error_details": details,
                      "stages": [{"error": "private-credential-value"}]}, "result": {"question_id": "7"}}
            with EvaluationCheckpoint(path, manifest) as checkpoint:
                checkpoint.append(record)
            with EvaluationCheckpoint(path, manifest, resume=True) as checkpoint:
                saved = checkpoint.records["7"]
            atomic_write_json(root / "predict.metrics.json", saved)
            for file in (path, root / "predict.metrics.json"):
                text = file.read_text(encoding="utf-8")
                self.assertNotIn("private-credential-value", text)
                self.assertNotIn("another-credential", text)
                self.assertIn("[REDACTED]", text)
                self.assertIn("RuntimeError", text)

    def test_cli_returns_failure_after_saving_incomplete_accounting_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predict.json"
            report = {"execution_accuracy": 0, "correct": 0, "total": 1,
                      "totals": {"total_tokens": None, "cost_usd": None, "accounting_complete": False},
                      "queries": [{"final_sql": None}]}
            def evaluate(*args, **kwargs):
                atomic_write_json(output.with_suffix(".metrics.json"), report)
            with patch("run_ablation.run_bird_evaluation", side_effect=evaluate), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "accounting is incomplete"):
                    run_ablation.main(["--dev-json", "unused.json", "--db-dir", "unused", "--output", str(output)])
            self.assertEqual(json.loads(output.with_suffix(".metrics.json").read_text()), report)


if __name__ == "__main__":
    unittest.main()

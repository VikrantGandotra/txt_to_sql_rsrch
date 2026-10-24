"""Question-level durability under interruption, corruption and changed controls."""

from contextlib import closing, redirect_stdout
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import bird_evaluator as evaluator
from src.evaluation_checkpoint import EvaluationCheckpoint, atomic_write_json
from tests.test_ablation import measured


class ResumeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mars_resume_")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.dev = self.root / "dev.json"
        self.db_dir = self.root / "databases"
        self.db = self.db_dir / "users/users.sqlite"
        self.db.parent.mkdir(parents=True)
        with closing(sqlite3.connect(self.db)) as connection:
            connection.executescript("CREATE TABLE users(name TEXT); INSERT INTO users VALUES('Ada');")
        self.items = [{"question_id": index, "db_id": "users", "question": "List users",
                       "SQL": "SELECT name FROM users"} for index in (7, 42, 99)]
        self.dev.write_text(json.dumps(self.items), encoding="utf-8")
        self.output = self.root / "predict.json"
        self.journal = self.output.with_suffix(".checkpoint.jsonl")

    def invoke(self, state):
        return {"final_sql": self.items[0]["SQL"], "metrics": measured(), "turn": 1}

    def run_evaluation(self, invoke=None, **kwargs):
        if kwargs.get("selection_mode") == "hybrid_cascade":
            artifact = self.root / "judge.json"
            artifact.write_text("{}", encoding="utf-8")
            kwargs.setdefault("verifier_path", str(artifact))
        with patch("src.bird_evaluator.build_sql_graph", return_value=SimpleNamespace(invoke=invoke or self.invoke)), \
             redirect_stdout(io.StringIO()):
            evaluator.run_bird_evaluation(str(self.dev), str(self.db_dir), str(self.output), **kwargs)
        return json.loads(self.output.with_suffix(".metrics.json").read_text(encoding="utf-8"))

    def test_interruption_resumes_only_pending_ids_and_preserves_seed_cost_and_ex(self):
        def interrupt(state):
            if state["question_id"] == "42":
                raise KeyboardInterrupt("simulated restart")
            return self.invoke(state)
        with self.assertRaises(KeyboardInterrupt):
            self.run_evaluation(interrupt)
        self.assertFalse(self.output.exists())
        self.assertEqual(len(self.journal.read_text(encoding="utf-8").splitlines()), 2)
        calls = []
        def resumed(state):
            calls.append((state["question_id"], state["rollout_seed"]))
            return self.invoke(state)
        original = evaluator._score_prediction
        with patch("src.bird_evaluator._score_prediction", wraps=original) as score:
            report = self.run_evaluation(resumed, resume=True)
        self.assertEqual(calls, [("42", 1), ("99", 2)])
        self.assertEqual(score.call_count, 2)
        self.assertEqual(report["resumed_questions"], 1)
        self.assertEqual(report["execution_accuracy"], 100)
        self.assertEqual(report["totals"]["prompt_tokens"], 30)
        self.assertAlmostEqual(report["totals"]["cost_usd"], 0.003)
        self.assertEqual(list(json.loads(self.output.read_text())), ["7", "42", "99"])

    def test_complete_resume_rebuilds_partial_final_files_and_paired_baseline_without_model_calls(self):
        report = self.run_evaluation(selection_mode="hybrid_cascade", num_candidates=2)
        self.output.write_text('{"torn":', encoding="utf-8")
        self.output.with_suffix(".self_consistency.json").unlink()
        with patch("src.bird_evaluator.build_sql_graph") as build, \
             patch("src.bird_evaluator._score_prediction") as score:
            restored = self.run_evaluation(resume=True, selection_mode="hybrid_cascade", num_candidates=2)
        build.assert_not_called()
        score.assert_not_called()
        self.assertEqual(restored["resumed_questions"], 3)
        self.assertEqual(restored["totals"], report["totals"])
        self.assertEqual(restored["self_consistency"], report["self_consistency"])
        self.assertEqual(restored["queries"], report["queries"])
        self.assertEqual(len(json.loads(self.output.read_text())), 3)

    def test_failed_questions_remain_committed_and_in_denominator(self):
        def fail(state):
            if state["question_id"] == "7":
                return {"final_sql": None, "metrics": measured(), "turn": 1}
            return self.invoke(state)
        self.run_evaluation(fail)
        never = Mock(side_effect=AssertionError("completed failures must not be retried"))
        report = self.run_evaluation(never, resume=True)
        never.assert_not_called()
        self.assertEqual(report["correct"], 2)
        self.assertEqual(report["total"], 3)
        self.assertIsNone(json.loads(self.output.read_text())["7"])

    def test_changed_configuration_dataset_database_and_source_are_rejected(self):
        self.run_evaluation()
        with self.assertRaisesRegex(ValueError, "temperature"):
            self.run_evaluation(resume=True, temperature=0.4)
        original = self.dev.read_text()
        self.dev.write_text(original + " ")
        with self.assertRaisesRegex(ValueError, "dataset_sha256"):
            self.run_evaluation(resume=True)
        self.dev.write_text(original)
        with patch("src.bird_evaluator._dependency_versions", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "dependencies"):
                self.run_evaluation(resume=True)
        original_hash = evaluator._file_hash
        def changed_hash(path):
            return "modified" if path and Path(path).name == "sql_agent_graph.py" else original_hash(path)
        with patch("src.bird_evaluator._file_hash", side_effect=changed_hash):
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                self.run_evaluation(resume=True)
        status = self.db.stat()
        os.utime(self.db, ns=(status.st_atime_ns, status.st_mtime_ns + 1000000000))
        with self.assertRaisesRegex(ValueError, "database_files"):
            self.run_evaluation(resume=True)

    def test_uncommitted_trailing_write_is_discarded_but_corrupt_commits_are_rejected(self):
        self.run_evaluation()
        original = self.journal.read_bytes()
        with self.journal.open("ab") as handle:
            handle.write(b'{"record": "interrupted UTF8 \xe2')
        self.run_evaluation(resume=True)
        self.assertEqual(self.journal.read_bytes(), original)
        lines = original.splitlines()
        event = json.loads(lines[1])
        event["record"]["prediction"] = None  # Do not update the checksum.
        lines[1] = json.dumps(event).encode()
        self.journal.write_bytes(b"\n".join(lines) + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_evaluation(resume=True)

    def test_existing_progress_is_not_overwritten_and_legacy_json_is_not_trusted_as_checkpoint(self):
        self.run_evaluation()
        original = self.journal.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_evaluation()
        self.assertEqual(self.journal.read_bytes(), original)
        self.journal.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "No resumable checkpoint"):
            self.run_evaluation(resume=True)

    def test_second_writer_is_rejected_and_lock_is_reusable_after_exit(self):
        manifest = {"question_ids": [], "created_at": "now"}
        with EvaluationCheckpoint(self.journal, manifest):
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                with EvaluationCheckpoint(self.journal, manifest, resume=True):
                    pass
        with EvaluationCheckpoint(self.journal, manifest, resume=True) as checkpoint:
            self.assertEqual(checkpoint.records, {})

    def test_atomic_final_write_preserves_existing_file_if_replace_fails(self):
        self.output.write_text('{"old": true}', encoding="utf-8")
        with patch("src.evaluation_checkpoint.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                atomic_write_json(self.output, {"new": True})
        self.assertEqual(json.loads(self.output.read_text()), {"old": True})
        self.assertFalse(list(self.root.glob(".predict.json.*")))


if __name__ == "__main__":
    unittest.main(verbosity=2)

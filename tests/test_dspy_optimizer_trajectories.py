"""Compilation uses trajectory observations and isolated training labels."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).resolve().parents[1] / ".dspy_cache"))

import dspy
from dspy.utils.dummies import DummyLM

from src.dspy_optimizer import (
    build_bird_trainset, build_trainset, build_verifier_trainset, compile_program,
    compile_verifier, collect_training_trajectories, TrajectoryTrainingProgram,
)
from src.dspy_telemetry import OpenRouterLM
from src.dspy_verifier import VERIFIER_MODEL, VERIFIER_MANIFEST_KEY


class TrajectoryCompilationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mars_compile_trajectory_")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.trainset = build_trainset(self.root / "training.sqlite")

    def test_training_forward_executes_probe_and_supplies_rows_to_final_action(self):
        example = self.trainset[0]
        lm = DummyLM([
            {"logical_plan": "Inspect maximum age, then return the matching name."},
            {"action": "probe", "rationale": "Find the maximum", "sql_query": "SELECT MAX(age) FROM users"},
            {"action": "final_answer", "rationale": "Age 85 is highest", "sql_query": example.sql_query},
        ])
        program = TrajectoryTrainingProgram(self.trainset)
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
            answer = program(**example.inputs())
        self.assertEqual(answer.action, "final_answer")
        self.assertEqual(len(answer.history), 2)
        self.assertEqual(answer.history[0]["action"], "probe")
        final_prompt = json.dumps(lm.history[-1]["messages"])
        self.assertIn("85", final_prompt)
        self.assertIn("<observation>", final_prompt)
        self.assertNotIn(str(self.root), final_prompt)

    def test_verifier_bootstrap_saves_both_classes_without_label_inputs(self):
        examples = build_verifier_trainset(self.trainset)
        for example in examples:
            self.assertEqual(set(example.inputs()), {"question", "evidence", "schema_ddl", "trajectory"})
            self.assertNotIn('"correct"', example.trajectory)
            self.assertNotIn('"gold_sql"', example.trajectory)
        lm = DummyLM([{"rationale": "Evaluate answer logic", "score": float(example.correct)} for example in examples])
        artifact = self.root / "compiled_verifier.json"
        compiled = compile_verifier(artifact, lm=lm)
        self.assertEqual(len(compiled.judge.demos), 8)
        state = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertEqual(state[VERIFIER_MANIFEST_KEY]["training_source"], "synthetic_smoke_only")
        self.assertEqual(state[VERIFIER_MANIFEST_KEY]["score_kind"], "elicited_validity_rating")
        for demo in state["judge"]["demos"]:
            self.assertNotIn("correct", demo)
            self.assertNotIn("db_path", demo)
        self.assertIsNone(state["judge"]["lm"])

    def test_training_split_and_batch_candidate_import(self):
        with self.assertRaisesRegex(ValueError, "training split"):
            build_bird_trainset(self.root / "dev.json", self.root)
        labeled = [item.copy(question_id=str(index)) for index, item in enumerate(self.trainset)]
        path = self.root / "training_candidates.json"
        path.write_text(json.dumps({"queries": [{
            "question_id": "0", "candidates": [
                {"final_sql": self.trainset[0].sql_query, "history": [], "execution_accuracy": 0},
                {"final_sql": "SELECT NULL", "history": [], "execution_accuracy": 100},
            ],
        }]}), encoding="utf-8")
        examples = build_verifier_trainset(labeled, path)
        self.assertEqual([example.correct for example in examples], [True, False])
        self.assertNotIn("execution_accuracy", examples[0].trajectory)
        path.write_text(json.dumps([{"question_id": "held_out", "final_sql": "SELECT NULL"}]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outside the training split"):
            build_verifier_trainset(labeled, path)

    def test_collect_normalizes_ids_and_protects_artifacts_without_live_calls(self):
        training_path = self.root / "train.json"
        training_path.write_text(json.dumps([
            {"question": "Count users", "db_id": "mock", "SQL": "SELECT COUNT(*) FROM users"},
            {"question_id": 41, "question": "Oldest user", "db_id": "mock", "SQL": self.trainset[0].sql_query},
        ]), encoding="utf-8")
        output = self.root / "training_predictions.json"

        def fake_evaluate(dataset, databases, prediction_path, limit, **kwargs):
            rows = json.loads(Path(dataset).read_text(encoding="utf-8"))
            self.assertEqual([row["question_id"] for row in rows], [0, 41])
            self.assertEqual(kwargs["selection_mode"], "self_consistency")
            self.assertEqual(kwargs["num_candidates"], 8)
            Path(prediction_path).write_text("{}", encoding="utf-8")
            Path(prediction_path).with_suffix(".metrics.json").write_text(
                json.dumps({"manifest": {"model_name": VERIFIER_MODEL}, "queries": []}), encoding="utf-8")

        with patch("src.bird_evaluator.run_bird_evaluation", side_effect=fake_evaluate) as run:
            report_path = collect_training_trajectories(training_path, self.root, output,
                                                        prompt_mode="uncompiled_dspy")
            run.assert_called_once()
        manifest = json.loads(report_path.read_text(encoding="utf-8"))["manifest"]
        self.assertEqual(manifest["split_role"], "training_only")
        self.assertEqual(manifest["training_source"], str(training_path.resolve()))
        self.assertEqual(len(manifest["training_source_sha256"]), 64)
        with patch("src.bird_evaluator.run_bird_evaluation") as run:
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                collect_training_trajectories(training_path, self.root, training_path)
            run.assert_not_called()

    def test_compilation_retains_nested_provider_accounting(self):
        answers = []
        for example in self.trainset:
            answers.extend([
                {"logical_plan": "Use the requested aggregation or filters"},
                {"action": "final_answer", "rationale": "Answer question", "sql_query": example.sql_query},
            ])
        scripted = DummyLM(answers)
        lm = OpenRouterLM(model=VERIFIER_MODEL, cache=False)

        def response(prompt=None, messages=None, **kwargs):
            result = scripted.forward(prompt=prompt, messages=messages, **kwargs)
            result.model = VERIFIER_MODEL
            result.usage.update(prompt_tokens=10, completion_tokens=2, total_tokens=12, cost=0.0001)
            return result

        artifact = self.root / "generator.json"
        with patch.object(dspy.LM, "forward", side_effect=response):
            compile_program(artifact, lm=lm)
        state = json.loads(artifact.read_text(encoding="utf-8"))
        metrics = state["mars_generator_manifest"]["compilation_metrics"]
        self.assertEqual(metrics["lm_call_count"], 8)
        self.assertEqual(metrics["prompt_tokens"], 80)
        self.assertAlmostEqual(metrics["cost_usd"], 0.0008)
        sidecar = json.loads(artifact.with_suffix(".compilation.metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(sidecar["lm_call_count"], 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)

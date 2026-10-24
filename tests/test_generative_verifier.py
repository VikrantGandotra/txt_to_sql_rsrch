"""Compatibility and source guards for the retired probability selector."""

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from src.generative_verifier import select_with_generative


class RetiredGenerativeVerifierTests(unittest.TestCase):
    def test_legacy_import_delegates_to_dspy_rating_judge(self):
        expected = {"mode": "dspy_verifier", "score_kind": "elicited_validity_rating"}
        with patch("src.generative_verifier.select_with_dspy", return_value=expected) as judge:
            with self.assertWarns(DeprecationWarning):
                actual = select_with_generative("Q", "E", "DDL", [], verification_samples=2)
        judge.assert_called_once_with("Q", "E", "DDL", [], verification_samples=2)
        self.assertEqual(actual, expected)

    def test_no_pipeline_source_requests_probability_parameters(self):
        root = Path(__file__).resolve().parents[1]
        sources = [*root.glob("*.py"), *(root / "src").glob("*.py")]
        forbidden = {"logprobs", "top_logprobs"}
        for source in sources:
            with self.subTest(source=source.name):
                tree = ast.parse(source.read_text(encoding="utf-8-sig"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.keyword):
                        self.assertNotIn(node.arg, forbidden)
                    if isinstance(node, ast.Dict):
                        for key in node.keys:
                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                self.assertNotIn(key.value, forbidden)


if __name__ == "__main__":
    unittest.main(verbosity=2)

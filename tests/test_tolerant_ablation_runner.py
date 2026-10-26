"""Offline tests of the actual inline recovery validator; no model/network calls."""
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'run_full_ablation_tolerant.sh'
BASH = Path('C:/Program Files/Git/bin/bash.exe')


class TolerantValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data').mkdir()
        text = SCRIPT.read_text(encoding='utf-8')
        self.validator = text.split("<<'VALIDATE'\n", 1)[1].split('\nVALIDATE\n', 1)[0]
        self.stage = dict(stage='generator', turn=1, prompt_tokens=2, completion_tokens=1,
                          total_tokens=3, cost_usd=0.0, error=None, lm_call_count=1,
                          calls=[dict(prompt_tokens=2, completion_tokens=1,
                                      total_tokens=3, cost_usd=0.0)])
        self.metrics = dict(prompt_tokens=2, completion_tokens=1, total_tokens=3,
                            cost_usd=0.0, accounting_complete=True, stages=[self.stage])
        self.config = dict(prompt_mode='uncompiled_dspy', selection_mode='self_consistency',
                           num_candidates=1, max_workers=1, max_turns=1, temperature=0,
                           top_p=1, seed=0, verification_samples=1,
                           execution_timeout_seconds=30.0, cross_model_fallback=False,
                           llm_backend='vllm')
        self.queries = []
        for i in range(1534):
            q = dict(question_id=str(i), db_id='db', configuration=self.config,
                     execution_correct=True, total_rollout_turns=1, final_sql='SELECT 1',
                     candidates=[dict(candidate_index=0, metrics=copy.deepcopy(self.metrics))],
                     selection={}, **copy.deepcopy(self.metrics))
            for field in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost_usd'):
                q['observed_' + field] = q[field]
            self.queries.append(q)
        (self.root / 'data/dev.json').write_text(json.dumps([
            dict(question_id=i) for i in range(1534)]))
        self.error = ('RuntimeError: Evaluation saved, but provider token/cost accounting '
                      'is incomplete; inspect per-query errors before continuing the matrix')

    def reject_request(self, index=506):
        q = self.queries[index]
        rejection = dict(stage='generator', turn=2, calls=[], lm_call_count=0,
                         error=('ContextWindowExceededError: maximum context length is 32768 tokens. '
                                'prompt contains at least 32769 input tokens, which exceeds this limit.'))
        component = q['candidates'][0]['metrics']
        for field in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost_usd'):
            rejection[field] = None
            component[field] = q[field] = None
        component['stages'].append(rejection)
        component['accounting_complete'] = q['accounting_complete'] = False
        return rejection

    def validate(self, exit_code=0, log=None, stale=False):
        fields = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost_usd')
        totals = dict(accounting_complete=all(q['accounting_complete'] for q in self.queries))
        for f in fields:
            values = [q[f] for q in self.queries]
            totals[f] = sum(values) if all(v is not None for v in values) else None
            totals['observed_' + f] = sum(q['observed_' + f] for q in self.queries)
        path = self.root / 'predict.metrics.json'
        path.write_text(json.dumps(dict(schema_version=4, total=1534, correct=1534,
                                       execution_accuracy=100, queries=self.queries, totals=totals)))
        (self.root / 'predict.json').write_text(json.dumps({
            q['question_id']: 'SELECT 1\t----- bird -----\tdb' for q in self.queries}))
        attempt = self.root / 'attempt.log'
        attempt.write_text(log if log is not None else (
            'Traceback (most recent call last):\n  File "run_ablation.py", line 58\n' + self.error
            if exit_code else 'Evaluation saved\n'))
        if stale:
            os.utime(path, (1, 1))
        return subprocess.run([
            sys.executable, '-c', self.validator, str(path), str(exit_code), str(attempt),
            str(time.time_ns()), 'uncompiled_dspy', 'self_consistency', '1', '1', '0', '1',
            str(SCRIPT)], cwd=self.root, capture_output=True, text=True)

    def test_complete_report_passes_without_mutation(self):
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'tolerance.acceptance.json').is_file())

    def test_documented_context_rejection_passes_with_nulls_preserved(self):
        self.reject_request()
        result = self.validate(1)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.root / 'predict.metrics.json').read_text())
        self.assertIsNone(report['totals']['total_tokens'])
        self.assertIn('ACCEPTED', result.stdout)

    def test_exact_vllm_error_with_and_without_retained_usage(self):
        exact = ("ContextWindowExceededError: [Qwen/Qwen2.5-7B-Instruct] "
                 "litellm.ContextWindowExceededError: litellm.BadRequestError: "
                 "ContextWindowExceededError: OpenAIException - This model's maximum "
                 "context length is 32768 tokens. However, your prompt contains at least "
                 "32769 input tokens, which exceeds this limit even without any output "
                 "tokens. Please reduce the length of the input prompt. "
                 "(parameter=input_tokens, value=32769)")
        self.reject_request()['error'] = exact
        # A different stage may retain billed adapter calls before an overflow.
        self.queries[20]['candidates'][0]['metrics']['stages'][0]['error'] = exact
        result = self.validate(1)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_complete_usage_with_adapter_error_is_accepted(self):
        self.reject_request()
        self.queries[20]['candidates'][0]['metrics']['stages'][0]['error'] = 'AdapterParseError: malformed output'
        result = self.validate(1)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_generic_context_message_accepted_within_threshold(self):
        self.reject_request()['error'] = 'BadRequestError: context too long'
        result = self.validate(1)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ten_incomplete_queries_accepted_without_stage_evidence(self):
        for i in range(10):
            self.reject_request(i)
            self.queries[i]['candidates'][0]['metrics']['stages'] = []
        result = self.validate(1)
        self.assertEqual(result.returncode, 0, result.stderr)
        decision = json.loads((self.root / 'tolerance.acceptance.json').read_text())
        self.assertEqual(len(decision['incomplete_question_ids']), 10)
        self.assertEqual(decision['policy'], 'tolerant-count-v2')

    def test_eleven_incomplete_queries_rejected(self):
        for i in range(11):
            self.reject_request(i)
        self.assertNotEqual(self.validate(1).returncode, 0)

    def test_unexplained_missing_usage_accepted_within_threshold(self):
        self.reject_request()['error'] = 'Missing provider usage'
        result = self.validate(1)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_error_strings_do_not_control_telemetry_acceptance(self):
        self.reject_request()
        self.queries[20]['candidates'][0]['error'] = 'AuthenticationError: 401 Unauthorized'
        result = self.validate(1)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_arithmetic_mismatch_rejected(self):
        self.queries[20]['total_tokens'] = 99
        self.assertNotEqual(self.validate().returncode, 0)

    def test_unrelated_exit_and_stale_report_rejected(self):
        self.reject_request()
        self.assertNotEqual(self.validate(2).returncode, 0)
        self.assertNotEqual(self.validate(1, stale=True).returncode, 0)

    def test_acceptance_does_not_match_exception_or_traceback_strings(self):
        self.reject_request()['error'] = 'APITimeoutError: no usage returned'
        result = self.validate(1, log='AdapterParseError: arbitrary formatting\n')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_complete_query_stage_arithmetic_remains_strict(self):
        self.reject_request()
        self.queries[20]['candidates'][0]['metrics']['stages'][0]['calls'][0]['total_tokens'] = 50
        self.assertNotEqual(self.validate(1).returncode, 0)

    @unittest.skipUnless(BASH.is_file(), 'Git Bash required')
    def test_bash_syntax_and_matrix_match_original(self):
        subprocess.run([str(BASH), '-n', SCRIPT.as_posix()], check=True)
        commands = []
        for runner in ('run_full_ablation.sh', SCRIPT.name):
            result = subprocess.run([str(BASH), (ROOT / runner).as_posix(), '--dry-run',
                                     self.root.as_posix()], capture_output=True, text=True, check=True)
            commands.append([shlex.split(line) for line in result.stdout.splitlines()])
        self.assertEqual(commands[0], commands[1])
        self.assertEqual(len(commands[1]), 8)


if __name__ == '__main__':
    unittest.main()

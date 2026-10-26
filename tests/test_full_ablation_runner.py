"""Audit the cloud matrix commands without connecting to a GPU or provider."""

from pathlib import Path
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path('C:/Program Files/Git/bin/bash.exe')
BASH = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which('bash')


@unittest.skipUnless(BASH, 'Bash is required to validate the cloud runner')
class FullAblationRunnerTests(unittest.TestCase):
    def fake_run(self, directory, fail):
        directory = Path(directory)
        binary = directory / 'bin'
        binary.mkdir()
        python = binary / 'application-python'
        python.write_text('''#!/usr/bin/env bash
if [[ "${1:-}" == -m ]]; then
    printf '%s\\n' "$*" >> "$MARS_TEST_STATUS_LOG"
    exit 0
fi
exit "${MARS_TEST_EXIT:-0}"
''', encoding='utf-8')
        flock = binary / 'flock'
        flock.write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8')
        python.chmod(0o700)
        flock.chmod(0o700)
        env = dict(os.environ, MARS_PYTHON=python.as_posix(),
                   MARS_TEST_STATUS_LOG=(directory/'status_calls').as_posix(), MARS_TEST_EXIT=str(fail))
        wrapper = '''test_bin="$1"; test_output="$3"
if command -v cygpath >/dev/null; then
    test_bin=$(cygpath -u "$test_bin")
    test_output=$(cygpath -u "$test_output")
    export MARS_PYTHON=$(cygpath -u "$MARS_PYTHON")
    export MARS_TEST_STATUS_LOG=$(cygpath -u "$MARS_TEST_STATUS_LOG")
fi
export PATH="$test_bin:$PATH"
exec bash "$2" "$test_output"
'''
        result = subprocess.run([BASH, '-c', wrapper,
            'test-runner', binary.as_posix(), (ROOT/'run_full_ablation.sh').as_posix(), (directory/'output').as_posix()],
            env=env, text=True, capture_output=True)
        self.assertTrue((directory/'status_calls').exists(), result.stderr + result.stdout)
        return result, (directory/'status_calls').read_text()

    def test_preflight_failure_reports_failure_and_preserves_exit_code(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            result, calls = self.fake_run(directory, 23)
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertIn('--status failed', calls)
            self.assertNotIn('--status completed', calls)

    def test_successful_matrix_reports_completion_without_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            result, calls = self.fake_run(directory, 0)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--status completed', calls)
            self.assertNotIn('--status failed', calls)

    def commands(self, directory):
        result = subprocess.run([BASH, (ROOT / 'run_full_ablation.sh').as_posix(), '--dry-run',
                                 Path(directory).as_posix()], capture_output=True, text=True, check=True)
        return [shlex.split(line) for line in result.stdout.splitlines() if line.strip()]

    def test_all_eight_commands_match_pilot_controls_and_full_dev_limit(self):
        subprocess.run([BASH, '-n', (ROOT / 'run_full_ablation.sh').as_posix()], check=True)
        with tempfile.TemporaryDirectory() as directory:
            commands = self.commands(directory)
            self.assertEqual(len(commands), 8)
            outputs = set()
            for command in commands:
                def value(flag):
                    return command[command.index(flag)+1]
                self.assertEqual(value('--limit'), '1534')
                self.assertEqual(value('--seed'), '0')
                outputs.add(value('--output'))
                greedy = '/greedy/' in value('--output')
                self.assertEqual(value('--candidates'), '1' if greedy else '8')
                self.assertEqual(value('--workers'), '1' if greedy else '8')
                self.assertEqual(value('--max-turns'), '1' if greedy else '5')
                self.assertEqual(value('--temperature'), '0' if greedy else '0.8')
                self.assertEqual(value('--top-p'), '1' if greedy else '0.7')
                if greedy:
                    self.assertEqual(value('--selection'), 'self_consistency')
            self.assertEqual(len(outputs), 8)

    def test_resume_flag_is_used_only_for_existing_cloud_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / 'uncompiled_dspy/self_consistency/predict.checkpoint.jsonl'
            journal.parent.mkdir(parents=True)
            journal.write_text('not read by dry run')
            commands = self.commands(directory)
            self.assertIn('--resume', commands[0])
            self.assertTrue(all('--resume' not in command for command in commands[1:]))


if __name__ == '__main__':
    unittest.main()

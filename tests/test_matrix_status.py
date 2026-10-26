"""Status markers and notifications must not expose research data or credentials."""

from contextlib import redirect_stdout
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import unittest

from src.matrix_status import publish_status


class MatrixStatusTests(unittest.TestCase):
    def test_failure_and_completion_markers_are_distinct_and_restart_clears_old_markers(self):
        with tempfile.TemporaryDirectory() as directory, patch("dotenv.dotenv_values", return_value={}), \
             patch.dict("os.environ", {"MARS_NOTIFY_URL": ""}), redirect_stdout(io.StringIO()):
            root = Path(directory)
            publish_status(root, "failed", "compiled_dspy/dspy_verifier", 1)
            self.assertTrue((root / "FAILED.json").exists())
            publish_status(root, "running", "preflight")
            self.assertFalse((root / "FAILED.json").exists())
            publish_status(root, "completed", "all_eight_configurations")
            self.assertEqual(json.loads((root / "STATUS.json").read_text())["status"], "completed")
            self.assertTrue((root / "COMPLETED.json").exists())

    def test_push_contains_only_status_and_notification_failure_keeps_local_evidence(self):
        with tempfile.TemporaryDirectory() as directory, patch("dotenv.dotenv_values", return_value={}), \
             patch.dict("os.environ", {"MARS_NOTIFY_URL": "https://ntfy.example/private-topic", "MARS_NOTIFY_TOKEN": "test-token"}), \
             redirect_stdout(io.StringIO()) as output, patch("src.matrix_status.urlopen", side_effect=OSError("secret response")) as send:
            publish_status(directory, "failed", "uncompiled_dspy/self_consistency", 1)
            request = send.call_args.args[0]
            self.assertNotIn(b"test-token", request.data)
            self.assertNotIn(b"secret response", request.data)
            self.assertNotIn("secret response", output.getvalue())
            self.assertTrue((Path(directory) / "FAILED.json").exists())


if __name__ == "__main__":
    unittest.main()

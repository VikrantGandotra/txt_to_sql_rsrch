"""Back up live checkpoint prefixes without mutating the running experiment."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from runpod_results_backup import encoded, copy_checkpoint, create_snapshot, backup_once


class ResultsBackupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "experiment"
        self.project.mkdir()
        (self.project / "run_ablation.py").write_text("# frozen experiment")
        self.journal = self.project / "results/full/uncompiled/predict.checkpoint.jsonl"
        self.journal.parent.mkdir(parents=True)
        header = {"version": 1, "manifest": {"question_ids": ["0", "1"]}}
        record = {"question_id": "0", "metrics": {"question_id": "0", "total_tokens": 12},
                  "result": {"question_id": "0", "correct": True}}
        event = {"record": record, "sha256": hashlib.sha256(encoded(record)).hexdigest()}
        self.journal.write_bytes(encoded(header) + b"\n" + encoded(event) + b"\n")

    def test_torn_tail_is_excluded_without_modifying_source(self):
        with self.journal.open("ab") as handle:
            handle.write(b'{"unfinished":')
        before = self.journal.read_bytes()
        destination = self.root / "copy.jsonl"
        self.assertEqual(copy_checkpoint(self.journal, destination), 1)
        self.assertEqual(self.journal.read_bytes(), before)
        self.assertEqual(len(destination.read_text().splitlines()), 2)

    def test_corrupt_committed_record_is_not_published(self):
        content = self.journal.read_bytes().replace(b'"total_tokens": 12', b'"total_tokens": 13')
        self.journal.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "checksum"):
            copy_checkpoint(self.journal, self.root / "copy.jsonl")

    def test_snapshot_excludes_credentials_databases_and_external_links(self):
        (self.project / ".env").write_text("PRIVATE_KEY=never-copy")
        (self.project / "rclone.conf").write_text("secret-token")
        (self.project / "service-account.json").write_text("secret-token")
        data = self.project / "data"; data.mkdir(); (data / "dev.sqlite").write_bytes(b"database")
        stage = self.root / "snapshot"; stage.mkdir()
        before = self.journal.read_bytes()
        report = create_snapshot(self.project, stage)
        self.assertEqual(sum(report["checkpoint_questions"].values()), 1)
        self.assertFalse((stage / ".env").exists())
        self.assertFalse((stage / "rclone.conf").exists())
        self.assertFalse((stage / "service-account.json").exists())
        self.assertFalse((stage / "data").exists())
        self.assertEqual(self.journal.read_bytes(), before)

    def run_fake(self, *, fail_check=False):
        state = self.root / "state"; state.mkdir()
        args = SimpleNamespace(project=self.project, state=state, remote="marsdrive:dedicated-folder",
                               config=self.root/"private.conf", rclone="rclone")
        calls = []
        def remote(_args, *command):
            calls.append(command)
            if command[0] == "lsjson":return '[{"Name":"LAST_SUCCESS.json"}]'
            if command[0] == "cat":return '{"slot":1}'
            if command[0] == "check" and fail_check:raise RuntimeError("verification failed")
            return ""
        with patch("runpod_results_backup.rclone", side_effect=remote):
            if fail_check:
                with self.assertRaises(RuntimeError):backup_once(args)
            else:
                result = backup_once(args)
                self.assertEqual(result["slot"], 0)
                self.assertEqual(result["archive"], "slot-0.tar.gz")
                self.assertIsNotNone(result["verified_utc"])
        return calls, state

    def test_last_good_slot_is_preserved_and_pointer_is_published_only_after_verification(self):
        calls, state = self.run_fake()
        actions = [call[0] for call in calls]
        self.assertEqual(actions, ["mkdir", "lsjson", "cat", "copy", "check", "copyto"])
        self.assertTrue((state / "LAST_SUCCESS.json").is_file())
        self.assertEqual(list(state.glob("snapshot_*")), [])

    def test_verification_failure_does_not_publish_a_new_success_pointer(self):
        calls, state = self.run_fake(fail_check=True)
        self.assertNotIn("copyto", [call[0] for call in calls])
        self.assertFalse((state / "LAST_SUCCESS.json").exists())


if __name__ == "__main__":
    unittest.main()

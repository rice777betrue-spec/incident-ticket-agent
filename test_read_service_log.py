"""使用临时文件测试，不修改真实 service.log。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import read_service_log as reader


class LogTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(directory) / "service.log"
        self.enterContext(patch.object(reader, "LOG_PATH", self.path))

    def test_missing_log(self):
        result = reader.read_service_log()
        self.assertEqual(result["status"], "missing")
        self.assertEqual(result["lines"], [])
        self.assertFalse(self.path.exists())

    def test_empty_log(self):
        self.path.write_text("", encoding="utf-8")
        self.assertEqual(reader.read_service_log()["status"], "empty")

    def test_last_twenty_lines_and_no_changes(self):
        text = "\n".join(f"日志 {i}" for i in range(25)) + "\n"
        self.path.write_text(text, encoding="utf-8")
        before = self.path.read_bytes()
        result = reader.read_service_log()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["lines"], [f"日志 {i}" for i in range(5, 25)])
        self.assertEqual(self.path.read_bytes(), before)

    def test_unreadable_log(self):
        with patch.object(Path, "open", side_effect=PermissionError("test denied")):
            result = reader.read_service_log()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["lines"], [])

    def test_invalid_utf8_does_not_crash(self):
        self.path.write_bytes(b"request\xff\n")
        self.assertEqual(reader.read_service_log()["lines"], ["request\ufffd"])


if __name__ == "__main__":
    unittest.main()

"""报告组合测试：保留证据来源，不把历史日志当成本次健康结果。"""

import unittest
from datetime import datetime
from unittest.mock import patch

import report_service as reporter
from test_service_fault import demo_server


class ReportTests(unittest.TestCase):
    def test_tools_called_once_and_in_order(self):
        events = []
        current = {"evidence": ["测试服务正常"], "assessment": "本次健康检查通过，无需修复"}
        history = {"status": "empty", "lines": [], "message": "历史证据"}
        def check():
            events.append("check")
            return current
        def log():
            events.append("log")
            return history
        with patch.object(reporter, "diagnose_service", side_effect=check) as diagnose, patch.object(reporter, "read_service_log", side_effect=log) as read:
            report = reporter.collect_report()
        diagnose.assert_called_once_with()
        read.assert_called_once_with()
        self.assertEqual(events, ["check", "log"])
        self.assertEqual(report["current_check"], current)
        self.assertEqual(report["historical_log"], history)
        self.assertEqual(report["mode"], "read_only")
        start = datetime.fromisoformat(report["started_at"])
        finish = datetime.fromisoformat(report["finished_at"])
        self.assertIsNotNone(start.utcoffset())
        self.assertIsNotNone(finish.utcoffset())

    def test_unavailable_log_keeps_current_evidence(self):
        current = {"evidence": ["接口返回 HTTP 错误：503"], "assessment": "需要排查"}
        for status in ("missing", "empty", "error"):
            with self.subTest(status=status):
                history = {"status": status, "lines": [], "message": "没有可用日志"}
                with patch.object(reporter, "diagnose_service", return_value=current), patch.object(reporter, "read_service_log", return_value=history):
                    report = reporter.collect_report()
                self.assertEqual(report["current_check"], current)
                self.assertEqual(report["historical_log"], history)

    def test_real_fault_is_not_overridden_by_old_200(self):
        history = {"status": "ok", "lines": ['OLD: "GET /health HTTP/1.1" 200 -'], "message": "历史证据"}
        with demo_server(fault=True), patch.object(reporter, "read_service_log", return_value=history):
            report = reporter.collect_report()
        evidence = report["current_check"]["evidence"]
        self.assertIn("503", evidence[0])
        self.assertIn("open", evidence[1])
        self.assertIn("未独立验证", evidence[2])
        self.assertEqual(report["historical_log"], history)
        self.assertIn("历史 HTTP 200 不能证明本次服务健康", " ".join(report["notes"]))


if __name__ == "__main__":
    unittest.main()

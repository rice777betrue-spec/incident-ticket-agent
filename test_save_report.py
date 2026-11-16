"""只使用临时目录验证报告保存，不写入项目的真实报告目录。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import task_store

import ai_check as agent
from ai_client import ModelConnectionError
from test_ai_check import tool_message

import report_service as reporter


class SaveReportTests(unittest.TestCase):
    def setUp(self):
        task_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(task_store, "TASKS_PATH", task_root / "tasks.sqlite3"))
        root = self.enterContext(tempfile.TemporaryDirectory())
        self.directory = Path(root) / "reports"
        self.enterContext(patch.object(reporter, "REPORTS_DIR", self.directory))
        self.report = {
            "mode": "read_only",
            "current_check": {"evidence": ["接口返回 HTTP 错误：503"]},
            "historical_log": {"status": "missing", "lines": []},
        }

    def run_main(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(reporter, "collect_report", return_value=self.report) as collect, redirect_stdout(stdout), redirect_stderr(stderr):
            code = reporter.main(args)
        collect.assert_called_once_with()
        self.assertEqual(json.loads(stdout.getvalue()), self.report)
        return code, stderr.getvalue()

    def test_default_does_not_create_reports_directory(self):
        code, message = self.run_main([])
        self.assertEqual(code, 0)
        self.assertEqual(message, "")
        self.assertFalse(self.directory.exists())

    def test_save_flag_writes_same_report_as_terminal(self):
        code, message = self.run_main(["--save"])
        self.assertEqual(code, 0)
        paths = list(self.directory.glob("*.json"))
        self.assertEqual(len(paths), 1)
        self.assertEqual(json.loads(paths[0].read_text(encoding="utf-8")), self.report)
        self.assertIn("接口返回", paths[0].read_text(encoding="utf-8"))
        self.assertIn(str(paths[0]), message)

    def test_repeated_saves_preserve_previous_file(self):
        first = reporter.save_report(self.report)
        original = first.read_bytes()
        second = reporter.save_report(self.report)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(len(list(self.directory.glob("*.json"))), 2)

    def test_name_collision_does_not_overwrite(self):
        with patch.object(reporter, "datetime") as clock, patch.object(reporter, "uuid4") as unique:
            clock.now.return_value.astimezone.return_value.strftime.return_value = "fixed_time"
            unique.return_value.hex = "fixed_id"
            first = reporter.save_report(self.report)
            original = first.read_bytes()
            with self.assertRaises(FileExistsError):
                reporter.save_report({"different": True})
        self.assertEqual(first.read_bytes(), original)

    def test_save_error_keeps_terminal_report_and_returns_failure(self):
        with patch.object(reporter, "save_report", side_effect=PermissionError("测试：无写权限")):
            code, message = self.run_main(["--save"])
        self.assertEqual(code, 1)
        self.assertIn("报告保存失败", message)
        self.assertNotIn("报告已保存", message)


class AgentReportTests(unittest.TestCase):
    def setUp(self):
        task_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(task_store, "TASKS_PATH", task_root / "tasks.sqlite3"))
        root = self.enterContext(tempfile.TemporaryDirectory())
        self.directory = Path(root) / "reports"
        self.enterContext(patch.object(reporter, "REPORTS_DIR", self.directory))
        self.output = self.enterContext(redirect_stdout(io.StringIO()))
        self.model = self.enterContext(patch.object(agent, "request_message"))
        self.tool = self.enterContext(patch.object(agent, "call_tool"))
        self.session = self.enterContext(patch.object(agent, "RecoverySession")).return_value
        self.collect = self.enterContext(patch.object(reporter, "collect_report"))
        self.tool.return_value = {"tool": "check_service", "call_status": "completed",
                                  "result": {"healthy": False, "status_code": 503}}

    def loop(self, inputs):
        with patch("builtins.input", side_effect=inputs):
            agent.chat_loop()
        self.session.close.assert_called()
        self.collect.assert_not_called()

    def files(self):
        return list(self.directory.glob("*.md"))

    def test_both_save_aliases_reuse_same_evidence_without_model_or_check(self):
        self.model.side_effect = [tool_message(), {"content": "### 分析\n接口返回503，根因待确认。"}]
        self.loop(["检查服务", "保存报告", "/保存报告", "退出"])
        self.assertEqual(self.model.call_count, 2)
        self.tool.assert_called_once_with("check_service")
        self.session.start_with_approval.assert_not_called()
        paths = self.files()
        self.assertEqual(len(paths), 2)
        self.assertNotEqual(paths[0], paths[1])
        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn("检查服务", text)
            self.assertIn("### 分析\n接口返回503，根因待确认。", text)
            self.assertIn('"status_code": 503', text)
            self.assertIn('"checked_at":', text)
            self.assertIn("本轮完成时间", text)
            self.assertIn("文件保存时间", text)
            self.assertIn(str(path), self.output.getvalue())
            self.assertEqual(path.parent, self.directory)

    def test_no_report_does_not_create_directory_or_contact_ai(self):
        self.loop(["保存报告", "/保存报告", "退出"])
        self.assertFalse(self.directory.exists())
        self.model.assert_not_called()
        self.tool.assert_not_called()
        self.assertIn("暂无可保存", self.output.getvalue())

    def test_no_save_command_does_not_write_files(self):
        self.model.return_value = {"content": "解释健康检查。"}
        self.loop(["解释一下", "退出"])
        self.assertFalse(self.directory.exists())

    def test_plain_conversation_preserves_full_markdown_and_marks_no_checks(self):
        reply = "### 解释\n\n" + "保留完整分析与排版。\n\t- 项目\n" * 400
        self.model.return_value = {"content": reply}
        self.loop(["解释一下", "保存报告", "退出"])
        text = self.files()[0].read_text(encoding="utf-8")
        self.assertIn(reply.strip(), text)
        self.assertIn("本轮未执行检查", text)
        self.tool.assert_not_called()
        self.model.assert_called_once()

    def test_only_latest_turn_is_saved_not_prior_history(self):
        self.model.side_effect = [tool_message(), {"content": "先前检查503。"}, {"content": "这次只是概念解释。"}]
        self.loop(["检查", "解释一下", "保存报告", "退出"])
        text = self.files()[0].read_text(encoding="utf-8")
        self.assertNotIn("先前检查503", text)
        self.assertNotIn('"status_code":', text)
        self.assertIn("本轮未执行检查", text)
        self.assertIn("这次只是概念解释。", text)

    def test_clear_or_failed_turn_or_local_operation_invalidates_old_report(self):
        for command in ["清空", "/clear", "新的任务", "/启动服务"]:
            with self.subTest(command=command):
                self.model.reset_mock()
                self.model.side_effect = [{"content": "旧分析"}, ModelConnectionError("模拟失败")]
                self.session.start_with_approval.return_value = {"status": "cancelled", "message": "未启动"}
                self.loop(["解释一下", command, "保存报告", "退出"])
                self.assertFalse(self.directory.exists())

    def test_rejected_blank_and_long_input_preserve_latest_report(self):
        self.model.return_value = {"content": "完整分析"}
        self.loop(["解释一下", "", "字" * 2001, "保存报告", "退出"])
        self.assertEqual(len(self.files()), 1)
        self.model.assert_called_once()

    def test_write_error_is_truthful_and_retry_does_not_rerun_task(self):
        self.model.return_value = {"content": "完整分析"}
        with patch.object(agent, "save_agent_report", side_effect=[PermissionError("禁止写入"), self.directory / "mock.md"]) as save:
            self.loop(["解释一下", "保存报告", "保存报告", "退出"])
        self.assertEqual(save.call_count, 2)
        self.assertEqual(save.call_args_list[0], save.call_args_list[1])
        self.model.assert_called_once()
        self.tool.assert_not_called()
        self.assertEqual(self.output.getvalue().count("报告保存失败"), 1)
        self.assertEqual(self.output.getvalue().count("报告已保存："), 1)

    def test_question_summary_and_evidence_secrets_are_redacted(self):
        # 全部是测试用假值，不读取或依赖用户实际密钥。
        secret = "fake_private_credential_7391"
        with patch.dict(agent.os.environ, {"WORKBUDDY_API_KEY": secret}):
            question = f"检查 {secret} sk-fake123 password=secret123"
            self.model.side_effect = [tool_message(), {"content": f"分析 {secret} Bearer fake_bearer_value api_key=abc123"}]
            self.tool.return_value["result"]["message"] = "refresh_token=refresh_fake"
            self.loop([question, "保存报告", "退出"])
        text = self.files()[0].read_text(encoding="utf-8")
        for value in [secret, "sk-fake123", "secret123", "fake_bearer_value", "abc123", "refresh_fake"]:
            self.assertNotIn(value, text)
        self.assertIn("已隐藏", text)

    def test_snapshot_does_not_mutate_turn_and_keeps_original_evidence_time(self):
        evidence = {"checked_at": "2026-11-12T10:00:00+08:00", "tool_result": {"result": {"message": "api_key=fake_test"}}}
        turn = [{"role": "user", "content": "问题"}, {"role": "tool", "content": json.dumps(evidence)}, {"role": "assistant", "content": "完整分析"}]
        original = json.dumps(turn)
        report = agent.build_agent_report(turn)
        self.assertEqual(json.dumps(turn), original)
        self.assertEqual(report["evidence"][0]["checked_at"], evidence["checked_at"])
        self.assertNotIn("fake_test", json.dumps(report))

    def test_start_report_preserves_post_start_verification(self):
        self.model.side_effect = [tool_message("request_start_service"), {"content": "启动后复查通过。"}]
        self.session.start_with_approval.return_value = {
            "status": "verified", "message": "启动后复查通过",
            "checks": [{"stage": "before"}],
            "post_start_check": {"checked_at": "2026-11-13T10:00:00+08:00", "health": {"healthy": True}, "process_running": True},
        }
        self.loop(["帮我启动服务", "保存报告", "退出"])
        text = self.files()[0].read_text(encoding="utf-8")
        self.assertIn('"status": "verified"', text)
        self.assertIn('"post_start_check":', text)
        self.assertIn("2026-11-13T10:00:00+08:00", text)
        self.session.start_with_approval.assert_called_once()
        self.assertEqual(self.model.call_count, 2)
        self.tool.assert_not_called()

    def test_question_and_log_backticks_cannot_close_code_fences(self):
        report = {"completed_at": "fixed", "question": "```\n问题", "analysis": "分析", "evidence": [{"message": "`````日志"}]}
        text = reporter.save_agent_report(report).read_text(encoding="utf-8")
        self.assertIn("````\n```\n问题\n````", text)
        self.assertIn("``````json\n", text)
        self.assertTrue(text.endswith("``````\n"))

    def test_agent_filename_collision_never_overwrites(self):
        report = {"completed_at": "fixed", "question": "问题", "analysis": "分析", "evidence": []}
        with patch.object(reporter, "uuid4") as unique:
            unique.return_value.hex = "fixed"
            with patch.object(reporter, "datetime") as clock:
                clock.now.return_value.astimezone.return_value.strftime.return_value = "fixed"
                path = reporter.save_agent_report(report)
                original = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    reporter.save_agent_report(report)
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

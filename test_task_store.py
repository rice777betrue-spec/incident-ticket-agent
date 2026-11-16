"""任务档案测试：临时 SQLite、模拟模型/进程，不访问真实18080或反代。"""

import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from unittest.mock import Mock, patch

import ai_check as agent
from ai_client import ModelConnectionError
import recover_service as recovery
import task_store as tasks
from test_ai_check import tool_message


class TaskStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.root / "archive" / "tasks.sqlite3"
        self.store = self.enterContext(tasks.TaskStore(self.path, sanitizer=agent.redact_text))

    def test_reopen_preserves_problem_events_and_full_analysis(self):
        record = self.store.create("查一下服务")
        record.event("tool_result", {"healthy": False, "checked_at": "2026-11-13T10:00:00+08:00"})
        analysis = "### AI分析\n" + "根因尚未确认。" * 200
        record.finish("completed", {"analysis": analysis})
        self.store.close()
        with tasks.TaskStore(self.path) as reopened:
            task = reopened.get_task(record.id)
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["question"], "查一下服务")
        self.assertEqual(task["events"][-1]["payload"]["analysis"], analysis)
        self.assertEqual([e["kind"] for e in task["events"]], ["created", "tool_result", "completed"])
        sequences = [e["sequence"] for e in task["events"]]
        self.assertEqual(sequences, sorted(set(sequences)))
        self.assertTrue(all(e["recorded_at"] for e in task["events"]))

    def test_separate_process_can_read_without_replaying(self):
        record = self.store.create("跨进程读取")
        record.event("executing", {"note": "意图，不代表执行成功"})
        script = ("import sys; from task_store import TaskStore; "
                  "s=TaskStore(sys.argv[1]); t=s.get_task(sys.argv[2]); "
                  "print(t['status'], t['phase']); s.close()")
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", script, str(self.path), record.id],
                                cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("running executing", result.stdout)
        self.assertEqual(self.store.get_task(record.id)["status"], "running")

    def test_terminal_task_cannot_accept_events_or_be_executed_again(self):
        for status in sorted(tasks.TERMINAL_STATUSES):
            record = self.store.create(status)
            record.finish(status, {})
            with self.assertRaises(tasks.TaskStoreError):
                record.event("executing", {})
            self.assertEqual(len(self.store.get_task(record.id)["events"]), 2)

    def test_competing_connections_cannot_finish_same_task_twice(self):
        record = self.store.create("test")
        with tasks.TaskStore(self.path) as other:
            record.finish("completed", {"winner": True})
            with self.assertRaises(tasks.TaskStoreError):
                other.append(record.id, "failed", {}, status="failed")
        self.assertEqual(self.store.get_task(record.id)["status"], "completed")

    def test_invalid_id_unknown_id_and_sql_are_not_accepted(self):
        for value in ["../../private", "task_a' OR 1=1 --", "x", None]:
            with self.subTest(value=value), self.assertRaises(tasks.TaskStoreError):
                self.store.get_task(value)
        self.assertIsNone(self.store.get_task("task_" + "0" * 32))

    def test_list_is_newest_first_and_bounded(self):
        for i in range(23):
            self.store.create(str(i))
        rows = self.store.list_tasks()
        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0]["question"], "22")
        self.assertEqual(rows[-1]["question"], "3")

    def test_secrets_redacted_recursively_in_all_saved_fields(self):
        secret = "custom-private-test-value"
        with patch.dict(os.environ, {"WORKBUDDY_API_KEY": secret}):
            record = self.store.create(f"{secret} sk-example-secret password=other-secret")
            record.event("tool_result", {secret: ["Bearer private-token", "api_key=another-secret"]})
            record.finish("failed", {"error": secret})
        snapshot = json.dumps(self.store.get_task(record.id))
        for value in [secret, "sk-example-secret", "other-secret", "private-token", "another-secret"]:
            self.assertNotIn(value, snapshot)
        self.assertIn("隐藏", str(self.store.get_task(record.id)))

    def test_bad_payload_rolls_back_state_and_event(self):
        record = self.store.create("test")
        with self.assertRaises(tasks.TaskStoreError):
            record.finish("completed", {"bad": float("nan")})
        task = self.store.get_task(record.id)
        self.assertEqual(task["status"], "running")
        self.assertEqual(task["phase"], "created")
        self.assertEqual(len(task["events"]), 1)

    def test_insert_failure_rolls_back_status_update(self):
        record = self.store.create("test")
        self.store.connection.execute("""CREATE TRIGGER block_event BEFORE INSERT ON task_events
                                     BEGIN SELECT RAISE(ABORT, 'test'); END""")
        with self.assertRaises(tasks.TaskStoreError):
            record.finish("completed", {})
        task = self.store.get_task(record.id)
        self.assertEqual((task["status"], task["phase"]), ("running", "created"))
        self.assertEqual(len(task["events"]), 1)

    def test_unsupported_schema_and_invalid_path_fail_cleanly(self):
        wrong = self.root / "future.sqlite3"
        with closing(sqlite3.connect(wrong)) as connection:
            connection.execute("PRAGMA user_version=99")
        with self.assertRaises(tasks.TaskStoreError):
            tasks.TaskStore(wrong)
        file = self.root / "file"
        file.write_text("not a folder", encoding="utf-8")
        with self.assertRaises(tasks.TaskStoreError):
            tasks.TaskStore(file / "tasks.sqlite3")


class TaskChatTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = root / "tasks.sqlite3"
        self.enterContext(patch.object(tasks, "TASKS_PATH", self.path))
        self.output = self.enterContext(redirect_stdout(io.StringIO()))
        self.model = self.enterContext(patch.object(agent, "request_message"))
        self.tool = self.enterContext(patch.object(agent, "call_tool"))
        self.session = self.enterContext(patch.object(agent, "RecoverySession")).return_value
        self.tool.return_value = {"tool": "check_service", "call_status": "completed", "result": {"healthy": False}}

    def loop(self, inputs):
        with patch("builtins.input", side_effect=inputs):
            agent.chat_loop()

    def rows(self):
        with tasks.TaskStore(self.path) as store:
            return [store.get_task(row["id"]) for row in store.list_tasks()]

    def test_check_evidence_and_genuine_model_markdown_saved(self):
        reply = "### 模型分析\n接口失败，但原因还没确认。\n" + "完整分析。" * 150
        self.model.side_effect = [tool_message(), {"content": reply}]
        self.loop(["检查服务", "退出"])
        task, = self.rows()
        self.assertEqual(task["status"], "completed")
        saved = task["events"][-1]["payload"]["report"]
        self.assertEqual(saved["analysis"], reply)
        self.assertEqual(saved["task_id"], task["id"])
        evidence = next(e["payload"] for e in task["events"] if e["kind"] == "tool_result")
        self.assertEqual(saved["evidence"], [evidence])
        model_evidence = json.loads(self.model.call_args.args[0][-1]["content"])
        self.assertEqual(evidence, model_evidence)
        self.assertFalse(evidence["tool_result"]["result"]["healthy"])

    def test_model_error_after_check_keeps_partial_evidence_and_no_fake_report(self):
        self.model.side_effect = [tool_message(), ModelConnectionError("模型断开")]
        self.loop(["检查", "退出"])
        task, = self.rows()
        self.assertEqual(task["status"], "failed")
        self.assertEqual(sum(e["kind"] == "tool_result" for e in task["events"]), 1)
        self.assertNotIn('"report"', json.dumps(task))
        self.tool.assert_called_once()

    def test_model_error_before_tools_is_archived(self):
        self.model.side_effect = ModelConnectionError("模型断开")
        self.loop(["检查", "退出"])
        task, = self.rows()
        self.assertEqual(task["status"], "failed")
        self.tool.assert_not_called()

    def test_interrupt_is_archived_and_session_closed(self):
        self.model.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.loop(["检查"])
        self.assertEqual(self.rows()[0]["status"], "interrupted")
        self.session.close.assert_called_once()

    def test_queries_after_reopen_do_not_call_model_tools_or_start(self):
        self.model.return_value = {"content": "仅解释，没有检查"}
        self.loop(["解释一下", "退出"])
        task_id = self.rows()[0]["id"]
        self.model.reset_mock()
        self.loop(["任务列表", f"查看任务 {task_id}", "/任务列表", f"/查看任务 {task_id}", "退出"])
        self.model.assert_not_called()
        self.tool.assert_not_called()
        self.session.start_with_approval.assert_not_called()
        self.assertIn("不保证当前服务状态", self.output.getvalue())
        self.assertEqual(len(self.rows()), 1)

    def test_invalid_query_does_not_become_model_question(self):
        self.loop(["/查看任务", "/查看任务 ../../secret", "/查看任务 task_" + "0" * 32, "退出"])
        self.model.assert_not_called()
        self.assertEqual(self.rows(), [])

    def test_clear_does_not_delete_tasks_or_restore_old_context(self):
        self.model.return_value = {"content": "未执行检查"}
        self.loop(["第一轮", "清空", "第二轮", "退出"])
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual(len(self.model.call_args.args[0]), 2)

    def test_create_failure_prevents_model_tool_and_local_start(self):
        with patch.object(tasks.TaskStore, "create", side_effect=tasks.TaskStoreError("写入失败")):
            self.loop(["检查", "/启动服务", "退出"])
        self.model.assert_not_called()
        self.tool.assert_not_called()
        self.session.start_with_approval.assert_not_called()

    def test_store_open_failure_prevents_all_actions(self):
        with patch.object(agent, "TaskStore", side_effect=tasks.TaskStoreError("无法打开")):
            self.loop(["/启动服务", "退出"])
        self.model.assert_not_called()
        self.session.start_with_approval.assert_not_called()

    def test_main_reports_store_open_failure_with_nonzero_exit_code(self):
        with patch.object(agent, "TaskStore", side_effect=tasks.TaskStoreError("不可用")):
            self.assertEqual(agent.main(["--chat"]), 1)
        self.model.assert_not_called()

    def test_report_export_includes_same_task_id(self):
        import report_service as reporter
        self.model.return_value = {"content": "本轮仅解释，没有检查"}
        with patch.object(reporter, "REPORTS_DIR", self.path.parent / "reports"):
            self.loop(["解释", "保存报告", "退出"])
        task, = self.rows()
        report, = (self.path.parent / "reports").glob("*.md")
        self.assertIn(task["id"], report.read_text(encoding="utf-8"))
        self.model.assert_called_once()
        self.tool.assert_not_called()

    def test_tool_intent_write_failure_prevents_side_effect(self):
        self.model.return_value = tool_message("request_start_service")
        original = tasks.TaskStore.append
        def append(store, task_id, kind, payload, **kwargs):
            if kind == "tool_requested":
                raise tasks.TaskStoreError("不可写")
            return original(store, task_id, kind, payload, **kwargs)
        with patch.object(tasks.TaskStore, "append", new=append):
            self.loop(["帮我启动", "退出"])
        self.session.start_with_approval.assert_not_called()
        self.assertEqual(self.rows()[0]["status"], "failed")

    def test_evidence_write_failure_stops_followup_model_request(self):
        self.model.return_value = tool_message()
        original = tasks.TaskStore.append
        def append(store, task_id, kind, payload, **kwargs):
            if kind == "tool_result":
                raise tasks.TaskStoreError("不可写")
            return original(store, task_id, kind, payload, **kwargs)
        with patch.object(tasks.TaskStore, "append", new=append):
            self.loop(["检查", "退出"])
        self.tool.assert_called_once()
        self.model.assert_called_once()
        self.assertIn("实际结果", self.output.getvalue())
        self.assertEqual(self.rows()[0]["status"], "failed")

    def test_failed_finish_is_not_presented_as_complete_or_exportable(self):
        self.model.return_value = {"content": "真实回答"}
        with patch.object(tasks.TaskRecord, "finish", side_effect=tasks.TaskStoreError("不可写")), \
                patch.object(agent, "save_agent_report") as save:
            self.loop(["解释", "保存报告", "退出"])
        save.assert_not_called()
        self.assertEqual(self.rows()[0]["status"], "running")
        self.assertIn("结束状态未能写入", self.output.getvalue())

    def test_saved_problem_evidence_analysis_and_model_error_are_redacted(self):
        key = "private-test-exact-key"
        self.tool.return_value["result"]["message"] = key
        self.model.side_effect = [tool_message(), {"content": f"**分析** {key}"}, ModelConnectionError(key)]
        with patch.dict(os.environ, {"WORKBUDDY_API_KEY": key}):
            self.loop([f"检查 {key}", f"解释 {key}", "退出"])
        self.assertNotIn(key, json.dumps(self.rows(), ensure_ascii=False))


class RecoveryTaskEventsTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = root / "tasks.sqlite3"
        self.enterContext(patch.object(tasks, "TASKS_PATH", self.path))
        self.store = self.enterContext(tasks.TaskStore(self.path, sanitizer=agent.redact_text))
        self.record = self.store.create("启动", "local_command")
        self.session = recovery.RecoverySession()
        self.enterContext(redirect_stdout(io.StringIO()))
        self.addCleanup(self.session.close)
        self.inspect = self.enterContext(patch.object(recovery, "inspect_start_eligibility", return_value={
            "allowed": True, "health": {"healthy": False, "message": "未监听"}, "port_state": "refused", "message": "可申请"}))
        self.approval = self.enterContext(patch.object(recovery, "ask_for_approval", return_value=True))
        self.start = self.enterContext(patch.object(recovery, "start_service"))
        self.process = self.start.return_value
        self.process.poll.return_value = None
        self.enterContext(patch.object(recovery, "wait_until_healthy", return_value=True))
        self.enterContext(patch.object(recovery, "check_service", return_value={"healthy": True}))

    def kinds(self):
        return [event["kind"] for event in self.store.get_task(self.record.id)["events"]]

    def test_actual_approval_execution_and_verification_chronology(self):
        result = agent.approved_start_result(self.session, self.record)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(self.kinds(), ["created", "start_precheck", "awaiting_approval", "approval_decision",
                                      "start_recheck", "executing", "verifying", "post_start_check"])
        self.assertEqual(self.inspect.call_count, 2)
        self.approval.assert_called_once_with()
        post = self.store.get_task(self.record.id)["events"][-1]["payload"]
        self.assertEqual(post, result["post_start_check"])

    def test_denial_never_reaches_execution(self):
        self.approval.return_value = False
        result = agent.approved_start_result(self.session, self.record)
        self.assertEqual(result["status"], "cancelled")
        self.assertNotIn("executing", self.kinds())
        self.start.assert_not_called()
        decision = self.store.get_task(self.record.id)["events"][-1]["payload"]
        self.assertFalse(decision["approved"])
        self.assertFalse(decision["approval_reusable"])

    def test_changed_state_after_approval_is_still_ineligible(self):
        self.inspect.side_effect = [self.inspect.return_value, {"allowed": False, "message": "已被占用"}]
        result = agent.approved_start_result(self.session, self.record)
        self.assertEqual(result["status"], "state_changed")
        self.start.assert_not_called()
        self.assertIn("start_recheck", self.kinds())

    def test_write_failure_at_each_preaction_boundary_fails_closed(self):
        for boundary in ["start_precheck", "awaiting_approval", "approval_decision", "start_recheck", "executing"]:
            self.approval.reset_mock()
            self.start.reset_mock()
            def observe(kind, payload):
                if kind == boundary:
                    raise tasks.TaskStoreError("不可写")
            with self.subTest(boundary=boundary), self.assertRaises(tasks.TaskStoreError):
                self.session.start_with_approval(observer=observe)
            self.start.assert_not_called()

    def test_write_failure_after_process_creation_cleans_only_owned_process(self):
        def observe(kind, payload):
            if kind == "verifying":
                raise tasks.TaskStoreError("不可写")
        with self.assertRaises(tasks.TaskStoreError):
            self.session.start_with_approval(observer=observe)
        self.start.assert_called_once()
        self.process.terminate.assert_called_once()
        self.assertIsNone(self.session.process)

    def test_model_failure_after_verified_start_keeps_actual_action_evidence(self):
        with patch.object(agent, "RecoverySession", return_value=self.session), \
                patch.object(agent, "request_message", side_effect=[
                    tool_message("request_start_service"), ModelConnectionError("总结时断开")]), \
                patch("builtins.input", side_effect=["检查并启动", "退出"]):
            agent.chat_loop()
        row = next(row for row in self.store.list_tasks() if row["question"] == "检查并启动")
        task = self.store.get_task(row["id"])
        self.assertEqual(task["status"], "failed")
        evidence = next(event["payload"] for event in task["events"] if event["kind"] == "tool_result")
        self.assertEqual(evidence["tool_result"]["result"]["status"], "verified")
        self.assertTrue(evidence["tool_result"]["result"]["post_start_check"]["health"]["healthy"])
        self.start.assert_called_once()
        self.process.terminate.assert_called_once()
        self.assertIsNone(self.session.process)

    def test_new_session_requires_new_local_approval(self):
        agent.approved_start_result(self.session, self.record)
        self.record.finish("completed", {})
        self.session.close()
        new_record = self.store.create("重新启动")
        self.approval.return_value = False
        agent.approved_start_result(self.session, new_record)
        self.assertEqual(self.approval.call_count, 2)
        self.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()

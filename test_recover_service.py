"""不操作真实服务：用模拟检查结果验证审批与进程清理边界。"""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import recover_service as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.health = self.enterContext(patch.object(recovery, "check_service"))
        self.health.return_value = {"healthy": False, "message": "连接失败"}
        self.port = self.enterContext(patch.object(recovery, "check_port", return_value="refused"))
        self.approval = self.enterContext(patch.object(recovery, "ask_for_approval", return_value=True))
        self.start = self.enterContext(patch.object(recovery, "start_service"))
        self.process = self.start.return_value
        self.process.poll.return_value = None
        self.ready = self.enterContext(patch.object(recovery, "wait_until_healthy", return_value=True))
        self.sleep = self.enterContext(patch.object(recovery.time, "sleep", side_effect=KeyboardInterrupt))

    def test_healthy_does_not_ask_or_start(self):
        self.health.return_value = {"healthy": True, "message": "正常"}
        recovery.main()
        self.approval.assert_not_called()
        self.start.assert_not_called()

    def test_refusal_does_not_start(self):
        self.approval.return_value = False
        recovery.main()
        self.start.assert_not_called()

    def test_uncertain_port_does_not_start(self):
        for state in ["open", "timeout", "unknown"]:
            self.port.return_value = state
            recovery.main()
        self.approval.assert_not_called()
        self.start.assert_not_called()

    def test_open_port_denial_does_not_claim_restart_is_ineffective(self):
        self.port.return_value = "open"
        result = recovery.inspect_start_eligibility()
        self.assertFalse(result["allowed"])
        self.assertEqual(result["port_state"], "open")
        self.assertIn("本项目启动工具不符合执行条件", result["message"])
        self.assertIn("不代表人工重启有效或无效", result["message"])
        self.approval.assert_not_called()
        self.start.assert_not_called()

    def test_rechecks_after_approval(self):
        self.health.side_effect = [
            {"healthy": False, "message": "异常"},
            {"healthy": True, "message": "已恢复"},
        ]
        recovery.main()
        self.start.assert_not_called()

    def test_success_and_ctrl_c_cleanup(self):
        recovery.main()
        self.start.assert_called_once_with()
        self.ready.assert_called_once_with(self.process)
        self.sleep.assert_called_once_with(0.2)
        self.process.terminate.assert_called_once_with()
        self.process.wait.assert_called_once_with()
        self.assertIn("恢复验证通过", self.output.getvalue())

    def test_failed_verification_cleans_up(self):
        self.ready.return_value = False
        self.process.wait.side_effect = None
        recovery.main()
        self.process.terminate.assert_called_once_with()
        self.assertNotIn("恢复验证通过", self.output.getvalue())

    def test_start_error_is_not_success(self):
        self.start.side_effect = OSError("test failure")
        recovery.main()
        self.assertIn("操作失败", self.output.getvalue())
        self.assertNotIn("恢复验证通过", self.output.getvalue())


class VerificationTests(unittest.TestCase):
    def test_polling_cases(self):
        for states, exit_code, expected in [([False, True], None, True), ([False] * 10, None, False), ([True], 1, False)]:
            with self.subTest(states=states, exit_code=exit_code):
                process = Mock()
                process.poll.return_value = exit_code
                with patch.object(recovery, "check_service", side_effect=[{"healthy": s} for s in states]), patch.object(recovery.time, "sleep"):
                    self.assertEqual(recovery.wait_until_healthy(process), expected)


class RecoverySessionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))
        self.health = self.enterContext(patch.object(recovery, "check_service", return_value={"healthy": False, "message": "连接失败"}))
        self.port = self.enterContext(patch.object(recovery, "check_port", return_value="refused"))
        self.approval = self.enterContext(patch.object(recovery, "ask_for_approval", return_value=True))
        self.start = self.enterContext(patch.object(recovery, "start_service"))
        self.process = self.start.return_value
        self.process.poll.return_value = None
        # 模拟进程启动后接口恢复；启动前仍按各测试设置的结果返回。
        self.health.side_effect = lambda: (
            {"healthy": True, "message": "测试服务正常"}
            if self.start.called else self.health.return_value
        )
        self.ready = self.enterContext(patch.object(recovery, "wait_until_healthy", return_value=True))
        self.session = recovery.RecoverySession()

    def test_approved_start_returns_without_stopping_child(self):
        result = self.session.start_with_approval()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(len(result["checks"]), 2)
        self.assertEqual(self.health.call_count, 3)
        self.assertEqual(self.port.call_count, 2)
        self.approval.assert_called_once_with()
        self.start.assert_called_once_with()
        self.ready.assert_called_once_with(self.process)
        self.process.terminate.assert_not_called()
        self.assertIs(self.session.process, self.process)

    def test_success_preserves_actual_post_start_health_not_pre_start_checks(self):
        healthy = {"healthy": True, "message": "本次实际健康响应", "status_code": 200}
        self.health.side_effect = [
            {"healthy": False, "message": "启动前失败"},
            {"healthy": False, "message": "确认后仍失败"}, healthy,
        ]
        result = self.session.start_with_approval()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["post_start_check"]["health"], healthy)
        self.assertTrue(result["post_start_check"]["process_running"])
        self.assertIn("checked_at", result["post_start_check"])
        self.assertTrue(all(not c["health"]["healthy"] for c in result["checks"]))
        self.assertEqual(self.health.call_count, 3)
        self.process.terminate.assert_not_called()

    def test_unhealthy_post_start_check_cannot_be_reported_as_verified(self):
        self.health.side_effect = [
            {"healthy": False, "message": "连接失败"},
            {"healthy": False, "message": "连接失败"},
            {"healthy": False, "status_code": 503, "message": "复查 HTTP503"},
        ]
        result = self.session.start_with_approval()
        self.assertEqual(result["status"], "verification_failed")
        self.assertEqual(result["post_start_check"]["health"]["status_code"], 503)
        self.process.terminate.assert_called_once_with()
        self.process.wait.assert_called_once_with(timeout=3)
        self.assertIsNone(self.session.process)

    def test_child_exits_during_post_start_health_check(self):
        def health_result():
            if self.start.called:
                self.process.poll.return_value = 1
                return {"healthy": True, "message": "接口正常但本次进程已退出"}
            return {"healthy": False, "message": "连接失败"}
        self.health.side_effect = health_result
        result = self.session.start_with_approval()
        self.assertEqual(result["status"], "verification_failed")
        self.assertFalse(result["post_start_check"]["process_running"])
        self.process.terminate.assert_not_called()
        self.process.wait.assert_called_once_with(timeout=3)
        self.assertIsNone(self.session.process)

    def test_post_start_check_error_cleans_only_owned_child(self):
        self.health.side_effect = [
            {"healthy": False, "message": "连接失败"},
            {"healthy": False, "message": "连接失败"}, OSError("模拟复查失败"),
        ]
        with self.assertRaises(OSError):
            self.session.start_with_approval()
        self.process.terminate.assert_called_once_with()
        self.assertIsNone(self.session.process)

    def test_denial_does_not_start(self):
        self.approval.return_value = False
        self.assertEqual(self.session.start_with_approval()["status"], "cancelled")
        self.start.assert_not_called()
        self.assertEqual(self.health.call_count, 1)

    def test_http_503_open_port_is_not_restarted(self):
        self.health.return_value = {"healthy": False, "status_code": 503, "message": "HTTP503"}
        self.port.return_value = "open"
        result = self.session.start_with_approval()
        self.assertEqual(result["status"], "ineligible")
        self.assertEqual(result["checks"][0]["health"]["status_code"], 503)
        self.approval.assert_not_called()
        self.start.assert_not_called()
        self.process.terminate.assert_not_called()

    def test_healthy_and_uncertain_states_never_ask(self):
        for state in ["timeout", "unknown", "unrecognized"]:
            with self.subTest(state=state):
                self.port.return_value = state
                self.assertEqual(self.session.start_with_approval()["status"], "ineligible")
        self.health.return_value = {"healthy": True, "message": "正常"}
        self.assertEqual(self.session.start_with_approval()["status"], "ineligible")
        self.approval.assert_not_called()
        self.start.assert_not_called()

    def test_state_changes_during_approval_block_start(self):
        self.port.side_effect = ["refused", "open"]
        self.assertEqual(self.session.start_with_approval()["status"], "state_changed")
        self.approval.assert_called_once()
        self.start.assert_not_called()

    def test_service_becomes_healthy_during_approval(self):
        self.health.side_effect = [{"healthy": False, "message": "失败"}, {"healthy": True, "message": "正常"}]
        self.assertEqual(self.session.start_with_approval()["status"], "state_changed")
        self.start.assert_not_called()

    def test_failed_verification_cleans_up(self):
        self.ready.return_value = False
        self.assertEqual(self.session.start_with_approval()["status"], "verification_failed")
        self.process.terminate.assert_called_once_with()
        self.process.wait.assert_called_once_with(timeout=3)
        self.assertIsNone(self.session.process)

    def test_interrupted_verification_cleans_up(self):
        self.ready.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.session.start_with_approval()
        self.process.terminate.assert_called_once_with()
        self.assertIsNone(self.session.process)

    def test_start_error_preserves_no_process(self):
        self.start.side_effect = OSError("mock spawn failure")
        with self.assertRaises(OSError):
            self.session.start_with_approval()
        self.assertIsNone(self.session.process)

    def test_approval_eof_or_ctrl_c_never_starts(self):
        for error in [EOFError(), KeyboardInterrupt()]:
            self.approval.side_effect = error
            with self.assertRaises(type(error)):
                self.session.start_with_approval()
        self.start.assert_not_called()

    def test_repeated_start_keeps_owned_child_without_new_approval(self):
        self.session.start_with_approval()
        self.assertEqual(self.session.start_with_approval()["status"], "already_owned")
        self.start.assert_called_once()
        self.approval.assert_called_once()
        self.process.terminate.assert_not_called()

    def test_close_only_owned_child_and_is_idempotent(self):
        self.session.close()
        self.process.terminate.assert_not_called()
        self.session.start_with_approval()
        self.session.close()
        self.session.close()
        self.process.terminate.assert_called_once_with()
        self.process.wait.assert_called_once_with(timeout=3)
        self.assertIsNone(self.session.process)

    def test_exited_child_is_reaped_without_termination(self):
        self.session.process = self.process
        self.process.poll.return_value = 1
        self.session.close()
        self.process.terminate.assert_not_called()
        self.process.wait.assert_called_once_with(timeout=3)

    def test_cleanup_timeout_kills_only_owned_child(self):
        self.session.process = self.process
        self.process.wait.side_effect = [recovery.subprocess.TimeoutExpired("mock", 3), 0]
        self.session.close()
        self.process.kill.assert_called_once_with()
        self.assertEqual(self.process.wait.call_count, 2)
        self.assertIsNone(self.session.process)

    def test_cleanup_error_does_not_claim_child_is_gone(self):
        self.session.process = self.process
        self.process.terminate.side_effect = OSError("mock failure")
        with self.assertRaises(OSError):
            self.session.close()
        self.assertIs(self.session.process, self.process)


class FixedStartTests(unittest.TestCase):
    def test_fixed_start_command_and_hidden_window(self):
        with patch.object(recovery.Path, "open") as log_open, patch.object(recovery.subprocess, "Popen") as popen:
            self.assertIs(recovery.start_service(), popen.return_value)
        log_open.assert_called_once_with("a", encoding="utf-8")
        args, kwargs = popen.call_args
        self.assertEqual(args[0], [recovery.sys.executable, "-u", str(recovery.PROJECT_DIR / "service.py")])
        self.assertEqual(kwargs["cwd"], recovery.PROJECT_DIR)
        self.assertEqual(kwargs["stdin"], recovery.subprocess.DEVNULL)
        self.assertEqual(kwargs["creationflags"], getattr(recovery.subprocess, "CREATE_NO_WINDOW", 0))
        self.assertNotIn("shell", kwargs)


if __name__ == "__main__":
    unittest.main()

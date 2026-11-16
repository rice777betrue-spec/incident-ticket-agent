"""验证统一入口：只调用允许的工具，不把调用完成误认为服务正常。"""

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import tool_runner as runner
from test_service_fault import demo_server


class ToolRunnerTests(unittest.TestCase):
    def test_only_three_read_tools_are_registered(self):
        expected = {"check_service", "diagnose_service", "read_service_log"}
        self.assertEqual(set(runner.TOOLS), expected)
        self.assertEqual(set(runner.TOOL_DESCRIPTIONS), expected)

    def test_calls_only_selected_tool_once_and_preserves_result(self):
        for selected in runner.TOOLS:
            with self.subTest(tool=selected):
                result = {"message": "原始结果", "healthy": False}
                tools = {name: Mock(return_value=result) for name in runner.TOOLS}
                with patch.dict(runner.TOOLS, tools, clear=True):
                    response = runner.call_tool(selected)
                self.assertEqual(response, {"tool": selected, "call_status": "completed", "result": result})
                tools[selected].assert_called_once_with()
                for name, tool in tools.items():
                    if name != selected:
                        tool.assert_not_called()

    def test_unknown_names_commands_and_invalid_types_are_rejected(self):
        tools = {name: Mock() for name in runner.TOOLS}
        with patch.dict(runner.TOOLS, tools, clear=True):
            for name in ("recover_service", "start_service", "check_service; echo test", "", None, [], {}):
                with self.subTest(name=name):
                    self.assertEqual(runner.call_tool(name)["call_status"], "error")
        for tool in tools.values():
            tool.assert_not_called()

    def test_unexpected_exception_is_not_a_diagnosis(self):
        with patch.dict(runner.TOOLS, {"check_service": Mock(side_effect=RuntimeError("测试异常"))}):
            response = runner.call_tool("check_service")
        self.assertEqual(response["call_status"], "error")
        self.assertIn("RuntimeError", response["error"])
        self.assertNotIn("result", response)

    def test_ctrl_c_is_not_swallowed(self):
        with patch.dict(runner.TOOLS, {"check_service": Mock(side_effect=KeyboardInterrupt)}):
            with self.assertRaises(KeyboardInterrupt):
                runner.call_tool("check_service")

    def test_real_fault_returns_completed_call_but_unhealthy_service(self):
        with demo_server(fault=True):
            response = runner.call_tool("check_service")
        self.assertEqual(response["call_status"], "completed")
        self.assertFalse(response["result"]["healthy"])
        self.assertEqual(response["result"]["status_code"], 503)

    def test_cli_without_name_lists_tools_without_calling(self):
        with patch.object(runner, "call_tool") as call, redirect_stdout(io.StringIO()) as output:
            code = runner.main([])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), runner.TOOL_DESCRIPTIONS)
        call.assert_not_called()

    def test_cli_returns_error_exit_code_for_unknown_tool(self):
        with redirect_stdout(io.StringIO()) as output:
            code = runner.main(["recover_service"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())["call_status"], "error")


if __name__ == "__main__":
    unittest.main()

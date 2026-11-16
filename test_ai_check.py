"""只使用模拟模型与工具，不访问真实模型或服务。"""
import copy
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import ai_check as agent
import ai_client


def tool_message(name="check_service", arguments="{}", call_id="call_test"):
    return {"tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}]}


class AICheckTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))
        self.request = self.enterContext(patch.object(agent, "request_message"))
        self.tool = self.enterContext(patch.object(agent, "call_tool"))
        self.tool.return_value = {"tool": "check_service", "call_status": "completed", "result": {"healthy": False, "message": "连接超时"}}
        self.request.side_effect = [tool_message(), {"content": "健康检查失败，根因尚未确认。"}]

    def test_full_flow_evidence_and_bounded_requests(self):
        self.assertEqual(agent.run_check(), self.tool.return_value)
        self.tool.assert_called_once_with("check_service")
        self.assertEqual(self.request.call_count, 2)
        self.assertEqual([t["function"]["name"] for t in self.request.call_args.kwargs["tools"]], ["check_service"])
        second = self.request.call_args
        self.assertEqual(second.kwargs["tool_choice"], "none")
        messages = second.args[0]
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["tool_call_id"], "call_test")
        self.assertEqual(json.loads(messages[-1]["content"]), self.tool.return_value)
        self.assertEqual(messages[-2]["tool_calls"][0]["function"]["name"], "check_service")

    def test_invalid_calls_never_execute(self):
        samples = [{"content": "我已经检查，一切正常"}, tool_message("recover_service"), tool_message("read_service_log"), tool_message("diagnose_service"), tool_message(arguments='{"command":"anything"}'), tool_message(arguments="null"), tool_message(arguments="[]"), tool_message(arguments="broken"), tool_message(arguments={}), tool_message(arguments=" " * 1025)]
        multi = tool_message(); multi["tool_calls"] *= 2; samples.append(multi)
        for field, value in [("id", ""), ("id", 1), ("type", "other"), ("function", None)]:
            message = tool_message(); message["tool_calls"][0][field] = value; samples.append(message)
        for sample in samples:
            with self.subTest(sample=sample):
                self.request.side_effect = [copy.deepcopy(sample)]
                with self.assertRaises(ai_client.ModelConnectionError):
                    agent.run_check()
                self.tool.assert_not_called()

    def test_second_call_failure_does_not_repeat_tool(self):
        self.request.side_effect = [tool_message(), ai_client.ModelConnectionError("超时")]
        with self.assertRaises(ai_client.ModelConnectionError):
            agent.run_check()
        self.tool.assert_called_once()
        self.assertEqual(self.request.call_count, 2)

    def test_repeated_tool_request_stops(self):
        self.request.side_effect = [tool_message(), tool_message()]
        with self.assertRaises(ai_client.ModelConnectionError):
            agent.run_check()
        self.tool.assert_called_once()

    def test_empty_explanation(self):
        self.request.side_effect = [tool_message(), {"content": None}]
        with self.assertRaises(ai_client.ModelConnectionError):
            agent.run_check()

    def test_tool_error_is_returned_as_evidence(self):
        self.tool.return_value = {"call_status": "error", "error": "检查程序异常"}
        agent.run_check()
        self.assertEqual(json.loads(self.request.call_args.args[0][-1]["content"])["call_status"], "error")

    def test_key_prompt_is_temporary(self):
        with patch.dict(os.environ, {"WORKBUDDY_API_KEY": "old-test-key"}), patch.object(agent.getpass, "getpass", return_value="new-test-key"), patch.object(agent, "run_check") as run:
            run.side_effect = lambda: self.assertEqual(os.environ["WORKBUDDY_API_KEY"], "new-test-key")
            self.assertEqual(agent.main(["--ask-key"]), 0)
            self.assertEqual(os.environ["WORKBUDDY_API_KEY"], "old-test-key")

    def test_blank_key_does_not_request(self):
        with patch.object(agent.getpass, "getpass", return_value=" "):
            self.assertEqual(agent.main(["--ask-key"]), 1)
        self.request.assert_not_called()

    def test_interrupt_exits(self):
        self.request.side_effect = KeyboardInterrupt()
        self.assertEqual(agent.main([]), 130)
        self.tool.assert_not_called()

    def test_client_serializes_tools_and_returns_call(self):
        with patch.object(ai_client, "urlopen") as urlopen:
            response = urlopen.return_value.__enter__.return_value
            response.read.return_value = json.dumps({"choices": [{"message": tool_message()}]}).encode()
            self.assertEqual(ai_client.request_message([{"role": "user", "content": "检查"}], tools=agent.TOOLS, tool_choice="auto"), tool_message())
            payload = json.loads(urlopen.call_args.args[0].data)
            self.assertEqual(payload["tools"], agent.TOOLS)
            self.assertEqual(payload["tool_choice"], "auto")


if __name__ == "__main__":
    unittest.main()

"""模拟模型接口测试；运行 unittest 不会消耗真实模型调用额度。"""

import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import ai_client as client


class AIClientTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.open = self.enterContext(patch.object(client, "urlopen"))
        self.response = MagicMock()
        self.open.return_value.__enter__.return_value = self.response
        self.response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "连接成功"}}]
        }, ensure_ascii=False).encode("utf-8")

    def test_exact_endpoint_model_and_prompt_no_tools(self):
        self.assertEqual(client.ask_ai("你好"), "连接成功")
        self.open.assert_called_once()
        request = self.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:7863/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data), {
            "model": "cn:deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
        })
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(self.open.call_args.kwargs["timeout"], 60)
        self.response.read.assert_called_once_with(client.MAX_RESPONSE_BYTES + 1)

    def test_optional_key_is_in_header_only(self):
        with patch.dict(os.environ, {"WORKBUDDY_API_KEY": "test-only-key"}):
            client.ask_ai("你好")
        request = self.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only-key")
        self.assertNotIn(b"test-only-key", request.data)

    def test_http_errors_are_closed_and_not_retried_or_echoed(self):
        for code in (401, 403, 404, 429, 500):
            with self.subTest(code=code):
                body = io.BytesIO(b"private upstream error")
                self.open.side_effect = HTTPError("http://localhost", code, "error", {}, body)
                with self.assertRaises(client.ModelConnectionError) as caught:
                    client.ask_ai("你好")
                self.assertIn(str(code), str(caught.exception))
                self.assertNotIn("private upstream", str(caught.exception))
                self.assertTrue(body.closed)
        self.assertEqual(self.open.call_count, 5)

    def test_400_displays_message_but_not_other_fields_or_key(self):
        key = "test-private-credential"
        body = io.BytesIO(json.dumps({"error": {
            "message": f"model not found; supplied credential: {key}",
            "debug": "do-not-print-this-field",
        }}).encode())
        self.open.side_effect = HTTPError("http://localhost", 400, "error", {}, body)
        with patch.dict(os.environ, {"WORKBUDDY_API_KEY": key}):
            with self.assertRaises(client.ModelConnectionError) as caught:
                client.ask_ai("你好")
        text = str(caught.exception)
        self.assertIn("HTTP 400", text)
        self.assertIn("model not found", text)
        self.assertNotIn(key, text)
        self.assertNotIn("do-not-print", text)
        self.assertTrue(body.closed)
        self.open.assert_called_once()

    def test_error_message_shapes(self):
        for data in ({"message": "unsupported field"}, {"detail": "unsupported field"}, {"error": "unsupported field"}):
            body = io.BytesIO(json.dumps(data).encode())
            error = HTTPError("http://localhost", 400, "error", {}, body)
            self.assertEqual(client.read_error_message(error, ""), "unsupported field")
            self.assertTrue(body.closed)

    def test_error_redaction_before_truncation(self):
        key = "fake-secret-" + "x" * 600
        message = f"prefix {key} Bearer other-secret sk-another-secret api_key=value"
        body = io.BytesIO(json.dumps({"error": {"message": message}}).encode())
        result = client.read_error_message(HTTPError("http://localhost", 400, "error", {}, body), key)
        for secret in ("fake-secret", "other-secret", "sk-another-secret", "value"):
            self.assertNotIn(secret, result)
        self.assertIn("隐藏", result)

    def test_oversized_error_body_is_not_partially_shown(self):
        body = io.BytesIO(b"x" * (client.MAX_ERROR_BYTES + 1))
        result = client.read_error_message(HTTPError("http://localhost", 400, "error", {}, body), "")
        self.assertIn("过大", result)
        self.assertTrue(body.closed)

    def test_error_body_read_failure_preserves_http_status(self):
        class TimeoutBody(io.BytesIO):
            def read(self, *args):
                raise TimeoutError()
        body = TimeoutBody()
        self.open.side_effect = HTTPError("http://localhost", 400, "error", {}, body)
        with self.assertRaises(client.ModelConnectionError) as caught:
            client.ask_ai("你好")
        self.assertIn("HTTP 400", str(caught.exception))
        self.assertTrue(body.closed)

    def test_connection_errors(self):
        for error in (URLError("test"), TimeoutError(), ConnectionRefusedError()):
            self.open.side_effect = error
            with self.assertRaises(client.ModelConnectionError):
                client.ask_ai("你好")

    def test_invalid_json_shape_or_empty_content(self):
        for body in (b"not json", b"\xff", b"null", b"[]", b"{}", b'{"choices":[]}', b'{"choices":[{"message":{"content":null}}]}', b'{"choices":[{"message":{"content":" "}}]}'):
            with self.subTest(body=body):
                self.response.read.return_value = body
                with self.assertRaises(client.ModelConnectionError):
                    client.ask_ai("你好")

    def test_oversized_response(self):
        self.response.read.return_value = b"x" * (client.MAX_RESPONSE_BYTES + 1)
        with self.assertRaisesRegex(client.ModelConnectionError, "过大"):
            client.ask_ai("你好")

    def test_ctrl_c_can_interrupt(self):
        self.open.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            client.ask_ai("你好")


if __name__ == "__main__":
    unittest.main()

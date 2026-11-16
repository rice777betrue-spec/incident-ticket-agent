"""验证错误正文解析：失败时保留状态码，不误报健康。"""

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import check_service as checker


class ErrorBodyTests(unittest.TestCase):
    def check_body(self, body):
        stream = io.BytesIO(body)
        error = HTTPError(checker.HEALTH_URL, 503, "Unavailable", {}, stream)
        with patch.object(checker, "urlopen", side_effect=error):
            result = checker.check_service()
        self.assertTrue(stream.closed)
        self.assertFalse(result["healthy"])
        self.assertEqual(result["status_code"], 503)
        return result

    def test_timeout_note_is_local_and_preserves_reported_numbers(self):
        for stage in ('connect', 'request', 'response_headers', 'response_body'):
            with self.subTest(stage=stage):
                dependency = {'status': 'timeout', 'stage': stage,
                              'elapsed_ms': 256.3, 'timeout_seconds': 0.25}
                body = {'reason': 'downstream_timeout', 'dependency': {
                    **dependency, 'response_note': 'FAKE_SUCCESS_LATENCY'},
                    'response_note': 'FAKE_SUCCESS_LATENCY'}
                result = self.check_body(json.dumps(body).encode())
                self.assertEqual(result['reported_dependency'], dependency)
                self.assertIn(checker.DEPENDENCY_TIMEOUT_NOTE, result['response_note'])
                self.assertNotIn('FAKE_SUCCESS_LATENCY', json.dumps(result))

    def test_timeout_note_not_attached_to_other_or_invalid_evidence(self):
        for dependency in (None, {'status': 'timeout'},
                           {'status': 'ok', 'stage': 'complete',
                            'elapsed_ms': 20, 'timeout_seconds': 0.25}):
            with self.subTest(dependency=dependency):
                result = self.check_body(json.dumps({'dependency': dependency}).encode())
                self.assertNotIn(checker.DEPENDENCY_TIMEOUT_NOTE, result['response_note'])

    def test_reported_reason(self):
        result = self.check_body(b'{"reason":"simulated_health_failure"}')
        self.assertEqual(result["reported_reason"], "simulated_health_failure")
        self.assertIn("未独立验证", result["response_note"])

    def test_invalid_json_and_encoding(self):
        for body in [b"", b"<html>Error</html>", b"\xff"]:
            with self.subTest(body=body):
                result = self.check_body(body)
                self.assertNotIn("reported_reason", result)
                self.assertIn("不是有效 JSON", result["response_note"])

    def test_missing_or_wrong_reason_type(self):
        for data in [{}, [], {"reason": None}, {"reason": 123}, {"reason": " "}]:
            with self.subTest(data=data):
                result = self.check_body(json.dumps(data).encode())
                self.assertNotIn("reported_reason", result)

    def test_oversized_body(self):
        result = self.check_body(b"x" * 4097)
        self.assertIn("超过 4096", result["response_note"])
        self.assertNotIn("reported_reason", result)

    def test_long_reason_is_marked_truncated(self):
        result = self.check_body(json.dumps({"reason": "x" * 501}).encode())
        self.assertEqual(len(result["reported_reason"]), 500)
        self.assertIn("前 500", result["response_note"])

    def test_body_timeout_preserves_http_status(self):
        class SlowBody(io.BytesIO):
            def read(self, *args):
                raise TimeoutError("test timeout")
        stream = SlowBody()
        error = HTTPError(checker.HEALTH_URL, 503, "Unavailable", {}, stream)
        with patch.object(checker, "urlopen", side_effect=error):
            result = checker.check_service()
        self.assertTrue(stream.closed)
        self.assertEqual(result["status_code"], 503)
        self.assertNotIn("reported_reason", result)


class DependencyEvidenceTests(unittest.TestCase):
    check_body = ErrorBodyTests.check_body

    def dependency(self):
        return {"status": "timeout", "stage": "response_headers",
                "timeout_seconds": 0.25, "elapsed_ms": 251.4}

    def parse(self, dependency):
        return self.check_body(json.dumps({"reason": "downstream_timeout", "dependency": dependency}).encode())

    def test_known_fields_preserved_unknown_fields_not_forwarded(self):
        data = self.dependency()
        result = self.parse({**data, "url": "private-url", "token": "private-token"})
        self.assertEqual(result["reported_dependency"], data)
        self.assertNotIn("private-", json.dumps(result))
        self.assertIn("未独立验证", result["response_note"])

    def test_invalid_status_or_stage_rejected_without_losing_http_error(self):
        for key in ("status", "stage"):
            for value in (None, [], {}, 3, "unknown", "private-text"):
                with self.subTest(key=key, value=value):
                    result = self.parse({**self.dependency(), key: value})
                    self.assertNotIn("reported_dependency", result)
                    self.assertEqual(result["reported_reason"], "downstream_timeout")

    def test_invalid_numbers_rejected(self):
        for key in ("timeout_seconds", "elapsed_ms"):
            for value in (True, None, "0.25", -1, float("nan"), float("inf"), 10 ** 100, 60001):
                with self.subTest(key=key, value=value):
                    self.assertNotIn("reported_dependency", self.parse({**self.dependency(), key: value}))
        self.assertNotIn("reported_dependency", self.parse({**self.dependency(), "timeout_seconds": 0}))
        self.assertNotIn("reported_dependency", self.parse({**self.dependency(), "timeout_seconds": 4}))

    def test_non_object_or_incomplete_dependency_rejected(self):
        for data in (None, [], "timeout", {}, {"status": "timeout"}):
            with self.subTest(data=data):
                self.assertNotIn("reported_dependency", self.parse(data))

    def test_metadata_without_reason_still_marked_unverified(self):
        result = self.check_body(json.dumps({"dependency": self.dependency()}).encode())
        self.assertIn("reported_dependency", result)
        self.assertIn("未独立验证", result["response_note"])


if __name__ == "__main__":
    unittest.main()

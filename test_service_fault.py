"""使用真实 HTTP 服务与随机端口验证，不占用用户的 18080 端口。"""

import io
import json
import threading
import unittest
from contextlib import contextmanager, redirect_stdout
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import check_service as checker
import diagnose_service as diagnosis
import recover_service as recovery
from service import DemoHandler, create_demo_server, validate_demo_config


class QuietHandler(DemoHandler):
    def log_message(self, *args):
        pass


@contextmanager
def demo_server(fault=False):
    # 保持旧测试的 True/False 用法，同时让新场景使用与命令行相同的工厂。
    mode = "health" if fault is True else (None if fault is False else fault)
    server = create_demo_server(port=0, fault=mode)
    server.RequestHandlerClass = QuietHandler
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with patch.object(checker, "HEALTH_URL", url + "/health"), patch.object(diagnosis, "PORT", server.server_port):
            yield url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class ServiceFaultTests(unittest.TestCase):
    def test_default_service_stays_healthy(self):
        with demo_server():
            self.assertTrue(checker.check_service()["healthy"])
            self.assertEqual(diagnosis.check_port(), "open")

    def test_fault_returns_503_and_explanation(self):
        with demo_server(fault=True) as url:
            with self.assertRaises(HTTPError) as caught:
                urlopen(url + "/health", timeout=3)
            with caught.exception as response:
                self.assertEqual(response.code, 503)
                body = json.load(response)
            self.assertEqual(body["reason"], "simulated_health_failure")
            health = checker.check_service()
            self.assertFalse(health["healthy"])
            self.assertEqual(health["status_code"], 503)
            self.assertEqual(health["reported_reason"], "simulated_health_failure")
            report = diagnosis.diagnose_service()
            self.assertIn("503", report["evidence"][0])
            self.assertIn("open", report["evidence"][1])
            self.assertIn("端口可以连接", report["assessment"])
            self.assertIn("未独立验证", report["evidence"][2])
            self.assertIn("simulated_health_failure", report["evidence"][2])

    def test_fault_does_not_trigger_duplicate_start(self):
        with demo_server(fault=True), patch.object(recovery, "start_service") as start, patch.object(recovery, "ask_for_approval") as approval, redirect_stdout(io.StringIO()) as output:
            recovery.main()
            start.assert_not_called()
            approval.assert_not_called()
            self.assertIn("不重复启动", output.getvalue())

    def test_unknown_route_remains_404_in_fault_mode(self):
        with demo_server(fault=True) as url:
            with self.assertRaises(HTTPError) as caught:
                urlopen(url + "/missing", timeout=3)
            with caught.exception as response:
                self.assertEqual(response.code, 404)


class ConfigFaultTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        self.assertEqual(validate_demo_config({"runtime_profile": "local-demo"}), [])

    def test_missing_config_field_is_detected(self):
        self.assertEqual(validate_demo_config({}), ["missing required field runtime_profile"])

    def test_invalid_config_value_is_not_exposed(self):
        errors = validate_demo_config({"runtime_profile": "private-demo-value"})
        self.assertEqual(errors, ["invalid field runtime_profile"])
        self.assertNotIn("private-demo-value", " ".join(errors))

    def test_non_object_config_is_rejected(self):
        self.assertTrue(validate_demo_config(None))

    def test_config_fault_is_visible_to_existing_diagnosis(self):
        with demo_server(fault="config"):
            report = diagnosis.diagnose_service()
            self.assertFalse(report["health"]["healthy"])
            self.assertEqual(report["health"]["status_code"], 503)
            self.assertIn("configuration_error", report["health"]["reported_reason"])
            self.assertIn("runtime_profile", report["health"]["reported_reason"])
            self.assertEqual(report["port"]["state"], "open")
            self.assertIn("未独立验证", " ".join(report["evidence"]))

    def test_config_fault_cannot_be_repaired_by_duplicate_start(self):
        with demo_server(fault="config"), patch.object(recovery, "start_service") as start, patch.object(recovery, "ask_for_approval") as approval, redirect_stdout(io.StringIO()):
            recovery.main()
            start.assert_not_called()
            approval.assert_not_called()

    def test_config_mode_keeps_unknown_routes_404(self):
        with demo_server(fault="config") as url:
            with self.assertRaises(HTTPError) as caught:
                urlopen(url + "/missing", timeout=3)
            with caught.exception as response:
                self.assertEqual(response.code, 404)

    def test_config_fault_does_not_pollute_next_server(self):
        with demo_server(fault="config"):
            self.assertFalse(checker.check_service()["healthy"])
        with demo_server():
            self.assertTrue(checker.check_service()["healthy"])

    def test_unknown_fault_is_rejected_before_binding_port(self):
        with self.assertRaises(ValueError):
            create_demo_server(port=0, fault="unknown")


if __name__ == "__main__":
    unittest.main()

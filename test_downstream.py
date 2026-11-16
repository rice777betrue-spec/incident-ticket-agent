"""真实本机依赖/随机端口故障，及有界协议解析；不操作18080或真实模型。"""

import io
import json
import socket
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import urlopen

import check_service as checker
import diagnose_service as diagnosis
import recover_service as recovery
import service
from demo_dependency import LocalDependency, probe_dependency
from test_service_fault import demo_server


class DependencyTests(unittest.TestCase):
    def dependency(self, **kwargs):
        dependency = LocalDependency(**kwargs)
        self.addCleanup(dependency.close)
        return dependency

    def test_fast_dependency_really_answers(self):
        dependency = self.dependency()
        result = probe_dependency(dependency.port, timeout_seconds=1)
        self.assertTrue(dependency.server.request_seen.is_set())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stage"], "complete")
        self.assertGreaterEqual(result["elapsed_ms"], 0)

    def test_delayed_dependency_really_times_out_waiting_for_headers(self):
        dependency = self.dependency(delay_seconds=0.6)
        result = probe_dependency(dependency.port)
        self.assertTrue(dependency.server.request_seen.is_set())
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["stage"], "response_headers")
        self.assertEqual(result["timeout_seconds"], 0.25)
        # 不对调度时间设脆弱上限，只验证实际等待和阶段。
        self.assertGreaterEqual(result["elapsed_ms"], 150)

    def test_close_cleans_owned_listener_and_threads_and_is_idempotent(self):
        dependency = self.dependency(delay_seconds=2)
        self.assertEqual(probe_dependency(dependency.port)["status"], "timeout")
        dependency.close()
        dependency.close()
        self.assertFalse(dependency.thread.is_alive())
        self.assertEqual(dependency.server.fileno(), -1)
        self.assertFalse(any(t.is_alive() for t in dependency.server._threads))

    def test_refused_connection_is_not_response_timeout(self):
        with patch("demo_dependency.http.client.HTTPConnection") as factory:
            connection = factory.return_value
            connection.connect.side_effect = ConnectionRefusedError("private detail")
            result = probe_dependency(12345)
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["stage"], "connect")
            connection.close.assert_called_once()
            self.assertNotIn("private detail", json.dumps(result))

    def test_timeout_elapsed_includes_probe_and_cleanup_not_successful_response(self):
        with patch('demo_dependency.http.client.HTTPConnection') as factory, \
             patch('demo_dependency.time.monotonic', side_effect=[10.0, 10.2563]) as clock:
            connection = factory.return_value
            connection.getresponse.side_effect = TimeoutError()
            def check_cleanup_before_final_timestamp():
                self.assertEqual(clock.call_count, 1)
            connection.close.side_effect = check_cleanup_before_final_timestamp
            result = probe_dependency(12345)
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['stage'], 'response_headers')
        self.assertEqual(result['elapsed_ms'], 256.3)
        connection.close.assert_called_once()
        connection.getresponse.return_value.__enter__.assert_not_called()

    def test_timeout_preserves_stage_and_closes_connection(self):
        for operation, stage in (("connect", "connect"), ("request", "request"),
                                 ("getresponse", "response_headers")):
            with self.subTest(stage=stage), patch("demo_dependency.http.client.HTTPConnection") as factory:
                connection = factory.return_value
                getattr(connection, operation).side_effect = TimeoutError()
                result = probe_dependency(12345)
                self.assertEqual(result["status"], "timeout")
                self.assertEqual(result["stage"], stage)
                connection.close.assert_called_once()

    def test_bad_response_and_body_timeout_are_not_reported_healthy(self):
        cases = [(500, b"", "http_error", "complete"),
                 (200, b"not json", "invalid_response", "response_body"),
                 (200, b'{"status":"ok"}', "invalid_response", "complete"),
                 (200, b"x" * 257, "invalid_response", "complete"),
                 (200, TimeoutError(), "timeout", "response_body")]
        for code, body, status, stage in cases:
            with self.subTest(status=status, body=body), patch("demo_dependency.http.client.HTTPConnection") as factory:
                connection = factory.return_value
                response = connection.getresponse.return_value.__enter__.return_value
                response.status = code
                if isinstance(body, Exception):
                    response.read.side_effect = body
                else:
                    response.read.return_value = body
                result = probe_dependency(12345)
                self.assertEqual((result["status"], result["stage"]), (status, stage))
                connection.getresponse.return_value.__exit__.assert_called_once()
                connection.close.assert_called_once()

    def test_invalid_probe_arguments_rejected_before_connecting(self):
        with patch("demo_dependency.http.client.HTTPConnection") as factory:
            for port in (True, 0, -1, 65536, "12345"):
                with self.subTest(port=port), self.assertRaises(ValueError):
                    probe_dependency(port)
            for timeout in (True, 0, -1, 2, float("nan"), float("inf")):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    probe_dependency(12345, timeout_seconds=timeout)
            factory.assert_not_called()

    def test_invalid_delay_rejected_before_binding(self):
        with patch("demo_dependency.ThreadingHTTPServer") as factory:
            for delay in (True, -1, 3, float("nan"), float("inf"), "slow"):
                with self.subTest(delay=delay), self.assertRaises(ValueError):
                    LocalDependency(delay_seconds=delay)
            factory.assert_not_called()

    def test_thread_start_failure_closes_listener(self):
        fake_server = Mock()
        with patch("demo_dependency.ThreadingHTTPServer", return_value=fake_server), \
             patch("demo_dependency.threading.Thread.start", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                LocalDependency()
        fake_server.server_close.assert_called_once()


class DownstreamServiceTests(unittest.TestCase):
    def test_real_timeout_and_open_port_reach_diagnosis(self):
        with demo_server(fault="downstream"):
            report = diagnosis.diagnose_service()
        self.assertEqual(report["port"]["state"], "open")
        health = report["health"]
        self.assertFalse(health["healthy"])
        self.assertEqual(health["status_code"], 503)
        self.assertEqual(health["reported_reason"], "downstream_timeout")
        dependency = health["reported_dependency"]
        self.assertEqual((dependency["status"], dependency["stage"]), ("timeout", "response_headers"))
        self.assertIn("未独立验证", health["response_note"])
        self.assertIn(checker.DEPENDENCY_TIMEOUT_NOTE, health["response_note"])
        self.assertIn("elapsed_ms", report["evidence"][-1])

    def test_fast_downstream_makes_main_service_healthy_not_hardcoded_failure(self):
        with patch.object(service, "DOWNSTREAM_DELAY_SECONDS", 0), demo_server(fault="downstream"):
            self.assertTrue(checker.check_service()["healthy"])

    def test_downstream_timeout_does_not_ask_approval_or_start_duplicate_service(self):
        with demo_server(fault="downstream"), patch.object(recovery, "start_service") as start, \
             patch.object(recovery, "ask_for_approval") as approval, redirect_stdout(io.StringIO()):
            recovery.main()
        start.assert_not_called()
        approval.assert_not_called()

    def test_unknown_route_stays_404(self):
        with demo_server(fault="downstream") as url:
            with self.assertRaises(HTTPError) as caught:
                urlopen(url + "/missing", timeout=3)
            with caught.exception as response:
                self.assertEqual(response.code, 404)

    def test_main_close_also_closes_owned_dependency(self):
        server = service.create_demo_server(port=0, fault="downstream")
        self.addCleanup(server.server_close)
        dependency = server.dependency
        server.server_close()
        server.server_close()
        self.assertEqual(server.fileno(), -1)
        self.assertEqual(dependency.server.fileno(), -1)
        self.assertFalse(dependency.thread.is_alive())

    def test_other_modes_do_not_create_dependency(self):
        with patch.object(service, "LocalDependency") as factory:
            for mode in (None, "health", "config"):
                server = service.create_demo_server(port=0, fault=mode)
                try:
                    self.assertIsNone(server.dependency)
                finally:
                    server.server_close()
            factory.assert_not_called()

    def test_main_bind_failure_does_not_create_or_orphan_dependency(self):
        # 端口由本用例占用，并不接触其他进程。
        with socket.socket() as owned:
            owned.bind(("127.0.0.1", 0))
            owned.listen()
            with patch.object(service, "LocalDependency") as factory:
                with self.assertRaises(OSError):
                    service.create_demo_server(port=owned.getsockname()[1], fault="downstream")
                factory.assert_not_called()

    def test_dependency_creation_failure_closes_main_listener(self):
        original_close = service.DemoHTTPServer.server_close
        closed = []
        def tracked_close(server):
            original_close(server)
            closed.append(server.fileno())
        with patch.object(service, "LocalDependency", side_effect=RuntimeError("test")), \
             patch.object(service.DemoHTTPServer, "server_close", tracked_close):
            with self.assertRaises(RuntimeError):
                service.create_demo_server(port=0, fault="downstream")
        self.assertEqual(closed, [-1])

    def test_fault_does_not_pollute_next_server(self):
        with demo_server(fault="downstream"):
            self.assertFalse(checker.check_service()["healthy"])
        with demo_server():
            self.assertTrue(checker.check_service()["healthy"])


if __name__ == "__main__":
    unittest.main()

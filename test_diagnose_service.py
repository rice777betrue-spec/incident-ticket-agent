"""端口检查回归测试：不把超时等同于拒绝连接。"""

import unittest
from unittest.mock import patch

import diagnose_service as diagnosis


class PortTests(unittest.TestCase):
    def test_port_states_and_timeout_budget(self):
        for error, state in [(None, "open"), (ConnectionRefusedError(), "refused"), (TimeoutError(), "timeout"), (OSError(), "unknown")]:
            with self.subTest(state=state):
                with patch.object(diagnosis.socket, "create_connection", side_effect=error) as connect:
                    self.assertEqual(diagnosis.check_port(), state)
                    connect.assert_called_once_with((diagnosis.HOST, diagnosis.PORT), timeout=5)


class DiagnosisTests(unittest.TestCase):
    def test_healthy_preserves_structured_health_without_checking_port(self):
        health = {'healthy': True, 'message': '测试服务正常'}
        with patch.object(diagnosis, 'check_service', return_value=health) as check, \
                patch.object(diagnosis, 'check_port') as port:
            result = diagnosis.diagnose_service()
        check.assert_called_once_with()
        port.assert_not_called()
        self.assertEqual(result['health'], health)
        self.assertIsNone(result['port'])
        self.assertEqual(result['evidence'], [health['message']])
        self.assertEqual(result['assessment'], '本次健康检查通过，无需修复')

    def test_unhealthy_preserves_structured_health_and_each_port_state(self):
        health = {'healthy': False, 'message': '接口返回 HTTP 错误：503',
                  'status_code': 503, 'reported_reason': 'simulated_health_failure'}
        for state, assessment in [('open', '端口可以连接'), ('refused', '端口连接被拒绝'),
                                  ('timeout', '端口连接超时'), ('unknown', '端口检查发生连接错误')]:
            with self.subTest(state=state):
                with patch.object(diagnosis, 'check_service', return_value=health) as check, \
                        patch.object(diagnosis, 'check_port', return_value=state) as port:
                    result = diagnosis.diagnose_service()
                check.assert_called_once_with()
                port.assert_called_once_with()
                self.assertEqual(result['health'], health)
                self.assertEqual(result['port'], {'number': diagnosis.PORT, 'state': state})
                self.assertEqual(result['evidence'], [health['message'],
                    f'端口 {diagnosis.PORT} 检查结果：{state}',
                    '服务报告的原因（未独立验证）：simulated_health_failure'])
                self.assertIn(assessment, result['assessment'])


if __name__ == "__main__":
    unittest.main()

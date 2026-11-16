"""任务读取边界测试；不连接模型或启动服务。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import task_query_tools as tools
from task_store import TaskStore
import ai_check as core


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.root / 'tasks.sqlite3'
        with TaskStore(self.path) as store:
            task = store.create('旧排障')
            self.old = task.id
            task.event('tool_requested', {'tool': 'diagnose_service'})
            task.finish('completed', {'report': {'analysis': '历史结果'}})
            for _ in range(24):
                record = store.create('查询记录')
                record.event('tool_requested', {'tool': 'list_tasks'})
                record.finish('completed', {'report': {'analysis': '查询完成'}})

    def run_tool(self, name, args, path=None):
        return tools.run_archive_tool(name, args, path or self.path, 'task_'+'f'*32)

    def test_filter_before_limit_finds_old_operational_task(self):
        result = self.run_tool('list_tasks', {'category': 'troubleshooting', 'limit': 1})['result']
        self.assertEqual([t['id'] for t in result['tasks']], [self.old])
        self.assertFalse(result['has_more'])

    def test_limit_and_readonly_bytes(self):
        before = hashlib.sha256(self.path.read_bytes()).digest()
        result = self.run_tool('list_tasks', {'limit': 3})['result']
        self.assertEqual(len(result['tasks']), 3)
        self.assertTrue(result['has_more'])
        self.run_tool('get_task', {'task_id': self.old})
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).digest(), before)

    def test_missing_database_not_created(self):
        missing = self.root / 'absent.sqlite3'
        self.assertEqual(self.run_tool('list_tasks', {}, missing)['call_status'], 'error')
        self.assertFalse(missing.exists())

    def test_bad_database_returns_safe_error(self):
        bad = self.root / 'broken.sqlite3'
        bad.write_text('secret raw file', encoding='utf-8')
        result = self.run_tool('list_tasks', {}, bad)
        self.assertEqual(result['call_status'], 'error')
        self.assertNotIn('secret', json.dumps(result))

    def test_bounded_events_evidence_and_report_text(self):
        with TaskStore(self.path) as store:
            task = store.create('long')
            for i in range(40):
                task.event('tool_result', {'tool_result': {'tool': 'check_service', 'result': {'index': i}}})
            task.finish('completed', {'report': {'analysis': 'x'*5000}})
        result = self.run_tool('get_task', {'task_id': task.id})
        detail = result['result']
        self.assertEqual(len(detail['events']), 30)
        self.assertTrue(detail['events_truncated'])
        self.assertEqual(len(detail['operational_evidence']), 10)
        self.assertTrue(detail['evidence_truncated'])
        prepared = core.prepare_evidence(result)
        self.assertIn('[后文省略]', prepared['tool_result']['result']['analysis'])

    def test_pending_task_has_no_fabricated_report(self):
        with TaskStore(self.path) as store:
            task = store.create('等待审批')
            task.event('graph_paused', {'note': 'not approved'})
        result = self.run_tool('get_task', {'task_id': task.id})['result']
        self.assertIsNone(result['analysis'])
        self.assertEqual(result['task']['status'], 'running')
        self.assertEqual(result['task']['phase'], 'graph_paused')


if __name__ == '__main__':
    unittest.main()

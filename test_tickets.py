"""工单业务闭环测试。使用临时库与真实LangGraph调度，不调用真实模型或服务。"""
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from langgraph.checkpoint.memory import InMemorySaver

import studio_agent as studio
import ticket_tools
from ticket_store import DEFAULT_TODOS, TicketStore, read_ticket, observation, operational_events
from task_store import TaskStore


def call(name, **args):
    return {'tool_calls': [{'id': 'c_' + name, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}]}


def outcome(healthy):
    return {'tool': 'check_service', 'call_status': 'completed', 'result': {'healthy': healthy}}


class TicketTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.root / 'tickets.sqlite3'
        self.tasks = self.enterContext(TaskStore(self.root / 'tasks.sqlite3'))
        self.store = self.enterContext(TicketStore(self.path, sanitizer=studio.core.redact_text))

    def task(self, result=None):
        record = self.tasks.create('测试故障')
        if result:
            record.event('tool_result', {'tool_result': result})
        return self.tasks.get_task(record.id)

    def create(self):
        task = self.task()
        return self.store.write('create_ticket', {'title': '接口异常', 'description': '用户反馈不可达'}, task['id'], task['events'][-1]['sequence'])

    def ticket(self, id):
        return read_ticket(self.path, id)['ticket']

    def test_create_complete_pending_ticket_without_claiming_repair(self):
        id = self.create()
        ticket = self.ticket(id)
        self.assertEqual(id, 'INC-000001')
        self.assertEqual(ticket['status'], '待处理')
        self.assertEqual(ticket['owner'], '待指定')
        self.assertIn('待确认', ticket['root_cause'])
        self.assertEqual(ticket['service'], 'demo-service')
        self.assertIn('尚无', ticket['verification']['note'])

    def test_default_todo_follows_health_and_recurrence(self):
        id = self.create()
        self.assertEqual(self.ticket(id)['todo'], DEFAULT_TODOS['待处理'])
        for healthy, status in [(True, '已恢复'), (None, '待验证'), (False, '待处理')]:
            self.store.sync(id, self.task(outcome(healthy)), '保留AI原文')
            self.assertEqual(self.ticket(id)['todo'], DEFAULT_TODOS[status])

    def test_change_reported_updates_default_todo_until_verified(self):
        id = self.create()
        self.store.sync(id, self.task(outcome(True)), '正常')
        task = self.task()
        self.store.write('update_ticket', {'ticket_id': id, 'note': '用户报告再次处理', 'change_reported': True},
                         task['id'], task['events'][-1]['sequence'])
        self.assertEqual(self.ticket(id)['todo'], DEFAULT_TODOS['待验证'])
        self.store.sync(id, self.task(outcome(True)), '复查正常')
        self.assertEqual(self.ticket(id)['todo'], DEFAULT_TODOS['已恢复'])

    def test_explicit_todo_preserved_even_when_equal_to_default(self):
        for todo in ['联系小王检查依赖配置', DEFAULT_TODOS['待处理']]:
            id = self.create()
            task = self.task()
            self.store.write('update_ticket', {'ticket_id': id, 'note': '指定待办', 'todo': todo},
                             task['id'], task['events'][-1]['sequence'])
            self.store.sync(id, self.task(outcome(True)), '已恢复')
            with TicketStore(self.path):
                pass
            self.assertEqual(self.ticket(id)['todo'], todo)

    def test_old_round_does_not_roll_back_default_todo(self):
        id = self.create()
        old, new = self.task(outcome(False)), self.task(outcome(True))
        self.store.sync(id, new, '新检查正常')
        self.store.sync(id, old, '旧检查失败')
        self.assertEqual(self.ticket(id)['todo'], DEFAULT_TODOS['已恢复'])

    def test_legacy_default_repaired_on_write_open_without_new_evidence(self):
        id = self.create()
        task = self.task(outcome(True))
        self.store.sync(id, task, '原AI回答')
        with self.store.db:
            self.store.db.execute('UPDATE tickets SET todo=? WHERE id=?', (DEFAULT_TODOS['待处理'], id))
        before = self.ticket(id)
        # 只读查询不能顺便修库。
        self.assertEqual(before['todo'], DEFAULT_TODOS['待处理'])
        with TicketStore(self.path):
            pass
        after = self.ticket(id)
        self.assertEqual(after['todo'], DEFAULT_TODOS['已恢复'])
        for key in ['status', 'verification', 'history', 'latest_analysis']:
            self.assertEqual(after[key], before[key])
        with TicketStore(self.path):
            pass
        self.assertEqual(self.ticket(id)['updated_at'], after['updated_at'])

    def test_idempotent_create_and_conflicting_replay(self):
        task = self.task()
        args = {'title': '异常', 'description': '描述'}
        a = self.store.write('create_ticket', args, task['id'], 1)
        self.assertEqual(a, self.store.write('create_ticket', args, task['id'], 2))
        with self.assertRaises(ValueError):
            self.store.write('create_ticket', {**args, 'title': '另一个'}, task['id'], 3)
        self.assertEqual(len(read_ticket(self.path)['tickets']), 1)

    def test_sync_idempotent_health_and_reopen(self):
        id = self.create()
        task = self.task(outcome(True))
        self.store.sync(id, task, '仅验证当时健康')
        self.store.sync(id, task, '重放不能替换第一次报告')
        self.assertEqual(self.ticket(id)['status'], '已恢复')
        self.assertEqual(self.ticket(id)['latest_analysis'], '仅验证当时健康')
        self.assertEqual(len(self.ticket(id)['history']), 2)
        bad = self.task(outcome(False))
        self.store.sync(id, bad, '当前失败，重新处理')
        self.assertEqual(self.ticket(id)['status'], '待处理')
        self.assertEqual(self.ticket(id)['verification']['task_id'], bad['id'])

    def test_late_old_round_cannot_overwrite_new_health(self):
        id = self.create()
        old = self.task(outcome(False))
        new = self.task(outcome(True))
        self.store.sync(id, new, '新证据通过')
        self.store.sync(id, old, '旧轮延迟返回')
        self.assertEqual(self.ticket(id)['status'], '已恢复')
        self.assertEqual(self.ticket(id)['verification']['task_id'], new['id'])
        self.assertEqual(self.ticket(id)['latest_analysis'], '新证据通过')

    def test_history_logs_and_ai_claim_cannot_resolve(self):
        for name in ('get_task', 'get_ticket', 'read_service_log'):
            with self.subTest(name=name):
                id = self.create()
                task = self.task({'tool': name, 'call_status': 'completed', 'result': {'healthy': True, 'text': '已经修好了'}})
                self.store.sync(id, task, 'AI声称全部正常')
                self.assertEqual(self.ticket(id)['status'], '待处理')

    def test_unknown_health_is_not_success(self):
        id = self.create()
        good = self.task(outcome(True))
        self.store.sync(id, good, '健康')
        bad = self.task({'tool': 'check_service', 'call_status': 'error', 'result': {'healthy': True}})
        self.store.sync(id, bad, '工具异常')
        self.assertEqual(self.ticket(id)['status'], '待验证')

    def test_user_reported_change_is_not_verification(self):
        id = self.create()
        good = self.task(outcome(True)); self.store.sync(id, good, '正常')
        task = self.task()
        self.store.write('update_ticket', {'ticket_id': id, 'note': '用户说已修改配置', 'change_reported': True,
                                          'owner': '小王', 'todo': '重新检查健康接口'}, task['id'], task['events'][-1]['sequence'])
        self.store.sync(id, task, '用户自述，未验证')
        ticket = self.ticket(id)
        self.assertEqual(ticket['status'], '待验证')
        self.assertEqual(ticket['owner'], '小王')
        self.assertEqual(ticket['todo'], '重新检查健康接口')
        self.store.sync(id, good, '旧轮重放不影响待验证')
        self.assertEqual(self.ticket(id)['status'], '待验证')

    def test_fresh_check_after_reported_change_can_recover(self):
        id = self.create()
        record = self.tasks.create('配置改好了检查一下')
        seq = self.tasks.get_task(record.id)['events'][-1]['sequence']
        self.store.write('update_ticket', {'ticket_id': id, 'note': '用户修改', 'change_reported': True}, record.id, seq)
        record.event('tool_result', {'tool_result': outcome(True)})
        self.store.sync(id, self.tasks.get_task(record.id), '已验证健康')
        self.assertEqual(self.ticket(id)['status'], '已恢复')

    def test_diagnose_and_startup_verification(self):
        for result, expected in [
            ({'tool': 'diagnose_service', 'call_status': 'completed', 'result': {'health': {'healthy': True}}}, '已恢复'),
            ({'tool': 'request_start_service', 'call_status': 'completed', 'result': {'status': 'verified', 'post_start_check': {'health': {'healthy': True}, 'process_running': True}}}, '已恢复'),
            ({'tool': 'request_start_service', 'call_status': 'completed', 'result': {'status': 'verification_failed', 'post_start_check': {'health': {'healthy': True}, 'process_running': False}}}, '待处理'),
            ({'tool': 'request_start_service', 'call_status': 'completed', 'result': {'status': 'already_owned', 'healthy': True}}, '待处理'),
        ]:
            with self.subTest(result=result):
                id = self.create(); task = self.task(result)
                self.store.sync(id, task, '分析')
                self.assertEqual(self.ticket(id)['status'], expected)

    def test_new_failed_precheck_reopens_but_positive_precheck_does_not_resolve(self):
        id = self.create()
        good = self.task(outcome(True)); self.store.sync(id, good, '正常')
        task = self.task({'tool': 'request_start_service', 'call_status': 'completed',
                          'result': {'status': 'ineligible', 'checks': [{'health': {'healthy': False}}]}})
        self.store.sync(id, task, '端口开放不能重复启动')
        self.assertEqual(self.ticket(id)['status'], '待处理')
        task = self.task({'tool': 'request_start_service', 'call_status': 'completed',
                          'result': {'status': 'ineligible', 'checks': [{'health': {'healthy': True}}]}})
        self.store.sync(id, task, '没做独立复查')
        self.assertEqual(self.ticket(id)['status'], '待处理')

    def test_one_round_cannot_link_to_two_tickets(self):
        a, b = self.create(), self.create()
        task = self.task(outcome(True))
        self.store.sync(a, task, '验证')
        with self.assertRaises(ValueError):
            self.store.sync(b, task, '不准复制归属')

    def test_read_only_no_file_creation_no_mutation(self):
        absent = self.root / 'absent.sqlite3'
        self.assertEqual(read_ticket(absent)['tickets'], [])
        self.assertFalse(absent.exists())
        id = self.create(); before = self.path.read_bytes()
        read_ticket(self.path, id); read_ticket(self.path)
        self.assertEqual(before, self.path.read_bytes())

    def test_schema_rejects_status_paths_approval_and_invalid_ids(self):
        invalid = [('update_ticket', {'ticket_id': 'INC-000001', 'note': 'hello', 'status': '已恢复'}),
                   ('create_ticket', {'title': 'x', 'description': 'x', 'path': 'other.db'}),
                   ('continue_ticket', {'ticket_id': '../../secret'}),
                   ('list_tickets', {'approve': True}),
                   ('update_ticket', {'ticket_id': 'INC-000001', 'note': 'x', 'change_reported': 'true'})]
        for name, args in invalid:
            with self.subTest(name=name, args=args), self.assertRaises(studio.core.ModelConnectionError):
                ticket_tools.VALIDATORS[name](args)

    def test_missing_ticket_does_not_create_phantom(self):
        task = self.task()
        with self.assertRaises(ValueError):
            self.store.write('continue_ticket', {'ticket_id': 'INC-999999'}, task['id'], 1)
        self.assertEqual(read_ticket(self.path)['tickets'], [])

    def test_redaction_before_persistence(self):
        task = self.task()
        fake = 'sk-' + 'x' * 40
        id = self.store.write('create_ticket', {'title': '测试', 'description': fake}, task['id'], 1)
        self.store.sync(id, task, fake)
        self.assertNotIn(fake, json.dumps(self.ticket(id)))

    def test_bounded_history_and_persistence_after_reopen(self):
        id = self.create()
        for i in range(22):
            self.store.sync(id, self.task(outcome(False)), f'第{i}轮')
        with TicketStore(self.path) as reopened:
            self.assertEqual(reopened.db.execute('SELECT count(*) FROM ticket_events').fetchone()[0], 23)
        ticket = self.ticket(id)
        self.assertTrue(ticket['history_has_more'])
        self.assertEqual(len(ticket['history']), 20)


class TicketGraphTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.model = self.enterContext(patch.object(studio.core, 'request_message'))
        self.tool = self.enterContext(patch.object(studio.core, 'call_tool', return_value=outcome(False)))
        self.spawn = self.enterContext(patch.object(studio.recovery, 'start_service', side_effect=AssertionError('不准启动')))
        self.engine = studio.StudioAgent(self.root)
        self.addCleanup(self.engine.session.close)
        self.graph = self.engine.build(InMemorySaver())
        self.config = {'configurable': {'thread_id': 'ticket-test'}, 'recursion_limit': 25}

    def run_round(self, question, responses):
        self.model.side_effect = responses
        return self.graph.invoke({'question': question}, self.config)

    def create(self):
        return self.run_round('服务访问不了，排查并建单', [call('create_ticket', title='服务不可达', description='用户反馈访问失败'), call('check_service'), {'content': 'AI的真实分析，未确认根因'}])

    def test_same_thread_two_tasks_one_ticket_with_recovery(self):
        first = self.create()
        id = first['active_ticket_id']
        self.assertEqual(first['ticket']['status'], '待处理')
        self.assertEqual(first['phase'], 'done')
        self.assertEqual(first['answer'], 'AI的真实分析，未确认根因')
        self.tool.return_value = outcome(True)
        second = self.run_round('配置改好了再检查', [call('check_service'), {'content': '本次健康验证通过，根因仍待确认'}])
        self.assertEqual(second['active_ticket_id'], id)
        self.assertNotEqual(first['task_id'], second['task_id'])
        self.assertEqual(second['ticket']['status'], '已恢复')
        self.assertEqual(len(read_ticket(self.engine.ticket_path)['tickets']), 1)
        rounds = [e for e in second['ticket']['history'] if e['kind'] == 'round']
        self.assertEqual([e['task_id'] for e in rounds], [first['task_id'], second['task_id']])
        with TaskStore(self.engine.store_path) as store:
            self.assertEqual(store.get_task(second['task_id'])['events'][-1]['payload']['report']['ticket_id'], id)
        self.spawn.assert_not_called()

    def test_cross_thread_requires_explicit_binding(self):
        first = self.create(); id = first['active_ticket_id']
        self.config['configurable']['thread_id'] = 'another-thread'
        new = self.run_round('检查服务', [call('check_service'), {'content': '异常'}])
        self.assertEqual(new['active_ticket_id'], '')
        self.assertEqual(new['ticket'], {})
        self.graph = studio.StudioAgent(self.root).build(InMemorySaver())
        new = self.run_round('继续处理 ' + id, [call('continue_ticket', ticket_id=id), call('check_service'), {'content': '仍异常'}])
        self.assertEqual(new['active_ticket_id'], id)
        continuation = next(m for m in new['turn'] if m.get('tool_call_id') == 'c_continue_ticket')
        self.assertIn('AI的真实分析', continuation['content'])
        self.assertEqual(len([e for e in new['ticket']['history'] if e['kind'] == 'round']), 2)

    def test_plain_chat_and_ticket_queries_do_not_modify_ticket(self):
        first = self.create()
        before = self.engine.ticket_path.read_bytes()
        chat = self.run_round('你好', [{'content': '你好'}])
        self.assertEqual(chat['ticket'], {})
        self.assertNotIn('ticket_id', chat['report'])
        self.run_round('查看工单', [call('get_ticket', ticket_id=first['active_ticket_id']), {'content': '这是历史工单'}])
        self.assertEqual(before, self.engine.ticket_path.read_bytes())

    def test_get_other_ticket_does_not_switch_binding(self):
        first = self.create()
        other = self.create()
        response = self.run_round('只查看第一张', [call('get_ticket', ticket_id=first['active_ticket_id']), {'content': '只读'}])
        self.assertEqual(response['active_ticket_id'], other['active_ticket_id'])

    def test_write_node_visible_and_mixed_batch_rejected_before_any_action(self):
        self.assertIn('write_ticket', self.graph.get_graph().nodes)
        batch = {'tool_calls': call('create_ticket', title='标题', description='描述')['tool_calls'] + call('check_service')['tool_calls']}
        with self.assertRaises(studio.GraphRunError):
            self.run_round('建单', [batch])
        self.tool.assert_not_called()
        self.assertFalse(self.engine.ticket_path.exists())

    def test_two_writes_same_round_rejected(self):
        with self.assertRaises(studio.GraphRunError):
            self.run_round('建单', [call('create_ticket', title='标题', description='描述'), call('continue_ticket', ticket_id='INC-000001')])
        self.assertEqual(len(read_ticket(self.engine.ticket_path)['tickets']), 1)
        self.tool.assert_not_called()

    def test_failed_model_keeps_created_ticket_pending_and_task_evidence(self):
        with self.assertRaises(studio.GraphRunError):
            self.run_round('建单排查', [call('create_ticket', title='标题', description='描述'), call('check_service'), RuntimeError('模型失败')])
        self.assertEqual(read_ticket(self.engine.ticket_path, 'INC-000001')['ticket']['status'], '待处理')
        with TaskStore(self.engine.store_path) as store:
            task = store.list_tasks()[0]
            self.assertEqual(task['status'], 'failed')
            self.assertTrue(operational_events(store.get_task(task['id'])))

    def test_sync_failure_does_not_report_task_completed(self):
        with patch.object(TicketStore, 'sync', side_effect=sqlite3.OperationalError('disk full')):
            with self.assertRaises(studio.GraphRunError):
                self.create()
        with TaskStore(self.engine.store_path) as store:
            self.assertEqual(store.list_tasks()[0]['status'], 'failed')

    def test_write_node_replay_keeps_single_ticket(self):
        self.model.side_effect = [call('create_ticket', title='故障', description='描述')]
        state = self.engine.prepare({'question': '建单'})
        state.update(self.engine._analyze(state))
        first = self.engine._write_ticket(state)
        replay = self.engine._write_ticket(state)
        self.assertEqual(first['active_ticket_id'], replay['active_ticket_id'])
        self.assertEqual(len(read_ticket(self.engine.ticket_path)['tickets']), 1)

    def test_ticket_write_cannot_be_in_read_node(self):
        state = self.engine.prepare({'question': '建单'})
        state['pending_calls'] = call('create_ticket', title='故障', description='描述')['tool_calls']
        with self.assertRaises(studio.GraphRunError):
            self.engine._read_tools(state)
        self.assertFalse(self.engine.ticket_path.exists())

    def test_budget_three_still_allows_create_diagnose_logs(self):
        self.tool.side_effect = [{'tool': 'diagnose_service', 'call_status': 'completed', 'result': {'health': {'healthy': False}}},
                                 {'tool': 'read_service_log', 'call_status': 'completed', 'result': {'lines': ['503']}}]
        result = self.run_round('建单排查', [call('create_ticket', title='标题', description='描述'), call('diagnose_service'), call('read_service_log'), {'content': '待排查'}])
        self.assertEqual(result['used_names'], ['create_ticket', 'diagnose_service', 'read_service_log'])
        self.assertEqual(self.model.call_args.kwargs['tool_choice'], 'none')
        self.assertEqual(result['ticket']['status'], '待处理')

    def test_user_change_note_then_check_updates_same_ticket(self):
        id = self.create()['active_ticket_id']
        self.tool.return_value = outcome(True)
        result = self.run_round('我改了配置，由小王继续', [call('update_ticket', ticket_id=id, note='用户说已改配置', owner='小王', change_reported=True), call('check_service'), {'content': '本轮验证健康'}])
        self.assertEqual(result['ticket']['owner'], '小王')
        self.assertEqual(result['ticket']['status'], '已恢复')
        self.assertEqual(result['ticket']['todo'], DEFAULT_TODOS['已恢复'])
        self.spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()

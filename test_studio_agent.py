"""正式 Studio 入口测试：使用真实图调度/临时档案，模型和进程全部模拟。"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import studio_agent as studio
from task_store import TaskStore


def call(name='check_service', id='c1'):
    return {'tool_calls': [{'id': id, 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}]}


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.model = self.enterContext(patch.object(studio.core, 'request_message'))
        self.tool = self.enterContext(patch.object(studio.core, 'call_tool'))
        self.tool.return_value = {'tool': 'check_service', 'call_status': 'completed',
                                  'result': {'healthy': False, 'message': 'HTTP 503'}}
        self.inspect = self.enterContext(patch.object(studio.recovery, 'inspect_start_eligibility'))
        self.inspect.return_value = {'allowed': True, 'health': {'healthy': False, 'message': '不可达'},
                                     'port_state': 'refused', 'message': '可以申请启动'}
        self.terminal = self.enterContext(patch.object(studio.recovery, 'ask_for_approval',
                                                       side_effect=AssertionError('不准读取终端')))
        self.spawn = self.enterContext(patch.object(studio.recovery, 'start_service'))
        self.process = Mock()
        self.process.poll.return_value = None
        self.spawn.return_value = self.process
        self.enterContext(patch.object(studio.recovery, 'wait_until_healthy', return_value=True))
        self.enterContext(patch.object(studio.recovery, 'check_service', return_value={'healthy': True}))
        self.engine = studio.StudioAgent(self.root)
        self.addCleanup(self.engine.session.close)
        self.graph = self.engine.build(InMemorySaver())
        self.config = {'configurable': {'thread_id': 'studio-test'}, 'recursion_limit': 25}

    def run_question(self, question='检查服务'):
        return self.graph.invoke({'question': question}, self.config)

    def events(self, task_id):
        with TaskStore(self.engine.store_path) as store:
            return store.get_task(task_id)

    def pause(self):
        self.model.return_value = call(studio.core.START_TOOL_NAME)
        result = self.run_question('请启动固定测试服务')
        self.assertTrue(result['__interrupt__'])
        self.spawn.assert_not_called()
        return result

    def resume(self, decision):
        self.model.return_value = {'content': '根据实际结果总结'}
        return self.graph.invoke(Command(resume=decision), self.config)

    def test_direct_answer_and_report_persist(self):
        self.model.return_value = {'content': '完整回答'}
        result = self.run_question('解释一下')
        self.assertEqual(result['answer'], '完整回答')
        task = self.events(result['task_id'])
        self.assertEqual(task['status'], 'completed')
        self.assertEqual(task['events'][-1]['payload']['report'], result['report'])
        self.tool.assert_not_called()

    def test_recurrence_guidance_reaches_model_without_rewriting_analysis(self):
        # 模拟只验证规则送达、旧推断不删除和新分析原文保存；不证明真实模型遵从。
        old_claim = '同一配置下有时200、有时503，说明下游偶发变慢。'
        self.model.return_value = {'content': old_claim}
        self.run_question('解释之前的记录')
        self.model.reset_mock()
        answer = ('历史检查曾成功，本轮检查失败，说明故障再次出现。'
                  '尚未确认两次检查之间的配置、进程或运行模式是否变化，'
                  '因此不能判断为同一配置下的间歇性故障。')
        self.model.side_effect = [call('diagnose_service'), call('read_service_log', 'c2'),
                                  {'content': answer}]
        self.tool.side_effect = [
            {'tool': 'diagnose_service', 'call_status': 'completed', 'result': {
                'health': {'healthy': False, 'status_code': 503,
                           'reported_reason': 'downstream_timeout'},
                'port': {'number': 18080, 'state': 'open'}}},
            {'tool': 'read_service_log', 'call_status': 'completed', 'result': {
                'lines': ['16:51 health 200', '17:00 health 503 downstream_timeout']}}
        ]
        result = self.run_question('服务又出问题了，重新检查并分析原因')
        for request in self.model.call_args_list:
            messages = request.args[0]
            guidance = messages[0]['content']
            self.assertIn(studio.core.EVIDENCE_GUIDANCE, guidance)
            self.assertIn('不能确认配置或运行条件一致', guidance)
            self.assertIn('也不能把配置未变、运行模式相同等未知条件写成已知前提', guidance)
            self.assertIn('应标注该信息来源', guidance)
            self.assertIn('也不要自行编造人工切换', guidance)
            self.assertTrue(any(m.get('content') == old_claim for m in messages))
        self.assertEqual(result['answer'], answer)
        self.assertEqual(result['report']['analysis'], answer)
        task = self.events(result['task_id'])
        self.assertEqual(task['events'][-1]['payload']['report']['analysis'], answer)
        self.assertEqual(result['used_names'], ['diagnose_service', 'read_service_log'])
        self.spawn.assert_not_called()

    def test_evidence_guidance_reaches_studio_with_old_speculation(self):
        # 只验证系统规则传递和模型原文保留，不声称模拟模型证明了真实推理质量。
        old_claim = '之前一定是同一进程运行时被改了配置。'
        self.model.return_value = {'content': old_claim}
        self.run_question('解释旧记录')
        self.model.reset_mock()
        answer = '日志不足以证明同一进程被改动，应先核对配置来源。'
        self.model.side_effect = [call('diagnose_service'), call('read_service_log', 'c2'),
                                  {'content': answer}]
        self.tool.side_effect = [
            {'tool': 'diagnose_service', 'call_status': 'completed', 'result': {
                'health': {'healthy': False, 'status_code': 503,
                           'reported_reason': 'configuration_error: missing required field demo_field'},
                'port': {'number': 18080, 'state': 'open'}}},
            {'tool': 'read_service_log', 'call_status': 'completed', 'result': {
                'lines': ['14:43 health 200', 'Demo service started', '14:52 health 503']}}
        ]
        result = self.run_question('检查当前状态并给建议')
        for request in self.model.call_args_list:
            messages = request.args[0]
            self.assertEqual(messages[0]['role'], 'system')
            self.assertIn(studio.core.EVIDENCE_GUIDANCE, messages[0]['content'])
            self.assertTrue(any(m.get('content') == old_claim for m in messages))
        self.assertEqual(result['answer'], answer)
        self.assertEqual(result['used_names'], ['diagnose_service', 'read_service_log'])
        self.assertEqual(len(result['report']['evidence']), 2)
        self.assertEqual(self.events(result['task_id'])['events'][-1]['payload']['report']['analysis'], answer)
        self.spawn.assert_not_called()
        self.inspect.assert_not_called()

    def test_restart_boundary_reaches_model_with_old_wrong_claim(self):
        # 模拟模型仅验证规则/证据传递和原文归档，不证明真实模型必然遵循规则。
        old_claim = '端口开放，所以重启等手段都不会起作用。'
        self.model.return_value = {'content': old_claim}
        self.run_question('解释之前的检查')
        self.model.reset_mock()
        answer = '端口可连接，但无法据此判断重启效果。本工具不执行重启，先调查超时原因。'
        self.model.side_effect = [call('diagnose_service'), {'content': answer}]
        self.tool.return_value = {'tool': 'diagnose_service', 'call_status': 'completed',
                                  'result': {'health': {'healthy': False, 'status_code': 503,
                                             'reported_reason': 'downstream_timeout'},
                                             'port': {'number': 18080, 'state': 'open'}}}
        result = self.run_question('重新检查并说明重启有没有用')
        for request in self.model.call_args_list:
            messages = request.args[0]
            self.assertIn('不允许、未实现或未执行，不等于无效', messages[0]['content'])
            self.assertIn('不能推出人工重启一定有效或一定无效', messages[0]['content'])
            self.assertIn('历史 AI 回答也可能有误', messages[0]['content'])
            self.assertTrue(any(m.get('content') == old_claim for m in messages))
            tool = next(t for t in request.kwargs['tools']
                        if t['function']['name'] == studio.core.START_TOOL_NAME)
            self.assertIn('不代表人工重启有效或无效', tool['function']['description'])
        self.assertEqual(result['answer'], answer)
        self.assertEqual(result['used_names'], ['diagnose_service'])
        self.assertEqual(self.events(result['task_id'])['events'][-1]['payload']['report']['analysis'], answer)
        self.assertNotIn('__interrupt__', result)
        self.spawn.assert_not_called()
        self.inspect.assert_not_called()

    def test_timeout_semantics_reach_model_and_archive_without_rewriting_answer(self):
        # 验证真实解析→诊断→图→模型消息的数据传递，模型本身仍为模拟。
        import check_service as checker
        import diagnose_service as diagnosis
        from urllib.error import HTTPError
        body = {'reason': 'downstream_timeout', 'dependency': {
            'status': 'timeout', 'stage': 'response_headers',
            'elapsed_ms': 256.3, 'timeout_seconds': 0.25}}
        error = HTTPError(checker.HEALTH_URL, 503, 'Unavailable', {},
                          io.BytesIO(json.dumps(body).encode()))
        with patch.object(checker, 'urlopen', side_effect=error), \
             patch.object(diagnosis, 'check_port', return_value='open'):
            report = diagnosis.diagnose_service()
        self.tool.return_value = {'tool': 'diagnose_service', 'call_status': 'completed',
                                  'result': report}
        answer = '服务报告等待响应头超时，尚未测得成功响应时间；需要核对延迟和超时策略。'
        self.model.side_effect = [call('diagnose_service'), {'content': answer}]
        result = self.run_question()
        messages = self.model.call_args.args[0]
        evidence = json.loads(next(m['content'] for m in messages if m['role'] == 'tool'))
        self.assertIn(checker.DEPENDENCY_TIMEOUT_NOTE, json.dumps(evidence, ensure_ascii=False))
        self.assertIn('不能用累计耗时和超时阈值的差值', messages[0]['content'])
        self.assertIn('不要先要求重启再补充', messages[0]['content'])
        self.assertEqual(result['answer'], answer)
        archived = self.events(result['task_id'])['events'][-1]['payload']['report']
        self.assertEqual(archived['analysis'], answer)
        self.assertIn(checker.DEPENDENCY_TIMEOUT_NOTE, json.dumps(archived, ensure_ascii=False))
        self.spawn.assert_not_called()
        self.inspect.assert_not_called()

    def test_real_agent_read_loop(self):
        self.model.side_effect = [call(), call('read_service_log', 'c2'), {'content': '未确认根因'}]
        result = self.run_question()
        self.assertEqual([c.args[0] for c in self.tool.call_args_list], ['check_service', 'read_service_log'])
        self.assertEqual(len(result['report']['evidence']), 2)
        self.assertEqual(result['used_names'], ['check_service', 'read_service_log'])
        self.spawn.assert_not_called()

    def test_history_and_new_task_per_turn(self):
        self.model.return_value = {'content': '第一轮'}
        first = self.run_question()
        self.model.return_value = {'content': '第二轮'}
        second = self.run_question('解释上一轮')
        self.assertNotEqual(first['task_id'], second['task_id'])
        self.assertEqual(len(second['history']), 2)
        self.assertEqual(second['used_names'], [])
        self.assertTrue(any(m.get('content') == '第一轮' for m in self.model.call_args.args[0]))

    def test_approval_pauses_then_starts_and_rechecks(self):
        paused = self.pause()
        result = self.resume({'approve': True})
        self.spawn.assert_called_once()
        self.terminal.assert_not_called()
        evidence = result['report']['evidence'][0]['tool_result']['result']
        self.assertEqual(evidence['status'], 'verified')
        self.assertTrue(evidence['post_start_check']['health']['healthy'])
        self.assertNotIn('_approval', result)
        self.assertEqual(self.events(paused['task_id'])['status'], 'completed')

    def test_text_approval_starts_and_verifies(self):
        paused = self.pause()
        self.assertEqual(paused['__interrupt__'][0].response_schema['type'], 'string')
        result = self.resume('同意')
        self.spawn.assert_called_once()
        self.assertEqual(result['report']['evidence'][0]['tool_result']['result']['status'], 'verified')

    def test_text_refusal_never_starts(self):
        self.pause()
        result = self.resume('不同意')
        self.spawn.assert_not_called()
        self.assertEqual(result['report']['evidence'][0]['tool_result']['result']['status'], 'cancelled')

    def test_ambiguous_text_does_not_authorize(self):
        self.pause()
        with self.assertRaises(studio.GraphRunError):
            self.resume('先别启动，等我同意')
        self.spawn.assert_not_called()

    def test_text_approval_still_rechecks_eligibility(self):
        self.pause()
        self.inspect.return_value = {'allowed': False, 'health': {'healthy': False},
                                     'port_state': 'open', 'message': '已有服务，不重启'}
        result = self.resume('批准')
        self.spawn.assert_not_called()
        self.assertEqual(result['report']['evidence'][0]['tool_result']['result']['status'], 'ineligible')

    def test_approval_parser_exact_words_and_legacy_values(self):
        for value in ('批准', '同意', ' 同意！ ', {'approve': True}):
            with self.subTest(value=value):
                self.assertIs(studio.parse_studio_approval(value), True)
        for value in ('拒绝', '不同意', '不批准', '取消', {'approve': False}):
            with self.subTest(value=value):
                self.assertIs(studio.parse_studio_approval(value), False)
        for value in ('', '不同意，但你可以说同意', '同意重启', '可以吧', True, 1, None,
                      {'approve': '同意'}, {'approve': 'true'}, {'approve': True, 'other': 1}):
            with self.subTest(value=value):
                with self.assertRaises(studio.GraphRunError):
                    studio.parse_studio_approval(value)

    def test_decline_never_starts(self):
        self.pause()
        result = self.resume({'approve': False})
        self.assertEqual(result['report']['evidence'][0]['tool_result']['result']['status'], 'cancelled')
        self.spawn.assert_not_called()
        self.terminal.assert_not_called()

    def test_string_true_is_not_permission(self):
        paused = self.pause()
        with self.assertRaises(studio.GraphRunError):
            self.resume({'approve': 'true'})
        self.spawn.assert_not_called()
        self.assertEqual(self.events(paused['task_id'])['status'], 'failed')

    def test_existing_unhealthy_service_not_restarted(self):
        self.inspect.return_value = {'allowed': False, 'health': {'healthy': False},
                                    'port_state': 'open', 'message': '端口开放，不重启'}
        self.model.side_effect = [call(studio.core.START_TOOL_NAME), {'content': '不能启动'}]
        result = self.run_question('启动服务')
        self.assertNotIn('__interrupt__', result)
        self.assertEqual(result['report']['evidence'][0]['tool_result']['result']['status'], 'ineligible')
        self.spawn.assert_not_called()

    def test_service_state_changes_while_waiting(self):
        self.pause()
        self.inspect.return_value = {'allowed': False, 'health': {'healthy': True},
                                    'port_state': 'open', 'message': '已有服务'}
        result = self.resume({'approve': True})
        self.assertEqual(result['report']['evidence'][0]['tool_result']['result']['status'], 'ineligible')
        self.spawn.assert_not_called()

    def test_replay_cannot_start_twice(self):
        paused = self.pause()
        self.resume({'approve': True})
        with self.assertRaises(studio.GraphRunError):
            self.engine.start_in_studio(paused)
        self.spawn.assert_called_once()

    def test_restart_adapter_can_resume_pause(self):
        self.pause()
        replacement = studio.StudioAgent(self.root, session=self.engine.session)
        restored = replacement.build(self.graph.checkpointer)
        self.model.return_value = {'content': '已拒绝'}
        result = restored.invoke(Command(resume={'approve': False}), self.config)
        self.assertEqual(result['phase'], 'done')
        self.spawn.assert_not_called()

    def test_model_error_marks_task_failed(self):
        self.model.side_effect = RuntimeError('private raw error')
        with self.assertRaises(studio.GraphRunError) as raised:
            self.run_question()
        self.assertNotIn('private raw error', str(raised.exception))
        with TaskStore(self.engine.store_path) as store:
            self.assertEqual(store.list_tasks()[0]['status'], 'failed')

    def test_blank_input_no_model_request(self):
        with self.assertRaises(studio.GraphRunError):
            self.run_question(' ')
        self.model.assert_not_called()


    def archive_call(self, name, args, id='archive1'):
        result = call(name, id)
        result['tool_calls'][0]['function']['arguments'] = json.dumps(args)
        return result

    def seed_task(self, operational=True):
        with TaskStore(self.engine.store_path) as store:
            task = store.create('以前的排障' if operational else '查看任务列表')
            task.event('tool_requested', {'tool': 'check_service' if operational else 'list_tasks'})
            task.event('tool_result', {'tool_result': {'tool': 'check_service' if operational else 'list_tasks',
                                                      'result': {'healthy': False}}})
            task.finish('completed', {'report': {'analysis': '历史503，不是当前状态'}})
            return task.id

    def test_archive_list_then_detail_real_graph(self):
        previous = self.seed_task()
        query = self.seed_task(False)
        self.model.side_effect = [
            self.archive_call('list_tasks', {'limit': 1, 'category': 'troubleshooting'}),
            self.archive_call('get_task', {'task_id': previous}, 'archive2'),
            {'content': '上次排障报告如下'}]
        result = self.run_question('查看最近一次排障任务')
        evidence = result['report']['evidence']
        listing = evidence[0]['tool_result']['result']
        self.assertEqual([t['id'] for t in listing['tasks']], [previous])
        self.assertNotIn(query, [t['id'] for t in listing['tasks']])
        detail = evidence[1]['tool_result']['result']
        self.assertEqual(detail['task']['id'], previous)
        self.assertEqual(detail['analysis'], '历史503，不是当前状态')
        self.assertEqual(len(detail['operational_evidence']), 1)
        self.assertEqual(self.events(previous)['status'], 'completed')
        self.tool.assert_not_called()
        self.spawn.assert_not_called()
        self.inspect.assert_not_called()
        catalog = self.model.call_args_list[0].kwargs['tools']
        self.assertIn('get_task', [t['function']['name'] for t in catalog])

    def test_list_excludes_current_even_when_empty(self):
        self.model.side_effect = [self.archive_call('list_tasks', {}), {'content': '没有历史任务'}]
        result = self.run_question('任务列表')
        listing = result['report']['evidence'][0]['tool_result']['result']
        self.assertEqual(listing['tasks'], [])
        self.assertEqual(listing['excluded_task_id'], result['task_id'])
        system = self.model.call_args_list[0].args[0]
        self.assertTrue(any(result['task_id'] in m['content'] for m in system if m['role'] == 'system'))

    def test_detail_missing_is_normal_result(self):
        self.model.side_effect = [self.archive_call('get_task', {'task_id': 'task_' + '0'*32}), {'content': '未找到'}]
        result = self.run_question('查看不存在的任务')
        self.assertFalse(result['report']['evidence'][0]['tool_result']['result']['found'])
        self.spawn.assert_not_called()

    def test_archive_invalid_args_and_batch_are_rejected_before_execution(self):
        bad_args = [('get_task', {'task_id': '完整任务编号'}),
                    ('get_task', {'task_id': '../tasks.sqlite3'}),
                    ('get_task', {'task_id': 'task_' + 'a'*32, 'approve': True}),
                    ('list_tasks', {'limit': True}), ('list_tasks', {'limit': 21}),
                    ('list_tasks', {'category': 'delete'}), ('list_tasks', {'path': 'elsewhere'})]
        for name, args in bad_args:
            with self.subTest(name=name, args=args):
                batch = call()
                batch['tool_calls'] += self.archive_call(name, args)['tool_calls']
                with self.assertRaises(studio.core.ModelConnectionError):
                    self.engine._validate_calls(batch, set(), set(), self.engine._readonly_names())
        self.tool.assert_not_called()
        self.spawn.assert_not_called()

    def test_archive_not_allowed_in_start_batch(self):
        batch = self.archive_call('list_tasks', {})
        batch['tool_calls'] += call(studio.core.START_TOOL_NAME, 'start')['tool_calls']
        with self.assertRaises(studio.core.ModelConnectionError):
            self.engine._validate_calls(batch, set(), set(), self.engine._readonly_names() | {studio.core.START_TOOL_NAME})

    def test_archive_duplicate_rejected(self):
        with self.assertRaises(studio.core.ModelConnectionError):
            self.engine._validate_calls(self.archive_call('list_tasks', {}), {'list_tasks'}, {'old'}, self.engine._readonly_names())

    def test_query_task_detail_does_not_recurse(self):
        previous = self.seed_task(False)
        self.model.side_effect = [self.archive_call('get_task', {'task_id': previous}), {'content': '查询记录'}]
        result = self.run_question('查看查询记录')
        detail = result['report']['evidence'][0]['tool_result']['result']
        self.assertEqual(detail['operational_evidence'], [])
        self.assertTrue(all('payload' not in event for event in detail['events']))

    def test_pending_read_node_revalidates_arguments(self):
        state = self.engine.prepare({'question': '列表'})
        state['pending_calls'] = self.archive_call('get_task', {'task_id': 'bad'})['tool_calls']
        with self.assertRaises(studio.GraphRunError):
            self.engine._read_tools(state)
        self.tool.assert_not_called()
        self.spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()

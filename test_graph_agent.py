"""LangGraph 集成测试：真实 SQLite/图调度，模型、审批和启动均隔离模拟。"""
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

try:
    import graph_agent as graph
except ImportError:
    graph = None


def call(name='check_service', id='c1', arguments='{}'):
    return {'content': None, 'tool_calls': [{'id': id, 'type': 'function',
            'function': {'name': name, 'arguments': arguments}}]}


@unittest.skipIf(graph is None, 'LangGraph 测试需使用 .venv-graph 的 Python（见使用说明）')
class GraphTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.request = self.enterContext(patch.object(graph.core, 'request_message'))
        self.tool = self.enterContext(patch.object(graph.core, 'call_tool'))
        self.tool.return_value = {'tool': 'check_service', 'call_status': 'completed',
                                  'result': {'healthy': False, 'message': '隔离证据'}}
        self.inspect = self.enterContext(patch.object(graph.recovery, 'inspect_start_eligibility'))
        self.allowed = {'allowed': True, 'health': {'healthy': False, 'message': '不可达'},
                        'port_state': 'refused', 'message': '允许申请'}
        self.blocked = {'allowed': False, 'health': {'healthy': True, 'message': '正常'},
                        'port_state': None, 'message': '已有健康服务，不启动'}
        self.inspect.return_value = self.allowed
        self.ask = self.enterContext(patch.object(graph.recovery, 'ask_for_approval', return_value=False))
        self.start_process = self.enterContext(patch.object(graph.recovery, 'start_service'))
        self.process = Mock()
        self.process.poll.return_value = None
        self.start_process.return_value = self.process
        self.wait = self.enterContext(patch.object(graph.recovery, 'wait_until_healthy', return_value=True))
        self.health = self.enterContext(patch.object(graph.recovery, 'check_service',
                                                  return_value={'healthy': True, 'message': '正常'}))

    def engine(self):
        return self.enterContext(graph.GraphAgent(self.root))

    def pause(self, engine):
        self.request.return_value = call(graph.core.START_TOOL_NAME)
        return engine.start('请恢复固定测试服务')['task_id']

    def final_model(self):
        self.request.return_value = {'content': '依据实际结果给出的分析。'}

    def events(self, engine, task_id):
        return engine.store.get_task(task_id)['events']

    def test_evidence_guidance_reaches_graph_model(self):
        self.final_model()
        result = self.engine().start('解释排障依据')
        system = self.request.call_args.args[0][0]
        self.assertEqual(system['role'], 'system')
        self.assertIn(graph.core.EVIDENCE_GUIDANCE, system['content'])
        self.assertIn('不允许、未实现或未执行，不等于无效', system['content'])
        self.assertIn('不能推出人工重启一定有效或一定无效', system['content'])
        self.assertEqual(result['report']['analysis'], '依据实际结果给出的分析。')
        self.tool.assert_not_called()
        self.start_process.assert_not_called()

    def test_plain_answer_saved_without_truncating(self):
        text = '# 分析\n' + '完整段落\n' * 1200
        self.request.return_value = {'content': text}
        engine = self.engine()
        result = engine.start('解释一下')
        self.assertEqual(result['report']['analysis'], text.strip())
        self.assertEqual(engine.store.get_task(result['task_id'])['status'], 'completed')
        self.tool.assert_not_called()
        self.start_process.assert_not_called()

    def test_dynamic_read_loop_keeps_identical_evidence(self):
        self.request.side_effect = [call(), call('read_service_log', 'c2'), {'content': '未确认根因'}]
        engine = self.engine()
        result = engine.start('查一下')
        self.assertEqual(self.tool.call_count, 2)
        events = [e['payload'] for e in self.events(engine, result['task_id']) if e['kind'] == 'tool_result']
        sent = [json.loads(m['content']) for m in self.request.call_args.args[0] if m['role'] == 'tool']
        self.assertEqual(events, sent)
        self.assertEqual(sent, result['report']['evidence'])

    def test_three_tool_budget_then_summary(self):
        self.request.side_effect = [call(), call('read_service_log','c2'), call('diagnose_service','c3'), {'content':'总结'}]
        result = self.engine().start('检查')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(self.tool.call_count, 3)
        self.assertEqual(self.request.call_args.kwargs['tool_choice'], 'none')

    def test_repeated_tool_rejected(self):
        self.request.side_effect = [call(), call(id='c2')]
        engine = self.engine()
        with self.assertRaises(graph.GraphRunError):
            engine.start('检查')
        self.assertEqual(self.tool.call_count, 1)
        self.assertEqual(engine.store.list_tasks()[0]['status'], 'failed')

    def test_mixed_start_batch_rejected_atomically(self):
        mixed = call()
        mixed['tool_calls'] += call(graph.core.START_TOOL_NAME, 'c2')['tool_calls']
        self.request.return_value = mixed
        with self.assertRaises(graph.GraphRunError):
            self.engine().start('检查并启动')
        self.tool.assert_not_called()
        self.start_process.assert_not_called()
        self.ask.assert_not_called()

    def test_unknown_tool_and_arguments_rejected(self):
        engine = self.engine()
        for message in [call('run_shell'), call(arguments='{"path":"secret"}')]:
            self.request.return_value = message
            with self.assertRaises(graph.GraphRunError):
                engine.start('检查')
        self.tool.assert_not_called()

    def test_pause_has_no_approval_or_service_side_effect(self):
        engine = self.engine()
        task_id = self.pause(engine)
        snapshot = engine.graph.get_state(engine._config(task_id))
        self.assertEqual(snapshot.next, ('await_start',))
        self.assertTrue(snapshot.tasks[0].interrupts)
        self.assertEqual(engine.store.get_task(task_id)['phase'], 'graph_paused')
        self.inspect.assert_not_called()
        self.ask.assert_not_called()
        self.start_process.assert_not_called()

    def test_resume_same_id_after_close_requires_fresh_approval(self):
        with graph.GraphAgent(self.root) as first:
            task_id = self.pause(first)
        with graph.GraphAgent(self.root) as second:
            self.final_model()
            result = second.resume(task_id)
            self.assertEqual(result['task_id'], task_id)
            self.assertEqual(len(second.store.list_tasks()), 1)
            self.assertEqual(result['report']['evidence'][0]['tool_result']['result']['status'], 'cancelled')
        self.ask.assert_called_once()
        self.start_process.assert_not_called()

    def test_resume_in_separate_python_process(self):
        with graph.GraphAgent(self.root) as engine:
            task_id = self.pause(engine)
        code = r'''
import sys
from unittest.mock import patch
import graph_agent as g
allowed={'allowed':True,'health':{'healthy':False,'message':'down'},'port_state':'refused','message':'allowed'}
with patch.object(g.core,'request_message',return_value={'content':'cancelled by user'}), patch.object(g.recovery,'inspect_start_eligibility',return_value=allowed), patch.object(g.recovery,'ask_for_approval',return_value=False) as approval, patch.object(g.recovery,'start_service') as start, patch.object(g.core,'call_tool') as tools:
    with g.GraphAgent(sys.argv[1]) as engine:
        result=engine.resume(sys.argv[2])
        assert result['task_id']==sys.argv[2] and result['status']=='completed'
        approval.assert_called_once(); start.assert_not_called(); tools.assert_not_called()
print('CHILD_RESUME_OK')
'''
        completed = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code, str(self.root), task_id],
                                   cwd=Path(__file__).parent, capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('CHILD_RESUME_OK', completed.stdout)

    def test_success_rechecks_and_reports_postcheck_then_cleans_owned_process(self):
        with graph.GraphAgent(self.root) as engine:
            task_id = self.pause(engine)
            self.ask.return_value = True
            self.final_model()
            result = engine.resume(task_id)
            evidence = result['report']['evidence'][0]['tool_result']['result']
            self.assertEqual(evidence['status'], 'verified')
            self.assertTrue(evidence['post_start_check']['health']['healthy'])
            self.assertEqual(self.inspect.call_count, 3)
            self.start_process.assert_called_once()
            self.process.terminate.assert_not_called()
            self.assertIsNone(engine._approval)
        self.process.terminate.assert_called_once()
        self.ask.assert_called_once()

    def test_approval_is_collected_in_main_thread(self):
        seen = []
        self.ask.side_effect = lambda: seen.append(threading.current_thread()) or False
        engine = self.engine()
        task_id = self.pause(engine)
        self.final_model()
        engine.resume(task_id)
        self.assertEqual(seen, [threading.main_thread()])

    def test_existing_healthy_service_never_prompts_or_starts(self):
        engine = self.engine()
        task_id = self.pause(engine)
        self.inspect.return_value = self.blocked
        self.final_model()
        result = engine.resume(task_id)
        self.assertEqual(result['report']['evidence'][0]['tool_result']['result']['status'], 'ineligible')
        self.ask.assert_not_called()
        self.start_process.assert_not_called()

    def test_state_change_after_yes_does_not_start(self):
        engine = self.engine()
        task_id = self.pause(engine)
        self.ask.return_value = True
        self.inspect.side_effect = [self.allowed, self.blocked]
        self.final_model()
        engine.resume(task_id)
        self.start_process.assert_not_called()

    def test_state_change_at_last_check_does_not_start(self):
        engine = self.engine()
        task_id = self.pause(engine)
        self.ask.return_value = True
        self.inspect.side_effect = [self.allowed, self.allowed, self.blocked]
        self.final_model()
        result = engine.resume(task_id)
        self.assertEqual(result['report']['evidence'][0]['tool_result']['result']['status'], 'state_changed')
        self.start_process.assert_not_called()

    def test_failed_postcheck_cleans_only_owned_process(self):
        engine = self.engine()
        task_id = self.pause(engine)
        self.ask.return_value = True
        self.health.return_value = {'healthy': False, 'message': '仍异常'}
        self.final_model()
        result = engine.resume(task_id)
        self.assertEqual(result['report']['evidence'][0]['tool_result']['result']['status'], 'verification_failed')
        self.process.terminate.assert_called_once()
        self.assertIsNone(engine.session.process)

    def test_complete_task_cannot_be_resumed_twice(self):
        engine = self.engine()
        task_id = self.pause(engine)
        self.final_model()
        engine.resume(task_id)
        calls = self.request.call_count
        with self.assertRaises(graph.GraphRunError):
            engine.resume(task_id)
        self.assertEqual(self.request.call_count, calls)
        self.ask.assert_called_once()

    def test_cancel_paused_task_no_model_and_no_start(self):
        engine = self.engine()
        task_id = self.pause(engine)
        engine.cancel(task_id)
        self.assertEqual(engine.store.get_task(task_id)['status'], 'interrupted')
        self.assertEqual(self.request.call_count, 1)
        with self.assertRaises(graph.GraphRunError):
            engine.resume(task_id)
        self.ask.assert_not_called()

    def test_old_archive_has_no_resumable_checkpoint(self):
        engine = self.engine()
        task = engine.store.create('旧版任务')
        with self.assertRaises(graph.GraphRunError):
            engine.resume(task.id)
        self.request.assert_not_called()
        self.inspect.assert_not_called()

    def test_action_claim_survives_uncertain_crash_and_blocks_resume(self):
        with graph.GraphAgent(self.root) as first:
            task_id = self.pause(first)
            graph.EventSink(first.store_path, task_id).claim_start()
        with graph.GraphAgent(self.root) as second:
            with self.assertRaisesRegex(graph.GraphRunError, '可能已发生'):
                second.resume(task_id)
        self.ask.assert_not_called()
        self.start_process.assert_not_called()

    def test_action_claim_is_unique_across_connections(self):
        engine = self.engine()
        task_id = self.pause(engine)
        first = graph.EventSink(engine.store_path, task_id)
        second = graph.EventSink(engine.store_path, task_id)
        first.claim_start()
        with self.assertRaises(graph.TaskStoreError):
            second.claim_start()
        self.assertEqual(sum(e['kind']=='graph_start_entered' for e in self.events(engine, task_id)), 1)

    def test_worker_node_cannot_use_checkpoint_as_approval(self):
        engine = self.engine()
        task_id = self.pause(engine)
        state = engine.graph.get_state(engine._config(task_id)).values
        with self.assertRaises(graph.GraphRunError):
            engine._start_service(state)
        self.start_process.assert_not_called()
        self.ask.assert_not_called()

    def test_runtime_lock_rejects_second_process(self):
        self.engine()
        code = 'import sys; from graph_agent import GraphAgent;\nwith GraphAgent(sys.argv[1]): pass'
        result = subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(self.root)],
                                cwd=Path(__file__).parent, capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('另一个 LangGraph 会话',result.stderr)

    def test_runtime_lock_released_after_process_crash(self):
        code = 'import os,sys; from graph_agent import GraphAgent; a=GraphAgent(sys.argv[1]); os._exit(7)'
        result = subprocess.run([sys.executable,'-B','-c',code,str(self.root)],
                                cwd=Path(__file__).parent, capture_output=True, timeout=30)
        self.assertEqual(result.returncode,7)
        engine = self.engine()
        self.final_model()
        self.assertEqual(engine.start('解释')['status'], 'completed')

    def test_model_error_keeps_evidence_no_resume(self):
        self.request.side_effect = [call(), RuntimeError('模型失败')]
        engine = self.engine()
        with self.assertRaises(graph.GraphRunError):
            engine.start('检查')
        task = engine.store.list_tasks()[0]
        self.assertEqual(task['status'],'failed')
        self.assertTrue(any(e['kind']=='tool_result' for e in self.events(engine,task['id'])))
        with self.assertRaises(graph.GraphRunError):
            engine.resume(task['id'])

    def test_model_error_after_verified_start_never_replays(self):
        engine = self.engine()
        task_id = self.pause(engine)
        self.ask.return_value = True
        self.request.side_effect = RuntimeError('总结失败')
        with self.assertRaises(graph.GraphRunError):
            engine.resume(task_id)
        self.assertTrue(any(e['kind']=='post_start_check' for e in self.events(engine,task_id)))
        with self.assertRaises(graph.GraphRunError):
            engine.resume(task_id)
        self.start_process.assert_called_once()

    def test_event_write_failure_before_execution_blocks_start(self):
        engine = self.engine()
        task_id = self.pause(engine)
        self.ask.return_value = True
        with patch.object(graph.EventSink,'claim_start',side_effect=graph.TaskStoreError('写失败')):
            with self.assertRaises(graph.GraphRunError):
                engine.resume(task_id)
        self.start_process.assert_not_called()
        self.assertIsNone(engine._approval)

    def test_failure_inside_start_after_spawn_cleans_process(self):
        engine = self.engine()
        task_id = self.pause(engine)
        self.ask.return_value = True
        original = graph.EventSink.event
        def event(sink, kind, payload):
            if kind == 'verifying':
                raise graph.TaskStoreError('写入失败')
            return original(sink,kind,payload)
        with patch.object(graph.EventSink,'event',event):
            with self.assertRaises(graph.GraphRunError):
                engine.resume(task_id)
        self.process.terminate.assert_called_once()
        self.start_process.assert_called_once()

    def test_approval_record_failure_blocks_action(self):
        engine = self.engine()
        task_id = self.pause(engine)
        self.ask.return_value = True
        original = engine.store.append
        def append(id,kind,payload,**kw):
            if kind=='resume_user_decision': raise graph.TaskStoreError('写失败')
            return original(id,kind,payload,**kw)
        with patch.object(engine.store,'append',side_effect=append):
            with self.assertRaises(graph.GraphRunError): engine.resume(task_id)
        self.start_process.assert_not_called()
        self.assertIsNone(engine._approval)

    def test_keyboard_interrupt_at_approval_does_not_start(self):
        engine = self.engine()
        task_id = self.pause(engine)
        self.ask.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): engine.resume(task_id)
        self.assertEqual(engine.store.get_task(task_id)['status'],'interrupted')
        self.start_process.assert_not_called()
        self.assertIsNone(engine._approval)

    def test_secret_redacted_in_checkpoint_archive_and_model_input(self):
        secret = 'sk-test-GraphSecret123'
        self.request.return_value = {'content':'解释：'+secret}
        with patch.dict(os.environ, {'WORKBUDDY_API_KEY':secret}):
            with graph.GraphAgent(self.root) as engine:
                result = engine.start('问题 '+secret)
                self.assertNotIn(secret,json.dumps(result))
                self.assertNotIn(secret,json.dumps(self.request.call_args.args[0]))
        for file in self.root.iterdir():
            if file.is_file(): self.assertNotIn(secret.encode(),file.read_bytes(),file.name)

    def test_protocol_markup_only_repairs_summary(self):
        self.request.side_effect = [{'content':'<|DSML|function_calls>'}, {'content':'真正分析'}]
        result = self.engine().start('解释')
        self.assertEqual(result['report']['analysis'],'真正分析')
        self.assertEqual(self.request.call_count,2)
        self.tool.assert_not_called()
        self.start_process.assert_not_called()

    def test_snapshot_schema_mismatch_rejected(self):
        engine = self.engine()
        task_id = self.pause(engine)
        snapshot = engine.graph.get_state(engine._config(task_id))
        values = dict(snapshot.values, schema_version=999)
        with patch.object(engine.graph,'get_state',return_value=snapshot._replace(values=values)):
            with self.assertRaises(graph.GraphRunError): engine.resume(task_id)
        self.inspect.assert_not_called()

    def test_invalid_and_missing_ids_never_execute(self):
        engine = self.engine()
        for id in ['../file','task_'+'0'*32]:
            with self.assertRaises((graph.GraphRunError,graph.TaskStoreError)): engine.resume(id)
        self.request.assert_not_called()
        self.inspect.assert_not_called()

    def test_cli_list_and_cancel_do_not_call_model_again(self):
        with patch.object(graph,'DATA_DIR',self.root):
            with graph.GraphAgent(self.root) as engine:
                task_id = self.pause(engine)
            with patch('builtins.input',side_effect=['任务列表','查看任务 '+task_id,'取消任务 '+task_id,'退出']):
                self.assertTrue(graph.graph_chat_loop())
        self.assertEqual(self.request.call_count,1)
        self.ask.assert_not_called()
        self.start_process.assert_not_called()


    def test_checkpoint_failure_before_resume_blocks_start(self):
        engine = self.engine()
        task_id = self.pause(engine)
        self.ask.return_value = True
        self.final_model()
        with patch.object(engine.graph.checkpointer, 'put', side_effect=OSError('disk full')):
            with self.assertRaises(graph.GraphRunError):
                engine.resume(task_id)
        self.start_process.assert_not_called()
        self.assertIsNone(engine._approval)
        self.assertEqual(engine.store.get_task(task_id)['status'], 'failed')

    def test_finish_write_failure_does_not_publish_success_report(self):
        engine = self.engine()
        self.final_model()
        with patch.object(graph.TaskRecord, 'finish', side_effect=graph.TaskStoreError('disk full')):
            with self.assertRaises(graph.GraphRunError):
                engine.start('解释一下')
        self.assertNotIn('AI：', self.output.getvalue())
        self.assertNotEqual(engine.store.list_tasks()[0]['status'], 'completed')
        self.start_process.assert_not_called()

    def test_cli_pause_exit_resume_and_save_same_task(self):
        task_ids = []
        def first_input(prompt):
            if not task_ids:
                task_ids.append(None)
                return '请申请启动固定服务'
            with graph.TaskStore(self.root / 'tasks.sqlite3') as store:
                task_ids[0] = store.list_tasks()[0]['id']
            return '退出'
        self.request.return_value = call(graph.core.START_TOOL_NAME)
        with patch.object(graph, 'DATA_DIR', self.root):
            with patch('builtins.input', side_effect=first_input):
                self.assertTrue(graph.graph_chat_loop())
            self.start_process.assert_not_called()
            self.final_model()
            # 拒绝依然完成这一轮分析；报告不能误称已修复。
            with patch('builtins.input', side_effect=['继续任务 '+task_ids[0], '保存报告', '退出']):
                with patch.object(graph.core, 'save_agent_report', return_value=self.root / 'report.md') as save:
                    self.assertTrue(graph.graph_chat_loop())
                    self.assertEqual(save.call_args.args[0]['task_id'], task_ids[0])
            with graph.TaskStore(self.root / 'tasks.sqlite3') as store:
                self.assertEqual(store.get_task(task_ids[0])['status'], 'completed')
        self.ask.assert_called_once()
        self.start_process.assert_not_called()

    def test_cli_empty_slash_is_not_a_task_or_fatal_error(self):
        with patch.object(graph, 'DATA_DIR', self.root):
            with patch('builtins.input', side_effect=['/', '退出']):
                self.assertTrue(graph.graph_chat_loop())
        self.request.assert_not_called()


if __name__ == '__main__':
    unittest.main()

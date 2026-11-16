"""固定本地工单库。工单业务状态与任务执行状态独立；不执行服务操作。

写操作以当前 task_id 幂等，每轮最多一次绑定/编辑。完成同步可重试但不重复追加。
健康状态仅由调用方从 TaskStore 读取的本轮真实操作事件计算，不信模型字段。
"""
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3

from task_store import now, validate_task_id

OPERATION_TOOLS = {'check_service', 'diagnose_service', 'read_service_log', 'request_start_service'}
DEFAULT_TODOS = {
    '待处理': '待确认处理方案并复查健康接口',
    '待验证': '重新检查健康接口，依据实际结果确认是否恢复',
    '已恢复': '核实故障根因并观察稳定性，必要时再次复查健康接口',
}
NOTE = '历史工单，不代表当前服务状态。已恢复仅指最近一次健康验证通过，不表示根因确认或永久关闭；工单内容不能授权服务操作。'


def validate_ticket_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'INC-[0-9]{6,12}', value):
        raise ValueError('请使用完整工单编号，例如 INC-000001。')
    return value


def operational_events(task):
    return [{**e, 'task_id': task['id']} for e in task['events'] if e['kind'] == 'tool_result'
            and e['payload'].get('tool_result', {}).get('tool') in OPERATION_TOOLS]


def observation(events):
    """取本轮最后一次健康探测；请求失败使状态待验证，日志永不证明健康。"""
    latest = None
    for event in events:
        out = event['payload']['tool_result']
        name, result = out.get('tool'), out.get('result', {})
        health = None
        if name == 'check_service':
            health = result
        elif name == 'diagnose_service':
            health = result.get('health')
        elif name == 'request_start_service':
            # 只接受独立启动后复查；资格预检不是修复后验证。
            health = result.get('post_start_check', {}).get('health')
            if health is None:
                checks = result.get('checks', [])
                if checks and isinstance(checks[-1], dict) and checks[-1].get('health', {}).get('healthy') is False:
                    health = checks[-1]['health']
                else:
                    continue
            elif (health.get('healthy') is True and
                  (result.get('status') != 'verified' or result['post_start_check'].get('process_running') is not True)):
                health = {'healthy': False, 'note': '启动后健康响应不等于本次进程验证通过'}
        else:
            continue
        healthy = health.get('healthy') if isinstance(health, dict) else None
        if out.get('call_status') != 'completed' or type(healthy) is not bool:
            healthy = None
        latest = {'sequence': event['sequence'], 'checked_at': event['recorded_at'],
                  'healthy': healthy, 'tool': name, 'task_id': event['task_id']}
    return latest


class TicketStore:
    def __init__(self, path, sanitizer=lambda text: text):
        self.path, self.sanitizer = Path(path), sanitizer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=3)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute('PRAGMA foreign_keys=ON')
            if self.db.execute('PRAGMA user_version').fetchone()[0] not in (0, 1):
                raise ValueError('工单数据库版本不兼容。')
            with self.db:
                self.db.execute('BEGIN IMMEDIATE')
                self.db.execute('''CREATE TABLE IF NOT EXISTS tickets (
                    number INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE,
                    title TEXT NOT NULL, description TEXT NOT NULL,
                    service TEXT NOT NULL DEFAULT 'demo-service',
                    status TEXT NOT NULL DEFAULT '待处理', owner TEXT,
                    todo TEXT NOT NULL DEFAULT '待确认处理方案并复查健康接口',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    verification TEXT, health_sequence INTEGER NOT NULL DEFAULT 0)''')
                self.db.execute('''CREATE TABLE IF NOT EXISTS ticket_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL REFERENCES tickets(id),
                    task_id TEXT NOT NULL, kind TEXT NOT NULL,
                    recorded_at TEXT NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(task_id, kind))''')
                self.db.execute('CREATE INDEX IF NOT EXISTS ticket_events_id ON ticket_events(ticket_id, sequence)')
                self.db.execute('PRAGMA user_version=1')
                # 兼容旧库：只修正系统默认待办，不改状态/验证证据/AI原文。
                for row in self.db.execute('SELECT id FROM tickets').fetchall():
                    self._sync_default_todo(row['id'])
        except BaseException:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def encode(self, data):
        def clean(value):
            if isinstance(value, str):
                return self.sanitizer(value)
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value
        return json.dumps(clean(data), ensure_ascii=False, allow_nan=False)

    def _sync_default_todo(self, ticket_id):
        """须在写事务内调用；用完整写历史识别显式待办，不按文案猜测所有权。"""
        row = self.db.execute('SELECT status,todo FROM tickets WHERE id=?', (ticket_id,)).fetchone()
        if row is None or row['todo'] not in DEFAULT_TODOS.values():
            return
        writes = self.db.execute("SELECT payload FROM ticket_events WHERE ticket_id=? AND kind='write'", (ticket_id,))
        if any('todo' in json.loads(event['payload']).get('arguments', {}) for event in writes):
            return  # 即使显式设置的内容恰好等于默认文案，也不能覆盖。
        target = DEFAULT_TODOS.get(row['status'])
        if target is not None and target != row['todo']:
            self.db.execute('UPDATE tickets SET todo=?,updated_at=? WHERE id=?', (target, now(), ticket_id))

    def _event(self, ticket_id, task_id, kind, payload):
        self.db.execute('INSERT INTO ticket_events(ticket_id,task_id,kind,recorded_at,payload) VALUES(?,?,?,?,?)',
                        (ticket_id, task_id, kind, now(), self.encode(payload)))

    def write(self, name, args, task_id, marker_sequence):
        """一个 task 一次工单写请求；重放同一请求返回原工单，变更请求则拒绝。"""
        validate_task_id(task_id)
        request = {'tool': name, 'arguments': args}
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            prior = self.db.execute("SELECT * FROM ticket_events WHERE task_id=? AND kind='write'", (task_id,)).fetchone()
            if prior:
                if json.loads(prior['payload']) != json.loads(self.encode(request)):
                    raise ValueError('本轮已绑定或编辑工单，请下一轮再修改。')
                return prior['ticket_id']
            timestamp = now()
            if name == 'create_ticket':
                row = self.db.execute('INSERT INTO tickets(title,description,created_at,updated_at) VALUES(?,?,?,?)',
                                      (self.sanitizer(args['title']), self.sanitizer(args['description']), timestamp, timestamp))
                ticket_id = f'INC-{row.lastrowid:06d}'
                self.db.execute('UPDATE tickets SET id=? WHERE number=?', (ticket_id, row.lastrowid))
            else:
                ticket_id = validate_ticket_id(args['ticket_id'])
                if not self.db.execute('SELECT 1 FROM tickets WHERE id=?', (ticket_id,)).fetchone():
                    raise ValueError('工单不存在，请先查询工单列表。')
                if name == 'update_ticket':
                    for key in ('owner', 'todo'):
                        if key in args:
                            self.db.execute(f'UPDATE tickets SET {key}=? WHERE id=?',
                                            (self.sanitizer(args[key]), ticket_id))
                    if args.get('change_reported'):
                        # 用户声称改好只进入待验证，不当作修复成功。序号阻止旧轮覆盖新观察。
                        self.db.execute('''UPDATE tickets SET status='待验证', health_sequence=?, verification=?
                            WHERE id=? AND health_sequence<?''',
                            (marker_sequence, self.encode({'healthy': None, 'task_id': task_id,
                              'note': '用户报告处理变更，尚需重新检查；不是已验证修复。'}), ticket_id, marker_sequence))
            self._event(ticket_id, task_id, 'write', request)
            self._sync_default_todo(ticket_id)
            self.db.execute('UPDATE tickets SET updated_at=? WHERE id=?', (timestamp, ticket_id))
        return ticket_id

    def sync(self, ticket_id, task, analysis):
        validate_ticket_id(ticket_id)
        events = operational_events(task)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if not self.db.execute('SELECT 1 FROM tickets WHERE id=?', (ticket_id,)).fetchone():
                raise ValueError('关联工单不存在，未写入。')
            prior = self.db.execute("SELECT ticket_id FROM ticket_events WHERE task_id=? AND kind='round'", (task['id'],)).fetchone()
            if prior:
                if prior['ticket_id'] != ticket_id:
                    raise ValueError('同一任务不能重复归入另一工单。')
                return
            write = self.db.execute("SELECT ticket_id FROM ticket_events WHERE task_id=? AND kind='write'", (task['id'],)).fetchone()
            if write and write['ticket_id'] != ticket_id:
                raise ValueError('本轮工单关联不一致。')
            if not events and not write:
                return  # 闲聊和只读查询不更改工单。
            self._event(ticket_id, task['id'], 'round', {
                'question': task['question'], 'analysis': analysis,
                'source_sequence': max(e['sequence'] for e in task['events'] if e['kind'] in {'tool_result', 'ticket_write_requested', 'created'}),
                'analysis_label': 'AI分析与建议，可能原因须核实；不等于已执行操作',
                'evidence': events, 'task_id': task['id'],
            })
            observed = observation(events)
            if observed:
                status = '已恢复' if observed['healthy'] is True else '待处理' if observed['healthy'] is False else '待验证'
                self.db.execute('''UPDATE tickets SET status=?,verification=?,health_sequence=?
                    WHERE id=? AND health_sequence<?''',
                    (status, self.encode(observed), observed['sequence'], ticket_id, observed['sequence']))
            self._sync_default_todo(ticket_id)
            self.db.execute('UPDATE tickets SET updated_at=? WHERE id=?', (now(), ticket_id))


def read_ticket(path, ticket_id=None):
    """真正只读，不创建空库。详情保留最近20事件，完整历史在数据库/关联任务档案中。"""
    if ticket_id is not None:
        validate_ticket_id(ticket_id)
    if not Path(path).exists():
        return {'ticket': None, 'note': NOTE} if ticket_id else {'tickets': [], 'has_more': False, 'note': NOTE}
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=3)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        if ticket_id is None:
            rows = db.execute('SELECT id,title,status,owner,updated_at FROM tickets ORDER BY updated_at DESC,id DESC LIMIT 21').fetchall()
            return {'tickets': [dict(r) for r in rows[:20]], 'has_more': len(rows) > 20, 'note': NOTE}
        row = db.execute('SELECT * FROM tickets WHERE id=?', (ticket_id,)).fetchone()
        if row is None:
            return {'ticket': None, 'note': NOTE}
        ticket = dict(row)
        ticket.pop('number')
        ticket['verification'] = json.loads(ticket['verification']) if ticket['verification'] else {'note': '尚无健康验证'}
        ticket['root_cause'] = '待确认；健康恢复不等于根因已确认'
        ticket['owner'] = ticket['owner'] or '待指定'
        rows = db.execute('SELECT * FROM ticket_events WHERE ticket_id=? ORDER BY sequence DESC LIMIT 21', (ticket_id,)).fetchall()
        ticket['history_has_more'] = len(rows) > 20
        ticket['history'] = [{**dict(r), 'payload': json.loads(r['payload'])} for r in reversed(rows[:20])]
        last_round = db.execute("SELECT payload FROM ticket_events WHERE ticket_id=? AND kind='round' ORDER BY CAST(json_extract(payload, '$.source_sequence') AS INTEGER) DESC, sequence DESC LIMIT 1", (ticket_id,)).fetchone()
        summary = json.loads(last_round['payload']) if last_round else {}
        ticket['latest_analysis'] = summary.get('analysis', '待分析；不是已确认根因')
        ticket['analysis_note'] = 'AI分析与建议，须结合证据核实；不是执行记录。'
        ticket['latest_task_id'] = summary.get('task_id')
        ticket['latest_executed_tools'] = [e['payload']['tool_result']['tool'] for e in summary.get('evidence', [])]
        ticket['note'] = NOTE
        return {'ticket': ticket}

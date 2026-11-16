"""本地任务档案：每轮一条记录，不是 LangGraph checkpoint 或恢复授权。"""

from contextlib import closing
from datetime import datetime
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

TASKS_PATH = Path(__file__).resolve().parent / "data" / "tasks.sqlite3"
TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted"})


class TaskStoreError(Exception):
    """存储不可用或任务状态不允许写入；调用方必须停止后续操作。"""


def now():
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def validate_task_id(task_id):
    if not isinstance(task_id, str) or re.fullmatch(r"task_[0-9a-f]{32}", task_id) is None:
        raise TaskStoreError("任务编号格式无效，请从任务列表复制完整编号。")


class TaskStore:
    """短事务、显式提交；状态与事件同一事务，不存储模型请求/密钥配置。"""

    def __init__(self, path=None, sanitizer=None):
        self.connection = None
        self.sanitizer = sanitizer or (lambda value: value)
        try:
            path = Path(path) if path is not None else TASKS_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(path, timeout=2)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            with closing(self.connection.cursor()) as cursor:
                version = cursor.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, 1):
                    raise TaskStoreError("任务档案版本不兼容，未启动聊天。")
            with self.connection:
                self.connection.execute("""CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, question TEXT NOT NULL, origin TEXT NOT NULL,
                    status TEXT NOT NULL, phase TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
                self.connection.execute("""CREATE TABLE IF NOT EXISTS task_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(id),
                    kind TEXT NOT NULL, recorded_at TEXT NOT NULL, payload TEXT NOT NULL)""")
                self.connection.execute("CREATE INDEX IF NOT EXISTS task_events_task ON task_events(task_id, sequence)")
                self.connection.execute("PRAGMA user_version = 1")
        except (sqlite3.Error, OSError, TaskStoreError) as error:
            self.close()
            if isinstance(error, TaskStoreError):
                raise
            raise TaskStoreError("任务档案无法打开，请检查 data 文件夹、磁盘和写入权限。") from None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _clean(self, value):
        if isinstance(value, str):
            return self.sanitizer(value)
        if isinstance(value, dict):
            return {self.sanitizer(str(key)): self._clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._clean(item) for item in value]
        return value

    def _payload(self, value):
        return json.dumps(self._clean(value), ensure_ascii=False, allow_nan=False)

    def create(self, question, origin="ai_chat"):
        if not isinstance(question, str) or not question.strip() or origin not in {"ai_chat", "local_command"}:
            raise TaskStoreError("任务问题或来源无效。")
        task_id, timestamp = "task_" + uuid4().hex, now()
        try:
            question = self.sanitizer(question)
            payload = self._payload({"question": question, "origin": origin})
            with self.connection:
                self.connection.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                                        (task_id, question, origin, "running", "created", timestamp, timestamp))
                self.connection.execute("INSERT INTO task_events(task_id, kind, recorded_at, payload) VALUES (?, ?, ?, ?)",
                                        (task_id, "created", timestamp, payload))
        except (sqlite3.Error, TypeError, ValueError):
            raise TaskStoreError("任务建档失败；本轮未请求模型，也未执行工具。") from None
        return TaskRecord(self, task_id)

    def append(self, task_id, kind, payload, *, status="running", once=False):
        validate_task_id(task_id)
        if status not in TERMINAL_STATUSES | {"running"} or not isinstance(kind, str) or not kind:
            raise TaskStoreError("任务事件或状态无效。")
        timestamp = now()
        try:
            encoded = self._payload(payload)
            with self.connection:
                cursor = self.connection.execute(
                    "UPDATE tasks SET status=?, phase=?, updated_at=? WHERE id=? AND status='running'",
                    (status, kind, timestamp, task_id))
                if cursor.rowcount != 1:
                    raise TaskStoreError("任务不存在或本轮已结束，不能追加或重放操作。")
                # UPDATE 已取得写锁：唯一操作标记的检查与插入处于同一事务，防止重放。
                if once and self.connection.execute(
                    "SELECT 1 FROM task_events WHERE task_id=? AND kind=?", (task_id, kind)
                ).fetchone() is not None:
                    raise TaskStoreError("此操作已进入过执行流程；不能重放，请重新检查当前状态。")
                self.connection.execute("INSERT INTO task_events(task_id, kind, recorded_at, payload) VALUES (?, ?, ?, ?)",
                                        (task_id, kind, timestamp, encoded))
        except (sqlite3.Error, TypeError, ValueError):
            raise TaskStoreError("任务事件保存失败；已停止本轮后续操作，请勿假定此前操作未执行。") from None

    def list_tasks(self):
        try:
            rows = self.connection.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC, id DESC LIMIT 20").fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error:
            raise TaskStoreError("无法读取任务列表。") from None

    def get_task(self, task_id):
        validate_task_id(task_id)
        try:
            row = self.connection.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["events"] = [dict(event) for event in self.connection.execute(
                "SELECT sequence, kind, recorded_at, payload FROM task_events WHERE task_id=? ORDER BY sequence",
                (task_id,)).fetchall()]
            for event in result["events"]:
                event["payload"] = json.loads(event["payload"])
            return result
        except (sqlite3.Error, ValueError):
            raise TaskStoreError("无法读取任务档案。") from None


class TaskRecord:
    def __init__(self, store, task_id):
        self.store = store
        self.id = task_id

    def event(self, kind, payload):
        self.store.append(self.id, kind, payload)

    def finish(self, status, payload):
        if status not in TERMINAL_STATUSES:
            raise TaskStoreError("任务结束状态无效。")
        self.store.append(self.id, status, payload, status=status)

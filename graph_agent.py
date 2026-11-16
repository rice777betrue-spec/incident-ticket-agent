"""LangGraph 初版：可持久暂停的启动申请；不自动重试副作用或恢复任意失败节点。"""
from collections import deque
from contextlib import ExitStack, contextmanager
from functools import wraps
import json
import os
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langsmith import tracing_context

import ai_check as core
import recover_service as recovery
from recover_service import RecoverySession
from task_store import TaskRecord, TaskStore, TaskStoreError, validate_task_id

DATA_DIR = Path(__file__).resolve().parent / "data"
GRAPH_VERSION = 1
GRAPH_PROMPT = core.CHAT_PROMPT + (
    "当前使用 LangGraph。启动申请会先持久暂停，用户用本地命令继续任务后才进入重新检查和本地 y 审批。"
    "继续任务命令不是批准启动。恢复时历史证据仍只代表原收集时间，不能声称一直健康。"
    "不要声称能恢复任意错误或重新执行已结束任务。"
)


class GraphRunError(Exception):
    """可显示的流程错误；绝不据此自动重试启动。"""


class AgentState(TypedDict):
    schema_version: int
    task_id: str
    messages: list[dict]
    turn: list[dict]
    used_names: list[str]
    used_ids: list[str]
    pending_calls: list[dict]
    phase: str


def clean(value):
    if isinstance(value, str):
        return core.redact_text(value)
    if isinstance(value, dict):
        return {core.redact_text(str(k)): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


@contextmanager
def exclusive_runtime(path):
    """进程生命周期文件锁；不依赖会在崩溃后残留的 PID/标志文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    locked = False
    try:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError:
            raise GraphRunError("另一个 LangGraph 会话正在使用这些任务，请先退出那个会话。") from None
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class EventSink:
    """图节点在工作线程中执行；每次事件用独立短连接，不跨线程共享档案连接。"""
    def __init__(self, path, task_id):
        self.path, self.id = path, task_id

    def event(self, kind, payload):
        with TaskStore(self.path, sanitizer=core.redact_text) as store:
            store.append(self.id, kind, payload)

    def claim_start(self):
        with TaskStore(self.path, sanitizer=core.redact_text) as store:
            store.append(self.id, "graph_start_entered", {
                "note": "仅表示已进入启动流程，不证明启动发生；不能自动重试此操作。"
            }, once=True)


def safe_node(function):
    """避免原始异常/密钥进入框架的错误 checkpoint；interrupt 节点不使用此包装。"""
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            raise GraphRunError("步骤未完成（" + type(error).__name__ + "），已停止；请查任务档案，不自动重放。") from None
    return wrapped


class GraphAgent:
    def __init__(self, root=None, session=None):
        # Studio 使用服务端检查点；仅终端入口需要本地 SQLite saver。
        from langgraph.checkpoint.sqlite import SqliteSaver
        self.root = Path(root) if root is not None else DATA_DIR
        self.stack = ExitStack()
        try:
            self.stack.enter_context(exclusive_runtime(self.root / "graph-runtime.lock"))
            self.store_path = self.root / "tasks.sqlite3"
            self.store = self.stack.enter_context(TaskStore(self.store_path, sanitizer=core.redact_text))
            saver = self.stack.enter_context(SqliteSaver.from_conn_string(str(self.root / "graph-checkpoints.sqlite3")))
            saver.setup()
            self._approval = None  # 仅内存的一次性本地输入；从不进入图状态。
            self.session = session if session is not None else RecoverySession()
            self.stack.callback(self.session.close)
            builder = StateGraph(AgentState)
            builder.add_node("analyze", self._analyze)
            builder.add_node("read_tools", self._read_tools)
            builder.add_node("await_start", self._await_start)
            builder.add_node("start_service", self._start_service)
            builder.add_edge(START, "analyze")
            builder.add_conditional_edges("analyze", lambda state: state["phase"], {
                "read_tools": "read_tools", "await_start": "await_start", "done": END,
            })
            builder.add_edge("read_tools", "analyze")
            builder.add_edge("await_start", "start_service")
            builder.add_edge("start_service", "analyze")
            self.graph = builder.compile(checkpointer=saver)
        except BaseException:
            self.stack.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stack.close()

    def _sink(self, state):
        return EventSink(self.store_path, state["task_id"])

    @staticmethod
    def _config(task_id):
        return {"configurable": {"thread_id": task_id}, "recursion_limit": 20}

    def _tool_catalog(self):
        return core.CHAT_TOOLS

    def _readonly_names(self):
        return core.ALLOWED_TOOLS

    def _validate_calls(self, message, used_names, used_ids, allowed_names):
        return core.validate_calls(message, used_names, used_ids, allowed_names)

    def _allowed_names(self):
        return self._readonly_names() | {core.START_TOOL_NAME}

    def _call_phase(self, calls):
        return "await_start" if calls[0]["function"]["name"] == core.START_TOOL_NAME else "read_tools"

    def _run_read_tool(self, call, state):
        return core.call_tool(call["function"]["name"])

    @safe_node
    def _analyze(self, state):
        names = set(state["used_names"])
        at_limit = len(names) >= core.MAX_TOOL_STEPS
        tools = self._tool_catalog() if at_limit else [
            t for t in self._tool_catalog() if t["function"]["name"] not in names]
        self._sink(state).event("analyzing", {"tools_used": len(names), "engine": "langgraph"})
        response = clean(core.request_message(list(state["messages"]), tools=tools,
                                             tool_choice="none" if at_limit else "auto"))
        if at_limit or response.get("tool_calls") in (None, []):
            reply = core.redact_text(core.report_reply(response, state["messages"]))
            final = {"role": "assistant", "content": reply}
            return {"turn": state["turn"] + [final], "messages": state["messages"] + [final],
                    "phase": "done", "pending_calls": []}
        calls = self._validate_calls(response, names, set(state["used_ids"]),
                                     self._allowed_names())
        assistant = {"role": "assistant", "content": None, "tool_calls": calls}
        return {"pending_calls": calls, "turn": state["turn"] + [assistant],
                "messages": state["messages"] + [assistant],
                "phase": self._call_phase(calls)}

    def _result(self, state, call, outcome):
        evidence = core.prepare_evidence(outcome)
        self._sink(state).event("tool_result", evidence)
        print("[实际结果] " + json.dumps(evidence, ensure_ascii=False), flush=True)
        return {"role": "tool", "tool_call_id": call["id"],
                "content": json.dumps(evidence, ensure_ascii=False)}

    @safe_node
    def _read_tools(self, state):
        # 再验一次；不能从损坏的待执行状态绕过白名单或把启动混入只读批次。
        calls = self._validate_calls({"tool_calls": state["pending_calls"]}, set(state["used_names"]),
                                    set(state["used_ids"]), self._readonly_names())
        results = []
        for call in calls:
            name = call["function"]["name"]
            self._sink(state).event("tool_requested", {"tool": name, "call_id": call["id"]})
            results.append(self._result(state, call, self._run_read_tool(call, state)))
        return {"turn": state["turn"] + results, "messages": state["messages"] + results,
                "used_names": state["used_names"] + [c["function"]["name"] for c in calls],
                "used_ids": state["used_ids"] + [c["id"] for c in calls], "pending_calls": [], "phase": "analyze"}

    @staticmethod
    def _await_start(state):
        # 节点恢复时会从开头重跑，所以 interrupt 前没有 I/O、记账或启动副作用。
        answer = interrupt({"kind": "start_confirmation_gate", "task_id": state["task_id"],
                            "message": "已暂停。继续任务只进入新的检查与本地 y 审批，不代表批准。"})
        if answer != "request_fresh_local_confirmation":
            raise GraphRunError("恢复信号无效；未批准、未启动。")
        return {"phase": "start_service"}

    @safe_node
    def _start_service(self, state):
        calls = core.validate_calls({"tool_calls": state["pending_calls"]}, set(state["used_names"]),
                                   set(state["used_ids"]), {core.START_TOOL_NAME})
        call = calls[0]
        sink = self._sink(state)
        if type(self._approval) is not bool:
            raise GraphRunError("缺少本次本地确认，禁止从检查点重放启动。")
        # 先持久写唯一标记。即使节点重放/进程崩溃，也不能第二次进入本地启动流程。
        sink.claim_start()
        sink.event("tool_requested", {"tool": core.START_TOOL_NAME, "call_id": call["id"]})
        result = core.approved_start_result(self.session, sink, approver=self._take_approval)
        message = self._result(state, call, {
            "tool": core.START_TOOL_NAME, "call_status": "error" if result["status"] == "error" else "completed",
            "result": result,
        })
        return {"turn": state["turn"] + [message], "messages": state["messages"] + [message],
                "used_names": state["used_names"] + [core.START_TOOL_NAME],
                "used_ids": state["used_ids"] + [call["id"]], "pending_calls": [], "phase": "analyze"}

    def start(self, question, history=()):
        if not isinstance(question, str) or not question.strip() or len(question) > core.MAX_INPUT_CHARS:
            raise GraphRunError("请输入1至2000字的任务。")
        question = core.redact_text(question.strip())
        task = self.store.create(question)
        print("[任务编号] " + task.id, flush=True)
        turn = [{"role": "user", "content": question}]
        messages = [{"role": "system", "content": GRAPH_PROMPT}]
        for previous in list(history)[-core.MAX_HISTORY_TURNS:]:
            messages.extend(clean(previous))
        state = {"schema_version": GRAPH_VERSION, "task_id": task.id, "messages": messages + turn,
                 "turn": turn, "used_names": [], "used_ids": [], "pending_calls": [], "phase": "analyze"}
        task.event("graph_created", {"schema_version": GRAPH_VERSION})
        return self._invoke(task, state)

    def _pending_task(self, task_id):
        validate_task_id(task_id)
        task = self.store.get_task(task_id)
        if task is None or task["status"] != "running":
            raise GraphRunError("任务不存在或已结束，不能继续或重放；请另提一个新任务。")
        if not any(e["kind"] == "graph_created" and e["payload"].get("schema_version") == GRAPH_VERSION
                   for e in task["events"]):
            raise GraphRunError("这是旧版档案或不兼容任务，不是可恢复的图检查点。")
        if any(e["kind"] == "graph_start_entered" for e in task["events"]):
            raise GraphRunError("此任务已进入过启动流程，操作可能已发生；禁止自动重放，请新建只读检查核实。")
        with tracing_context(enabled=False):
            snapshot = self.graph.get_state(self._config(task_id))
        if (snapshot.values.get("schema_version") != GRAPH_VERSION or
                snapshot.values.get("task_id") != task_id or snapshot.next != ("await_start",) or
                not any(t.interrupts for t in snapshot.tasks) or
                any(t.error for t in snapshot.tasks)):
            raise GraphRunError("任务不在可安全恢复的审批等待点；不自动恢复崩溃或失败节点，请新建只读检查。")
        return TaskRecord(self.store, task_id)

    def _take_approval(self):
        # 只接受本次 resume 在主线程采集的输入，绝不从历史事件/checkpoint 取批准。
        decision, self._approval = self._approval, None
        if type(decision) is not bool:
            raise GraphRunError("本次批准已失效，不能重放。")
        return decision

    def resume(self, task_id):
        task = self._pending_task(task_id)
        try:
            task.event("graph_resume_requested", {"approval_reusable": False,
                                                 "note": "继续不是批准；旧证据仅代表原收集时刻"})
            print("[继续原任务] " + task_id + "；旧证据不是当前状态，将重新检查并单独询问 y。", flush=True)
            # input 必须在主线程，否则 Ctrl+C 可能把主线程卡在等待工作线程退出。
            before = recovery.inspect_start_eligibility()
            task.event("resume_precheck", before)
            print(core.redact_text(before["message"]), flush=True)
            if before["allowed"]:
                print("只启动固定 service.py；退出会话会清理本次进程。不停止或重启已有服务。", flush=True)
                try:
                    self._approval = recovery.ask_for_approval()
                except EOFError:
                    self._approval = False
            else:
                self._approval = False
            task.event("resume_user_decision", {"approved": self._approval, "source": "local_input" if before["allowed"] else "ineligible",
                                                "approval_reusable": False})
        except (KeyboardInterrupt, EOFError):
            self._approval = None
            core.finish_task_safely(task, "interrupted", {"reason": "本地确认中断，没有进入启动流程"})
            raise
        except Exception:
            self._approval = None
            core.finish_task_safely(task, "failed", {"reason": "恢复前检查或审批记录失败；未进入启动流程"})
            raise GraphRunError("恢复前检查或审批记录失败；未进入启动流程。") from None
        try:
            return self._invoke(task, Command(resume="request_fresh_local_confirmation"))
        finally:
            self._approval = None

    def cancel(self, task_id):
        task = self._pending_task(task_id)
        task.finish("interrupted", {"reason": "用户取消等待中的任务；未进入启动流程"})

    def _invoke(self, task, value):
        try:
            with tracing_context(enabled=False):
                # 同步提交每步检查点后才继续下个节点，不采用后台异步持久化。
                result = self.graph.invoke(value, self._config(task.id), durability="sync")
            if result.get("__interrupt__"):
                task.event("graph_paused", {"reason": "等待用户继续后进行新的本地审批", "approval_reusable": False})
                print("[已暂停] 可退出后再来。输入：继续任务 " + task.id, flush=True)
                print("若不再处理，输入：取消任务 " + task.id, flush=True)
                return {"task_id": task.id, "status": "paused"}
            if result.get("phase") != "done":
                raise GraphRunError("流程未正常结束，不能生成成功报告。")
            report = core.build_agent_report(result["turn"])
            report["task_id"] = task.id
            task.finish("completed", {"report": report})
            if not result["used_names"]:
                print("[本轮没有执行检查、读取日志或申请启动；历史结果不代表当前状态。]", flush=True)
            print("AI：" + report["analysis"], flush=True)
            return {"task_id": task.id, "status": "completed", "report": report, "turn": result["turn"]}
        except (KeyboardInterrupt, EOFError):
            core.finish_task_safely(task, "interrupted", {"reason": "执行期间被中断；不自动重试或复用批准"})
            raise
        except Exception as error:
            core.finish_task_safely(task, "failed", {"reason": "图流程或落盘失败", "error_type": type(error).__name__})
            raise GraphRunError("本轮未完成，已有证据可查。操作可能已发生，不自动续办或重试；可新建只读检查。") from None


def graph_chat_loop():
    history = deque(maxlen=core.MAX_HISTORY_TURNS)
    latest_report = None
    print("LangGraph 对话已启动：正常描述任务；任务列表 / 查看任务 编号 / 继续任务 编号 / 取消任务 编号。")
    print("命令可带或不带 /。启动申请会暂停；继续只进入新检查和本地 y 审批，不沿用旧授权。")
    print("data 中保存档案与图检查点（含基础遮挡后的上下文、证据和分析，非加密）；不要输入密钥。")
    print("保存报告导出本次会话最近的完整报告；清空只清聊天记忆；退出会清理本会话创建的服务。")
    print("旧版任务只能查看；初版只支持从审批等待点继续，不支持任意失败节点恢复。")
    try:
        with GraphAgent() as engine:
            while True:
                try:
                    text = input("\n你：").strip()
                except EOFError:
                    return True
                if not text:
                    continue
                if text.lower() in {"退出", "exit", "quit"}:
                    return True
                try:
                    if core.show_task_command(text, engine.store):
                        continue
                    command = text.lstrip("/")
                    if command == "清空":
                        history.clear()
                        latest_report = None
                        print("聊天记忆已清空；任务档案、检查点和报告未删除。")
                        continue
                    if command == "保存报告":
                        if latest_report is None:
                            print("本次会话没有可保存的完整报告；可查看任务档案。")
                        else:
                            print("报告已保存：" + str(core.save_agent_report(latest_report)))
                        continue
                    parts = command.split(maxsplit=1)
                    if not parts:
                        print("请输入任务描述或命令，例如：任务列表。")
                        continue
                    if parts[0] in {"继续任务", "取消任务"}:
                        if len(parts) != 2:
                            print("用法：" + parts[0] + " 完整任务编号")
                            continue
                        latest_report = None
                        if parts[0] == "取消任务":
                            engine.cancel(parts[1].strip())
                            print("任务已取消，没有启动服务。")
                            continue
                        result = engine.resume(parts[1].strip())
                    else:
                        if command == "启动服务":
                            text = "请申请启动本项目固定测试服务。"
                        latest_report = None
                        result = engine.start(text, history)
                    if result["status"] == "completed":
                        latest_report = result["report"]
                        history.append(result["turn"])
                except (GraphRunError, TaskStoreError) as error:
                    print(core.redact_text(str(error)))
                except OSError:
                    print("文件操作未完成，请检查磁盘及权限；不要假定报告已保存。")
    except KeyboardInterrupt:
        print("\n已退出；等待中的任务可继续，执行中中断的任务不会自动重放。")
        return True
    except Exception as error:
        print("LangGraph 会话未能安全完成（" + type(error).__name__ + "）；不会自动启动或重试。")
        return False

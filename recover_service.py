"""受限服务启动：检查、人工确认、再次检查、启动及验证。"""

from datetime import datetime
import subprocess
import sys
import time
from pathlib import Path

from check_service import check_service
from diagnose_service import check_port

PROJECT_DIR = Path(__file__).resolve().parent


def ask_for_approval():
    answer = input("是否允许启动测试服务？输入 y 同意，其他输入取消：")
    return answer.strip().lower() == "y"


def inspect_start_eligibility():
    """只读检查：端口明确拒绝连接时，才允许尝试启动固定测试服务。"""
    health = check_service()
    if health["healthy"]:
        return {"allowed": False, "health": health, "port_state": None,
                "message": "无需修复，不重复启动。"}
    port_state = check_port()
    messages = {
        "open": ("端口可以连接，但健康检查失败；本项目启动工具不符合执行条件，"
                 "不重复启动，也不停止或重启现有进程。"
                 "这不代表人工重启有效或无效，其效果尚未验证；请先检查接口和日志。"),
        "timeout": "端口检查超时，尚不能确认原因；不启动，请继续排查。",
        "unknown": "端口检查发生其他连接错误；不启动，请继续排查。",
        "refused": "端口连接被拒绝，可以尝试启动本项目的测试服务。",
    }
    return {"allowed": port_state == "refused", "health": health, "port_state": port_state,
            "message": messages.get(port_state, "端口状态无法识别；不启动，请继续排查。")}


def can_start_service():
    """保留独立恢复脚本的布尔接口。"""
    evidence = inspect_start_eligibility()
    print(evidence["health"]["message"])
    print(evidence["message"])
    return evidence["allowed"]


def start_service():
    # 固定运行本项目文件，不接受用户输入的任意命令。
    # 日志写到文件，子进程不另开窗口，也不抢占当前终端的输入。
    with (PROJECT_DIR / "service.log").open("a", encoding="utf-8") as log:
        return subprocess.Popen(
            [sys.executable, "-u", str(PROJECT_DIR / "service.py")],
            cwd=PROJECT_DIR,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


def wait_until_healthy(process):
    # 启动需要时间：最多检查 10 次，而不是执行命令后立即宣布成功。
    for _ in range(10):
        if process.poll() is not None:
            return False
        if check_service()["healthy"] and process.poll() is None:
            return True
        time.sleep(0.3)
    return False


class RecoverySession:
    """聊天只管理自己创建的子进程，绝不按端口查杀别人的服务。"""

    def __init__(self):
        self.process = None

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            # 仅用于收回本会话创建的进程，不操作外部监听进程。
            self.process.kill()
            self.process.wait(timeout=3)
        self.process = None
        print("本次会话启动的测试进程已结束。")

    def start_with_approval(self, observer=None, approver=None):
        # 先落盘意图，再跨越审批/副作用边界；保存失败必须抛出，不能继续启动。
        def record(kind, payload):
            if observer is not None:
                observer(kind, payload)

        keep_running = False
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    keep_running = True
                    return {"status": "already_owned", "message": "本次会话启动的进程仍在运行，不重复启动；健康状态需另行检查。"}
                self.close()
            before = inspect_start_eligibility()
            record("start_precheck", before)
            if not before["allowed"]:
                return {"status": "ineligible", "checks": [before], "message": before["message"]}
            print(before["health"]["message"])
            print(before["message"])
            print("只启动本项目固定的 service.py，不重启已有服务；退出对话或按 Ctrl+C 时会停止本次启动的进程。")
            record("awaiting_approval" if approver is None else "consuming_fresh_local_approval",
                   {"operation": "start_fixed_demo_service", "approval_reusable": False})
            approved = ask_for_approval() if approver is None else approver()
            record("approval_decision", {"approved": approved, "source": "local_input", "approval_reusable": False})
            if not approved:
                return {"status": "cancelled", "checks": [before], "message": "你未确认，已取消；没有启动服务。"}
            after = inspect_start_eligibility()
            record("start_recheck", after)
            if not after["allowed"]:
                return {"status": "state_changed", "checks": [before, after],
                        "message": "确认后重新检查，已不符合启动条件；未启动。" + after["message"]}
            record("executing", {"operation": "start_fixed_demo_service", "note": "仅为执行意图，不证明进程已创建"})
            self.process = start_service()
            record("verifying", {"note": "启动调用已返回，健康验证尚未完成"})
            if not wait_until_healthy(self.process):
                return {"status": "verification_failed", "checks": [before, after],
                        "message": "已尝试启动，但未通过健康验证；将清理本次启动的进程，不能宣告恢复成功。"}
            # 启动前的两次检查不是启动后证据。额外复查并保留真实响应，
            # 不占模型工具额度，即使启动是第3个工具也必须完成。
            health = check_service()
            post_start_check = {
                "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "health": health,
                "process_running": self.process.poll() is None,
            }
            record("post_start_check", post_start_check)
            if not health["healthy"] or not post_start_check["process_running"]:
                return {"status": "verification_failed", "checks": [before, after],
                        "post_start_check": post_start_check,
                        "message": "启动后复查未通过：健康接口异常或本次进程已退出；将清理本次进程，不能宣告恢复成功。"}
            keep_running = True
            return {"status": "verified", "checks": [before, after],
                    "post_start_check": post_start_check,
                    "message": "启动后复查通过：本次启动的进程仍在运行，/health 返回正常。可继续对话；退出时会清理本次进程。"}
        finally:
            if not keep_running:
                self.close()


def main():
    process = None
    try:
        if not can_start_service():
            return
        if not ask_for_approval():
            print("已取消，不执行任何操作。")
            return
        # 用户输入期间，服务可能已被别人启动，因此执行前再检查一次。
        if not can_start_service():
            return
        process = start_service()
        if not wait_until_healthy(process):
            print("未验证到恢复成功，将清理本次启动的进程。请查看 service.log。")
            return
        print("恢复验证通过：本次启动进程仍在运行，/health 返回预期的正常状态。")
        print("保持此窗口打开；按 Ctrl+C 结束演示并停止本次启动的测试服务。")
        # Windows 上不无限阻塞在 wait() 中，定期回到 Python 处理 Ctrl+C。
        while process.poll() is None:
            time.sleep(0.2)
        print("测试进程已退出；当前健康状态需要重新检查。")
    except (KeyboardInterrupt, EOFError):
        print("\n已结束操作。")
    except OSError as error:
        print(f"操作失败，不能宣告修复成功：{error}")
    finally:
        # 只清理本次创建的进程，不按端口或进程名称结束其他服务。
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait()
            print("本次启动的测试进程已结束。")


if __name__ == "__main__":
    main()

"""读取固定服务日志的末尾记录，只收集证据。"""

import json
from collections import deque
from pathlib import Path

# 固定读取本项目的日志，不因终端所在目录不同而读错文件。
LOG_PATH = Path(__file__).resolve().parent / "service.log"


def read_service_log():
    """返回最近 20 行；日志是历史记录，不代表服务当前一定正常。"""
    try:
        with LOG_PATH.open("r", encoding="utf-8", errors="replace") as log:
            # 逐行读取，只保留最后 20 行。
            lines = [line.rstrip("\r\n") for line in deque(log, maxlen=20)]
    except FileNotFoundError:
        return {
            "status": "missing",
            "lines": [],
            "message": "尚未找到服务日志，不能据此判断服务是否运行。",
        }
    except OSError as error:
        return {
            "status": "error",
            "lines": [],
            "message": f"日志读取失败：{error}",
        }

    return {
        "status": "ok" if lines else "empty",
        "lines": lines,
        "message": "日志仅作历史证据；当前状态仍需调用健康检查确认。",
    }


if __name__ == "__main__":
    print(json.dumps(read_service_log(), ensure_ascii=False, indent=2))

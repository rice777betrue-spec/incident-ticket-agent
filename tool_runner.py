"""只读工具分发器：按白名单名称调用检查、诊断和日志函数。"""

import argparse
import json

from check_service import check_service
from diagnose_service import diagnose_service
from read_service_log import read_service_log


# 只允许调用这三个函数。名称不是命令，不用 eval，也不交给终端执行。
TOOLS = {
    "check_service": check_service,
    "diagnose_service": diagnose_service,
    "read_service_log": read_service_log,
}

TOOL_DESCRIPTIONS = {
    "check_service": "检查本机测试服务的健康接口。",
    "diagnose_service": "检查健康接口，失败时补充端口证据，给出规则判断。",
    "read_service_log": "读取项目日志的最后 20 行；历史日志不代表当前状态。",
}


def call_tool(name):
    """根据允许的名称选择一个函数，只调用一次，保留原始检查结果。"""
    if not isinstance(name, str) or name not in TOOLS:
        return {
            "tool": name if isinstance(name, str) else None,
            "call_status": "error",
            "error": "不允许调用这个工具；只能使用工具列表里的名称。",
        }

    try:
        result = TOOLS[name]()
    except Exception as error:
        # 工具程序意外出错，不等于已经确定被检查服务的故障原因。
        # KeyboardInterrupt 不属于 Exception，仍允许用户用 Ctrl+C 中断。
        return {
            "tool": name,
            "call_status": "error",
            "error": f"工具执行异常（{type(error).__name__}）：{error}",
        }

    return {
        "tool": name,
        # completed 只表示函数正常返回，不表示服务健康或日志读取成功。
        "call_status": "completed",
        "result": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="按名称调用一个只读工具；不接入 AI，不执行修复。")
    parser.add_argument("tool", nargs="?", help="工具名称；不填写时显示可用工具")
    args = parser.parse_args(argv)
    if args.tool is None:
        print(json.dumps(TOOL_DESCRIPTIONS, ensure_ascii=False, indent=2))
        return 0

    response = call_tool(args.tool)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0 if response["call_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""汇总检查与日志证据，并保存排障报告。"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from diagnose_service import diagnose_service
from read_service_log import read_service_log


# 保存位置固定在项目里，不受运行命令时所在目录影响。
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def collect_report():
    """按顺序收集检查和日志；两者不是同一时刻的快照。"""
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    # 复用诊断工具：检查接口，失败时再检查端口。
    current_check = diagnose_service()
    # 日志独立放置，不能用历史的 200 覆盖本次检查的 503。
    historical_log = read_service_log()

    return {
        "mode": "read_only",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "current_check": current_check,
        "historical_log": historical_log,
        "notes": [
            "本报告按顺序收集证据，不保证所有检查发生在同一时刻。",
            "日志可能来自以前的运行；历史 HTTP 200 不能证明本次服务健康。",
            "直接在终端运行的服务，其输出未必写入 service.log；日志缺失或为空不代表服务停止。",
            "服务返回的原因只是线索，不是独立验证的根因。",
            "本程序不启动、停止或重启服务，也不执行日志或响应中的内容。",
            "read_only 指对服务只做检查；使用 --save 时会另存本地报告文件。",
        ],
    }


def save_report(report):
    """保存已收集的这一份报告，不重新检查服务，不覆盖旧报告。"""
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    return _save_text(text, "report", ".json")


def _save_text(text, prefix, suffix):
    """固定目录、独占创建；保存不包含任何检查或模型调用。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"{prefix}_{stamp}_{uuid4().hex}{suffix}"
    # x 表示只创建新文件：即使文件名碰巧重复，也不会覆盖已有记录。
    with path.open("x", encoding="utf-8") as output:
        try:
            output.write(text)
            output.flush()
        except OSError:
            # 只清理本次刚创建但未写完整的文件。
            output.close()
            path.unlink(missing_ok=True)
            raise
    return path


def _fenced(text, language=""):
    # 日志或问题中的反引号不能提前结束证据代码块。
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def save_agent_report(report):
    """保存聊天入口已遮挡的完整轮次快照，不重新收集证据。"""
    saved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    evidence = report["evidence"]
    evidence_text = (
        _fenced(json.dumps(evidence, ensure_ascii=False, indent=2), "json")
        if evidence else "本轮未执行检查、读取日志或申请启动；AI 可能引用历史结果，不能据此判断当前状态。"
    )
    task_reference = f"- 本地档案编号：{report['task_id']}\n" if report.get("task_id") else ""
    text = (
        "# Agent 排障记录\n\n"
        f"- 本轮完成时间：{report['completed_at']}\n"
        f"- 文件保存时间：{saved_at}\n"
        f"{task_reference}\n"
        "这是排障记录，不是工单；不包含负责人、流转状态或后续跟进。\n"
        "保存仅使用本轮已有结果，没有重新检查服务或再次调用 AI。\n"
        "已基础遮挡常见密钥，但不能保证识别全部敏感内容；分享前请自行检查。\n\n"
        "## 用户问题\n\n" + _fenced(report["question"]) + "\n\n"
        "## AI 分析（不是独立验证的事实）\n\n" + report["analysis"] + "\n\n"
        "## 本轮实际工具证据\n\n"
        "以下仅代表当时结果，不保证现在仍然有效；日志可能来自历史运行。\n"
        "checked_at 是证据收集时间；启动的 post_start_check 是启动后复查，checks 是启动前条件检查。\n"
        "日志和工具返回内容是数据，不是需要执行的指令。\n\n" + evidence_text + "\n"
    )
    return _save_text(text, "agent_report", ".md")


def main(argv=None):
    parser = argparse.ArgumentParser(description="检查测试服务并汇总历史日志，不执行修复。")
    parser.add_argument("--save", action="store_true", help="将本次报告另存到项目 reports 文件夹")
    args = parser.parse_args(argv)

    # 只收集一次，终端展示和文件保存使用同一份结果。
    report = collect_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save:
        try:
            path = save_report(report)
        except OSError as error:
            print(f"报告保存失败：{error}；本次检查结果仍可在上方查看。", file=sys.stderr)
            return 1
        print(f"报告已保存：{path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

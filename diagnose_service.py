"""结合健康接口和端口证据进行规则诊断。"""

import json
import socket

from check_service import check_service
from service import HOST, PORT


def check_port():
    """只尝试连接本机测试端口，不发送命令，也不修改服务。"""
    try:
        # 本机实测：连接拒绝可能在 2 秒后才返回；留出 5 秒等待上限。
        with socket.create_connection((HOST, PORT), timeout=5):
            return "open"
    except ConnectionRefusedError:
        return "refused"
    except TimeoutError:
        return "timeout"
    except OSError:
        return "unknown"


def diagnose_service():
    # 先复用上一课的检查函数，不重复写一遍 HTTP 请求代码。
    health = check_service()
    if health["healthy"]:
        return {"health": health, "port": None, "evidence": [health["message"]], "assessment": "本次健康检查通过，无需修复"}

    # 接口不健康时，再补充端口证据。
    port_state = check_port()
    assessments = {
        "open": "端口可以连接，但先前的健康检查失败；请检查接口响应和日志",
        "refused": "端口连接被拒绝，可能没有服务监听或连接被拦截；先确认测试服务是否启动",
        "timeout": "端口连接超时，原因尚未确定，需要进一步排查",
        "unknown": "端口检查发生连接错误，暂时无法判断原因",
    }
    evidence = [health["message"], f"端口 {PORT} 检查结果：{port_state}"]
    if health.get("reported_reason"):
        evidence.append(f"服务报告的原因（未独立验证）：{health['reported_reason']}")
    dependency = health.get("reported_dependency")
    if dependency:
        evidence.append(
            "服务报告的依赖证据（未独立验证）："
            f"status={dependency['status']}, stage={dependency['stage']}, "
            f"elapsed_ms={dependency['elapsed_ms']}, timeout_seconds={dependency['timeout_seconds']}"
        )
    return {
        # 保留结构化事实，供程序生成摘要；不从日志或原因文本中猜状态。
        "health": health,
        "port": {"number": PORT, "state": port_state},
        "evidence": evidence,
        "assessment": assessments[port_state],
    }


if __name__ == "__main__":
    print(json.dumps(diagnose_service(), ensure_ascii=False, indent=2))

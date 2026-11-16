"""只读健康接口检查，不启动或停止服务。"""

import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

HEALTH_URL = "http://127.0.0.1:18080/health"

# 本地固定探测协议的解释，不从 HTTP 正文接受说明或推断修复参数。
DEPENDENCY_TIMEOUT_NOTE = (
    "本项目依赖探测字段说明（不是独立测量）：status=timeout 时，elapsed_ms 是从探测开始"
    "到超时失败并清理连接后的累计耗时，不是下游成功响应耗时，也不是 stage 阶段单独耗时。"
    "stage 只表示超时发生阶段；response_headers 表示等待响应头时超时，未测得成功收到响应头的时间。"
    "timeout_seconds 是阻塞操作的超时设置，不是整个探测的严格总时限。"
    "elapsed_ms 与 timeout_seconds 换算后的差值不能说明只差几毫秒就会成功，"
    "不能据此认定阈值过紧或给出应该调大的具体数值；应进一步核对依赖延迟和超时策略。"
    "这些数值仍由服务报告，尚未独立验证。"
)


def parse_dependency_evidence(data):
    """错误正文只保留已知状态/阶段和有界数字，不转发 URL、任意文字等字段。"""
    if not isinstance(data, dict):
        return None
    status = data.get("status")
    stage = data.get("stage")
    if not isinstance(status, str) or status not in {
        "ok", "timeout", "unavailable", "http_error", "invalid_response",
    }:
        return None
    if not isinstance(stage, str) or stage not in {
        "connect", "request", "response_headers", "response_body", "complete",
    }:
        return None
    numbers = {}
    for key, maximum in (("timeout_seconds", 3), ("elapsed_ms", 60000)):
        value = data.get(key)
        if type(value) not in (int, float) or not 0 <= value <= maximum:
            return None
        if not math.isfinite(value) or (key == "timeout_seconds" and value == 0):
            return None
        numbers[key] = value
    return {"status": status, "stage": stage, **numbers}


def check_service():
    # 1. 请求健康检查接口，最多等待 3 秒。
    try:
        with urlopen(HEALTH_URL, timeout=3) as response:
            status_code = response.status
            data = json.load(response)
    except HTTPError as error:
        result = {
            "healthy": False,
            "message": f"接口返回 HTTP 错误：{error.code}",
            "status_code": error.code,
        }
        # HTTP 出错时仍可能有响应正文；只读取少量内容作为线索。
        try:
            with error:
                body = error.read(4097)
            if len(body) > 4096:
                result["response_note"] = "错误正文超过 4096 字节，未解析。"
                return result
            data = json.loads(body)
        except (OSError, ValueError, UnicodeError):
            result["response_note"] = "错误正文读取失败或不是有效 JSON；保留 HTTP 错误码。"
            return result

        # 只提取文本 reason，不把它当成已验证根因，也不把内容当命令执行。
        reason = data.get("reason") if isinstance(data, dict) else None
        if isinstance(reason, str) and reason.strip():
            result["reported_reason"] = reason[:500]
            result["response_note"] = "原因由服务报告，尚未独立验证。"
            if len(reason) > 500:
                result["response_note"] += "原因文本仅保留前 500 个字符。"
        else:
            result["response_note"] = "错误正文未提供有效的文本 reason 字段。"
        dependency = parse_dependency_evidence(data.get("dependency")) if isinstance(data, dict) else None
        if dependency is not None:
            result["reported_dependency"] = dependency
            result["response_note"] += " 依赖状态、阶段与耗时也由服务报告，尚未独立验证。"
            if dependency["status"] == "timeout":
                result["response_note"] += " " + DEPENDENCY_TIMEOUT_NOTE
        return result
    except (URLError, OSError):
        # 连不上只是现象，不能直接认定服务没启动。
        return {"healthy": False, "message": "连接失败或超时，具体原因待排查"}
    except (ValueError, UnicodeError):
        return {"healthy": False, "message": "接口返回的内容不是有效 JSON"}

    # 2. 不仅要连得上，还要确认是预期服务，并且返回了正常状态。
    if (
        status_code == 200
        and isinstance(data, dict)
        and data.get("service") == "demo-service"
        and data.get("status") == "ok"
    ):
        return {"healthy": True, "message": "测试服务正常"}

    return {"healthy": False, "message": "接口可以访问，但响应不符合健康条件"}


if __name__ == "__main__":
    # 3. 把检查结果打印出来；以后 AI 调用的就是上面的函数。
    result = check_service()
    print(json.dumps(result, ensure_ascii=False, indent=2))

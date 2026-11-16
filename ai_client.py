"""模型HTTP适配层：发送消息与工具说明，校验响应并遮挡错误中的凭据。"""

import json
import os
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:7863/v1"
MODEL = "cn:deepseek-v4.1-flash"
TIMEOUT = 60
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_ERROR_BYTES = 8192


class ModelConnectionError(Exception):
    """给使用者看的连接错误，只附带经过遮挡的结构化错误说明。"""



def read_error_message(error, api_key):
    """只读取有限长度的 JSON 错误说明；不展示整个响应或请求头。"""
    try:
        with error:
            body = error.read(MAX_ERROR_BYTES + 1)
        if len(body) > MAX_ERROR_BYTES:
            return "错误正文过大，未展示；请在反代面板查看。"
        data = json.loads(body)
    except (OSError, ValueError, UnicodeError):
        return "没有可读取的 JSON 错误说明；请在反代面板查看。"

    message = None
    if isinstance(data, dict):
        detail = data.get("error")
        if isinstance(detail, dict):
            message = detail.get("message")
        elif isinstance(detail, str):
            message = detail
        if not isinstance(message, str) or not message.strip():
            message = data.get("message") or data.get("detail")
    if not isinstance(message, str) or not message.strip():
        return "响应未提供文本错误说明；请在反代面板查看。"

    # 先遮挡再截断，避免泄露密钥的前半段。不输出其他 JSON 字段。
    if api_key:
        message = message.replace(api_key, "[密钥已隐藏]")
    message = re.sub(r"(?i)\bbearer\s+[^\s\"'<>，。]+", "Bearer [已隐藏]", message)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[密钥已隐藏]", message)
    message = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password)\s*[:=]\s*[^\s,;\"'<>]+",
        r"\1=[已隐藏]", message,
    )
    # 防止控制字符影响终端显示；错误说明只是数据，不作为命令执行。
    message = " ".join(message.split())
    message = "".join(char for char in message if char.isprintable())
    return message[:500] + ("（后文省略）" if len(message) > 500 else "")


def request_message(messages, tools=None, tool_choice=None):
    """发送一次聊天请求；只返回消息，不执行模型提出的工具调用。"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    # 如果反代需要鉴权，在本机终端设置环境变量，不把密钥写进代码。
    api_key = os.environ.get("WORKBUDDY_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        code = error.code
        detail = read_error_message(error, api_key)
        hints = {
            400: "请求未被接受，具体原因需结合反代返回的说明判断。",
            401: "反代要求鉴权或密钥无效，请在本机设置 WORKBUDDY_API_KEY。",
            403: "请求被拒绝，请检查反代访问权限。",
            404: "请确认反代支持 /v1/chat/completions，以及模型名称是否有效。",
            429: "请求受到限流或额度限制；本程序不会自动重试。",
        }
        hint = hints.get(code, "请在反代面板检查请求错误；本程序不会自动重试。")
        raise ModelConnectionError(
            f"模型接口返回 HTTP {code}。{hint}\n反代返回的错误说明（未独立验证）：{detail}"
        ) from None
    except (URLError, OSError):
        raise ModelConnectionError(
            "模型请求连接失败或等待超时，请确认反代已启动、7863 端口可用，并查看反代状态。"
        ) from None

    if len(body) > MAX_RESPONSE_BYTES:
        raise ModelConnectionError("模型响应过大，未解析。")
    try:
        data = json.loads(body)
        message = data["choices"][0]["message"]
    except (ValueError, UnicodeError, KeyError, IndexError, TypeError):
        raise ModelConnectionError("收到响应，但不是预期的聊天 JSON 格式，需要检查反代兼容性。") from None
    if not isinstance(message, dict):
        raise ModelConnectionError("收到响应，但 message 不是有效对象。")
    return message


def ask_ai(text):
    """原来的纯文字连接测试；不提供工具。"""
    message = request_message([{"role": "user", "content": text}])
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ModelConnectionError("收到响应，但没有有效的文字回复，暂不能确认文字对话可用。")
    return content


def main():
    print(f"连接地址：{BASE_URL}")
    print(f"请求模型：{MODEL}")
    print("正在发送一条连接测试消息，不发送项目日志、不调用工具……", flush=True)
    try:
        reply = ask_ai("这是一条连接测试消息。请只回复：连接成功。不要调用任何工具。")
    except ModelConnectionError as error:
        print(f"连接测试未通过：{error}")
        return 1
    except KeyboardInterrupt:
        print("\n已取消等待。")
        return 130
    print("收到模型文字回复：")
    print(reply)
    print("文字连接测试通过；尚未接入工具调用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

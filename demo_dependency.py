"""演示服务自己管理的本机 HTTP 依赖；只用于故障复现，不访问外网。"""

import http.client
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
DEFAULT_TIMEOUT_SECONDS = 0.25


class DependencyHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(1)

    def do_GET(self):
        if self.path != "/ready":
            self.send_error(404)
            return
        self.server.request_seen.set()
        # 真实地延迟响应头；退出时唤醒，不留下长时间睡眠的工作线程。
        if self.server.stopping.wait(self.server.delay_seconds):
            return
        body = b'{"service":"demo-dependency","status":"ok"}'
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            # 调用方等待超时后会关闭连接，这是本演示中的预期行为。
            pass

    def log_message(self, *args):
        pass


class LocalDependency:
    """拥有一个随机端口和线程；close 只清理自己创建的资源，可重复调用。"""

    def __init__(self, *, delay_seconds=0):
        if (type(delay_seconds) not in (int, float)
                or not math.isfinite(delay_seconds) or not 0 <= delay_seconds <= 2):
            raise ValueError("dependency delay must be between 0 and 2 seconds")
        self.server = ThreadingHTTPServer((HOST, 0), DependencyHandler)
        # close 等待本依赖的工作线程退出，而非遗留后台线程。
        self.server.daemon_threads = False
        self.server.delay_seconds = delay_seconds
        self.server.request_seen = threading.Event()
        self.server.stopping = threading.Event()
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.05),
            name="demo-dependency", daemon=True,
        )
        self._closed = False
        try:
            self.thread.start()
        except BaseException:
            self.server.server_close()
            raise

    @property
    def port(self):
        return self.server.server_port

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.server.stopping.set()
        self.server.shutdown()  # 从 serve_forever 所在线程以外调用。
        self.server.server_close()
        self.thread.join()


def probe_dependency(port, *, timeout_seconds=DEFAULT_TIMEOUT_SECONDS):
    """只请求指定的本机依赖。超时是各阻塞操作的等待上限，不声称是全程硬期限。"""
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("invalid local dependency port")
    if (type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 1):
        raise ValueError("invalid dependency timeout")
    started = time.monotonic()
    stage = "connect"
    connection = http.client.HTTPConnection(HOST, port, timeout=timeout_seconds)
    try:
        connection.connect()
        stage = "request"
        connection.request("GET", "/ready")
        stage = "response_headers"
        with connection.getresponse() as response:
            if response.status != 200:
                status = "http_error"
            else:
                stage = "response_body"
                body = response.read(257)
                data = json.loads(body) if len(body) <= 256 else None
                status = ("ok" if isinstance(data, dict)
                          and data.get("service") == "demo-dependency"
                          and data.get("status") == "ok" else "invalid_response")
        stage = "complete"
    except TimeoutError:
        status = "timeout"
    except (OSError, http.client.HTTPException):
        status = "unavailable"
    except (ValueError, UnicodeError):
        status = "invalid_response"
    finally:
        connection.close()
    return {
        "status": status,
        "stage": stage,
        "timeout_seconds": timeout_seconds,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
    }

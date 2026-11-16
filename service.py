"""仅监听本机的集成测试目标服务，支持可控故障场景。"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from demo_dependency import LocalDependency, probe_dependency

HOST = "127.0.0.1"  # 只允许本机访问，不对外网开放。
PORT = 18080
FAULT_MODES = ("health", "config", "downstream")
DOWNSTREAM_DELAY_SECONDS = 0.6
DEFAULT_DEMO_CONFIG = {"runtime_profile": "local-demo"}


def validate_demo_config(config):
    """校验演示配置，不读取真实环境变量，也不返回任何密钥或配置值。"""
    if not isinstance(config, dict):
        return ["demo configuration must be an object"]
    if "runtime_profile" not in config:
        return ["missing required field runtime_profile"]
    if config["runtime_profile"] != "local-demo":
        return ["invalid field runtime_profile"]
    return []


class DemoHTTPServer(ThreadingHTTPServer):
    def __init__(self, *args, **kwargs):
        self.dependency = None
        super().__init__(*args, **kwargs)

    def server_close(self):
        try:
            super().server_close()
        finally:
            if self.dependency is not None:
                self.dependency.close()


def create_demo_server(*, port=PORT, fault=None):
    """只监听本机；测试可用 port=0 随机分配端口，不接触正在运行的18080。"""
    if fault is not None and fault not in FAULT_MODES:
        raise ValueError("unsupported demo fault")
    config = dict(DEFAULT_DEMO_CONFIG)
    if fault == "config":
        # 真正删掉必填字段，再由健康检查中的校验发现，而不是直接写死错误响应。
        config.pop("runtime_profile")
    # 先绑定主服务，避免主端口占用时创建一个孤立的依赖服务。
    server = DemoHTTPServer((HOST, port), DemoHandler)
    try:
        server.health_fault = fault == "health"
        server.demo_config = config
        if fault == "downstream":
            server.dependency = LocalDependency(delay_seconds=DOWNSTREAM_DELAY_SECONDS)
        return server
    except BaseException:
        server.server_close()
        raise


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """浏览器或检查工具发送 GET 请求时，执行这里的代码。"""
        if self.path == "/health":
            config_errors = validate_demo_config(
                getattr(self.server, "demo_config", DEFAULT_DEMO_CONFIG)
            )
            if config_errors:
                status_code = 503
                reason = "configuration_error: " + "; ".join(config_errors)
                result = {"service": "demo-service", "status": "error", "reason": reason}
                # 日志只打印固定字段名，不打印配置值或密钥。
                self.log_error("%s", reason)
            elif getattr(self.server, "health_fault", False):
                # 人为注入的演示故障：进程仍在运行，但健康接口返回错误。
                status_code = 503
                result = {
                    "service": "demo-service",
                    "status": "error",
                    "reason": "simulated_health_failure",
                }
            elif getattr(self.server, "dependency", None) is not None:
                # 实际调用本机依赖，按真实返回结果决定健康状态。
                evidence = probe_dependency(self.server.dependency.port)
                if evidence["status"] == "ok":
                    status_code = 200
                    result = {"service": "demo-service", "status": "ok"}
                else:
                    status_code = 503
                    reason = "downstream_" + evidence["status"]
                    result = {"service": "demo-service", "status": "error",
                              "reason": reason, "dependency": evidence}
                    self.log_error("%s stage=%s elapsed_ms=%s timeout_seconds=%s",
                                   reason, evidence["stage"], evidence["elapsed_ms"],
                                   evidence["timeout_seconds"])
            else:
                status_code = 200
                result = {"service": "demo-service", "status": "ok"}
        else:
            status_code = 404
            result = {"error": "not_found", "hint": "Visit /health"}

        body = json.dumps(result).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="本机测试服务，可手动模拟健康接口故障。")
    parser.add_argument(
        "--fault", choices=FAULT_MODES,
        help="health：健康故障；config：演示配置缺失；downstream：真实本机依赖响应超时",
    )
    args = parser.parse_args()

    server = create_demo_server(fault=args.fault)
    print(f"Demo service: http://{HOST}:{PORT}/health", flush=True)
    if server.health_fault:
        print("[DEMO FAULT] /health returns 503; the process and port remain available.", flush=True)
    if args.fault == "config":
        print("[DEMO FAULT] Required demo configuration field runtime_profile is missing.", flush=True)
    if args.fault == "downstream":
        print("[DEMO FAULT] Local dependency delays response headers beyond the caller timeout.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nService stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

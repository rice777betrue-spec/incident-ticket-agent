# 正式 Agent：密钥只保存在本次进程环境，不写文件。
param(
    [switch]$AskKey,
    [ValidateRange(1024, 65535)][int]$Port = 2025
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-studio\Scripts\python.exe'
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "端口 $Port 已占用。请在原 Studio 服务终端按 Ctrl+C 停止后，再运行此命令。"
}
$previousKey = $env:WORKBUDDY_API_KEY
$env:PYTHONUTF8 = '1'
$env:LANGSMITH_TRACING = 'false'
$env:LANGCHAIN_TRACING_V2 = 'false'
$env:LANGGRAPH_CLI_NO_ANALYTICS = '1'
Push-Location $PSScriptRoot
try {
    if ($AskKey) {
        $secureKey = Read-Host '请输入 WorkBuddy 密钥（输入隐藏，不保存）' -AsSecureString
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
        try { $env:WORKBUDDY_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim() }
        finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
            $secureKey.Dispose()
        }
    }
    if ([string]::IsNullOrWhiteSpace($env:WORKBUDDY_API_KEY)) { throw '没有读到密钥，请用 -AskKey 在本机输入。' }
    Write-Host "正式 Agent: service_agent"
    Write-Host "Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:$Port"
    Write-Host '检查结果和按需读取的日志会发给模型；任务档案本地保存。启动需在 Studio 单独审批。'
    Write-Host '关闭浏览器不会停止本地服务；在此终端按 Ctrl+C 退出。'
    & $python -m langgraph_cli dev --host 127.0.0.1 --port $Port --no-browser
} finally {
    if ($AskKey) { $env:WORKBUDDY_API_KEY = $previousKey }
    Pop-Location
}

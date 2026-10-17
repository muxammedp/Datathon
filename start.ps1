param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
try {
    $runningService = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
    if ($runningService.service -eq 'ForteVision') {
        Write-Host "ForteVision is already running: http://127.0.0.1:$Port" -ForegroundColor Green
        Write-Host 'API key: data/api-key.txt'
        exit 0
    }
} catch {}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run setup.ps1 first.' }
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PYTHONIOENCODING = 'utf-8'
& '.\.venv\Scripts\python.exe' -c "from app.main import API_KEY; print('Local API key: ' + API_KEY)"
if ($LASTEXITCODE -ne 0) { throw 'Application initialization failed.' }
Write-Host "Open http://127.0.0.1:$Port and enter the key using Access." -ForegroundColor Green
Write-Host 'Keep this window open. Press Ctrl+C to stop.'
& '.\.venv\Scripts\python.exe' -m uvicorn app.main:app --host 127.0.0.1 --port $Port --no-access-log

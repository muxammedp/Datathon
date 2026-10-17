$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
}
& '.\.venv\Scripts\python.exe' -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA installation failed.' }
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& '.\.venv\Scripts\python.exe' scripts/download_video_model.py
if ($LASTEXITCODE -ne 0) { throw 'Model download failed. Run setup.ps1 again to resume.' }
if (Test-Path -LiteralPath 'models\forte-video-lora-4b-v1\metadata.json') {
    Write-Host 'Setup complete. Run Start.cmd.' -ForegroundColor Green
} else {
    Write-Host 'Base model ready. Copy the trained adapter into models\forte-video-lora-4b-v1 before starting.' -ForegroundColor Yellow
}

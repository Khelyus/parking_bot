$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

$envText = Get-Content ".env" -Raw
if ($envText -match "TELEGRAM_BOT_TOKEN=replace-with-your-token") {
    Write-Host "Set TELEGRAM_BOT_TOKEN in .env before starting the bot." -ForegroundColor Yellow
    exit 1
}

python -m parking_bot --demo

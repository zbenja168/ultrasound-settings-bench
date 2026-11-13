# One-time setup on a new Windows laptop. From the repo folder, in PowerShell:
#     powershell -ExecutionPolicy Bypass -File setup.ps1
# Installs Python 3.12 if missing (via winget), builds .venv, runs the tests,
# and lists the capture devices so you know which --source number is the Cam Link.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Python {
    foreach ($cmd in @("py -3.12", "python")) {
        try {
            $v = & cmd /c "$cmd --version 2>&1"
            if ($v -match "Python 3\.(1[0-9])") { return $cmd }
        } catch {}
    }
    return $null
}

Write-Host "== Python" -ForegroundColor Cyan
$py = Find-Python
if (-not $py) {
    Write-Host "Python 3.10+ not found; installing Python 3.12 with winget..."
    winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements --silent
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    $py = Find-Python
    if (-not $py) { throw "Python installed but not on PATH yet. Close PowerShell, open a new one, and run setup.ps1 again." }
}
Write-Host "using: $py"

Write-Host "== Virtual environment and packages" -ForegroundColor Cyan
if (-not (Test-Path ".venv\Scripts\python.exe")) { cmd /c "$py -m venv .venv" }
& .venv\Scripts\python.exe -m pip install -q --upgrade pip
& .venv\Scripts\python.exe -m pip install -q -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host "== Tests (includes a real Venue Go frame)" -ForegroundColor Cyan
& .venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "SOME TESTS FAILED - send the output above before bench day" -ForegroundColor Red; exit 1 }

Write-Host "== Capture devices (plug the Cam Link in first; the Venue Go should show 1920x1080)" -ForegroundColor Cyan
& .venv\Scripts\python.exe scripts\list_devices.py

Write-Host ""
Write-Host "Setup done. Bench-day commands are in docs\LAB_DAY.md (section 'Venue Go commands')." -ForegroundColor Green

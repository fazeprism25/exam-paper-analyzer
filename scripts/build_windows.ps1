<#
.SYNOPSIS
    Builds the Windows desktop app (PyInstaller onedir) and the
    ExamPaperAnalyzer-Setup.exe installer (Inno Setup).

.DESCRIPTION
    Reproduces locally what .github/workflows/build-windows.yml runs on CI.
    Run from the repository root:  .\scripts\build_windows.ps1

    Requires:
      - Python with requirements.txt + pyinstaller installed
      - Inno Setup 6 (ISCC.exe) on PATH, or pass -InnoSetupCompiler
#>
param(
    [string]$InnoSetupCompiler = "ISCC.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

Write-Host "== Installing build dependency (pyinstaller) ==" -ForegroundColor Cyan
python -m pip install --upgrade pyinstaller

Write-Host "== Cleaning previous build output ==" -ForegroundColor Cyan
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$RepoRoot\build"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$RepoRoot\dist"

Write-Host "== Running PyInstaller ==" -ForegroundColor Cyan
pyinstaller "$RepoRoot\packaging\exampapersorter.spec" --noconfirm --distpath "$RepoRoot\dist" --workpath "$RepoRoot\build"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

$appDir = Join-Path $RepoRoot "dist\ExamPaperAnalyzer"
if (-not (Test-Path "$appDir\ExamPaperAnalyzer.exe")) {
    throw "Expected build output not found: $appDir\ExamPaperAnalyzer.exe"
}
Write-Host "Built: $appDir" -ForegroundColor Green

$innoCmd = Get-Command $InnoSetupCompiler -ErrorAction SilentlyContinue
if (-not $innoCmd) {
    Write-Warning "Inno Setup compiler '$InnoSetupCompiler' not found on PATH -- skipping installer step."
    Write-Warning "Install Inno Setup 6 (https://jrsoftware.org/isinfo.php) to produce ExamPaperAnalyzer-Setup.exe."
    exit 0
}

Write-Host "== Building installer with Inno Setup ==" -ForegroundColor Cyan
& $innoCmd.Source "$RepoRoot\packaging\windows\installer.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed" }

Write-Host "Done. Installer output: $RepoRoot\dist\installer\ExamPaperAnalyzer-Setup.exe" -ForegroundColor Green

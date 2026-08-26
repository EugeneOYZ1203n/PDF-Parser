# Installs the Tesseract OCR engine binary (via winget) on Windows --
# rastervec's TesseractOcrBackend (rastervec/helpers/ocr_backend.py) needs
# tesseract.exe on PATH or at one of its known fallback locations; this is
# that binary, not the pytesseract Python wrapper (already in
# requirements.txt).
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\setup_tesseract.ps1

$ErrorActionPreference = "Stop"

if (Get-Command tesseract -ErrorAction SilentlyContinue) {
    $existing = (Get-Command tesseract).Source
    Write-Host "tesseract already on PATH: $existing"
    tesseract --version
    exit 0
}

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Error "winget not found. Install Tesseract manually: https://github.com/UB-Mannheim/tesseract/wiki"
    exit 1
}

Write-Host "Installing Tesseract OCR via winget..."
winget install --id UB-Mannheim.TesseractOCR --source winget --accept-source-agreements --accept-package-agreements

$defaultPath = "C:\Program Files\Tesseract-OCR\tesseract.exe"
if (Test-Path $defaultPath) {
    Write-Host "Installed at $defaultPath"
    & $defaultPath --version
    Write-Host "rastervec's TesseractOcrBackend finds this automatically (no env var needed) --"
    Write-Host "see _TESSERACT_CANDIDATES in rastervec/helpers/ocr_backend.py."
} else {
    Write-Warning "Install finished but tesseract.exe wasn't found at the expected path ($defaultPath)."
    Write-Warning "Set RASTERVEC_TESSERACT_CMD to wherever winget put it."
}

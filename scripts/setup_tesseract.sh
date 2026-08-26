#!/usr/bin/env bash
# Installs the Tesseract OCR engine binary (via apt) on Linux/WSL --
# rastervec's TesseractOcrBackend (rastervec/helpers/ocr_backend.py) needs
# `tesseract` on PATH; this is that binary, not the pytesseract Python
# wrapper (already in requirements.txt).
#
# Usage: bash scripts/setup_tesseract.sh
set -euo pipefail

if command -v tesseract >/dev/null 2>&1; then
    echo "tesseract already on PATH: $(command -v tesseract)"
    tesseract --version
    exit 0
fi

if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get not found -- install tesseract-ocr via your distro's package manager." >&2
    exit 1
fi

echo "Installing tesseract-ocr via apt..."
sudo apt-get update
sudo apt-get install -y tesseract-ocr

echo "Installed: $(command -v tesseract)"
tesseract --version
echo "rastervec's TesseractOcrBackend finds this automatically (no env var needed)."

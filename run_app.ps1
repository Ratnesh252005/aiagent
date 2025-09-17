# PowerShell script to run the RAG Document Assistant

Write-Host "🚀 Starting RAG Document Assistant..." -ForegroundColor Green
Write-Host "=" * 50

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python first." -ForegroundColor Red
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "❌ .env file not found. Please run setup first." -ForegroundColor Red
    Write-Host "Run: python setup.py" -ForegroundColor Yellow
    exit 1
}

# Check if requirements are installed
Write-Host "🔍 Checking if packages are installed..." -ForegroundColor Blue

try {
    python -c "import streamlit, pinecone, google.generativeai, sentence_transformers, PyPDF2" 2>$null
    Write-Host "✅ All required packages are installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Some packages are missing. Installing..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Start the Streamlit app
Write-Host "🌐 Starting Streamlit application..." -ForegroundColor Blue
Write-Host "The app will open in your default browser automatically." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application." -ForegroundColor Yellow
Write-Host ""

streamlit run app.py

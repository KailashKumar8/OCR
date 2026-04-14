@echo off
cd /d "%~dp0"

if "%OCR_PROVIDER%"=="" set "OCR_PROVIDER=nvidia"

if "%OCR_API_KEY%"=="" (
  if /I "%OCR_PROVIDER%"=="openai" (
    set /p OCR_API_KEY=Enter OpenAI API key (starts with sk-): 
  ) else (
    set /p OCR_API_KEY=Enter NVIDIA API key: 
  )
)

if /I "%OCR_PROVIDER%"=="nvidia" (
  if "%OCR_MODEL%"=="" set "OCR_MODEL=nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
  if "%OCR_API_URL%"=="" set "OCR_API_URL=https://integrate.api.nvidia.com/v1/chat/completions"
) else (
  if "%OCR_MODEL%"=="" set "OCR_MODEL=gpt-4.1"
  if "%OCR_API_URL%"=="" set "OCR_API_URL=https://api.openai.com/v1/responses"
)

set "OCR_ENGINE=openai"
set "APP_URL=http://127.0.0.1:5000"

start "OCR Server" cmd /k "set OCR_PROVIDER=%OCR_PROVIDER%&& set OCR_API_KEY=%OCR_API_KEY%&& set OCR_MODEL=%OCR_MODEL%&& set OCR_API_URL=%OCR_API_URL%&& set OCR_ENGINE=%OCR_ENGINE%&& .\venv\Scripts\python.exe app.py"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$deadline = (Get-Date).AddSeconds(45);" ^
  "do {" ^
  "  try { Invoke-WebRequest -Uri '%APP_URL%' -UseBasicParsing | Out-Null; Start-Process '%APP_URL%'; exit 0 }" ^
  "  catch { Start-Sleep -Milliseconds 500 }" ^
  "} while ((Get-Date) -lt $deadline);" ^
  "Write-Host 'The OCR server did not become ready in time. Check the OCR Server window for errors.'; exit 1"

if errorlevel 1 pause

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

echo Starting OCR server...
echo Provider: %OCR_PROVIDER%
echo Model: %OCR_MODEL%
echo Wait until you see: Running on http://127.0.0.1:5000
echo Keep this window open while using the site.
.\venv\Scripts\python.exe app.py
pause

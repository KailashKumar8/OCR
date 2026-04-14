# Run This Project On A New PC (College)

## 1) Copy project folder
Copy the whole `ocr_basic` folder to the new PC.

Important folders/files to keep:
- `models/` (trained classifier model)
- `templates/`
- `uploads/` (optional, old outputs)
- `app.py`
- `openai_ocr.py`
- `requirements_college.txt`

## 2) Install Python
Install Python **3.10.x** (this project was built on `3.10.11`).

## 3) Open PowerShell in project folder
```powershell
cd D:\ocr_basic
```

## 4) Create virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 5) Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements_college.txt
```

## 6) Set OCR environment variables
```powershell
$env:OCR_PROVIDER="nvidia"
$env:OCR_API_KEY="nvapi-your-real-key"
$env:OCR_API_URL="https://integrate.api.nvidia.com/v1/chat/completions"
$env:OCR_MODEL="nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
$env:OCR_ENGINE="openai"
```

## 7) Run app
```powershell
python app.py
```
Open:
`http://127.0.0.1:5000`

## 8) Quick verify
After upload, output panel should show:
- `OCR Engine: NVIDIA API`
- `Processing Time: ... seconds`
- Openable full image links/buttons

## 9) If API error appears
- Make sure the key is real
- Start app from the same terminal where env vars were set
- Reopen browser with `Ctrl + F5`

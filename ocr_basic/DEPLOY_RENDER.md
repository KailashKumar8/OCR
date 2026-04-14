# Deploy To Render

## 1) Push this project to GitHub
Use the folder that contains `render.yaml` as repo root.

## 2) Create Render service
1. In Render, choose **New +** -> **Blueprint**.
2. Connect your GitHub repo.
3. Render will auto-read `render.yaml` and create the web service.

If you do not use Blueprint:
- Environment: `Python`
- Root Directory: `ocr_basic`
- Build Command: `pip install --upgrade pip && pip install -r requirements.txt`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180`

## 3) Add environment variables
Set these in Render service settings:
- `OCR_PROVIDER` = `openai` (or `nvidia`)
- `OCR_API_KEY` = your real API key
- `OCR_MODEL` = `gpt-4.1` (or your provider model)
- `OCR_ENGINE` = `openai`

## 4) Deploy and verify
After deploy finishes, open your Render URL and test an image upload.

## Notes
- Render free instances have ephemeral disk; files in `uploads/` are temporary.
- Classifier support (`document_classifier.py`) is optional in production. If needed, uncomment `torch` and `torchvision` in `requirements.txt`.

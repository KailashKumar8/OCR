import os
import time
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from ocr_pipeline import build_preprocess_variants
from openai_ocr import extract_text_with_chatgpt, normalize_chatgpt_ocr_text
from report_utils import create_ocr_report_pdf

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OCR_ENGINE = (os.environ.get("OCR_ENGINE", "openai") or "openai").strip().lower()
_document_classifier = None
_document_classifier_checked = False
_predict_document_type = None
_document_classifier_mtime = None

# create uploads folder if missing
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def get_document_classifier():
    global _document_classifier, _document_classifier_checked, _predict_document_type, _document_classifier_mtime

    model_path = Path("models/document_classifier.pt")
    model_mtime = model_path.stat().st_mtime if model_path.exists() else None

    should_reload = (not _document_classifier_checked) or (model_mtime != _document_classifier_mtime)
    if should_reload:
        _document_classifier_checked = True
        _document_classifier_mtime = model_mtime
        try:
            from document_classifier import load_document_classifier, predict_document_type

            _document_classifier = load_document_classifier()
            _predict_document_type = predict_document_type
        except FileNotFoundError:
            _document_classifier = None
            _predict_document_type = None
        except Exception:
            _document_classifier = None
            _predict_document_type = None
    return _document_classifier, _predict_document_type


def current_ocr_engine_label() -> str:
    # Keep backend provider details private from the frontend UI.
    return "Secure OCR Service"


def run_best_ocr(image, image_path: str, preprocess_path: str, document_hint: str = ""):
    text = extract_text_with_chatgpt(
        image_paths=[image_path],
        document_hint=document_hint,
    )
    text = normalize_chatgpt_ocr_text(text)
    preview_image = cv2.imread(preprocess_path)
    if preview_image is None:
        preview_image = image.copy()
    return text, preview_image, current_ocr_engine_label()


def render_home(**kwargs):
    context = {
        "text": "",
        "image": "",
        "original_image": "",
        "preprocessed_image": "",
        "report_file": "",
        "document_type": "",
        "document_confidence": None,
        "processing_seconds": None,
        "ocr_engine": current_ocr_engine_label(),
    }
    context.update(kwargs)
    return render_template("index.html", **context)


@app.route("/")
def home():
    return render_home()


# route to serve images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_home()

    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return render_home(text="No file selected.")

    safe_name = secure_filename(uploaded.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return render_home(text="Unsupported file type. Use PNG or JPG.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(safe_name).stem
    image_filename = f"{base_name}_{timestamp}{ext}"
    image_path = str(UPLOAD_FOLDER / image_filename)
    uploaded.save(image_path)

    image = cv2.imread(image_path)
    if image is None:
        return render_home(text="Could not read uploaded image.")

    process_started = time.perf_counter()
    predicted_document_type = ""
    predicted_document_confidence = None
    classifier, predict_document_type = get_document_classifier()
    if classifier is not None and predict_document_type is not None:
        try:
            predicted_document_type, predicted_document_confidence = predict_document_type(image, classifier)
        except Exception:
            predicted_document_type = ""
            predicted_document_confidence = None

    preprocess_variants = build_preprocess_variants(image)
    preprocess_variant = preprocess_variants.get("sharpen")
    preprocess_filename = f"{base_name}_{timestamp}_preprocessed.png"
    preprocess_path = str(UPLOAD_FOLDER / preprocess_filename)
    if preprocess_variant is not None:
        cv2.imwrite(preprocess_path, preprocess_variant)
    else:
        cv2.imwrite(preprocess_path, image)

    ocr_engine_used = current_ocr_engine_label()
    try:
        extracted_text, annotated_image, ocr_engine_used = run_best_ocr(
            image=image,
            image_path=image_path,
            preprocess_path=preprocess_path,
            document_hint=predicted_document_type,
        )
    except Exception as exc:
        return render_home(
            text=f"OCR failed: {exc}",
            document_type=predicted_document_type,
            document_confidence=predicted_document_confidence,
            processing_seconds=time.perf_counter() - process_started,
            ocr_engine=ocr_engine_used,
        )

    result_filename = f"{base_name}_{timestamp}_ocr.png"
    result_path = str(UPLOAD_FOLDER / result_filename)
    cv2.imwrite(result_path, annotated_image)

    report_filename = f"{base_name}_{timestamp}_report.pdf"
    report_path = str(UPLOAD_FOLDER / report_filename)
    try:
        create_ocr_report_pdf(
            output_path=report_path,
            original_path=image_path,
            preprocessed_path=preprocess_path,
            annotated_path=result_path,
            extracted_text=extracted_text,
        )
    except Exception as exc:
        return render_home(
            text=f"Report generation failed: {exc}",
            document_type=predicted_document_type,
            document_confidence=predicted_document_confidence,
            processing_seconds=time.perf_counter() - process_started,
            ocr_engine=ocr_engine_used,
        )

    final_text = extracted_text if extracted_text else "No readable text detected."
    processing_seconds = time.perf_counter() - process_started
    return render_home(
        text=final_text,
        image=result_filename,
        original_image=image_filename,
        preprocessed_image=preprocess_filename,
        report_file=report_filename,
        document_type=predicted_document_type,
        document_confidence=predicted_document_confidence,
        processing_seconds=processing_seconds,
        ocr_engine=ocr_engine_used,
    )


if __name__ == "__main__":
    host = (os.environ.get("HOST", "127.0.0.1") or "127.0.0.1").strip()
    port = int((os.environ.get("PORT", "5000") or "5000").strip())
    app.run(host=host, port=port, debug=False, use_reloader=False)

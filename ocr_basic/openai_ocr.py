import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from typing import Any


DEFAULT_OCR_MODEL = "gpt-4.1"
DEFAULT_NVIDIA_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
NVIDIA_CHAT_COMPLETIONS_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def _guess_mime_type(image_path: str) -> str:
    guessed, _ = mimetypes.guess_type(image_path)
    if guessed in {"image/jpeg", "image/png", "image/webp"}:
        return guessed
    return "image/jpeg"


def _build_data_url(image_path: str) -> str:
    mime_type = _guess_mime_type(image_path)
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_output_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    if isinstance(output_text, list):
        joined = "\n".join(str(x) for x in output_text if str(x).strip()).strip()
        if joined:
            return joined

    output = payload.get("output", [])
    chunks: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
                    text_part = part["text"].strip()
                    if text_part:
                        chunks.append(text_part)
    if chunks:
        return "\n".join(chunks).strip()

    choices = payload.get("choices", [])
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _extract_chat_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not isinstance(choices, list):
        return ""

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
            if chunks:
                return "\n".join(chunks).strip()
    return ""


def _build_prompt(document_hint: str | None = None) -> str:
    hint = (document_hint or "").strip()
    if hint:
        return (
            f"Extract text from this image with highest possible OCR accuracy. "
            f"The document type is likely '{hint}'. "
            "You may receive original and preprocessed versions of the same page. "
            "Use whichever version is clearest and produce one final transcription. "
            "Preserve natural reading order, line breaks, section headers, paragraph spacing, and word spacing. "
            "Do not merge words together. "
            "Output only the extracted text."
        )
    return (
        "Extract text from this image with highest possible OCR accuracy. "
        "You may receive original and preprocessed versions of the same page. "
        "Use whichever version is clearest and produce one final transcription. "
        "Preserve natural reading order, line breaks, section headers, paragraph spacing, and word spacing. "
        "Do not merge words together. "
        "Output only the extracted text."
    )


def normalize_chatgpt_ocr_text(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    cleaned = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", cleaned)
    cleaned = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", cleaned)
    cleaned = re.sub(r"([,:;])(?=[A-Za-z0-9])", r"\1 ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _resolve_provider() -> str:
    provider = (os.environ.get("OCR_PROVIDER", "openai") or "openai").strip().lower()
    return provider or "openai"


def _resolve_api_key(provider: str) -> str:
    generic_key = (os.environ.get("OCR_API_KEY", "") or "").strip()
    if generic_key:
        return generic_key
    if provider == "nvidia":
        return (os.environ.get("NVIDIA_API_KEY", "") or "").strip()
    return (os.environ.get("OPENAI_API_KEY", "") or "").strip()


def _resolve_model(provider: str, explicit_model: str | None) -> str:
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()

    generic_model = (os.environ.get("OCR_MODEL", "") or "").strip()
    if generic_model:
        return generic_model

    if provider == "nvidia":
        nvidia_model = (os.environ.get("NVIDIA_OCR_MODEL", "") or "").strip()
        return nvidia_model or DEFAULT_NVIDIA_MODEL

    openai_model = (os.environ.get("OPENAI_OCR_MODEL", "") or "").strip()
    return openai_model or DEFAULT_OCR_MODEL


def _resolve_api_url(provider: str) -> str:
    custom_url = (os.environ.get("OCR_API_URL", "") or "").strip()
    if custom_url:
        return custom_url
    if provider == "nvidia":
        return NVIDIA_CHAT_COMPLETIONS_URL
    return OPENAI_RESPONSES_URL


def _build_openai_payload(image_paths: list[str], prompt: str, selected_model: str) -> dict[str, Any]:
    content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
    for path in image_paths:
        data_url = _build_data_url(path)
        content.append({"type": "input_image", "image_url": data_url})

    return {
        "model": selected_model,
        "input": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_output_tokens": 4000,
    }


def _build_nvidia_payload(image_paths: list[str], prompt: str, selected_model: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        data_url = _build_data_url(path)
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    return {
        "model": selected_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4000,
        "temperature": 0,
    }


def extract_text_with_chatgpt(
    image_paths: str | list[str],
    document_hint: str | None = None,
    model: str | None = None,
    timeout_seconds: int = 120,
) -> str:
    provider = _resolve_provider()
    api_key = _resolve_api_key(provider)
    if not api_key:
        raise RuntimeError("OCR API key is not set. Set OCR_API_KEY (or provider-specific API key env var).")

    selected_model = _resolve_model(provider, model)
    normalized_paths = [image_paths] if isinstance(image_paths, str) else [p for p in image_paths if p]
    if not normalized_paths:
        raise RuntimeError("No image paths provided to OCR")

    prompt = _build_prompt(document_hint=document_hint)
    if provider == "openai":
        payload = _build_openai_payload(normalized_paths, prompt, selected_model)
    elif provider == "nvidia":
        payload = _build_nvidia_payload(normalized_paths, prompt, selected_model)
    else:
        raise RuntimeError(f"Unsupported OCR_PROVIDER: {provider}")

    api_url = _resolve_api_url(provider)
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider.upper()} OCR request failed: HTTP {exc.code} - {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{provider.upper()} OCR request failed: {exc.reason}") from exc

    parsed = json.loads(body)
    if provider == "nvidia":
        extracted = _extract_chat_completion_text(parsed)
    else:
        extracted = _extract_output_text(parsed)

    if not extracted:
        raise RuntimeError(f"{provider.upper()} OCR returned empty text")
    return normalize_chatgpt_ocr_text(extracted)

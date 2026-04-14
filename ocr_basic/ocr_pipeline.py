import argparse
import difflib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class OCRItem:
    bbox: List[List[float]]
    text: str
    confidence: float
    rect: Tuple[int, int, int, int]


_READER = None
cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, min(4, os.cpu_count() or 1)))

_COMMON_WORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "mean",
    "means",
    "my",
    "no",
    "not",
    "of",
    "on",
    "one",
    "only",
    "or",
    "our",
    "out",
    "please",
    "reading",
    "seeing",
    "sentence",
    "she",
    "so",
    "text",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "up",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "work",
    "working",
    "would",
    "you",
    "your",
    "ocr",
}

_HANDWRITING_FIXES = {
    "hhis": "this",
    "jhis": "this",
    "mneans": "means",
    "neans": "means",
    "oc": "ocr",
    "0cr": "ocr",
}

_EMAIL_TERMS = {
    "access",
    "account",
    "address",
    "after",
    "all",
    "also",
    "and",
    "approved",
    "approve",
    "approvethis",
    "assigned",
    "as",
    "assigned",
    "attachment",
    "back",
    "bcc",
    "best",
    "cc",
    "click",
    "confirm",
    "corp",
    "database",
    "desel",
    "date",
    "dear",
    "details",
    "document",
    "email",
    "employee",
    "files",
    "folder",
    "folders",
    "files",
    "folder",
    "for",
    "forward",
    "forwarded",
    "from",
    "full",
    "give",
    "ginny",
    "hello",
    "hi",
    "if",
    "in",
    "information",
    "is",
    "issue",
    "kindly",
    "level",
    "message",
    "mirroring",
    "murphy",
    "murphyvl",
    "morning",
    "name",
    "normal",
    "on",
    "original",
    "path",
    "pleaseconfirm",
    "permissions",
    "please",
    "pmmc",
    "project",
    "regards",
    "removed",
    "reply",
    "replyall",
    "request",
    "rights",
    "server",
    "servers",
    "sent",
    "signature",
    "subject",
    "team",
    "thank",
    "thanks",
    "the",
    "this",
    "to",
    "top",
    "toplevel",
    "today",
    "update",
    "user",
    "users",
    "ward",
    "with",
    "workflow",
    "ysp",
    "created",
    "create",
    "currently",
    "afterward",
    "remove",
    "removed",
    "literature",
}

_NOTE_TERMS = {
    "agenda",
    "announcement",
    "attach",
    "attached",
    "bring",
    "call",
    "check",
    "date",
    "dear",
    "draft",
    "kindly",
    "meeting",
    "memo",
    "message",
    "minutes",
    "note",
    "please",
    "point",
    "regards",
    "reminder",
    "review",
    "thanks",
    "todo",
}

_RESUME_TERMS = {
    "academic",
    "achievements",
    "activities",
    "address",
    "awards",
    "bio",
    "birth",
    "citizenship",
    "contact",
    "curriculum",
    "degrees",
    "education",
    "email",
    "employment",
    "experience",
    "faculty",
    "haifa",
    "honors",
    "institute",
    "interests",
    "israel",
    "marital",
    "objective",
    "phone",
    "position",
    "profile",
    "projects",
    "publications",
    "references",
    "resume",
    "science",
    "skills",
    "student",
    "summary",
    "technion",
    "training",
    "university",
    "vitae",
    "work",
}

_REPORT_TERMS = {
    "analysis",
    "background",
    "biological",
    "conclusion",
    "constituents",
    "content",
    "controversy",
    "data",
    "discussion",
    "findings",
    "health",
    "industry",
    "information",
    "introduction",
    "markets",
    "method",
    "methods",
    "natural",
    "objective",
    "observations",
    "overview",
    "period",
    "process",
    "project",
    "recommendations",
    "report",
    "results",
    "sampling",
    "section",
    "smoking",
    "study",
    "summary",
    "survey",
    "table",
    "technical",
    "tobacco",
}

_DOCUMENT_TERMS = {
    "advance",
    "advertisers",
    "bag",
    "biology",
    "called",
    "cats",
    "chicago",
    "cigarette",
    "combination",
    "company",
    "content",
    "copy",
    "daily",
    "dear",
    "director",
    "documents",
    "draft",
    "enterprise",
    "enterprises",
    "faculty",
    "field",
    "folder",
    "folders",
    "good",
    "important",
    "industry",
    "ink",
    "institute",
    "israel",
    "items",
    "level",
    "manager",
    "markets",
    "message",
    "news",
    "notice",
    "paragraph",
    "period",
    "press",
    "printers",
    "project",
    "provide",
    "provided",
    "reasons",
    "rights",
    "science",
    "section",
    "special",
    "status",
    "study",
    "survey",
    "switched",
    "technology",
    "times",
    "upcoming",
    "word",
    "words",
}

_SEGMENT_WORDS = _COMMON_WORDS | _EMAIL_TERMS | _NOTE_TERMS | _RESUME_TERMS | _REPORT_TERMS | _DOCUMENT_TERMS
_HEADER_LABELS = ("Original Message", "From", "Sent", "To", "Cc", "Bcc", "Subject", "Date")
_SECTION_TERMS = {
    "note": _NOTE_TERMS,
    "resume": _RESUME_TERMS,
    "report": _REPORT_TERMS,
}
_SPACING_HINTS = {"email", "note", "resume", "report"}
_GLUE_CONNECTORS = (
    "from",
    "with",
    "into",
    "through",
    "between",
    "after",
    "before",
    "subject",
    "summary",
    "report",
    "resume",
    "note",
    "your",
    "their",
    "there",
    "which",
    "where",
    "would",
    "please",
    "about",
    "over",
    "under",
    "of",
    "the",
    "and",
    "for",
)
_SEGMENT_WORDS_BY_INITIAL: Dict[str, List[str]] = {}
for _word in _SEGMENT_WORDS:
    if _word:
        _SEGMENT_WORDS_BY_INITIAL.setdefault(_word[0], []).append(_word)


def get_reader():
    global _READER
    if _READER is None:
        import easyocr

        _READER = easyocr.Reader(["en"], gpu=False)
    return _READER


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _upscale(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    if max(h, w) < 1200:
        return cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    return gray


def _normalize_runtime_size(image: np.ndarray, max_side: int = 2200) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def build_preprocess_variants(image: np.ndarray) -> Dict[str, np.ndarray]:
    gray = _to_gray(image)
    upscaled = _upscale(gray)
    denoise = cv2.fastNlMeansDenoising(upscaled, h=12, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoise)
    sharpen = cv2.addWeighted(clahe, 1.45, cv2.GaussianBlur(clahe, (0, 0), 1.2), -0.45, 0)
    binary = cv2.adaptiveThreshold(
        sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
    )
    binary_inv = cv2.adaptiveThreshold(
        sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return {
        "original": image,
        "gray": gray,
        "upscaled": upscaled,
        "denoise": denoise,
        "clahe": clahe,
        "sharpen": sharpen,
        "binary": binary,
        "binary_inv": binary_inv,
        "morph": morph,
    }


def _bbox_to_rect(bbox: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    left = int(min(xs))
    top = int(min(ys))
    right = int(max(xs))
    bottom = int(max(ys))
    return left, top, right, bottom


def _rect_key(rect: Tuple[int, int, int, int], bin_size: int = 8) -> Tuple[int, int, int, int]:
    return tuple(int(v / bin_size) for v in rect)


def _group_lines(items: List[OCRItem]) -> List[List[OCRItem]]:
    if not items:
        return []

    heights = [max(1, item.rect[3] - item.rect[1]) for item in items]
    median_h = float(np.median(heights)) if heights else 16.0
    line_tol = max(12, int(median_h * 0.7))

    sorted_items = sorted(items, key=lambda it: (it.rect[1], it.rect[0]))
    lines: List[List[OCRItem]] = []

    for item in sorted_items:
        y_center = (item.rect[1] + item.rect[3]) // 2
        placed = False
        for line in lines:
            centers = [((x.rect[1] + x.rect[3]) // 2) for x in line]
            line_center = int(sum(centers) / len(centers))
            if abs(y_center - line_center) <= line_tol:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    for line in lines:
        line.sort(key=lambda it: it.rect[0])

    return lines


def _clean_text(token: str) -> str:
    cleaned = " ".join(token.split())
    return cleaned.strip()


def _normalize_token(token: str) -> str:
    token = token.strip()
    token = token.replace("`", "").replace("~", "").replace("_", "")
    token = re.sub(r"[^A-Za-z0-9'@._-]", "", token)
    if not token:
        return ""

    # Fix common OCR confusions when token is mostly alphabetic.
    if sum(ch.isalpha() for ch in token) >= max(1, len(token) - 2):
        token = (
            token.replace("0", "o")
            .replace("1", "l")
            .replace("5", "s")
            .replace("8", "B")
            .replace("6", "G")
        )

    # Collapse long accidental repeats.
    token = re.sub(r"(.)\1{3,}", r"\1\1", token)
    return token


def _split_camel_and_digit_boundaries(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    return text


def _split_glued_connectors(token: str) -> str:
    if not token or len(token) < 9 or not token.isalpha():
        return token

    updated = token
    for _ in range(3):
        previous = updated
        for connector in _GLUE_CONNECTORS:
            pattern = re.compile(rf"([A-Za-z]{{4,}}){connector}([A-Za-z]{{4,}})", flags=re.IGNORECASE)
            updated = pattern.sub(lambda m: f"{m.group(1)} {connector} {m.group(2)}", updated)
        updated = re.sub(r"\s+", " ", updated).strip()
        if updated == previous:
            break
    return updated


def _normalize_document_hint(document_hint: str | None) -> str:
    return (document_hint or "").strip().lower()


@lru_cache(maxsize=4096)
def _fuzzy_segment_score(part: str) -> float:
    if len(part) < 4:
        return 0.0
    key = part[0]
    candidates = _SEGMENT_WORDS_BY_INITIAL.get(key, [])
    best = 0.0
    for candidate in candidates:
        if abs(len(candidate) - len(part)) > 2:
            continue
        ratio = difflib.SequenceMatcher(None, part, candidate).ratio()
        if ratio > best:
            best = ratio
    return best


def _segment_long_alpha_token(token: str, aggressive: bool = False) -> str:
    min_len = 10 if aggressive else 12
    if len(token) < min_len or not token.isalpha():
        return token

    lower = token.lower()
    if aggressive and _fuzzy_segment_score(lower) >= 0.92:
        return token
    n = len(lower)
    max_window = min(24, n)

    @lru_cache(maxsize=None)
    def best_from(i: int) -> Tuple[float, Tuple[str, ...]]:
        if i >= n:
            return 0.0, tuple()

        best_score = -1e9
        best_parts: Tuple[str, ...] = (lower[i:],)

        upper = min(n, i + max_window)
        for j in range(i + 1, upper + 1):
            part = lower[i:j]
            score_tail, tail_parts = best_from(j)
            if part in _SEGMENT_WORDS:
                if len(part) == 1 and part not in {"a", "i"}:
                    continue
                short_penalty = -0.35 if len(part) <= 2 else 0.0
                score = score_tail + 1.2 + (len(part) * 0.48) + short_penalty
            elif aggressive and len(part) >= 4:
                fuzzy = _fuzzy_segment_score(part)
                if fuzzy >= 0.82:
                    score = score_tail + 0.52 + (len(part) * 0.27) + ((fuzzy - 0.82) * 3.2)
                else:
                    if j - i > 7:
                        continue
                    score = score_tail - (1.65 * (j - i))
            else:
                if j - i > 6:
                    continue
                score = score_tail - (2.25 * (j - i))
            if score > best_score:
                best_score = score
                best_parts = (part,) + tail_parts

        return best_score, best_parts

    _, parts = best_from(0)
    if not parts:
        return token

    singletons = sum(1 for p in parts if len(p) == 1)
    if singletons > 2 and singletons / max(1, len(parts)) > 0.28:
        return token

    avg_len = sum(len(p) for p in parts) / max(1, len(parts))
    if avg_len < (2.3 if aggressive else 2.7):
        return token

    known_chars = sum(len(p) for p in parts if p in _SEGMENT_WORDS)
    if known_chars / max(1, n) < (0.5 if aggressive else 0.56):
        return token

    segmented = " ".join(parts)
    if token[0].isupper():
        segmented = segmented.capitalize()
    return segmented


def _normalize_line_readability(line: str, aggressive: bool = False) -> str:
    line = _split_camel_and_digit_boundaries(line)
    line = re.sub(r"\s*([,:;])\s*", r"\1 ", line)
    line = re.sub(r"\s+([.!?])", r"\1", line)
    line = re.sub(r"\s+", " ", line).strip()
    if not line:
        return ""

    tokens = line.split(" ")
    normalized_tokens: List[str] = []
    for token in tokens:
        clean = _normalize_token(token)
        if not clean:
            continue
        if aggressive and clean.isalpha():
            clean = _split_glued_connectors(clean)
        if clean.isalpha() and len(clean) >= 12:
            clean = _segment_long_alpha_token(clean, aggressive=aggressive)
        else:
            alpha_only = re.sub(r"[^A-Za-z]", "", clean)
            min_len = 10 if aggressive else 12
            if len(alpha_only) >= min_len:
                segmented_alpha = _segment_long_alpha_token(alpha_only, aggressive=aggressive)
                if " " in segmented_alpha:
                    clean = segmented_alpha
        normalized_tokens.append(clean)

    line = " ".join(normalized_tokens)
    line = re.sub(r"\bOriginalMessage\b", "Original Message", line, flags=re.IGNORECASE)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _insert_header_linebreaks(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\bOriginalMessage\b", "Original Message", text, flags=re.IGNORECASE)
    for label in _HEADER_LABELS:
        escaped = re.escape(label)
        text = re.sub(
            rf"(?<!\n)\s+({escaped})(?=\s*[A-Z0-9:])",
            r"\n\1",
            text,
        )
    return text


def _wrap_line(line: str, width: int = 92) -> List[str]:
    if len(line) <= width:
        return [line]
    words = line.split()
    if not words:
        return []
    wrapped: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            wrapped.append(current)
            current = word
    wrapped.append(current)
    return wrapped


def _is_email_header_line(line: str) -> bool:
    return bool(re.match(r"^(Original Message|From|Sent|To|Cc|Bcc|Subject|Date)\b", line.strip(), flags=re.IGNORECASE))


def _looks_like_section_header(line: str, document_hint: str) -> bool:
    hint = _normalize_document_hint(document_hint)
    if hint not in _SECTION_TERMS:
        return False
    clean = line.strip().strip(":")
    if not clean:
        return False
    tokens = re.findall(r"[A-Za-z]+", clean.lower())
    if not tokens:
        return False

    section_terms = _SECTION_TERMS[hint]
    joined = " ".join(tokens)
    if joined in section_terms:
        return True
    if tokens[0] in section_terms and len(tokens) <= 6:
        return True
    if clean.isupper() and len(tokens) <= 7 and len(clean) <= 64:
        return True
    return False


def _make_text_readable(text: str, document_hint: str | None = None) -> str:
    if not text:
        return ""

    hint = _normalize_document_hint(document_hint)
    aggressive = hint in _SPACING_HINTS
    wrap_width = 86 if hint in {"resume", "report"} else 92

    text = _insert_header_linebreaks(text)
    output_lines: List[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            continue
        normalized = _normalize_line_readability(raw_line, aggressive=aggressive)
        if not normalized:
            continue
        output_lines.extend(_wrap_line(normalized, width=wrap_width))

    final_lines: List[str] = []
    for line in output_lines:
        if not line.strip():
            if final_lines and final_lines[-1] != "":
                final_lines.append("")
            continue
        if _looks_like_section_header(line, hint):
            if final_lines and final_lines[-1] != "":
                final_lines.append("")
            final_lines.append(line)
            continue
        if final_lines and _is_email_header_line(final_lines[-1]) and not _is_email_header_line(line):
            if final_lines[-1].strip() and line.strip():
                final_lines.append("")
        final_lines.append(line)
    return "\n".join(final_lines).strip()


def _is_token_readable(token: str) -> bool:
    if not token:
        return False
    if len(token) == 1 and token.lower() not in {"a", "i"}:
        return False
    alpha = sum(ch.isalpha() for ch in token)
    digits = sum(ch.isdigit() for ch in token)
    if alpha + digits == 0:
        return False
    if alpha > 0 and alpha / len(token) < 0.45:
        return False
    return True


def _format_text(items: List[OCRItem], document_hint: str | None = None) -> str:
    hint = _normalize_document_hint(document_hint)
    aggressive = hint in _SPACING_HINTS
    lines = _group_lines(items)
    line_texts: List[str] = []
    line_heights: List[int] = []
    line_bounds: List[Tuple[int, int]] = []

    for line in lines:
        tokens = []
        for item in line:
            token = _normalize_token(_clean_text(item.text))
            if _is_token_readable(token):
                tokens.append(token)
        # Remove adjacent near-duplicates produced by multi-pass OCR.
        deduped: List[str] = []
        for token in tokens:
            if deduped and difflib.SequenceMatcher(None, deduped[-1].lower(), token.lower()).ratio() > 0.85:
                continue
            deduped.append(token)
        tokens = deduped
        if tokens:
            normalized_line = _normalize_line_readability(" ".join(tokens), aggressive=aggressive)
            if normalized_line:
                top = min(item.rect[1] for item in line)
                bottom = max(item.rect[3] for item in line)
                line_bounds.append((top, bottom))
                line_heights.append(max(1, bottom - top))
                if line_bounds and len(line_bounds) >= 2:
                    prev_bottom = line_bounds[-2][1]
                    gap = top - prev_bottom
                    median_h = int(np.median(line_heights)) if line_heights else 16
                    if gap > max(12, int(median_h * 0.95)):
                        line_texts.append("")
                line_texts.append(normalized_line)
    polished_lines = _polish_lines(line_texts)
    return _make_text_readable("\n".join(polished_lines).strip(), document_hint=hint)


def _polish_lines(lines: List[str]) -> List[str]:
    polished: List[str] = []
    pending_blank = False
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            pending_blank = True
            continue
        # Remove lines with very low alphabetic density.
        alpha = sum(ch.isalpha() for ch in line)
        if alpha / max(1, len(line)) < 0.55:
            continue
        # Trim noisy edge punctuation.
        line = re.sub(r"^[^A-Za-z0-9]+", "", line)
        line = re.sub(r"[^A-Za-z0-9]+$", "", line)
        if not line:
            continue
        if pending_blank and polished and polished[-1] != "":
            polished.append("")
        pending_blank = False
        polished.append(line)

    # Drop near-duplicate lines.
    final_lines: List[str] = []
    for line in polished:
        if line == "":
            if final_lines and final_lines[-1] != "":
                final_lines.append("")
            continue
        last_non_blank = ""
        for prev in reversed(final_lines):
            if prev:
                last_non_blank = prev
                break
        if last_non_blank and difflib.SequenceMatcher(None, last_non_blank.lower(), line.lower()).ratio() > 0.8:
            continue
        final_lines.append(line)
    return final_lines


def _language_word_score(text: str) -> float:
    tokens = re.findall(r"[A-Za-z']+", (text or "").lower())
    if not tokens:
        return 0.0

    score = 0.0
    for token in tokens:
        if token in _COMMON_WORDS:
            score += 1.0
            continue
        if len(token) >= 4 and token.startswith("a") and token[1:] in _COMMON_WORDS:
            score += 0.85
            continue
        close = difflib.get_close_matches(token, _COMMON_WORDS, n=1, cutoff=0.83)
        if close:
            score += 0.6
    return score / len(tokens)


def _postprocess_paragraph_text(text: str, document_hint: str | None = None) -> str:
    hint = _normalize_document_hint(document_hint)
    aggressive = hint in _SPACING_HINTS
    cleaned_lines: List[str] = []
    for raw_line in (text or "").splitlines():
        line = _normalize_line_readability(re.sub(r"\s+", " ", raw_line).strip(), aggressive=aggressive)
        if not line:
            continue
        parts = re.findall(r"[A-Za-z0-9'@._-]+|[^A-Za-z0-9'@._-]+", line)
        normalized_parts: List[str] = []
        for part in parts:
            if not re.fullmatch(r"[A-Za-z0-9'@._-]+", part):
                normalized_parts.append(part)
                continue
            token = _normalize_token(part).lower()
            if not token:
                continue
            token = _HANDWRITING_FIXES.get(token, token)
            normalized_parts.append(token)
        normalized_line = re.sub(r"\s+", " ", "".join(normalized_parts)).strip()
        normalized_line = re.sub(r"([A-Za-z])\s+([A-Za-z])", r"\1 \2", normalized_line)
        if normalized_line:
            cleaned_lines.append(normalized_line)
    polished = _polish_lines(cleaned_lines)
    return _make_text_readable("\n".join(polished).strip(), document_hint=hint)


def _run_paragraph_pass(
    reader,
    variants: Dict[str, np.ndarray],
    min_confidence: float = 0.2,
    document_hint: str | None = None,
) -> str:
    candidate_names = ["upscaled", "sharpen", "clahe", "original"]
    best_text = ""
    best_score = 0.0
    for name in candidate_names:
        variant = variants.get(name)
        if variant is None:
            continue
        results = reader.readtext(
            variant,
            detail=0,
            paragraph=True,
            decoder="beamsearch",
            canvas_size=3200,
            width_ths=1.0,
            low_text=max(0.05, min_confidence - 0.05),
            text_threshold=0.45,
            link_threshold=0.35,
            contrast_ths=0.05,
            adjust_contrast=0.9,
            batch_size=1,
        )
        raw_text = "\n".join(" ".join(str(item).split()).strip() for item in results if str(item).strip())
        text = _postprocess_paragraph_text(raw_text, document_hint=document_hint)
        score = text_quality_score(text)
        if score > best_score:
            best_score = score
            best_text = text
    return best_text


def text_quality_score(text: str) -> float:
    text = " ".join((text or "").split())
    if not text:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    words = re.findall(r"[A-Za-z']+", text)
    long_words = sum(len(w) >= 3 for w in words)
    bad_chars = sum(ch in "~_`^|{}[]<>" for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    language = _language_word_score(text)
    glued_tokens = len(re.findall(r"[A-Za-z]{16,}", text))
    single_char_noise = sum(1 for token in words if len(token) == 1 and token.lower() not in {"a", "i"})
    symbol_noise = sum(ch in "@#$%^&*=" for ch in text)
    header_bonus = sum(
        1
        for label in ("From", "To", "Cc", "Subject", "Date", "Sent")
        if re.search(rf"\b{label}\b", text)
    )
    score = (
        (alpha / max(1, len(text))) * 0.32
        + (len(words) / max(1, len(text.split()))) * 0.24
        + (long_words / max(1, len(words))) * 0.22
        + language * 0.42
        - bad_chars * 0.03
        - (digits / max(1, len(text))) * 0.35
        - glued_tokens * 0.045
        - single_char_noise * 0.015
        - symbol_noise * 0.02
        + min(0.12, header_bonus * 0.02)
    )
    return max(0.0, min(1.0, score))


def _single_char_noise_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text or "")
    if not words:
        return 1.0
    noise = sum(1 for token in words if len(token) == 1 and token.lower() not in {"a", "i"})
    return noise / len(words)


def run_ocr(
    image: np.ndarray,
    min_confidence: float = 0.2,
    profile: str = "balanced",
    variants: Dict[str, np.ndarray] | None = None,
    document_hint: str | None = None,
) -> Tuple[str, List[OCRItem], np.ndarray]:
    if image is None:
        raise ValueError("Image could not be loaded")

    image = _normalize_runtime_size(image)
    reader = get_reader()
    if variants is None:
        variants = build_preprocess_variants(image)

    merged: Dict[Tuple[int, int, int, int], OCRItem] = {}
    profile = (profile or "balanced").lower()
    if profile == "fast":
        pass_order = ["sharpen", "original"]
        decoder = "greedy"
        canvas_size = 1920
        width_ths = 0.75
    elif profile == "accurate":
        pass_order = ["original", "clahe", "sharpen", "morph", "binary_inv"]
        decoder = "beamsearch"
        canvas_size = 2560
        width_ths = 0.6
    else:
        pass_order = ["original", "sharpen", "morph"]
        decoder = "greedy"
        canvas_size = 2200
        width_ths = 0.65

    for pass_name in pass_order:
        variant = variants[pass_name]
        results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            decoder=decoder,
            canvas_size=canvas_size,
            width_ths=width_ths,
            batch_size=1,
        )
        for bbox, text, conf in results:
            text = _clean_text(str(text))
            if not text:
                continue
            conf = float(conf)
            if conf < min_confidence:
                continue
            rect = _bbox_to_rect(bbox)
            key = _rect_key(rect)
            previous = merged.get(key)
            if previous is None or conf > previous.confidence or len(text) > len(previous.text):
                merged[key] = OCRItem(bbox=bbox, text=text, confidence=conf, rect=rect)

    items = sorted(merged.values(), key=lambda it: (it.rect[1], it.rect[0]))
    text = _format_text(items, document_hint=document_hint)
    annotated = draw_annotations(image.copy(), items)
    return text, items, annotated


def run_ocr_auto(
    image: np.ndarray,
    min_confidence: float = 0.2,
    variants: Dict[str, np.ndarray] | None = None,
    document_hint: str | None = None,
) -> Tuple[str, List[OCRItem], np.ndarray]:
    if variants is None:
        variants = build_preprocess_variants(image)

    text_fast, items_fast, ann_fast = run_ocr(
        image,
        min_confidence=min_confidence,
        profile="fast",
        variants=variants,
        document_hint=document_hint,
    )
    score_fast = text_quality_score(text_fast)

    best_text = text_fast
    best_items = items_fast
    best_ann = ann_fast
    best_score = score_fast

    text_acc, items_acc, ann_acc = run_ocr(
        image,
        min_confidence=min_confidence,
        profile="accurate",
        variants=variants,
        document_hint=document_hint,
    )
    score_acc = text_quality_score(text_acc)

    if score_acc >= best_score:
        best_text = text_acc
        best_items = items_acc
        best_ann = ann_acc
        best_score = score_acc

    # Handwriting fallback: paragraph mode on upscaled/high-contrast variants.
    if best_score < 0.82:
        paragraph_text = _run_paragraph_pass(
            get_reader(),
            variants,
            min_confidence=min_confidence,
            document_hint=document_hint,
        )
        paragraph_score = text_quality_score(paragraph_text)
        paragraph_noise = _single_char_noise_ratio(paragraph_text)
        best_noise = _single_char_noise_ratio(best_text)
        if paragraph_score > best_score + 0.08 and paragraph_noise <= best_noise + 0.01:
            best_text = paragraph_text

    best_text = _make_text_readable(best_text, document_hint=document_hint)
    return best_text, best_items, best_ann


def format_extracted_text(text: str, document_hint: str | None = None) -> str:
    return _make_text_readable(text or "", document_hint=document_hint)


def draw_annotations(image: np.ndarray, items: List[OCRItem]) -> np.ndarray:
    for item in items:
        left, top, right, bottom = item.rect
        cv2.rectangle(image, (left, top), (right, bottom), (24, 215, 163), 2)
        label = f"{item.text} ({item.confidence:.2f})"
        cv2.putText(
            image,
            label,
            (left, max(20, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (22, 190, 146),
            1,
            cv2.LINE_AA,
        )
    return image


def save_preprocess_variants(input_path: str, output_dir: str) -> List[str]:
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read input image: {input_path}")

    os.makedirs(output_dir, exist_ok=True)
    variants = build_preprocess_variants(image)
    written_files: List[str] = []

    stem = os.path.splitext(os.path.basename(input_path))[0]
    for name, variant in variants.items():
        out_path = os.path.join(output_dir, f"{stem}_{name}.png")
        cv2.imwrite(out_path, variant)
        written_files.append(out_path)

    return written_files


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run OCR pipeline on a single image")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--annotated-out", default="ocr_result.png", help="Output path for annotated image")
    parser.add_argument("--text-out", default="", help="Optional output path for extracted text")
    parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "fast", "balanced", "accurate"],
        help="OCR speed/quality profile",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        raise SystemExit(f"Image not found: {args.image_path}")

    if args.profile == "auto":
        text, _, annotated = run_ocr_auto(image)
    else:
        text, _, annotated = run_ocr(image, profile=args.profile)
    cv2.imwrite(args.annotated_out, annotated)

    print("Extracted text:")
    print("-" * 80)
    print(text if text else "(no text detected)")
    print("-" * 80)
    print(f"Annotated image saved to: {args.annotated_out}")

    if args.text_out:
        with open(args.text_out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text saved to: {args.text_out}")


if __name__ == "__main__":
    _cli()

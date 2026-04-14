import os
import textwrap
from datetime import datetime
from typing import List

from PIL import Image, ImageDraw, ImageFont


PAGE_W = 1240
PAGE_H = 1754
MARGIN = 80
LINE_GAP = 14
FONT = ImageFont.load_default()
TITLE_FONT = ImageFont.load_default()


def _new_page() -> Image.Image:
    return Image.new("RGB", (PAGE_W, PAGE_H), "white")


def _draw_multiline(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, max_chars: int, line_height: int) -> int:
    wrapped_lines: List[str] = []
    for paragraph in (text or "").splitlines() or [""]:
        chunk = textwrap.wrap(paragraph, width=max_chars) or [""]
        wrapped_lines.extend(chunk)
    for line in wrapped_lines:
        draw.text((x, y), line, fill="black", font=FONT)
        y += line_height
    return y


def _fit_image_on_page(path: str, caption: str) -> Image.Image:
    page = _new_page()
    draw = ImageDraw.Draw(page)
    draw.text((MARGIN, MARGIN), caption, fill="black", font=TITLE_FONT)

    img = Image.open(path).convert("RGB")
    max_w = PAGE_W - 2 * MARGIN
    max_h = PAGE_H - (MARGIN * 2 + 80)
    img.thumbnail((max_w, max_h))

    x = (PAGE_W - img.width) // 2
    y = MARGIN + 50 + (max_h - img.height) // 2
    page.paste(img, (x, y))
    draw.rectangle((x - 2, y - 2, x + img.width + 2, y + img.height + 2), outline="black", width=2)
    return page


def create_ocr_report_pdf(
    output_path: str,
    original_path: str,
    preprocessed_path: str,
    annotated_path: str,
    extracted_text: str,
) -> None:
    pages: List[Image.Image] = []

    cover = _new_page()
    draw = ImageDraw.Draw(cover)
    y = MARGIN
    draw.text((MARGIN, y), "OCR Evaluation Report", fill="black", font=TITLE_FONT)
    y += 40
    draw.text((MARGIN, y), f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fill="black", font=FONT)
    y += 40
    draw.text((MARGIN, y), f"Original Image: {os.path.basename(original_path)}", fill="black", font=FONT)
    y += 30
    draw.text((MARGIN, y), f"Preprocessed Image: {os.path.basename(preprocessed_path)}", fill="black", font=FONT)
    y += 30
    draw.text((MARGIN, y), f"Annotated Image: {os.path.basename(annotated_path)}", fill="black", font=FONT)
    y += 40
    draw.text((MARGIN, y), "Extracted Text (Preview):", fill="black", font=FONT)
    y += 28
    preview = extracted_text if extracted_text else "No readable text detected."
    preview = preview[:1800]
    _draw_multiline(draw, preview, MARGIN, y, max_chars=130, line_height=18)
    pages.append(cover)

    pages.append(_fit_image_on_page(original_path, "Original Image"))
    pages.append(_fit_image_on_page(preprocessed_path, "Preprocessed Image"))
    pages.append(_fit_image_on_page(annotated_path, "OCR Annotated Image"))

    full_text = extracted_text if extracted_text else "No readable text detected."
    lines: List[str] = []
    for paragraph in full_text.splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=130) or [""])

    max_lines_per_page = 85
    idx = 0
    page_no = 1
    while idx < len(lines):
        page = _new_page()
        draw = ImageDraw.Draw(page)
        draw.text((MARGIN, MARGIN), f"Extracted Text (Page {page_no})", fill="black", font=TITLE_FONT)
        y = MARGIN + 38
        for _ in range(max_lines_per_page):
            if idx >= len(lines):
                break
            draw.text((MARGIN, y), lines[idx], fill="black", font=FONT)
            y += 18
            idx += 1
        pages.append(page)
        page_no += 1

    pages[0].save(output_path, "PDF", resolution=100.0, save_all=True, append_images=pages[1:])

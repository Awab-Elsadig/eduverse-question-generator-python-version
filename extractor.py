import fitz
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

IMAGE_MIN_WIDTH = 80
IMAGE_MIN_HEIGHT = 80


def clean_text(text: str) -> str:
    replacements = {
        "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
        " ": " ", "−": "-", "–": "-", "—": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_page_range(
    pdf_path: str,
    start_page: int,
    end_page: int,
    output_dir: Path,
    prefix: str = "img",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extract text and embedded images from a 1-indexed page range.
    Images are saved to output_dir. Returns (full_text, images_list).
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    images: List[Dict[str, Any]] = []

    start = max(0, start_page - 1)
    end = min(len(doc) - 1, end_page - 1)

    seen_xrefs: set = set()

    for page_idx in range(start, end + 1):
        page = doc[page_idx]
        page_num = page_idx + 1

        text = clean_text(page.get_text("text", sort=True))
        full_text += f"\n\n--- PAGE {page_num} ---\n{text}"

        for img_idx, img_info in enumerate(page.get_images(full=True)):
            try:
                xref = img_info[0]
                width = img_info[2]
                height = img_info[3]

                if width < IMAGE_MIN_WIDTH or height < IMAGE_MIN_HEIGHT:
                    continue
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                base_image = doc.extract_image(xref)
                img_ext = base_image["ext"]
                filename = f"{prefix}_page{page_num}_img{img_idx + 1}.{img_ext}"
                img_path = output_dir / filename

                with open(img_path, "wb") as f:
                    f.write(base_image["image"])

                images.append({
                    "filename": filename,
                    "page": page_num,
                    "width": width,
                    "height": height,
                })
            except Exception:
                continue

    doc.close()
    return full_text, images

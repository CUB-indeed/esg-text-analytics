#!/usr/bin/env python3
"""
pdf_clean_pipeline.py

PDF -> cleaned + parsable sentence-level dataset (JSONL and/or CSV).

Install:
  pip install pymupdf

Optional OCR support (for scanned PDFs):
  pip install pillow pytesseract
  # plus install the Tesseract binary on your system (outside Python)

Run:
  python pdf_clean_pipeline.py /path/to/input.pdf --out output.jsonl
  python pdf_clean_pipeline.py /path/to/input.pdf --out output.jsonl --csv output.csv
  python pdf_clean_pipeline.py /path/to/input.pdf --out output.jsonl --ocr
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional

import fitz  # PyMuPDF


# Small abbreviation list to avoid bad sentence splits on common cases.
# (You can expand this if your documents are heavy on legal/financial abbreviations.)
ABBREVIATIONS = [
    "e.g.", "i.e.", "etc.",
    "mr.", "mrs.", "ms.", "dr.", "prof.",
    "inc.", "ltd.", "co.", "corp.",
    "vs.", "fig.", "eq.", "no.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
]


@dataclass
class PageText:
    page: int
    text: str
    method: str  # "text" or "ocr"


def extract_pages(pdf_path: str, *, ocr: bool, ocr_lang: str, min_chars_per_page: int) -> List[PageText]:
    """Extract per-page text using embedded PDF text; optionally OCR pages that look empty."""
    doc = fitz.open(pdf_path)
    pages: List[PageText] = []

    # Lazy-import OCR deps only if requested
    pytesseract = None
    Image = None
    if ocr:
        try:
            import pytesseract as _pytesseract
            from PIL import Image as _Image
            pytesseract = _pytesseract
            Image = _Image
        except Exception as e:
            raise RuntimeError(
                "OCR requested (--ocr), but pytesseract/Pillow are not available. "
                "Install with: pip install pillow pytesseract"
            ) from e

    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            method = "text"

            # If requested, OCR pages that look like scans (very little embedded text)
            if ocr and len(text.strip()) < min_chars_per_page:
                pix = page.get_pixmap(dpi=300, alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = pytesseract.image_to_string(img, lang=ocr_lang)
                method = "ocr"

            pages.append(PageText(page=i + 1, text=text, method=method))
    finally:
        doc.close()

    return pages


def detect_repeated_margin_lines(
    pages: List[PageText],
    *,
    top_n: int = 4,
    bottom_n: int = 4,
    freq_ratio: float = 0.6,
) -> set[str]:
    """
    Heuristic header/footer remover:
    - Look at top_n and bottom_n non-empty lines per page
    - If a line repeats on >= freq_ratio of pages, treat it as header/footer
    """
    counts: Dict[str, int] = {}
    total = max(1, len(pages))

    def good_candidate(line: str) -> bool:
        line = line.strip()
        if not line:
            return False
        if len(line) < 3 or len(line) > 140:
            return False
        if re.fullmatch(r"\d{1,4}", line):  # pure page numbers
            return False
        return True

    for p in pages:
        lines = [ln.strip() for ln in (p.text or "").splitlines()]
        lines = [ln for ln in lines if ln]

        # Use a set per page so a short page can't double-count the same line
        margin_lines = set(lines[:top_n] + lines[-bottom_n:])
        for ln in margin_lines:
            if good_candidate(ln):
                counts[ln] = counts.get(ln, 0) + 1

    threshold = max(2, int(math.ceil(total * freq_ratio)))
    return {ln for ln, c in counts.items() if c >= threshold}


def remove_margin_lines(text: str, remove_set: set[str], *, top_n: int = 4, bottom_n: int = 4) -> str:
    """Remove detected header/footer lines only if they appear near the top/bottom of the page."""
    lines = [ln.rstrip("\n") for ln in (text or "").splitlines()]

    nonblank_idxs = [i for i, ln in enumerate(lines) if ln.strip()]
    if not nonblank_idxs:
        return text or ""

    top_idxs = nonblank_idxs[:top_n]
    bottom_idxs = nonblank_idxs[-bottom_n:]
    margin_idxs = top_idxs + bottom_idxs

    for idx in margin_idxs:
        s = lines[idx].strip()

        # remove repeated headers/footers
        if s in remove_set:
            lines[idx] = ""
            continue

        # remove page-number-only footers/headers
        if re.fullmatch(r"\d{1,4}", s):
            lines[idx] = ""
            continue

        # common footer keyword (optional, but usually helps)
        if re.search(r"\bconfidential\b", s, flags=re.IGNORECASE):
            lines[idx] = ""

    return "\n".join(lines)


def clean_text(text: str) -> str:
    """Normalize and unwrap PDF layout artifacts into readable text."""
    if not text:
        return ""

    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00A0", " ")  # non-breaking space

    # Normalize newlines
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove control chars except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

    # De-hyphenate line breaks: sustain-\nability -> sustainability
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

    # Convert single newlines (line wraps) into spaces; keep paragraph breaks (\n\n)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Whitespace cleanup
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\n+", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter that works decently on report-style text without extra downloads.
    """
    if not text:
        return []

    tmp = text

    # Protect abbreviations by replacing '.' with '<DOT>' in the matched abbreviation
    for ab in ABBREVIATIONS:
        tmp = re.sub(
            re.escape(ab),
            lambda m: m.group(0).replace(".", "<DOT>"),
            tmp,
            flags=re.IGNORECASE,
        )

    # Protect decimal numbers like 12.5
    tmp = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", tmp)

    # Split on sentence-ending punctuation followed by whitespace and a likely sentence start
    parts = re.split(r'(?<=[.!?])\s+(?=(?:["“”‘’\(\[])?[A-Z0-9])', tmp)

    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = p.replace("<DOT>", ".")
        out.append(p)

    return out


def pdf_to_records(
    pdf_path: str,
    *,
    doc_id: Optional[str],
    ocr: bool,
    ocr_lang: str,
    min_chars_per_page: int,
    min_sentence_chars: int,
) -> List[Dict]:
    pages = extract_pages(pdf_path, ocr=ocr, ocr_lang=ocr_lang, min_chars_per_page=min_chars_per_page)
    remove_set = detect_repeated_margin_lines(pages)

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_id = doc_id or base

    records: List[Dict] = []
    sent_id = 0

    for p in pages:
        without_margins = remove_margin_lines(p.text, remove_set)
        cleaned = clean_text(without_margins)

        for s in split_sentences(cleaned):
            if len(s.strip()) < min_sentence_chars:
                continue
            sent_id += 1
            records.append(
                {
                    "doc_id": doc_id,
                    "source_file": os.path.basename(pdf_path),
                    "page": p.page,
                    "sent_id": sent_id,
                    "text": s,
                    "extraction_method": p.method,  # "text" or "ocr"
                }
            )

    return records


def write_jsonl(records: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(records: List[Dict], out_path: str) -> None:
    if not records:
        # still write an empty file with header
        fieldnames = ["doc_id", "source_file", "page", "sent_id", "text", "extraction_method"]
    else:
        fieldnames = list(records[0].keys())

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF -> cleaned parsable dataset (sentence-level JSONL/CSV).")
    ap.add_argument("pdf", help="Path to input PDF")
    ap.add_argument("--out", required=True, help="Path to output JSONL file")
    ap.add_argument("--csv", default=None, help="Optional path to output CSV file")
    ap.add_argument("--doc-id", default=None, help="Optional doc_id override (default: filename stem)")

    ap.add_argument("--ocr", action="store_true", help="Use OCR for pages with too little extracted text")
    ap.add_argument("--ocr-lang", default="eng", help="Tesseract language (default: eng)")
    ap.add_argument("--min-chars-per-page", type=int, default=80, help="OCR pages if extracted text < this many chars")

    ap.add_argument("--min-sentence-chars", type=int, default=8, help="Drop very short sentence fragments")

    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"Input PDF not found: {args.pdf}")

    records = pdf_to_records(
        args.pdf,
        doc_id=args.doc_id,
        ocr=args.ocr,
        ocr_lang=args.ocr_lang,
        min_chars_per_page=args.min_chars_per_page,
        min_sentence_chars=args.min_sentence_chars,
    )

    write_jsonl(records, args.out)
    if args.csv:
        write_csv(records, args.csv)

    print(f"Done. Wrote {len(records)} sentences to: {args.out}")
    if args.csv:
        print(f"Wrote CSV to: {args.csv}")


if __name__ == "__main__":
    main()

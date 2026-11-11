#!/usr/bin/env python3
"""
testing.py
PDF -> cleaned sentence-level JSONL/CSV
"""

from __future__ import annotations
import argparse, csv, json, math, os, re, unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional
import fitz  # PyMuPDF

ABBREVIATIONS = ["e.g.", "i.e.", "etc.", "mr.", "mrs.", "ms.", "dr.", "prof.",
                 "inc.", "ltd.", "co.", "corp.", "vs.", "fig.", "eq.", "no.",
                 "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
                 "sep.", "sept.", "oct.", "nov.", "dec."]

@dataclass
class PageText:
    page: int
    text: str
    method: str


def extract_pages(pdf_path: str, *, ocr: bool, ocr_lang: str, min_chars_per_page: int) -> List[PageText]:
    doc = fitz.open(pdf_path)
    pages: List[PageText] = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        pages.append(PageText(page=i + 1, text=text, method="text"))

    doc.close()
    return pages


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00A0", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\x00-\x1F]", "", text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def split_sentences(text: str) -> List[str]:
    if not text:
        return []

    tmp = text

    for ab in ABBREVIATIONS:
        tmp = re.sub(re.escape(ab), lambda m: m.group(0).replace(".", "<DOT>"), tmp, flags=re.IGNORECASE)

    tmp = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", tmp)

    parts = re.split(r'(?<=[.!?])\s+', tmp)

    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(p.replace("<DOT>", "."))

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

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_id = doc_id or base

    records: List[Dict] = []
    sent_id = 0

    for p in pages:
        cleaned = clean_text(p.text)

        for s in split_sentences(cleaned):
            if len(s.strip()) < min_sentence_chars:
                continue
            sent_id += 1
            records.append({
                "doc_id": doc_id,
                "source_file": os.path.basename(pdf_path),
                "page": p.page,
                "sent_id": sent_id,
                "text": s,
                "extraction_method": p.method,
            })

    return records


def write_jsonl(records: List[Dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(records: List[Dict], out_path: str) -> None:
    if not records:
        return

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        for r in records:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    records = pdf_to_records(
        args.pdf,
        doc_id=None,
        ocr=False,
        ocr_lang="eng",
        min_chars_per_page=80,
        min_sentence_chars=8,
    )

    write_jsonl(records, args.out)

    if args.csv:
        write_csv(records, args.csv)

    print(f"Done: {len(records)} sentences")


if __name__ == "__main__":
    main()
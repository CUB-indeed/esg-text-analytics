import pdfplumber
import json
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PDF_DIR = Path("pdfs")
OUTPUT_DIR = Path("data/raw_text")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIAGNOSTIC_FILE = Path("data/raw_text/diagnostic_report.txt")

# ── Find column split ──────────────────────────────────────────────────────────
def find_column_split(page):
    words = page.extract_words()
    if not words:
        return page.width / 2

    page_width = page.width
    slices = 100
    slice_width = page_width / slices
    coverage = [0] * slices

    for w in words:
        x0 = float(w["x0"])
        x1 = float(w["x1"])
        for i in range(slices):
            slice_start = i * slice_width
            slice_end = slice_start + slice_width
            if x0 < slice_end and x1 > slice_start:
                coverage[i] += 1

    middle_start = int(slices * 0.3)
    middle_end = int(slices * 0.7)

    min_coverage = float('inf')
    split_slice = slices // 2

    for i in range(middle_start, middle_end):
        if coverage[i] < min_coverage:
            min_coverage = coverage[i]
            split_slice = i

    return split_slice * slice_width + slice_width / 2


# ── Column detection ───────────────────────────────────────────────────────────
def detect_columns(page):
    words = page.extract_words()
    if not words:
        return "single"

    page_mid = page.width / 2
    left_words = [w for w in words if float(w["x0"]) < page_mid]
    right_words = [w for w in words if float(w["x0"]) >= page_mid]

    left_ratio = len(left_words) / len(words)
    right_ratio = len(right_words) / len(words)

    if left_ratio > 0.2 and right_ratio > 0.2:
        return "multi"
    return "single"


# ── Single column extraction ───────────────────────────────────────────────────
def extract_single_column(page):
    text = page.extract_text()
    if text:
        return text.strip()
    return ""


# ── Multi column extraction ────────────────────────────────────────────────────
def extract_multi_column(page):
    words = page.extract_words()
    if not words:
        return ""

    split_x = find_column_split(page)

    left_words = [w for w in words if float(w["x1"]) <= split_x]
    right_words = [w for w in words if float(w["x0"]) >= split_x]

    left_words_sorted = sorted(
        left_words,
        key=lambda w: (round(float(w["top"]) / 5) * 5, float(w["x0"]))
    )
    right_words_sorted = sorted(
        right_words,
        key=lambda w: (round(float(w["top"]) / 5) * 5, float(w["x0"]))
    )

    def group_into_lines(word_list, y_tolerance=5):
        if not word_list:
            return []
        lines = []
        current_line = [word_list[0]]
        for word in word_list[1:]:
            if abs(float(word["top"]) - float(current_line[0]["top"])) <= y_tolerance:
                current_line.append(word)
            else:
                lines.append(
                    sorted(current_line, key=lambda w: float(w["x0"]))
                )
                current_line = [word]
        lines.append(sorted(current_line, key=lambda w: float(w["x0"])))
        return lines

    def lines_to_text(lines):
        return "\n".join(
            " ".join(word["text"] for word in line)
            for line in lines
        )

    left_lines = group_into_lines(left_words_sorted)
    right_lines = group_into_lines(right_words_sorted)

    left_text = lines_to_text(left_lines)
    right_text = lines_to_text(right_lines)

    return left_text + "\n\n" + right_text


# ── Process one PDF ────────────────────────────────────────────────────────────
def process_pdf(pdf_path):
    bank_name = pdf_path.stem
    pages_text = []
    diagnostic = {
        "bank": bank_name,
        "total_pages": 0,
        "single_column_pages": 0,
        "multi_column_pages": 0,
        "empty_pages": 0,
        "flagged_pages": []
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            diagnostic["total_pages"] = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                layout = detect_columns(page)

                if layout == "multi":
                    diagnostic["multi_column_pages"] += 1
                    text = extract_multi_column(page)
                else:
                    diagnostic["single_column_pages"] += 1
                    text = extract_single_column(page)

                if not text or len(text.split()) < 10:
                    diagnostic["flagged_pages"].append(page_num)
                    diagnostic["empty_pages"] += 1
                    continue

                pages_text.append({
                    "page": page_num,
                    "layout": layout,
                    "text": text
                })

    except Exception as e:
        print(f"ERROR processing {bank_name}: {e}")
        diagnostic["error"] = str(e)

    return bank_name, pages_text, diagnostic


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in pdfs/ directory.")
        return

    print(f"Found {len(pdf_files)} PDF files.\n")

    all_diagnostics = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        bank_name, pages_text, diagnostic = process_pdf(pdf_path)

        output_file = OUTPUT_DIR / f"{bank_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pages_text, f, ensure_ascii=False, indent=2)

        all_diagnostics.append(diagnostic)

        print(f"  Pages: {diagnostic['total_pages']} | "
              f"Single: {diagnostic['single_column_pages']} | "
              f"Multi: {diagnostic['multi_column_pages']} | "
              f"Flagged: {len(diagnostic['flagged_pages'])}")

    with open(DIAGNOSTIC_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("EXTRACTION DIAGNOSTIC REPORT\n")
        f.write("=" * 60 + "\n\n")

        for d in all_diagnostics:
            f.write(f"Bank: {d['bank']}\n")
            f.write(f"  Total pages     : {d['total_pages']}\n")
            f.write(f"  Single column   : {d['single_column_pages']}\n")
            f.write(f"  Multi column    : {d['multi_column_pages']}\n")
            f.write(f"  Empty/skipped   : {d['empty_pages']}\n")
            if d['flagged_pages']:
                f.write(f"  Flagged pages   : {d['flagged_pages']}\n")
            if 'error' in d:
                f.write(f"  ERROR           : {d['error']}\n")
            f.write("\n")

    print(f"\nDone. Diagnostic report saved to {DIAGNOSTIC_FILE}")


if __name__ == "__main__":
    main()
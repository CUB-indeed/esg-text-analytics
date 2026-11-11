import os
import re
import pdfplumber
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================
PDF_PATH = "/home/coder/project/Allianz_Group_Sustainability_Report_2023-web.pdf"
OUTPUT_PATH = "data/output/confidence_results.csv"


# ==============================
# TEXT EXTRACTION
# ==============================
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found")

    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i+1} ---\n" + page_text

    return text


# ==============================
# CLEAN NUMBER
# ==============================
def clean_number(value):
    if value is None:
        return None

    value = value.replace(",", "").strip()

    try:
        return float(value)
    except:
        return None


# ==============================
# ESG PATTERNS (IMPROVED)
# ==============================
ESG_PATTERNS = {
    "Scope 1": r"Scope\s*1[\s\S]{0,120}?(\d{1,3}(?:[,\.\s]\d{3})*|\d+)\s*(tCO2|tCO2e|tons|tonnes)",
    "Scope 2": r"Scope\s*2[\s\S]{0,120}?(\d{1,3}(?:[,\.\s]\d{3})*|\d+)\s*(tCO2|tCO2e|tons|tonnes)",
    "Scope 3": r"Scope\s*3[\s\S]{0,120}?(\d{1,3}(?:[,\.\s]\d{3})*|\d+)\s*(tCO2|tCO2e|tons|tonnes)"
}


# ==============================
# SCORING FUNCTIONS
# ==============================
def context_score(context):
    keywords = ["scope", "emission", "co2", "total"]
    return sum(k in context.lower() for k in keywords) / len(keywords)


def numeric_plausibility(value):
    if value is None:
        return 0
    if 1e2 < value < 1e10:
        return 1
    return 0.3


def unit_score(unit):
    valid = ["tco2", "tco2e", "tons", "tonnes"]
    return 1 if unit.lower() in valid else 0.5


# ==============================
# EXTRACTION ENGINE
# ==============================
def extract_with_confidence(text):
    results = []

    for kpi, pattern in ESG_PATTERNS.items():

        matches = list(re.finditer(pattern, text, re.IGNORECASE))

        print(f"{kpi} matches: {len(matches)}")

        for match in matches:
            raw_value = match.group(1)
            unit = match.group(2) if len(match.groups()) > 1 else ""

            value = clean_number(raw_value)

            start, end = match.span()
            context = text[max(0, start-150): end+150]

            c_score = context_score(context)
            n_score = numeric_plausibility(value)
            u_score = unit_score(unit)

            confidence = np.mean([c_score, n_score, u_score])

            results.append({
                "KPI": kpi,
                "Value": value,
                "Unit": unit,
                "Confidence": round(confidence, 3),
                "Context": context
            })

    return results


# ==============================
# SAFE CONSISTENCY CHECK
# ==============================
def consistency_adjustment(df):

    if df is None or df.empty or "KPI" not in df.columns:
        print("⚠️ No valid extracted data for consistency adjustment")
        return df

    adjusted = []

    for kpi in df["KPI"].unique():

        subset = df[df["KPI"] == kpi]

        values = subset["Value"].dropna().values

        if len(values) > 1:
            mean_val = np.mean(values)
            std_val = np.std(values) if np.std(values) != 0 else 1

            for _, row in subset.iterrows():

                z = abs(row["Value"] - mean_val) / std_val
                penalty = min(z / 5, 1)

                new_conf = row["Confidence"] * (1 - penalty)

                adjusted.append({
                    **row,
                    "Adj Confidence": round(new_conf, 3)
                })

        else:
            for _, row in subset.iterrows():
                adjusted.append({
                    **row,
                    "Adj Confidence": row["Confidence"]
                })

    return pd.DataFrame(adjusted)


# ==============================
# SELECT BEST RESULTS
# ==============================
def select_best(df):
    if df.empty:
        return df

    return (
        df.sort_values("Adj Confidence", ascending=False)
          .groupby("KPI")
          .head(2)
    )


# ==============================
# MAIN PIPELINE
# ==============================
if __name__ == "__main__":

    print("📄 Extracting PDF text...")
    text = extract_text_from_pdf(PDF_PATH)

    print("📊 Running ESG extraction...")
    raw_results = extract_with_confidence(text)

    print("Total matches:", len(raw_results))

    if len(raw_results) == 0:
        print("⚠️ No ESG patterns found. Check PDF structure or regex.")
        exit()

    df = pd.DataFrame(raw_results)

    print("🔧 Applying consistency adjustment...")
    df = consistency_adjustment(df)

    print("🎯 Selecting best results...")
    final_df = select_best(df)

    final_df["Source"] = "Allianz_2023"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Saved to:", OUTPUT_PATH)
    print(final_df.head())
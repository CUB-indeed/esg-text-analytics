# ============================================================
# ESG MODEL INFERENCE PIPELINE
# Uses trained V1 / FINAL model
# ============================================================

# =========================
# 1. IMPORTS
# =========================

import os
import re
import joblib
import torch
import pdfplumber
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from collections import defaultdict
from bertopic import BERTopic

from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

# =========================
# 2. CONFIG
# =========================

MODEL_PATH = "models/esg_model_v1"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

PDF_PATH = "/home/coder/project/Allianz_Group_Sustainability_Report_2023-web.pdf"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 3. LOAD MODEL + TOKENIZER + LABEL ENCODER
# =========================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

label_encoder = joblib.load(LABEL_ENCODER_PATH)

# =========================
# 4. ESG KPI REGEX
# (same as training idea, simplified reuse)
# =========================

ESG_PATTERNS = {

    "Scope 1 Emissions":
        r"Scope\s*1[\s\S]{0,120}?(\d[\d,\.]*)\s*(tCO2e|tCO2|tons|tonnes)",

    "Scope 2 Emissions":
        r"Scope\s*2[\s\S]{0,120}?(\d[\d,\.]*)\s*(tCO2e|tCO2|tons|tonnes)",

    "Scope 3 Emissions":
        r"Scope\s*3[\s\S]{0,120}?(\d[\d,\.]*)\s*(tCO2e|tCO2|tons|tonnes)",

    "Renewable Energy":
        r"renewable energy[\s\S]{0,80}?(\d+\.?\d*)\s*%",

    "Women Employees":
        r"women[\s\S]{0,80}?(\d+\.?\d*)\s*%",

    "Board Independence":
        r"independent directors[\s\S]{0,80}?(\d+\.?\d*)\s*%"
}

# =========================
# 5. PDF EXTRACTION
# =========================

def extract_pdf_pages(pdf_path):
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text:
                pages.append({
                    "page": i + 1,
                    "text": text
                })

    return pages

# =========================
# 6. CLEAN TEXT
# =========================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\x00", " ")
    return text.strip()

# =========================
# 7. CHUNKING
# =========================

def split_into_chunks(pages, min_length=80):

    chunks = []

    for page in pages:

        sentences = re.split(r'(?<=[.!?])\s+', page["text"])

        for sent in sentences:

            sent = clean_text(sent)

            if len(sent) > min_length:

                chunks.append({
                    "page": page["page"],
                    "text": sent
                })

    return pd.DataFrame(chunks)

# =========================
# 8. BERT INFERENCE
# =========================

def bert_predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    pred = probs.argmax().item()
    confidence = probs.max().item()

    label = label_encoder.inverse_transform([pred])[0]

    return label, confidence

# =========================
# 9. BERTopic (inference mode)
# =========================

def run_bertopic(texts):

    umap_model = UMAP(random_state=42)
    vectorizer_model = CountVectorizer(stop_words="english")

    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=umap_model,
        vectorizer_model=vectorizer_model
    )

    topics, probs = topic_model.fit_transform(texts)

    return topic_model, topics, probs

# =========================
# 10. KPI EXTRACTION
# =========================

def clean_number(value):
    try:
        return float(value.replace(",", ""))
    except:
        return None


def extract_kpis(text):

    results = []

    for kpi, pattern in ESG_PATTERNS.items():

        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:

            value = clean_number(match.group(1))

            unit = match.group(2) if len(match.groups()) > 1 else ""

            context = text[max(0, match.start()-150): match.end()+150]

            results.append({
                "KPI": kpi,
                "Value": value,
                "Unit": unit,
                "Context": context
            })

    return results

# =========================
# 11. CONFIDENCE SCORE
# =========================

def compute_regex_confidence(context, value):

    keyword_score = sum([
        k in context.lower()
        for k in ["emission", "scope", "co2", "energy"]
    ]) / 4

    numeric_score = 1 if value and value > 0 else 0.3

    return np.mean([keyword_score, numeric_score])

# =========================
# 12. MAIN PIPELINE
# =========================

def run_pipeline(pdf_path):

    print("Extracting PDF...")
    pages = extract_pdf_pages(pdf_path)

    print("Chunking text...")
    df = split_into_chunks(pages)

    print("Running BERT inference...")
    labels = []
    scores = []

    for text in df["text"]:

        label, conf = bert_predict(text)

        labels.append(label)
        scores.append(conf)

    df["ESG_Label"] = labels
    df["BERT_Confidence"] = scores

    print("Running KPI extraction...")

    extracted = []

    for _, row in df.iterrows():

        kpis = extract_kpis(row["text"])

        for kpi in kpis:

            regex_conf = compute_regex_confidence(
                kpi["Context"],
                kpi["Value"]
            )

            final_conf = np.mean([regex_conf, row["BERT_Confidence"]])

            extracted.append({
                "Page": row["page"],
                "Text": row["text"],
                "ESG_Label": row["ESG_Label"],
                "BERT_Confidence": row["BERT_Confidence"],

                "KPI": kpi["KPI"],
                "Value": kpi["Value"],
                "Unit": kpi["Unit"],

                "Regex_Confidence": regex_conf,
                "Final_Confidence": final_conf
            })

    final_df = pd.DataFrame(extracted)

    final_df.to_csv(
        os.path.join(OUTPUT_DIR, "esg_inference_results.csv"),
        index=False
    )

    print("DONE")

    return final_df

# =========================
# 13. RUN
# =========================

if __name__ == "__main__":

    results = run_pipeline(PDF_PATH)

    print(results.head())
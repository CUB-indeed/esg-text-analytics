# ============================================================
# ESG
# ============================================================

import os
import re
import pdfplumber
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

PDF_PATH = "/home/coder/project/Allianz_Group_Sustainability_Report_2023-web.pdf"

DATASET_OUTPUT = "data/all_reports_dataset.csv"
ANNOTATION_OUTPUT = "data/to_annotate.csv"

os.makedirs("data", exist_ok=True)

# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = text.encode("utf-8", "ignore").decode("utf-8")

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d{1,3}\s*/\s*\d{1,3}", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    return text.strip()


# ============================================================
# ESG FILTER
# ============================================================

def is_esg_relevant(text):
    text = text.lower()

    keywords = [
        "carbon", "emission", "ghg", "co2", "net zero",
        "renewable", "energy", "water", "waste",
        "diversity", "human rights", "safety",
        "governance", "risk", "compliance", "privacy"
    ]

    return sum(k in text for k in keywords) >= 2


# ============================================================
# ONTOLOGY
# ============================================================

ontology = {
    "E": [
        "ghg emissions", "greenhouse gas", "co2", "ch4", "n2o",
        "scope 1", "scope 2 market-based", "scope 2 location-based",
        "scope 3", "carbon intensity", "net zero",
        "renewable energy", "energy efficiency",
        "water consumption", "waste management", "biodiversity"
    ],
    "S": [
        "human rights", "diversity", "inclusion",
        "employee training", "health and safety",
        "community investment", "customer satisfaction",
        "labor practices", "workplace safety"
    ],
    "G": [
        "corporate governance", "board oversight",
        "audit committee", "business ethics",
        "anti corruption", "fraud", "bribery",
        "risk management", "data privacy"
    ]
}


# ============================================================
# WEAK LABEL
# ============================================================

def weak_label(text):
    text = text.lower()

    scores = {"E": 0, "S": 0, "G": 0}

    for label, keys in ontology.items():
        for k in keys:
            if k in text:
                scores[label] += 1

    if max(scores.values()) == 0:
        return None

    return max(scores, key=scores.get)


# ============================================================
# EXTRACT TEXT
# ============================================================

def extract_text(pdf_path):
    out = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):

            text = page.extract_text()

            if text:
                text = clean_text(text)

                if len(text) > 100 and is_esg_relevant(text):
                    out.append({
                        "text": text,
                        "page": i + 1
                    })

    return out


# ============================================================
# EXTRACT TABLES
# ============================================================

def extract_tables(pdf_path):
    out = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):

            tables = page.extract_tables()

            if tables:
                for table in tables:

                    rows = []

                    for row in table:
                        if row:
                            row_text = " ".join([str(x) for x in row if x])
                            row_text = clean_text(row_text)

                            if len(row_text) > 20:
                                rows.append(row_text)

                    if rows:
                        out.append({
                            "text": " | ".join(rows),
                            "page": i + 1
                        })

    return out


# ============================================================
# BUILD DATASET (SINGLE PDF)
# ============================================================

def build_dataset(pdf_path):

    print(f"Processing PDF: {os.path.basename(pdf_path)}")

    all_rows = []

    data = extract_text(pdf_path) + extract_tables(pdf_path)

    for r in data:
        r["source_file"] = os.path.basename(pdf_path)
        all_rows.append(r)

    df = pd.DataFrame(all_rows)

    if len(df) == 0:
        return df

    # REMOVE DUPLICATES
    df["text"] = df["text"].str.strip()
    df = df.drop_duplicates(subset=["text"])

    df.to_csv(DATASET_OUTPUT, index=False)

    print(f"Saved dataset → {DATASET_OUTPUT}")

    return df


# ============================================================
# EXPORT ANNOTATION
# ============================================================

def export_for_annotation(df):

    results = []

    for _, row in df.iterrows():

        label = weak_label(row["text"])

        if label is None:
            continue

        results.append({
            "text": row["text"],
            "model_label": label,
            "confidence": 0.5,
            "human_label": ""
        })

    out = pd.DataFrame(results)

    # FINAL DEDUP
    out = out.drop_duplicates(subset=["text"])

    out.to_csv(ANNOTATION_OUTPUT, index=False)

    print(f"Saved annotation → {ANNOTATION_OUTPUT}")


# ============================================================
# RUN STAGE 1
# ============================================================

if __name__ == "__main__":

    df = build_dataset(PDF_PATH)

    print("Export annotation file...")
    export_for_annotation(df)

    print("DONE STAGE 1")

#########################

# ============================================================
# ESG FINAL MODEL TRAINING (FROM HUMAN LABELS)
# ============================================================

import pandas as pd
import torch
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from torch.utils.data import Dataset

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "bert-base-uncased"

OUTPUT_DIR_FINAL = "models/esg_model_final"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET CLASS
# ============================================================

class ESGDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):

        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final():

    print("Loading human labeled dataset...")

    df = pd.read_csv("data/human_labeled.csv")

    # clean data
    df = df.dropna(subset=["human_label", "text"])
    df = df[df["human_label"].isin(["E", "S", "G"])]

    print(f"Dataset size: {len(df)}")

    # encode labels
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["human_label"])

    joblib.dump(le, "models/label_encoder_final.pkl")

    # split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ESGDataset(
        train_df["text"].tolist(),
        train_df["label_id"].tolist(),
        tokenizer
    )

    test_dataset = ESGDataset(
        test_df["text"].tolist(),
        test_df["label_id"].tolist(),
        tokenizer
    )

    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(le.classes_)
    ).to(device)

    # training args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR_FINAL,
        num_train_epochs=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    print("Training final model...")

    trainer.train()

    # save model
    model.save_pretrained(OUTPUT_DIR_FINAL)
    tokenizer.save_pretrained(OUTPUT_DIR_FINAL)

    print(f"Model saved → {OUTPUT_DIR_FINAL}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    train_final()
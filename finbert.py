# =========================
# 1. IMPORT
# =========================
import fitz  # PyMuPDF
import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =========================
# 2. DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# 3. LOAD PDF
# =========================
PDF_PATH = "/home/coder/project/Allianz_Group_Sustainability_Report_2023-web.pdf"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            texts.append(text)
    return texts

pages = extract_text_from_pdf(PDF_PATH)

# =========================
# 4. SPLIT TEXT
# =========================
def split_text(texts):
    chunks = []
    for t in texts:
        sentences = t.split(".")
        for s in sentences:
            s = s.strip()
            if len(s) > 50:
                chunks.append(s)
    return chunks

text_chunks = split_text(pages)
df = pd.DataFrame({"text": text_chunks})

# =========================
# 5. ONTOLOGY LABELING
# =========================
ontology = {
    "E": ["emission", "energy", "carbon", "climate"],
    "S": ["employee", "diversity", "health", "safety"],
    "G": ["board", "governance", "compliance"]
}

def match_ontology(text):
    text = text.lower()
    for category, keywords in ontology.items():
        for k in keywords:
            if k in text:
                return category
    return "Unknown"

df["label"] = df["text"].apply(match_ontology)
df = df[df["label"] != "Unknown"]

# =========================
# 6. CLEAN DATA
# =========================
df = df.drop_duplicates()
df = df[df["text"].str.len() > 50]
df = df.sample(min(len(df), 1000), random_state=42)

# balance sederhana
min_count = df["label"].value_counts().min()
df = df.groupby("label").sample(n=min_count, random_state=42)

# =========================
# 7. ENCODE LABEL
# =========================
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

# =========================
# 8. TRAIN TEST SPLIT
# =========================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# =========================
# 9. MODEL & TOKENIZER
# =========================
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(le.classes_)
).to(device)

# =========================
# 10. TOKENIZATION
# =========================
train_encodings = tokenizer(
    list(train_df["text"]),
    padding=True,
    truncation=True,
    return_tensors="pt"
)

test_encodings = tokenizer(
    list(test_df["text"]),
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# =========================
# 11. DATASET CLASS
# =========================
class ESGDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ESGDataset(train_encodings, train_df["label_id"].tolist())
test_dataset = ESGDataset(test_encodings, test_df["label_id"].tolist())

# =========================
# 12. TRAINING
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    logging_dir="./logs",
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# =========================
# 13. EVALUATION
# =========================
preds = trainer.predict(test_dataset)
y_pred = preds.predictions.argmax(axis=1)

print("\n=== Classification Report ===")
print(classification_report(
    test_df["label_id"],
    y_pred,
    target_names=le.classes_
))

# =========================
# 14. INFERENCE
# =========================
def predict(text):
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pred = outputs.logits.argmax().item()
    return le.inverse_transform([pred])[0]

# =========================
# 15. TEST INFERENCE
# =========================
sample_text = "Carbon emissions decreased by 20% in 2024"
print("\nPrediction:", predict(sample_text))

# =========================
# SAVE RESULT
# =========================

test_df = test_df.copy()
test_df["pred_label_id"] = y_pred
test_df["pred_label"] = le.inverse_transform(y_pred)

df.to_csv("esg_dataset_full.csv", index=False)
test_df.to_csv("esg_test_predictions.csv", index=False)

print("Data saved successfully!")
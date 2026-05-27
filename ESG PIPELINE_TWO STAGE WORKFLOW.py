# ESG PIPELINE — TWO STAGE WORKFLOW

# ============================================================
# STAGE 1
# GENERATE ESG CANDIDATE DATA
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIG
# =========================================================

PDF_DIR = "/home/coder/project/Reports_10_Industries"
OUTPUT_DIR = "/home/coder/project/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# MODEL
# =========================================================

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================================================
# ESG ONTOLOGY
# =========================================================

ontology = {
    "E": "emission carbon climate energy renewable water waste biodiversity sustainability green net zero",
    "S": "employee diversity health safety human rights training labor wellbeing inclusion education community",
    "G": "board governance compliance ethics audit risk regulation policy transparency"
}

ontology_vec = {
    k: embed_model.encode(v)
    for k, v in ontology.items()
}

# =========================================================
# HELPERS
# =========================================================


def clean(text):
    return re.sub(r"\s+", " ", text.lower()) if text else ""



def split_sentences(text):
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) > 20
    ]



def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

# =========================================================
# PDF EXTRACTION
# =========================================================


def extract_pdf(path):

    text = []

    try:
        with pdfplumber.open(path) as pdf:

            for page_num, page in enumerate(pdf.pages[:15]):

                t = page.extract_text()

                if t:
                    text.append(f"[PAGE {page_num+1}] {t}")

    except Exception as e:
        print("ERROR:", path, e)

    return clean(" ".join(text))

# =========================================================
# ESG DETECTION
# =========================================================


def detect_esg(sentence, emb):

    best_cat = None
    best_score = 0

    for k in ontology_vec:

        score = cosine(emb, ontology_vec[k])

        if score > best_score:
            best_score = score
            best_cat = k

    if best_score < 0.25:
        return None, 0

    return best_cat, best_score

# =========================================================
# MAIN PROCESSING
# =========================================================

files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

all_esg_sentences = []
summary_rows = []

for f in files:

    print("Processing:", f)

    path = os.path.join(PDF_DIR, f)

    text = extract_pdf(path)

    sentences = split_sentences(text)

    if len(sentences) == 0:
        continue

    embeddings = embed_model.encode(sentences)

    rows = []

    for i, sentence in enumerate(sentences):

        cat, score = detect_esg(sentence, embeddings[i])

        if cat:

            rows.append({
                "company": f,
                "sentence": sentence,
                "predicted_esg": cat,
                "semantic_score": float(score)
            })

    if len(rows) == 0:
        continue

    df_tmp = pd.DataFrame(rows)

    all_esg_sentences.append(df_tmp)

    summary_rows.append({
        "company": f,
        "E": sum(df_tmp["predicted_esg"] == "E"),
        "S": sum(df_tmp["predicted_esg"] == "S"),
        "G": sum(df_tmp["predicted_esg"] == "G"),
        "avg_semantic_score": df_tmp["semantic_score"].mean(),
        "total_esg_sentences": len(df_tmp)
    })

# =========================================================
# FINAL DATA
# =========================================================


df_esg = pd.concat(all_esg_sentences, ignore_index=True)
df_summary = pd.DataFrame(summary_rows)

# =========================================================
# SAVE CSV
# =========================================================

summary_path = os.path.join(OUTPUT_DIR, "stage1_esg_summary.csv")
sentence_path = os.path.join(OUTPUT_DIR, "stage1_esg_sentences.csv")


df_summary.to_csv(summary_path, index=False)
df_esg.to_csv(sentence_path, index=False)

# =========================================================
# VISUALIZATION
# =========================================================

plt.figure(figsize=(8,5))
df_summary[["E","S","G"]].sum().plot(kind="bar")
plt.title("Total ESG Mentions")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage1_esg_distribution.png"))

plt.figure(figsize=(8,5))
df_summary["avg_semantic_score"].plot(kind="hist", bins=10)
plt.title("Semantic Similarity Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage1_similarity_distribution.png"))

print("\nDONE")
print(df_summary.head())
print("\nSaved:")
print(summary_path)
print(sentence_path)

# ============================================================
# HUMAN LABELING STEP
# ============================================================

# ============================================================
# STAGE 2
# EVALUATION + FINBERT + BERTOPIC
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score
)

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from bertopic import BERTopic

# =========================================================
# CONFIG
# =========================================================

OUTPUT_DIR = "/home/coder/project/outputs"
GROUND_TRUTH_FILE = "/home/coder/project/esg_ground_truth.csv"

# =========================================================
# LOAD DATA
# =========================================================


df = pd.read_csv(GROUND_TRUTH_FILE)

# remove empty labels

df = df[df["human_label"].notna()]

# =========================================================
# EVALUATION OF RULE-BASED ESG
# =========================================================

print("\n============================")
print("RULE-BASED ESG EVALUATION")
print("============================")


y_true = df["human_label"]
y_pred = df["predicted_esg"]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted")
rec = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

balanced_acc = balanced_accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("Balanced Accuracy:", balanced_acc)
print("Cohen Kappa:", kappa)
print("Matthews Corrcoef:", mcc)

print("\nClassification Report")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix")
print(cm)

# =========================================================
# VISUALIZE CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(len(np.unique(y_true))), np.unique(y_true))
plt.yticks(range(len(np.unique(y_true))), np.unique(y_true))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage2_confusion_matrix.png"))

# =========================================================
# TRAIN SUPERVISED ESG CLASSIFIER
# =========================================================

print("\n============================")
print("TRAINING SUPERVISED ESG MODEL")
print("============================")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

X = embed_model.encode(df["sentence"].tolist())
y = df["human_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("\nSUPERVISED MODEL RESULTS")
print("Accuracy:", accuracy_score(y_test, pred))
print("F1:", f1_score(y_test, pred, average="weighted"))

print("\nClassification Report")
print(classification_report(y_test, pred))

# =========================================================
# PREDICT FINAL ESG LABELS
# =========================================================

final_pred = clf.predict(X)

df["final_esg_prediction"] = final_pred

# =========================================================
# FINBERT SENTIMENT
# =========================================================

print("\n============================")
print("FINBERT ANALYSIS")
print("============================")

finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)


def finbert_sentiment(text):

    try:

        res = finbert(text[:256])[0]

        label = res["label"].lower()
        score = res["score"]

        return label, score

    except:
        return "neutral", 0

sentiment_labels = []
sentiment_scores = []

for sentence in df["sentence"]:

    label, score = finbert_sentiment(sentence)

    sentiment_labels.append(label)
    sentiment_scores.append(score)


df["sentiment"] = sentiment_labels
df["sentiment_score"] = sentiment_scores

# =========================================================
# BERTOPIC
# =========================================================

print("\n============================")
print("BERTOPIC ANALYSIS")
print("============================")

# use only validated ESG text

validated_df = df[df["human_label"] != "None"]

texts = validated_df["sentence"].tolist()


topic_model = BERTopic(min_topic_size=5)


topics, probs = topic_model.fit_transform(texts)

validated_df["topic"] = topics

# =========================================================
# TOPIC INFO
# =========================================================


topic_info = topic_model.get_topic_info()

print("\nTOPIC INFO")
print(topic_info.head())

# =========================================================
# EXTRACT TOP WORDS
# =========================================================


def get_topic_name(topic_id):

    words = topic_model.get_topic(topic_id)

    if words is None:
        return "Unknown"

    top_words = [w[0] for w in words[:5]]

    return " | ".join(top_words)

validated_df["topic_keywords"] = validated_df["topic"].apply(get_topic_name)

# =========================================================
# TOPIC INSIGHT SUMMARY
# =========================================================

insights = []

for topic_id in validated_df["topic"].unique():

    topic_subset = validated_df[
        validated_df["topic"] == topic_id
    ]

    insights.append({
        "topic": topic_id,
        "keywords": get_topic_name(topic_id),
        "documents": len(topic_subset),
        "avg_sentiment": topic_subset["sentiment_score"].mean(),
        "dominant_esg": topic_subset["human_label"].mode()[0]
    })


df_insights = pd.DataFrame(insights)

print("\nTOPIC INSIGHTS")
print(df_insights.head())

# =========================================================
# VISUALIZATION
# =========================================================

plt.figure(figsize=(8,5))
df["human_label"].value_counts().plot(kind="bar")
plt.title("Human ESG Labels")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage2_human_labels.png"))

plt.figure(figsize=(8,5))
df["sentiment"].value_counts().plot(kind="bar")
plt.title("FinBERT Sentiment Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage2_sentiment_distribution.png"))

plt.figure(figsize=(8,5))
df_insights.groupby("dominant_esg")["documents"].sum().plot(kind="bar")
plt.title("Topic Distribution by ESG")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage2_topic_distribution.png"))

# =========================================================
# SAVE RESULTS
# =========================================================

validated_df.to_csv(
    os.path.join(OUTPUT_DIR, "stage2_final_esg_analysis.csv"),
    index=False
)


df_insights.to_csv(
    os.path.join(OUTPUT_DIR, "stage2_topic_insights.csv"),
    index=False
)

print("\nDONE")
print("Results saved to:")
print(OUTPUT_DIR)

# =========================================================
# VISUALIZATION
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# =========================================================
# LOAD DATASET
# =========================================================
file_path = "/home/coder/project/outputs/stage2_final_esg_analysis.csv"
df = pd.read_csv(file_path)

# =========================================================
# STYLE
# =========================================================
sns.set_theme(style="white")

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

# =========================================================
# FIGURE (KEY FIX: wider, less tall)
# =========================================================
fig, axes = plt.subplots(
    2, 2,
    figsize=(6.2, 5.2),   # <-- IMPORTANT CHANGE (WIDER, LESS HEIGHT)
    dpi=300
)

# tighten spacing heavily (critical fix)
plt.subplots_adjust(hspace=0.35, wspace=0.35)

# =========================================================
# 1. ESG CATEGORY DISTRIBUTION
# =========================================================
esg_counts = df["final_esg_prediction"].value_counts().head(5)

sns.barplot(
    x=esg_counts.index,
    y=esg_counts.values,
    ax=axes[0, 0]
)

axes[0, 0].set_title("ESG Category Distribution", fontweight="bold")
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("Frequency")

# =========================================================
# 2. SENTIMENT (PIE - compact legend bottom)
# =========================================================
sentiment_counts = df["sentiment"].value_counts()

wedges, _, _ = axes[0, 1].pie(
    sentiment_counts.values,
    labels=None,
    autopct="%1.0f%%",
    startangle=90,
    pctdistance=0.72,
    textprops={"fontsize": 8}
)

axes[0, 1].set_title("Sentiment Distribution", fontweight="bold")

# legend bottom but closer (NO extra height waste)
axes[0, 1].legend(
    wedges,
    sentiment_counts.index,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),  # <-- pulled UP (important fix)
    ncol=3,
    frameon=False
)

# =========================================================
# 3. TOP KEYWORDS (more compact)
# =========================================================
if "topic_keywords" in df.columns:

    text = " ".join(df["topic_keywords"].dropna().astype(str))
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    stopwords = {
        "the", "and", "for", "with", "that", "this",
        "from", "have", "has", "into", "their",
        "about", "will", "were", "been", "being"
    }

    filtered = [w for w in words if w not in stopwords]

    keyword_counts = Counter(filtered)
    top_keywords = keyword_counts.most_common(4)

    keywords = [k[0] for k in top_keywords]
    counts = [k[1] for k in top_keywords]

    sns.barplot(
        x=counts,
        y=keywords,
        ax=axes[1, 0]
    )

    axes[1, 0].set_title("Top ESG Keywords", fontweight="bold")
    axes[1, 0].set_xlabel("Freq")
    axes[1, 0].set_ylabel("")

# =========================================================
# 4. HEATMAP (COMPACT FIX)
# =========================================================
heatmap_data = pd.crosstab(
    df["final_esg_prediction"],
    df["sentiment"]
)

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    linewidths=0.15,
    annot_kws={"size": 7},
    ax=axes[1, 1]
)

axes[1, 1].set_title("ESG vs Sentiment", fontweight="bold")

# rotate but slightly angled (NOT 90°, too tall visually)
axes[1, 1].set_xticklabels(
    axes[1, 1].get_xticklabels(),
    rotation=45,
    ha="right"
)

axes[1, 1].set_xlabel("")
axes[1, 1].set_ylabel("")

# =========================================================
# FINAL LAYOUT (IMPORTANT FIX)
# =========================================================
plt.tight_layout(pad=0.6)   # <-- tighter packing

# =========================================================
# SAVE
# =========================================================
output_path = "/home/coder/project/outputs/esg_summary_figure.png"

plt.savefig(
    output_path,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("Saved:", output_path)

# =========================================================
# ESG Topic Bubble Chart (Clean & Readable)
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
file_path = "/home/coder/project/outputs/stage2_topic_insights.csv"
df = pd.read_csv(file_path)

# Remove noise topic
df = df[df["topic"] != -1]

# ---------------------------------------------------------
# Color Mapping
# ---------------------------------------------------------
colors = {
    "E": "green",
    "S": "blue",
    "G": "orange"
}

# ---------------------------------------------------------
# Clean Keywords Function
# ---------------------------------------------------------
def get_representative_keywords(kw_string, top_n=2):

    keywords = [k.strip() for k in kw_string.split("|")]

    blacklist = {
        "group", "groups", "the", "and", "our", "for", "of", "to",
        "volkswagen", "bayer", "siemens", "tui"
    }

    filtered = [k for k in keywords if k.lower() not in blacklist]

    if len(filtered) == 0:
        filtered = keywords

    return " / ".join(filtered[:top_n])

# ---------------------------------------------------------
# Plot Setup (A4 column optimized)
# ---------------------------------------------------------
plt.figure(figsize=(4.8, 5.5), dpi=DPI)

# ---------------------------------------------------------
# Scatter Plot (Reduced Density)
# ---------------------------------------------------------
for esg in df["dominant_esg"].unique():

    subset = df[df["dominant_esg"] == esg]

    plt.scatter(
        subset["avg_sentiment"],
        subset["documents"],
        s=subset["documents"] * 8,
        c=colors.get(esg, "gray"),
        alpha=0.6,
        edgecolors="black",
        label=esg
    )

# ---------------------------------------------------------
# Label Only Top Topics (Important Fix)
# ---------------------------------------------------------
df_labeled = df.sort_values("documents", ascending=False).head(12)

texts = []

for _, row in df_labeled.iterrows():

    keyword = get_representative_keywords(row["keywords"], top_n=2)

    txt = plt.text(
        row["avg_sentiment"],
        row["documents"],
        keyword,
        fontsize=9
    )

    texts.append(txt)

adjust_text(
    texts,
    arrowprops=dict(
        arrowstyle="-",
        color="gray",
        lw=0.5
    )
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
plt.xlabel("Average Sentiment", fontsize=11)
plt.ylabel("Number of Documents", fontsize=11)

plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(title="ESG Type", fontsize=9)

# ---------------------------------------------------------
# Save Figure
# ---------------------------------------------------------
output_path = "/home/coder/project/outputs/esg_topic_bubble_chart.png"

plt.savefig(
    output_path,
    dpi=DPI,
    bbox_inches="tight"
)

plt.close()

print("Saved:", output_path)

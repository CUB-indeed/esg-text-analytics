# ESG Semantic Analytics Pipeline

An end-to-end NLP system for extracting, classifying, and analyzing Environmental, Social, and Governance (ESG) insights from corporate PDF reports using semantic similarity, supervised learning, sentiment analysis, and topic modeling.

---

## 📌 Project Overview

This project transforms unstructured corporate sustainability reports into structured ESG intelligence using a two-stage NLP pipeline.

### 🔹 Stage 1: ESG Extraction Pipeline
- Extract text from PDF reports using `pdfplumber`
- Split text into sentences
- Compute semantic similarity using SentenceTransformers
- Classify sentences into ESG categories:
  - **E (Environmental)**
  - **S (Social)**
  - **G (Governance)**
- Generate company-level ESG summaries
- Save sentence-level ESG dataset

### 🔹 Stage 2: Advanced NLP Analytics
- Evaluate rule-based ESG classification vs human labels
- Train supervised ESG classifier (Logistic Regression + embeddings)
- Perform sentiment analysis using **FinBERT**
- Topic modeling using **BERTopic**
- Generate ESG topic insights
- Produce advanced visualizations

---

## 🧠 Methods & Models Used

- Sentence Transformers (`all-MiniLM-L6-v2`)
- Cosine similarity for semantic matching
- Logistic Regression classifier
- FinBERT (`ProsusAI/finbert`)
- BERTopic for topic modeling
- PDF parsing with `pdfplumber`
- Visualization using `matplotlib` and `seaborn`

---

## 📊 Outputs

The pipeline generates:

### 📁 Datasets
- ESG sentence-level dataset
- ESG company summary
- Human-labeled evaluation dataset
- Topic modeling results

### 📈 Metrics
- Accuracy, Precision, Recall, F1-score
- Balanced Accuracy
- Cohen’s Kappa
- Matthews Correlation Coefficient
- Confusion Matrix

### 📉 Visualizations
- ESG distribution chart
- Semantic similarity histogram
- Confusion matrix heatmap
- Sentiment distribution
- ESG vs sentiment heatmap
- ESG topic bubble chart

---

## 📂 Project Structure


.
├── stage1_esg_pipeline.py
├── stage2_esg_evaluation.py
├── esg_ground_truth.csv
│
├── Reports_10_Industries/ # Input PDF reports
│
├── outputs/
│ ├── stage1_esg_summary.csv
│ ├── stage1_esg_sentences.csv
│ ├── stage2_final_esg_analysis.csv
│ ├── stage2_topic_insights.csv
│ ├── esg_summary_figure.png
│ ├── esg_topic_bubble_chart.png
│ └── ...


---

## ⚙️ Installation

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pdfplumber sentence-transformers transformers bertopic adjustText

🚀 How to Run
Step 1: Run ESG Extraction Pipeline
python stage1_pipeline.py
Step 2: Run Evaluation & Advanced Analytics
python stage2_pipeline.py

📌 Key Features
Hybrid ESG classification (rules + machine learning)
Human benchmark evaluation
Financial sentiment analysis (FinBERT)
Topic discovery (BERTopic)
Full evaluation metrics suite
Publication-ready visualizations
Scalable PDF processing pipeline

📌 Use Cases
ESG reporting automation
Sustainability analytics
Financial NLP research
Corporate risk analysis
Investment decision support systems

🧠 Pipeline Workflow
PDF ingestion
Text extraction
Sentence segmentation
Semantic embedding
ESG classification
Sentiment analysis
Topic modeling
Aggregation & visualization

📊 Example Insights
ESG distribution across industries
Sentiment trends in sustainability reports
Governance vs Environmental focus comparison
Topic clusters of ESG narratives
Model performance against human labeling

🧑‍💻 Author
Name: Widyasmoro Priatmojo
Role: PhD Student
Focus: ESG Analytics, Financial NLP, Topic Modeling

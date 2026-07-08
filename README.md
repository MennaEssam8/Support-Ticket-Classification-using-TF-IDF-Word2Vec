# Support Ticket Classification using TF-IDF & Word2Vec

Baseline classification and word-embedding feature engineering for routing customer
support tickets into predefined categories, using two feature representations
(TF-IDF and Word2Vec) and a shared Logistic Regression classifier, evaluated
side by side.

## Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Streamlit demo](#streamlit-demo)
- [Limitations, bias, and ethics](#limitations-bias-and-ethics)
- [Next steps (Phase 2 ideas)](#next-steps-phase-2-ideas)

## Overview

Support teams need incoming tickets routed to the right queue quickly. This
phase builds and compares two baseline text classifiers for that task:

1. **TF-IDF + Logistic Regression** — sparse bag-of-n-grams features.
2. **Word2Vec (averaged) + Logistic Regression** — dense embeddings trained
   on the ticket corpus, averaged per document.

Both share the same preprocessing pipeline, train/test split, and classifier
family, so the comparison isolates the effect of the feature representation
itself.

## Dataset

**[Bitext Customer Support LLM Chatbot Training Dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset)**
— a real, publicly available dataset of **26,872** customer support
utterances, generated with Bitext's hybrid NLP/NLG methodology from real
seed text and curated by computational linguists (not hand-written
templates). It includes realistic noise such as typos and run-on words.

| Column | Used as | Description |
|---|---|---|
| `text` (`instruction`) | model input | the customer's message |
| `category` | **classification target** | 11 coarse categories: ACCOUNT, ORDER, REFUND, INVOICE, PAYMENT, SHIPPING, DELIVERY, FEEDBACK, CONTACT, SUBSCRIPTION, CANCEL |
| `intent` | kept, unused in Phase 1 | 27 fine-grained intents (e.g. `cancel_order`, `track_refund`) — a natural target for a future, finer-grained classifier |

Full EDA (class balance, ticket length distribution, top vocabulary) is in
[`reports/evaluation_report_m1.txt`](reports/evaluation_report_m1.txt) and
[`reports/figures/`](reports/figures/).

## Architecture

```
                         ┌─────────────────────┐
                         │  raw_tickets.csv     │
                         │  (text, category)    │
                         └──────────┬───────────┘
                                    │
                              run_eda()
                                    │
                    ┌───────────────┴───────────────┐
                    │      preprocess_corpus()        │
                    │  clean → tokenize → stopwords   │
                    │         → lemmatize             │
                    └───────────────┬───────────────┘
                                    │
                  train/test split (80/20, stratified)
                                    │
            ┌───────────────────────┴───────────────────────┐
            │                                                 │
   ┌────────▼─────────┐                             ┌────────▼─────────┐
   │  TfidfVectorizer  │                             │  Word2Vec (skip- │
   │  (1-2 grams)      │                             │  gram, size=100) │
   └────────┬─────────┘                             └────────┬─────────┘
            │                                                 │
   sparse TF-IDF matrix                              average word vectors
            │                                          per document
            │                                                 │
   ┌────────▼─────────┐                             ┌────────▼─────────┐
   │ LogisticRegression│                             │ LogisticRegression│
   │  (TF-IDF branch)  │                             │  (W2V branch)     │
   └────────┬─────────┘                             └────────┬─────────┘
            │                                                 │
            └───────────────────────┬───────────────────────┘
                                    │
              evaluate_model() + plot_confusion_matrix()
                    + plot_roc_pr_curves() + comparison table
                                    │
                         evaluation_report_m1.txt
                                    │
                    saved models/vectorizers (pickle/joblib)
                                    │
                          app.py (Streamlit demo)
```

## Project structure

```
Support-Ticket-Classification-using-TF-IDF-Word2Vec/
├── data/
│   └── raw_tickets.csv                 # Bitext dataset (text, category, intent)
├── models/
│   ├── tfidf_baseline_model.pkl        # LogisticRegression on TF-IDF features
│   ├── word2vec_model.pkl              # trained gensim Word2Vec model
│   ├── word2vec_baseline_model.pkl     # LogisticRegression on averaged W2V vectors
│   └── label_classes.pkl               # sorted list of category labels
├── reports/
│   ├── evaluation_report_m1.txt        # full report: EDA + metrics + bias/ethics
│   ├── model_comparison.csv            # metrics table (machine-readable)
│   └── figures/
│       ├── class_distribution.png
│       ├── text_length_distribution.png
│       ├── top_words_overall.png
│       ├── confusion_matrix_TF-IDF.png
│       ├── confusion_matrix_Word2Vec.png
│       ├── roc_curves_TF-IDF.png
│       ├── roc_curves_Word2Vec.png
│       ├── pr_curves_TF-IDF.png
│       └── pr_curves_Word2Vec.png
├── src/
│   ├── data_pipeline.py                # the full modular pipeline (this is the deliverable)
│   └── tfidf_vectorizer.pkl            # fitted TfidfVectorizer
├── app.py                              # Streamlit demo for classifying new tickets
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The pipeline downloads a handful of small NLTK corpora on first run
(`stopwords`, `wordnet`, `omw-1.4`, `punkt`, `punkt_tab`) — this requires
internet access once; after that they're cached locally.

## Usage

**1. Run the full training pipeline** (loads data → EDA → preprocess →
train both models → evaluate → save everything):

```bash
cd src
python data_pipeline.py
```

This regenerates every file under `models/`, `reports/`, and
`src/tfidf_vectorizer.pkl`. It takes well under a minute on the full 26.8k-row
dataset on a laptop CPU.

**2. Launch the interactive demo:**

```bash
# from the project root, after step 1 has produced the model artifacts
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Results

Evaluated on a stratified 20% held-out test set (5,375 tickets), full
classification reports and confusion matrices are in
[`reports/evaluation_report_m1.txt`](reports/evaluation_report_m1.txt).

| Model | Accuracy | Precision (wtd) | Recall (wtd) | F1 (wtd) | F1 (macro) | ROC AUC (micro) | Avg Precision (micro) |
|---|---|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | 0.9980 | 0.9980 | 0.9980 | 0.9980 | 0.9979 | 1.0000 | 0.9999 |
| **Word2Vec (avg) + Logistic Regression** | 0.9968 | 0.9968 | 0.9968 | 0.9968 | 0.9968 | 0.9999 | 0.9996 |

Both baselines score very highly on this dataset — the categories map onto
fairly distinctive vocabulary (e.g. "refund", "cancel", "invoice"), so even
a linear classifier over sparse or averaged-dense features separates them
well. TF-IDF has a small, consistent edge, which matches the general pattern
that sparse lexical features tend to outperform averaged embeddings on
short, keyword-driven text where mean-pooling dilutes discriminative words.
See the full report for per-class breakdowns, confusion matrices, and
ROC/PR curves — the near-ceiling scores here are also a reason to treat this
as a *baseline to beat*, not a finished production model (see
[Limitations](#limitations-bias-and-ethics)).

## Streamlit demo

`app.py` loads the saved artifacts and lets you type in a new, unseen ticket
to see:

- The predicted category from each model, side by side
- Per-class prediction probabilities as bar charts
- An agreement indicator — when the two independently-trained models
  disagree, the demo flags the ticket as a good candidate for human review
  rather than fully automated routing

## Limitations, bias, and ethics

The full write-up — covering dataset representativeness, class imbalance,
language/dialect bias, embedding bias, deployment risks (misclassification
cost asymmetry, automation bias, feedback loops, PII), and recommended
mitigations — lives in the **"BIAS, LIMITATIONS, AND ETHICAL
CONSIDERATIONS"** section of
[`reports/evaluation_report_m1.txt`](reports/evaluation_report_m1.txt).
Please read it before considering any production use of these models.

## Next steps (Phase 2 ideas)

- Fine-tune/validate on a sample of real, de-identified tickets from an
  actual support queue rather than relying solely on the Bitext benchmark.
- Try the finer-grained `intent` column (27 classes) as a target.
- Add class weighting / resampling to address category imbalance.
- Swap in a transformer-based encoder (e.g. a sentence-embedding model) as a
  third feature representation to compare against TF-IDF and Word2Vec.
- Add a confidence threshold + human-review queue for low-confidence or
  high-stakes categories (REFUND, PAYMENT, ACCOUNT) in the Streamlit demo.
- Add PII detection/redaction before any logging of raw ticket text.

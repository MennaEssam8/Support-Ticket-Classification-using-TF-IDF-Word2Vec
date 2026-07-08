"""
Phase 1: Baseline Classification and Word Embedding Feature Engineering
for the Customer Support Ticket Classification System.

Dataset: Bitext Customer Support LLM Chatbot Training Dataset (real data)
Source:  https://github.com/bitext/customer-support-llm-chatbot-training-dataset
         (Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv)
Target column used for classification: `category` (11 coarse ticket categories,
e.g. ACCOUNT, ORDER, REFUND, INVOICE, PAYMENT, SHIPPING, DELIVERY, FEEDBACK,
CONTACT, SUBSCRIPTION, CANCEL). The finer-grained `intent` column (27 classes)
is retained in the data for optional future work but is not the Phase 1 target.

This script is organized into modular functions covering:
    1. Data loading
    2. Exploratory Data Analysis (EDA) — class balance, text length, vocabulary
    3. Text preprocessing (tokenization, stop word removal, lemmatization)
    4. TF-IDF feature engineering + baseline classifier (Logistic Regression)
    5. Word2Vec feature engineering (averaged word vectors) + classifier
    6. Evaluation: accuracy/precision/recall/F1, confusion matrices, ROC & PR curves
    7. Model comparison table
    8. Serialization of all models/vectorizers with pickle/joblib
    9. Generation of an evaluation report, including a bias/ethics section
"""

import os
import re
import string
import pickle
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts/servers
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.utils import resample

from gensim.models import Word2Vec


# --------------------------------------------------------------------------
# Paths (relative to project root; this file lives in <root>/src/)
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw_tickets.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

TFIDF_VECTORIZER_PATH = os.path.join(SRC_DIR, "tfidf_vectorizer.pkl")
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, "tfidf_baseline_model.pkl")
WORD2VEC_MODEL_PATH = os.path.join(MODELS_DIR, "word2vec_model.pkl")
W2V_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "word2vec_baseline_model.pkl")
LABEL_LIST_PATH = os.path.join(MODELS_DIR, "label_classes.pkl")
REPORT_PATH = os.path.join(REPORTS_DIR, "evaluation_report_m1.txt")
COMPARISON_CSV_PATH = os.path.join(REPORTS_DIR, "model_comparison.csv")

RANDOM_STATE = 42
W2V_VECTOR_SIZE = 100
# Cap dataset size for reasonable local run time; set to None to use all rows.
MAX_ROWS = None

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# 0. Setup: ensure NLTK resources are available
# --------------------------------------------------------------------------
def ensure_nltk_resources():
    """Downloads required NLTK corpora if not already present."""
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def load_data(path=DATA_PATH, max_rows=MAX_ROWS):
    """Loads the raw ticket dataset from CSV.

    Expected columns: ticket_id, text, category (intent optional/extra).
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "category"]).reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = resample(
            df, n_samples=max_rows, replace=False,
            stratify=df["category"], random_state=RANDOM_STATE,
        ).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# --------------------------------------------------------------------------
def run_eda(df, figures_dir=FIGURES_DIR):
    """Generates and saves EDA plots + returns a text summary block.

    Produces:
        - class_distribution.png : bar chart of ticket counts per category
        - text_length_distribution.png : histogram of char/word lengths
        - top_words_overall.png : most frequent tokens (post-cleaning, pre-lemmatization)
    """
    os.makedirs(figures_dir, exist_ok=True)
    summary_lines = []

    # --- Class distribution ---
    counts = df["category"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=counts.values, y=counts.index, hue=counts.index,
                      palette="viridis", legend=False)
    ax.set_xlabel("Number of tickets")
    ax.set_ylabel("Category")
    ax.set_title("Ticket Category Distribution (Bitext Customer Support Dataset)")
    for i, v in enumerate(counts.values):
        ax.text(v + 5, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "class_distribution.png"), dpi=150)
    plt.close()

    imbalance_ratio = counts.max() / counts.min()
    summary_lines.append(
        f"Class imbalance ratio (largest / smallest category): {imbalance_ratio:.2f}x "
        f"({counts.idxmax()}: {counts.max()} vs {counts.idxmin()}: {counts.min()})"
    )

    # --- Text length distribution ---
    char_lens = df["text"].str.len()
    word_lens = df["text"].str.split().apply(len)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(char_lens, bins=30, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Ticket Length (characters)")
    axes[0].set_xlabel("Characters")
    sns.histplot(word_lens, bins=20, ax=axes[1], color="#DD8452")
    axes[1].set_title("Ticket Length (words)")
    axes[1].set_xlabel("Words")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "text_length_distribution.png"), dpi=150)
    plt.close()

    summary_lines.append(
        f"Ticket length: mean {word_lens.mean():.1f} words "
        f"(min {word_lens.min()}, max {word_lens.max()}, median {word_lens.median():.0f})"
    )

    # --- Top words overall (simple frequency, using light cleaning only) ---
    all_tokens = []
    stop_words = set(stopwords.words("english"))
    for t in df["text"].sample(min(5000, len(df)), random_state=RANDOM_STATE):
        cleaned = re.sub(r"[^a-zA-Z\s]", " ", t.lower())
        all_tokens.extend([w for w in cleaned.split() if w not in stop_words and len(w) > 2])
    freq = pd.Series(all_tokens).value_counts().head(20)
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(x=freq.values, y=freq.index, hue=freq.index,
                      palette="mako", legend=False)
    ax.set_xlabel("Frequency")
    ax.set_title("Top 20 Most Frequent Words (sampled, stopwords removed)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "top_words_overall.png"), dpi=150)
    plt.close()

    return "\n".join(summary_lines), counts


# --------------------------------------------------------------------------
# 3. Text preprocessing pipeline
# --------------------------------------------------------------------------
_STOPWORDS = None
_LEMMATIZER = None


def _get_preprocessing_tools():
    global _STOPWORDS, _LEMMATIZER
    if _STOPWORDS is None:
        _STOPWORDS = set(stopwords.words("english"))
    if _LEMMATIZER is None:
        _LEMMATIZER = WordNetLemmatizer()
    return _STOPWORDS, _LEMMATIZER


def clean_text(text):
    """Lowercases text, expands Bitext placeholder tags, and strips URLs,
    punctuation, and digits."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Bitext-style placeholders like {{Order Number}} -> "order number"
    text = re.sub(r"\{\{(.*?)\}\}", lambda m: re.sub(r"[_\-]", " ", m.group(1)), text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text):
    """Full preprocessing pipeline: clean -> tokenize -> remove stopwords ->
    lemmatize. Returns a list of processed tokens.
    """
    stop_words, lemmatizer = _get_preprocessing_tools()
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def preprocess_corpus(texts):
    """Applies preprocess_text to an iterable of raw documents.

    Returns:
        token_lists: list[list[str]]  (needed for Word2Vec training)
        joined_texts: list[str]       (needed for TF-IDF, which expects strings)
    """
    token_lists = [preprocess_text(t) for t in texts]
    joined_texts = [" ".join(tokens) for tokens in token_lists]
    return token_lists, joined_texts


# --------------------------------------------------------------------------
# 4. TF-IDF feature engineering + baseline classifier
# --------------------------------------------------------------------------
def build_tfidf_features(train_texts, test_texts, max_features=8000):
    """Fits a TF-IDF vectorizer on training text and transforms both splits."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test


def train_tfidf_classifier(X_train, y_train):
    """Trains a Logistic Regression classifier on TF-IDF features."""
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, C=5.0)
    clf.fit(X_train, y_train)
    return clf


# --------------------------------------------------------------------------
# 5. Word2Vec feature engineering + classifier
# --------------------------------------------------------------------------
def train_word2vec(token_lists, vector_size=W2V_VECTOR_SIZE, window=5, min_count=2):
    """Trains a Word2Vec model (skip-gram) on the tokenized corpus."""
    model = Word2Vec(
        sentences=token_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        workers=4,
        epochs=15,
        seed=RANDOM_STATE,
    )
    return model


def document_to_avg_vector(tokens, w2v_model, vector_size=W2V_VECTOR_SIZE):
    """Converts a single tokenized document into an averaged Word2Vec vector.

    Out-of-vocabulary tokens are skipped. Documents with no in-vocabulary
    tokens yield a zero vector.
    """
    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


def corpus_to_avg_vectors(token_lists, w2v_model, vector_size=W2V_VECTOR_SIZE):
    """Converts a list of tokenized documents into a matrix of averaged
    Word2Vec vectors, shape (n_documents, vector_size).
    """
    return np.vstack(
        [document_to_avg_vector(tokens, w2v_model, vector_size) for tokens in token_lists]
    )


def train_word2vec_classifier(X_train, y_train):
    """Trains a Logistic Regression classifier on averaged Word2Vec features."""
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, C=5.0)
    clf.fit(X_train, y_train)
    return clf


# --------------------------------------------------------------------------
# 6. Evaluation: metrics, confusion matrix, ROC/PR curves
# --------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name):
    """Computes accuracy, precision, recall, F1 (weighted + macro) and a
    full classification report / confusion matrix for a fitted model.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    report_text = classification_report(y_test, y_pred, zero_division=0)
    labels_sorted = sorted(y_test.unique())
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "precision_macro": precision_m,
        "recall_macro": recall_m,
        "f1_macro": f1_m,
        "classification_report": report_text,
        "confusion_matrix": conf_matrix,
        "labels": labels_sorted,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(conf_matrix, labels, title, save_path):
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=True)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_pr_curves(model, X_test, y_test, labels, model_name, save_dir):
    """Plots one-vs-rest ROC curves and Precision-Recall curves (per class +
    micro-average) for a multiclass classifier that supports predict_proba.
    """
    os.makedirs(save_dir, exist_ok=True)
    y_test_bin = label_binarize(y_test, classes=labels)
    y_score = model.predict_proba(X_test)

    # ----- ROC curves -----
    plt.figure(figsize=(8, 7))
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, label=f"micro-average (AUC = {auc_micro:.3f})",
              color="black", linewidth=2.5, linestyle="--")

    colors = sns.color_palette("husl", len(labels))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], alpha=0.8,
                  label=f"{label} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k:", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (One-vs-Rest) — {model_name}")
    plt.legend(loc="lower right", fontsize=7, ncol=2)
    plt.tight_layout()
    roc_path = os.path.join(save_dir, f"roc_curves_{model_name}.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # ----- Precision-Recall curves -----
    plt.figure(figsize=(8, 7))
    precision_micro, recall_micro, _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
    ap_micro = average_precision_score(y_test_bin, y_score, average="micro")
    plt.plot(recall_micro, precision_micro, color="black", linewidth=2.5,
              linestyle="--", label=f"micro-average (AP = {ap_micro:.3f})")

    for i, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, color=colors[i], alpha=0.8,
                  label=f"{label} (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves (One-vs-Rest) — {model_name}")
    plt.legend(loc="lower left", fontsize=7, ncol=2)
    plt.tight_layout()
    pr_path = os.path.join(save_dir, f"pr_curves_{model_name}.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()

    return {"roc_auc_micro": auc_micro, "avg_precision_micro": ap_micro,
            "roc_path": roc_path, "pr_path": pr_path}


# --------------------------------------------------------------------------
# 7. Model comparison table
# --------------------------------------------------------------------------
def build_comparison_table(results_list, curve_metrics_list):
    rows = []
    for results, curve_metrics in zip(results_list, curve_metrics_list):
        rows.append({
            "Model": results["model_name"],
            "Accuracy": round(results["accuracy"], 4),
            "Precision (weighted)": round(results["precision_weighted"], 4),
            "Recall (weighted)": round(results["recall_weighted"], 4),
            "F1 (weighted)": round(results["f1_weighted"], 4),
            "F1 (macro)": round(results["f1_macro"], 4),
            "ROC AUC (micro)": round(curve_metrics["roc_auc_micro"], 4),
            "Avg Precision (micro)": round(curve_metrics["avg_precision_micro"], 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 8. Serialization helpers
# --------------------------------------------------------------------------
def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_joblib(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


# --------------------------------------------------------------------------
# 9. Report generation (includes EDA summary, comparison table, bias/ethics)
# --------------------------------------------------------------------------
def build_evaluation_report(df, eda_summary, class_counts, tfidf_results, w2v_results,
                             tfidf_curves, w2v_curves, comparison_df):
    lines = []
    lines.append("=" * 78)
    lines.append("EVALUATION REPORT — PHASE 1")
    lines.append("Baseline Classification and Word Embedding Feature Engineering")
    lines.append("Customer Support Ticket Classification System")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Dataset: Bitext Customer Support LLM Chatbot Training Dataset (real data)")
    lines.append("Source: github.com/bitext/customer-support-llm-chatbot-training-dataset")
    lines.append(f"Dataset size (used in this run): {len(df)} tickets")
    lines.append(f"Number of categories: {df['category'].nunique()}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("EXPLORATORY DATA ANALYSIS (EDA)")
    lines.append("-" * 78)
    lines.append("Class distribution:")
    for cat, count in class_counts.items():
        pct = 100 * count / len(df)
        lines.append(f"  - {cat:<15s}: {count:>5d} tickets ({pct:5.1f}%)")
    lines.append("")
    lines.append(eda_summary)
    lines.append("")
    lines.append("EDA figures saved to reports/figures/:")
    lines.append("  - class_distribution.png")
    lines.append("  - text_length_distribution.png")
    lines.append("  - top_words_overall.png")
    lines.append("")

    for results, curves in ((tfidf_results, tfidf_curves), (w2v_results, w2v_curves)):
        lines.append("-" * 78)
        lines.append(f"MODEL: {results['model_name']}")
        lines.append("-" * 78)
        lines.append(f"Accuracy             : {results['accuracy']:.4f}")
        lines.append(f"Precision (weighted)  : {results['precision_weighted']:.4f}")
        lines.append(f"Recall    (weighted)  : {results['recall_weighted']:.4f}")
        lines.append(f"F1-score  (weighted)  : {results['f1_weighted']:.4f}")
        lines.append(f"Precision (macro)     : {results['precision_macro']:.4f}")
        lines.append(f"Recall    (macro)     : {results['recall_macro']:.4f}")
        lines.append(f"F1-score  (macro)     : {results['f1_macro']:.4f}")
        lines.append(f"ROC AUC (micro-avg)   : {curves['roc_auc_micro']:.4f}")
        lines.append(f"Avg Precision (micro) : {curves['avg_precision_micro']:.4f}")
        lines.append("")
        lines.append("Per-class classification report:")
        lines.append(results["classification_report"])
        lines.append(f"Confusion matrix figure saved to: reports/figures/confusion_matrix_{results['model_name']}.png")
        lines.append(f"ROC curve figure saved to:        {os.path.relpath(curves['roc_path'], REPORTS_DIR)}")
        lines.append(f"PR curve figure saved to:         {os.path.relpath(curves['pr_path'], REPORTS_DIR)}")
        lines.append("")

    lines.append("=" * 78)
    lines.append("MODEL COMPARISON TABLE")
    lines.append("=" * 78)
    lines.append(comparison_df.to_string(index=False))
    lines.append("")
    lines.append("(Full table also saved to reports/model_comparison.csv)")
    better = "TF-IDF" if tfidf_results["f1_weighted"] >= w2v_results["f1_weighted"] else "Word2Vec"
    lines.append("")
    lines.append(
        f"Based on weighted F1-score, the {better} + Logistic Regression pipeline "
        f"performed better on this held-out test set."
    )
    lines.append(
        "TF-IDF typically holds an advantage on this kind of short, template-influenced "
        "support-ticket text because sparse lexical features (specific words/n-grams like "
        "'cancel', 'invoice', 'refund') map very directly onto category labels. Averaged "
        "Word2Vec vectors compress each document to a single dense vector, discarding word "
        "order and diluting sparse, highly discriminative terms with the surrounding "
        "context — a known limitation of simple mean-pooling for short texts."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Bias / limitations / ethics section (required non-functional item)
    # ------------------------------------------------------------------
    lines.append("=" * 78)
    lines.append("BIAS, LIMITATIONS, AND ETHICAL CONSIDERATIONS")
    lines.append("=" * 78)
    lines.append("""
1. Data source and representativeness
   - This dataset (Bitext Customer Support LLM Chatbot Training Dataset) is a
     hybrid *synthetic* dataset: real natural-language seed text was expanded
     using NLP/NLG technology and curated by computational linguists, rather
     than being sampled directly from a live support queue of a specific
     company. It is far more linguistically realistic than hand-written
     templates (it includes typos, run-on words, register variation), but it
     was generated to be broadly representative across many verticals, not
     tailored to any one product, customer base, or support workflow. A
     production deployment should be fine-tuned/validated on a sample of the
     organization's own real tickets before going live.
   - The dataset's `flags` field (not used as a model feature here) encodes
     which linguistic variation each utterance represents (colloquial,
     polite, offensive, etc.). This is a useful signal that the underlying
     utterance distribution was deliberately engineered for diversity, which
     is good for robustness testing but means the class-conditional language
     style may not perfectly match a specific real deployment's tone.

2. Class balance and label bias
   - Category classes are naturally imbalanced in this dataset (see EDA
     above — the largest category has several times more tickets than the
     smallest). Logistic Regression trained without class weighting will
     tend to be biased toward the majority class in a real-world imbalanced
     stream; per-class precision/recall (not just accuracy) should always be
     monitored, and `class_weight='balanced'` or resampling should be
     evaluated in future phases if minority-category miss rates matter.
   - The 11-category taxonomy used here (ACCOUNT, ORDER, REFUND, etc.) is a
     coarse grouping of 27 finer-grained intents. Tickets that are genuinely
     ambiguous between categories are forced into a single label by the
     original dataset's annotation scheme, which can encode arbitrary
     boundary decisions into what the model treats as ground truth.

3. Language, dialect, and demographic bias
   - Preprocessing (tokenization, stop word lists, lemmatization) and both
     feature representations are English-only and tuned to relatively
     standard English, albeit with some typos/informal phrasing represented
     in the data. Customers who write in non-native English, regional
     dialects, or code-switch between languages are likely to see degraded
     classification accuracy — an indirect bias against non-native English
     speakers that should be measured explicitly with dialect-stratified
     evaluation before deployment.
   - Word2Vec embeddings, even trained on a task-specific corpus, can encode
     spurious co-occurrence patterns from that corpus. If a broader
     general-purpose pretrained embedding is substituted in later phases,
     it should be audited for encoded social bias (e.g. gender/occupation
     stereotypes) before use.

4. Operational and deployment risks
   - Asymmetric misclassification cost: a REFUND or PAYMENT ticket
     misrouted as FEEDBACK or CONTACT could delay a financially sensitive
     issue and cause real customer harm. This baseline optimizes for
     overall accuracy/F1 and does not account for the fact that some
     misclassifications are costlier than others.
   - Automation bias: support staff may over-trust automated routing labels
     and under-scrutinize confidently-but-incorrectly routed tickets. A
     human-in-the-loop review step and a confidence threshold for automatic
     routing versus manual review is recommended before production use —
     the Streamlit demo in this repo surfaces prediction probabilities for
     exactly this reason.
   - Feedback loops: if this classifier's own predictions are later used as
     training labels for retraining, systematic bias will compound over
     time rather than self-correct. Human-reviewed labels should be used
     for any retraining data.
   - Privacy: real support tickets (unlike this dataset's placeholder tags
     such as {{Order Number}}) often contain genuine PII — names, emails,
     account numbers. This phase does not implement PII detection/redaction;
     any pipeline that stores or logs raw ticket text in production should
     add PII detection/redaction and comply with applicable data protection
     regulations.

5. Recommended mitigations for later phases
   - Validate on a held-out sample of the organization's own real, de-
     identified support tickets, not only on the Bitext benchmark.
   - Monitor per-class precision/recall/AUC in production, not just
     aggregate accuracy, and set differentiated confidence thresholds for
     high-stakes categories (e.g. REFUND, PAYMENT, ACCOUNT).
   - Periodically re-audit the model for performance drift and dialect/
     demographic disparities as real traffic patterns evolve.
   - Maintain a human review path for low-confidence or high-stakes
     predictions rather than fully automating ticket routing.
""".strip("\n"))
    lines.append("")
    lines.append("=" * 78)
    lines.append("END OF REPORT")
    lines.append("=" * 78)

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_pipeline():
    print("[1/10] Ensuring NLTK resources are available...")
    ensure_nltk_resources()

    print("[2/10] Loading raw ticket data...")
    df = load_data()

    print("[3/10] Running Exploratory Data Analysis (EDA)...")
    eda_summary, class_counts = run_eda(df)

    print("[4/10] Preprocessing text (tokenize -> remove stopwords -> lemmatize)...")
    token_lists, joined_texts = preprocess_corpus(df["text"].tolist())
    df["tokens"] = token_lists
    df["clean_text"] = joined_texts

    print("[5/10] Splitting into train/test sets (80/20, stratified)...")
    (
        train_tokens, test_tokens,
        train_texts, test_texts,
        y_train, y_test,
    ) = train_test_split(
        df["tokens"].tolist(),
        df["clean_text"].tolist(),
        df["category"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["category"],
    )
    labels_sorted = sorted(df["category"].unique())
    save_pickle(labels_sorted, LABEL_LIST_PATH)

    # ---------------- TF-IDF branch ----------------
    print("[6/10] Building TF-IDF features and training baseline classifier...")
    tfidf_vectorizer, X_train_tfidf, X_test_tfidf = build_tfidf_features(train_texts, test_texts)
    tfidf_clf = train_tfidf_classifier(X_train_tfidf, y_train)

    save_pickle(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
    save_joblib(tfidf_clf, TFIDF_MODEL_PATH)

    tfidf_results = evaluate_model(tfidf_clf, X_test_tfidf, y_test, "TF-IDF")
    plot_confusion_matrix(
        tfidf_results["confusion_matrix"], tfidf_results["labels"],
        "Confusion Matrix — TF-IDF + Logistic Regression",
        os.path.join(FIGURES_DIR, "confusion_matrix_TF-IDF.png"),
    )
    tfidf_curves = plot_roc_pr_curves(
        tfidf_clf, X_test_tfidf, y_test, labels_sorted, "TF-IDF", FIGURES_DIR
    )

    # ---------------- Word2Vec branch ----------------
    print("[7/10] Training Word2Vec embeddings and averaged-vector classifier...")
    w2v_model = train_word2vec(train_tokens)
    save_joblib(w2v_model, WORD2VEC_MODEL_PATH)

    X_train_w2v = corpus_to_avg_vectors(train_tokens, w2v_model)
    X_test_w2v = corpus_to_avg_vectors(test_tokens, w2v_model)

    w2v_clf = train_word2vec_classifier(X_train_w2v, y_train)
    save_joblib(w2v_clf, W2V_CLASSIFIER_PATH)

    w2v_results = evaluate_model(w2v_clf, X_test_w2v, y_test, "Word2Vec")
    plot_confusion_matrix(
        w2v_results["confusion_matrix"], w2v_results["labels"],
        "Confusion Matrix — Word2Vec (avg) + Logistic Regression",
        os.path.join(FIGURES_DIR, "confusion_matrix_Word2Vec.png"),
    )
    w2v_curves = plot_roc_pr_curves(
        w2v_clf, X_test_w2v, y_test, labels_sorted, "Word2Vec", FIGURES_DIR
    )

    # ---------------- Comparison table ----------------
    print("[8/10] Building model comparison table...")
    comparison_df = build_comparison_table(
        [tfidf_results, w2v_results], [tfidf_curves, w2v_curves]
    )
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)

    # ---------------- Report ----------------
    print("[9/10] Generating evaluation report...")
    report_text = build_evaluation_report(
        df, eda_summary, class_counts, tfidf_results, w2v_results,
        tfidf_curves, w2v_curves, comparison_df,
    )
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("[10/10] Pipeline complete.")
    print(f"  - TF-IDF vectorizer saved to: {TFIDF_VECTORIZER_PATH}")
    print(f"  - TF-IDF classifier saved to: {TFIDF_MODEL_PATH}")
    print(f"  - Word2Vec model saved to:    {WORD2VEC_MODEL_PATH}")
    print(f"  - Word2Vec classifier saved to: {W2V_CLASSIFIER_PATH}")
    print(f"  - Evaluation report saved to: {REPORT_PATH}")
    print(f"  - Model comparison table saved to: {COMPARISON_CSV_PATH}")
    print(f"  - Figures saved to: {FIGURES_DIR}")
    print()
    print(comparison_df.to_string(index=False))

    return {
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_clf": tfidf_clf,
        "w2v_model": w2v_model,
        "w2v_clf": w2v_clf,
        "tfidf_results": tfidf_results,
        "w2v_results": w2v_results,
        "comparison_df": comparison_df,
    }


if __name__ == "__main__":
    run_pipeline()

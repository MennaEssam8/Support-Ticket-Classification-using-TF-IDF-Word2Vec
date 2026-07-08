"""
Streamlit demo for the Customer Support Ticket Classification System (Phase 1).

Lets a user type in a new support ticket and see:
    - The predicted category from the TF-IDF + Logistic Regression model
    - The predicted category from the Word2Vec (avg) + Logistic Regression model
    - Per-class prediction probabilities for both models (bar charts)
    - A side-by-side agreement indicator

Run with:
    streamlit run app.py

Expects the trained artifacts produced by `src/data_pipeline.py` to already
exist under `models/` and `src/tfidf_vectorizer.pkl`. Run the pipeline first
if you haven't:
    python src/data_pipeline.py
"""

import os
import re
import string

import numpy as np
import pandas as pd
import joblib
import pickle
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

TFIDF_VECTORIZER_PATH = os.path.join(SRC_DIR, "tfidf_vectorizer.pkl")
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, "tfidf_baseline_model.pkl")
WORD2VEC_MODEL_PATH = os.path.join(MODELS_DIR, "word2vec_model.pkl")
W2V_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "word2vec_baseline_model.pkl")
LABEL_LIST_PATH = os.path.join(MODELS_DIR, "label_classes.pkl")

W2V_VECTOR_SIZE = 100

EXAMPLE_TICKETS = [
    "I was charged twice for my subscription this month, can you refund the extra charge?",
    "How do I reset my password, the reset link never arrives in my inbox",
    "Can I change the shipping address for order 48213 before it ships?",
    "I want to cancel my account and delete all my data permanently",
    "Do you offer a student discount on the annual plan?",
]


# --------------------------------------------------------------------------
# Preprocessing (mirrors src/data_pipeline.py so features match training)
# --------------------------------------------------------------------------
@st.cache_resource
def ensure_nltk_resources():
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
    return True


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\{\{(.*?)\}\}", lambda m: re.sub(r"[_\-]", " ", m.group(1)), text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text, stop_words, lemmatizer):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def document_to_avg_vector(tokens, w2v_model, vector_size=W2V_VECTOR_SIZE):
    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


# --------------------------------------------------------------------------
# Cached model loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    missing = [p for p in [TFIDF_VECTORIZER_PATH, TFIDF_MODEL_PATH,
                            WORD2VEC_MODEL_PATH, W2V_CLASSIFIER_PATH]
               if not os.path.exists(p)]
    if missing:
        return None, missing

    with open(TFIDF_VECTORIZER_PATH, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    tfidf_clf = joblib.load(TFIDF_MODEL_PATH)
    w2v_model = joblib.load(WORD2VEC_MODEL_PATH)
    w2v_clf = joblib.load(W2V_CLASSIFIER_PATH)

    artifacts = {
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_clf": tfidf_clf,
        "w2v_model": w2v_model,
        "w2v_clf": w2v_clf,
    }
    return artifacts, []


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict(text, artifacts, stop_words, lemmatizer):
    tokens = preprocess_text(text, stop_words, lemmatizer)
    clean = " ".join(tokens)

    # TF-IDF branch
    X_tfidf = artifacts["tfidf_vectorizer"].transform([clean])
    tfidf_clf = artifacts["tfidf_clf"]
    tfidf_pred = tfidf_clf.predict(X_tfidf)[0]
    tfidf_proba = tfidf_clf.predict_proba(X_tfidf)[0]
    tfidf_classes = tfidf_clf.classes_

    # Word2Vec branch
    vec = document_to_avg_vector(tokens, artifacts["w2v_model"]).reshape(1, -1)
    w2v_clf = artifacts["w2v_clf"]
    w2v_pred = w2v_clf.predict(vec)[0]
    w2v_proba = w2v_clf.predict_proba(vec)[0]
    w2v_classes = w2v_clf.classes_

    return {
        "tokens": tokens,
        "tfidf_pred": tfidf_pred,
        "tfidf_proba": pd.Series(tfidf_proba, index=tfidf_classes).sort_values(ascending=False),
        "w2v_pred": w2v_pred,
        "w2v_proba": pd.Series(w2v_proba, index=w2v_classes).sort_values(ascending=False),
    }


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫", layout="wide")
    st.title("🎫 Customer Support Ticket Classifier — Phase 1 Demo")
    st.caption(
        "Compares a TF-IDF + Logistic Regression baseline against a Word2Vec "
        "(averaged embeddings) + Logistic Regression baseline, trained on the "
        "Bitext Customer Support dataset."
    )

    ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    artifacts, missing = load_artifacts()
    if artifacts is None:
        st.error(
            "Trained model artifacts were not found. Please run "
            "`python src/data_pipeline.py` from the project root first to "
            "generate them.\n\nMissing files:\n" + "\n".join(f"- {m}" for m in missing)
        )
        st.stop()

    with st.sidebar:
        st.header("About")
        st.write(
            "This demo classifies a new, unseen support ticket into one of the "
            "trained categories using two different feature engineering "
            "approaches, so you can visually compare their confidence and "
            "agreement on the same input."
        )
        st.write("**Categories the models were trained on:**")
        st.write(", ".join(sorted(artifacts["tfidf_clf"].classes_)))
        st.divider()
        st.caption(
            "⚠️ This is a Phase 1 baseline demo for illustration purposes. "
            "See `reports/evaluation_report_m1.txt` for full bias, limitation, "
            "and ethical-risk documentation before considering production use."
        )

    example = st.selectbox(
        "Try an example ticket (or type your own below):",
        ["— choose an example —"] + EXAMPLE_TICKETS,
    )
    default_text = "" if example == "— choose an example —" else example

    ticket_text = st.text_area(
        "Enter a support ticket:",
        value=default_text,
        height=120,
        placeholder="e.g. I was charged twice this month, can I get a refund?",
    )

    classify_clicked = st.button("Classify ticket", type="primary")

    if classify_clicked:
        if not ticket_text.strip():
            st.warning("Please enter some ticket text first.")
        else:
            result = predict(ticket_text, artifacts, stop_words, lemmatizer)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("TF-IDF + Logistic Regression")
                st.metric("Predicted category", result["tfidf_pred"])
                st.bar_chart(result["tfidf_proba"].head(8))
                st.caption(f"Top confidence: {result['tfidf_proba'].iloc[0]:.1%}")

            with col2:
                st.subheader("Word2Vec (avg) + Logistic Regression")
                st.metric("Predicted category", result["w2v_pred"])
                st.bar_chart(result["w2v_proba"].head(8))
                st.caption(f"Top confidence: {result['w2v_proba'].iloc[0]:.1%}")

            st.divider()
            if result["tfidf_pred"] == result["w2v_pred"]:
                st.success(
                    f"Both models agree: **{result['tfidf_pred']}**. "
                    "High agreement between independently-trained models is a "
                    "reasonable (though not sufficient) signal of confidence."
                )
            else:
                st.warning(
                    f"Models disagree — TF-IDF says **{result['tfidf_pred']}**, "
                    f"Word2Vec says **{result['w2v_pred']}**. "
                    "Ambiguous or borderline tickets like this are good "
                    "candidates for human review rather than fully automated "
                    "routing."
                )

            with st.expander("Preprocessed tokens used for prediction"):
                st.write(result["tokens"])


if __name__ == "__main__":
    main()

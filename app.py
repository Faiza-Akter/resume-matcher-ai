import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import CONFIG
from src.preprocess import clean_text, build_pair_text
from src.resume_parser import extract_text


# -------------------------
# Load trained model files
# -------------------------
@st.cache_resource
def load_model_artifacts():
    if not CONFIG.vectorizer_path.exists() or not CONFIG.classifier_path.exists():
        return None, None, {}

    vectorizer = joblib.load(CONFIG.vectorizer_path)
    classifier = joblib.load(CONFIG.classifier_path)

    # Load accuracy and metrics
    metrics = {}
    if CONFIG.label_info_path.exists():
        try:
            with open(CONFIG.label_info_path, "r", encoding="utf-8") as f:
                label_info = json.load(f)
            metrics = label_info.get("metrics", {}) or {}
        except Exception:
            metrics = {}

    return vectorizer, classifier, metrics


vectorizer, classifier, metrics = load_model_artifacts()


# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="AI Resume Matcher", page_icon="🤖", layout="wide")

st.title("🤖 AI Resume Matcher using NLP + Machine Learning")
st.write("Upload resumes and compare them against any job description.")

# Show model status + accuracy
c1, c2 = st.columns([0.6, 0.4])
with c1:
    if vectorizer is None or classifier is None:
        st.warning("⚠️ Model not found. Train the model first: `py -u -m src.model --train`")
    else:
        st.success("✅ Model loaded successfully.")

with c2:
    if metrics:
        st.info(f"📊 **Training Accuracy (dataset): {metrics.get('accuracy', 'N/A')}%**")
    else:
        st.caption("Metrics not found (train once to generate models/label_info.json).")


# -------------------------
# Job Description Input
# -------------------------
st.subheader("📌 Step 1: Enter Job Description")
job_description = st.text_area(
    "Paste the job description below:",
    height=180,
    placeholder="Example: Looking for a Python Developer with experience in ML, NLP, APIs, SQL..."
)

# -------------------------
# Resume Upload
# -------------------------
st.subheader("📌 Step 2: Upload Resumes (PDF, DOCX, TXT)")
uploaded_files = st.file_uploader(
    "Upload multiple resumes:",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Optional settings
with st.expander("⚙️ Scoring Settings", expanded=False):
    st.write("**Similarity Score** is used for ranking (job text vs resume text).")
    strong_thr = st.slider("Strong match threshold (similarity %)", 10, 90, 35)
    moderate_thr = st.slider("Moderate match threshold (similarity %)", 5, 80, 20)

    st.write("**ML Prediction** is separate (trained classifier probability).")
    show_clean_preview = st.checkbox("Show cleaned text preview for top resume", value=False)


# -------------------------
# Similarity (Ranking) Score
# -------------------------
def compute_similarity_scores(job_text: str, resume_texts: list[str]) -> list[float]:
    """
    Returns cosine similarity scores (0..1) between job text and each resume.
    Uses local TF-IDF fit on current job + resumes for better ranking.
    """
    job_clean = clean_text(job_text)
    resumes_clean = [clean_text(t) for t in resume_texts]

    corpus = [job_clean] + resumes_clean
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = tfidf.fit_transform(corpus)

    # similarity job (row 0) vs each resume (rows 1..)
    sims = cosine_similarity(mat[0:1], mat[1:]).flatten()
    return [float(x) for x in sims]


# -------------------------
# ML Prediction Score (Classifier)
# -------------------------
def compute_ml_probability(job_text: str, resume_text: str) -> float | None:
    """
    Returns probability of class 1 (match) from the trained model.
    If model is not available, returns None.
    """
    if vectorizer is None or classifier is None:
        return None

    combined = build_pair_text(job_text, resume_text)  # consistent with training preprocessing
    X = vectorizer.transform([combined])

    if hasattr(classifier, "predict_proba"):
        return float(classifier.predict_proba(X)[0][1])
    # fallback
    pred = int(classifier.predict(X)[0])
    return 1.0 if pred == 1 else 0.0


def similarity_category(sim_percent: float, strong_t: float, moderate_t: float) -> str:
    if sim_percent >= strong_t:
        return "🟢 Strong Match"
    if sim_percent >= moderate_t:
        return "🟡 Moderate Match"
    return "🔴 Weak Match"


# -------------------------
# Submit Button
# -------------------------
if st.button("🔍 Match Resumes"):
    if not job_description.strip():
        st.error("❌ Please enter a job description.")
        st.stop()

    if not uploaded_files:
        st.error("❌ Please upload at least one resume.")
        st.stop()

    # Extract resume texts
    resume_names = []
    resume_texts = []
    for file in uploaded_files:
        text = extract_text(file.name, file.read())
        if not text.strip():
            st.warning(f"⚠ Could not read text from file: {file.name}")
            continue
        resume_names.append(file.name)
        resume_texts.append(text)

    if not resume_texts:
        st.error("❌ No readable resumes found. Try TXT/DOCX or a text-based PDF.")
        st.stop()

    # Compute ranking similarity scores
    sims = compute_similarity_scores(job_description, resume_texts)
    sims_percent = [s * 100.0 for s in sims]

    # Build results
    rows = []
    for name, raw_text, sim_p in zip(resume_names, resume_texts, sims_percent):
        ml_prob = compute_ml_probability(job_description, raw_text)

        rows.append({
            "Resume": name,
            "SimilarityScore(%)": round(sim_p, 2),
            "Category": similarity_category(sim_p, strong_thr, moderate_thr),
            "ML_Match_Prob(%)": (round(ml_prob * 100.0, 2) if ml_prob is not None else None),
            "ML_Prediction": ("Match ✅" if (ml_prob is not None and ml_prob >= 0.5) else ("Not Match ❌" if ml_prob is not None else "N/A")),
            "_raw": raw_text,
        })

    df = pd.DataFrame(rows).sort_values("SimilarityScore(%)", ascending=False).reset_index(drop=True)

    st.subheader("📌 Match Results (Ranked by Similarity)")
    st.dataframe(df.drop(columns=["_raw"]), use_container_width=True)

    # Optional preview
    if show_clean_preview:
        top = df.iloc[0]
        st.markdown("### 🔍 Cleaned text preview (Top resume)")
        st.write("**Cleaned Job Description:**")
        st.code(clean_text(job_description)[:2000] + ("..." if len(clean_text(job_description)) > 2000 else ""))

        st.write(f"**Cleaned Resume:** ({top['Resume']})")
        st.code(clean_text(top["_raw"])[:2000] + ("..." if len(clean_text(top['_raw'])) > 2000 else ""))

    st.caption(
        "✅ SimilarityScore is used for ranking (cosine similarity between job & resume). "
        "🧠 ML_Match_Prob is the trained model’s probability output."
    )
